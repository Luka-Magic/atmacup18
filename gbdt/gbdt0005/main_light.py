import hydra
import re
import wandb
import pandas as pd
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Union

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
import lightgbm
import matplotlib.pyplot as plt

from utils import seed_everything, AverageMeter
from feature_block import run_block, NumericBlock, LabelEncodingBlock, CountEncodingBlock, AggBlock


GBDT_DIR = Path.cwd()
GBDT_ID =  Path.cwd().name
ROOT_DIR = GBDT_DIR.parents[2]

DATA_DIR = ROOT_DIR / 'data'
ORIGINAL_DATA_DIR = DATA_DIR / 'original_data/atmaCup#18_dataset'
CREATED_DATA_DIR = DATA_DIR / 'created_data'

OUTPUT_DIR = ROOT_DIR / 'outputs'

ID_COLUMNS = ['ID']
META_COLUMNS = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake', 'brakePressed', 'gas', 'gasPressed', 'gearShifter', 'leftBlinker', 'rightBlinker']
TARGET_COLUMNS = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']

def split_data(cfg, df):
    scene_ser = df['ID'].apply(lambda x: x.split('_')[0])

    df['fold'] = -1
    group_kfold = GroupKFold(n_splits=cfg.n_folds)
    for ifold, (_, valid_index) in enumerate(group_kfold.split(df, groups=scene_ser)):
        df.loc[valid_index, 'fold'] = ifold
    return df

def mae(gt: np.array, pred: np.array):
    abs_diff = np.abs(gt - pred)
    score = np.mean(abs_diff.reshape(-1, ))
    return float(score)

def common_preprocess(target_df: pd.DataFrame) -> Union[pd.DataFrame, List[str]]:
    '''
    処理
    ----
    - boolのcolをintに変換
    - scene, scene_sec, scene_countを追加
    '''
    num_columns = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake', 'brakePressed', 'gas', 'gasPressed',  'leftBlinker', 'rightBlinker']
    
    # boolのcol
    bool_columns = ['brakePressed', 'gasPressed', 'leftBlinker', 'rightBlinker']
    target_df[bool_columns] = target_df[bool_columns].astype(int)

    target_df['scene'] = target_df['ID'].str.split('_').str[0]
    target_df['scene_sec'] = target_df['ID'].str.split('_').str[1].astype(int)

    count_df = target_df.groupby('scene').size()
    target_df['scene_count'] = target_df['scene'].map(count_df)
    num_columns.append(['scene_sec', 'scene_count'])
    
    # 2. steeringAngleDeg を度からラジアンに変換
    target_df["steeringAngleRad"] = np.deg2rad(target_df["steeringAngleDeg"])
    num_columns.append("steeringAngleRad")

    # 3. 三角関数の特徴量を作成
    target_df["steeringAngle_sin"] = np.sin(target_df["steeringAngleRad"])
    target_df["steeringAngle_cos"] = np.cos(target_df["steeringAngleRad"])
    num_columns.extend(["steeringAngle_sin", "steeringAngle_cos"])

    # 4. 交互作用特徴量を作成
    target_df["speed_steering"] = target_df["vEgo"] * target_df["steeringAngleRad"]  # 速度とステアリング角度の組み合わせ
    target_df["acc_steeringTorque"] = target_df["aEgo"] * target_df["steeringTorque"]  # 加速度とステアリングトルクの組み合わせ
    num_columns.extend(["speed_steering", "acc_steeringTorque"])

    # 5. 対数変換
    target_df["vEgo_positive"] = target_df["vEgo"].clip(lower=0) + 1e-6
    target_df["log_vEgo"] = np.log(target_df["vEgo_positive"])
    num_columns.append("log_vEgo")

    # # 6. 加速度の変化率
    # target_df["jerk"] = target_df.groupby("scene")["aEgo"].diff()
    # num_columns.append("jerk")

    # 7. ステアリング角度とトルクの変化率
    # target_df["steeringAngleRate"] = target_df.groupby("scene")["steeringAngleRad"].diff()
    # target_df["steeringTorqueRate"] = target_df.groupby("scene")["steeringTorque"].diff()
    # num_columns.extend(["steeringAngleRate", "steeringTorqueRate"])

    # 8. 二乗・絶対値特徴量
    target_df["vEgo_squared"] = target_df["vEgo"] ** 2
    target_df["steeringAngleRad_squared"] = target_df["steeringAngleRad"] ** 2
    target_df["aEgo_squared"] = target_df["aEgo"] ** 2
    num_columns.extend(["vEgo_squared", "steeringAngleRad_squared", "aEgo_squared"])

    # 9. 移動平均や移動和
    target_df["vEgo_roll_mean"] = target_df.groupby("scene")["vEgo"].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
    target_df["aEgo_roll_mean"] = target_df.groupby("scene")["aEgo"].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
    num_columns.extend(["vEgo_roll_mean", "aEgo_roll_mean"])
    
    return target_df, num_columns

# 信号機に関する特徴量を追加
def add_traffic_light_feature(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:
    '''
    処理
    ----
    - 信号機の数をを追加 (jsonの中のlistの長さ)
    '''
    tl_dir = ORIGINAL_DATA_DIR / 'traffic_lights'
    tl_json_paths = list(tl_dir.glob('*.json'))

    traffic_light_columns = []

    traffic_lights = []
    id_class_list = []
    for tl_json_path in tqdm(tl_json_paths, total=len(tl_json_paths)):
        tl_id = tl_json_path.stem
        traffic_light = json.load(open(tl_json_path))

        traffic_lights.append(traffic_light)

        for traffic_light in traffic_light:
            id_class_list.append((tl_id.split('.')[0], traffic_light['class']))

    # 信号機の数
    counts = [len(traffic_light) for traffic_light in traffic_lights]
    
    # traffic_lights_df = pd.DataFrame(id_class_list, columns=['ID', 'class'])

    tl_ids = [json_path.stem for json_path in tl_json_paths]
    traffic_lights_df = pd.DataFrame({
        'ID': tl_ids,
        'traffic_lights_counts': counts
    })

    train_df = pd.merge(train_df, traffic_lights_df, on='ID', how='left')
    test_df = pd.merge(test_df, traffic_lights_df, on='ID', how='left')
    return train_df, test_df


# epipolarの特徴量を追加
def add_epipolar_feature(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_epipolar_df: pd.DataFrame,
        test_epipolar_df: pd.DataFrame
    ):
    '''
    処理
    ----
    - epipolar_dfの特徴量を追加
    '''
    epipolar_columns = [c for c in train_epipolar_df.columns if re.search('^r', c)]
    train_df = pd.merge(train_df, train_epipolar_df, on='ID', how='left')
    test_df = pd.merge(test_df, test_epipolar_df, on='ID', how='left')
    # -1をNaNに変換
    train_df[epipolar_columns] = train_df[epipolar_columns].replace(-1, np.nan)
    test_df[epipolar_columns] = test_df[epipolar_columns].replace(-1, np.nan)
    return train_df, test_df, epipolar_columns


# oofの特徴量を追加
def add_oof_feature(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        img_oof_paths: List[Path],
        img_submissions_paths: List[Path],
        oof_feature: bool = False,
        oof_v: bool = False
    ):
    '''
    処理
    ----
    - oof_dfの特徴量を追加
    '''
    assert len(img_oof_paths) == len(img_submissions_paths), 'len(img_oof_paths) != len(img_submissions_paths)'
    
    oof_feat_columns = []
    for img_oof_path, img_submission_path in zip(img_oof_paths, img_submissions_paths):
        img_oof_df = pd.read_csv(img_oof_path, index_col=0)
        img_oof_name = img_oof_path.parent.name

        _oof_feat_columns  = [f'{img_oof_name}_{c}' for c in TARGET_COLUMNS]
        pred_columns = [f'pred_{i}' for i in TARGET_COLUMNS]

        if oof_v:
            # 速度を求める
            for i in range(6):
                for r in ['x', 'y', 'z']:
                    if i == 0:
                        img_oof_df[f'v_{r}_{i}'] = img_oof_df[f'pred_{r}_{i}']
                    else:
                        img_oof_df[f'v_{r}_{i}'] = img_oof_df[f'pred_{r}_{i}'] - img_oof_df[f'pred_{r}_{i-1}']
                        _oof_feat_columns.append(f'{img_oof_name}_v_{r}_{i}')
                        pred_columns.append(f'v_{r}_{i}')
            # 加速度を求める
            for i in range(6):
                for r in ['x', 'y', 'z']:
                    if i == 0:
                        img_oof_df[f'a_{r}_{i}'] = img_oof_df[f'v_{r}_{i}']
                    else:
                        img_oof_df[f'a_{r}_{i}'] = img_oof_df[f'v_{r}_{i}'] - img_oof_df[f'v_{r}_{i-1}']
                        _oof_feat_columns.append(f'{img_oof_name}_a_{r}_{i}')
                        pred_columns.append(f'a_{r}_{i}')
        
        if oof_feature:
            feature_columns = [c for c in img_oof_df.columns if re.search('^feature_', c)]
            _oof_feat_columns += [f'{img_oof_name}_{c}' for c in feature_columns]
            pred_columns += feature_columns

        img_oof_df.sort_values(by='ID', inplace=True)
        img_oof_df.reset_index(drop=True, inplace=True)
        assert train_df.shape[0] == img_oof_df.shape[0], f'train_df.shape[0] ({train_df.shape[0]}) != img_oof_df.shape[0] ({img_oof_df.shape[0]})'
        train_df[_oof_feat_columns] = img_oof_df[pred_columns]

        img_submission_df = pd.read_csv(img_submission_path)
        target_columns = TARGET_COLUMNS.copy()

        if oof_v:
            # 速度を求める
            for i in range(6):
                for r in ['x', 'y', 'z']:
                    if i == 0:
                        img_submission_df[f'v_{r}_{i}'] = img_submission_df[f'{r}_{i}']
                    else:
                        img_submission_df[f'v_{r}_{i}'] = img_submission_df[f'{r}_{i}'] - img_submission_df[f'{r}_{i-1}']
                        target_columns.append(f'v_{r}_{i}')
            # 加速度を求める
            for i in range(6):
                for r in ['x', 'y', 'z']:
                    if i == 0:
                        img_submission_df[f'a_{r}_{i}'] = img_submission_df[f'v_{r}_{i}']
                    else:
                        img_submission_df[f'a_{r}_{i}'] = img_submission_df[f'v_{r}_{i}'] - img_submission_df[f'v_{r}_{i-1}']
                        target_columns.append(f'a_{r}_{i}')
        
        if oof_feature:
            target_columns += feature_columns
        
        test_df[_oof_feat_columns] = img_submission_df[target_columns]

        oof_feat_columns.extend(_oof_feat_columns)
    
    return train_df, test_df, oof_feat_columns


# shift特徴量を追加
def make_shift_feature(target_df, use_feat_columns):
    shift_count = 1
    shift_range = list(range(-shift_count, shift_count+1))
    shift_range = [x for x in shift_range if x != 0]

    target_df['ori_idx'] = target_df.index

    target_df = target_df.sort_values(['scene', 'scene_sec']).reset_index(drop=True)

    shift_feat_columns = []
    for shift in shift_range:
        for col in use_feat_columns:
            shift_col = f'{col}_shift{shift}'
            target_df[shift_col] = target_df.groupby('scene')[col].shift(shift)
            shift_feat_columns.append(shift_col)

            diff_col = f'{col}_diff{shift}'
            target_df[diff_col] = target_df[col] - target_df[shift_col]
            shift_feat_columns.append(diff_col)

    target_df = target_df.sort_values('ori_idx').reset_index(drop=True)
    target_df = target_df.drop('ori_idx', axis=1)

    return target_df, shift_feat_columns

def add_feature_block(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_columns: List[str] = [],
        agg_num_columns: List[str] = [],
        cat_label_columns: List[str] = [],
        cat_count_columns: List[str] = [],
        cat_te_columns: List[str] = [],

    ):
    '''
    処理
    ----
    - feature_blocksの処理を実行
    '''
    train_num = len(train_df)

    # ======= train_df, test_dfを結合して処理 =======
    whole_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    blocks = [
        *[NumericBlock(col) for col in num_columns],
        *[LabelEncodingBlock(col) for col in cat_label_columns],
        *[CountEncodingBlock(col) for col in cat_count_columns],
        # *[AggBlock(group_col, target_columns=agg_num_columns,
        #            agg_columns=['mean', 'max', 'min', 'std']) for group_col in ['scene']],
    ]
    whole_feat_df = run_block(whole_df, blocks, is_fit=True)

    # ======= train_df, test_df 別々に処理 =======

    train_df, test_df = whole_df.iloc[:train_num], whole_df.iloc[train_num:].drop(
        columns=TARGET_COLUMNS).reset_index(drop=True)
    train_feat, test_feat = whole_feat_df.iloc[:train_num], whole_feat_df.iloc[train_num:].reset_index(
        drop=True)

    blocks = [
        # *[TargetEncodingBlock(col, TARGET_COLUMNS) for col in cat_te_columns]
    ]

    _df = run_block(train_df, blocks, is_fit=True)
    train_feat = pd.concat([train_feat, _df], axis=1)
    _df = run_block(test_df, blocks, is_fit=False)
    test_feat = pd.concat([test_feat, _df], axis=1)

    return train_df, test_df, train_feat, test_feat

## ====================================================

# gbdtモデル
class LightGBM:
    def __init__(
            self,
            lgb_params,
            save_dir=None,
            categorical_feature=None,
            model_name='lgb',
            stopping_rounds=50
        ) -> None:

        self.save_dir = save_dir
        self.lgb_params = lgb_params
        self.categorical_feature = categorical_feature

        # saveの切り替え用
        self.model_name = model_name

        self.stopping_rounds = stopping_rounds

    def fit(self, x_train, y_train, **fit_params) -> None:

        X_val, y_val = fit_params['eval_set'][0]
        del fit_params['eval_set']

        train_dataset = lightgbm.Dataset(
            x_train, y_train, categorical_feature=self.categorical_feature)

        val_dataset = lightgbm.Dataset(
            X_val, y_val, categorical_feature=self.categorical_feature)

        self.model = lightgbm.train(
            params=self.lgb_params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=[lightgbm.early_stopping(stopping_rounds=self.stopping_rounds,
                                            verbose=True),
                        lightgbm.log_evaluation(500)],
            **fit_params
        )

    # def save(self, fold):
    #     save_to = self.save_dir / f'lgb_fold_{fold}_{self.model_name}.txt'
    #     self.model.save_model(save_to)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


def get_model(
        cfg,
        model_name
    ):
    lgb_params = {
        'objective': 'regression',
        'boosting_type': cfg.boosting_type,
        'verbose': -1,
        'n_jobs': 8,
        'seed': cfg.seed,
        'learning_rate': cfg.learning_rate,
        # 'num_class': CFG.num_class, # multiclassなら必要
        'metric': 'mae',
        'num_leaves': cfg.num_leaves,
        'max_depth': cfg.max_depth,
        'subsample': cfg.subsample,
        'colsample_bytree': cfg.colsample_bytree,
        'min_data_in_leaf': cfg.min_data_in_leaf,
        'bagging_seed': cfg.seed,
        'feature_fraction_seed': cfg.seed,
        'drop_seed': cfg.seed,
    }
    # save_log_dir = SAVE_DIR / 'log'
    # save_log_dir.mkdir(exist_ok=True, parents=True)

    model = LightGBM(
                lgb_params=lgb_params,
                save_dir=None,
                model_name=model_name
    )

    return model

def get_fit_params(cfg, model_name):
    params = {
        'num_boost_round': 100000
    }
    return params

def get_result(result_df):
    pred_cols = [f'pred_{i}' for i in range(len(TARGET_COLUMNS))]

    preds = result_df[pred_cols].values
    labels = result_df[TARGET_COLUMNS].values

    eval_func = eval('mae')
    best_score = eval_func(labels, preds)

    print(f'best_score: {best_score:<.4f}')
    return best_score


@hydra.main(config_path='config', config_name='config_light')
def main(cfg: DictConfig):
    raw_train_df = pd.read_csv(ORIGINAL_DATA_DIR / 'train_features.csv')
    raw_test_df = pd.read_csv(ORIGINAL_DATA_DIR / 'test_features.csv')
    ss_df = pd.read_csv(ORIGINAL_DATA_DIR / 'atmaCup18__sample_submit.csv')

    seed_everything(cfg.seed)

    y = raw_train_df[TARGET_COLUMNS]
    train_with_fold_df = split_data(cfg, raw_train_df)

    oof_predictions = np.zeros((raw_train_df.shape[0], len(TARGET_COLUMNS)))
    test_predictions = np.zeros((raw_test_df.shape[0], len(TARGET_COLUMNS)))

    # =======
    fold = cfg.fold
    target_column = cfg.c
    # =======

    train_indices = train_with_fold_df['fold'] != fold
    valid_indices = train_with_fold_df['fold'] == fold

    # preprocess
    train_df, common_num_columns = common_preprocess(train_with_fold_df)
    test_df, _ = common_preprocess(raw_test_df)

    # traffic_light
    if cfg.use_traffic_light:
        train_df, test_df = add_traffic_light_feature(train_df, test_df)
    
    # epipolar
    if cfg.use_epipolar:
        train_epipolar_df = pd.read_csv(CREATED_DATA_DIR / 'data0004' / 'train_features_r.csv')
        test_epipolar_df = pd.read_csv(CREATED_DATA_DIR / 'data0004' / 'test_features_r.csv')
        train_df, test_df, epipolar_columns = add_epipolar_feature(
            train_df,
            test_df,
            train_epipolar_df,
            test_epipolar_df
        )
    else:
        epipolar_columns = []
            
    # oof
    if cfg.oof_ids is not None and len(cfg.oof_ids) > 0:
        img_oof_paths = []
        img_submissions_paths = []
        for oof_id in cfg.oof_ids:
            if cfg.oof_feature:
                img_oof_paths.append(OUTPUT_DIR / 'exp' / oof_id / 'oof_feature.csv')
                img_submissions_paths.append(OUTPUT_DIR / 'exp' / oof_id / f'submission_feature_fold{fold}.csv')
            else:
                img_oof_paths.append(OUTPUT_DIR / 'exp' / oof_id / 'oof.csv')
                img_submissions_paths.append(OUTPUT_DIR / 'exp' / oof_id / f'submission_fold{fold}.csv')

        train_df, test_df, oof_feat_columns = add_oof_feature(
            train_df,
            test_df,
            img_oof_paths,
            img_submissions_paths,
            oof_feature=cfg.oof_feature,
            oof_v=cfg.oof_v
        )
    else:
        oof_feat_columns = []

    # shift
    use_shift_columns = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake', 'brakePressed', 'gas', 'gasPressed',  'leftBlinker', 'rightBlinker']
    if cfg.oof_shift:
        use_shift_columns += oof_feat_columns
    train_df, shift_columns = make_shift_feature(train_df, use_shift_columns)
    test_df, shift_columns = make_shift_feature(test_df, use_shift_columns)

    # feature block
    num_columns = common_num_columns
    num_columns += oof_feat_columns
    num_columns += shift_columns
    num_columns += epipolar_columns
    if cfg.use_traffic_light:
        num_columns += ['traffic_lights_counts']

    agg_num_columns = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake', 'gas']

    cat_label_columns = ['gearShifter']
    cat_count_columns = []
    cat_te_columns = []

    train_df, test_df, train_feat, test_feat = add_feature_block(
        train_df,
        test_df,
        num_columns=num_columns,
        agg_num_columns=agg_num_columns,
        cat_label_columns=cat_label_columns,
        cat_count_columns=cat_count_columns,
        cat_te_columns=cat_te_columns
    )

    print(f'feature columns:', train_feat.columns)
    print(f'num feature columns:', len(train_feat.columns))

    train_df.to_csv(f'tmp_train.csv', index=False)
    test_df.to_csv(f'tmp_test.csv', index=False)

    target_idx = TARGET_COLUMNS.index(target_column)
    print(f'fold: {fold}, target_column: {target_column}')

    x_train = train_feat.loc[train_indices]
    x_valid = train_feat.loc[valid_indices]
    y_train = train_df.loc[train_indices, target_column]
    y_valid = train_df.loc[valid_indices, target_column]

    model_name = f'lgb_{target_column}'
    model = get_model(cfg, model_name)

    fit_params = get_fit_params(cfg, model_name)

    fit_params_fold = fit_params.copy()
    fit_params_fold['eval_set'] = [(x_valid, y_valid)]

    model.fit(x_train, y_train, **fit_params_fold)

    oof_predictions[valid_indices, target_idx] = model.predict(x_valid)
    test_predictions[:, target_idx] += model.predict(test_feat)
    eval_func = eval('mae')
    score_fold = eval_func(y.loc[valid_indices, target_column].values, oof_predictions[valid_indices, target_idx])
    
    lightgbm.plot_importance(model.model, figsize=(8,20), max_num_features=50, importance_type='gain')
    plt.savefig(f'importance_{target_column}.png')

    print(f'fold: {fold}, score: {score_fold:<.4f}')

if __name__ == '__main__':
    main()