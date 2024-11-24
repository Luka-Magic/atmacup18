import hydra
import re
import gc
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

from utils import seed_everything, AverageMeter
from feature_block import run_block, NumericBlock, LabelEncodingBlock, CountEncodingBlock, AggBlock

GBDT_DIR = Path.cwd()
GBDT_ID =  Path.cwd().name
ROOT_DIR = GBDT_DIR.parents[2]

DATA_DIR = ROOT_DIR / 'data'
ORIGINAL_DATA_DIR = DATA_DIR / 'original_data/atmaCup#18_dataset'
CREATED_DATA_DIR = DATA_DIR / 'created_data'

OUTPUT_DIR = ROOT_DIR / 'outputs'

SAVE_DIR = OUTPUT_DIR / 'gbdt' / GBDT_ID
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# WANDB_DIR = SAVE_DIR / 'wandb'
# WANDB_DIR.mkdir(parents=True, exist_ok=True)

ID_COLUMNS = ['ID']
META_COLUMNS = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brakePressed', 'gas', 'gasPressed', 'gearShifter', 'leftBlinker', 'rightBlinker']
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
    num_cols = []
    
    # brake消す
    if 'brake' in target_df.columns:
        target_df.drop('brake', axis=1, inplace=True)
    
    # boolのcol
    bool_columns = ['brakePressed', 'gasPressed', 'leftBlinker', 'rightBlinker']
    target_df[bool_columns] = target_df[bool_columns].astype(int)

    target_df['scene'] = target_df['ID'].str.split('_').str[0]
    target_df['scene_sec'] = target_df['ID'].str.split('_').str[1].astype(int)

    target_df['ori_idx'] = target_df.index
    
    # sceneでsort
    target_df.sort_values(by=['scene', 'scene_sec'], inplace=True)
    # 1. sceneの特徴量
    count_df = target_df.groupby('scene').size()
    target_df['scene_count'] = target_df['scene'].map(count_df)
    
    scene_sec_from_zero = target_df.groupby('scene').apply(lambda x:x['scene_sec'] - x['scene_sec'].min()).reset_index()['scene_sec'].values
    target_df['scene_sec_from_zero'] = scene_sec_from_zero
    target_df['scene_sec_rank'] = target_df.groupby('scene')['scene_sec'].rank(method='first').astype(int)
    
    num_cols.append(['scene_sec', 'scene_count', 'scene_sec_from_zero', 'scene_sec_rank'])
    
    # 2. steeringAngleDeg を度からラジアンに変換
    target_df["steeringAngleRad"] = np.deg2rad(target_df["steeringAngleDeg"])
    num_cols.append("steeringAngleRad")

    # 3. 三角関数の特徴量を作成
    target_df["steeringAngle_sin"] = np.sin(target_df["steeringAngleRad"])
    target_df["steeringAngle_cos"] = np.cos(target_df["steeringAngleRad"])
    num_cols.extend(["steeringAngle_sin", "steeringAngle_cos"])

    # 4. 交互作用特徴量を作成
    target_df["speed_steering"] = target_df["vEgo"] * target_df["steeringAngleRad"]  # 速度とステアリング角度の組み合わせ
    target_df["acc_steeringTorque"] = target_df["aEgo"] * target_df["steeringTorque"]  # 加速度とステアリングトルクの組み合わせ
    num_cols.extend(["speed_steering", "acc_steeringTorque"])

    # 5. 対数変換
    target_df["vEgo_positive"] = target_df["vEgo"].clip(lower=0) + 1e-6
    target_df["log_vEgo"] = np.log(target_df["vEgo_positive"])
    num_cols.append("log_vEgo")

    # 6. 加速度の変化率
    target_df["jerk"] = target_df.groupby("scene")["aEgo"].diff()
    num_cols.append("jerk")

    # 7. ステアリング角度とトルクの変化率
    target_df["steeringAngleRate"] = target_df.groupby("scene")["steeringAngleRad"].diff()
    target_df["steeringTorqueRate"] = target_df.groupby("scene")["steeringTorque"].diff()
    num_cols.extend(["steeringAngleRate", "steeringTorqueRate"])

    # 8. 二乗・絶対値特徴量
    target_df["vEgo_squared"] = target_df["vEgo"] ** 2
    target_df["steeringAngleRad_squared"] = target_df["steeringAngleRad"] ** 2
    target_df["aEgo_squared"] = target_df["aEgo"] ** 2
    num_cols.extend(["vEgo_squared", "steeringAngleRad_squared", "aEgo_squared"])

    # 9. 移動平均や移動和
    target_df["vEgo_roll_mean"] = target_df.groupby("scene")["vEgo"].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
    target_df["aEgo_roll_mean"] = target_df.groupby("scene")["aEgo"].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
    num_cols.extend(["vEgo_roll_mean", "aEgo_roll_mean"])
    
    # IDでsortしなおす
    target_df = target_df.sort_values('ori_idx').reset_index(drop=True)
    target_df = target_df.drop('ori_idx', axis=1)
    
    return target_df, num_cols

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
    traffic_lights_df = pd.read_csv(CREATED_DATA_DIR / 'data0005' / 'traffic_light.csv')
    
    # classという名前があれなのでclass_nameに変える
    traffic_lights_df.rename(columns={'class': 'class_name'}, inplace=True)

    traffic_lights_df['bbox_c_x'] = traffic_lights_df.apply(lambda x:(x['bbox_2'] + x['bbox_0']) / 2 , axis=1)
    traffic_lights_df['bbox_c_y'] = traffic_lights_df.apply(lambda x:(x['bbox_3'] + x['bbox_1']) / 2 , axis=1)
    traffic_lights_df['bbox_aspect'] = traffic_lights_df.apply(lambda x:(x['bbox_2'] - x['bbox_0']) / (x['bbox_3'] - x['bbox_1']) , axis=1)
    traffic_lights_df['bbox_area'] = traffic_lights_df.apply(lambda x:(x['bbox_2'] - x['bbox_0']) * (x['bbox_3'] - x['bbox_1']), axis=1)

    # 面積が30以上のものを削除
    traffic_lights_df = traffic_lights_df.query('bbox_area < 30').reset_index(drop=True)

    # 信号の数
    tl_count = traffic_lights_df.groupby(['ID']).size().reset_index().rename(columns={0: 'n_traffic_lights'})
    traffic_lights_df = traffic_lights_df.merge(tl_count, on='ID')

    # bboxをstrにして一意に
    traffic_lights_df['bbox_str'] = traffic_lights_df.apply(lambda x:f'[{x["bbox_0"]:.3f}, {x["bbox_1"]:.3f}, {x["bbox_2"]:.3f}, {x["bbox_3"]:.3f}]', axis=1)

    # id, bbox, classでsort
    traffic_lights_df.sort_values(by=['ID', 'bbox_str', 'class_name'], inplace=True)

    # 一つの信号機に対するclassの組み合わせを取得
    same_tl_df = traffic_lights_df.groupby(['ID', 'bbox_str'])['class_name'].unique().reset_index().rename(columns={'class_name': 'class_unique'})

    # 一つの信号機に対するclassの組み合わせの個数を取得
    same_tl_size_df = traffic_lights_df.groupby(['ID', 'bbox_str'])['class_name'].nunique().reset_index().rename(columns={'class_name': 'n_signs'})

    # 一つの信号の情報をマージ
    traffic_lights_df = traffic_lights_df.merge(same_tl_df, on=['ID', 'bbox_str'])
    traffic_lights_df = traffic_lights_df.merge(same_tl_size_df, on=['ID', 'bbox_str'])

    # IDに対して最大の面積の信号のみ使う
    area_df = traffic_lights_df.drop_duplicates(['ID', 'bbox_str'])[['ID', 'bbox_str', 'bbox_area']]
    area_df = area_df.groupby(['ID'])[['bbox_area']].rank(ascending=False).astype(int)
    traffic_lights_df = traffic_lights_df.loc[area_df.query('bbox_area == 1').index]
    
    # 必要な特徴量に厳選
    tl_feature_df = traffic_lights_df[['ID', 'bbox_c_x', 'bbox_c_y', 'bbox_aspect', 'bbox_area', 'n_signs', 'n_traffic_lights', 'class_unique']].copy()
    # 最大面積の信号機の各信号を特徴量に
    sign_columns = ['green', 'yellow', 'red', 'straight', 'left', 'right', 'empty', 'other']
    tl_feature_df.reset_index(drop=True, inplace=True)
    
    # 各信号があるかをチェック
    tl_feature_df[[f'sign_{c}' for c in sign_columns]] = 0
    for i, row in tqdm(tl_feature_df.iterrows(), total=len(tl_feature_df)):
        for class_name in row['class_unique']:
            tl_feature_df.loc[i, f'sign_{class_name}'] = 1
    tl_feature_df.drop('class_unique', axis=1, inplace=True)

    traffic_columns = [c for c in tl_feature_df.columns if c != 'ID']
    
    train_df = pd.merge(train_df, tl_feature_df, on='ID', how='left')
    test_df = pd.merge(test_df, tl_feature_df, on='ID', how='left')
    return train_df, test_df, traffic_columns


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
    shift_diff_count = 1
    shift_range = list(range(-shift_count, shift_count+1))
    shift_range = [x for x in shift_range if x != 0]
    shift_diff_range = list(range(-shift_diff_count, shift_diff_count+1))
    shift_diff_range = [x for x in shift_diff_range if x != 0]

    target_df['ori_idx'] = target_df.index

    target_df = target_df.sort_values(['scene', 'scene_sec']).reset_index(drop=True)

    shift_feat_columns = []
    for shift in shift_range:
        for col in use_feat_columns:
            shift_col = f'{col}_shift{shift}'
            target_df[shift_col] = target_df.groupby('scene')[col].shift(shift)
            shift_feat_columns.append(shift_col)
    
    for shift in shift_diff_range:
        for col in use_feat_columns:
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

import xgboost as xgb
from pathlib import Path

# XGBoostモデル
class XGBoost:
    def __init__(
            self,
            xgb_params,
            save_dir=None,
            model_name='xgb',
            stopping_rounds=50
        ) -> None:

        self.save_dir = save_dir
        self.xgb_params = xgb_params

        # saveの切り替え用
        self.model_name = model_name

        self.stopping_rounds = stopping_rounds

    def fit(self, x_train, y_train, **fit_params) -> None:
        eval_set = fit_params['eval_set']
        X_val, y_val = eval_set[0]

        del fit_params['eval_set']
        train_dataset = xgb.DMatrix(data=x_train, label=y_train)
        val_dataset = xgb.DMatrix(data=X_val, label=y_val)

        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=train_dataset,
            evals=[(train_dataset, 'train'), (val_dataset, 'validation')],
            early_stopping_rounds=self.stopping_rounds,
            verbose_eval=500,
            **fit_params
        )

    def save(self, fold):
        save_to = self.save_dir / f'xgb_fold_{fold}_{self.model_name}.json'
        self.model.save_model(str(save_to))

    def predict(self, x):
        dmatrix = xgb.DMatrix(data=x)
        return self.model.predict(dmatrix)

    def predict_proba(self, x):
        dmatrix = xgb.DMatrix(data=x)
        return self.model.predict(dmatrix)  # For binary/multiclass classification

def get_model(
        cfg,
        model_name
    ):
    xgb_params = {
        'objective': 'reg:squarederror',  # regression objective
        'booster': cfg.boosting_type,
        'verbosity': 0,
        'n_jobs': 8,
        'seed': cfg.seed,
        'eta': cfg.eta,
        # 'num_class': CFG.num_class, # multiclassなら必要
        'eval_metric': 'mae',
        'max_depth': cfg.max_depth,
        'subsample': cfg.subsample,
        'colsample_bytree': cfg.colsample_bytree,
        'min_child_weight': cfg.min_data_in_leaf,
    }
    save_log_dir = SAVE_DIR / 'log'
    save_log_dir.mkdir(exist_ok=True, parents=True)

    model = XGBoost(
                xgb_params=xgb_params,
                save_dir=save_log_dir,
                model_name=model_name
    )

    return model

def get_fit_params(cfg, model_name):
    params = {
        'num_boost_round': 100000
    }
    return params

def get_result(result_df):
    pred_cols = [f'pred_{i}' for i in TARGET_COLUMNS]

    preds = result_df[pred_cols].values
    labels = result_df[TARGET_COLUMNS].values

    eval_func = eval('mae')
    best_score = eval_func(labels, preds)

    print(f'best_score: {best_score:<.4f}')
    return best_score


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    raw_train_df = pd.read_csv(ORIGINAL_DATA_DIR / 'train_features.csv')
    raw_test_df = pd.read_csv(ORIGINAL_DATA_DIR / 'test_features.csv')
    ss_df = pd.read_csv(ORIGINAL_DATA_DIR / 'atmaCup18__sample_submit.csv')

    seed_everything(cfg.seed)

    y = raw_train_df[TARGET_COLUMNS]
    train_with_fold_df = split_data(cfg, raw_train_df)

    if cfg.train:
        oof_predictions = np.zeros((raw_train_df.shape[0], len(TARGET_COLUMNS)))
        test_predictions = np.zeros((raw_test_df.shape[0], len(TARGET_COLUMNS)))
        
        times = [0, 1, 2, 3, 4, 5]
        coors = ['x', 'y', 'z']

        for time_idx, time in enumerate(times):
            # preprocess
            if cfg.use_pre_pred:
                # 一つ前に予測した値を特徴量に追加
                pre_pred_columns = [f'pre_pred_{r}_{i}' for i in range(time_idx) for r in coors]
                train_pre_pred_df = pd.DataFrame(oof_predictions[:, :time_idx*3], columns=pre_pred_columns)
                test_pre_pred_df = pd.DataFrame(test_predictions[:, :time_idx*3], columns=pre_pred_columns)
            else:
                pre_pred_columns = []
                train_pre_pred_df = None
                test_pre_pred_df = None

            for fold in range(cfg.n_folds):
                train_df = train_with_fold_df.copy()
                test_df = raw_test_df.copy()

                train_indices = train_df['fold'] != fold
                valid_indices = train_df['fold'] == fold
                
                # pre_pred
                train_df = pd.concat([train_df, train_pre_pred_df], axis=1)
                test_df = pd.concat([test_df, test_pre_pred_df], axis=1)

                train_df, common_num_columns = common_preprocess(train_df)
                test_df, _ = common_preprocess(test_df)

                # traffic_light
                if cfg.use_traffic_light:
                    train_df, test_df, traffic_columns = add_traffic_light_feature(train_df, test_df)

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
                use_shift_columns = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brakePressed', 'gas', 'gasPressed',  'leftBlinker', 'rightBlinker']
                if cfg.oof_shift:
                    use_shift_columns += oof_feat_columns
                train_df, shift_columns = make_shift_feature(train_df, use_shift_columns)
                test_df, shift_columns = make_shift_feature(test_df, use_shift_columns)

                # feature block
                num_columns = common_num_columns
                num_columns += oof_feat_columns
                num_columns += shift_columns
                num_columns += epipolar_columns
                num_columns += pre_pred_columns
                if cfg.use_traffic_light:
                    num_columns += traffic_columns

                agg_num_columns = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'gas']

                cat_label_columns = ['gearShifter']
                cat_count_columns = []
                cat_te_columns = []

                # drop duplicate columns
                tmp = train_df.copy()
                old_columns = set(tmp.columns)
                tmp = tmp.T.drop_duplicates().T
                new_columns = set(tmp.columns)
                drop_columns = old_columns - new_columns
                del tmp
                for col in drop_columns:
                    if col in num_columns:
                        num_columns.remove(col)
                    if col in agg_num_columns:
                        agg_num_columns.remove(col)
                    if col in cat_label_columns:
                        cat_label_columns.remove(col)
                    if col in cat_count_columns:
                        cat_count_columns.remove(col)
                    if col in cat_te_columns:
                        cat_te_columns.remove(col)

                test_df = test_df.drop(drop_columns, axis=1)

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
    
                gc.collect()

                for coor_idx, corr in enumerate(coors):
                    target_column = f'{corr}_{time}'
                    target_idx = time_idx * 3 + coor_idx

                    print(f'fold: {fold}, target_column: {target_column}')

                    x_train = train_feat.loc[train_indices]
                    x_valid = train_feat.loc[valid_indices]
                    y_train = train_df.loc[train_indices, target_column]
                    y_valid = train_df.loc[valid_indices, target_column]

                    model_name = f'xgb_{target_column}'  # モデル名をXGBoost用に変更
                    model = get_model(cfg, model_name)

                    fit_params = get_fit_params(cfg, model_name)

                    fit_params_fold = fit_params.copy()
                    fit_params_fold['eval_set'] = [(x_valid, y_valid)]

                    model.fit(x_train, y_train, **fit_params_fold)

                    if hasattr(model, 'save'):
                        model.save(fold)

                    if hasattr(model, 'plot_importance'):
                        model.plot_importance(fold)

                    oof_predictions[valid_indices, target_idx] = model.predict(x_valid)
                    test_predictions[:, target_idx] += model.predict(test_feat)

                eval_func = eval('mae')
                score_fold = eval_func(y.loc[valid_indices, target_column].values, oof_predictions[valid_indices, target_idx])
                print(f'fold: {fold}, column: {target_column}, score: {score_fold:<.4f}')

        eval_func = eval('mae')
        score = eval_func(y.values, oof_predictions)

        print(f'score: {score:<.4f}')

        pred_columns = [f'pred_{i}' for i in TARGET_COLUMNS]

        oof = train_df.copy()
        oof[pred_columns] = oof_predictions
        oof[TARGET_COLUMNS] = y

        oof_feat = train_feat.copy()
        oof_feat[pred_columns] = oof_predictions
        oof_feat[TARGET_COLUMNS] = y

        best_score = get_result(oof)
        print(f'best_score: {best_score:<.4f}')

        oof.to_csv(SAVE_DIR / 'oof.csv', index=False)
        oof_feat.to_csv(SAVE_DIR / 'oof_feat.csv', index=False)

        test_predictions /= cfg.n_folds

        test_feat[TARGET_COLUMNS] = test_predictions
        test_feat.to_csv(SAVE_DIR / 'submission_oof.csv', index=False)
        test_feat[TARGET_COLUMNS].to_csv(SAVE_DIR / f'submission_gbdt.csv', index=False)

        print('sample_sub_len: ', len(ss_df))
        print('sub_len: ', len(test_feat))

    
if __name__ == '__main__':
    main()