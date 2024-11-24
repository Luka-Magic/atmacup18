import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_features_with_all_data(train_df, test_df):
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    for gear_shift in all_df.gearShifter.value_counts().index:
        col_name = f"label_{gear_shift}"
        all_df[col_name] = 0
        all_df.loc[all_df.gearShifter == gear_shift, col_name] = 1
    all_df.drop(columns=['gearShifter', 'brake'], inplace=True)
    train_df = all_df[all_df.is_train == 1].drop(columns=['is_train'])
    test_df = all_df[all_df.is_train == 0].drop(columns=['is_train'])
    return train_df, test_df
    

def create_features(df):
    features = df.copy()

    remain_columns = set(features.columns)
    delete_cols = set()
    
    # Scene IDを抽出
    features['scene_id'] = features['ID'].apply(lambda x: x.split('_')[0])
    delete_cols.add('scene_id')

    # 平均化など
    features.vEgo = features.vEgo / 30
    features.aEgo = features.aEgo
    features.steeringAngleDeg = features.steeringAngleDeg / 400
    features.steeringTorque = features.steeringTorque / 600
    
    # fillna
    features.fillna(0, inplace=True)

    # 1. 基本的な統計量の計算（各シーンごと）
    scene_stats = features.groupby('scene_id').agg({
        'vEgo': ['mean', 'std', 'min', 'max', 'median'],
        'aEgo': ['mean', 'std', 'min', 'max', 'median'],
        'steeringAngleDeg': ['mean', 'std', 'min', 'max', 'median'],
        'steeringTorque': ['mean', 'std', 'min', 'max', 'median'],
        'gas': ['mean', 'max', 'sum']
    })

    # 2. 運転スタイルに関する特徴量
    features['harsh_acceleration'] = (features['aEgo'] > 2.0).astype(int)
    features['harsh_braking'] = (features['aEgo'] < -2.0).astype(int)

    # 急ハンドルの回数
    features['sharp_steering'] = (abs(features['steeringAngleDeg']) > 45).astype(int)

    # シーンごとの集計
    driving_style = features.groupby('scene_id').agg({
        'harsh_acceleration': 'sum',
        'harsh_braking': 'sum',
        'sharp_steering': 'sum'
    })

    # 3. 運転の滑らかさに関する特徴量
    features['speed_change_rate'] = features.groupby('scene_id')['vEgo'].diff()

    # ステアリングの変化率
    features['steering_change_rate'] = features.groupby('scene_id')['steeringAngleDeg'].diff()

    # 変化率の統計量
    smoothness_stats = features.groupby('scene_id').agg({
        'speed_change_rate': ['std', 'max', 'min'],
        'steering_change_rate': ['std', 'max', 'min']
    })

    # 4. ペダル操作に関する特徴量
    pedal_features = features.groupby('scene_id').agg({
        'brakePressed': 'sum',  # ブレーキを踏んでいる時間
        'gasPressed': 'sum',    # アクセルを踏んでいる時間
        'gas': 'mean'          # 平均アクセル開度
    })

    # 5. ウィンカー使用に関する特徴量
    blinker_features = features.groupby('scene_id').agg({
        'leftBlinker': 'sum',   # 左ウィンカーの使用時間
        'rightBlinker': 'sum'   # 右ウィンカーの使用時間
    })

    # 6. 高度な特徴量
    features['speed_zone'] = pd.qcut(features['vEgo'], q=5, labels=['very_slow', 'slow', 'medium', 'fast', 'very_fast'])
    speed_profile = pd.crosstab(features['scene_id'], features['speed_zone'], normalize='index')

    # 加速度プロファイルの特徴
    features['acc_zone'] = pd.qcut(features['aEgo'], q=5, labels=['hard_brake', 'mild_brake', 'neutral', 'mild_acc', 'hard_acc'])
    acc_profile = pd.crosstab(features['scene_id'], features['acc_zone'], normalize='index')

    # 7. 時系列特徴量
    def lag_features(group):
        return pd.Series({
            'speed_autocorr': group['vEgo'].autocorr(),
            'steering_autocorr': group['steeringAngleDeg'].autocorr()
        })

    time_features = features.groupby('scene_id').apply(lag_features)
    
    # 8. 運転の一貫性に関する特徴量
    def consistency_features(group):
        return pd.Series({
            'speed_consistency': 1 - (group['vEgo'].std() / (group['vEgo'].mean() + 1e-6)),
            'steering_consistency': 1 - (group['steeringAngleDeg'].std() / (abs(group['steeringAngleDeg']).mean() + 1e-6))
        })

    consistency = features.groupby('scene_id').apply(consistency_features)

    # 9. その他の特徴量
    others = features.copy()
    # steeringAngleDeg を度からラジアンに変換
    others["steeringAngleRad"] = np.deg2rad(others["steeringAngleDeg"])
    # 三角関数の特徴量を作成
    others["steeringAngle_sin"] = np.sin(others["steeringAngleRad"])
    others["steeringAngle_cos"] = np.cos(others["steeringAngleRad"])
    # 交互作用特徴量を作成
    others["speed_steering"] = others["vEgo"] * others["steeringAngleRad"]  # 速度とステアリング角度の組み合わせ
    others["acc_steeringTorque"] = others["aEgo"] * others["steeringTorque"]  # 加速度とステアリングトルクの組み合わせ
    # 対数変換
    others["vEgo_positive"] = others["vEgo"].clip(lower=0) + 1e-6
    others["log_vEgo"] = np.log(others["vEgo_positive"])
    # 加速度の変化率
    others["jerk"] = others.groupby("scene_id")["aEgo"].diff()
    # ステアリング角度とトルクの変化率
    others["steeringAngleRate"] = others.groupby("scene_id")["steeringAngleRad"].diff()
    others["steeringTorqueRate"] = others.groupby("scene_id")["steeringTorque"].diff()
    # 二乗・絶対値特徴量
    others["vEgo_squared"] = others["vEgo"] ** 2
    others["steeringAngleRad_squared"] = others["steeringAngleRad"] ** 2
    others["aEgo_squared"] = others["aEgo"] ** 2
    # 移動平均や移動和
    others["vEgo_roll_mean"] = others.groupby("scene_id")["vEgo"].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
    others["aEgo_roll_mean"] = others.groupby("scene_id")["aEgo"].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)

    others = others[['steeringAngleRad', 'steeringAngle_sin', 'steeringAngle_cos', 'speed_steering', 'acc_steeringTorque', 'vEgo_positive', 'log_vEgo', 'jerk', 'steeringAngleRate', 'steeringTorqueRate', 'vEgo_squared', 'steeringAngleRad_squared', 'aEgo_squared', 'vEgo_roll_mean', 'aEgo_roll_mean']]

    # 全ての特徴量を結合
    all_features = pd.concat([
        scene_stats,
        driving_style,
        smoothness_stats,
        pedal_features,
        blinker_features,
        speed_profile,
        acc_profile,
        time_features,
        consistency
    ], axis=1)

    # カラム名を平坦化
    all_features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in all_features.columns]
    # 重複を削除
    all_features = all_features.loc[:,~all_features.columns.duplicated()]

    df_feature = features[list(remain_columns) + ['scene_id']].merge(all_features.fillna(0), left_on="scene_id", right_index=True)

    df_feature = pd.concat([df_feature, others], axis=1)
    
    df_feature.drop(columns=delete_cols, inplace=True)
    return df_feature

if __name__ == '__main__':
    from pathlib import Path
    ROOT_DIR = Path.cwd().parents[2]
    df = pd.read_csv(ROOT_DIR / 'data/original_data/atmaCup#18_dataset/train_features.csv')
    features = create_features(df)
    print(features.columns)
    features.to_csv('./features.csv')