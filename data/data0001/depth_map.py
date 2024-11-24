import os
import sys
from glob import glob
from PIL import Image
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from tqdm import tqdm

# ==================================
# Dataの前処理に関するConfig
# ==================================
EXP_DIR = Path.cwd()
EXP_ID = EXP_DIR.name
ROOT_DIR = EXP_DIR.parents[2]
DATA_DIR = ROOT_DIR / 'data'
ORIGINAL_DATA_DIR = DATA_DIR / 'original_data'
SAVE_DIR = ROOT_DIR / 'data' / 'created_data' / EXP_ID
SAVE_DIR.mkdir(exist_ok=True, parents=True)

def get_relative_path(path):
    return os.path.join(ORIGINAL_DATA_DIR, path)

# 画像へのパス
image_path_root_list = [
    get_relative_path("atmaCup#18_dataset/images/{ID}/image_t.png"),
    get_relative_path("atmaCup#18_dataset/images/{ID}/image_t-0.5.png"),
    get_relative_path("atmaCup#18_dataset/images/{ID}/image_t-1.0.png")
]

# 特徴量のパス
train_feature_path = get_relative_path("atmaCup#18_dataset/train_features.csv")
traffic_light_path = get_relative_path("atmaCup#18_dataset/traffic_lights/{ID}.json")

# 信号機の情報へのパス
test_feature_path = get_relative_path("atmaCup#18_dataset/test_features.csv")

# ========================================
# DataFrameの読み込み
# ========================================
df_feature_train = pd.read_csv(train_feature_path)
df_feature_test = pd.read_csv(test_feature_path)

# =======================================
# 画像のパスの追加
# =======================================
df_feature_train["img_path_t_00"] = [image_path_root_list[0].format(ID=ID) for ID in df_feature_train.ID]
df_feature_train["img_path_t_05"] = [image_path_root_list[1].format(ID=ID) for ID in df_feature_train.ID]
df_feature_train["img_path_t_10"] = [image_path_root_list[2].format(ID=ID) for ID in df_feature_train.ID]

df_feature_test["img_path_t_00"] = [image_path_root_list[0].format(ID=ID) for ID in df_feature_test.ID]
df_feature_test["img_path_t_05"] = [image_path_root_list[1].format(ID=ID) for ID in df_feature_test.ID]
df_feature_test["img_path_t_10"] = [image_path_root_list[2].format(ID=ID) for ID in df_feature_test.ID]

df_feature = pd.concat([df_feature_train, df_feature_test], axis=0, ignore_index=True)

# =======================================
# Depth Mapの生成と保存
# =======================================
depth_anything_v2 = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

for i in tqdm(range(len(df_feature))):
    row = df_feature.iloc[i]
    depth_dir = SAVE_DIR / "depth" / row.ID
    if depth_dir.exists():
        continue
    depth_dir.mkdir(parents=True, exist_ok=True)

    for t, image_path_root in enumerate(image_path_root_list):
        img_pil = Image.open(image_path_root.format(ID=row.ID))
        pred = depth_anything_v2(img_pil)
        depth_path = depth_dir / Path(image_path_root).name
        pred["depth"].save(depth_path)