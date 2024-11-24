import numpy as np
import matplotlib.pyplot as plt
import gc
from PIL import Image
import requests
from transformers import pipeline
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd
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
# セグメンテーションの結果を取得
# =======================================
labels = [set(['car', 'truck', 'bus', 'motorcycle', 'bicycle'])]
    # set('traffic light'),
    # set('stop sign'),
    # set('light'),
    # set('road'),
    # set('person')


generator = pipeline(model="facebook/detr-resnet-50-panoptic", device=0)

for i in tqdm(range(len(df_feature))):
    row = df_feature.iloc[i]
    segmentation_dir = SAVE_DIR / "segmentation" / row.ID
    if segmentation_dir.exists():
        continue
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    for t, image_path_root in enumerate(image_path_root_list):
        img_pil = Image.open(image_path_root.format(ID=row.ID))
        preds = generator(img_pil)
        segmentation_path = segmentation_dir / Path(image_path_root).name
        # 画像サイズと同じサイズの配列を作成
        arrays = np.zeros((img_pil.size[1], img_pil.size[0])).astype(np.uint8)
        for pred in preds:
            pred_label = pred["label"]
            for i, target_label_set in enumerate(labels):
                if pred_label in target_label_set:
                    arrays += (i+1) * np.array(pred["mask"])
        # np.save(segmentation_path, arrays)
        # 1次元のImage.pillに変換
        img_pil = Image.fromarray(arrays)
        img_pil.save(segmentation_path)