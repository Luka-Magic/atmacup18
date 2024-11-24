import sys
import numpy as np
import numpy.linalg as LA
import cv2
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import OrderedDict

EXP_DIR = Path.cwd()
EXP_ID = EXP_DIR.name
ROOT_DIR = EXP_DIR.parents[2]
DATA_DIR = ROOT_DIR / 'data'
ORIGINAL_DATA_DIR = DATA_DIR / 'original_data/atmaCup#18_dataset'
SAVE_DIR = ROOT_DIR / 'data' / 'created_data' / EXP_ID
SAVE_DIR.mkdir(exist_ok=True, parents=True)

def calc_R(
        img_1,
        img_2
    ):
    #特徴検出の関数を設定 detector
    #--Detector character points
#     detector = cv2.AKAZE_create()
    detector = cv2.SIFT_create()

    #マッチング関数を設定 match
#     match = cv2.BFMatcher()
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    match = cv2.FlannBasedMatcher(index_params,search_params)

    k1, d1 = detector.detectAndCompute(img_1, None)
    k2, d2 = detector.detectAndCompute(img_2, None)

    #------
    try:
        matches = match.knnMatch(d1, d2, k=2)
    except:
        return None, 'match'

    #マッチングペアの確認：ペアの数が8以下なら、ストップ
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append(m)
    except:
        return None, 'match_result'

    MIN_MATCH_COUNT = 8
    if len(good) > MIN_MATCH_COUNT:
        ptsCAM1i = np.int32([ k1[m.queryIdx].pt for m in good ])
        ptsCAM2i = np.int32([ k2[m.trainIdx].pt for m in good ])
    else:
        return None, 'count'
    
    #カメラ内部パラメータを設定　K
#     w = 128*3
#     h = 64*3
#     tans = np.tan(np.deg2rad(36.54))
#     K = np.array([[w/2.0/tans,0,w/2.0],[0,w/2.0/tans,h/2.0],[0,0,1]])
    K = np.array([[226.16438356, 0., 63.62426614],
                             [0., 224.82352941, 11.76],
                             [0., 0., 1.]])
    
    F, mask = cv2.findFundamentalMat(ptsCAM2i, ptsCAM1i, cv2.FM_LMEDS)
    
    if F is None:
        return None, 'F'
     
    #基本行列の取得
    E = np.dot(np.dot(K.T,F),K)

    #Rの取得
    #次正方行列 U,Σ,V(転置行列)を求める
    U, S, Vt = LA.svd(E, full_matrices=True)

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = np.dot(np.dot(U,W),Vt)
    WT = W.T
    R2 = np.dot(np.dot(U,WT),Vt)

    #候補のRを選ぶ。トレースから推定
    if np.abs(np.trace(R1)) > np.abs(np.trace(R2)):
        R = R1
    else:
        R = R2

    #Rに反転が入ってた場合は、再反転して戻す。
    if np.trace(R) < 0 :
        R = -R
    return R, None

def main(
        restart=False,
        restart_from=0,
        phase='train'
    ):

    df = pd.read_csv(ORIGINAL_DATA_DIR / f'{phase}_features.csv')

    if restart:
        r_df = pd.read_csv(SAVE_DIR / f'{phase}_features_r_copy.csv')
    else:
        r_columns = [f'r{n+1}_{i//3}_{i%3}' for n in range(2) for i in range(9)]
        r_df = df[['ID']].copy()
        r_df[r_columns] = -1.0
        r_df = r_df.astype({'ID': str, **{col: np.float16 for col in r_columns}})

    errors = {
        'match': 0,
        'match_result': 0,
        'count': 0,
        'F': 0
    }
    pbar = tqdm(enumerate(df['ID']), total=len(df))
    none_counter = 0
    for i, id_ in pbar:
        if i < restart_from:
            continue
        imgs = []
        for postfix in ['-1.0', '-0.5', '']:
            img = cv2.imread(str(ORIGINAL_DATA_DIR / 'images' / id_ / f'image_t{postfix}.png'))[:, :, ::-1]
            img = cv2.resize(img, ((128*10), (64*10)))
            imgs.append(img)
        
        r1, error1 = calc_R(imgs[0], imgs[1])
        r2, error2 = calc_R(imgs[1], imgs[2])
        if r1 is None:
            none_counter += 1
            errors[error1] += 1
        else:
            r_df.iloc[i, 1:10] = r1.reshape(-1).astype(np.float16)
        if r2 is None:
            none_counter += 1
            errors[error2] += 1
        else:
            r_df.iloc[i, 10:19] = r2.reshape(-1).astype(np.float16)
        
        pbar.set_postfix(OrderedDict(
                none=f'{none_counter}/{i+1}',
                error_match=f'{errors["match"]}/{none_counter}',
                error_match_result=f'{errors["match_result"]}/{none_counter}',
                error_count=f'{errors["count"]}/{none_counter}',
                error_f=f'{errors["F"]}/{none_counter}',
            )
        )

        if i % 100 == 0:
            r_df.to_csv(SAVE_DIR / f'{phase}_features_r.csv', index=False)
    r_df.to_csv(SAVE_DIR / f'{phase}_features_r.csv', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', '-r', action='store_true')
    parser.add_argument('--restart_from', '-rf', type=int, default=0)
    parser.add_argument('--phase', '-p', type=str, default='train')
    args = parser.parse_args()
    main(args.restart, args.restart_from, args.phase)