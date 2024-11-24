import hydra
import io
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import gc
import json
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import DictConfig, OmegaConf

import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Callable
from sklearn.model_selection import GroupKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import AverageMeter, seed_everything
from get_feature import create_features_with_all_data, create_features
import wandb
import torchinfo
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

from traffic_light_to_img import TrafficLightMaskGenerator

EXP_DIR = Path.cwd()
EXP_ID = EXP_DIR.name
ROOT_DIR = EXP_DIR.parents[2]
DATA_DIR = ROOT_DIR / 'data'
SAVE_DIR = ROOT_DIR / 'outputs' / 'exp' / EXP_ID
SAVE_DIR.mkdir(exist_ok=True, parents=True)
WANDB_DIR = SAVE_DIR / 'wandb'
WANDB_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_DATA_DIR = DATA_DIR / 'original_data/atmaCup#18_dataset'
CREATED_DATA_DIR = DATA_DIR / 'created_data'

ID_COLUMNS = ['ID']
META_COLUMNS = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake', 'brakePressed', 'gas', 'gasPressed', 'gearShifter', 'leftBlinker', 'rightBlinker']
TARGET_COLUMNS = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']


def split_data(cfg, df, fold):
    scene_ser = df['ID'].apply(lambda x: x.split('_')[0])

    group_kfold = GroupKFold(n_splits=cfg.n_folds)
    for ifold, (train_index, valid_index) in enumerate(group_kfold.split(df, groups=scene_ser)):
        if ifold == fold:
            train_df = df.iloc[train_index]
            valid_df = df.iloc[valid_index]
            break
    return train_df, valid_df


class Atma18Dataset(Dataset):
    def __init__(self,
                 cfg,
                 df,
                 feat_columns=None,
                 transform=None,
                 return_label=True
        ):
        super().__init__()
        self.df = df
        self.feat_columns = list(feat_columns)
        self.transform = transform
        self.return_label = return_label
        if cfg.use_traffic_light:
            self.use_traffic_light = True
            self.img_h = cfg.img_h
            self.img_w = cfg.img_w
            self.traffic_light_generator = TrafficLightMaskGenerator(image_size=[self.img_h, self.img_w], normalize=True)
        if cfg.use_depth:
            self.use_depth = True
            self.depth_dir = CREATED_DATA_DIR / cfg.depth_dir_id / "depth"
        if cfg.use_vehicle_segmentation:
            self.use_vehicle_segmentation = True
            self.vehicle_segmentation_dir = CREATED_DATA_DIR / cfg.vehicle_segmentation_dir_id / "segmentation"

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # image
        img_seq = []
        for postfix in ['-1.0', '-0.5', '']:
            img = np.array(Image.open(str(ORIGINAL_DATA_DIR / "images" / row['ID'] / f'image_t{postfix}.png')))
            img_seq.append(img)

        img = np.concatenate(img_seq, axis=2)

        if self.transform:
            img = self.transform(image=img)['image']

        # depth
        if self.use_depth:
            depth_seq = []
            for postfix in ['-1.0', '-0.5', '']:
                depth = np.array(Image.open(str(self.depth_dir / row['ID'] / f'image_t{postfix}.png')))
                depth = depth.astype(np.float32)[:, :, np.newaxis]
                depth = depth / 255.0
                depth_seq.append(depth)
            depth_seq = np.concatenate(depth_seq, axis=2)
            depth_seq = depth_seq.transpose(2, 0, 1)
            depths = torch.tensor(depth_seq)
            img = torch.cat([img, depths], dim=0)
        
        # traffic light
        if self.traffic_light_generator is not None:
            with open(ORIGINAL_DATA_DIR / 'traffic_lights' / f'{row["ID"]}.json', 'r') as f:
                traffic_light_json = json.load(f)
            traffic_image = self.traffic_light_generator.generate_masks(traffic_light_json)
            traffic_image = traffic_image.astype(np.float32).transpose(2, 0, 1)
            # to tensor
            traffic_image = torch.tensor(traffic_image)
            img = torch.cat([img, traffic_image], dim=0)
        
        # vehicle segmentation
        if self.use_vehicle_segmentation:
            vehicle_segmentation_seq = []
            for postfix in ['-1.0', '-0.5', '']:
                vehicle_segmentation = np.array(Image.open(str(self.vehicle_segmentation_dir / row['ID'] / f'image_t{postfix}.png')))
                vehicle_segmentation = vehicle_segmentation.astype(np.float32)[:, :, np.newaxis]
                vehicle_segmentation = vehicle_segmentation / 255.0
                vehicle_segmentation_seq.append(vehicle_segmentation)
            vehicle_segmentation_seq = np.concatenate(vehicle_segmentation_seq, axis=2)
            vehicle_segmentation_seq = vehicle_segmentation_seq.transpose(2, 0, 1)
            vehicle_segmentations = torch.tensor(vehicle_segmentation_seq)
            img = torch.cat([img, vehicle_segmentations], dim=0)

        # meta
        meta = row[self.feat_columns].values.astype(np.float32)

        if self.return_label:
            target = row[TARGET_COLUMNS].values.astype(np.float32)
            return img, meta, target
        else:
            return img, meta


def get_transforms(cfg, phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(cfg.img_h, cfg.img_w),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ], p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406] * 3, std=[0.229, 0.224, 0.225] * 3),
            ToTensorV2(),
        ])
    elif phase == 'valid':
        return A.Compose([
            A.Resize(cfg.img_h, cfg.img_w),
            A.Normalize(mean=[0.485, 0.456, 0.406] * 3, std=[0.229, 0.224, 0.225] * 3),
            ToTensorV2(),
        ])


def get_dataloader(cfg, df, feat_columns, phase):
    if phase == 'train':
        dataset = Atma18Dataset(
            cfg,
            df,
            feat_columns,
            transform=get_transforms(cfg, phase),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train_bs,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    elif phase == 'valid':
        dataset = Atma18Dataset(
            cfg,
            df,
            feat_columns,
            transform=get_transforms(cfg, phase),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.valid_bs,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return dataloader


class Atma18Model(nn.Module):
    def __init__(
            self,
            name: str = 'resnet18',
            img_in_chans: int = 9,
            output_dim: int = 18,
            meta_in_chans: int = 11,
            meta_out_chans: int = 128,
        ):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=0, in_chans=img_in_chans)
        self.mlp = nn.Sequential(
            nn.Linear(meta_in_chans, 128),
            nn.ReLU(),
            nn.Linear(128, meta_out_chans),
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.model.num_features + meta_out_chans, output_dim)

    def feature(self, x):
        x = self.model(x)
        return x

    def forward(self, x, meta):
        x = self.model(x)
        meta = self.mlp(meta)

        x = torch.cat([x, meta], dim=-1)
        x = self.relu(x)
        x = self.fc(x)
        return x

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def mae(gt: np.array, pred: np.array) -> float:
    abs_diff = np.abs(gt - pred)
    score = np.mean(abs_diff.reshape(-1, ))
    return float(score)


def train_one_epoch(
    cfg,
    fold: int,
    epoch: int,
    save_dir: Path,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    metric_fn: Callable,# ただの関数
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    scaler: torch.cuda.amp.GradScaler,
    total_images: int,
):
    model.train()
    train_losses  = AverageMeter()
    train_score = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for step, (imgs, metas, labels) in pbar:
        imgs = imgs.to(device).float()
        metas = metas.to(device).float()
        labels = labels.to(device).float()
        bs = imgs.size(0)
        lr = get_lr(optimizer)

        optimizer.zero_grad()
        with autocast(enabled=cfg.use_amp):
            output = model(imgs, metas)
            loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        if cfg.scheduler_step_time == 'steps':
            scheduler.step()
        
        # update meters
        train_losses.update(loss.item(), bs)
        score = metric_fn(
            output.detach().cpu().numpy(),
            labels.detach().cpu().numpy()
        )
        train_score.update(score, bs)

        # update pbar
        pbar.set_description(
            f'[TRAIN epoch {epoch}/{cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(loss=train_losses.avg, score=train_score.avg, lr=lr))

        # wandb
        total_images += bs
        wandb.log({
            'train_loss': train_losses.avg,
            'train_score': train_score.avg,
            'lr': lr,
            'total_images': total_images,
        })
    
    if cfg.scheduler_step_time == 'epoch':
        scheduler.step()

    return train_losses.avg, train_score.avg, total_images


def valid_one_epoch(
    cfg,
    fold: int,
    epoch: int,
    save_dir: Path,
    valid_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    metric_fn: Callable,
    device: torch.device,
):
    model.eval()

    valid_losses = AverageMeter()
    valid_score = AverageMeter()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for step, (imgs, metas, labels) in pbar:
        imgs = imgs.to(device).float()
        metas = metas.to(device).float()
        labels = labels.to(device).float()
        bs = imgs.size(0)

        with torch.no_grad():
            output = model(imgs, metas)
            loss = loss_fn(output, labels)

        # update meters
        valid_losses.update(loss.item(), bs)
        score = metric_fn(
            output.detach().cpu().numpy(),
            labels.detach().cpu().numpy()
        )
        valid_score.update(score, bs)
        
        # update pbar
        pbar.set_description(
            f'[VALID epoch {epoch}/{cfg.n_epochs}]')
        pbar.set_postfix(OrderedDict(loss=valid_losses.avg, score=valid_score.avg))
    return valid_losses.avg, valid_score.avg


def test_function(
    cfg,
    fold: int,
    save_dir: Path,
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
):
    model.eval()
    preds = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, (imgs, metas) in pbar:
        imgs = imgs.to(device).float()
        metas = metas.to(device).float()

        with torch.no_grad():
            output = model(imgs, metas)
            preds.append(output.detach().cpu().numpy())
    preds = np.concatenate(preds)
    return preds

def test_function_for_feature(
    cfg,
    fold: int,
    save_dir: Path,
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device
):
    model.eval()
    preds = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, (imgs, metas) in pbar:
        imgs = imgs.to(device).float()
        metas = metas.to(device).float()

        with torch.no_grad():
            output = model.feature(imgs)
            preds.append(output.detach().cpu().numpy())
    preds = np.concatenate(preds).astype(np.float16)
    return preds



def send_image(df):
    intrinsic_matrix = np.array([[226.16438356, 0., 63.62426614],
                             [0., 224.82352941, 11.76],
                             [0., 0., 1.]])

    def camera_to_image(P_camera, intrinsic_matrix):
        P_image_homogeneous = np.dot(intrinsic_matrix, P_camera)
        P_image = P_image_homogeneous[:2] / P_image_homogeneous[2]
        return P_image
    
    def project_trajectory_to_image_coordinate_system(trajectory: np.ndarray, intrinsic_matrix: np.ndarray):
        """車両中心座標系で表現されたtrajectoryをカメラ座標系に投影する"""
        # カメラの設置されている高さ(1.22m)まで座標系をズラす
        trajectory_with_offset = trajectory.copy()
        trajectory_with_offset[:, 2] = trajectory_with_offset[:, 2] + 1.22

        # 座標の取り方を変更する
        road_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        trajectory_camera = trajectory_with_offset @ road_to_camera
        trajectory_image = np.array([camera_to_image(p, intrinsic_matrix) for p in trajectory_camera if p[2] > 0])
        return trajectory_image


    def overlay_trajectory(
            trajectory_gt: np.ndarray,
            trajectory_pred: np.ndarray,
            image_id: str,
            intrinsic_matrix: np.ndarray,
            save_path=None,
            score=None,
            figsize=(5.12, 2.56),
        ):
        images_dir = ORIGINAL_DATA_DIR / "images"
        image = Image.open(images_dir / image_id / "image_t.png")
        
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_axis_off()
        ax.imshow(image)
        try:
            trajectory_gt_img = project_trajectory_to_image_coordinate_system(trajectory_gt, intrinsic_matrix)
            ax.plot(
                trajectory_gt_img[:, 0],
                trajectory_gt_img[:, 1],
                marker="o",
                color="blue",
                alpha=1.0,
                markersize=3,
                linestyle="solid",
            )
        except:
            pass
        try:
            trajectory_pred_img = project_trajectory_to_image_coordinate_system(trajectory_pred, intrinsic_matrix)
            ax.plot(
                trajectory_pred_img[:, 0],
                trajectory_pred_img[:, 1],
                marker="o",
                color="red",
                alpha=1.0,
                markersize=3,
                linestyle="solid",
            )
        except:
            pass

        ax.set_xlim(0, 128)
        ax.set_ylim(64, 0)

        fig.savefig(save_path, format="png", dpi=240) #解像度は調整

    SAVE_IMAGE_DIR = SAVE_DIR / 'images'
    SAVE_IMAGE_DIR.mkdir(exist_ok=True, parents=True)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        id_ = row['ID']
        score = row['score']
        img = cv2.imread(str(ORIGINAL_DATA_DIR / 'images' / id_ / f'image_t.png'))
        trajectory_gt = row[TARGET_COLUMNS].values.reshape(6, 3)
        trajectory_pred = row[[f'pred_{col}' for col in TARGET_COLUMNS]].values.reshape(6, 3)
        
        save_path = SAVE_IMAGE_DIR / f'{id_}.png'
        if save_path.exists():
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay_trajectory(
            trajectory_gt,
            trajectory_pred,
            id_,
            intrinsic_matrix,
            save_path=save_path,
            score=score
        )
        if i % 100 == 0:
            # cacheの解法
            gc.collect()


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    wandb.login()

    raw_train_df = pd.read_csv(ORIGINAL_DATA_DIR / 'train_features.csv')
    raw_test_df = pd.read_csv(ORIGINAL_DATA_DIR / 'test_features.csv')
    ss_df = pd.read_csv(ORIGINAL_DATA_DIR / 'atmaCup18__sample_submit.csv')

    seed_everything(cfg.seed)

    df, test_df = create_features_with_all_data(raw_train_df, raw_test_df)

    device = torch.device(cfg.device)
    
    if cfg.train:
        for fold in cfg.use_folds:
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True)
            wandb.init(
                project=cfg.wandb_project,
                entity='luka-magic',
                name=EXP_ID if not cfg.debug else f'{EXP_ID}_debug',
                mode='online' if cfg.wandb else 'disabled',
                config=wandb.config,
                dir=WANDB_DIR,
                settings=wandb.Settings(disable_git=True, save_code=False)
            )
            wandb.config.fold = fold
            
            best_info = {
                'epoch': -1,
                'best_score': np.inf,
            }
            total_images = 0

            train_df, valid_df = split_data(cfg, df, fold)
            train_df = create_features(train_df)
            valid_df = create_features(valid_df)

            feat_columns = set(train_df.columns) - set(ID_COLUMNS + TARGET_COLUMNS)
            feat_columns = sorted(list(feat_columns))

            # 特徴量カラムは正規化
            # for col in feat_columns:
            #     # もしnanがあったら0にする
            #     if train_df[col].std() < 1e-7:
            #         train_df[col] = 0.0
            #     train_df[col] = (train_df[col] - train_df[col].mean()) / train_df[col].std()
            #     if valid_df[col].std() < 1e-7:
            #         valid_df[col] = 0.0
            #     valid_df[col] = (valid_df[col] - valid_df[col].mean()) / valid_df[col].std()

            train_df = train_df.fillna(0.0)
            valid_df = valid_df.fillna(0.0)


            wandb.config.n_meta_columns = len(feat_columns)
            
            img_in_chans = cfg.img_in_chans
            if cfg.use_traffic_light:
                img_in_chans += 8
            if cfg.use_depth:
                img_in_chans += 3
            if cfg.use_vehicle_segmentation:
                img_in_chans += 3

            train_dl = get_dataloader(cfg, train_df, feat_columns, 'train')
            valid_dl = get_dataloader(cfg, valid_df,feat_columns, 'valid')

            model = Atma18Model(
                name=cfg.model_name,
                img_in_chans=img_in_chans,
                output_dim=cfg.output_dim,
                meta_in_chans=len(feat_columns),
                meta_out_chans=cfg.meta_out_chans,
            )
            model.to(cfg.device)

            # torchinfo
            torchinfo.summary(model, input_size=[(cfg.train_bs, img_in_chans, cfg.img_h, cfg.img_w), (cfg.train_bs, len(feat_columns))])

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            if cfg.scheduler == 'OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=cfg.lr,
                    steps_per_epoch=len(train_dl),
                    epochs=cfg.n_epochs,
                    pct_start=cfg.pct_start,
                    div_factor=cfg.div_factor,
                    final_div_factor=cfg.final_div_factor,
                )
            elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=cfg.T_0,
                    T_mult=cfg.T_mult,
                    eta_min=cfg.eta_min,
                    last_epoch=-1,
                )
            loss_fn = nn.L1Loss()
            metric_fn = mae
            scaler = GradScaler(enabled=cfg.use_amp)

            for epoch in range(1, cfg.n_epochs+1):
                train_loss, train_score, total_images = train_one_epoch(
                    cfg,
                    fold,
                    epoch,
                    SAVE_DIR,
                    train_dl,
                    valid_dl,
                    model,
                    loss_fn,
                    metric_fn,
                    device,
                    optimizer,
                    scheduler,
                    scaler,
                    total_images
                )

                valid_loss, valid_score = valid_one_epoch(
                    cfg,
                    fold,
                    epoch,
                    SAVE_DIR,
                    valid_dl,
                    model,
                    loss_fn,
                    metric_fn,
                    device,
                )

                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_score': train_score,
                    'valid_loss': valid_loss,
                    'valid_score': valid_score,
                })
                print("\n" + "=" * 80)
                print(f'Fold {fold} | Epoch {epoch}/{cfg.n_epochs}')
                print(f'    TRAIN:')
                print(f'            loss: {train_loss:.6f}')
                print(f'            score: {train_score:.6f}')
                print(f'    VALID:')
                print(f'            loss: {valid_loss:.6f}')
                print(f'            score: {valid_score:.6f}')

                if valid_score < best_info['best_score']:
                    best_info['epoch'] = epoch
                    best_info['best_score'] = valid_score
                    torch.save(
                        {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'best_score': valid_score,
                        },
                        SAVE_DIR / f'fold{fold}_best.pth'
                    )

                    wandb.run.summary['best_score'] = valid_score
                    print(f'    *Best score: {best_info["best_score"]:.6f} (epoch: {best_info["epoch"]})')
                else:
                    print(f'    Best score: {best_info["best_score"]:.6f} (epoch: {best_info["epoch"]})')
                print("=" * 80)
            del model, optimizer, scheduler, loss_fn, metric_fn, scaler, train_dl, valid_dl, train_df, valid_df, train_loss, train_score, valid_loss, valid_score
            gc.collect()
            torch.cuda.empty_cache()
            wandb.finish()
    
    # oof
    if cfg.oof:
        oof_df = None
        if cfg.oof_feature:
            oof_feature_df = None
        
        for fold in cfg.use_folds:
            _, valid_df = split_data(cfg, df, fold)
            valid_feat_df = create_features(valid_df)
            feat_columns = set(valid_feat_df.columns) - set(ID_COLUMNS + TARGET_COLUMNS)
            feat_columns = sorted(list(feat_columns))
            
            img_in_chans = cfg.img_in_chans
            if cfg.use_traffic_light:
                img_in_chans += 8
            if cfg.use_depth:
                img_in_chans += 3
            if cfg.use_vehicle_segmentation:
                img_in_chans += 3
            
            valid_ds = Atma18Dataset(
                cfg,
                valid_feat_df,
                feat_columns=feat_columns,
                transform=get_transforms(cfg, 'valid'),
                return_label=False
            )
            valid_dl = DataLoader(
                valid_ds,
                batch_size=cfg.valid_bs,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
                pin_memory=True
            )

            model = Atma18Model(
                name=cfg.model_name,
                img_in_chans=img_in_chans,
                output_dim=cfg.output_dim,
                meta_in_chans=len(feat_columns),
                meta_out_chans=cfg.meta_out_chans,
            )
            model.to(cfg.device)
            model.load_state_dict(torch.load(SAVE_DIR / f'fold{fold}_best.pth')['model'])
            
            oof_pred = test_function(
                cfg,
                fold,
                SAVE_DIR,
                valid_dl,
                model,
                device,
            )

            if cfg.oof_feature:
                oof_features = test_function_for_feature(
                    cfg,
                    fold,
                    SAVE_DIR,
                    valid_dl,
                    model,
                    device,
                )
            
            del model, valid_ds, valid_dl, _, valid_feat_df
            gc.collect()
            torch.cuda.empty_cache()

            one_oof_df = valid_df[TARGET_COLUMNS].copy()
            # column名にpostfixを追加
            pred_columns = [f'pred_{col}' for col in one_oof_df.columns]
            one_oof_df.columns = pred_columns

            one_oof_df.iloc[:, :] = oof_pred
            one_oof_df['fold'] = fold
            # valid_dfと結合
            one_oof_df = pd.concat([valid_df, one_oof_df], axis=1)

            if cfg.oof_feature:
                one_oof_feature_df = pd.DataFrame(oof_features, columns=[f'feature_{i}' for i in range(oof_features.shape[1])], index=valid_df.index)
                # valid_dfと結合
                one_oof_feature_df = pd.concat([one_oof_df, one_oof_feature_df], axis=1)
            
            # maeを計算
            one_oof_df['score'] = one_oof_df.apply(lambda x: mae(x[TARGET_COLUMNS].values, x[pred_columns].values), axis=1)
            print('=' * 80)
            print(f"Fold {fold} OOF MAE: {one_oof_df['score'].mean():.6f}")
            print('=' * 80)
            oof_df = pd.concat([oof_df, one_oof_df], axis=0)
            if cfg.oof_feature:
                oof_feature_df = pd.concat([oof_feature_df, one_oof_feature_df], axis=0)
                print(len(oof_df), len(oof_feature_df))
            
            if cfg.oof_save_image:
                send_image(
                    one_oof_df
                )
        print('=' * 80)
        print(f"OOF MAE: {oof_df['score'].mean():.6f}")
        print('=' * 80)
        oof_df.to_csv(SAVE_DIR / 'oof.csv')
        if cfg.oof_feature:
            oof_feature_df.to_csv(SAVE_DIR / 'oof_feature.csv')


    if cfg.test:
        test_df = create_features(test_df)
        feat_columns = set(test_df.columns) - set(ID_COLUMNS + TARGET_COLUMNS)
        feat_columns = sorted(list(feat_columns))

        # foldで平均を取る
        assert len(cfg.ensemble_folds) > 0, f"test_cfg.use_fold must be set"
        assert len(cfg.ensemble_folds) == len(cfg.ensemble_model_weight), f"len(test_cfg.use_folds) must be equal to len(test_cfg.model_weight)"

        if not all([(SAVE_DIR / f'fold{fold}_best.pth').exists() for fold in cfg.ensemble_folds]):
            print("Not all models exist")
            return

        test_preds = []
        test_features = []
        for i, fold in enumerate(cfg.ensemble_folds):
            print(f"score of fold{fold}: {torch.load(SAVE_DIR / f'fold{fold}_best.pth')['best_score']}")
            
            img_in_chans = cfg.img_in_chans
            if cfg.use_traffic_light:
                img_in_chans += 8
            if cfg.use_depth:
                img_in_chans += 3
            if cfg.use_vehicle_segmentation:
                img_in_chans += 3

            test_ds = Atma18Dataset(
                cfg,
                test_df,
                feat_columns,
                transform=get_transforms(cfg, 'valid'),
                return_label=False
            )
            test_dl = DataLoader(
                test_ds,
                batch_size=cfg.valid_bs,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
                pin_memory=True
            )
            model = Atma18Model(
                name=cfg.model_name,
                img_in_chans=img_in_chans,
                output_dim=cfg.output_dim,
                meta_in_chans=len(feat_columns),
                meta_out_chans=cfg.meta_out_chans,
            )
            model.to(cfg.device)
            model.load_state_dict(torch.load(SAVE_DIR / f'fold{fold}_best.pth')['model'])
            test_pred = test_function(
                cfg,
                fold,
                SAVE_DIR,
                test_dl,
                model,
                device,
            )
            if cfg.test_feature:
                test_feature = test_function_for_feature(
                    cfg,
                    fold,
                    SAVE_DIR,
                    test_dl,
                    model,
                    device
                )
            del model, test_ds, test_dl
            gc.collect()
            torch.cuda.empty_cache()
            test_preds.append(test_pred * cfg.ensemble_model_weight[i])
            if cfg.test_feature:
                test_features.append(test_feature * cfg.ensemble_model_weight[i])
            
            ss_fold_df = ss_df.copy()
            ss_fold_df.iloc[:, :] = test_pred
            ss_fold_df.to_csv(SAVE_DIR / f'submission_fold{fold}.csv', index=False)
            
            if cfg.test_feature:
                ss_feature_fold_df = pd.DataFrame(test_feature, columns=[f'feature_{i}' for i in range(test_feature.shape[1])])
                ss_feature_fold_df = pd.concat([ss_fold_df, ss_feature_fold_df], axis=1)
                ss_feature_fold_df.to_csv(SAVE_DIR / f'submission_feature_fold{fold}.csv', index=False)
        
        # model_weightで重み付け
        test_preds = np.sum(test_preds, axis=0) / np.sum(cfg.ensemble_model_weight)
        test_features = np.sum(test_features, axis=0) / np.sum(cfg.ensemble_model_weight)
        ss_df.iloc[:, :] = test_preds
        ss_df.to_csv(SAVE_DIR / 'submission.csv', index=False)

        if cfg.test_feature:
            ss_feature_df = pd.DataFrame(test_features, columns=[f'feature_{i}' for i in range(test_features.shape[1])])
            ss_feature_df = pd.concat([ss_df, ss_feature_df], axis=1)
            ss_feature_df.to_csv(SAVE_DIR / 'submission_feature.csv', index=False)

if __name__ == '__main__':
    main()

