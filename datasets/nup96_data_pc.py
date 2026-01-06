"""
NUP96点云数据集
用于PVCNN生成式模型训练

支持两种模式:
1. 从预处理好的H5文件加载
2. 实时从清洗后的CSV文件随机裁剪生成
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import h5py
import logging
from typing import Optional, List, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    load_csv_pointcloud, 
    smart_crop_with_augmentation,
    normalize_pointcloud,
    denormalize_pointcloud,
    augment_pointcloud
)


class NUP96PointClouds(Dataset):
    """
    NUP96核孔复合物点云数据集
    
    Args:
        mode: 数据加载模式
            - 'h5': 从预处理好的H5文件加载
            - 'realtime': 实时从CSV文件随机裁剪
        h5_path: H5文件路径（mode='h5'时必需）
        csv_dir: 清洗后CSV目录路径（mode='realtime'时必需）
        num_samples: 样本数量（mode='realtime'时必需）
        num_points: 每个样本的点数
        split: 数据集划分 ('train', 'val', 'test')
        normalize_per_shape: 是否按单个形状归一化
        normalize_std_per_axis: 是否按轴归一化标准差
        crop_ratio_x: x方向裁剪比例
        crop_ratio_y: y方向裁剪比例
        center_sampling: 是否从中心附近采样
        use_fps: 是否使用最远点采样
        random_subsample: 是否每次获取时随机采样
        all_points_mean: 预计算的均值（用于验证/测试集）
        all_points_std: 预计算的标准差
        enable_z_filter: 是否启用z轴离群点过滤
        z_filter_method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
        z_filter_params: 过滤参数字典
    """
    
    def __init__(self,
                 mode: str = 'h5',
                 h5_path: str = None,
                 csv_dir: str = None,
                 num_samples: int = 1024,
                 num_points: int = 2048,
                 split: str = 'train',
                 normalize_per_shape: bool = True,
                 normalize_std_per_axis: bool = False,
                 crop_ratio_x: float = 0.1,
                 crop_ratio_y: float = 0.1,
                 center_sampling: bool = True,
                 use_fps: bool = True,
                 random_subsample: bool = False,
                 all_points_mean: np.ndarray = None,
                 all_points_std: np.ndarray = None,
                 enable_z_filter: bool = True,
                 z_filter_method: str = 'iqr',
                 z_filter_params: dict = None):
        
        self.mode = mode
        self.split = split
        self.num_points = num_points
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.random_subsample = random_subsample
        
        # 实时裁剪参数
        self.crop_ratio_x = crop_ratio_x
        self.crop_ratio_y = crop_ratio_y
        self.center_sampling = center_sampling
        self.use_fps = use_fps
        
        # z轴离群点过滤参数
        self.enable_z_filter = enable_z_filter
        self.z_filter_method = z_filter_method
        self.z_filter_params = z_filter_params or {'iqr_k': 1.5}
        
        if mode == 'h5':
            self._load_from_h5(h5_path, all_points_mean, all_points_std)
        elif mode == 'realtime':
            self._setup_realtime(csv_dir, num_samples, all_points_mean, all_points_std)
        else:
            raise ValueError(f"Unknown mode: {mode}, expected 'h5' or 'realtime'")
        
        print(f"NUP96PointClouds ({mode} mode, {split}): {len(self)} samples, {self.num_points} points each")
    
    def _load_from_h5(self, h5_path: str, 
                     all_points_mean: np.ndarray = None,
                     all_points_std: np.ndarray = None):
        """从H5文件加载数据"""
        if h5_path is None:
            raise ValueError("h5_path is required for h5 mode")
        
        with h5py.File(h5_path, 'r') as f:
            self.all_points = f['points'][:]  # (B, N, 3)
            
            if all_points_mean is not None and all_points_std is not None:
                # 使用提供的归一化参数（用于验证/测试集）
                self.all_points_mean = all_points_mean
                self.all_points_std = all_points_std
            else:
                self.all_points_mean = f['mean'][:]
                self.all_points_std = f['std'][:]
            
            self.normalize_per_shape = f.attrs.get('normalize_per_shape', True)
        
        # 分割训练/测试点
        # 类似ShapeNet，80%用于训练，20%用于测试
        train_size = int(0.8 * self.all_points.shape[1])
        self.train_points = self.all_points[:, :train_size]
        self.test_points = self.all_points[:, train_size:]
        
        self.tr_sample_size = min(train_size, self.num_points)
        self.te_sample_size = min(self.all_points.shape[1] - train_size, self.num_points)
    
    def _setup_realtime(self, csv_dir: str, num_samples: int,
                       all_points_mean: np.ndarray = None,
                       all_points_std: np.ndarray = None):
        """设置实时裁剪模式"""
        if csv_dir is None:
            raise ValueError("csv_dir is required for realtime mode")
        
        self.csv_dir = csv_dir
        self.num_samples = num_samples
        
        # 加载所有源点云
        csv_files = sorted(glob(os.path.join(csv_dir, "*.csv")))
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {csv_dir}")
        
        self.source_points = []
        for csv_path in csv_files:
            points, _ = load_csv_pointcloud(csv_path, coord_cols=['x [nm]', 'y [nm]', 'z [nm]'])
            self.source_points.append(points)
        
        print(f"Loaded {len(self.source_points)} source point clouds for realtime cropping")
        
        # 预生成样本用于归一化参数计算
        if all_points_mean is None or all_points_std is None:
            self._pregenerate_samples()
        else:
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
            self.all_points = None
            self.train_points = None
            self.test_points = None
    
    def _pregenerate_samples(self):
        """预生成样本用于计算归一化参数"""
        # 数据增强配置
        augmentation_config = {
            'enable_rotation_z': True,
            'enable_rotation_xy': False,
            'enable_flip': True,
            'enable_jitter': False,
            'enable_scale': False,
        }
        
        samples = []
        while len(samples) < self.num_samples:
            source_idx = np.random.randint(0, len(self.source_points))
            points = self.source_points[source_idx]
            
            # 使用智能裁剪+数据增强+离群点过滤
            cropped = smart_crop_with_augmentation(
                points,
                target_points=self.num_points,
                initial_crop_ratio=self.crop_ratio_x,
                enable_augmentation=True,
                augmentation_config=augmentation_config,
                tolerance_high=1.5,
                tolerance_low=0.8,
                max_attempts=50,
                center_sampling=self.center_sampling,
                enable_z_filter=self.enable_z_filter,
                z_filter_method=self.z_filter_method,
                z_filter_params=self.z_filter_params
            )
            
            if cropped is not None:
                samples.append(cropped)
        
        self.all_points = np.stack(samples, axis=0)  # (B, N, 3)
        
        # 计算归一化参数
        normalized, self.all_points_mean, self.all_points_std = normalize_pointcloud(
            self.all_points,
            normalize_per_shape=self.normalize_per_shape,
            normalize_std_per_axis=self.normalize_std_per_axis
        )
        
        self.all_points = normalized
        
        # 分割训练/测试点
        train_size = int(0.8 * self.all_points.shape[1])
        self.train_points = self.all_points[:, :train_size]
        self.test_points = self.all_points[:, train_size:]
        
        self.tr_sample_size = min(train_size, self.num_points)
        self.te_sample_size = min(self.all_points.shape[1] - train_size, self.num_points)
    
    def get_pc_stats(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取点云的归一化参数"""
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, -1)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s
        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)
    
    def __len__(self) -> int:
        if self.mode == 'realtime' and self.all_points is None:
            return self.num_samples
        return len(self.train_points)
    
    def __getitem__(self, idx: int) -> dict:
        if self.mode == 'realtime' and self.all_points is None:
            # 实时生成模式
            return self._get_realtime_sample(idx)
        
        # 从预加载数据获取
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size, replace=False)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        
        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size, replace=False)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()
        
        m, s = self.get_pc_stats(idx)
        
        return {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': m,
            'std': s,
            'cate_idx': 0,  # NUP96只有一个类别
            'sid': 'nup96',
            'mid': f'sample_{idx:04d}'
        }
    
    def _get_realtime_sample(self, idx: int) -> dict:
        """实时生成样本"""
        # 数据增强配置
        augmentation_config = {
            'enable_rotation_z': True,
            'enable_rotation_xy': False,
            'enable_flip': True,
            'enable_jitter': False,
            'enable_scale': False,
        }
        
        # 随机选择源点云
        source_idx = np.random.randint(0, len(self.source_points))
        points = self.source_points[source_idx]
        
        # 使用智能裁剪+数据增强+离群点过滤
        cropped = None
        while cropped is None:
            cropped = smart_crop_with_augmentation(
                points,
                target_points=self.num_points,
                initial_crop_ratio=self.crop_ratio_x,
                enable_augmentation=True,
                augmentation_config=augmentation_config,
                tolerance_high=1.5,
                tolerance_low=0.8,
                max_attempts=50,
                center_sampling=self.center_sampling,
                enable_z_filter=self.enable_z_filter,
                z_filter_method=self.z_filter_method,
                z_filter_params=self.z_filter_params
            )
            if cropped is None:
                # 换一个源点云
                source_idx = np.random.randint(0, len(self.source_points))
                points = self.source_points[source_idx]
        
        # 归一化
        normalized, mean, std = normalize_pointcloud(
            cropped,
            normalize_per_shape=True,
            normalize_std_per_axis=self.normalize_std_per_axis
        )
        
        # 分割训练/测试
        train_size = int(0.8 * self.num_points)
        tr_out = torch.from_numpy(normalized[:train_size]).float()
        te_out = torch.from_numpy(normalized[train_size:]).float()
        
        return {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': mean,
            'std': std,
            'cate_idx': 0,
            'sid': 'nup96',
            'mid': f'realtime_{idx:04d}'
        }
    
    def renormalize(self, mean: np.ndarray, std: np.ndarray):
        """使用新的参数重新归一化"""
        if self.all_points is None:
            return
        
        # 反归一化
        self.all_points = denormalize_pointcloud(
            self.all_points, self.all_points_mean, self.all_points_std
        )
        
        # 使用新参数归一化
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - mean) / std
        
        train_size = int(0.8 * self.all_points.shape[1])
        self.train_points = self.all_points[:, :train_size]
        self.test_points = self.all_points[:, train_size:]


def get_nup96_datasets(data_root: str = "/home/djx/data/nup96-large",
                      mode: str = 'h5',
                      num_points: int = 2048,
                      num_samples: int = 1024,
                      enable_z_filter: bool = True,
                      z_filter_method: str = 'iqr',
                      z_filter_params: dict = None) -> Tuple[NUP96PointClouds, NUP96PointClouds]:
    """
    获取NUP96训练和验证数据集
    
    Args:
        data_root: 数据根目录
        mode: 加载模式 ('h5' 或 'realtime')
        num_points: 每个样本的点数
        num_samples: 样本数量（realtime模式）
        enable_z_filter: 是否启用z轴离群点过滤
        z_filter_method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
        z_filter_params: 过滤参数字典
        
    Returns:
        train_dataset, val_dataset
    """
    if z_filter_params is None:
        z_filter_params = {'iqr_k': 1.5}
    
    if mode == 'h5':
        h5_path = os.path.join(data_root, "6-pc-blocks.h5")
        train_dataset = NUP96PointClouds(
            mode='h5',
            h5_path=h5_path,
            num_points=num_points,
            split='train',
            normalize_per_shape=True,
            random_subsample=True
        )
        
        # 验证集使用相同的归一化参数
        val_dataset = NUP96PointClouds(
            mode='h5',
            h5_path=h5_path,
            num_points=num_points,
            split='val',
            normalize_per_shape=True,
            random_subsample=False,
            all_points_mean=train_dataset.all_points_mean,
            all_points_std=train_dataset.all_points_std
        )
    else:
        csv_dir = os.path.join(data_root, "1-clean-csv")
        train_dataset = NUP96PointClouds(
            mode='realtime',
            csv_dir=csv_dir,
            num_samples=num_samples,
            num_points=num_points,
            split='train',
            normalize_per_shape=True,
            random_subsample=True,
            enable_z_filter=enable_z_filter,
            z_filter_method=z_filter_method,
            z_filter_params=z_filter_params
        )
        
        val_dataset = NUP96PointClouds(
            mode='realtime',
            csv_dir=csv_dir,
            num_samples=num_samples // 4,
            num_points=num_points,
            split='val',
            normalize_per_shape=True,
            random_subsample=False,
            all_points_mean=train_dataset.all_points_mean,
            all_points_std=train_dataset.all_points_std,
            enable_z_filter=enable_z_filter,
            z_filter_method=z_filter_method,
            z_filter_params=z_filter_params
        )
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # 测试代码
    import time
    
    print("Testing H5 mode...")
    # 注意：需要先运行数据预处理流水线生成H5文件
    try:
        train_ds, val_ds = get_nup96_datasets(mode='h5')
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        
        sample = train_ds[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Train points shape: {sample['train_points'].shape}")
        print(f"Test points shape: {sample['test_points'].shape}")
    except Exception as e:
        print(f"H5 mode test failed (expected if H5 not generated yet): {e}")
    
    print("\nTesting realtime mode...")
    try:
        train_ds, val_ds = get_nup96_datasets(mode='realtime', num_samples=10)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        
        start = time.time()
        sample = train_ds[0]
        print(f"Sample generation time: {time.time() - start:.3f}s")
        print(f"Sample keys: {sample.keys()}")
        print(f"Train points shape: {sample['train_points'].shape}")
    except Exception as e:
        print(f"Realtime mode test failed: {e}")
