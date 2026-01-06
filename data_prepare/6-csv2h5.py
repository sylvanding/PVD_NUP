#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
6-csv2h5.py: 将CSV点云块归一化并保存为H5格式

功能:
1. 读取5-random-crop-batch-csv/目录下的点云块
2. 进行归一化（参考ShapeNet15kPointClouds的归一化方法）
3. 保存为H5格式，同时保存归一化参数
4. 适配PVCNN generation模型的输入格式
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
from glob import glob
import logging
from typing import Tuple, List

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    load_csv_pointcloud, setup_logging,
    normalize_pointcloud, denormalize_pointcloud
)


def load_all_pointclouds(input_dir: str, 
                        coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                        logger: logging.Logger = None) -> Tuple[np.ndarray, List[str]]:
    """
    加载目录下所有点云并堆叠
    
    Args:
        input_dir: 输入目录
        coord_cols: 坐标列名
        logger: 日志记录器
        
    Returns:
        all_points: (B, N, 3) numpy数组
        filenames: 文件名列表
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    all_points = []
    filenames = []
    
    for csv_path in csv_files:
        points, _ = load_csv_pointcloud(csv_path, coord_cols=coord_cols)
        all_points.append(points)
        filenames.append(os.path.basename(csv_path))
    
    # 检查点数是否一致
    num_points = [p.shape[0] for p in all_points]
    if len(set(num_points)) > 1:
        logger.warning(f"点云点数不一致: {set(num_points)}")
        # 找到最小点数，截断到相同大小
        min_points = min(num_points)
        logger.warning(f"截断到最小点数: {min_points}")
        all_points = [p[:min_points] for p in all_points]
    
    all_points = np.stack(all_points, axis=0).astype(np.float32)  # (B, N, 3)
    logger.info(f"加载完成: {all_points.shape}")
    
    return all_points, filenames


def save_to_h5(output_path: str,
              points: np.ndarray,
              mean: np.ndarray,
              std: np.ndarray,
              filenames: List[str] = None,
              normalize_per_shape: bool = True,
              logger: logging.Logger = None):
    """
    保存点云到H5文件
    
    Args:
        output_path: 输出文件路径
        points: 归一化后的点云 (B, N, 3)
        mean: 均值
        std: 标准差
        filenames: 原始文件名列表
        normalize_per_shape: 归一化方式
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 保存归一化后的点云
        f.create_dataset('points', data=points, dtype=np.float32)
        
        # 保存归一化参数
        f.create_dataset('mean', data=mean, dtype=np.float32)
        f.create_dataset('std', data=std, dtype=np.float32)
        
        # 保存元数据
        f.attrs['num_samples'] = points.shape[0]
        f.attrs['num_points'] = points.shape[1]
        f.attrs['normalize_per_shape'] = normalize_per_shape
        
        # 保存文件名（如果提供）
        if filenames is not None:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('filenames', data=filenames, dtype=dt)
    
    logger.info(f"已保存到: {output_path}")
    logger.info(f"  样本数: {points.shape[0]}")
    logger.info(f"  点数: {points.shape[1]}")
    logger.info(f"  归一化方式: {'per_shape' if normalize_per_shape else 'global'}")


def load_from_h5(h5_path: str, 
                logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    从H5文件加载点云数据
    
    Args:
        h5_path: H5文件路径
        logger: 日志记录器
        
    Returns:
        points: 归一化后的点云 (B, N, 3)
        mean: 均值
        std: 标准差
        metadata: 元数据字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    with h5py.File(h5_path, 'r') as f:
        points = f['points'][:]
        mean = f['mean'][:]
        std = f['std'][:]
        
        metadata = {
            'num_samples': f.attrs['num_samples'],
            'num_points': f.attrs['num_points'],
            'normalize_per_shape': f.attrs['normalize_per_shape'],
        }
        
        if 'filenames' in f:
            metadata['filenames'] = [fn.decode('utf-8') if isinstance(fn, bytes) else fn 
                                    for fn in f['filenames'][:]]
    
    logger.info(f"从 {h5_path} 加载:")
    logger.info(f"  样本数: {metadata['num_samples']}")
    logger.info(f"  点数: {metadata['num_points']}")
    
    return points, mean, std, metadata


def csv_to_h5(input_dir: str, output_path: str,
             normalize_per_shape: bool = True,
             normalize_std_per_axis: bool = False,
             logger: logging.Logger = None):
    """
    将CSV点云转换为归一化的H5格式
    
    Args:
        input_dir: CSV文件目录
        output_path: H5输出文件路径
        normalize_per_shape: 是否按单个形状归一化
        normalize_std_per_axis: 是否按轴归一化标准差
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 加载所有点云
    all_points, filenames = load_all_pointclouds(input_dir, logger=logger)
    
    # 归一化
    logger.info("正在归一化点云...")
    normalized_points, mean, std = normalize_pointcloud(
        all_points,
        normalize_per_shape=normalize_per_shape,
        normalize_std_per_axis=normalize_std_per_axis
    )
    
    # 输出统计信息
    logger.info(f"归一化前 - 范围: [{all_points.min():.2f}, {all_points.max():.2f}]")
    logger.info(f"归一化后 - 范围: [{normalized_points.min():.4f}, {normalized_points.max():.4f}]")
    logger.info(f"归一化后 - 均值: {normalized_points.mean():.6f}, 标准差: {normalized_points.std():.6f}")
    
    # 保存
    save_to_h5(
        output_path, normalized_points, mean, std,
        filenames=filenames,
        normalize_per_shape=normalize_per_shape,
        logger=logger
    )


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "5-random-crop-batch-csv")
    OUTPUT_PATH = os.path.join(DATA_ROOT, "6-pc-blocks.h5")
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始将CSV点云转换为H5格式...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出文件: {OUTPUT_PATH}")
    
    # 执行转换
    # 使用全局归一化（normalize_per_shape=False）
    # 这样所有点云共享同一组归一化参数（mean, std），
    # 便于在生成模型推理时使用固定参数进行反归一化
    csv_to_h5(
        INPUT_DIR, OUTPUT_PATH,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        logger=logger
    )
    
    logger.info("转换完成!")
    
    # 验证：加载并检查
    logger.info("\n验证H5文件...")
    points, mean, std, metadata = load_from_h5(OUTPUT_PATH, logger=logger)
    
    # 反归一化测试
    logger.info("\n反归一化测试...")
    denormalized = denormalize_pointcloud(points, mean, std)
    logger.info(f"反归一化后 - 范围: [{denormalized.min():.2f}, {denormalized.max():.2f}]")


if __name__ == "__main__":
    main()
