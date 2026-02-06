#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3-ccp-csv2h5.py: 将CCP样本归一化并保存为H5格式

功能:
1. 读取outputs/{stage}/csv/目录下的CCP样本
2. 使用最值归一化（每个样本独立归一化到[-1, 1]或[0, 1]）
3. 保存为H5格式，同时保存归一化参数
4. 适配PVCNN generation模型的输入格式
"""

import os
import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
from glob import glob
import logging
from typing import Tuple, List

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    load_csv_pointcloud, setup_logging,
    farthest_point_sampling
)

# CCP发育时期列表
STAGES = ['early', 'mid_early', 'middle', 'mid_late', 'late', 'mature']


def normalize_minmax(points: np.ndarray, 
                    center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    最值归一化点云（单个样本）
    
    将点云归一化到 [-1, 1] (center=True) 或 [0, 1] (center=False)
    
    Args:
        points: (N, 3) 点云坐标
        center: 是否居中到[-1, 1]，否则归一化到[0, 1]
        
    Returns:
        normalized: 归一化后的点云
        min_vals: 最小值
        max_vals: 最大值
    """
    min_vals = points.min(axis=0)  # (3,)
    max_vals = points.max(axis=0)  # (3,)
    
    # 避免除零
    range_vals = max_vals - min_vals
    range_vals = np.maximum(range_vals, 1e-8)
    
    if center:
        # 归一化到 [-1, 1]
        normalized = 2 * (points - min_vals) / range_vals - 1
    else:
        # 归一化到 [0, 1]
        normalized = (points - min_vals) / range_vals
    
    return normalized.astype(np.float32), min_vals.astype(np.float32), max_vals.astype(np.float32)


def denormalize_minmax(normalized: np.ndarray,
                      min_vals: np.ndarray,
                      max_vals: np.ndarray,
                      center: bool = True) -> np.ndarray:
    """
    反归一化点云
    
    Args:
        normalized: 归一化的点云
        min_vals: 最小值
        max_vals: 最大值
        center: 是否从[-1, 1]反归一化
        
    Returns:
        denormalized: 原始尺度的点云
    """
    range_vals = max_vals - min_vals
    
    if center:
        # 从 [-1, 1] 反归一化
        denormalized = (normalized + 1) / 2 * range_vals + min_vals
    else:
        # 从 [0, 1] 反归一化
        denormalized = normalized * range_vals + min_vals
    
    return denormalized


def load_all_samples(input_dir: str,
                    coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                    target_points: int = None,
                    logger: logging.Logger = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    加载目录下所有CCP样本
    
    Args:
        input_dir: 输入目录
        coord_cols: 坐标列名
        target_points: 目标点数（如果指定，会使用FPS采样到相同点数）
        logger: 日志记录器
        
    Returns:
        all_points: 点云列表
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
        
        # 如果指定了目标点数，进行采样
        if target_points is not None:
            if len(points) > target_points:
                points = farthest_point_sampling(points, target_points)
            elif len(points) < target_points:
                # 点数不足，跳过
                logger.warning(f"跳过 {os.path.basename(csv_path)}: 点数不足 ({len(points)} < {target_points})")
                continue
        
        all_points.append(points)
        filenames.append(os.path.basename(csv_path))
    
    logger.info(f"加载完成: {len(all_points)} 个样本")
    
    return all_points, filenames


def save_to_h5_minmax(output_path: str,
                     points_list: List[np.ndarray],
                     filenames: List[str] = None,
                     center: bool = True,
                     logger: logging.Logger = None):
    """
    使用最值归一化保存点云到H5文件（每个样本独立归一化）
    
    Args:
        output_path: 输出文件路径
        points_list: 点云列表
        filenames: 原始文件名列表
        center: 是否居中归一化到[-1, 1]
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    B = len(points_list)
    
    # 检查点数是否一致
    num_points = [p.shape[0] for p in points_list]
    if len(set(num_points)) > 1:
        logger.warning(f"点云点数不一致: {set(num_points)}")
        min_points = min(num_points)
        logger.warning(f"截断到最小点数: {min_points}")
        points_list = [p[:min_points] for p in points_list]
        num_points = [min_points] * B
    
    N = num_points[0]
    
    # 归一化每个样本
    normalized_list = []
    min_vals_list = []
    max_vals_list = []
    
    for i, points in enumerate(points_list):
        normalized, min_vals, max_vals = normalize_minmax(points, center=center)
        normalized_list.append(normalized)
        min_vals_list.append(min_vals)
        max_vals_list.append(max_vals)
    
    # 堆叠为数组
    all_points = np.stack(normalized_list, axis=0)  # (B, N, 3)
    all_min_vals = np.stack(min_vals_list, axis=0)  # (B, 3)
    all_max_vals = np.stack(max_vals_list, axis=0)  # (B, 3)
    
    # 保存到H5
    with h5py.File(output_path, 'w') as f:
        # 保存归一化后的点云
        f.create_dataset('points', data=all_points, dtype=np.float32)
        
        # 保存归一化参数
        f.create_dataset('min_vals', data=all_min_vals, dtype=np.float32)
        f.create_dataset('max_vals', data=all_max_vals, dtype=np.float32)
        
        # 保存元数据
        f.attrs['num_samples'] = B
        f.attrs['num_points'] = N
        f.attrs['normalize_method'] = 'minmax'
        f.attrs['normalize_center'] = center
        f.attrs['normalize_per_shape'] = True  # 每个样本独立归一化
        
        # 保存文件名
        if filenames is not None:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('filenames', data=filenames, dtype=dt)
    
    # 统计信息
    logger.info(f"已保存到: {output_path}")
    logger.info(f"  样本数: {B}")
    logger.info(f"  点数: {N}")
    logger.info(f"  归一化方式: minmax (per_shape)")
    logger.info(f"  归一化范围: {'[-1, 1]' if center else '[0, 1]'}")
    logger.info(f"  归一化后 - 范围: [{all_points.min():.4f}, {all_points.max():.4f}]")
    logger.info(f"  归一化后 - 均值: {all_points.mean():.6f}")


def load_from_h5_minmax(h5_path: str,
                       logger: logging.Logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    从H5文件加载最值归一化的点云数据
    
    Args:
        h5_path: H5文件路径
        logger: 日志记录器
        
    Returns:
        points: 归一化后的点云 (B, N, 3)
        min_vals: 最小值 (B, 3)
        max_vals: 最大值 (B, 3)
        metadata: 元数据字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    with h5py.File(h5_path, 'r') as f:
        points = f['points'][:]
        min_vals = f['min_vals'][:]
        max_vals = f['max_vals'][:]
        
        metadata = {
            'num_samples': f.attrs['num_samples'],
            'num_points': f.attrs['num_points'],
            'normalize_method': f.attrs.get('normalize_method', 'minmax'),
            'normalize_center': f.attrs.get('normalize_center', True),
            'normalize_per_shape': f.attrs.get('normalize_per_shape', True),
        }
        
        if 'filenames' in f:
            metadata['filenames'] = [fn.decode('utf-8') if isinstance(fn, bytes) else fn
                                    for fn in f['filenames'][:]]
    
    logger.info(f"从 {h5_path} 加载:")
    logger.info(f"  样本数: {metadata['num_samples']}")
    logger.info(f"  点数: {metadata['num_points']}")
    
    return points, min_vals, max_vals, metadata


def csv_to_h5_minmax(input_dir: str,
                    output_path: str,
                    target_points: int = None,
                    center: bool = True,
                    coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                    logger: logging.Logger = None):
    """
    将CSV点云转换为最值归一化的H5格式
    
    Args:
        input_dir: CSV文件目录
        output_path: H5输出文件路径
        target_points: 目标点数
        center: 是否居中归一化到[-1, 1]
        coord_cols: 坐标列名
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 加载所有样本
    points_list, filenames = load_all_samples(
        input_dir, coord_cols=coord_cols,
        target_points=target_points, logger=logger
    )
    
    if len(points_list) == 0:
        logger.error("没有找到有效的点云样本!")
        return
    
    # 输出原始数据统计
    all_points_raw = np.concatenate([p.reshape(-1, 3) for p in points_list], axis=0)
    logger.info(f"归一化前 - 范围: [{all_points_raw.min():.2f}, {all_points_raw.max():.2f}]")
    
    # 保存
    save_to_h5_minmax(
        output_path, points_list,
        filenames=filenames,
        center=center,
        logger=logger
    )


def main():
    parser = argparse.ArgumentParser(description='将CCP样本CSV转换为H5格式')
    parser.add_argument('--stage', type=str, default='early',
                       choices=STAGES,
                       help='发育时期 (default: early)')
    parser.add_argument('--target-points', type=int, default=None,
                       help='目标点数，确保所有样本点数一致 (default: None, 使用原始点数)')
    parser.add_argument('--center', action='store_true', default=True,
                       help='归一化到 [-1, 1] (default: True)')
    parser.add_argument('--no-center', action='store_false', dest='center',
                       help='归一化到 [0, 1]')
    parser.add_argument('--output-dir', type=str,
                       default='/home/djx/data0/pvd_nup/ccp_stages/outputs',
                       help='数据根目录')
    
    args = parser.parse_args()
    
    # 设置路径
    input_dir = os.path.join(args.output_dir, args.stage, "csv")
    output_path = os.path.join(args.output_dir, args.stage, f"ccp_{args.stage}.h5")
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("开始将CCP样本CSV转换为H5格式...")
    logger.info("=" * 60)
    logger.info(f"发育时期: {args.stage}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"目标点数: {args.target_points if args.target_points else '使用原始点数'}")
    logger.info(f"归一化范围: {'[-1, 1]' if args.center else '[0, 1]'}")
    
    # 执行转换
    csv_to_h5_minmax(
        input_dir, output_path,
        target_points=args.target_points,
        center=args.center,
        logger=logger
    )
    
    logger.info("转换完成!")
    
    # 验证：加载并检查
    logger.info("\n验证H5文件...")
    points, min_vals, max_vals, metadata = load_from_h5_minmax(output_path, logger=logger)
    
    # 反归一化测试
    logger.info("\n反归一化测试（第一个样本）...")
    sample_idx = 0
    logger.info(f"  样本 {sample_idx} 原始范围 (min_vals): {min_vals[sample_idx]}")
    logger.info(f"  样本 {sample_idx} 原始范围 (max_vals): {max_vals[sample_idx]}")
    denormalized = denormalize_minmax(points[sample_idx], min_vals[sample_idx], max_vals[sample_idx], center=args.center)
    logger.info(f"  反归一化后 - 范围: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
    
    # 全局统计
    logger.info("\n全局统计（所有样本）...")
    logger.info(f"  所有样本 min_vals 范围: [{min_vals.min():.2f}, {min_vals.max():.2f}]")
    logger.info(f"  所有样本 max_vals 范围: [{max_vals.min():.2f}, {max_vals.max():.2f}]")


if __name__ == "__main__":
    main()
