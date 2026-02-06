#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1-npc-samples.py: 处理NPC点云样本

功能:
1. 读取指定子文件夹的NPC点云CSV文件
2. 过滤点数不足的样本
3. 对点数过多的样本使用最远点采样
4. 归一化到坐标原点（平移）
5. 输出到processed/{subfolder}/csv/目录
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import logging
from typing import Tuple, List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    save_csv_pointcloud, setup_logging,
    move_to_origin, farthest_point_sampling
)

# NPC子文件夹列表
SUBFOLDERS = [
    'rotated_density_0_9',
    'rotated_density_0_7',
    'rotated_density_0_5',
    'fixed_density_0_9',
    'fixed_density_0_7',
    'fixed_density_0_5'
]

# NPC数据的坐标列名 (与CCP相同)
NPC_COORD_COLS = ['x', 'y', 'z']


def load_npc_pointcloud(csv_path: str,
                       coord_cols: List[str] = NPC_COORD_COLS) -> np.ndarray:
    """
    加载NPC点云
    
    Args:
        csv_path: CSV文件路径
        coord_cols: 坐标列名
        
    Returns:
        points: (N, 3) 点云坐标
    """
    df = pd.read_csv(csv_path)
    points = df[coord_cols].values.astype(np.float64)
    return points


def process_npc_sample(points: np.ndarray,
                       target_points: int,
                       min_points: int,
                       use_fps: bool = True) -> Optional[np.ndarray]:
    """
    处理单个NPC样本：过滤、采样、归一化
    
    Args:
        points: (N, 3) NPC点云
        target_points: 目标点数
        min_points: 最小点数阈值（小于此值直接丢弃）
        use_fps: 是否使用最远点采样（否则随机采样）
        
    Returns:
        processed_points: 处理后的点云 或 None（如果点数不足）
    """
    num_points = len(points)
    
    # 点数不足，丢弃
    if num_points < min_points:
        return None
    
    # 点数足够但需要采样
    if num_points > target_points:
        if use_fps:
            sampled = farthest_point_sampling(points, target_points)
        else:
            indices = np.random.choice(num_points, target_points, replace=False)
            sampled = points[indices]
    elif num_points == target_points:
        sampled = points
    else:
        # 点数在 [min_points, target_points) 范围
        # 直接使用原始点云
        sampled = points
    
    # 归一化到原点
    processed, _ = move_to_origin(sampled)
    
    return processed


def process_subfolder(input_dir: str,
                      output_dir: str,
                      subfolder: str,
                      target_points: int = 2048,
                      min_points: int = 100,
                      use_fps: bool = True,
                      seed: int = 42,
                      logger: logging.Logger = None) -> dict:
    """
    处理指定子文件夹的NPC样本
    
    Args:
        input_dir: 数据根目录
        output_dir: 输出根目录
        subfolder: 子文件夹名称
        target_points: 每个样本的目标点数
        min_points: 最小点数阈值
        use_fps: 是否使用最远点采样
        seed: 随机种子
        logger: 日志记录器
        
    Returns:
        stats: 处理统计信息
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    np.random.seed(seed)
    
    # 设置路径
    subfolder_input_dir = os.path.join(input_dir, subfolder)
    subfolder_output_dir = os.path.join(output_dir, subfolder, "csv")
    
    if not os.path.exists(subfolder_input_dir):
        logger.error(f"输入目录不存在: {subfolder_input_dir}")
        return {}
    
    Path(subfolder_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = sorted(glob(os.path.join(subfolder_input_dir, "*.csv")))
    logger.info(f"[{subfolder}] 找到 {len(csv_files)} 个NPC点云文件")
    
    stats = {
        'subfolder': subfolder,
        'total_samples': len(csv_files),
        'samples_too_small': 0,
        'samples_processed': 0,
        'point_counts': [],
        'original_point_counts': []
    }
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        
        try:
            points = load_npc_pointcloud(csv_path)
            original_count = len(points)
            stats['original_point_counts'].append(original_count)
            
            if original_count < min_points:
                stats['samples_too_small'] += 1
                logger.debug(f"  跳过 {filename}: 点数不足 ({original_count} < {min_points})")
                continue
            
            # 处理样本
            processed = process_npc_sample(
                points, target_points, min_points, use_fps
            )
            
            if processed is None:
                stats['samples_too_small'] += 1
                continue
            
            # 保存样本
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(subfolder_output_dir, f"{name_without_ext}.csv")
            
            # 使用标准列名保存
            save_csv_pointcloud(output_path, processed, header=['x [nm]', 'y [nm]', 'z [nm]'])
            
            stats['point_counts'].append(len(processed))
            stats['samples_processed'] += 1
            
            if stats['samples_processed'] % 100 == 0:
                logger.info(f"  [{subfolder}] 已处理: {stats['samples_processed']}/{len(csv_files)}")
                
        except Exception as e:
            logger.error(f"  处理 {filename} 时出错: {str(e)}")
            continue
    
    logger.info(f"\n[{subfolder}] 处理完成!")
    logger.info(f"  总样本数: {stats['total_samples']}")
    logger.info(f"  点数不足被过滤: {stats['samples_too_small']}")
    logger.info(f"  成功处理数: {stats['samples_processed']}")
    
    if stats['point_counts']:
        logger.info(f"  处理后点数范围: [{min(stats['point_counts'])}, {max(stats['point_counts'])}]")
        logger.info(f"  处理后点数均值: {np.mean(stats['point_counts']):.1f}")
    
    if stats['original_point_counts']:
        logger.info(f"  原始点数范围: [{min(stats['original_point_counts'])}, {max(stats['original_point_counts'])}]")
        logger.info(f"  原始点数均值: {np.mean(stats['original_point_counts']):.1f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='处理NPC点云样本')
    parser.add_argument('--subfolder', type=str, default='rotated_density_0_9',
                       choices=SUBFOLDERS,
                       help='子文件夹 (default: rotated_density_0_9)')
    parser.add_argument('--target-points', type=int, default=2048,
                       help='每个样本的目标点数 (default: 2048)')
    parser.add_argument('--min-points', type=int, default=100,
                       help='最小点数阈值 (default: 100)')
    parser.add_argument('--use-fps', action='store_true', default=True,
                       help='使用最远点采样 (default: True)')
    parser.add_argument('--no-fps', action='store_false', dest='use_fps',
                       help='使用随机采样')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--input-dir', type=str,
                       default='/home/djx/repos/biolocsim/outputs/npc_batch',
                       help='输入数据根目录')
    parser.add_argument('--output-dir', type=str,
                       default='/home/djx/data0/pvd_nup/npc_batch/processed',
                       help='输出数据根目录')
    
    args = parser.parse_args()
    
    # ========== 执行处理 ==========
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("开始处理NPC点云样本...")
    logger.info("=" * 60)
    logger.info(f"子文件夹: {args.subfolder}")
    logger.info(f"输入目录: {args.input_dir}/{args.subfolder}")
    logger.info(f"输出目录: {args.output_dir}/{args.subfolder}/csv")
    logger.info(f"目标点数: {args.target_points}")
    logger.info(f"最小点数阈值: {args.min_points}")
    logger.info(f"采样方式: {'FPS' if args.use_fps else '随机'}")
    
    stats = process_subfolder(
        args.input_dir, args.output_dir,
        subfolder=args.subfolder,
        target_points=args.target_points,
        min_points=args.min_points,
        use_fps=args.use_fps,
        seed=args.seed,
        logger=logger
    )
    
    # 保存统计信息
    stats_dir = os.path.join(args.output_dir, args.subfolder)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    stats_path = os.path.join(stats_dir, "extraction_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"统计信息已保存到: {stats_path}")
    
    logger.info("\n处理完成!")


if __name__ == "__main__":
    main()
