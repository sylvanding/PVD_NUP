#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11-clustered-samples.py: 从聚类结果中提取单个核孔样本

功能:
1. 读取9-clustering-csv/目录下的聚类点云
2. 将每个聚类（核孔）分离为单独的CSV文件
3. 过滤点数不足的核孔
4. 对点数过多的核孔使用最远点采样
5. 归一化到坐标原点（平移）
6. 可设置目标样本数量，达到后停止
7. 输出到11-clustered-samples-csv/目录
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import logging
from typing import Tuple, List, Optional
from collections import defaultdict
import random

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    save_csv_pointcloud, setup_logging,
    move_to_origin, farthest_point_sampling
)


def load_clustered_pointcloud(csv_path: str,
                             coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                             label_col: str = 'cluster_id') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载带聚类标签的点云
    
    Args:
        csv_path: CSV文件路径
        coord_cols: 坐标列名
        label_col: 聚类标签列名
        
    Returns:
        points: (N, 3) 点云坐标
        labels: (N,) 聚类标签
    """
    df = pd.read_csv(csv_path)
    points = df[coord_cols].values.astype(np.float64)
    labels = df[label_col].values.astype(np.int32)
    return points, labels


def extract_clusters(points: np.ndarray, 
                    labels: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """
    提取所有聚类
    
    Args:
        points: (N, 3) 点云坐标
        labels: (N,) 聚类标签
        
    Returns:
        clusters: [(cluster_id, cluster_points), ...] 列表
    """
    clusters = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue
        mask = labels == label
        cluster_points = points[mask]
        clusters.append((int(label), cluster_points))
    
    return clusters


def compute_aspect_ratio(points: np.ndarray) -> float:
    """
    计算点云在XY平面上的长宽比
    
    Args:
        points: (N, 3) 点云坐标
        
    Returns:
        aspect_ratio: 长宽比（总是 >= 1.0，较大边/较小边）
    """
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    
    # 避免除零
    if min(x_range, y_range) < 1e-6:
        return float('inf')
    
    # 返回较大值/较小值，使得比值总是 >= 1.0
    return max(x_range, y_range) / min(x_range, y_range)


def process_cluster(cluster_points: np.ndarray,
                   target_points: int,
                   min_points: int,
                   use_fps: bool = True,
                   max_aspect_ratio: float = 2.0,
                   max_retries: int = 10) -> Optional[np.ndarray]:
    """
    处理单个聚类：过滤、采样、归一化
    
    Args:
        cluster_points: (N, 3) 聚类点云
        target_points: 目标点数
        min_points: 最小点数阈值（小于此值直接丢弃）
        use_fps: 是否使用最远点采样（否则随机采样）
        max_aspect_ratio: 最大允许长宽比（1.0表示完美正方形，默认2.0）
        max_retries: 随机采样时的最大重试次数
        
    Returns:
        processed_points: (target_points, 3) 或 None（如果点数不足或长宽比不合格）
    """
    num_points = len(cluster_points)
    
    # 点数不足，丢弃
    if num_points < min_points:
        return None
    
    # 检查原始点云长宽比
    aspect_ratio = compute_aspect_ratio(cluster_points)
    if aspect_ratio > max_aspect_ratio:
        return None
    
    # 点数足够但需要采样
    if num_points > target_points:
        if use_fps:
            sampled = farthest_point_sampling(cluster_points, target_points)
            # FPS采样后也需要检查长宽比
            if compute_aspect_ratio(sampled) > max_aspect_ratio:
                return None
        else:
            # 随机采样：可能需要多次尝试才能得到满足长宽比要求的结果
            sampled = None
            for _ in range(max_retries):
                indices = np.random.choice(num_points, target_points, replace=False)
                candidate = cluster_points[indices]
                if compute_aspect_ratio(candidate) <= max_aspect_ratio:
                    sampled = candidate
                    break
            if sampled is None:
                # 所有重试都失败了，丢弃这个聚类
                return None
    elif num_points == target_points:
        sampled = cluster_points
    else:
        # 点数在 [min_points, target_points) 范围
        # 直接使用原始点云
        sampled = cluster_points
    
    # 归一化到原点
    processed, _ = move_to_origin(sampled)
    
    return processed


def extract_samples_from_files(input_dir: str,
                              output_dir: str,
                              num_samples: int = 1024,
                              target_points: int = 2048,
                              min_points: int = 100,
                              max_aspect_ratio: float = 2.0,
                              use_fps: bool = True,
                              shuffle: bool = True,
                              seed: int = 42,
                              coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                              logger: logging.Logger = None) -> dict:
    """
    从所有聚类文件中提取样本
    
    Args:
        input_dir: 聚类CSV文件目录
        output_dir: 输出目录
        num_samples: 目标样本数量
        target_points: 每个样本的目标点数
        min_points: 最小点数阈值
        max_aspect_ratio: 最大允许长宽比（1.0=完美正方形，2.0=长是宽的2倍）
        use_fps: 是否使用最远点采样
        shuffle: 是否随机打乱样本顺序
        seed: 随机种子
        coord_cols: 坐标列名
        logger: 日志记录器
        
    Returns:
        stats: 提取统计信息
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    np.random.seed(seed)
    random.seed(seed)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    logger.info(f"找到 {len(csv_files)} 个聚类CSV文件")
    
    # 收集所有有效的聚类
    all_clusters = []  # [(source_file, cluster_id, points, aspect_ratio), ...]
    
    stats = {
        'total_clusters_found': 0,
        'clusters_too_small': 0,
        'clusters_bad_aspect_ratio': 0,
        'clusters_accepted': 0,
        'samples_saved': 0,
        'source_files': defaultdict(int),
        'point_counts': [],
        'aspect_ratios': []
    }
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        logger.info(f"处理: {filename}")
        
        try:
            points, labels = load_clustered_pointcloud(csv_path, coord_cols=coord_cols)
            clusters = extract_clusters(points, labels)
            
            logger.info(f"  找到 {len(clusters)} 个聚类")
            stats['total_clusters_found'] += len(clusters)
            
            for cluster_id, cluster_points in clusters:
                if len(cluster_points) < min_points:
                    stats['clusters_too_small'] += 1
                    continue
                
                # 检查长宽比
                aspect_ratio = compute_aspect_ratio(cluster_points)
                if aspect_ratio > max_aspect_ratio:
                    stats['clusters_bad_aspect_ratio'] += 1
                    continue
                
                all_clusters.append((filename, cluster_id, cluster_points, aspect_ratio))
                stats['clusters_accepted'] += 1
                
        except Exception as e:
            logger.error(f"处理 {filename} 时出错: {str(e)}")
            continue
    
    logger.info(f"\n总共找到 {stats['total_clusters_found']} 个聚类")
    logger.info(f"点数不足被过滤: {stats['clusters_too_small']}")
    logger.info(f"长宽比不合格被过滤: {stats['clusters_bad_aspect_ratio']}")
    logger.info(f"可用聚类: {stats['clusters_accepted']}")
    
    # 随机打乱
    if shuffle:
        random.shuffle(all_clusters)
    
    # 提取样本
    sample_idx = 0
    clusters_failed_after_sampling = 0
    for source_file, cluster_id, cluster_points, aspect_ratio in all_clusters:
        if sample_idx >= num_samples:
            break
        
        # 处理聚类（采样后会重新检查长宽比）
        processed = process_cluster(
            cluster_points, target_points, min_points, use_fps,
            max_aspect_ratio=max_aspect_ratio  # 采样后重新检查长宽比
        )
        
        if processed is None:
            clusters_failed_after_sampling += 1
            continue
        
        # 计算采样后的实际长宽比
        final_aspect_ratio = compute_aspect_ratio(processed)
        
        # 保存样本
        # 文件名格式: {idx:05d}_{source}_{cluster_id}.csv
        source_name = os.path.splitext(source_file)[0][:30]  # 截断太长的文件名
        output_filename = f"{sample_idx:05d}_{source_name}_c{cluster_id}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        save_csv_pointcloud(output_path, processed, header=coord_cols)
        
        stats['source_files'][source_file] += 1
        stats['point_counts'].append(len(processed))
        stats['aspect_ratios'].append(final_aspect_ratio)  # 记录采样后的实际长宽比
        sample_idx += 1
        
        if (sample_idx) % 100 == 0:
            logger.info(f"  已保存: {sample_idx}/{num_samples}")
    
    stats['samples_saved'] = sample_idx
    stats['clusters_failed_after_sampling'] = clusters_failed_after_sampling
    
    # 转换defaultdict为普通dict
    stats['source_files'] = dict(stats['source_files'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"提取完成!")
    logger.info(f"目标样本数: {num_samples}")
    logger.info(f"实际保存数: {stats['samples_saved']}")
    logger.info(f"采样后长宽比不合格被过滤: {clusters_failed_after_sampling}")
    logger.info(f"目标点数: {target_points}")
    logger.info(f"最小点数阈值: {min_points}")
    logger.info(f"最大长宽比阈值: {max_aspect_ratio}")
    
    if stats['point_counts']:
        logger.info(f"实际点数范围: [{min(stats['point_counts'])}, {max(stats['point_counts'])}]")
        logger.info(f"实际点数均值: {np.mean(stats['point_counts']):.1f}")
    
    if stats['aspect_ratios']:
        logger.info(f"长宽比范围: [{min(stats['aspect_ratios']):.2f}, {max(stats['aspect_ratios']):.2f}]")
        logger.info(f"长宽比均值: {np.mean(stats['aspect_ratios']):.2f}")
    
    return stats


def main():
    # ========== 配置参数 ==========
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "9-clustering-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "11-clustered-samples-csv")
    STATS_PATH = os.path.join(OUTPUT_DIR, "extraction_stats.json")
    
    # 采样参数
    NUM_SAMPLES = 4096      # 目标样本数量
    TARGET_POINTS = 40      # 每个样本的目标点数
    MIN_POINTS = 40         # 最小点数阈值（小于此值的核孔直接丢弃）
    MAX_ASPECT_RATIO = 1.1  # 最大长宽比（1.0=完美正方形，1.5=允许长是宽的1.5倍）
    USE_FPS = False          # 使用最远点采样（否则随机采样）
    SHUFFLE = True          # 随机打乱样本顺序
    SEED = 42               # 随机种子
    
    # ========== 执行提取 ==========
    logger = setup_logging()
    
    logger.info("开始从聚类结果中提取核孔样本...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"目标样本数: {NUM_SAMPLES}")
    logger.info(f"目标点数: {TARGET_POINTS}")
    logger.info(f"最小点数阈值: {MIN_POINTS}")
    logger.info(f"最大长宽比: {MAX_ASPECT_RATIO}")
    
    stats = extract_samples_from_files(
        INPUT_DIR, OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        target_points=TARGET_POINTS,
        min_points=MIN_POINTS,
        max_aspect_ratio=MAX_ASPECT_RATIO,
        use_fps=USE_FPS,
        shuffle=SHUFFLE,
        seed=SEED,
        logger=logger
    )
    
    # 保存统计信息
    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"统计信息已保存到: {STATS_PATH}")
    
    logger.info("\n提取完成!")


if __name__ == "__main__":
    main()
