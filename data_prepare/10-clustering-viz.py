#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10-clustering-viz.py: 可视化聚类结果

功能:
1. 读取9-clustering-csv/目录下的聚类点云
2. 按聚类标签着色可视化
3. 保存到10-clustering-viz-png/目录
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import Tuple, List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import setup_logging


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


def visualize_clustered_pointcloud(points: np.ndarray,
                                   labels: np.ndarray,
                                   output_path: str,
                                   figsize: Tuple[int, int] = (12, 12),
                                   point_size: float = 0.1,
                                   cmap: str = 'tab20',
                                   title: Optional[str] = None,
                                   dpi: int = 150,
                                   show_noise: bool = True,
                                   noise_color: str = 'lightgray',
                                   noise_alpha: float = 0.3):
    """
    可视化聚类点云（按聚类标签着色）
    
    Args:
        points: (N, 3) 点云坐标
        labels: (N,) 聚类标签 (-1表示噪声)
        output_path: 输出文件路径
        figsize: 图像大小
        point_size: 点大小
        cmap: 颜色映射
        title: 图像标题
        dpi: 图像分辨率
        show_noise: 是否显示噪声点
        noise_color: 噪声点颜色
        noise_alpha: 噪声点透明度
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 分离噪声点和聚类点
    noise_mask = labels == -1
    cluster_mask = ~noise_mask
    
    # 绘制噪声点（在底层）
    if show_noise and np.any(noise_mask):
        ax.scatter(points[noise_mask, 0], points[noise_mask, 1],
                  c=noise_color, s=point_size * 0.5, alpha=noise_alpha,
                  label=f'Noise ({np.sum(noise_mask)})')
    
    # 绘制聚类点
    if np.any(cluster_mask):
        cluster_points = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        
        # 使用离散颜色映射
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        # 创建颜色映射
        colormap = plt.colormaps.get_cmap(cmap)
        colors = [colormap(i % 20 / 20) for i in range(n_clusters)]
        
        # 为每个点分配颜色
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        point_colors = [colors[label_to_idx[l]] for l in cluster_labels]
        
        scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           c=point_colors, s=point_size, alpha=0.8)
    
    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    ax.set_aspect('equal')
    
    # 添加统计信息到标题
    num_clusters = len(np.unique(labels[labels != -1]))
    num_noise = np.sum(labels == -1)
    if title:
        full_title = f"{title}\n{num_clusters} clusters, {num_noise} noise points"
    else:
        full_title = f"{num_clusters} clusters, {num_noise} noise points"
    ax.set_title(full_title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def visualize_single_file(csv_path: str, output_dir: str) -> str:
    """可视化单个聚类CSV文件"""
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}.png")
    
    # 加载聚类点云
    points, labels = load_clustered_pointcloud(csv_path)
    
    visualize_clustered_pointcloud(
        points, labels, output_path,
        figsize=(12, 12),
        point_size=0.1,
        cmap='tab20',
        title=name_without_ext,
        dpi=150
    )
    
    return output_path


def visualize_csv_files(input_dir: str, 
                       output_dir: str,
                       max_workers: int = 4,
                       logger: logging.Logger = None):
    """
    批量可视化聚类CSV点云文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        max_workers: 并行工作进程数
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(visualize_single_file, csv_path, output_dir): csv_path
            for csv_path in csv_files
        }
        
        for future in as_completed(futures):
            csv_path = futures[future]
            filename = os.path.basename(csv_path)
            try:
                output_path = future.result()
                logger.info(f"已可视化: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                logger.error(f"可视化 {filename} 时出错: {str(e)}")


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "9-clustering-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "10-clustering-viz-png")
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始可视化聚类点云...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # 执行可视化
    visualize_csv_files(INPUT_DIR, OUTPUT_DIR, max_workers=4, logger=logger)
    
    logger.info("可视化完成!")


if __name__ == "__main__":
    main()
