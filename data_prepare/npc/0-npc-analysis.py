#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0-npc-analysis.py: 分析NPC点云数据的点数分布

功能:
1. 统计指定子文件夹或所有子文件夹的点云点数分布
2. 计算平均点数、标准差、最小/最大值、分位数等
3. 生成点数分布直方图
4. 帮助决定目标点数的设置

NPC数据目录结构:
outputs/npc_batch/
├── rotated_density_0_9/      # 随机旋转 + 标记密度0.9
├── rotated_density_0_7/      # 随机旋转 + 标记密度0.7
├── rotated_density_0_5/      # 随机旋转 + 标记密度0.5
├── fixed_density_0_9/        # 固定方向 + 标记密度0.9
├── fixed_density_0_7/        # 固定方向 + 标记密度0.7
└── fixed_density_0_5/        # 固定方向 + 标记密度0.5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import logging
from typing import Dict, List
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# NPC子文件夹列表
SUBFOLDERS = [
    'rotated_density_0_9',
    'rotated_density_0_7',
    'rotated_density_0_5',
    'fixed_density_0_9',
    'fixed_density_0_7',
    'fixed_density_0_5'
]

# NPC数据的坐标列名
NPC_COORD_COLS = ['x', 'y', 'z']


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def count_points_in_file(csv_path: str) -> int:
    """统计单个CSV文件的点数"""
    df = pd.read_csv(csv_path)
    return len(df)


def analyze_subfolder(input_dir: str, subfolder: str, logger: logging.Logger) -> Dict:
    """
    分析单个子文件夹的点数分布
    
    Args:
        input_dir: 数据根目录
        subfolder: 子文件夹名称
        logger: 日志记录器
        
    Returns:
        统计信息字典
    """
    subfolder_dir = os.path.join(input_dir, subfolder)
    
    if not os.path.exists(subfolder_dir):
        logger.warning(f"目录不存在: {subfolder_dir}")
        return {}
    
    csv_files = sorted(glob(os.path.join(subfolder_dir, "*.csv")))
    
    if len(csv_files) == 0:
        logger.warning(f"[{subfolder}] 没有找到CSV文件")
        return {}
    
    # 统计每个文件的点数
    point_counts = []
    for csv_path in csv_files:
        try:
            count = count_points_in_file(csv_path)
            point_counts.append(count)
        except Exception as e:
            logger.error(f"读取 {csv_path} 出错: {e}")
            continue
    
    if len(point_counts) == 0:
        return {}
    
    point_counts = np.array(point_counts)
    
    # 计算统计量
    stats = {
        'subfolder': subfolder,
        'num_samples': len(point_counts),
        'mean': float(np.mean(point_counts)),
        'std': float(np.std(point_counts)),
        'min': int(np.min(point_counts)),
        'max': int(np.max(point_counts)),
        'median': float(np.median(point_counts)),
        'q25': float(np.percentile(point_counts, 25)),
        'q75': float(np.percentile(point_counts, 75)),
        'q10': float(np.percentile(point_counts, 10)),
        'q90': float(np.percentile(point_counts, 90)),
        'point_counts': point_counts.tolist()
    }
    
    return stats


def print_subfolder_stats(stats: Dict, logger: logging.Logger):
    """打印单个子文件夹的统计信息"""
    if not stats:
        return
    
    subfolder = stats['subfolder']
    logger.info(f"\n{'='*50}")
    logger.info(f"子文件夹: {subfolder}")
    logger.info(f"{'='*50}")
    logger.info(f"  样本数量: {stats['num_samples']}")
    logger.info(f"  点数范围: [{stats['min']}, {stats['max']}]")
    logger.info(f"  平均点数: {stats['mean']:.1f}")
    logger.info(f"  标准差:   {stats['std']:.1f}")
    logger.info(f"  中位数:   {stats['median']:.1f}")
    logger.info(f"  10%分位:  {stats['q10']:.1f}")
    logger.info(f"  25%分位:  {stats['q25']:.1f}")
    logger.info(f"  75%分位:  {stats['q75']:.1f}")
    logger.info(f"  90%分位:  {stats['q90']:.1f}")
    
    # 建议的目标点数
    logger.info(f"\n  建议目标点数:")
    logger.info(f"    保守 (10%分位): {int(stats['q10'])}")
    logger.info(f"    中等 (25%分位): {int(stats['q25'])}")
    logger.info(f"    激进 (中位数):  {int(stats['median'])}")


def plot_histogram(all_stats: List[Dict], output_path: str, logger: logging.Logger):
    """
    绘制所有子文件夹的点数分布直方图
    
    Args:
        all_stats: 所有子文件夹的统计信息列表
        output_path: 输出图片路径
        logger: 日志记录器
    """
    num_subfolders = len(all_stats)
    if num_subfolders == 0:
        logger.warning("没有数据可绘制")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 6))
    
    for i, stats in enumerate(all_stats):
        if i >= 6:
            break
        
        ax = axes[i]
        point_counts = stats['point_counts']
        
        ax.hist(point_counts, bins=30, color=colors[i], edgecolor='white', alpha=0.8)
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
        ax.axvline(stats['median'], color='orange', linestyle='-.', linewidth=2, label=f"Median: {stats['median']:.0f}")
        ax.axvline(stats['q25'], color='green', linestyle=':', linewidth=2, label=f"Q25: {stats['q25']:.0f}")
        
        ax.set_xlabel('Point Count', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"{stats['subfolder']} (n={stats['num_samples']})", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(all_stats), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n直方图已保存到: {output_path}")


def plot_combined_histogram(all_stats: List[Dict], output_path: str, logger: logging.Logger):
    """
    绘制所有子文件夹合并的点数分布直方图
    
    Args:
        all_stats: 所有子文件夹的统计信息列表
        output_path: 输出图片路径
        logger: 日志记录器
    """
    if len(all_stats) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 合并所有点数
    all_counts = []
    labels = []
    for stats in all_stats:
        all_counts.extend(stats['point_counts'])
        labels.extend([stats['subfolder']] * len(stats['point_counts']))
    
    all_counts = np.array(all_counts)
    
    # 绘制直方图
    ax.hist(all_counts, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    
    # 添加统计线
    mean_val = np.mean(all_counts)
    median_val = np.median(all_counts)
    q25_val = np.percentile(all_counts, 25)
    q10_val = np.percentile(all_counts, 10)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_val:.0f}")
    ax.axvline(median_val, color='orange', linestyle='-.', linewidth=2, label=f"Median: {median_val:.0f}")
    ax.axvline(q25_val, color='green', linestyle=':', linewidth=2, label=f"Q25: {q25_val:.0f}")
    ax.axvline(q10_val, color='purple', linestyle=':', linewidth=2, label=f"Q10: {q10_val:.0f}")
    
    ax.set_xlabel('Point Count', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'All Subfolders Combined (n={len(all_counts)})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"合并直方图已保存到: {output_path}")


def print_summary_table(all_stats: List[Dict], logger: logging.Logger):
    """打印汇总表格"""
    if len(all_stats) == 0:
        return
    
    logger.info(f"\n{'='*100}")
    logger.info("汇总统计")
    logger.info(f"{'='*100}")
    
    # 表头
    header = f"{'子文件夹':<24} {'样本数':>8} {'平均':>8} {'标准差':>8} {'最小':>8} {'最大':>8} {'Q10':>8} {'Q25':>8} {'中位数':>8}"
    logger.info(header)
    logger.info("-" * 100)
    
    # 每个子文件夹
    for stats in all_stats:
        row = f"{stats['subfolder']:<24} {stats['num_samples']:>8} {stats['mean']:>8.1f} {stats['std']:>8.1f} {stats['min']:>8} {stats['max']:>8} {stats['q10']:>8.1f} {stats['q25']:>8.1f} {stats['median']:>8.1f}"
        logger.info(row)
    
    # 汇总
    if len(all_stats) > 1:
        all_counts = []
        for stats in all_stats:
            all_counts.extend(stats['point_counts'])
        all_counts = np.array(all_counts)
        
        logger.info("-" * 100)
        total_row = f"{'全部':<24} {len(all_counts):>8} {np.mean(all_counts):>8.1f} {np.std(all_counts):>8.1f} {np.min(all_counts):>8} {np.max(all_counts):>8} {np.percentile(all_counts, 10):>8.1f} {np.percentile(all_counts, 25):>8.1f} {np.median(all_counts):>8.1f}"
        logger.info(total_row)
    
    logger.info(f"\n{'='*100}")
    logger.info("建议目标点数选择:")
    logger.info(f"{'='*100}")
    
    if len(all_stats) > 1:
        all_counts = []
        for stats in all_stats:
            all_counts.extend(stats['point_counts'])
        all_counts = np.array(all_counts)
        
        logger.info(f"  全局 Q10 (保守):  {int(np.percentile(all_counts, 10))}")
        logger.info(f"  全局 Q25 (中等):  {int(np.percentile(all_counts, 25))}")
        logger.info(f"  全局 中位数:      {int(np.median(all_counts))}")
        logger.info(f"  全局 最小值:      {int(np.min(all_counts))}")


def main():
    parser = argparse.ArgumentParser(description='分析NPC点云数据的点数分布')
    parser.add_argument('--subfolder', type=str, default='all',
                       choices=SUBFOLDERS + ['all'],
                       help='子文件夹，设为"all"分析所有子文件夹 (default: all)')
    parser.add_argument('--input-dir', type=str,
                       default='/home/djx/repos/biolocsim/outputs/npc_batch',
                       help='输入数据根目录')
    parser.add_argument('--output-dir', type=str,
                       default='/home/djx/repos/biolocsim/outputs/npc_batch/analysis',
                       help='输出目录（用于保存直方图）')
    parser.add_argument('--no-plot', action='store_true', default=False,
                       help='不生成直方图')
    
    args = parser.parse_args()
    
    # 确定要分析的子文件夹
    if args.subfolder == 'all':
        subfolders_to_analyze = SUBFOLDERS
    else:
        subfolders_to_analyze = [args.subfolder]
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("NPC点云数据点数分析")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"分析子文件夹: {subfolders_to_analyze}")
    
    # 分析每个子文件夹
    all_stats = []
    for subfolder in subfolders_to_analyze:
        stats = analyze_subfolder(args.input_dir, subfolder, logger)
        if stats:
            all_stats.append(stats)
            print_subfolder_stats(stats, logger)
    
    # 打印汇总表格
    if len(all_stats) > 0:
        print_summary_table(all_stats, logger)
    
    # 绘制直方图
    if not args.no_plot and len(all_stats) > 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 分子文件夹直方图
        hist_path = os.path.join(args.output_dir, "npc_point_count_distribution.png")
        plot_histogram(all_stats, hist_path, logger)
        
        # 合并直方图
        if len(all_stats) > 1:
            combined_path = os.path.join(args.output_dir, "npc_point_count_distribution_combined.png")
            plot_combined_histogram(all_stats, combined_path, logger)
    
    logger.info("\n分析完成!")


if __name__ == "__main__":
    main()
