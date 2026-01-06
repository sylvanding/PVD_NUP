#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1-clean.py: 清洗原始CSV点云文件

功能:
1. 读取0-origin-csv/目录下所有csv文件
2. 检测表头中的坐标列(xnm, ynm, znm或类似变体)
3. 将点云左下角移动到坐标原点
4. 输出每个csv的统计信息(坐标范围、均值、点数)
5. 重命名表头为标准格式并保存到1-clean-csv/目录
"""

import os
import sys
import logging
from pathlib import Path
from glob import glob

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    load_csv_pointcloud, save_csv_pointcloud, 
    move_to_origin, get_pointcloud_stats, setup_logging
)


def clean_csv_files(input_dir: str, output_dir: str, logger: logging.Logger = None):
    """
    清洗CSV点云文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    # 统计信息汇总
    all_stats = []
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"处理文件: {filename}")
        
        try:
            # 加载点云
            points, header = load_csv_pointcloud(csv_path)
            logger.info(f"原始表头: {header}")
            
            # 获取原始统计信息
            stats_before = get_pointcloud_stats(points)
            
            # 移动到原点
            points_shifted, min_coords = move_to_origin(points)
            
            # 获取移动后的统计信息
            stats_after = get_pointcloud_stats(points_shifted)
            
            # 输出统计信息
            logger.info(f"点云点数: {stats_after['num_points']}")
            logger.info(f"原始坐标范围:")
            logger.info(f"  X: [{stats_before['x_min']:.2f}, {stats_before['x_max']:.2f}], 范围: {stats_before['x_range']:.2f}")
            logger.info(f"  Y: [{stats_before['y_min']:.2f}, {stats_before['y_max']:.2f}], 范围: {stats_before['y_range']:.2f}")
            logger.info(f"  Z: [{stats_before['z_min']:.2f}, {stats_before['z_max']:.2f}], 范围: {stats_before['z_range']:.2f}")
            logger.info(f"移动后坐标范围:")
            logger.info(f"  X: [{stats_after['x_min']:.2f}, {stats_after['x_max']:.2f}], 范围: {stats_after['x_range']:.2f}")
            logger.info(f"  Y: [{stats_after['y_min']:.2f}, {stats_after['y_max']:.2f}], 范围: {stats_after['y_range']:.2f}")
            logger.info(f"  Z: [{stats_after['z_min']:.2f}, {stats_after['z_max']:.2f}], 范围: {stats_after['z_range']:.2f}")
            logger.info(f"均值 (移动后):")
            logger.info(f"  X均值: {stats_after['x_mean']:.2f}")
            logger.info(f"  Y均值: {stats_after['y_mean']:.2f}")
            logger.info(f"  Z均值: {stats_after['z_mean']:.2f}")
            
            # 保存清洗后的文件
            output_path = os.path.join(output_dir, filename)
            save_csv_pointcloud(output_path, points_shifted, 
                              header=['x [nm]', 'y [nm]', 'z [nm]'],
                              precision=2)
            logger.info(f"已保存到: {output_path}")
            
            # 记录统计信息
            all_stats.append({
                'filename': filename,
                **stats_after
            })
            
        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    # 输出汇总信息
    logger.info(f"\n{'='*60}")
    logger.info("汇总统计信息:")
    logger.info(f"{'文件名':<60} {'点数':>10} {'X范围':>12} {'Y范围':>12} {'Z范围':>12}")
    logger.info("-" * 110)
    for stats in all_stats:
        logger.info(f"{stats['filename']:<60} {stats['num_points']:>10} {stats['x_range']:>12.2f} {stats['y_range']:>12.2f} {stats['z_range']:>12.2f}")
    
    return all_stats


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "0-origin-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "1-clean-csv")
    
    # 设置日志
    log_file = os.path.join(DATA_ROOT, "1-clean.log")
    logger = setup_logging(log_file)
    
    logger.info("开始清洗CSV点云文件...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # 执行清洗
    stats = clean_csv_files(INPUT_DIR, OUTPUT_DIR, logger)
    
    logger.info(f"\n清洗完成! 共处理 {len(stats)} 个文件")


if __name__ == "__main__":
    main()
