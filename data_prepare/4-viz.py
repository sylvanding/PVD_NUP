#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4-viz.py: 可视化裁剪后的点云块

功能:
1. 读取3-random-crop-csv/目录下所有csv文件
2. 生成2D可视化图像
3. 保存到4-random-crop-viz-png/目录
"""

import os
import sys
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import load_csv_pointcloud, visualize_pointcloud_2d, setup_logging


def visualize_single_file(csv_path: str, output_dir: str) -> str:
    """可视化单个CSV文件"""
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}.png")
    
    points, _ = load_csv_pointcloud(csv_path, coord_cols=['x [nm]', 'y [nm]', 'z [nm]'])
    
    visualize_pointcloud_2d(
        points, output_path,
        color_by_z=True,
        figsize=(8, 8),
        point_size=1.0,
        cmap='viridis',
        title=f"{name_without_ext} ({len(points)} points)",
        dpi=100
    )
    
    return output_path


def visualize_csv_files(input_dir: str, output_dir: str, 
                       max_workers: int = 4, logger: logging.Logger = None):
    """
    批量可视化CSV点云文件
    
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
    INPUT_DIR = os.path.join(DATA_ROOT, "3-random-crop-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "4-random-crop-viz-png")
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始可视化裁剪后的点云块...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # 执行可视化
    visualize_csv_files(INPUT_DIR, OUTPUT_DIR, max_workers=4, logger=logger)
    
    logger.info("可视化完成!")


if __name__ == "__main__":
    main()
