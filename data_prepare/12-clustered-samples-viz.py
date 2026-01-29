#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12-clustered-samples-viz.py: 可视化提取的核孔样本

功能:
1. 读取11-clustered-samples-csv/目录下的核孔样本
2. 生成2D可视化图像（俯视图，颜色按z轴）
3. 支持选择可视化前N个样本
4. 保存到12-clustered-samples-viz-png/目录
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
    """可视化单个核孔样本"""
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}.png")
    
    points, _ = load_csv_pointcloud(csv_path, coord_cols=['x [nm]', 'y [nm]', 'z [nm]'])
    
    visualize_pointcloud_2d(
        points, output_path,
        color_by_z=True,
        figsize=(6, 6),
        point_size=2.0,
        cmap='viridis',
        title=f"{name_without_ext[:40]}\n({len(points)} points)",
        dpi=100
    )
    
    return output_path


def visualize_samples(input_dir: str,
                     output_dir: str,
                     max_samples: int = 100,
                     max_workers: int = 4,
                     logger: logging.Logger = None):
    """
    批量可视化核孔样本
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        max_samples: 最大可视化样本数
        max_workers: 并行工作进程数
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    total_files = len(csv_files)
    logger.info(f"找到 {total_files} 个CSV文件")
    
    # 限制可视化数量
    if max_samples is not None and max_samples < total_files:
        csv_files = csv_files[:max_samples]
        logger.info(f"将可视化前 {max_samples} 个样本")
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(visualize_single_file, csv_path, output_dir): csv_path
            for csv_path in csv_files
        }
        
        completed = 0
        for future in as_completed(futures):
            csv_path = futures[future]
            filename = os.path.basename(csv_path)
            completed += 1
            try:
                output_path = future.result()
                if completed % 20 == 0 or completed == len(csv_files):
                    logger.info(f"进度: {completed}/{len(csv_files)} - 已可视化: {filename}")
            except Exception as e:
                logger.error(f"可视化 {filename} 时出错: {str(e)}")


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "11-clustered-samples-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "12-clustered-samples-viz-png")
    
    # 可视化参数
    MAX_SAMPLES = 100  # 最大可视化样本数，设为None表示全部可视化
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始可视化核孔样本...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"最大可视化数: {MAX_SAMPLES}")
    
    # 执行可视化
    visualize_samples(
        INPUT_DIR, OUTPUT_DIR,
        max_samples=MAX_SAMPLES,
        max_workers=4,
        logger=logger
    )
    
    logger.info("可视化完成!")


if __name__ == "__main__":
    main()
