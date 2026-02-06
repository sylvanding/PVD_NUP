#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2-ccp-samples-viz.py: 可视化CCP点云样本

功能:
1. 读取outputs/{stage}/csv/目录下的CCP样本
2. 生成2D可视化图像（俯视图，颜色按z轴）
3. 支持选择可视化前N个样本
4. 保存到outputs/{stage}/viz/目录
"""

import os
import sys
import argparse
from pathlib import Path
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import load_csv_pointcloud, visualize_pointcloud_2d, setup_logging

# CCP发育时期列表
STAGES = ['early', 'mid_early', 'middle', 'mid_late', 'late', 'mature']


def visualize_single_file(csv_path: str, output_dir: str) -> str:
    """可视化单个CCP样本"""
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
        title=f"{name_without_ext}\n({len(points)} points)",
        dpi=100
    )
    
    return output_path


def visualize_samples(input_dir: str,
                     output_dir: str,
                     max_samples: int = 100,
                     max_workers: int = 4,
                     logger: logging.Logger = None):
    """
    批量可视化CCP样本
    
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
    
    if total_files == 0:
        logger.warning(f"目录中没有CSV文件: {input_dir}")
        return
    
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
    parser = argparse.ArgumentParser(description='可视化CCP点云样本')
    parser.add_argument('--stage', type=str, default='early',
                       choices=STAGES,
                       help='发育时期 (default: early)')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='最大可视化样本数，设为0表示全部 (default: 10)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='并行工作进程数 (default: 4)')
    parser.add_argument('--output-dir', type=str,
                       default='/home/djx/data0/pvd_nup/ccp_stages/outputs',
                       help='数据根目录')
    
    args = parser.parse_args()
    
    # 设置路径
    input_dir = os.path.join(args.output_dir, args.stage, "csv")
    output_dir = os.path.join(args.output_dir, args.stage, "viz")
    
    max_samples = args.max_samples if args.max_samples > 0 else None
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("开始可视化CCP样本...")
    logger.info("=" * 60)
    logger.info(f"发育时期: {args.stage}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"最大可视化数: {max_samples}")
    
    # 执行可视化
    visualize_samples(
        input_dir, output_dir,
        max_samples=max_samples,
        max_workers=args.max_workers,
        logger=logger
    )
    
    logger.info("可视化完成!")


if __name__ == "__main__":
    main()
