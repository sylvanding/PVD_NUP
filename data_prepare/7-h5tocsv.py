#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
7-h5tocsv.py: 将H5格式点云转换回CSV（反归一化）

功能:
1. 读取6-pc-blocks.h5文件
2. 反归一化点云
3. 保存为CSV格式（可选择只转换前N个）
4. 保存到7-h5tocsv-csv/目录
"""

import os
import sys
import numpy as np
from pathlib import Path
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    save_csv_pointcloud, setup_logging, denormalize_pointcloud
)

# 导入6-csv2h5.py中的加载函数
from importlib.util import spec_from_file_location, module_from_spec

def import_module_from_file(module_name, file_path):
    """从文件路径导入模块"""
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入H5加载模块
h5_module = import_module_from_file(
    "csv2h5", 
    os.path.join(Path(__file__).parent, "6-csv2h5.py")
)


def h5_to_csv(h5_path: str, output_dir: str,
             num_samples: int = None,
             precision: int = 2,
             logger: logging.Logger = None):
    """
    将H5格式点云转换为CSV
    
    Args:
        h5_path: H5文件路径
        output_dir: 输出目录
        num_samples: 转换的样本数（None表示全部）
        precision: 坐标精度
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 加载H5文件
    points, mean, std, metadata = h5_module.load_from_h5(h5_path, logger=logger)
    
    # 确定转换数量
    total_samples = points.shape[0]
    if num_samples is None:
        num_samples = total_samples
    else:
        num_samples = min(num_samples, total_samples)
    
    logger.info(f"将转换 {num_samples}/{total_samples} 个样本")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取文件名（如果有）
    filenames = metadata.get('filenames', None)
    
    # 反归一化并保存
    for i in range(num_samples):
        # 获取当前样本的归一化参数
        if metadata['normalize_per_shape']:
            sample_mean = mean[i]  # (1, 3) or (1, 1)
            sample_std = std[i]
        else:
            sample_mean = mean
            sample_std = std
        
        # 反归一化
        denormalized = denormalize_pointcloud(points[i], sample_mean, sample_std)
        # 确保输出是2D (N, 3)，处理广播产生的额外维度
        denormalized = np.squeeze(denormalized)
        
        # 确定输出文件名
        if filenames is not None and i < len(filenames):
            output_name = filenames[i]
        else:
            output_name = f"{i:04d}.csv"
        
        output_path = os.path.join(output_dir, output_name)
        
        # 保存
        save_csv_pointcloud(output_path, denormalized, precision=precision)
        
        if (i + 1) % 10 == 0 or i == num_samples - 1:
            logger.info(f"  已转换: {i + 1}/{num_samples}")
    
    logger.info(f"转换完成! 已保存到 {output_dir}")


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_PATH = os.path.join(DATA_ROOT, "6-pc-blocks.h5")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "7-h5tocsv-csv")
    
    # 设置参数
    NUM_SAMPLES = 20  # 只转换前10个样本
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始将H5点云转换为CSV格式...")
    logger.info(f"输入文件: {INPUT_PATH}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"转换样本数: {NUM_SAMPLES}")
    
    # 执行转换
    h5_to_csv(
        INPUT_PATH, OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        precision=2,
        logger=logger
    )
    
    logger.info("转换完成!")


if __name__ == "__main__":
    main()
