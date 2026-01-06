#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-random-crop-batch.py: 批量生成点云块

功能:
1. 调用3-random-crop.py中的智能裁剪函数
2. 批量生成大量点云块（如1024个）
3. 支持自定义数据增强配置
4. 保存到5-random-crop-batch-csv/目录
"""

import os
import sys
from pathlib import Path
from glob import glob
import logging
from typing import Dict, List, Tuple
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import setup_logging

# 导入3-random-crop.py中的类和函数
from importlib.util import spec_from_file_location, module_from_spec

def import_module_from_file(module_name, file_path):
    """从文件路径导入模块"""
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入裁剪模块
crop_module = import_module_from_file(
    "random_crop", 
    os.path.join(Path(__file__).parent, "3-random-crop.py")
)

# 导入关键类和函数
CropConfig = crop_module.CropConfig
batch_crop_pointclouds = crop_module.batch_crop_pointclouds
crop_from_multiple_files_parallel = crop_module.crop_from_multiple_files_parallel
save_cropped_samples = crop_module.save_cropped_samples


def batch_crop(input_dir: str, 
               output_dir: str,
               num_samples: int = 1024,
               target_points: int = 2048,
               initial_crop_ratio: float = 0.1,
               enable_augmentation: bool = True,
               augmentation_config: Dict = None,
               tolerance_high: float = 1.5,
               tolerance_low: float = 0.8,
               max_attempts: int = 50,
               enable_z_filter: bool = True,
               z_filter_method: str = 'iqr',
               z_filter_params: Dict = None,
               logger: logging.Logger = None) -> List[Tuple[int, np.ndarray, str]]:
    """
    批量裁剪点云块
    
    Args:
        input_dir: 输入目录（1-clean-csv/）
        output_dir: 输出目录
        num_samples: 需要生成的样本数
        target_points: 每个样本的点数
        initial_crop_ratio: 初始裁剪比例
        enable_augmentation: 是否启用数据增强
        augmentation_config: 增强配置
        tolerance_high: 高容忍阈值
        tolerance_low: 低容忍阈值
        max_attempts: 最大尝试次数
        enable_z_filter: 是否启用z轴离群点过滤
        z_filter_method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
        z_filter_params: 过滤参数字典
        logger: 日志记录器
        
    Returns:
        samples: 裁剪后的样本列表
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 创建配置
    config = CropConfig(
        target_points=target_points,
        initial_crop_ratio=initial_crop_ratio,
        tolerance_high=tolerance_high,
        tolerance_low=tolerance_low,
        max_attempts=max_attempts,
        enable_augmentation=enable_augmentation,
        augmentation_config=augmentation_config,
        enable_z_filter=enable_z_filter,
        z_filter_method=z_filter_method,
        z_filter_params=z_filter_params
    )
    
    # 获取所有CSV文件
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    if len(csv_files) == 0:
        raise ValueError(f"在 {input_dir} 中没有找到CSV文件")
    
    logger.info(f"找到 {len(csv_files)} 个源文件")
    
    # 执行裁剪
    samples = crop_from_multiple_files_parallel(
        csv_files,
        num_samples=num_samples,
        config=config,
        logger=logger
    )
    
    # 保存结果
    save_cropped_samples(samples, output_dir, precision=2, logger=logger)
    
    return samples


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "1-clean-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "5-random-crop-batch-csv")
    
    # ========================================
    # 主要参数配置
    # ========================================
    NUM_SAMPLES = 2048  # 生成大批量样本
    TARGET_POINTS = 2048  # 每个样本的点数
    INITIAL_CROP_RATIO = 0.1  # 初始裁剪比例 (x和y方向各1/10)
    
    # 智能裁剪参数
    TOLERANCE_HIGH = 1.5  # 超过目标点数1.5倍时缩小裁剪框
    TOLERANCE_LOW = 0.8   # 低于目标点数0.8倍时增大裁剪框
    MAX_ATTEMPTS = 50     # 每个样本最大尝试次数
    
    # 数据增强配置
    ENABLE_AUGMENTATION = True
    AUGMENTATION_CONFIG = {
        'enable_rotation_z': True,    # 绕Z轴随机旋转
        'enable_rotation_xy': False,  # 绕XY轴小角度旋转
        'enable_flip': True,          # 随机翻转
        'enable_jitter': False,       # 噪声抖动
        'enable_scale': False,        # 随机缩放
    }
    
    # z轴离群点过滤配置
    ENABLE_Z_FILTER = True  # 启用z轴离群点过滤
    Z_FILTER_METHOD = 'statistical'  # 过滤方法: 'zscore', 'iqr', 'percentile', 'statistical'
    Z_FILTER_PARAMS = {
        'statistical_neighbors': 20,
        'statistical_std_ratio': 2.0,
    }
    
    # ========================================
    # 执行批量裁剪
    # ========================================
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("开始批量生成点云块...")
    logger.info("=" * 60)
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"样本数: {NUM_SAMPLES}")
    logger.info(f"目标点数: {TARGET_POINTS}")
    logger.info(f"初始裁剪比例: {INITIAL_CROP_RATIO}")
    logger.info(f"容忍阈值: 高={TOLERANCE_HIGH}, 低={TOLERANCE_LOW}")
    logger.info(f"数据增强: {'开启' if ENABLE_AUGMENTATION else '关闭'}")
    logger.info(f"z轴离群点过滤: {'开启' if ENABLE_Z_FILTER else '关闭'}")
    if ENABLE_Z_FILTER:
        logger.info(f"  过滤方法: {Z_FILTER_METHOD}, 参数: {Z_FILTER_PARAMS}")
    
    # 执行批量裁剪
    samples = batch_crop(
        INPUT_DIR, OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        target_points=TARGET_POINTS,
        initial_crop_ratio=INITIAL_CROP_RATIO,
        enable_augmentation=ENABLE_AUGMENTATION,
        augmentation_config=AUGMENTATION_CONFIG,
        tolerance_high=TOLERANCE_HIGH,
        tolerance_low=TOLERANCE_LOW,
        max_attempts=MAX_ATTEMPTS,
        enable_z_filter=ENABLE_Z_FILTER,
        z_filter_method=Z_FILTER_METHOD,
        z_filter_params=Z_FILTER_PARAMS,
        logger=logger
    )
    
    logger.info("=" * 60)
    logger.info(f"批量生成完成! 共生成 {len(samples)} 个样本")
    logger.info(f"成功率: {len(samples)/NUM_SAMPLES*100:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
