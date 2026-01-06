#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3-random-crop.py: 智能随机裁剪点云块

功能:
1. 从1-clean-csv/目录下的csv文件中随机选择和裁剪小块
2. 支持单文件和多文件随机裁剪
3. 智能调整裁剪比例以达到目标点数:
   - 超出太多: 缩小CROP_RATIO
   - 超出不多: 使用FPS降采样
   - 略低于目标: 增大CROP_RATIO
   - 远低于目标: 重新选择区域
4. 支持裁剪前数据增强(旋转、翻转)
5. 从点云中心附近采样
6. 将小块左下角移动到坐标原点
7. 保存到3-random-crop-csv/目录
"""

import os
import sys
import numpy as np
from pathlib import Path
from glob import glob
import logging
from typing import List, Optional, Tuple, Dict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import (
    load_csv_pointcloud, save_csv_pointcloud,
    move_to_origin, setup_logging,
    smart_crop_with_augmentation,
    adaptive_crop_pointcloud,
    augment_pointcloud,
    farthest_point_sampling
)


# ============================================================================
# 配置类
# ============================================================================

class CropConfig:
    """裁剪配置类"""
    
    def __init__(self,
                 target_points: int = 2048,
                 initial_crop_ratio: float = 0.1,
                 tolerance_high: float = 1.5,
                 tolerance_low: float = 0.8,
                 min_points_threshold: float = 0.5,
                 max_attempts: int = 50,
                 center_sampling: bool = True,
                 center_range: float = 0.5,
                 enable_augmentation: bool = True,
                 augmentation_config: Dict = None,
                 enable_z_filter: bool = True,
                 z_filter_method: str = 'iqr',
                 z_filter_params: Dict = None):
        """
        Args:
            target_points: 目标点数
            initial_crop_ratio: 初始裁剪比例 (x和y方向，各为1/10)
            tolerance_high: 高容忍阈值 (超过target*tolerance_high需缩小裁剪框)
            tolerance_low: 低容忍阈值 (低于target*tolerance_low需增大裁剪框)
            min_points_threshold: 最小点数比例阈值 (低于此值重新采样)
            max_attempts: 最大尝试次数
            center_sampling: 是否从中心附近采样
            center_range: 中心采样范围
            enable_augmentation: 是否启用数据增强
            augmentation_config: 增强配置字典
            enable_z_filter: 是否启用z轴离群点过滤
            z_filter_method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
            z_filter_params: 过滤参数字典
                - zscore: {'threshold': 3.0}
                - iqr: {'k': 1.5}
                - percentile: {'lower': 1.0, 'upper': 99.0}
                - statistical: {'nb_neighbors': 20, 'std_ratio': 2.0}
        """
        self.target_points = target_points
        self.initial_crop_ratio = initial_crop_ratio
        self.tolerance_high = tolerance_high
        self.tolerance_low = tolerance_low
        self.min_points_threshold = min_points_threshold
        self.max_attempts = max_attempts
        self.center_sampling = center_sampling
        self.center_range = center_range
        self.enable_augmentation = enable_augmentation
        self.augmentation_config = augmentation_config or {
            'enable_rotation_z': True,
            'enable_rotation_xy': False,
            'enable_flip': True,
            'enable_jitter': False,
            'enable_scale': False,
        }
        # 离群点过滤配置
        self.enable_z_filter = enable_z_filter
        self.z_filter_method = z_filter_method
        self.z_filter_params = z_filter_params or self._get_default_filter_params(z_filter_method)
    
    @staticmethod
    def _get_default_filter_params(method: str) -> Dict:
        """获取默认的过滤参数"""
        defaults = {
            'zscore': {'zscore_threshold': 3.0},
            'iqr': {'iqr_k': 1.5},
            'percentile': {'percentile_lower': 1.0, 'percentile_upper': 99.0},
            'statistical': {'statistical_neighbors': 20, 'statistical_std_ratio': 2.0},
        }
        return defaults.get(method, {})
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'target_points': self.target_points,
            'initial_crop_ratio': self.initial_crop_ratio,
            'tolerance_high': self.tolerance_high,
            'tolerance_low': self.tolerance_low,
            'min_points_threshold': self.min_points_threshold,
            'max_attempts': self.max_attempts,
            'center_sampling': self.center_sampling,
            'center_range': self.center_range,
            'enable_augmentation': self.enable_augmentation,
            'augmentation_config': self.augmentation_config,
            'enable_z_filter': self.enable_z_filter,
            'z_filter_method': self.z_filter_method,
            'z_filter_params': self.z_filter_params,
        }


# ============================================================================
# 核心裁剪函数
# ============================================================================

def crop_single_pointcloud(points: np.ndarray,
                          config: CropConfig = None) -> Optional[np.ndarray]:
    """
    从单个点云中智能裁剪一个小块
    
    Args:
        points: (N, 3) numpy数组
        config: 裁剪配置
        
    Returns:
        cropped_points: 裁剪后的点云，如果失败返回None
    """
    if config is None:
        config = CropConfig()
    
    return smart_crop_with_augmentation(
        points,
        target_points=config.target_points,
        initial_crop_ratio=config.initial_crop_ratio,
        enable_augmentation=config.enable_augmentation,
        augmentation_config=config.augmentation_config,
        tolerance_high=config.tolerance_high,
        tolerance_low=config.tolerance_low,
        min_points_threshold=config.min_points_threshold,
        max_attempts=config.max_attempts,
        center_sampling=config.center_sampling,
        enable_z_filter=config.enable_z_filter,
        z_filter_method=config.z_filter_method,
        z_filter_params=config.z_filter_params
    )


def crop_from_single_file(csv_path: str,
                         num_samples: int = 1,
                         config: CropConfig = None) -> List[np.ndarray]:
    """
    从单个CSV文件中裁剪多个点云块
    
    Args:
        csv_path: CSV文件路径
        num_samples: 裁剪样本数
        config: 裁剪配置
        
    Returns:
        samples: 裁剪后的点云列表
    """
    if config is None:
        config = CropConfig()
    
    points, _ = load_csv_pointcloud(csv_path, coord_cols=['x [nm]', 'y [nm]', 'z [nm]'])
    
    samples = []
    for _ in range(num_samples):
        cropped = crop_single_pointcloud(points, config)
        if cropped is not None:
            samples.append(cropped)
    
    return samples


def crop_from_multiple_files(csv_paths: List[str],
                            num_samples: int = 100,
                            config: CropConfig = None,
                            seed: int = None) -> List[Tuple[np.ndarray, str]]:
    """
    从多个CSV文件中随机选择并裁剪点云块
    
    Args:
        csv_paths: CSV文件路径列表
        num_samples: 总共需要裁剪的样本数
        config: 裁剪配置
        seed: 随机种子
        
    Returns:
        samples: (点云, 源文件名) 元组列表
    """
    if config is None:
        config = CropConfig()
    
    if seed is not None:
        np.random.seed(seed)
    
    # 预加载所有点云
    all_points = {}
    for csv_path in csv_paths:
        points, _ = load_csv_pointcloud(csv_path, coord_cols=['x [nm]', 'y [nm]', 'z [nm]'])
        all_points[csv_path] = points
    
    samples = []
    for _ in range(num_samples):
        # 随机选择一个文件
        csv_path = np.random.choice(csv_paths)
        points = all_points[csv_path]
        
        # 裁剪
        cropped = crop_single_pointcloud(points, config)
        
        if cropped is not None:
            samples.append((cropped, os.path.basename(csv_path)))
    
    return samples


def _worker_crop(args):
    """并行裁剪的工作函数"""
    idx, csv_paths, all_points_dict, config_dict = args
    
    # 每个worker使用不同的随机种子
    np.random.seed(idx + int.from_bytes(os.urandom(4), byteorder='little'))
    
    # 从字典重建配置
    config = CropConfig(**config_dict)
    
    # 随机选择一个文件
    csv_path = np.random.choice(csv_paths)
    points = all_points_dict[csv_path]
    
    # 裁剪
    cropped = crop_single_pointcloud(points, config)
    
    if cropped is not None:
        return (idx, cropped, os.path.basename(csv_path))
    return None


def crop_from_multiple_files_parallel(csv_paths: List[str],
                                     num_samples: int = 100,
                                     config: CropConfig = None,
                                     logger: logging.Logger = None) -> List[Tuple[int, np.ndarray, str]]:
    """
    从多个CSV文件中随机选择并裁剪点云块（支持并行）
    
    Args:
        csv_paths: CSV文件路径列表
        num_samples: 总共需要裁剪的样本数
        config: 裁剪配置
        logger: 日志记录器
        
    Returns:
        samples: (索引, 点云, 源文件名) 元组列表
    """
    if config is None:
        config = CropConfig()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 预加载所有点云
    logger.info("正在加载所有点云数据...")
    all_points = {}
    for csv_path in csv_paths:
        points, _ = load_csv_pointcloud(csv_path, coord_cols=['x [nm]', 'y [nm]', 'z [nm]'])
        all_points[csv_path] = points
        logger.info(f"  加载: {os.path.basename(csv_path)} ({len(points)} points)")
    
    logger.info(f"开始智能裁剪 {num_samples} 个样本...")
    logger.info(f"  目标点数: {config.target_points}")
    logger.info(f"  初始裁剪比例: {config.initial_crop_ratio}")
    logger.info(f"  数据增强: {'开启' if config.enable_augmentation else '关闭'}")
    if config.enable_augmentation:
        aug_str = []
        if config.augmentation_config.get('enable_rotation_z'):
            aug_str.append('Z轴旋转')
        if config.augmentation_config.get('enable_flip'):
            aug_str.append('翻转')
        if config.augmentation_config.get('enable_rotation_xy'):
            aug_str.append('XY轴旋转')
        if config.augmentation_config.get('enable_jitter'):
            aug_str.append('抖动')
        if config.augmentation_config.get('enable_scale'):
            aug_str.append('缩放')
        logger.info(f"  增强方式: {', '.join(aug_str)}")
    logger.info(f"  z轴离群点过滤: {'开启' if config.enable_z_filter else '关闭'}")
    if config.enable_z_filter:
        logger.info(f"    过滤方法: {config.z_filter_method}")
        logger.info(f"    过滤参数: {config.z_filter_params}")
    
    samples = []
    failed_count = 0
    config_dict = config.to_dict()
    
    # 准备参数
    args_list = [
        (i, csv_paths, all_points, config_dict)
        for i in range(num_samples)
    ]
    
    # 串行处理（由于numpy数组在进程间传递开销大）
    for i, args in enumerate(args_list):
        result = _worker_crop(args)
        if result is not None:
            samples.append(result)
        else:
            failed_count += 1
        
        if (i + 1) % max(1, num_samples // 10) == 0:
            logger.info(f"  进度: {i + 1}/{num_samples} (成功: {len(samples)}, 失败: {failed_count})")
    
    logger.info(f"裁剪完成! 成功: {len(samples)}, 失败: {failed_count}")
    
    return samples


def save_cropped_samples(samples: List[Tuple[int, np.ndarray, str]], 
                        output_dir: str,
                        precision: int = 2,
                        logger: logging.Logger = None):
    """
    保存裁剪后的样本
    
    Args:
        samples: (索引, 点云, 源文件名) 元组列表
        output_dir: 输出目录
        precision: 坐标精度
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, points, source_file in samples:
        output_path = os.path.join(output_dir, f"{idx:04d}.csv")
        save_csv_pointcloud(output_path, points, precision=precision)
    
    logger.info(f"已保存 {len(samples)} 个样本到 {output_dir}")


# ============================================================================
# 便捷接口函数（用于外部调用）
# ============================================================================

def batch_crop_pointclouds(input_dir: str,
                          output_dir: str,
                          num_samples: int = 1024,
                          target_points: int = 2048,
                          initial_crop_ratio: float = 0.1,
                          enable_augmentation: bool = True,
                          augmentation_config: Dict = None,
                          enable_z_filter: bool = True,
                          z_filter_method: str = 'iqr',
                          z_filter_params: Dict = None,
                          logger: logging.Logger = None) -> List[Tuple[int, np.ndarray, str]]:
    """
    批量裁剪点云的便捷接口
    
    Args:
        input_dir: 输入目录（包含清洗后的CSV文件）
        output_dir: 输出目录
        num_samples: 需要生成的样本数
        target_points: 每个样本的点数
        initial_crop_ratio: 初始裁剪比例
        enable_augmentation: 是否启用数据增强
        augmentation_config: 增强配置
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
    if output_dir:
        save_cropped_samples(samples, output_dir, precision=2, logger=logger)
    
    return samples


def print_source_distribution(samples: List[Tuple[int, np.ndarray, str]], 
                              logger: logging.Logger = None):
    """
    打印源文件使用分布统计
    
    Args:
        samples: (索引, 点云, 源文件名) 元组列表
        logger: 日志记录器
    """
    from collections import Counter
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not samples:
        logger.warning("没有样本，无法统计源文件分布")
        return
    
    # 统计每个源文件的使用次数
    source_files = [source_file for _, _, source_file in samples]
    source_counter = Counter(source_files)
    total_samples = len(samples)
    
    logger.info("=" * 60)
    logger.info("源文件使用分布统计:")
    logger.info("-" * 60)
    
    # 按使用次数降序排列
    for source_file, count in source_counter.most_common():
        percentage = count / total_samples * 100
        bar = "█" * int(percentage / 2)  # 简单的进度条可视化
        logger.info(f"  {source_file}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    logger.info("-" * 60)
    logger.info(f"共使用 {len(source_counter)} 个源文件，生成 {total_samples} 个样本")
    
    # 计算均匀度指标
    if len(source_counter) > 1:
        expected_ratio = 1.0 / len(source_counter)
        max_ratio = max(count / total_samples for count in source_counter.values())
        min_ratio = min(count / total_samples for count in source_counter.values())
        
        logger.info(f"理想占比: {expected_ratio*100:.1f}% | 实际范围: {min_ratio*100:.1f}% ~ {max_ratio*100:.1f}%")
        
        # 警告如果分布极不均匀
        if max_ratio > expected_ratio * 2:
            logger.warning(f"⚠️ 警告: 源文件分布不均匀，最高占比({max_ratio*100:.1f}%)超过理想占比的2倍!")
    
    logger.info("=" * 60)


def main():
    # 设置路径
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "1-clean-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "3-random-crop-csv")
    
    # 设置参数
    NUM_SAMPLES = 100  # 先生成少量样本用于测试
    TARGET_POINTS = 2048  # 每个样本的点数
    INITIAL_CROP_RATIO = 0.1  # 初始裁剪比例
    
    # 数据增强配置
    AUGMENTATION_CONFIG = {
        'enable_rotation_z': True,    # 绕Z轴随机旋转
        'enable_rotation_xy': False,  # 绕XY轴小角度旋转（可能改变Z方向视角）
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
    # 其他方法的参数示例:
    # Z_FILTER_METHOD = 'iqr', Z_FILTER_PARAMS = {'iqr_k': 1.5}
    # Z_FILTER_METHOD = 'zscore', Z_FILTER_PARAMS = {'zscore_threshold': 3.0}
    # Z_FILTER_METHOD = 'percentile', Z_FILTER_PARAMS = {'percentile_lower': 1.0, 'percentile_upper': 99.0}
    # Z_FILTER_METHOD = 'statistical', Z_FILTER_PARAMS = {'statistical_neighbors': 20, 'statistical_std_ratio': 2.0}
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("开始智能随机裁剪点云...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"样本数: {NUM_SAMPLES}")
    logger.info(f"目标点数: {TARGET_POINTS}")
    logger.info(f"初始裁剪比例: {INITIAL_CROP_RATIO}")
    logger.info(f"z轴离群点过滤: {'开启' if ENABLE_Z_FILTER else '关闭'}")
    if ENABLE_Z_FILTER:
        logger.info(f"  过滤方法: {Z_FILTER_METHOD}")
        logger.info(f"  过滤参数: {Z_FILTER_PARAMS}")
    
    # 执行批量裁剪
    samples = batch_crop_pointclouds(
        INPUT_DIR, OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        target_points=TARGET_POINTS,
        initial_crop_ratio=INITIAL_CROP_RATIO,
        enable_augmentation=True,
        augmentation_config=AUGMENTATION_CONFIG,
        enable_z_filter=ENABLE_Z_FILTER,
        z_filter_method=Z_FILTER_METHOD,
        z_filter_params=Z_FILTER_PARAMS,
        logger=logger
    )
    
    logger.info(f"智能随机裁剪完成! 共生成 {len(samples)} 个样本")
    
    # 打印源文件使用分布统计
    print_source_distribution(samples, logger)


if __name__ == "__main__":
    main()
