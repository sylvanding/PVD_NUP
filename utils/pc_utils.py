"""
点云处理通用工具函数
用于NUP96大规模点云数据的预处理、可视化和数据增强
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Optional, List, Union
import os


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """设置日志配置"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_csv_pointcloud(csv_path: str, coord_cols: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    加载CSV格式的点云文件
    
    Args:
        csv_path: CSV文件路径
        coord_cols: 坐标列名列表，如果为None则自动检测
        
    Returns:
        points: (N, 3) numpy数组
        header: 原始表头列表
    """
    df = pd.read_csv(csv_path)
    header = list(df.columns)
    
    if coord_cols is None:
        # 自动检测坐标列
        coord_cols = []
        for axis in ['x', 'y', 'z']:
            found = False
            for col in df.columns:
                col_lower = col.lower()
                if axis in col_lower and 'nm' in col_lower:
                    coord_cols.append(col)
                    found = True
                    break
            if not found:
                raise ValueError(f"无法找到{axis}轴坐标列，表头为: {header}")
    
    points = df[coord_cols].values.astype(np.float64)
    return points, header


def save_csv_pointcloud(csv_path: str, points: np.ndarray, 
                        header: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                        precision: int = 2):
    """
    保存点云为CSV格式
    
    Args:
        csv_path: 输出文件路径
        points: (N, 3) numpy数组
        header: 列名
        precision: 坐标精度（小数位数）
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(points, columns=header)
    # 设置精度
    df = df.round(precision)
    df.to_csv(csv_path, index=False)


def move_to_origin(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将点云左下角移动到坐标原点
    
    Args:
        points: (N, 3) numpy数组
        
    Returns:
        shifted_points: 移动后的点云
        min_coords: 原始最小坐标（用于反向移动）
    """
    min_coords = points.min(axis=0)
    shifted_points = points - min_coords
    return shifted_points, min_coords


def get_pointcloud_stats(points: np.ndarray) -> dict:
    """
    获取点云统计信息
    
    Args:
        points: (N, 3) numpy数组
        
    Returns:
        统计信息字典
    """
    stats = {
        'num_points': len(points),
        'x_min': points[:, 0].min(),
        'x_max': points[:, 0].max(),
        'x_range': points[:, 0].max() - points[:, 0].min(),
        'y_min': points[:, 1].min(),
        'y_max': points[:, 1].max(),
        'y_range': points[:, 1].max() - points[:, 1].min(),
        'z_min': points[:, 2].min(),
        'z_max': points[:, 2].max(),
        'z_range': points[:, 2].max() - points[:, 2].min(),
        'x_mean': points[:, 0].mean(),
        'y_mean': points[:, 1].mean(),
        'z_mean': points[:, 2].mean(),
    }
    return stats


def visualize_pointcloud_2d(points: np.ndarray, out_file: str,
                            color_by_z: bool = True,
                            figsize: Tuple[int, int] = (10, 10),
                            point_size: float = 0.1,
                            cmap: str = 'viridis',
                            title: Optional[str] = None,
                            dpi: int = 150):
    """
    2D可视化点云（从上往下看，颜色按z轴）
    
    Args:
        points: (N, 3) numpy数组
        out_file: 输出文件路径
        color_by_z: 是否按z轴着色
        figsize: 图像大小
        point_size: 点大小
        cmap: 颜色映射
        title: 图像标题
        dpi: 图像分辨率
    """
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by_z:
        z_values = np.asarray(points[:, 2]).flatten().astype(np.float64)
        scatter = ax.scatter(points[:, 0], points[:, 1], 
                           c=z_values, s=point_size, 
                           cmap=cmap, alpha=0.8,
                           vmin=float(z_values.min()), vmax=float(z_values.max()))
        plt.colorbar(scatter, ax=ax, label='z [nm]')
    else:
        ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.8)
    
    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def farthest_point_sampling(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    最远点采样 (Farthest Point Sampling, FPS)
    
    Args:
        points: (N, 3) numpy数组
        num_points: 采样点数
        
    Returns:
        sampled_points: (num_points, 3) numpy数组
    """
    N = points.shape[0]
    if N <= num_points:
        return points
    
    # 初始化
    sampled_indices = np.zeros(num_points, dtype=np.int64)
    distances = np.full(N, np.inf)
    
    # 随机选择第一个点
    sampled_indices[0] = np.random.randint(0, N)
    
    for i in range(1, num_points):
        # 计算所有点到最新采样点的距离
        last_point = points[sampled_indices[i-1]]
        dist_to_last = np.sum((points - last_point) ** 2, axis=1)
        
        # 更新最小距离
        distances = np.minimum(distances, dist_to_last)
        
        # 选择距离最远的点
        sampled_indices[i] = np.argmax(distances)
    
    return points[sampled_indices]


def random_crop_pointcloud(points: np.ndarray, 
                          crop_ratio_x: float = 0.1, 
                          crop_ratio_y: float = 0.1,
                          center_sampling: bool = True,
                          center_range: float = 0.5) -> np.ndarray:
    """
    从点云中随机裁剪一个小块
    
    Args:
        points: (N, 3) numpy数组
        crop_ratio_x: x方向裁剪比例
        crop_ratio_y: y方向裁剪比例
        center_sampling: 是否从中心附近采样
        center_range: 中心采样范围（相对于可采样区域）
        
    Returns:
        cropped_points: 裁剪后的点云
    """
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    crop_size_x = x_range * crop_ratio_x
    crop_size_y = y_range * crop_ratio_y
    
    # 可采样的中心点范围
    sample_x_min = x_min + crop_size_x / 2
    sample_x_max = x_max - crop_size_x / 2
    sample_y_min = y_min + crop_size_y / 2
    sample_y_max = y_max - crop_size_y / 2
    
    if center_sampling:
        # 限制到中心区域
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        available_x = (sample_x_max - sample_x_min) * center_range / 2
        available_y = (sample_y_max - sample_y_min) * center_range / 2
        
        sample_x_min = center_x - available_x
        sample_x_max = center_x + available_x
        sample_y_min = center_y - available_y
        sample_y_max = center_y + available_y
    
    # 随机选择裁剪中心
    crop_center_x = np.random.uniform(sample_x_min, sample_x_max)
    crop_center_y = np.random.uniform(sample_y_min, sample_y_max)
    
    # 裁剪边界
    crop_x_min = crop_center_x - crop_size_x / 2
    crop_x_max = crop_center_x + crop_size_x / 2
    crop_y_min = crop_center_y - crop_size_y / 2
    crop_y_max = crop_center_y + crop_size_y / 2
    
    # 选择z轴全部范围
    mask = (
        (points[:, 0] >= crop_x_min) & (points[:, 0] <= crop_x_max) &
        (points[:, 1] >= crop_y_min) & (points[:, 1] <= crop_y_max)
    )
    
    return points[mask]


def random_crop_with_target_points(points: np.ndarray,
                                   target_points: int,
                                   crop_ratio_x: float = 0.1,
                                   crop_ratio_y: float = 0.1,
                                   min_points: int = 100,
                                   max_attempts: int = 100,
                                   center_sampling: bool = True,
                                   use_fps: bool = True) -> Optional[np.ndarray]:
    """
    随机裁剪点云块，确保点数符合要求
    
    Args:
        points: (N, 3) numpy数组
        target_points: 目标点数
        crop_ratio_x: x方向裁剪比例
        crop_ratio_y: y方向裁剪比例
        min_points: 最小点数阈值
        max_attempts: 最大尝试次数
        center_sampling: 是否从中心附近采样
        use_fps: 是否使用最远点采样（否则随机采样）
        
    Returns:
        cropped_points: 裁剪后的点云，如果失败返回None
    """
    for _ in range(max_attempts):
        cropped = random_crop_pointcloud(
            points, crop_ratio_x, crop_ratio_y, center_sampling
        )
        
        if len(cropped) >= min_points:
            if len(cropped) > target_points:
                # 降采样到目标点数
                if use_fps:
                    cropped = farthest_point_sampling(cropped, target_points)
                else:
                    indices = np.random.choice(len(cropped), target_points, replace=False)
                    cropped = cropped[indices]
            elif len(cropped) < target_points:
                # 点数不足，继续尝试
                continue
            
            # 移动到原点
            cropped, _ = move_to_origin(cropped)
            return cropped
    
    return None


def normalize_pointcloud(points: np.ndarray, 
                        normalize_per_shape: bool = False,
                        normalize_std_per_axis: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    归一化点云
    参考 ShapeNet15kPointClouds 的归一化方法
    
    Args:
        points: (N, 3) 或 (B, N, 3) numpy数组
        normalize_per_shape: 是否按单个形状归一化
        normalize_std_per_axis: 是否按轴归一化标准差
        
    Returns:
        normalized_points: 归一化后的点云
        mean: 均值
        std: 标准差
    """
    if points.ndim == 2:
        points = points[np.newaxis, ...]  # (1, N, 3)
        squeeze = True
    else:
        squeeze = False
    
    B, N, C = points.shape
    
    if normalize_per_shape:
        mean = points.mean(axis=1, keepdims=True)  # (B, 1, 3)
        if normalize_std_per_axis:
            std = points.std(axis=1, keepdims=True)  # (B, 1, 3)
        else:
            std = points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)  # (B, 1, 1)
    else:
        mean = points.reshape(-1, C).mean(axis=0).reshape(1, 1, C)  # (1, 1, 3)
        if normalize_std_per_axis:
            std = points.reshape(-1, C).std(axis=0).reshape(1, 1, C)  # (1, 1, 3)
        else:
            std = points.reshape(-1).std().reshape(1, 1, 1)  # (1, 1, 1)
    
    # 避免除零
    std = np.maximum(std, 1e-8)
    
    normalized = (points - mean) / std
    
    if squeeze:
        normalized = normalized.squeeze(0)
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    
    return normalized, mean, std


def denormalize_pointcloud(normalized_points: np.ndarray, 
                          mean: np.ndarray, 
                          std: np.ndarray) -> np.ndarray:
    """
    反归一化点云
    
    Args:
        normalized_points: 归一化的点云
        mean: 均值
        std: 标准差
        
    Returns:
        原始尺度的点云
    """
    return normalized_points * std + mean


def batch_normalize_pointclouds(points_list: List[np.ndarray],
                               normalize_per_shape: bool = True,
                               normalize_std_per_axis: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    批量归一化点云列表
    
    Args:
        points_list: 点云列表，每个元素是(N, 3)数组
        normalize_per_shape: 是否按单个形状归一化
        normalize_std_per_axis: 是否按轴归一化标准差
        
    Returns:
        normalized_batch: (B, N, 3) numpy数组
        means: 均值数组
        stds: 标准差数组
    """
    # 确保所有点云点数一致
    num_points = points_list[0].shape[0]
    assert all(p.shape[0] == num_points for p in points_list), "所有点云的点数必须一致"
    
    # Stack成batch
    batch = np.stack(points_list, axis=0)  # (B, N, 3)
    
    return normalize_pointcloud(batch, normalize_per_shape, normalize_std_per_axis)


# ============================================================================
# 数据增强函数
# ============================================================================

def rotate_pointcloud_z(points: np.ndarray, angle: float = None) -> np.ndarray:
    """
    绕z轴旋转点云
    
    Args:
        points: (N, 3) numpy数组
        angle: 旋转角度（弧度），如果为None则随机
        
    Returns:
        rotated_points: 旋转后的点云
    """
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    
    # 以点云中心为旋转中心
    center = points.mean(axis=0)
    centered = points - center
    rotated = centered @ rotation_matrix.T
    
    return rotated + center


def rotate_pointcloud_xy(points: np.ndarray, 
                        angle_x: float = None, 
                        angle_y: float = None,
                        max_angle: float = np.pi / 18) -> np.ndarray:
    """
    绕x和y轴小角度旋转点云（轻微倾斜）
    
    Args:
        points: (N, 3) numpy数组
        angle_x: 绕x轴旋转角度（弧度）
        angle_y: 绕y轴旋转角度（弧度）
        max_angle: 最大随机角度（默认10度）
        
    Returns:
        rotated_points: 旋转后的点云
    """
    if angle_x is None:
        angle_x = np.random.uniform(-max_angle, max_angle)
    if angle_y is None:
        angle_y = np.random.uniform(-max_angle, max_angle)
    
    # 绕x轴旋转
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    # 绕y轴旋转
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    rot_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    rotation_matrix = rot_x @ rot_y
    
    # 以点云中心为旋转中心
    center = points.mean(axis=0)
    centered = points - center
    rotated = centered @ rotation_matrix.T
    
    return rotated + center


def flip_pointcloud(points: np.ndarray, 
                   flip_x: bool = None, 
                   flip_y: bool = None) -> np.ndarray:
    """
    翻转点云
    
    Args:
        points: (N, 3) numpy数组
        flip_x: 是否沿x轴翻转，如果为None则随机
        flip_y: 是否沿y轴翻转，如果为None则随机
        
    Returns:
        flipped_points: 翻转后的点云
    """
    if flip_x is None:
        flip_x = np.random.rand() > 0.5
    if flip_y is None:
        flip_y = np.random.rand() > 0.5
    
    result = points.copy()
    center = points.mean(axis=0)
    
    if flip_x:
        result[:, 0] = 2 * center[0] - result[:, 0]
    if flip_y:
        result[:, 1] = 2 * center[1] - result[:, 1]
    
    return result


def jitter_pointcloud(points: np.ndarray, 
                     sigma: float = 0.01, 
                     clip: float = 0.05) -> np.ndarray:
    """
    对点云添加随机噪声抖动
    
    Args:
        points: (N, 3) numpy数组
        sigma: 噪声标准差（相对于点云尺度）
        clip: 噪声裁剪范围
        
    Returns:
        jittered_points: 抖动后的点云
    """
    # 计算点云尺度
    scale = np.max(points.max(axis=0) - points.min(axis=0))
    
    noise = np.clip(
        np.random.normal(0, sigma * scale, points.shape),
        -clip * scale,
        clip * scale
    )
    
    return points + noise


def scale_pointcloud(points: np.ndarray, 
                    scale_factor: float = None,
                    scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    缩放点云
    
    Args:
        points: (N, 3) numpy数组
        scale_factor: 缩放因子，如果为None则随机
        scale_range: 随机缩放范围
        
    Returns:
        scaled_points: 缩放后的点云
    """
    if scale_factor is None:
        scale_factor = np.random.uniform(*scale_range)
    
    center = points.mean(axis=0)
    centered = points - center
    scaled = centered * scale_factor
    
    return scaled + center


def augment_pointcloud(points: np.ndarray,
                      enable_rotation_z: bool = True,
                      enable_rotation_xy: bool = False,
                      enable_flip: bool = True,
                      enable_jitter: bool = False,
                      enable_scale: bool = False,
                      jitter_sigma: float = 0.01,
                      scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    综合数据增强函数
    
    Args:
        points: (N, 3) numpy数组
        enable_rotation_z: 是否启用z轴旋转
        enable_rotation_xy: 是否启用xy轴小角度旋转
        enable_flip: 是否启用翻转
        enable_jitter: 是否启用噪声抖动
        enable_scale: 是否启用缩放
        jitter_sigma: 抖动标准差
        scale_range: 缩放范围
        
    Returns:
        augmented_points: 增强后的点云
    """
    result = points.copy()
    
    if enable_rotation_z:
        result = rotate_pointcloud_z(result)
    
    if enable_rotation_xy:
        result = rotate_pointcloud_xy(result)
    
    if enable_flip:
        result = flip_pointcloud(result)
    
    if enable_scale:
        result = scale_pointcloud(result, scale_range=scale_range)
    
    if enable_jitter:
        result = jitter_pointcloud(result, sigma=jitter_sigma)
    
    return result


# ============================================================================
# 离群点过滤函数
# ============================================================================

def filter_outliers_zscore(points: np.ndarray,
                          axis: int = 2,
                          threshold: float = 3.0) -> np.ndarray:
    """
    使用Z-score方法过滤指定轴上的离群点
    
    Args:
        points: (N, 3) numpy数组
        axis: 要过滤的轴 (0=x, 1=y, 2=z)
        threshold: Z-score阈值，超过此值的点被视为离群点
        
    Returns:
        filtered_points: 过滤后的点云
    """
    values = points[:, axis]
    mean = np.mean(values)
    std = np.std(values)
    
    if std < 1e-8:  # 避免除零
        return points
    
    z_scores = np.abs((values - mean) / std)
    mask = z_scores < threshold
    
    return points[mask]


def filter_outliers_iqr(points: np.ndarray,
                       axis: int = 2,
                       k: float = 1.5) -> np.ndarray:
    """
    使用IQR(四分位距)方法过滤指定轴上的离群点
    
    Args:
        points: (N, 3) numpy数组
        axis: 要过滤的轴 (0=x, 1=y, 2=z)
        k: IQR倍数，默认1.5（标准箱线图）
        
    Returns:
        filtered_points: 过滤后的点云
    """
    values = points[:, axis]
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    mask = (values >= lower_bound) & (values <= upper_bound)
    
    return points[mask]


def filter_outliers_percentile(points: np.ndarray,
                              axis: int = 2,
                              lower_percentile: float = 1.0,
                              upper_percentile: float = 99.0) -> np.ndarray:
    """
    使用百分位数方法过滤指定轴上的离群点
    
    Args:
        points: (N, 3) numpy数组
        axis: 要过滤的轴 (0=x, 1=y, 2=z)
        lower_percentile: 下限百分位数
        upper_percentile: 上限百分位数
        
    Returns:
        filtered_points: 过滤后的点云
    """
    values = points[:, axis]
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    
    mask = (values >= lower_bound) & (values <= upper_bound)
    
    return points[mask]


def filter_outliers_statistical(points: np.ndarray,
                               nb_neighbors: int = 20,
                               std_ratio: float = 2.0) -> np.ndarray:
    """
    使用统计方法过滤离群点（基于邻域距离）
    需要安装open3d库
    
    Args:
        points: (N, 3) numpy数组
        nb_neighbors: 邻域点数
        std_ratio: 标准差倍数阈值
        
    Returns:
        filtered_points: 过滤后的点云
    """
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 统计离群点去除
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        return points[ind]
    except ImportError:
        # 如果没有open3d，使用Z-score方法作为后备
        return filter_outliers_zscore(points, axis=2, threshold=3.0)


def filter_z_outliers(points: np.ndarray,
                     method: str = 'iqr',
                     zscore_threshold: float = 3.0,
                     iqr_k: float = 1.5,
                     percentile_lower: float = 1.0,
                     percentile_upper: float = 99.0,
                     statistical_neighbors: int = 20,
                     statistical_std_ratio: float = 2.0) -> np.ndarray:
    """
    过滤z轴离群点的统一接口
    
    Args:
        points: (N, 3) numpy数组
        method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
        zscore_threshold: Z-score阈值
        iqr_k: IQR倍数
        percentile_lower: 下限百分位数
        percentile_upper: 上限百分位数
        statistical_neighbors: 统计方法邻域点数
        statistical_std_ratio: 统计方法标准差倍数
        
    Returns:
        filtered_points: 过滤后的点云
    """
    if method == 'zscore':
        return filter_outliers_zscore(points, axis=2, threshold=zscore_threshold)
    elif method == 'iqr':
        return filter_outliers_iqr(points, axis=2, k=iqr_k)
    elif method == 'percentile':
        return filter_outliers_percentile(
            points, axis=2, 
            lower_percentile=percentile_lower,
            upper_percentile=percentile_upper
        )
    elif method == 'statistical':
        return filter_outliers_statistical(
            points,
            nb_neighbors=statistical_neighbors,
            std_ratio=statistical_std_ratio
        )
    else:
        raise ValueError(f"Unknown outlier filter method: {method}")


# ============================================================================
# 智能裁剪函数
# ============================================================================

def adaptive_crop_pointcloud(points: np.ndarray,
                            target_points: int,
                            initial_crop_ratio: float = 0.1,
                            tolerance_high: float = 1.5,
                            tolerance_low: float = 0.8,
                            min_ratio: float = 0.02,
                            max_ratio: float = 0.3,
                            center_sampling: bool = True,
                            center_range: float = 0.5,
                            enable_z_filter: bool = True,
                            z_filter_method: str = 'iqr',
                            z_filter_params: dict = None) -> Tuple[np.ndarray, float, float]:
    """
    自适应裁剪点云，动态调整裁剪比例以达到目标点数
    
    策略:
    - 裁剪后先进行z轴离群点过滤
    - 超出太多(>tolerance_high): 缩小裁剪比例
    - 超出不多: 使用FPS降采样
    - 略低(<tolerance_low): 增大裁剪比例
    - 远低于目标: 返回None，需要重新选择区域
    
    Args:
        points: (N, 3) numpy数组
        target_points: 目标点数
        initial_crop_ratio: 初始裁剪比例
        tolerance_high: 高容忍阈值（超过target*tolerance_high需缩小）
        tolerance_low: 低容忍阈值（低于target*tolerance_low需增大）
        min_ratio: 最小裁剪比例
        max_ratio: 最大裁剪比例
        center_sampling: 是否从中心附近采样
        center_range: 中心采样范围
        enable_z_filter: 是否启用z轴离群点过滤
        z_filter_method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
        z_filter_params: 过滤参数字典
        
    Returns:
        (cropped_points, final_ratio_x, final_ratio_y) 或 (None, ratio, ratio) 如果失败
    """
    if z_filter_params is None:
        z_filter_params = {}
    
    crop_ratio = initial_crop_ratio
    max_iterations = 10
    
    for iteration in range(max_iterations):
        cropped = random_crop_pointcloud(
            points, crop_ratio, crop_ratio, center_sampling, center_range
        )
        
        if len(cropped) == 0:
            # 空区域，增大比例
            crop_ratio = min(crop_ratio * 1.5, max_ratio)
            continue
        
        # z轴离群点过滤
        if enable_z_filter:
            cropped = filter_z_outliers(cropped, method=z_filter_method, **z_filter_params)
        
        num_cropped = len(cropped)
        
        if num_cropped == 0:
            # 过滤后为空，增大比例
            crop_ratio = min(crop_ratio * 1.5, max_ratio)
            continue
        
        ratio = num_cropped / target_points
        
        if ratio > tolerance_high:
            # 超出太多，缩小裁剪比例
            shrink_factor = np.sqrt(target_points / num_cropped)
            crop_ratio = max(crop_ratio * shrink_factor * 0.9, min_ratio)
            
        elif ratio > 1.0:
            # 超出不多，使用FPS降采样
            cropped = farthest_point_sampling(cropped, target_points)
            cropped, _ = move_to_origin(cropped)
            return cropped, crop_ratio, crop_ratio
            
        elif ratio >= tolerance_low:
            # 略低于目标，增大裁剪比例
            expand_factor = np.sqrt(target_points / num_cropped)
            crop_ratio = min(crop_ratio * expand_factor * 1.1, max_ratio)
            
        else:
            # 远低于目标（可能是噪声或空白区域）
            # 返回None让调用者重新选择区域
            return None, crop_ratio, crop_ratio
    
    # 迭代结束，返回最后的结果
    if num_cropped >= target_points:
        cropped = farthest_point_sampling(cropped, target_points)
        cropped, _ = move_to_origin(cropped)
        return cropped, crop_ratio, crop_ratio
    
    return None, crop_ratio, crop_ratio


def smart_crop_with_augmentation(points: np.ndarray,
                                target_points: int,
                                initial_crop_ratio: float = 0.1,
                                enable_augmentation: bool = True,
                                augmentation_config: dict = None,
                                tolerance_high: float = 1.5,
                                tolerance_low: float = 0.8,
                                min_points_threshold: float = 0.3,
                                max_attempts: int = 50,
                                center_sampling: bool = True,
                                enable_z_filter: bool = True,
                                z_filter_method: str = 'iqr',
                                z_filter_params: dict = None) -> Optional[np.ndarray]:
    """
    智能裁剪点云，带数据增强和离群点过滤
    
    Args:
        points: (N, 3) numpy数组
        target_points: 目标点数
        initial_crop_ratio: 初始裁剪比例
        enable_augmentation: 是否启用数据增强
        augmentation_config: 增强配置字典
        tolerance_high: 高容忍阈值
        tolerance_low: 低容忍阈值
        min_points_threshold: 最小点数比例阈值（低于此值重新采样）
        max_attempts: 最大尝试次数
        center_sampling: 是否从中心附近采样
        enable_z_filter: 是否启用z轴离群点过滤
        z_filter_method: 过滤方法 ('zscore', 'iqr', 'percentile', 'statistical')
        z_filter_params: 过滤参数字典
        
    Returns:
        cropped_points: (target_points, 3) numpy数组，失败返回None
    """
    if augmentation_config is None:
        augmentation_config = {
            'enable_rotation_z': True,
            'enable_rotation_xy': False,
            'enable_flip': True,
            'enable_jitter': False,
            'enable_scale': False,
        }
    
    if z_filter_params is None:
        z_filter_params = {}
    
    for attempt in range(max_attempts):
        # 数据增强
        if enable_augmentation:
            augmented = augment_pointcloud(points, **augmentation_config)
        else:
            augmented = points.copy()
        
        # 自适应裁剪（包含离群点过滤）
        cropped, ratio_x, ratio_y = adaptive_crop_pointcloud(
            augmented,
            target_points=target_points,
            initial_crop_ratio=initial_crop_ratio,
            tolerance_high=tolerance_high,
            tolerance_low=tolerance_low,
            center_sampling=center_sampling,
            enable_z_filter=enable_z_filter,
            z_filter_method=z_filter_method,
            z_filter_params=z_filter_params
        )
        
        if cropped is not None and len(cropped) == target_points:
            return cropped
    
    return None
