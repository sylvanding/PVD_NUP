#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
9-clustering.py: 对NUP96点云中的核孔复合物进行聚类

功能:
1. 读取1-clean-csv/目录下的点云文件
2. 使用聚类算法识别单个核孔复合物
3. 保存聚类结果（新增cluster_id列）到9-clustering-csv/
4. 输出聚类分析结果（JSON格式）

聚类方法:
- DBSCAN: 基于密度的聚类，适合不规则形状
- HDBSCAN: 自适应版本的DBSCAN
- 2D投影聚类: 在XY平面上聚类（因为核孔在XY平面上分布）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN
from collections import Counter

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pc_utils import load_csv_pointcloud, setup_logging


@dataclass
class ClusterStats:
    """聚类统计信息"""
    filename: str
    total_points: int
    num_clusters: int
    noise_points: int
    noise_ratio: float
    avg_points_per_cluster: float
    std_points_per_cluster: float
    min_points_per_cluster: int
    max_points_per_cluster: int
    median_points_per_cluster: float
    cluster_sizes: List[int]
    anomaly_clusters: List[Dict]  # 异常聚类（过大或过小）


def cluster_dbscan_2d(points: np.ndarray, 
                      eps: float = 80.0, 
                      min_samples: int = 5) -> np.ndarray:
    """
    使用DBSCAN在XY平面上聚类
    
    核孔复合物直径约100nm，使用2D聚类因为核孔在XY平面上分布
    
    Args:
        points: (N, 3) 点云坐标
        eps: DBSCAN邻域半径 (nm)，核孔直径约100nm，建议60-100
        min_samples: 最小邻域点数
        
    Returns:
        labels: (N,) 聚类标签，-1表示噪声
    """
    # 只使用XY坐标进行聚类
    xy_points = points[:, :2]
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(xy_points)
    
    return labels


def cluster_dbscan_3d(points: np.ndarray, 
                      eps: float = 80.0, 
                      min_samples: int = 5,
                      z_weight: float = 0.5) -> np.ndarray:
    """
    使用DBSCAN进行3D聚类，z轴可加权
    
    Args:
        points: (N, 3) 点云坐标
        eps: DBSCAN邻域半径
        min_samples: 最小邻域点数
        z_weight: z轴权重（<1表示降低z轴影响）
        
    Returns:
        labels: (N,) 聚类标签
    """
    # 对z轴进行加权
    weighted_points = points.copy()
    weighted_points[:, 2] *= z_weight
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = clustering.fit_predict(weighted_points)
    
    return labels


def try_hdbscan(points: np.ndarray,
                min_cluster_size: int = 10,
                min_samples: int = 5,
                use_2d: bool = True) -> Optional[np.ndarray]:
    """
    尝试使用HDBSCAN进行聚类
    
    Args:
        points: (N, 3) 点云坐标
        min_cluster_size: 最小聚类大小
        min_samples: 最小样本数
        use_2d: 是否只使用XY坐标
        
    Returns:
        labels: (N,) 聚类标签，如果HDBSCAN不可用返回None
    """
    try:
        import hdbscan
        
        if use_2d:
            cluster_points = points[:, :2]
        else:
            cluster_points = points
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            core_dist_n_jobs=-1
        )
        labels = clusterer.fit_predict(cluster_points)
        
        return labels
    except ImportError:
        return None


def analyze_clusters(labels: np.ndarray,
                    filename: str,
                    anomaly_threshold_low: int = 10,
                    anomaly_threshold_high: int = 500) -> ClusterStats:
    """
    分析聚类结果
    
    Args:
        labels: 聚类标签
        filename: 文件名
        anomaly_threshold_low: 异常小聚类阈值
        anomaly_threshold_high: 异常大聚类阈值
        
    Returns:
        ClusterStats: 聚类统计信息
    """
    # 统计每个聚类的点数
    counter = Counter(labels)
    
    # 噪声点数
    noise_points = counter.get(-1, 0)
    
    # 聚类大小列表（排除噪声）
    cluster_sizes = [count for label, count in counter.items() if label != -1]
    
    # 聚类数量
    num_clusters = len(cluster_sizes)
    
    # 异常聚类检测
    anomaly_clusters = []
    for label, count in counter.items():
        if label == -1:
            continue
        if count < anomaly_threshold_low:
            anomaly_clusters.append({
                'cluster_id': int(label),
                'num_points': count,
                'type': 'too_small'
            })
        elif count > anomaly_threshold_high:
            anomaly_clusters.append({
                'cluster_id': int(label),
                'num_points': count,
                'type': 'too_large'
            })
    
    # 计算统计量
    if num_clusters > 0:
        cluster_sizes_arr = np.array(cluster_sizes)
        avg_points = float(np.mean(cluster_sizes_arr))
        std_points = float(np.std(cluster_sizes_arr))
        min_points = int(np.min(cluster_sizes_arr))
        max_points = int(np.max(cluster_sizes_arr))
        median_points = float(np.median(cluster_sizes_arr))
    else:
        avg_points = 0.0
        std_points = 0.0
        min_points = 0
        max_points = 0
        median_points = 0.0
    
    return ClusterStats(
        filename=filename,
        total_points=len(labels),
        num_clusters=num_clusters,
        noise_points=noise_points,
        noise_ratio=noise_points / len(labels) if len(labels) > 0 else 0.0,
        avg_points_per_cluster=avg_points,
        std_points_per_cluster=std_points,
        min_points_per_cluster=min_points,
        max_points_per_cluster=max_points,
        median_points_per_cluster=median_points,
        cluster_sizes=cluster_sizes,
        anomaly_clusters=anomaly_clusters
    )


def save_clustered_pointcloud(csv_path: str, 
                              points: np.ndarray, 
                              labels: np.ndarray,
                              coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                              precision: int = 2):
    """
    保存带聚类标签的点云
    
    Args:
        csv_path: 输出文件路径
        points: (N, 3) 点云坐标
        labels: (N,) 聚类标签
        coord_cols: 坐标列名
        precision: 坐标精度
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        coord_cols[0]: points[:, 0],
        coord_cols[1]: points[:, 1],
        coord_cols[2]: points[:, 2],
        'cluster_id': labels
    })
    
    # 设置精度
    for col in coord_cols:
        df[col] = df[col].round(precision)
    
    df.to_csv(csv_path, index=False)


def cluster_pointcloud(points: np.ndarray,
                      method: str = 'dbscan_2d',
                      eps: float = 80.0,
                      min_samples: int = 5,
                      min_cluster_size: int = 10,
                      z_weight: float = 0.5) -> np.ndarray:
    """
    聚类点云的统一接口
    
    Args:
        points: (N, 3) 点云坐标
        method: 聚类方法 ('dbscan_2d', 'dbscan_3d', 'hdbscan_2d', 'hdbscan_3d')
        eps: DBSCAN邻域半径
        min_samples: 最小邻域点数
        min_cluster_size: HDBSCAN最小聚类大小
        z_weight: 3D聚类时z轴权重
        
    Returns:
        labels: (N,) 聚类标签
    """
    if method == 'dbscan_2d':
        return cluster_dbscan_2d(points, eps=eps, min_samples=min_samples)
    elif method == 'dbscan_3d':
        return cluster_dbscan_3d(points, eps=eps, min_samples=min_samples, z_weight=z_weight)
    elif method == 'hdbscan_2d':
        labels = try_hdbscan(points, min_cluster_size=min_cluster_size, 
                            min_samples=min_samples, use_2d=True)
        if labels is None:
            logging.warning("HDBSCAN不可用，回退到DBSCAN")
            return cluster_dbscan_2d(points, eps=eps, min_samples=min_samples)
        return labels
    elif method == 'hdbscan_3d':
        labels = try_hdbscan(points, min_cluster_size=min_cluster_size, 
                            min_samples=min_samples, use_2d=False)
        if labels is None:
            logging.warning("HDBSCAN不可用，回退到DBSCAN")
            return cluster_dbscan_3d(points, eps=eps, min_samples=min_samples, z_weight=z_weight)
        return labels
    else:
        raise ValueError(f"未知的聚类方法: {method}")


def process_all_files(input_dir: str,
                     output_dir: str,
                     method: str = 'dbscan_2d',
                     eps: float = 80.0,
                     min_samples: int = 5,
                     min_cluster_size: int = 10,
                     z_weight: float = 0.5,
                     coord_cols: List[str] = ['x [nm]', 'y [nm]', 'z [nm]'],
                     logger: logging.Logger = None) -> List[ClusterStats]:
    """
    处理所有点云文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        method: 聚类方法
        eps: DBSCAN邻域半径
        min_samples: 最小邻域点数
        min_cluster_size: HDBSCAN最小聚类大小
        z_weight: 3D聚类时z轴权重
        coord_cols: 坐标列名
        logger: 日志记录器
        
    Returns:
        all_stats: 所有文件的聚类统计列表
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    all_stats = []
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"处理文件: {filename}")
        
        try:
            # 加载点云
            points, _ = load_csv_pointcloud(csv_path, coord_cols=coord_cols)
            logger.info(f"点数: {len(points)}")
            
            # 聚类
            labels = cluster_pointcloud(
                points, 
                method=method,
                eps=eps,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                z_weight=z_weight
            )
            
            # 分析聚类结果
            stats = analyze_clusters(labels, filename)
            all_stats.append(stats)
            
            logger.info(f"聚类数: {stats.num_clusters}")
            logger.info(f"噪声点: {stats.noise_points} ({stats.noise_ratio*100:.1f}%)")
            logger.info(f"平均每核孔点数: {stats.avg_points_per_cluster:.1f} ± {stats.std_points_per_cluster:.1f}")
            logger.info(f"核孔点数范围: [{stats.min_points_per_cluster}, {stats.max_points_per_cluster}]")
            if stats.anomaly_clusters:
                logger.warning(f"异常核孔数: {len(stats.anomaly_clusters)}")
            
            # 保存结果
            output_path = os.path.join(output_dir, filename)
            save_clustered_pointcloud(output_path, points, labels, coord_cols=coord_cols)
            logger.info(f"已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_stats


def save_analysis_report(all_stats: List[ClusterStats],
                        output_path: str,
                        method: str,
                        params: dict,
                        logger: logging.Logger = None):
    """
    保存聚类分析报告
    
    Args:
        all_stats: 所有文件的聚类统计列表
        output_path: 输出JSON文件路径
        method: 使用的聚类方法
        params: 聚类参数
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 汇总统计
    total_clusters = sum(s.num_clusters for s in all_stats)
    total_points = sum(s.total_points for s in all_stats)
    total_noise = sum(s.noise_points for s in all_stats)
    
    all_cluster_sizes = []
    for s in all_stats:
        all_cluster_sizes.extend(s.cluster_sizes)
    
    if all_cluster_sizes:
        cluster_sizes_arr = np.array(all_cluster_sizes)
        global_avg = float(np.mean(cluster_sizes_arr))
        global_std = float(np.std(cluster_sizes_arr))
        global_median = float(np.median(cluster_sizes_arr))
        global_min = int(np.min(cluster_sizes_arr))
        global_max = int(np.max(cluster_sizes_arr))
    else:
        global_avg = global_std = global_median = 0.0
        global_min = global_max = 0
    
    # 异常聚类汇总
    all_anomalies = []
    for s in all_stats:
        for a in s.anomaly_clusters:
            a['source_file'] = s.filename
            all_anomalies.append(a)
    
    report = {
        'method': method,
        'parameters': params,
        'summary': {
            'total_files': len(all_stats),
            'total_points': total_points,
            'total_clusters': total_clusters,
            'total_noise_points': total_noise,
            'global_noise_ratio': total_noise / total_points if total_points > 0 else 0.0,
            'global_avg_points_per_cluster': global_avg,
            'global_std_points_per_cluster': global_std,
            'global_median_points_per_cluster': global_median,
            'global_min_points_per_cluster': global_min,
            'global_max_points_per_cluster': global_max,
            'total_anomaly_clusters': len(all_anomalies)
        },
        'anomaly_clusters': all_anomalies,
        'per_file_stats': [asdict(s) for s in all_stats]
    }
    
    # 保存报告
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("聚类分析报告汇总:")
    logger.info(f"  总文件数: {len(all_stats)}")
    logger.info(f"  总点数: {total_points}")
    logger.info(f"  总核孔数: {total_clusters}")
    logger.info(f"  总噪声点: {total_noise} ({total_noise/total_points*100:.1f}%)")
    logger.info(f"  平均每核孔点数: {global_avg:.1f} ± {global_std:.1f}")
    logger.info(f"  中位数: {global_median:.1f}")
    logger.info(f"  核孔点数范围: [{global_min}, {global_max}]")
    logger.info(f"  异常核孔数: {len(all_anomalies)}")
    logger.info(f"报告已保存到: {output_path}")


def main():
    # ========== 配置参数 ==========
    DATA_ROOT = "/home/djx/data/nup96-large"
    INPUT_DIR = os.path.join(DATA_ROOT, "1-clean-csv")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "9-clustering-csv")
    REPORT_PATH = os.path.join(OUTPUT_DIR, "clustering_analysis.json")
    
    # 聚类参数
    # 核孔复合物直径约100nm，选择eps=80作为合理的聚类半径
    CLUSTERING_METHOD = 'dbscan_2d'  # 推荐使用2D聚类，因为核孔在XY平面分布
    EPS = 80.0  # DBSCAN邻域半径 (nm)
    MIN_SAMPLES = 5  # 最小邻域点数
    MIN_CLUSTER_SIZE = 10  # HDBSCAN最小聚类大小
    Z_WEIGHT = 0.5  # 3D聚类时z轴权重
    
    # ========== 执行聚类 ==========
    logger = setup_logging()
    
    logger.info("开始对NUP96点云进行核孔聚类...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"聚类方法: {CLUSTERING_METHOD}")
    logger.info(f"参数: eps={EPS}, min_samples={MIN_SAMPLES}")
    
    # 处理所有文件
    all_stats = process_all_files(
        INPUT_DIR, OUTPUT_DIR,
        method=CLUSTERING_METHOD,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        min_cluster_size=MIN_CLUSTER_SIZE,
        z_weight=Z_WEIGHT,
        logger=logger
    )
    
    # 保存分析报告
    params = {
        'eps': EPS,
        'min_samples': MIN_SAMPLES,
        'min_cluster_size': MIN_CLUSTER_SIZE,
        'z_weight': Z_WEIGHT
    }
    save_analysis_report(all_stats, REPORT_PATH, CLUSTERING_METHOD, params, logger)
    
    logger.info("\n聚类完成!")


if __name__ == "__main__":
    main()
