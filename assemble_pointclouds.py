"""
NUP96 点云拼接脚本

将多个生成的核孔复合物点云拼接成一个大的点云，避免重叠。

用法:
    python assemble_pointclouds.py --input_dir ./generated_nups --output assembled_nup.csv
    
    # 或从 H5 文件加载
    python assemble_pointclouds.py --input_h5 ./generated_nups/generated_nups.h5 --output assembled_nup.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Tuple, Optional


def load_from_csv_dir(csv_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    从 CSV 目录加载所有点云
    
    Args:
        csv_dir: CSV 文件目录
        
    Returns:
        points: (N, M, 3) 所有点云
        filenames: 文件名列表
    """
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        raise ValueError(f"在 {csv_dir} 中没有找到 CSV 文件")
    
    print(f"从 {csv_dir} 加载 {len(csv_files)} 个点云")
    
    all_points = []
    filenames = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # 尝试不同的列名格式
        if 'x [nm]' in df.columns:
            points = df[['x [nm]', 'y [nm]', 'z [nm]']].values
        elif 'x' in df.columns:
            points = df[['x', 'y', 'z']].values
        else:
            # 假设前三列是 x, y, z
            points = df.iloc[:, :3].values
        
        all_points.append(points)
        filenames.append(csv_file.stem)
    
    return np.array(all_points), filenames


def load_from_h5(h5_path: str) -> Tuple[np.ndarray, None]:
    """
    从 H5 文件加载点云
    
    Args:
        h5_path: H5 文件路径
        
    Returns:
        points: (N, M, 3) 所有点云（反归一化后的）
        filenames: None
    """
    print(f"从 {h5_path} 加载点云")
    
    with h5py.File(h5_path, 'r') as f:
        if 'points_denormalized' in f:
            points = f['points_denormalized'][:]
        elif 'points' in f:
            points = f['points'][:]
        else:
            raise ValueError("H5 文件中没有找到 'points_denormalized' 或 'points' 数据集")
        
        num_samples = f.attrs.get('num_samples', len(points))
        print(f"  加载了 {num_samples} 个点云")
    
    return points, None


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云的边界框
    
    Args:
        points: (N, 3) 点云
        
    Returns:
        min_coords: (3,) 最小坐标
        max_coords: (3,) 最大坐标
    """
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    return min_coords, max_coords


def compute_size(points: np.ndarray) -> np.ndarray:
    """
    计算点云的尺寸
    
    Args:
        points: (N, 3) 点云
        
    Returns:
        size: (3,) 在每个维度上的尺寸
    """
    min_coords, max_coords = compute_bounding_box(points)
    return max_coords - min_coords


def check_overlap(pos1: np.ndarray, size1: np.ndarray, 
                  pos2: np.ndarray, size2: np.ndarray, 
                  margin: float = 0.0) -> bool:
    """
    检查两个边界框是否重叠
    
    Args:
        pos1: 第一个点云的位置（中心或左下角）
        size1: 第一个点云的尺寸
        pos2: 第二个点云的位置
        size2: 第二个点云的尺寸
        margin: 额外的间隔
        
    Returns:
        是否重叠
    """
    # 使用左下角坐标
    for d in range(3):  # x, y, z
        if pos1[d] + size1[d] + margin <= pos2[d]:
            return False
        if pos2[d] + size2[d] + margin <= pos1[d]:
            return False
    return True


def place_random_no_overlap(all_points: np.ndarray, 
                            margin: float = 10.0,
                            max_attempts: int = 1000,
                            area_scale: float = 5.0,
                            seed: Optional[int] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    随机放置点云，避免重叠
    
    Args:
        all_points: (N, M, 3) 所有点云
        margin: 点云之间的最小间隔 (nm)
        max_attempts: 每个点云的最大尝试次数
        area_scale: 放置区域相对于点云总尺寸的缩放因子
        seed: 随机种子
        
    Returns:
        assembled: 拼接后的大点云 (N_total, 3)
        positions: 每个点云的放置位置列表
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_clouds = len(all_points)
    
    # 计算每个点云的尺寸
    sizes = []
    for i in range(num_clouds):
        size = compute_size(all_points[i])
        sizes.append(size)
    sizes = np.array(sizes)
    
    # 计算平均尺寸
    avg_size = sizes.mean(axis=0)
    max_size = sizes.max(axis=0)
    
    # 估算放置区域大小
    # 假设大致按网格排列，计算需要多大的区域
    grid_dim = int(np.ceil(np.sqrt(num_clouds)))
    area_size = np.array([
        grid_dim * (max_size[0] + margin) * area_scale,
        grid_dim * (max_size[1] + margin) * area_scale,
        max_size[2] * 2  # z 方向给一些余量
    ])
    
    print(f"放置区域大小: {area_size} nm")
    print(f"平均点云尺寸: {avg_size} nm")
    print(f"间隔: {margin} nm")
    
    placed_positions = []  # 已放置点云的位置
    placed_sizes = []      # 已放置点云的尺寸
    
    assembled_points = []
    positions = []
    
    for i in range(num_clouds):
        cloud = all_points[i]
        cloud_size = sizes[i]
        
        placed = False
        for attempt in range(max_attempts):
            # 随机生成位置
            pos = np.array([
                np.random.uniform(0, area_size[0] - cloud_size[0]),
                np.random.uniform(0, area_size[1] - cloud_size[1]),
                np.random.uniform(-cloud_size[2]/2, cloud_size[2]/2)
            ])
            
            # 检查与已放置点云的重叠
            overlap = False
            for j, (placed_pos, placed_size) in enumerate(zip(placed_positions, placed_sizes)):
                if check_overlap(pos, cloud_size, placed_pos, placed_size, margin):
                    overlap = True
                    break
            
            if not overlap:
                # 成功放置
                placed_positions.append(pos)
                placed_sizes.append(cloud_size)
                
                # 将点云居中到边界框的左下角
                cloud_min = cloud.min(axis=0)
                translated = cloud - cloud_min + pos
                
                assembled_points.append(translated)
                positions.append(pos)
                placed = True
                break
        
        if not placed:
            print(f"警告: 点云 {i} 在 {max_attempts} 次尝试后无法放置，强制放置")
            # 强制放置在一个较远的位置
            fallback_pos = np.array([
                i * (max_size[0] + margin * 2),
                0,
                0
            ])
            cloud_min = cloud.min(axis=0)
            translated = cloud - cloud_min + fallback_pos
            assembled_points.append(translated)
            positions.append(fallback_pos)
            placed_positions.append(fallback_pos)
            placed_sizes.append(cloud_size)
        
        if (i + 1) % 10 == 0:
            print(f"  已放置 {i+1}/{num_clouds} 个点云")
    
    assembled = np.vstack(assembled_points)
    print(f"拼接完成! 总共 {len(assembled)} 个点")
    
    return assembled, positions


def place_grid(all_points: np.ndarray, 
               margin: float = 10.0,
               cols: Optional[int] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    网格放置点云
    
    Args:
        all_points: (N, M, 3) 所有点云
        margin: 点云之间的间隔 (nm)
        cols: 每行放置的数量，None 则自动计算
        
    Returns:
        assembled: 拼接后的大点云 (N_total, 3)
        positions: 每个点云的放置位置列表
    """
    num_clouds = len(all_points)
    
    if cols is None:
        cols = int(np.ceil(np.sqrt(num_clouds)))
    
    # 计算每个点云的尺寸
    sizes = []
    for i in range(num_clouds):
        size = compute_size(all_points[i])
        sizes.append(size)
    sizes = np.array(sizes)
    
    # 使用最大尺寸作为网格单元大小
    max_size = sizes.max(axis=0)
    cell_size = max_size + margin
    
    assembled_points = []
    positions = []
    
    for i in range(num_clouds):
        cloud = all_points[i]
        
        row = i // cols
        col = i % cols
        
        pos = np.array([
            col * cell_size[0],
            row * cell_size[1],
            0
        ])
        
        # 将点云居中到网格单元
        cloud_min = cloud.min(axis=0)
        cloud_center = (cloud.max(axis=0) + cloud_min) / 2
        cell_center = pos + cell_size / 2
        
        # 平移到网格单元中心
        offset = cell_center - cloud_center
        offset[2] = 0  # 保持 z 方向不变
        translated = cloud + offset
        
        assembled_points.append(translated)
        positions.append(pos)
    
    assembled = np.vstack(assembled_points)
    print(f"网格拼接完成! 总共 {len(assembled)} 个点 ({cols} 列)")
    
    return assembled, positions


def save_assembled(assembled: np.ndarray, output_path: str, 
                   positions: Optional[List[np.ndarray]] = None):
    """
    保存拼接后的点云
    
    Args:
        assembled: (N, 3) 拼接后的点云
        output_path: 输出路径
        positions: 各个点云的位置信息（可选）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据扩展名选择保存格式
    if output_path.suffix.lower() == '.csv':
        df = pd.DataFrame(assembled, columns=['x [nm]', 'y [nm]', 'z [nm]'])
        df.to_csv(output_path, index=False)
        print(f"保存 CSV 到: {output_path}")
    
    elif output_path.suffix.lower() == '.h5':
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('points', data=assembled)
            f.attrs['num_points'] = len(assembled)
            if positions is not None:
                f.create_dataset('positions', data=np.array(positions))
        print(f"保存 H5 到: {output_path}")
    
    else:
        # 默认保存为 CSV
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame(assembled, columns=['x [nm]', 'y [nm]', 'z [nm]'])
        df.to_csv(csv_path, index=False)
        print(f"保存 CSV 到: {csv_path}")


def visualize_assembled(assembled: np.ndarray, output_path: str, 
                        sample_ratio: float = 1.0):
    """
    可视化拼接后的点云
    
    Args:
        assembled: (N, 3) 拼接后的点云
        output_path: 输出图片路径
        sample_ratio: 采样比例（大点云时降采样以加快可视化）
    """
    import matplotlib.pyplot as plt
    
    # 如果点太多，进行采样
    if sample_ratio < 1.0 and len(assembled) > 10000:
        num_sample = int(len(assembled) * sample_ratio)
        indices = np.random.choice(len(assembled), num_sample, replace=False)
        points = assembled[indices]
    else:
        points = assembled
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY 视图
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Y [nm]')
    ax.set_title('XY View (color: Z)')
    ax.set_aspect('equal')
    
    # XZ 视图
    ax = axes[1]
    ax.scatter(points[:, 0], points[:, 2], c=points[:, 1], cmap='plasma', s=0.5, alpha=0.6)
    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Z [nm]')
    ax.set_title('XZ View (color: Y)')
    ax.set_aspect('equal')
    
    # YZ 视图
    ax = axes[2]
    ax.scatter(points[:, 1], points[:, 2], c=points[:, 0], cmap='coolwarm', s=0.5, alpha=0.6)
    ax.set_xlabel('Y [nm]')
    ax.set_ylabel('Z [nm]')
    ax.set_title('YZ View (color: X)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"可视化保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='NUP96 点云拼接')
    
    # 输入源（二选一）
    parser.add_argument('--input_dir', type=str, default="/data0/djx/pvd_nup/output/20260109_044249_clustering/generated_nups",
                        help='包含 CSV 文件的输入目录')
    parser.add_argument('--input_h5', type=str, default=None,
                        help='输入 H5 文件路径')
    
    # 输出
    parser.add_argument('--output', type=str, default='/data0/djx/pvd_nup/output/20260109_044249_clustering/generated_nups/assembled_nup.csv',
                        help='输出文件路径 (支持 .csv 或 .h5)')
    
    # 放置参数
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'grid'],
                        help='放置模式: random (随机避免重叠) 或 grid (网格排列)')
    parser.add_argument('--margin', type=float, default=20.0,
                        help='点云之间的最小间隔 (nm)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--cols', type=int, default=None,
                        help='网格模式下每行的列数')
    parser.add_argument('--max_attempts', type=int, default=1000,
                        help='随机模式下每个点云的最大放置尝试次数')
    parser.add_argument('--area_scale', type=float, default=3.0,
                        help='随机模式下放置区域的缩放因子')
    
    # 可视化
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='生成可视化图片')
    parser.add_argument('--viz_output', type=str, default=None,
                        help='可视化输出路径 (默认为 output 同名 .png)')
    
    args = parser.parse_args()
    
    # 加载点云
    if args.input_h5:
        all_points, filenames = load_from_h5(args.input_h5)
    elif args.input_dir:
        csv_dir = Path(args.input_dir)
        if (csv_dir / 'csv').exists():
            csv_dir = csv_dir / 'csv'
        all_points, filenames = load_from_csv_dir(csv_dir)
    else:
        # 默认路径
        default_h5 = './generated_nups/generated_nups.h5'
        if os.path.exists(default_h5):
            all_points, filenames = load_from_h5(default_h5)
        else:
            raise ValueError("请指定 --input_dir 或 --input_h5")
    
    print(f"加载了 {len(all_points)} 个点云，每个 {all_points.shape[1]} 点")
    
    # 拼接
    if args.mode == 'random':
        assembled, positions = place_random_no_overlap(
            all_points, 
            margin=args.margin,
            max_attempts=args.max_attempts,
            area_scale=args.area_scale,
            seed=args.seed
        )
    else:  # grid
        assembled, positions = place_grid(
            all_points,
            margin=args.margin,
            cols=args.cols
        )
    
    # 打印统计
    print(f"\n拼接后点云统计:")
    print(f"  总点数: {len(assembled)}")
    print(f"  X 范围: [{assembled[:, 0].min():.2f}, {assembled[:, 0].max():.2f}] nm")
    print(f"  Y 范围: [{assembled[:, 1].min():.2f}, {assembled[:, 1].max():.2f}] nm")
    print(f"  Z 范围: [{assembled[:, 2].min():.2f}, {assembled[:, 2].max():.2f}] nm")
    
    # 保存
    save_assembled(assembled, args.output, positions)
    
    # 可视化
    if args.visualize:
        if args.viz_output:
            viz_path = args.viz_output
        else:
            viz_path = str(Path(args.output).with_suffix('.png'))
        visualize_assembled(assembled, viz_path)
    
    print("\n拼接完成!")


if __name__ == '__main__':
    main()
