# NUP96 点云数据预处理流水线

本目录包含NUP96核孔复合物大规模点云数据的预处理脚本，用于PVCNN生成式模型训练。

## 数据概述

- **数据来源**: `/home/djx/data/nup96-large/0-origin-csv/`
- **数据格式**: CSV文件，包含xnm, ynm, znm坐标列
- **数据规模**: 16个大规模点云文件，每个包含4万-19万个点
- **坐标精度**: 小数点后2位 (nm级别)

## 数据处理流水线

数据存储在 `/home/djx/data/nup96-large/` 目录下：

```
/home/djx/data/nup96-large/
├── 0-origin-csv/           # 原始CSV点云文件
├── 1-clean-csv/            # 清洗后的CSV（坐标移动到原点）
├── 2-clean-viz-png/        # 清洗后点云的2D可视化
├── 3-random-crop-csv/      # 随机裁剪的小块样本（测试用）
├── 4-random-crop-viz-png/  # 裁剪样本的2D可视化
├── 5-random-crop-batch-csv/# 批量生成的点云块
├── 6-pc-blocks.h5          # 归一化后的H5格式数据
├── 7-h5tocsv-csv/          # H5转回CSV的验证文件
└── 8-h5tocsv-viz-png/      # H5转换验证的可视化
```

## 处理脚本说明

### 1. `1-clean.py` - 数据清洗

```bash
python data_prepare/1-clean.py
```

功能:
- 读取原始CSV文件
- 自动检测坐标列（xnm, ynm, znm等变体）
- 将点云左下角移动到坐标原点
- 输出统计信息（坐标范围、均值、点数）
- 标准化表头为 `x [nm]`, `y [nm]`, `z [nm]`

### 2. `2-viz.py` - 清洗数据可视化

```bash
python data_prepare/2-viz.py
```

功能:
- 生成大规模点云的2D俯视图
- 颜色按z轴高度着色

### 3. `3-random-crop.py` - 智能随机裁剪

```bash
python data_prepare/3-random-crop.py
```

功能:
- 从大规模点云中智能随机裁剪小块
- **智能调整裁剪比例**以达到目标点数
- **数据增强**: 裁剪前可选旋转、翻转等增强
- 从点云中心附近采样

#### 智能裁剪策略

| 条件 | 动作 |
|------|------|
| ratio > 1.5 (超出太多) | 缩小裁剪框 |
| 1.0 < ratio <= 1.5 | 使用FPS降采样 |
| 0.8 <= ratio < 1.0 | 增大裁剪框 |
| ratio < 0.8 (远低于目标) | 重新选择区域 |

#### 参数配置

```python
NUM_SAMPLES = 10           # 样本数（测试用）
TARGET_POINTS = 2048       # 目标点数
INITIAL_CROP_RATIO = 0.1   # 初始裁剪比例
TOLERANCE_HIGH = 1.5       # 高容忍阈值
TOLERANCE_LOW = 0.8        # 低容忍阈值

# 数据增强配置
AUGMENTATION_CONFIG = {
    'enable_rotation_z': True,    # 绕Z轴随机旋转
    'enable_flip': True,          # 随机翻转
    'enable_rotation_xy': False,  # 绕XY轴旋转
    'enable_jitter': False,       # 噪声抖动
    'enable_scale': False,        # 随机缩放
}

# z轴离群点过滤配置
ENABLE_Z_FILTER = True     # 启用z轴离群点过滤
Z_FILTER_METHOD = 'iqr'    # 过滤方法
Z_FILTER_PARAMS = {'iqr_k': 1.5}  # 过滤参数
```

#### z轴离群点过滤

裁剪后会自动过滤z轴上的离群点，防止生成的样本包含噪声点。

支持的过滤方法：

| 方法 | 说明 | 参数 |
|------|------|------|
| `zscore` | Z-score统计方法 | `zscore_threshold`: 阈值，默认3.0 |
| `iqr` | 四分位距方法（推荐） | `iqr_k`: IQR倍数，默认1.5 |
| `percentile` | 百分位数方法 | `percentile_lower`, `percentile_upper`: 上下百分位 |
| `statistical` | Open3D统计方法 | `statistical_neighbors`, `statistical_std_ratio` |

过滤后若点数不足，会自动增大裁剪框重新裁剪。

### 4. `4-viz.py` - 裁剪样本可视化

```bash
python data_prepare/4-viz.py
```

### 5. `5-random-crop-batch.py` - 批量生成

```bash
python data_prepare/5-random-crop-batch.py
```

功能:
- 批量生成大量点云块（默认1024个）
- 使用智能裁剪和数据增强

### 6. `6-csv2h5.py` - 归一化并转H5

```bash
python data_prepare/6-csv2h5.py
```

### 7. `7-h5tocsv.py` - H5转CSV验证

```bash
python data_prepare/7-h5tocsv.py
```

### 8. `8-viz.py` - 验证可视化

```bash
python data_prepare/8-viz.py
```

## 工具函数

位于 `utils/pc_utils.py`:

### 数据增强

```python
from utils.pc_utils import (
    rotate_pointcloud_z,          # 绕Z轴旋转
    rotate_pointcloud_xy,         # 绕XY轴小角度旋转
    flip_pointcloud,              # 翻转点云
    jitter_pointcloud,            # 添加噪声抖动
    scale_pointcloud,             # 缩放点云
    augment_pointcloud,           # 综合数据增强
)
```

### 离群点过滤

```python
from utils.pc_utils import (
    filter_outliers_zscore,       # Z-score方法过滤
    filter_outliers_iqr,          # IQR四分位距方法过滤
    filter_outliers_percentile,   # 百分位数方法过滤
    filter_outliers_statistical,  # Open3D统计方法过滤
    filter_z_outliers,            # 统一接口
)
```

### 智能裁剪

```python
from utils.pc_utils import (
    adaptive_crop_pointcloud,         # 自适应裁剪（含离群点过滤）
    smart_crop_with_augmentation,     # 智能裁剪+增强+过滤
)
```

## 快速开始

```bash
cd /home/djx/repos/PVD_NUP

python data_prepare/1-clean.py
python data_prepare/2-viz.py
python data_prepare/3-random-crop.py
python data_prepare/4-viz.py
python data_prepare/5-random-crop-batch.py
python data_prepare/6-csv2h5.py
python data_prepare/7-h5tocsv.py
python data_prepare/8-viz.py
```

## 训练数据加载

```python
from datasets.nup96_data_pc import NUP96PointClouds

# H5模式
dataset = NUP96PointClouds(
    mode='h5',
    h5_path='/home/djx/data/nup96-large/6-pc-blocks.h5',
    num_points=2048
)

# 实时模式
dataset = NUP96PointClouds(
    mode='realtime',
    csv_dir='/home/djx/data/nup96-large/1-clean-csv',
    num_samples=1024,
    num_points=2048
)
```
