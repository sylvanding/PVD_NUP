# NPC 点云数据处理脚本

本目录包含用于处理 NPC (Nuclear Pore Complex, 核孔复合物) 点云数据的脚本，将原始 CSV 数据转换为可供模型训练的 H5 格式。

## 推荐工作流程

```bash
# 1. 首先分析点数分布，决定目标点数
python 0-npc-analysis.py --subfolder all

# 2. 根据分析结果运行处理流程
python run_all.py --subfolder rotated_density_0_9 --target-points <根据分析结果设定>
```

## 数据组织

输入数据位于 `outputs/npc_batch/`，按旋转类型和标记密度组织：
```
outputs/npc_batch/
├── rotated_density_0_9/      # 随机旋转 + 标记密度0.9
│   ├── npc_0001.csv
│   ├── npc_0002.csv
│   └── ... (共1024个文件)
├── rotated_density_0_7/      # 随机旋转 + 标记密度0.7
├── rotated_density_0_5/      # 随机旋转 + 标记密度0.5
├── fixed_density_0_9/        # 固定方向 + 标记密度0.9
├── fixed_density_0_7/        # 固定方向 + 标记密度0.7
└── fixed_density_0_5/        # 固定方向 + 标记密度0.5
```

输出数据保存到 `outputs/npc_batch/processed/`，同样按子文件夹组织：
```
outputs/npc_batch/processed/
├── rotated_density_0_9/
│   ├── csv/                          # 处理后的CSV文件
│   ├── npc_rotated_density_0_9.h5    # H5格式训练数据
│   └── extraction_stats.json
├── rotated_density_0_7/
│   └── ...
└── ...
```

## 脚本说明

### 0-npc-analysis.py
分析 NPC 点云数据的点数分布，帮助决定目标点数：
- 统计平均点数、标准差、分位数等
- 生成点数分布直方图
- 给出目标点数建议

```bash
# 分析所有子文件夹
python 0-npc-analysis.py --subfolder all

# 分析单个子文件夹
python 0-npc-analysis.py --subfolder rotated_density_0_9

# 不生成直方图
python 0-npc-analysis.py --subfolder all --no-plot
```

### 1-npc-samples.py
处理 NPC 点云样本：
- 过滤点数不足的样本
- 对点数过多的样本使用最远点采样 (FPS)
- 归一化到坐标原点（平移）

```bash
python 1-npc-samples.py --subfolder rotated_density_0_9 --target-points 2048 --min-points 100
```

### 3-npc-csv2h5.py
将 CSV 点云转换为 H5 格式：
- 使用最值归一化（每个样本独立归一化到 [-1, 1]）
- 保存归一化参数用于反归一化

```bash
python 3-npc-csv2h5.py --subfolder rotated_density_0_9 --target-points 2048 --center
```

### run_all.py
一次运行上述两个脚本的流水线。

```bash
# 处理单个子文件夹
python run_all.py --subfolder rotated_density_0_9 --target-points 2048

# 处理所有子文件夹
python run_all.py --subfolder all --target-points 2048
```

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--subfolder` | all | 子文件夹 (rotated_density_0_9/0_7/0_5, fixed_density_0_9/0_7/0_5, all) |
| `--target-points` | 2048 | 每个样本的目标点数 |
| `--min-points` | 100 | 最小点数阈值（小于此值的样本被丢弃） |
| `--use-fps` | True | 使用最远点采样 |
| `--center` | True | 归一化到 [-1, 1]（否则 [0, 1]） |
| `--seed` | 42 | 随机种子 |
| `--input-dir` | outputs/npc_batch | 输入数据根目录 |
| `--output-dir` | outputs/npc_batch/processed | 输出数据根目录 |

## 使用示例

```bash
# 在项目根目录运行
cd /home/djx/repos/PVD_NUP

# 处理单个子文件夹（随机旋转 + 密度0.9）
python data_prepare/npc/run_all.py --subfolder rotated_density_0_9 --target-points 2048

# 处理固定方向的所有密度
python data_prepare/npc/run_all.py --subfolder fixed_density_0_9
python data_prepare/npc/run_all.py --subfolder fixed_density_0_7
python data_prepare/npc/run_all.py --subfolder fixed_density_0_5

# 批量处理所有子文件夹
python data_prepare/npc/run_all.py --subfolder all --target-points 2048
```

## H5 文件格式

生成的 H5 文件包含：
- `points`: 归一化后的点云 (B, N, 3)，float32
- `min_vals`: 每个样本的最小值 (B, 3)，用于反归一化
- `max_vals`: 每个样本的最大值 (B, 3)，用于反归一化
- `filenames`: 原始文件名列表

元数据 (attrs):
- `num_samples`: 样本数量
- `num_points`: 每个样本的点数
- `normalize_method`: 'minmax'
- `normalize_center`: True/False
- `normalize_per_shape`: True
