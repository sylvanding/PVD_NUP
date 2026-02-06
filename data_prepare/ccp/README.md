# CCP 点云数据处理脚本

本目录包含用于处理 CCP (Clathrin-Coated Pit) 点云数据的脚本，将原始 CSV 数据转换为可供模型训练的 H5 格式。

## 推荐工作流程

```bash
# 1. 首先分析点数分布，决定目标点数
python 0-ccp-analysis.py --stage all

# 2. 根据分析结果运行处理流程
python run_all.py --stage early --target-points <根据分析结果设定>
```

## 数据组织

输入数据位于 `/home/djx/data0/pvd_nup/ccp_stages/`，按发育时期组织：
- `early/` - 早期阶段
- `mid_early/` - 中早期阶段
- `middle/` - 中期阶段
- `mid_late/` - 中晚期阶段
- `late/` - 晚期阶段
- `mature/` - 成熟阶段

输出数据保存到 `/home/djx/data0/pvd_nup/ccp_stages/outputs/`，同样按时期组织：
```
outputs/
├── early/
│   ├── csv/           # 处理后的CSV文件
│   ├── viz/           # 可视化图像
│   ├── ccp_early.h5   # H5格式训练数据
│   └── extraction_stats.json
├── mid_early/
│   └── ...
└── ...
```

## 脚本说明

### 0-ccp-analysis.py
分析 CCP 点云数据的点数分布，帮助决定目标点数：
- 统计平均点数、标准差、分位数等
- 生成点数分布直方图
- 给出目标点数建议

```bash
# 分析所有时期
python 0-ccp-analysis.py --stage all

# 分析单个时期
python 0-ccp-analysis.py --stage early

# 不生成直方图
python 0-ccp-analysis.py --stage all --no-plot
```

### 1-ccp-samples.py
处理 CCP 点云样本：
- 过滤点数不足的样本
- 对点数过多的样本使用最远点采样 (FPS)
- 归一化到坐标原点（平移）

```bash
python 1-ccp-samples.py --stage early --target-points 2048 --min-points 100
```

### 2-ccp-samples-viz.py
可视化处理后的 CCP 样本：
- 生成 2D 俯视图
- 颜色按 z 轴高度着色

```bash
python 2-ccp-samples-viz.py --stage early --max-samples 100
```

### 3-ccp-csv2h5.py
将 CSV 点云转换为 H5 格式：
- 使用最值归一化（每个样本独立归一化到 [-1, 1]）
- 保存归一化参数用于反归一化

```bash
python 3-ccp-csv2h5.py --stage early --target-points 2048 --center
```

### run_all.py
一次运行上述三个脚本的流水线。

```bash
# 处理单个时期
python run_all.py --stage early --target-points 2048

# 处理所有时期
python run_all.py --stage all --target-points 2048

# 跳过可视化（加速处理）
python run_all.py --stage all --skip-viz
```

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage` | all | 发育时期 (early/mid_early/middle/mid_late/late/mature/all) |
| `--target-points` | 2048 | 每个样本的目标点数 |
| `--min-points` | 100 | 最小点数阈值（小于此值的样本被丢弃） |
| `--use-fps` | True | 使用最远点采样 |
| `--skip-viz` | False | 跳过可视化步骤 |
| `--center` | True | 归一化到 [-1, 1]（否则 [0, 1]） |
| `--seed` | 42 | 随机种子 |

## 使用示例

```bash
# 在项目根目录运行
cd /home/djx/repos/PVD_NUP

# 处理 early 时期，目标点数 512
python data_prepare/ccp/run_all.py --stage early --target-points 512

# 批量处理所有时期，跳过可视化
python data_prepare/ccp/run_all.py --stage all --target-points 2048 --skip-viz
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
