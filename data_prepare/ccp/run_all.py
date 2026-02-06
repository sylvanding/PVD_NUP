#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py: 一次运行CCP数据处理的全部三个脚本

功能:
1. 运行 1-ccp-samples.py 处理点云样本
2. 运行 2-ccp-samples-viz.py 可视化样本
3. 运行 3-ccp-csv2h5.py 转换为H5格式

支持:
- 选择单个发育时期处理
- 选择所有发育时期批量处理
- 跳过可视化步骤（加速处理）
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# CCP发育时期列表
STAGES = ['early', 'mid_early', 'middle', 'mid_late', 'late', 'mature']


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def run_script(script_path: str, args: list, logger: logging.Logger) -> bool:
    """
    运行Python脚本
    
    Args:
        script_path: 脚本路径
        args: 命令行参数列表
        logger: 日志记录器
        
    Returns:
        成功返回True，失败返回False
    """
    cmd = [sys.executable, script_path] + args
    logger.info(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"脚本执行失败: {e}")
        return False


def process_stage(stage: str,
                  script_dir: str,
                  input_dir: str,
                  output_dir: str,
                  target_points: int,
                  min_points: int,
                  use_fps: bool,
                  skip_viz: bool,
                  max_viz_samples: int,
                  center: bool,
                  seed: int,
                  logger: logging.Logger) -> bool:
    """
    处理单个发育时期
    
    Args:
        stage: 发育时期名称
        script_dir: 脚本目录
        input_dir: 输入数据根目录
        output_dir: 输出数据根目录
        target_points: 目标点数
        min_points: 最小点数阈值
        use_fps: 是否使用最远点采样
        skip_viz: 是否跳过可视化
        max_viz_samples: 最大可视化样本数
        center: 是否居中归一化
        seed: 随机种子
        logger: 日志记录器
        
    Returns:
        成功返回True
    """
    logger.info("=" * 70)
    logger.info(f"开始处理发育时期: {stage}")
    logger.info("=" * 70)
    
    # Step 1: 处理点云样本
    logger.info(f"\n[Step 1/3] 处理点云样本...")
    script1 = os.path.join(script_dir, "1-ccp-samples.py")
    args1 = [
        "--stage", stage,
        "--target-points", str(target_points),
        "--min-points", str(min_points),
        "--seed", str(seed),
        "--input-dir", input_dir,
        "--output-dir", output_dir
    ]
    if use_fps:
        args1.append("--use-fps")
    else:
        args1.append("--no-fps")
    
    if not run_script(script1, args1, logger):
        logger.error(f"[{stage}] Step 1 失败!")
        return False
    
    # Step 2: 可视化（可选）
    if not skip_viz:
        logger.info(f"\n[Step 2/3] 可视化样本...")
        script2 = os.path.join(script_dir, "2-ccp-samples-viz.py")
        args2 = [
            "--stage", stage,
            "--max-samples", str(max_viz_samples),
            "--output-dir", output_dir
        ]
        
        if not run_script(script2, args2, logger):
            logger.error(f"[{stage}] Step 2 失败!")
            return False
    else:
        logger.info(f"\n[Step 2/3] 跳过可视化...")
    
    # Step 3: 转换为H5
    logger.info(f"\n[Step 3/3] 转换为H5格式...")
    script3 = os.path.join(script_dir, "3-ccp-csv2h5.py")
    args3 = [
        "--stage", stage,
        "--target-points", str(target_points),
        "--output-dir", output_dir
    ]
    if center:
        args3.append("--center")
    else:
        args3.append("--no-center")
    
    if not run_script(script3, args3, logger):
        logger.error(f"[{stage}] Step 3 失败!")
        return False
    
    logger.info(f"\n[{stage}] 处理完成!")
    return True


def main():
    parser = argparse.ArgumentParser(description='一次运行CCP数据处理的全部脚本')
    parser.add_argument('--stage', type=str, default=None,
                       choices=STAGES + ['all'],
                       help='发育时期，设为"all"处理所有时期 (default: all)')
    parser.add_argument('--target-points', type=int, default=2048,
                       help='每个样本的目标点数 (default: 2048)')
    parser.add_argument('--min-points', type=int, default=100,
                       help='最小点数阈值 (default: 100)')
    parser.add_argument('--use-fps', action='store_true', default=True,
                       help='使用最远点采样 (default: True)')
    parser.add_argument('--no-fps', action='store_false', dest='use_fps',
                       help='使用随机采样')
    parser.add_argument('--skip-viz', action='store_true', default=False,
                       help='跳过可视化步骤 (default: False)')
    parser.add_argument('--max-viz-samples', type=int, default=100,
                       help='最大可视化样本数 (default: 100)')
    parser.add_argument('--center', action='store_true', default=True,
                       help='归一化到 [-1, 1] (default: True)')
    parser.add_argument('--no-center', action='store_false', dest='center',
                       help='归一化到 [0, 1]')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--input-dir', type=str,
                       default='/home/djx/data0/pvd_nup/ccp_stages',
                       help='输入数据根目录')
    parser.add_argument('--output-dir', type=str,
                       default='/home/djx/data0/pvd_nup/ccp_stages/outputs',
                       help='输出数据根目录')
    
    args = parser.parse_args()
    
    # 确定要处理的时期
    if args.stage is None or args.stage == 'all':
        stages_to_process = STAGES
    else:
        stages_to_process = [args.stage]
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("CCP数据处理流水线")
    logger.info("=" * 70)
    logger.info(f"待处理时期: {stages_to_process}")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"目标点数: {args.target_points}")
    logger.info(f"最小点数阈值: {args.min_points}")
    logger.info(f"采样方式: {'FPS' if args.use_fps else '随机'}")
    logger.info(f"跳过可视化: {args.skip_viz}")
    logger.info(f"归一化范围: {'[-1, 1]' if args.center else '[0, 1]'}")
    
    # 处理每个时期
    results = {}
    for stage in stages_to_process:
        success = process_stage(
            stage=stage,
            script_dir=str(script_dir),
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_points=args.target_points,
            min_points=args.min_points,
            use_fps=args.use_fps,
            skip_viz=args.skip_viz,
            max_viz_samples=args.max_viz_samples,
            center=args.center,
            seed=args.seed,
            logger=logger
        )
        results[stage] = success
    
    # 输出汇总
    logger.info("\n" + "=" * 70)
    logger.info("处理汇总")
    logger.info("=" * 70)
    for stage, success in results.items():
        status = "成功 ✓" if success else "失败 ✗"
        logger.info(f"  {stage}: {status}")
    
    # 检查是否全部成功
    if all(results.values()):
        logger.info("\n所有时期处理完成!")
    else:
        failed = [s for s, r in results.items() if not r]
        logger.error(f"\n以下时期处理失败: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
