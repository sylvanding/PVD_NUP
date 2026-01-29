"""
NUP96 点云生成推理脚本

加载训练好的 PVD 模型，生成多个核孔复合物点云。

用法:
    python inference_nup.py --checkpoint /path/to/epoch_159.pth --num_samples 40 --output_dir ./generated_nups
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path
from datetime import datetime

# 确保项目路径在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.pvcnn_generation import PVCNN2Base
from model.pointcnn_generation import PointCNN2
from model.gravnetconv_generation import GravNet2


class PVCNN2(PVCNN2Base):
    """PVCNN 模型架构（与训练时一致）"""
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


def get_model_class(model_type):
    """Get model class based on model type string."""
    if model_type == 'pvcnn':
        return PVCNN2
    elif model_type == 'pointcnn':
        return PointCNN2
    elif model_type == 'gravnet':
        return GravNet2
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'pvcnn', 'pointcnn', or 'gravnet'")


class GaussianDiffusion:
    """高斯扩散过程（与 test_generation.py 一致）"""
    
    def __init__(self, betas, loss_type='mse', model_mean_type='eps', model_var_type='fixedsmall'):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):
        """
        计算 p(x_{t-1} | x_t) 的均值和方差（与 test_generation.py 一致）
        """
        model_output = denoise_fn(data, t)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model（与 test_generation.py 一致）
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data, t=t, clip_denoised=clip_denoised, return_pred_xstart=True
        )
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples（与 test_generation.py 一致）
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """
        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t


class InferenceModel:
    """用于推理的模型封装"""
    
    def __init__(self, checkpoint_path, model_type='pointcnn', device='cuda',
                 # Model configuration
                 embed_dim=64, attention=False, dropout=0.05, npoints=40,
                 # PVCNN specific
                 voxel_resolution_multiplier=2,
                 # PointCNN specific
                 pointcnn_width_multiplier=4.0, pointcnn_base_channels=128,
                 pointcnn_kernel_size=32, pointcnn_num_layers=4, pointcnn_downsample_ratio=0.8,
                 # GravNet specific
                 gravnet_width_multiplier=2.0, gravnet_base_channels=128,
                 gravnet_space_dimensions=6, gravnet_propagate_dimensions=16,
                 gravnet_k=32, gravnet_num_layers=8, gravnet_downsample_ratio=0.8):
        
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        
        # 加载 checkpoint
        print(f"加载 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 解析归一化参数
        self.use_minmax_norm = ckpt.get('use_minmax_norm', True)
        self.norm_min_vals = ckpt.get('norm_min_vals', None)
        self.norm_max_vals = ckpt.get('norm_max_vals', None)
        self.norm_mean = ckpt.get('norm_mean', np.array([0., 0., 0.]))
        self.norm_std = ckpt.get('norm_std', np.array([1., 1., 1.]))
        
        print(f"归一化模式: {'minmax' if self.use_minmax_norm else 'standard'}")
        if self.use_minmax_norm and self.norm_min_vals is not None:
            print(f"  min_vals: {self.norm_min_vals}")
            print(f"  max_vals: {self.norm_max_vals}")
        
        # 创建 betas (与训练时一致)
        betas = self._get_betas('warm0.1', 1e-5, 0.008, 1000)
        # 使用与 test_generation.py 一致的 GaussianDiffusion
        self.diffusion = GaussianDiffusion(
            betas, 
            loss_type='mse', 
            model_mean_type='eps', 
            model_var_type='fixedsmall'
        )
        
        # 根据 model_type 创建模型
        print(f"模型类型: {model_type}")
        ModelClass = get_model_class(model_type)
        
        if model_type == 'pointcnn':
            self.model = ModelClass(
                num_classes=3,  # nc=3 (xyz)
                embed_dim=embed_dim,
                use_att=attention,
                dropout=dropout,
                extra_feature_channels=0,
                width_multiplier=pointcnn_width_multiplier,
                base_channels=pointcnn_base_channels,
                kernel_size=pointcnn_kernel_size,
                num_layers=pointcnn_num_layers,
                downsample_ratio=pointcnn_downsample_ratio,
                npoints=npoints
            )
        elif model_type == 'gravnet':
            self.model = ModelClass(
                num_classes=3,
                embed_dim=embed_dim,
                use_att=attention,
                dropout=dropout,
                extra_feature_channels=0,
                width_multiplier=gravnet_width_multiplier,
                base_channels=gravnet_base_channels,
                space_dimensions=gravnet_space_dimensions,
                propagate_dimensions=gravnet_propagate_dimensions,
                k=gravnet_k,
                num_layers=gravnet_num_layers,
                downsample_ratio=gravnet_downsample_ratio,
                npoints=npoints
            )
        else:
            # PVCNN
            self.model = ModelClass(
                num_classes=3,
                embed_dim=embed_dim,
                use_att=attention,
                dropout=dropout,
                extra_feature_channels=0,
                voxel_resolution_multiplier=voxel_resolution_multiplier
            )
        
        # 加载模型权重
        state_dict = ckpt['model_state']
        # 处理 DataParallel 保存的模型（移除 model. 和 module. 前缀）
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            # 先移除 'model.' 前缀，再移除 'module.' 前缀
            # 原始 key 可能是 'model.module.xxx' 的格式
            if new_k.startswith('model.'):
                new_k = new_k[6:]
            if new_k.startswith('module.'):
                new_k = new_k[7:]
            new_state_dict[new_k] = v
        
        # Debug: 打印前几个 key 用于调试
        print(f"原始 state_dict keys (前3个): {list(state_dict.keys())[:3]}")
        print(f"处理后 state_dict keys (前3个): {list(new_state_dict.keys())[:3]}")
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        
        # 尝试加载 EMA 权重
        if 'ema_shadow' in ckpt:
            print("使用 EMA 权重进行推理")
            ema_shadow = ckpt['ema_shadow']
            for name, param in self.model.named_parameters():
                if name in ema_shadow:
                    param.data = ema_shadow[name].to(device)
        
        print("模型加载完成")
    
    def _get_betas(self, schedule_type, b_start, b_end, time_num):
        if schedule_type == 'linear':
            betas = np.linspace(b_start, b_end, time_num)
        elif schedule_type.startswith('warm'):
            ratio = float(schedule_type.replace('warm', ''))
            betas = b_end * np.ones(time_num, dtype=np.float64)
            warmup_time = int(time_num * ratio)
            betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
        else:
            raise NotImplementedError(schedule_type)
        return betas
    
    def _denoise(self, data, t):
        return self.model(data, t)
    
    @torch.no_grad()
    def generate(self, num_samples, npoints=40, batch_size=16, clip_denoised=False):
        """
        生成多个核孔复合物点云
        
        Args:
            num_samples: 要生成的样本数量
            npoints: 每个样本的点数
            batch_size: 每批生成的数量
            clip_denoised: 是否裁剪去噪后的值
            
        Returns:
            generated_points: (num_samples, npoints, 3) numpy array，已反归一化到原始尺度
        """
        all_points = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch = min(batch_size, num_samples - i * batch_size)
            print(f"生成批次 {i+1}/{num_batches}，样本数: {current_batch}")
            
            shape = (current_batch, 3, npoints)  # (B, C, N)
            gen_points = self.diffusion.p_sample_loop(
                self._denoise, shape, self.device, clip_denoised=clip_denoised
            )
            
            # (B, C, N) -> (B, N, C)
            gen_points = gen_points.permute(0, 2, 1).cpu().numpy()
            all_points.append(gen_points)
        
        generated_points = np.concatenate(all_points, axis=0)[:num_samples]
        
        # 反归一化
        denormalized = self.denormalize(generated_points)
        
        return generated_points, denormalized
    
    def denormalize(self, normalized_points):
        """
        反归一化点云到原始尺度
        
        Args:
            normalized_points: (N_samples, N_points, 3) 归一化的点云
            
        Returns:
            denormalized: 原始尺度的点云
        """
        if self.use_minmax_norm and self.norm_min_vals is not None:
            # minmax 反归一化: 从 [-1, 1] 到原始尺度
            min_vals = np.asarray(self.norm_min_vals).flatten()
            max_vals = np.asarray(self.norm_max_vals).flatten()
            range_vals = max_vals - min_vals
            denormalized = (normalized_points + 1) / 2 * range_vals + min_vals
        else:
            # 标准归一化反变换
            mean = np.asarray(self.norm_mean).flatten()
            std = np.asarray(self.norm_std).flatten()
            denormalized = normalized_points * std + mean
        
        return denormalized


def save_results(generated_points, denormalized_points, output_dir, save_csv=True, save_h5=True):
    """
    保存生成的点云
    
    Args:
        generated_points: (N, M, 3) 归一化的点云
        denormalized_points: (N, M, 3) 反归一化的点云
        output_dir: 输出目录
        save_csv: 是否保存为 CSV
        save_h5: 是否保存为 H5
    """
    import pandas as pd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(exist_ok=True)
    
    # 保存每个样本为 CSV
    if save_csv:
        print(f"保存 CSV 文件到: {csv_dir}")
        for i in range(len(denormalized_points)):
            points = denormalized_points[i]
            df = pd.DataFrame(points, columns=['x [nm]', 'y [nm]', 'z [nm]'])
            df.to_csv(csv_dir / f'nup_{i:04d}.csv', index=False)
    
    # 保存为 H5 文件
    if save_h5:
        h5_path = output_dir / 'generated_nups.h5'
        print(f"保存 H5 文件到: {h5_path}")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('points_normalized', data=generated_points)
            f.create_dataset('points_denormalized', data=denormalized_points)
            f.attrs['num_samples'] = len(generated_points)
            f.attrs['num_points'] = generated_points.shape[1]
            f.attrs['generated_at'] = datetime.now().isoformat()
    
    print(f"生成完成! 共 {len(generated_points)} 个核孔复合物")
    return output_dir


def visualize_samples(denormalized_points, output_dir, num_viz=9):
    """可视化部分生成的点云"""
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    num_viz = min(num_viz, len(denormalized_points))
    cols = min(3, num_viz)
    rows = (num_viz + cols - 1) // cols if cols > 0 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i in range(num_viz):
        ax = axes[i]
        points = denormalized_points[i]
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=5, alpha=0.8)
        ax.set_xlabel('X [nm]')
        ax.set_ylabel('Y [nm]')
        ax.set_title(f'NUP Sample {i}')
        ax.set_aspect('equal')
    
    # 隐藏空白子图
    for i in range(num_viz, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'samples_overview.png', dpi=150)
    plt.close()
    
    print(f"可视化保存到: {viz_dir / 'samples_overview.png'}")


def main():
    parser = argparse.ArgumentParser(description='NUP96 点云生成推理')
    parser.add_argument('--checkpoint', type=str, 
                        default='/data0/djx/pvd_nup/output/20260109_044249_clustering/epoch_159.pth',
                        help='模型 checkpoint 路径')
    parser.add_argument('--num_samples', type=int, default=40,
                        help='要生成的核孔复合物数量')
    parser.add_argument('--npoints', type=int, default=40,
                        help='每个核孔复合物的点数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='每批生成的数量')
    parser.add_argument('--output_dir', type=str, default='/data0/djx/pvd_nup/output/20260109_044249_clustering/generated_nups',
                        help='输出目录')
    parser.add_argument('--gpu', type=int, default=7,
                        help='使用的 GPU 编号')
    parser.add_argument('--save_csv', action='store_true', default=True,
                        help='保存为 CSV 文件')
    parser.add_argument('--save_h5', action='store_true', default=True,
                        help='保存为 H5 文件')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='可视化生成的样本')
    parser.add_argument('--clip_denoised', action='store_true', default=False,
                        help='是否裁剪去噪值')
    
    # Model type and architecture params
    parser.add_argument('--model_type', type=str, default='pvcnn', 
                        choices=['pvcnn', 'pointcnn', 'gravnet'],
                        help='模型类型')
    parser.add_argument('--attention', type=bool, default=False,
                        help='是否使用 attention')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout 比率')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding 维度')
    
    # PVCNN specific
    parser.add_argument('--voxel_resolution_multiplier', type=int, default=2,
                        help='PVCNN voxel resolution multiplier')
    
    # PointCNN specific
    parser.add_argument('--pointcnn_width_multiplier', type=float, default=4.0,
                        help='PointCNN width multiplier')
    parser.add_argument('--pointcnn_base_channels', type=int, default=128,
                        help='PointCNN base channels')
    parser.add_argument('--pointcnn_kernel_size', type=int, default=32,
                        help='PointCNN kernel size')
    parser.add_argument('--pointcnn_num_layers', type=int, default=4,
                        help='PointCNN number of layers')
    parser.add_argument('--pointcnn_downsample_ratio', type=float, default=0.8,
                        help='PointCNN downsample ratio')
    
    # GravNet specific
    parser.add_argument('--gravnet_width_multiplier', type=float, default=2.0,
                        help='GravNet width multiplier')
    parser.add_argument('--gravnet_base_channels', type=int, default=128,
                        help='GravNet base channels')
    parser.add_argument('--gravnet_space_dimensions', type=int, default=6,
                        help='GravNet space dimensions')
    parser.add_argument('--gravnet_propagate_dimensions', type=int, default=16,
                        help='GravNet propagate dimensions')
    parser.add_argument('--gravnet_k', type=int, default=32,
                        help='GravNet number of neighbors')
    parser.add_argument('--gravnet_num_layers', type=int, default=8,
                        help='GravNet number of layers')
    parser.add_argument('--gravnet_downsample_ratio', type=float, default=0.8,
                        help='GravNet downsample ratio')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
        print("警告: CUDA 不可用，使用 CPU")
    
    print(f"使用设备: {device}")
    print(f"生成 {args.num_samples} 个核孔复合物，每个 {args.npoints} 点")
    
    # 创建模型并生成
    model = InferenceModel(
        args.checkpoint, 
        model_type=args.model_type,
        device=device,
        embed_dim=args.embed_dim,
        attention=args.attention,
        dropout=args.dropout,
        npoints=args.npoints,
        # PVCNN
        voxel_resolution_multiplier=args.voxel_resolution_multiplier,
        # PointCNN
        pointcnn_width_multiplier=args.pointcnn_width_multiplier,
        pointcnn_base_channels=args.pointcnn_base_channels,
        pointcnn_kernel_size=args.pointcnn_kernel_size,
        pointcnn_num_layers=args.pointcnn_num_layers,
        pointcnn_downsample_ratio=args.pointcnn_downsample_ratio,
        # GravNet
        gravnet_width_multiplier=args.gravnet_width_multiplier,
        gravnet_base_channels=args.gravnet_base_channels,
        gravnet_space_dimensions=args.gravnet_space_dimensions,
        gravnet_propagate_dimensions=args.gravnet_propagate_dimensions,
        gravnet_k=args.gravnet_k,
        gravnet_num_layers=args.gravnet_num_layers,
        gravnet_downsample_ratio=args.gravnet_downsample_ratio
    )
    
    generated, denormalized = model.generate(
        args.num_samples, 
        npoints=args.npoints,
        batch_size=args.batch_size,
        clip_denoised=args.clip_denoised
    )
    
    # 打印统计信息
    print(f"\n生成点云统计:")
    print(f"  归一化范围: [{generated.min():.4f}, {generated.max():.4f}]")
    print(f"  反归一化 X 范围: [{denormalized[:,:,0].min():.2f}, {denormalized[:,:,0].max():.2f}] nm")
    print(f"  反归一化 Y 范围: [{denormalized[:,:,1].min():.2f}, {denormalized[:,:,1].max():.2f}] nm")
    print(f"  反归一化 Z 范围: [{denormalized[:,:,2].min():.2f}, {denormalized[:,:,2].max():.2f}] nm")
    
    # 保存结果
    output_dir = save_results(generated, denormalized, args.output_dir, 
                              save_csv=args.save_csv, save_h5=args.save_h5)
    
    # 可视化
    if args.visualize:
        visualize_samples(denormalized, output_dir)
    
    print(f"\n全部完成! 结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
