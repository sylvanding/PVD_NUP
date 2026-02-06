"""
Point-Voxel Diffusion Training for NUP96 Point Cloud Generation
Modified to support NUP96 H5 dataset and TensorBoard visualization.
"""

# IMPORTANT: Set CUDA_VISIBLE_DEVICES before importing torch
# This must be done before any torch import to take effect
import sys
import os

def _set_cuda_visible_devices():
    """Parse --gpus argument and set CUDA_VISIBLE_DEVICES before torch import."""
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            gpus = sys.argv[i + 1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            print(f'[Pre-import] CUDA_VISIBLE_DEVICES set to: {gpus}')
            return
        elif arg.startswith('--gpus='):
            gpus = arg.split('=', 1)[1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            print(f'[Pre-import] CUDA_VISIBLE_DEVICES set to: {gpus}')
            return

_set_cuda_visible_devices()

import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import argparse
from pathlib import Path

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from utils.file_utils import *
from utils.visualize import *
from datetime import datetime  # 必须在 file_utils import * 之后，避免被覆盖
from utils.pc_utils import visualize_pointcloud_2d, denormalize_pointcloud
from model.pvcnn_generation import PVCNN2Base
from model.pointcnn_generation import PointCNN2
from model.gravnetconv_generation import GravNet2
import torch.distributed as dist

# Import dataset classes
from datasets.nup96_data_pc import (
    NUP96PointClouds, get_nup96_datasets,
    NUP96ClusteredPointClouds, get_clustered_nup96_datasets,
    CCPClusteredPointClouds, get_clustered_ccp_datasets, CCP_STAGES,
    NPCClusteredPointClouds, get_clustered_npc_datasets, NPC_SUBFOLDERS
)


'''
some utils
'''

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters() if p.grad is not None))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

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
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
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



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


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


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

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
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
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
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == torch.Size([B, D, N])
            assert eps_recon.shape == data_start.shape
            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class PVCNN2(PVCNN2Base):
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

    def __init__(self, num_classes, embed_dim, use_att,dropout, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
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


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        # Select model based on model_type argument
        ModelClass = get_model_class(args.model_type)
        
        if args.model_type == 'pointcnn':
            # PointCNN with configurable capacity
            self.model = ModelClass(
                num_classes=args.nc, 
                embed_dim=args.embed_dim, 
                use_att=args.attention,
                dropout=args.dropout, 
                extra_feature_channels=0,
                width_multiplier=args.pointcnn_width_multiplier,
                base_channels=args.pointcnn_base_channels,
                kernel_size=args.pointcnn_kernel_size,
                num_layers=args.pointcnn_num_layers,
                downsample_ratio=args.pointcnn_downsample_ratio,
                npoints=args.npoints  # Pass npoints for kernel size calculation
            )
        elif args.model_type == 'gravnet':
            # GravNet with configurable capacity
            self.model = ModelClass(
                num_classes=args.nc,
                embed_dim=args.embed_dim,
                use_att=args.attention,
                dropout=args.dropout,
                extra_feature_channels=0,
                width_multiplier=args.gravnet_width_multiplier,
                base_channels=args.gravnet_base_channels,
                space_dimensions=args.gravnet_space_dimensions,
                propagate_dimensions=args.gravnet_propagate_dimensions,
                k=args.gravnet_k,
                num_layers=args.gravnet_num_layers,
                downsample_ratio=args.gravnet_downsample_ratio,
                npoints=args.npoints
            )
        else:
            # PVCNN
            self.model = ModelClass(
                num_classes=args.nc, 
                embed_dim=args.embed_dim, 
                use_att=args.attention,
                dropout=args.dropout, 
                extra_feature_channels=0,
                voxel_resolution_multiplier=args.voxel_resolution_multiplier
            )

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def get_dataset(opt):
    """
    Get dataset based on dataset_type and dataset_mode
    
    Dataset types:
        - 'nup': NUP96 核孔复合物数据集
        - 'ccp': CCP (Clathrin-Coated Pit) 数据集
        - 'npc': NPC (Nuclear Pore Complex) 模拟数据集
    
    Modes (for NUP):
        - 'h5': 从预处理的 H5 文件加载（全局归一化）
        - 'realtime': 实时从 CSV 文件裁剪
        - 'clustered': 从聚类核孔 H5 文件加载（minmax 归一化）
    
    For CCP:
        - 总是使用 clustered 模式（minmax 归一化）
        - 需要指定 ccp_stage
    
    For NPC:
        - 总是使用 clustered 模式（minmax 归一化）
        - 需要指定 npc_subfolder
    """
    if opt.dataset_type == 'npc':
        # NPC 数据集
        train_dataset, val_dataset = get_clustered_npc_datasets(
            data_root=opt.npc_dataroot,
            subfolder=opt.npc_subfolder,
            num_points=opt.npoints
        )
    elif opt.dataset_type == 'ccp':
        # CCP 数据集
        train_dataset, val_dataset = get_clustered_ccp_datasets(
            data_root=opt.ccp_dataroot,
            stage=opt.ccp_stage,
            num_points=opt.npoints
        )
    elif opt.dataset_mode == 'clustered':
        # NUP96 clustered 模式
        train_dataset, val_dataset = get_clustered_nup96_datasets(
            data_root=opt.dataroot,
            num_points=opt.npoints
        )
    else:
        # NUP96 其他模式 (h5, realtime)
        train_dataset, val_dataset = get_nup96_datasets(
            data_root=opt.dataroot,
            mode=opt.dataset_mode,
            num_points=opt.npoints,
            num_samples=opt.num_samples
        )
    return train_dataset, val_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def save_generated_samples(gen_points, mean, std, epoch, output_dir, sample_type='eval',
                          normalize_mode='standard', min_vals=None, max_vals=None):
    """
    Save generated point cloud samples as CSV and visualize as 2D images.
    
    Args:
        gen_points: Generated points (B, 3, N) normalized
        mean: Normalization mean (用于 standard 模式)
        std: Normalization std (用于 standard 模式)
        epoch: Current epoch
        output_dir: Output directory
        sample_type: 'eval' or 'traj'
        normalize_mode: 'standard' 或 'minmax'
        min_vals: minmax 模式下的平均最小值 (3,)
        max_vals: minmax 模式下的平均最大值 (3,)
    """
    import pandas as pd
    
    # Create output directories
    img_dir = Path(output_dir) / 'generated_images'
    pts_dir = Path(output_dir) / 'generated_points'
    img_dir.mkdir(parents=True, exist_ok=True)
    pts_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and transpose to (B, N, 3)
    gen_points_np = gen_points.cpu().numpy().transpose(0, 2, 1)
    
    for i in range(min(len(gen_points_np), 16)):  # Save up to 16 samples
        points = gen_points_np[i]  # (N, 3) normalized
        
        # Denormalize to original scale (nm)
        if normalize_mode == 'minmax' and min_vals is not None and max_vals is not None:
            # minmax 反归一化: 从 [-1, 1] 到原始尺度
            min_flat = np.asarray(min_vals).flatten()
            max_flat = np.asarray(max_vals).flatten()
            range_vals = max_flat - min_flat
            points_denorm = (points + 1) / 2 * range_vals + min_flat
        else:
            # 标准归一化反变换
            mean_flat = np.asarray(mean).flatten()
            std_flat = np.asarray(std).flatten()
            points_denorm = denormalize_pointcloud(points, mean_flat, std_flat)
        
        # Ensure 2D shape (N, 3)
        points_denorm = np.asarray(points_denorm).reshape(-1, 3)
        
        # Save 2D visualization
        img_path = img_dir / f'epoch_{epoch:04d}_{sample_type}_sample_{i:02d}.png'
        visualize_pointcloud_2d(
            points_denorm,
            str(img_path),
            color_by_z=True,
            figsize=(8, 8),
            point_size=1.0,
            cmap='viridis',
            title=f'Generated NUP96 - Epoch {epoch} Sample {i}',
            dpi=150
        )
        
        # Save point cloud as CSV
        csv_path = pts_dir / f'epoch_{epoch:04d}_{sample_type}_sample_{i:02d}.csv'
        df = pd.DataFrame(points_denorm, columns=['x [nm]', 'y [nm]', 'z [nm]'])
        df.to_csv(csv_path, index=False)
    
    return img_dir, pts_dir


def train(gpu, opt, output_dir, noises_init):

    set_seed(opt)
    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')
        # Setup TensorBoard
        tb_dir = Path(output_dir) / 'tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f'TensorBoard log directory: {tb_dir}')
    else:
        writer = None

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter =  int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)


    ''' data '''
    train_dataset, _ = get_dataset(opt)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)
    
    # Get normalization parameters from dataset
    # 判断是否使用 minmax 归一化 (clustered 数据集: NUP96, CCP 或 NPC)
    use_minmax_norm = isinstance(train_dataset, (NUP96ClusteredPointClouds, CCPClusteredPointClouds, NPCClusteredPointClouds))
    
    if use_minmax_norm:
        # clustered 数据集: 使用平均 min/max 进行反归一化
        norm_min_vals, norm_max_vals = train_dataset.get_avg_norm_params()
        norm_mean = train_dataset.all_points_mean  # 兼容性
        norm_std = train_dataset.all_points_std    # 兼容性
    elif hasattr(train_dataset, 'all_points_mean') and hasattr(train_dataset, 'all_points_std'):
        norm_mean = train_dataset.all_points_mean
        norm_std = train_dataset.all_points_std
        norm_min_vals = None
        norm_max_vals = None
    else:
        norm_mean = np.array([0.0, 0.0, 0.0])
        norm_std = np.array([1.0, 1.0, 1.0])
        norm_min_vals = None
        norm_max_vals = None
    
    if should_diag:
        logger.info(f'Dataset size: {len(train_dataset)}')
        logger.info(f'Dataset mode: {opt.dataset_mode}')
        logger.info(f'Using minmax normalization: {use_minmax_norm}')
        if use_minmax_norm:
            logger.info(f'Avg min_vals: {norm_min_vals}')
            logger.info(f'Avg max_vals: {norm_max_vals}')
        else:
            logger.info(f'Normalization mean: {norm_mean}')
            logger.info(f'Normalization std: {norm_std}')


    '''
    create networks
    '''

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        # Use DataParallel with specified GPUs (remapped to 0, 1, 2, ...)
        if opt.num_gpus == 1:
            # Single GPU mode
            torch.cuda.set_device(0)
            model = model.cuda(0)
        else:
            # Multi-GPU with DataParallel
            def _transform_(m):
                return nn.parallel.DataParallel(m, device_ids=opt.remapped_gpu_ids)
            model = model.cuda()
            model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)
        logger.info(f'Model type: {opt.model_type}')

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    # Learning rate scheduler with warmup
    def get_lr_lambda(epoch):
        """Warmup + cosine decay schedule"""
        if epoch < opt.warmup_epochs:
            # Linear warmup
            return (epoch + 1) / opt.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - opt.warmup_epochs) / max(1, opt.niter - opt.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    if should_diag:
        logger.info(f'Using warmup for {opt.warmup_epochs} epochs + cosine decay')
    
    # Initialize EMA
    ema = EMA(model, decay=opt.ema_decay)
    if should_diag:
        logger.info(f'Using EMA with decay={opt.ema_decay}')

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, opt.niter):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(dataloader):
            x = data['train_points'].transpose(1,2)
            noises_batch = noises_init[data['idx']].transpose(1,2)

            '''
            train diffusion
            '''

            if opt.distribution_type == 'multi':
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)
            elif opt.distribution_type == 'single':
                # Data will be automatically distributed by DataParallel
                x = x.cuda()
                noises_batch = noises_batch.cuda()
            elif gpu is not None:
                x = x.cuda(gpu)
                noises_batch = noises_batch.cuda(gpu)

            loss = model.get_loss_iter(x, noises_batch).mean()

            optimizer.zero_grad()
            loss.backward()
            netpNorm, netgradNorm = getGradNorm(model)
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()
            
            # Update EMA after each step
            ema.update()
            
            # TensorBoard logging
            if should_diag and writer is not None:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Norm/param', netpNorm.item(), global_step)
                writer.add_scalar('Norm/grad', netgradNorm.item(), global_step)
                writer.add_scalar('LR', lr_scheduler.get_last_lr()[0], global_step)

            global_step += 1

            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             'netpNorm: {:>10.2f},   netgradNorm: {:>10.2f}     '
                             .format(
                        epoch, opt.niter, i, len(dataloader),loss.item(),
                    netpNorm, netgradNorm,
                        ))


        if (epoch + 1) % opt.diagIter == 0 and should_diag:

            logger.info('Diagnosis:')

            x_range = [x.min().item(), x.max().item()]
            kl_stats = model.all_kl(x)
            logger.info('      [{:>3d}/{:>3d}]    '
                         'x_range: [{:>10.4f}, {:>10.4f}],   '
                         'total_bpd_b: {:>10.4f},    '
                         'terms_bpd: {:>10.4f},  '
                         'prior_bpd_b: {:>10.4f}    '
                         'mse_bt: {:>10.4f}  '
                .format(
                epoch, opt.niter,
                *x_range,
                kl_stats['total_bpd_b'].item(),
                kl_stats['terms_bpd'].item(), kl_stats['prior_bpd_b'].item(), kl_stats['mse_bt'].item()
            ))
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Diag/total_bpd', kl_stats['total_bpd_b'].item(), epoch)
                writer.add_scalar('Diag/terms_bpd', kl_stats['terms_bpd'].item(), epoch)
                writer.add_scalar('Diag/prior_bpd', kl_stats['prior_bpd_b'].item(), epoch)
                writer.add_scalar('Diag/mse', kl_stats['mse_bt'].item(), epoch)



        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            logger.info('Generation: eval (using EMA weights)')

            model.eval()
            ema.apply_shadow()  # Use EMA weights for generation
            with torch.no_grad():

                x_gen_eval = model.gen_samples(new_x_chain(x, 10).shape, x.device, clip_denoised=False)
                x_gen_list = model.gen_sample_traj(new_x_chain(x, 1).shape, x.device, freq=40, clip_denoised=False)
                x_gen_all = torch.cat(x_gen_list, dim=0)

                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                logger.info('      [{:>3d}/{:>3d}]  '
                             'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                             'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                    .format(
                    epoch, opt.niter,
                    *gen_eval_range, *gen_stats,
                ))
                
                # Log generation stats to TensorBoard
                if writer is not None:
                    writer.add_scalar('Gen/mean', gen_stats[0].item(), epoch)
                    writer.add_scalar('Gen/std', gen_stats[1].item(), epoch)
                    writer.add_scalar('Gen/min', gen_eval_range[0], epoch)
                    writer.add_scalar('Gen/max', gen_eval_range[1], epoch)

            # Save generated samples with denormalization and 2D visualization
            img_dir, pts_dir = save_generated_samples(
                x_gen_eval, norm_mean, norm_std, epoch, output_dir, 'eval',
                normalize_mode='minmax' if use_minmax_norm else 'standard',
                min_vals=norm_min_vals, max_vals=norm_max_vals
            )
            logger.info(f'Saved generated samples to {img_dir} and {pts_dir}')

            # Original 3D batch visualization
            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                                       x_gen_eval.transpose(1, 2), None, None,
                                       None)

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                                       x_gen_all.transpose(1, 2), None,
                                       None,
                                       None)

            visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                                       None,
                                       None)

            ema.restore()  # Restore original weights after evaluation
            logger.info('Generation: train')
            model.train()


        if (epoch + 1) % opt.saveIter == 0:

            if should_diag:


                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'ema_shadow': ema.shadow,  # Save EMA weights
                    'norm_mean': norm_mean,
                    'norm_std': norm_std,
                    'use_minmax_norm': use_minmax_norm,
                    'norm_min_vals': norm_min_vals,
                    'norm_max_vals': norm_max_vals,
                }

                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))
                logger.info(f'Saved checkpoint to {output_dir}/epoch_{epoch}.pth')


            if opt.distribution_type == 'multi':
                dist.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])

        # Update learning rate scheduler at the end of each epoch
        lr_scheduler.step()

    # Close TensorBoard writer
    if should_diag and writer is not None:
        writer.close()
    
    if opt.distribution_type == 'multi':
        dist.destroy_process_group()

def main():
    opt = parse_args()
    
    # Set beta schedule based on dataset
    opt.beta_start = 1e-5
    opt.beta_end = 0.008
    opt.schedule_type = 'warm0.1'

    # GPU configuration
    # Note: CUDA_VISIBLE_DEVICES is already set before torch import (at script start)
    # Now torch.cuda.device_count() returns only the visible GPUs
    opt.num_gpus = torch.cuda.device_count()
    opt.remapped_gpu_ids = list(range(opt.num_gpus))
    
    if opt.num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available! Check your --gpus setting.")
    
    print(f'Visible GPUs after CUDA_VISIBLE_DEVICES: {opt.num_gpus}')
    print(f'Remapped GPU ids: {opt.remapped_gpu_ids}')
    print(f'Using model type: {opt.model_type}')
    print(f'Dataset type: {opt.dataset_type}')
    if opt.dataset_type == 'npc':
        print(f'NPC subfolder: {opt.npc_subfolder}')
        print(f'NPC dataroot: {opt.npc_dataroot}')
    elif opt.dataset_type == 'ccp':
        print(f'CCP stage: {opt.ccp_stage}')
        print(f'CCP dataroot: {opt.ccp_dataroot}')
    else:
        print(f'Dataset mode: {opt.dataset_mode}')
    
    if opt.model_type == 'pointcnn':
        print(f'PointCNN config:')
        print(f'  - width_multiplier: {opt.pointcnn_width_multiplier}')
        print(f'  - base_channels: {opt.pointcnn_base_channels}')
        print(f'  - kernel_size: {opt.pointcnn_kernel_size}')
        print(f'  - num_layers: {opt.pointcnn_num_layers}')
        print(f'  - downsample_ratio: {opt.pointcnn_downsample_ratio}')
    elif opt.model_type == 'gravnet':
        print(f'GravNet config:')
        print(f'  - width_multiplier: {opt.gravnet_width_multiplier}')
        print(f'  - base_channels: {opt.gravnet_base_channels}')
        print(f'  - space_dimensions: {opt.gravnet_space_dimensions}')
        print(f'  - propagate_dimensions: {opt.gravnet_propagate_dimensions}')
        print(f'  - k (nearest neighbors): {opt.gravnet_k}')
        print(f'  - num_layers: {opt.gravnet_num_layers}')
        print(f'  - downsample_ratio: {opt.gravnet_downsample_ratio}')

    # Create output directory with timestamp and experiment name
    # Format: {output_folder}/{timestamp}_{exp_name}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(opt.output_folder, f'{timestamp}_{opt.exp_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy source file to output directory for reproducibility
    copy_source(__file__, output_dir)
    
    print(f'Output directory: {output_dir}')

    ''' workaround '''
    train_dataset, _ = get_dataset(opt)
    # 更新 opt.npoints 以匹配实际数据集的点数
    actual_npoints = train_dataset.num_points
    if actual_npoints != opt.npoints:
        print(f"Updating npoints from {opt.npoints} to {actual_npoints} to match dataset")
        opt.npoints = actual_npoints
    noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        # Use specified number of GPUs for distributed training
        opt.ngpus_per_node = opt.num_gpus
        opt.world_size = opt.ngpus_per_node * opt.world_size
        print(f'Starting distributed training with {opt.ngpus_per_node} GPUs')
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        # Single mode: use DataParallel with specified GPUs
        if opt.num_gpus > 1:
            print(f'Starting DataParallel training with {opt.num_gpus} GPUs')
        train(None, opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    
    # Data settings
    parser.add_argument('--dataset_type', type=str, default='npc',
                        choices=['nup', 'ccp', 'npc'],
                        help='Dataset type: nup (NUP96 nuclear pore), ccp (Clathrin-Coated Pit), '
                             'or npc (Nuclear Pore Complex simulation)')
    parser.add_argument('--dataroot', default='/home/djx/data/nup96-large', 
                        help='Path to NUP96 data root')
    parser.add_argument('--dataset_mode', type=str, default='clustered',
                        choices=['h5', 'realtime', 'clustered'],
                        help='Dataset loading mode (for NUP): h5 (global norm), realtime (crop from CSV), '
                             'clustered (minmax norm, 13-clustered-pc-blocks.h5)')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Number of samples for realtime mode')
    
    # CCP specific settings
    parser.add_argument('--ccp_dataroot', default='/home/djx/data0/pvd_nup/ccp_stages/outputs',
                        help='Path to CCP data outputs root')
    parser.add_argument('--ccp_stage', type=str, default='early',
                        choices=['early', 'mid_early', 'middle', 'mid_late', 'late', 'mature'],
                        help='CCP developmental stage to use')
    
    # NPC specific settings
    parser.add_argument('--npc_dataroot', default='/data0/djx/pvd_nup/npc_batch/processed',
                        help='Path to NPC data outputs root')
    parser.add_argument('--npc_subfolder', type=str, default='rotated_density_0_9',
                        choices=['rotated_density_0_9', 'rotated_density_0_7', 'rotated_density_0_5',
                                 'fixed_density_0_9', 'fixed_density_0_7', 'fixed_density_0_5'],
                        help='NPC subfolder to use (rotation type + density)')

    # Training settings
    parser.add_argument('--bs', type=int, default=1, help='input batch size (larger batch helps stabilize diffusion training)')
    parser.add_argument('--workers', type=int, default=4, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', type=int, default=2048)
    
    '''model'''
    parser.add_argument('--beta_start', type=float, default=1e-5)
    parser.add_argument('--beta_end', type=float, default=0.008)
    parser.add_argument('--schedule_type', default='warm0.1')
    parser.add_argument('--time_num', type=int, default=1000)

    # Model params
    parser.add_argument('--model_type', type=str, default='pointcnn', choices=['pvcnn', 'pointcnn', 'gravnet'],
                        help='Model type: pvcnn (Point-Voxel CNN), pointcnn (PointCNN with XConv), or gravnet (GravNet with distance-weighted graph)')
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.05)  # 增加dropout防止过拟合
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--voxel_resolution_multiplier', type=int, default=2,
                        help='Voxel resolution multiplier (2 is better for small datasets, 4 causes sparse voxels)')
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')
    
    # PointCNN specific params (only used when model_type='pointcnn')
    parser.add_argument('--pointcnn_width_multiplier', type=float, default=4.0,
                        help='PointCNN width multiplier: 1.0=~2M params, 2.0=~8M params, 4.0=~32M params')
    parser.add_argument('--pointcnn_base_channels', type=int, default=128,
                        help='PointCNN base channel dimension (scaled by width_multiplier)')
    parser.add_argument('--pointcnn_kernel_size', type=int, default=32,
                        help='PointCNN XConv kernel size (number of neighbors)')
    parser.add_argument('--pointcnn_num_layers', type=int, default=4,
                        help='PointCNN number of encoder/decoder stages')
    parser.add_argument('--pointcnn_downsample_ratio', type=float, default=0.8,
                        help='PointCNN FPS downsampling ratio per stage')
    
    # GravNet specific params (only used when model_type='gravnet')
    parser.add_argument('--gravnet_width_multiplier', type=float, default=2.0,
                        help='GravNet width multiplier: 1.0=~2M params, 2.0=~8M params, 4.0=~32M params')
    parser.add_argument('--gravnet_base_channels', type=int, default=128,
                        help='GravNet base channel dimension (scaled by width_multiplier)')
    parser.add_argument('--gravnet_space_dimensions', type=int, default=6,
                        help='GravNet space dimensions for learnable neighbor finding (S in paper)')
    parser.add_argument('--gravnet_propagate_dimensions', type=int, default=16,
                        help='GravNet propagate dimensions (F_LR in paper)')
    parser.add_argument('--gravnet_k', type=int, default=32,
                        help='GravNet number of nearest neighbors')
    parser.add_argument('--gravnet_num_layers', type=int, default=8,
                        help='GravNet number of encoder/decoder stages')
    parser.add_argument('--gravnet_downsample_ratio', type=float, default=0.8,
                        help='GravNet FPS downsampling ratio per stage')

    # Optimizer (optimized for small dataset diffusion training)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (2e-4 works well for diffusion)')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam (0.9 is more stable than 0.5)')
    parser.add_argument('--decay', type=float, default=1e-4, help='weight decay for regularization')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping (essential for stable training)')
    parser.add_argument('--lr_gamma', type=float, default=0.9999, help='lr decay rate per step')
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                        help='Use learning rate scheduler (ExponentialLR)')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay rate for model averaging')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use for single GPU training. Ignored if --gpus is set.')
    parser.add_argument('--gpus', default="6,7", type=str,
                        help='Comma-separated GPU ids to use (e.g., "0,1,2" or "0,2,3"). '
                             'Overrides --gpu when set.')

    '''eval'''
    parser.add_argument('--saveIter', type=int, default=20, help='unit: epoch')
    parser.add_argument('--diagIter', type=int, default=50, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=5, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    # Output settings
    parser.add_argument('--output_folder', default='/data0/djx/pvd_nup/output',
                        help='Base output folder for saving experiments')
    parser.add_argument('--exp_name', default='nup96_pvd',
                        help='Experiment name (used in output folder naming)')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()
