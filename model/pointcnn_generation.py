"""
PointCNN-based model for point cloud generation using PyTorch Geometric's XConv.

This module implements a U-Net style architecture using PointCNN (XConv) layers
for point cloud diffusion models.

Configurable parameters for scaling model capacity:
- width_multiplier: Scales all channel dimensions
- kernel_size: Number of neighbors for XConv
- num_layers: Number of encoder/decoder stages
- base_channels: Base channel dimension
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import XConv, fps, knn_interpolate
from torch_geometric.nn import global_max_pool, global_mean_pool


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class XConvBlock(nn.Module):
    """
    A single XConv block with batch normalization, activation, and optional dropout.
    """
    def __init__(self, in_channels, out_channels, dim=3, kernel_size=16, 
                 hidden_channels=None, dilation=1, dropout=0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(out_channels // 2, 16)
        
        self.kernel_size = kernel_size
        self.conv = XConv(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, pos, batch):
        x = self.conv(x, pos, batch)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class PointCNNEncoder(nn.Module):
    """
    PointCNN Encoder using XConv layers with hierarchical downsampling.
    
    Uses Farthest Point Sampling (FPS) for downsampling between layers.
    
    Args:
        in_channels: Input feature channels (excluding position)
        embed_dim: Time embedding dimension
        base_channels: Base channel dimension (scaled by width_multiplier)
        width_multiplier: Multiplier for all channel dimensions
        kernel_size: Base number of neighbors for XConv (will be reduced for deeper layers)
        num_layers: Number of encoder stages (default: 4)
        dropout: Dropout rate
        downsample_ratio: Ratio for FPS downsampling per stage
        npoints: Number of input points (used to calculate appropriate kernel sizes)
    """
    
    def __init__(self, in_channels, embed_dim, base_channels=64, 
                 width_multiplier=1.0, kernel_size=16, num_layers=4,
                 dropout=0.1, downsample_ratio=0.25, npoints=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.downsample_ratio = downsample_ratio
        self.npoints = npoints
        
        # Scale base channels
        w = width_multiplier
        
        # Calculate point counts at each layer and appropriate kernel sizes
        # Layer i operates on points BEFORE downsampling
        point_counts = [npoints]
        for i in range(num_layers):
            next_count = max(int(point_counts[-1] * downsample_ratio), 1)
            point_counts.append(next_count)
        
        # Kernel size for each layer (must be <= number of points at that layer)
        self.kernel_sizes = []
        for i in range(num_layers):
            # Points available at layer i (before downsampling)
            available_points = point_counts[i]
            # Use min of requested kernel_size and available points
            # Also ensure at least 4 neighbors for meaningful convolution
            ks = min(kernel_size, max(available_points - 1, 4))
            self.kernel_sizes.append(ks)
        
        print(f"PointCNN Encoder: point_counts={point_counts}, kernel_sizes={self.kernel_sizes}")
        
        # Channel progression: base -> 2*base -> 4*base -> 8*base ...
        # Cap the channel growth to avoid memory issues
        self.channels = []
        for i in range(num_layers):
            ch = int(base_channels * min(2 ** i, 8) * w)  # Cap at 8x base
            self.channels.append(ch)
        
        # Build encoder layers
        self.conv_layers = nn.ModuleList()
        self.extra_conv_layers = nn.ModuleList()  # Additional convolutions per stage for more capacity
        
        prev_channels = in_channels + embed_dim  # First layer input
        for i, out_ch in enumerate(self.channels):
            ks = self.kernel_sizes[i]
            # Primary XConv
            hidden_ch = max(out_ch // 2, 32)
            self.conv_layers.append(
                XConvBlock(prev_channels, out_ch, kernel_size=ks,
                          hidden_channels=hidden_ch, dropout=dropout)
            )
            
            # Extra XConv for more capacity (residual-style)
            self.extra_conv_layers.append(
                XConvBlock(out_ch, out_ch, kernel_size=ks,
                          hidden_channels=hidden_ch, dropout=dropout)
            )
            
            prev_channels = out_ch + embed_dim  # Next layer input includes time embedding
    
    def forward(self, x, pos, batch, temb):
        """
        Args:
            x: Node features (B*N, C) or None
            pos: Positions (B*N, 3)
            batch: Batch indices (B*N,)
            temb: Time embedding (B, embed_dim)
        
        Returns:
            features_list: List of features at each scale
            pos_list: List of positions at each scale
            batch_list: List of batch indices at each scale
        """
        features_list = []
        pos_list = []
        batch_list = []
        
        # Store original
        pos_list.append(pos)
        batch_list.append(batch)
        features_list.append(x if x is not None else torch.zeros(pos.size(0), 0, device=pos.device))
        
        # Expand time embedding to match points
        def expand_temb(temb, batch_indices):
            return temb[batch_indices]
        
        current_x = x
        current_pos = pos
        current_batch = batch
        
        for i in range(self.num_layers):
            # Prepare input with time embedding
            temb_expanded = expand_temb(temb, current_batch)
            if current_x is not None and current_x.size(1) > 0:
                x_in = torch.cat([current_x, temb_expanded], dim=-1)
            else:
                x_in = temb_expanded
            
            # Apply XConv blocks
            x_out = self.conv_layers[i](x_in, current_pos, current_batch)
            x_out = self.extra_conv_layers[i](x_out, current_pos, current_batch) + x_out  # Residual
            
            # Downsample with FPS (ensure at least 1 point per sample)
            # Calculate how many points we'll have after downsampling
            num_points = current_pos.size(0)
            target_points = max(int(num_points * self.downsample_ratio), batch.max().item() + 1)
            actual_ratio = target_points / num_points
            
            idx = fps(current_pos, current_batch, ratio=actual_ratio)
            current_pos = current_pos[idx]
            current_batch = current_batch[idx]
            current_x = x_out[idx]
            
            features_list.append(current_x)
            pos_list.append(current_pos)
            batch_list.append(current_batch)
        
        return features_list, pos_list, batch_list


class PointCNNDecoder(nn.Module):
    """
    PointCNN Decoder using feature propagation (interpolation) and MLP layers.
    
    Args:
        encoder_channels: List of channel dimensions from encoder
        embed_dim: Time embedding dimension
        width_multiplier: Multiplier for channel dimensions
        dropout: Dropout rate
        knn_k: Number of neighbors for kNN interpolation
    """
    
    def __init__(self, encoder_channels, embed_dim, width_multiplier=1.0,
                 dropout=0.1, knn_k=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(encoder_channels)
        self.knn_k = knn_k
        
        w = width_multiplier
        
        # Build decoder layers (reverse order)
        # We need num_layers MLP layers for propagation from level num_layers to level 0
        self.fp_layers = nn.ModuleList()
        self.extra_fp_layers = nn.ModuleList()  # Additional MLPs for more capacity
        
        # Layer for: level num_layers -> level num_layers-1, ..., level 1 -> level 0
        for i in range(self.num_layers, 0, -1):
            # i goes from num_layers down to 1
            # Input: interpolated features from level i + skip features from level i-1 + time embedding
            # encoder_channels[j] corresponds to features_list[j+1]
            in_ch_interp = encoder_channels[i-1]  # channel from level i (encoder_channels[i-1])
            
            if i > 1:
                # Skip connection from level i-1 (encoder_channels[i-2])
                in_ch_skip = encoder_channels[i-2]
            else:
                # Level 0 has no encoder features (original input)
                in_ch_skip = 0
            
            in_ch = in_ch_interp + in_ch_skip + embed_dim
            
            if i > 1:
                out_ch = encoder_channels[i-2]
            else:
                # Final output channels
                out_ch = int(64 * w)
            
            # Primary MLP
            self.fp_layers.append(nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                Swish(),
                nn.Dropout(dropout)
            ))
            
            # Extra MLP for capacity
            self.extra_fp_layers.append(nn.Sequential(
                nn.Linear(out_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                Swish(),
                nn.Dropout(dropout)
            ))
        
        self.final_out_channels = int(64 * w)
    
    def forward(self, features_list, pos_list, batch_list, temb):
        """
        Args:
            features_list: List of features from encoder at each scale [level0, level1, ..., level_num_layers]
            pos_list: List of positions at each scale
            batch_list: List of batch indices at each scale
            temb: Time embedding (B, embed_dim)
        
        Returns:
            x: Final features at original resolution (N, C)
        """
        def expand_temb(temb, batch_indices):
            return temb[batch_indices]
        
        # Adaptive knn_k based on available points per batch
        def safe_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k):
            # Ensure k doesn't exceed available source points per batch
            # Count minimum points per batch in source
            if batch_x.numel() > 0:
                batch_counts = torch.bincount(batch_x)
                min_points_per_batch = batch_counts.min().item()
                safe_k = min(k, max(min_points_per_batch, 1))
            else:
                safe_k = 1
            return knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=safe_k)
        
        # Start from deepest features (last in list)
        # features_list[-1] = features_list[num_layers] corresponds to pos_list[num_layers]
        x = features_list[-1]
        
        # Propagate features up (from deepest to shallowest)
        # i goes from num_layers down to 1
        for layer_idx, i in enumerate(range(self.num_layers, 0, -1)):
            # Interpolate from level i to level i-1
            x_interp = safe_knn_interpolate(
                x, pos_list[i], pos_list[i-1],
                batch_list[i], batch_list[i-1], k=self.knn_k
            )
            
            # Concatenate with skip features and time embedding
            temb_expanded = expand_temb(temb, batch_list[i-1])
            
            if i > 1:
                # Skip connection from encoder features at level i-1
                x = torch.cat([x_interp, features_list[i-1], temb_expanded], dim=-1)
            else:
                # Level 0: no encoder features (original input has no features)
                x = torch.cat([x_interp, temb_expanded], dim=-1)
            
            # Apply MLP
            x = self.fp_layers[layer_idx](x)
            x = self.extra_fp_layers[layer_idx](x) + x  # Residual
        
        return x


class PointCNNBase(nn.Module):
    """
    PointCNN-based model for point cloud diffusion.
    
    Uses a U-Net style architecture with XConv layers for feature extraction
    and feature propagation for upsampling.
    
    Args:
        num_classes: Output channels (typically 3 for xyz prediction)
        embed_dim: Time embedding dimension
        dropout: Dropout rate
        extra_feature_channels: Additional input feature channels (beyond xyz)
        base_channels: Base channel dimension
        width_multiplier: Multiplier for all channel dimensions
        kernel_size: Number of neighbors for XConv
        num_layers: Number of encoder/decoder stages
        downsample_ratio: FPS downsampling ratio per stage
        npoints: Number of input points
    """
    
    def __init__(self, num_classes, embed_dim=64, dropout=0.1, 
                 extra_feature_channels=0, base_channels=64,
                 width_multiplier=1.0, kernel_size=16, num_layers=4,
                 downsample_ratio=0.25, npoints=2048, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels
        self.width_multiplier = width_multiplier
        self.npoints = npoints
        
        # Validate configuration
        self._validate_config(npoints, num_layers, downsample_ratio, kernel_size)
        
        w = width_multiplier
        
        # Time embedding MLP (also scaled)
        temb_dim = int(embed_dim * max(w, 1.0))
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, temb_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(temb_dim, embed_dim),
        )
        
        # Encoder
        self.encoder = PointCNNEncoder(
            in_channels=extra_feature_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            width_multiplier=width_multiplier,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
            downsample_ratio=downsample_ratio,
            npoints=npoints
        )
        
        # Decoder  
        self.decoder = PointCNNDecoder(
            encoder_channels=self.encoder.channels,
            embed_dim=embed_dim,
            width_multiplier=width_multiplier,
            dropout=dropout
        )
        
        # Final classifier (scaled)
        classifier_hidden = int(128 * w)
        self.classifier = nn.Sequential(
            nn.Linear(self.decoder.final_out_channels, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )
        
        # Print model info
        self._print_model_info()
    
    def _validate_config(self, npoints, num_layers, downsample_ratio, kernel_size):
        """Validate and warn about configuration issues."""
        # Calculate minimum points at deepest layer
        min_points = npoints
        for _ in range(num_layers):
            min_points = int(min_points * downsample_ratio)
        
        if min_points < 4:
            print(f"WARNING: With npoints={npoints}, num_layers={num_layers}, "
                  f"downsample_ratio={downsample_ratio}, deepest layer will have "
                  f"~{min_points} points. Consider reducing num_layers or increasing "
                  f"downsample_ratio for better performance.")
    
    def _print_model_info(self):
        """Print model configuration and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"PointCNN Model Info:")
        print(f"  - Width multiplier: {self.width_multiplier}")
        print(f"  - Encoder channels: {self.encoder.channels}")
        print(f"  - Encoder kernel sizes: {self.encoder.kernel_sizes}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def get_timestep_embedding(self, timesteps, device):
        """Generate sinusoidal timestep embeddings."""
        assert len(timesteps.shape) == 1
        
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb
    
    def forward(self, inputs, t):
        """
        Forward pass for diffusion model.
        
        Args:
            inputs: Point cloud tensor (B, C, N) where C=3 for xyz
            t: Timestep tensor (B,)
        
        Returns:
            Predicted noise (B, C, N)
        """
        B, C, N = inputs.shape
        device = inputs.device
        
        # Get time embedding
        temb = self.embedf(self.get_timestep_embedding(t, device))  # (B, embed_dim)
        
        # Convert from (B, C, N) to PyG format (B*N, C)
        pos = inputs.permute(0, 2, 1).reshape(B * N, C)  # (B*N, 3)
        
        # Create batch indices
        batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
        
        # No extra features beyond position for basic point cloud
        x = None
        if self.in_channels > 0:
            x = inputs[:, 3:, :].permute(0, 2, 1).reshape(B * N, -1)
        
        # Encode
        features_list, pos_list, batch_list = self.encoder(x, pos, batch, temb)
        
        # Decode
        x_out = self.decoder(features_list, pos_list, batch_list, temb)
        
        # Classify
        x_out = self.classifier(x_out)  # (B*N, num_classes)
        
        # Convert back to (B, C, N)
        x_out = x_out.reshape(B, N, self.num_classes).permute(0, 2, 1)
        
        return x_out


class PointCNN2(PointCNNBase):
    """
    PointCNN model with configurable capacity.
    
    Preset configurations:
    - width_multiplier=1.0: ~2M params (baseline)
    - width_multiplier=2.0: ~8M params (medium)
    - width_multiplier=4.0: ~32M params (large)
    
    Args:
        num_classes: Output channels (typically 3)
        embed_dim: Time embedding dimension
        use_att: Not used, kept for API compatibility
        dropout: Dropout rate
        extra_feature_channels: Extra input features
        width_multiplier: Channel width multiplier (1.0-4.0 recommended)
        voxel_resolution_multiplier: Not used, kept for API compatibility
        kernel_size: XConv kernel size (neighbors), default 16
        num_layers: Number of encoder/decoder stages, default 4
        base_channels: Base channel dimension, default 64
        npoints: Number of input points, default 2048
    """
    
    def __init__(self, num_classes, embed_dim, use_att=False, dropout=0.1,
                 extra_feature_channels=0, width_multiplier=1.0,
                 voxel_resolution_multiplier=1, kernel_size=16,
                 num_layers=4, base_channels=64, downsample_ratio=0.25,
                 npoints=2048):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            base_channels=base_channels,
            width_multiplier=width_multiplier,
            kernel_size=kernel_size,
            num_layers=num_layers,
            downsample_ratio=downsample_ratio,
            npoints=npoints
        )


# Preset configurations for different model sizes
class PointCNNSmall(PointCNN2):
    """Small PointCNN: ~1M params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 0.5
        kwargs['base_channels'] = 48
        super().__init__(num_classes, embed_dim, **kwargs)


class PointCNNMedium(PointCNN2):
    """Medium PointCNN: ~4M params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 1.5
        kwargs['base_channels'] = 64
        super().__init__(num_classes, embed_dim, **kwargs)


class PointCNNLarge(PointCNN2):
    """Large PointCNN: ~16M params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 2.5
        kwargs['base_channels'] = 96
        kwargs['kernel_size'] = 24
        super().__init__(num_classes, embed_dim, **kwargs)


class PointCNNXLarge(PointCNN2):
    """XLarge PointCNN: ~32M+ params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 4.0
        kwargs['base_channels'] = 128
        kwargs['kernel_size'] = 32
        kwargs['num_layers'] = 5
        super().__init__(num_classes, embed_dim, **kwargs)
