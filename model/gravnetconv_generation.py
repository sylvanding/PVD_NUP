"""
GravNet-based model for point cloud generation using PyTorch Geometric's GravNetConv.

This module implements a U-Net style architecture using GravNet layers
for point cloud diffusion models.

GravNet dynamically constructs the graph using nearest neighbors in a learnable
low-dimensional projection of the feature space. A second projection of the input
feature space is then propagated from the neighbors to each vertex using distance
weights derived by applying a Gaussian function to the distances.

Reference:
- "Learning Representations of Irregular Particle-detector Geometry 
   with Distance-weighted Graph Networks" (https://arxiv.org/abs/1902.07987)
- PyG docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/
             torch_geometric.nn.conv.GravNetConv.html

Configurable parameters for scaling model capacity:
- width_multiplier: Scales all channel dimensions
- space_dimensions: Dimensionality of learnable space for neighbor finding (S)
- propagate_dimensions: Number of features propagated between vertices (F_LR)
- k: Number of nearest neighbors
- num_layers: Number of encoder/decoder stages
- base_channels: Base channel dimension
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GravNetConv, fps, knn_interpolate
from torch_geometric.nn import global_max_pool, global_mean_pool


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GravNetBlock(nn.Module):
    """
    A single GravNet block with batch normalization, activation, and optional dropout.
    
    Args:
        in_channels: Input feature channels
        out_channels: Output feature channels
        space_dimensions: Dimensionality of learnable space for neighbor construction
        propagate_dimensions: Number of features propagated between vertices
        k: Number of nearest neighbors
        dropout: Dropout rate
    """
    def __init__(self, in_channels, out_channels, space_dimensions=4, 
                 propagate_dimensions=32, k=16, dropout=0.1):
        super().__init__()
        
        self.conv = GravNetConv(
            in_channels=in_channels,
            out_channels=out_channels,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x, batch):
        """
        Args:
            x: Node features (B*N, C)
            batch: Batch indices (B*N,)
        
        Returns:
            Updated node features (B*N, out_channels)
        """
        x = self.conv(x, batch)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GravNetEncoder(nn.Module):
    """
    GravNet Encoder using GravNetConv layers with hierarchical downsampling.
    
    Uses Farthest Point Sampling (FPS) for downsampling between layers.
    
    Args:
        in_channels: Input feature channels (including position)
        embed_dim: Time embedding dimension
        base_channels: Base channel dimension (scaled by width_multiplier)
        width_multiplier: Multiplier for all channel dimensions
        space_dimensions: Dimensionality of learnable space for GravNet
        propagate_dimensions: Number of features propagated in GravNet
        k: Number of nearest neighbors for GravNet
        num_layers: Number of encoder stages
        dropout: Dropout rate
        downsample_ratio: Ratio for FPS downsampling per stage
        npoints: Number of input points
    """
    
    def __init__(self, in_channels, embed_dim, base_channels=64, 
                 width_multiplier=1.0, space_dimensions=4, propagate_dimensions=32,
                 k=16, num_layers=4, dropout=0.1, downsample_ratio=0.25, npoints=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.downsample_ratio = downsample_ratio
        self.npoints = npoints
        
        # Scale dimensions
        w = width_multiplier
        
        # Calculate point counts at each layer to adjust k
        point_counts = [npoints]
        for i in range(num_layers):
            next_count = max(int(point_counts[-1] * downsample_ratio), 1)
            point_counts.append(next_count)
        
        # Adjust k for each layer (must be <= number of points at that layer)
        self.k_per_layer = []
        for i in range(num_layers):
            available_points = point_counts[i]
            ks = min(k, max(available_points - 1, 4))
            self.k_per_layer.append(ks)
        
        print(f"GravNet Encoder: point_counts={point_counts}, k_per_layer={self.k_per_layer}")
        
        # Channel progression: base -> 2*base -> 4*base -> 8*base ...
        self.channels = []
        for i in range(num_layers):
            ch = int(base_channels * min(2 ** i, 8) * w)  # Cap at 8x base
            self.channels.append(ch)
        
        # Scale propagate dimensions with width
        self.propagate_dims = [int(propagate_dimensions * w) for _ in range(num_layers)]
        
        # Build encoder layers
        self.conv_layers = nn.ModuleList()
        self.extra_conv_layers = nn.ModuleList()  # Additional convolutions for more capacity
        
        prev_channels = in_channels + embed_dim  # First layer input (xyz + time embedding)
        for i, out_ch in enumerate(self.channels):
            ks = self.k_per_layer[i]
            prop_dim = self.propagate_dims[i]
            
            # Primary GravNet
            self.conv_layers.append(
                GravNetBlock(prev_channels, out_ch, 
                            space_dimensions=space_dimensions,
                            propagate_dimensions=prop_dim,
                            k=ks, dropout=dropout)
            )
            
            # Extra GravNet for more capacity (residual-style)
            self.extra_conv_layers.append(
                GravNetBlock(out_ch, out_ch,
                            space_dimensions=space_dimensions,
                            propagate_dimensions=prop_dim,
                            k=ks, dropout=dropout)
            )
            
            prev_channels = out_ch + embed_dim + 3  # Next layer input: features + time embedding + position
    
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
            # Prepare input: concatenate position, features, and time embedding
            temb_expanded = expand_temb(temb, current_batch)
            
            # For GravNet, we pass all features including position
            if current_x is not None and current_x.size(1) > 0:
                x_in = torch.cat([current_pos, current_x, temb_expanded], dim=-1)
            else:
                x_in = torch.cat([current_pos, temb_expanded], dim=-1)
            
            # Apply GravNet blocks
            x_out = self.conv_layers[i](x_in, current_batch)
            x_out = self.extra_conv_layers[i](x_out, current_batch) + x_out  # Residual
            
            # Downsample with FPS
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


class GravNetDecoder(nn.Module):
    """
    GravNet Decoder using feature propagation (interpolation) and MLP layers.
    
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
        self.fp_layers = nn.ModuleList()
        self.extra_fp_layers = nn.ModuleList()
        
        for i in range(self.num_layers, 0, -1):
            in_ch_interp = encoder_channels[i-1]
            
            if i > 1:
                in_ch_skip = encoder_channels[i-2]
            else:
                in_ch_skip = 0
            
            in_ch = in_ch_interp + in_ch_skip + embed_dim
            
            if i > 1:
                out_ch = encoder_channels[i-2]
            else:
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
            features_list: List of features from encoder at each scale
            pos_list: List of positions at each scale
            batch_list: List of batch indices at each scale
            temb: Time embedding (B, embed_dim)
        
        Returns:
            x: Final features at original resolution (N, C)
        """
        def expand_temb(temb, batch_indices):
            return temb[batch_indices]
        
        def safe_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k):
            if batch_x.numel() > 0:
                batch_counts = torch.bincount(batch_x)
                min_points_per_batch = batch_counts.min().item()
                safe_k = min(k, max(min_points_per_batch, 1))
            else:
                safe_k = 1
            return knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=safe_k)
        
        x = features_list[-1]
        
        for layer_idx, i in enumerate(range(self.num_layers, 0, -1)):
            x_interp = safe_knn_interpolate(
                x, pos_list[i], pos_list[i-1],
                batch_list[i], batch_list[i-1], k=self.knn_k
            )
            
            temb_expanded = expand_temb(temb, batch_list[i-1])
            
            if i > 1:
                x = torch.cat([x_interp, features_list[i-1], temb_expanded], dim=-1)
            else:
                x = torch.cat([x_interp, temb_expanded], dim=-1)
            
            x = self.fp_layers[layer_idx](x)
            x = self.extra_fp_layers[layer_idx](x) + x  # Residual
        
        return x


class GravNetBase(nn.Module):
    """
    GravNet-based model for point cloud diffusion.
    
    Uses a U-Net style architecture with GravNetConv layers for feature extraction
    and feature propagation for upsampling.
    
    Key features of GravNet:
    - Dynamically constructs graphs using learnable low-dimensional projections
    - Distance-weighted message passing using Gaussian functions
    - Efficient for point cloud processing without explicit graph construction
    
    Args:
        num_classes: Output channels (typically 3 for xyz prediction)
        embed_dim: Time embedding dimension
        dropout: Dropout rate
        extra_feature_channels: Additional input feature channels (beyond xyz)
        base_channels: Base channel dimension
        width_multiplier: Multiplier for all channel dimensions
        space_dimensions: Dimensionality of learnable space for neighbor finding
        propagate_dimensions: Number of features propagated between vertices
        k: Number of nearest neighbors
        num_layers: Number of encoder/decoder stages
        downsample_ratio: FPS downsampling ratio per stage
        npoints: Number of input points
    """
    
    def __init__(self, num_classes, embed_dim=64, dropout=0.1, 
                 extra_feature_channels=0, base_channels=64,
                 width_multiplier=1.0, space_dimensions=4, 
                 propagate_dimensions=32, k=16, num_layers=4,
                 downsample_ratio=0.25, npoints=2048, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels
        self.width_multiplier = width_multiplier
        self.npoints = npoints
        
        # Validate configuration
        self._validate_config(npoints, num_layers, downsample_ratio, k)
        
        w = width_multiplier
        
        # Time embedding MLP
        temb_dim = int(embed_dim * max(w, 1.0))
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, temb_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(temb_dim, embed_dim),
        )
        
        # Input channels: position (3) + extra features
        # Note: GravNet takes all features including position
        encoder_in_channels = 3 + extra_feature_channels
        
        # Encoder
        self.encoder = GravNetEncoder(
            in_channels=encoder_in_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            width_multiplier=width_multiplier,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
            num_layers=num_layers,
            dropout=dropout,
            downsample_ratio=downsample_ratio,
            npoints=npoints
        )
        
        # Decoder  
        self.decoder = GravNetDecoder(
            encoder_channels=self.encoder.channels,
            embed_dim=embed_dim,
            width_multiplier=width_multiplier,
            dropout=dropout
        )
        
        # Final classifier
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
    
    def _validate_config(self, npoints, num_layers, downsample_ratio, k):
        """Validate and warn about configuration issues."""
        min_points = npoints
        for _ in range(num_layers):
            min_points = int(min_points * downsample_ratio)
        
        if min_points < 4:
            print(f"WARNING: With npoints={npoints}, num_layers={num_layers}, "
                  f"downsample_ratio={downsample_ratio}, deepest layer will have "
                  f"~{min_points} points. Consider reducing num_layers or increasing "
                  f"downsample_ratio for better performance.")
        
        if k > npoints:
            print(f"WARNING: k={k} is larger than npoints={npoints}. "
                  f"k will be automatically adjusted per layer.")
    
    def _print_model_info(self):
        """Print model configuration and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GravNet Model Info:")
        print(f"  - Width multiplier: {self.width_multiplier}")
        print(f"  - Encoder channels: {self.encoder.channels}")
        print(f"  - k per layer: {self.encoder.k_per_layer}")
        print(f"  - Propagate dimensions: {self.encoder.propagate_dims}")
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
        
        # Extra features beyond position
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


class GravNet2(GravNetBase):
    """
    GravNet model with configurable capacity.
    
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
        space_dimensions: GravNet space dimensions (default 4)
        propagate_dimensions: GravNet propagate dimensions (default 32)
        k: Number of nearest neighbors (default 16)
        num_layers: Number of encoder/decoder stages (default 4)
        base_channels: Base channel dimension (default 64)
        downsample_ratio: FPS downsampling ratio (default 0.25)
        npoints: Number of input points (default 2048)
    """
    
    def __init__(self, num_classes, embed_dim, use_att=False, dropout=0.1,
                 extra_feature_channels=0, width_multiplier=1.0,
                 voxel_resolution_multiplier=1, space_dimensions=4,
                 propagate_dimensions=32, k=16, num_layers=4, 
                 base_channels=64, downsample_ratio=0.25, npoints=2048):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            base_channels=base_channels,
            width_multiplier=width_multiplier,
            space_dimensions=space_dimensions,
            propagate_dimensions=propagate_dimensions,
            k=k,
            num_layers=num_layers,
            downsample_ratio=downsample_ratio,
            npoints=npoints
        )


# Preset configurations for different model sizes
class GravNetSmall(GravNet2):
    """Small GravNet: ~1M params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 0.5
        kwargs['base_channels'] = 48
        super().__init__(num_classes, embed_dim, **kwargs)


class GravNetMedium(GravNet2):
    """Medium GravNet: ~4M params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 1.5
        kwargs['base_channels'] = 64
        super().__init__(num_classes, embed_dim, **kwargs)


class GravNetLarge(GravNet2):
    """Large GravNet: ~16M params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 2.5
        kwargs['base_channels'] = 96
        kwargs['k'] = 24
        super().__init__(num_classes, embed_dim, **kwargs)


class GravNetXLarge(GravNet2):
    """XLarge GravNet: ~32M+ params"""
    def __init__(self, num_classes, embed_dim, **kwargs):
        kwargs['width_multiplier'] = 4.0
        kwargs['base_channels'] = 128
        kwargs['k'] = 32
        kwargs['num_layers'] = 5
        super().__init__(num_classes, embed_dim, **kwargs)
