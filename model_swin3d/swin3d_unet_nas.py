"""
Swin3D UNet NAS (EXTENDED SEARCH SPACE)
- Supports search over:
    * FFN op choices (MixedOp3D)
    * Attention variants (MixedAttention3D: different window sizes / heads)
    * Block gate (scalar): sigmoid(gate_alpha) * block_output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PatchMerging3D, PatchExpanding3D, window_partition, window_reverse
from .nas_ops import MixedOp3D, MixedAttention3D, AlphaManager
from .swin3d_block import LayerNormChannel, DropPath

# -------------------------
# Helper wrapper for a single block (NAS-aware)
# -------------------------
class Swin3DBlock_NAS(nn.Module):
    def __init__(self, dim, input_resolution, attn_candidates, ffn_op_names, shift_size=(0,0,0), drop_path=0.0):
        """
        attn_candidates: list of dicts {"window_size":(wd,wh,ww), "num_heads": h}
        ffn_op_names: list of strings e.g. ["conv3","conv1","dw","identity"]
        """
        super().__init__()
        self.dim = dim
        self.D, self.H, self.W = input_resolution
        self.shift_size = shift_size
        self.window_size = attn_candidates[0]["window_size"]  # primary window for mask shape (we expect candidates have same N)
        self.norm1 = LayerNormChannel(dim)
        self.norm2 = LayerNormChannel(dim)

        # attention: either MixedAttention or standard single
        self.attn = MixedAttention3D(dim, attn_candidates)

        self.drop_path = DropPath(drop_path)

        # ffn mixed op
        self.ffn = MixedOp3D(dim, dim, op_names=ffn_op_names)

        # gate alpha (set externally). We'll use sigmoid(gate) to multiply residual
        self.gate = None  # scalar tensor (alpha) set via AlphaManager

        # cached mask
        self.register_buffer("attn_mask", None, persistent=False)

    def calculate_mask(self, x):
        if self.attn_mask is not None:
            return self.attn_mask
        D, H, W = self.D, self.H, self.W
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size
        from .layers import compute_attn_mask
        mask = compute_attn_mask(D, H, W, (wd,wh,ww), (sd,sh,sw), x.device)
        self.attn_mask = mask
        return mask

    def forward(self, x, mode='soft', temp=1.0, hard=False):
        """
        x: (B, C, D, H, W)
        mode: 'soft' for soft forward, 'gumbel' for Gumbel-Softmax sampling
        temp: temperature for Gumbel-Softmax
        hard: hard=True for ST estimator (one-hot in forward, gradient in backward)
        """
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        shortcut = x
        x_norm = self.norm1(x)

        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            shifted = torch.roll(x_norm, shifts=(-sd,-sh,-sw), dims=(2,3,4))
        else:
            shifted = x_norm

        # partition windows
        wd, wh, ww = self.window_size
        windows = window_partition(shifted, (wd,wh,ww))  # (nW*B, wd,wh,ww, C)
        N = wd*wh*ww
        windows_flat = windows.view(-1, N, C)

        mask = self.calculate_mask(x_norm)
        
        # Attention forward
        if mode == 'gumbel':
            attn_out, attn_w = self.attn.forward_gumbel(windows_flat, mask=mask, temp=temp, hard=hard)
            # Store weights for logging (optional)
            self._last_attn_weights = attn_w.detach().cpu().numpy()
        else:  # mode == 'soft'
            attn_out = self.attn.forward_soft(windows_flat, mask=mask)
        
        attn_out = attn_out.view(-1, wd, wh, ww, C)
        shifted_x = window_reverse(attn_out, (wd,wh,ww), B, D, H, W)

        if sd or sh or sw:
            x_attn = torch.roll(shifted_x, shifts=(sd,sh,sw), dims=(2,3,4))
        else:
            x_attn = shifted_x

        # block gating: gate is a scalar alpha -> use sigmoid(gate) to scale the block output
        if self.gate is not None:
            gate_val = torch.sigmoid(self.gate)
        else:
            gate_val = 1.0
        x = shortcut + self.drop_path(gate_val * x_attn)

        # FFN
        shortcut2 = x
        x2 = self.norm2(x)
        
        # FFN forward
        if mode == 'gumbel':
            ffn_out, ffn_w = self.ffn.forward_gumbel(x2, temp=temp, hard=hard)
            # Store weights for logging (optional)
            self._last_ffn_weights = ffn_w.detach().cpu().numpy()
        else:  # mode == 'soft'
            ffn_out = self.ffn.forward_soft(x2)
        
        x = shortcut2 + self.drop_path(ffn_out)
        return x

# -------------------------
# Stage builder using NAS blocks
# -------------------------
class SwinStage_NAS(nn.Module):
    def __init__(self, dim, resolution, depth, attn_candidates, ffn_op_names, drop_path_list, downsample=True):
        super().__init__()
        D,H,W = resolution
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0,0,0) if (i%2==0) else (attn_candidates[0]["window_size"][0]//2,
                                              attn_candidates[0]["window_size"][1]//2,
                                              attn_candidates[0]["window_size"][2]//2)
            blk = Swin3DBlock_NAS(dim, resolution, attn_candidates, ffn_op_names, shift_size=shift, drop_path=drop_path_list[i])
            self.blocks.append(blk)
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# -------------------------
# Main model (NAS-enabled)
# -------------------------
class SwinUNet3D_NAS(nn.Module):
    """
    Extended NAS-enabled SwinUNet3D.
    search_space argument defines attention candidates and ffn options.
    """
    def __init__(self,
                 in_channels=1,
                 num_classes=4,
                 dims=(48,96,192,384),
                 depths=(2,2,2,2),
                 window_candidates=None,
                 ffn_op_names=None,
                 drop_path_rate=0.1,
                 resolution=(128,128,128),
                 nas=True):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dims = dims
        self.depths = depths
        self.resolution = resolution
        self.nas = nas

        # default search candidates
        if window_candidates is None:
            # IMPORTANT: All candidates must use the same window_size for MixedAttention3D to work
            # Different window_sizes cause relative_position_bias size mismatch
            # We only search over num_heads to avoid this issue
            window_candidates = [
                {"window_size": (2,7,7), "num_heads": 2},
                {"window_size": (2,7,7), "num_heads": 4},
                {"window_size": (2,7,7), "num_heads": 6},
            ]
        if ffn_op_names is None:
            ffn_op_names = ["conv3","conv1","dw","identity","conv5"]

        # patch embed
        self.patch_embed = nn.Conv3d(in_channels, dims[0], kernel_size=2, stride=2)

        # build stages: need to keep track of total alpha groups
        alpha_groups = []  # list of int lengths: [ffn_len, attn_len, gate_len]*num_blocks

        # create stages as modules but we need to pass candidates
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        idx = 0

        self.stages = nn.ModuleList()
        for s in range(len(depths)):
            dim = dims[s]
            depth = depths[s]
            drop_list = dpr[idx: idx+depth]; idx += depth
            res = (
                resolution[0] // (2**s),
                resolution[1] // (2**s),
                resolution[2] // (2**s)
            )
            stage = SwinStage_NAS(dim, res, depth, window_candidates, ffn_op_names, drop_list, downsample=(s<3))
            self.stages.append(stage)

            # collect alpha groups per block
            for b in range(depth):
                alpha_groups.append(len(ffn_op_names))   # ffn choices
                alpha_groups.append(len(window_candidates))  # attention choices
                alpha_groups.append(1)  # gate (scalar)

        # head / decoder
        self.up_blocks = nn.ModuleList()
        for i in range(3,0,-1):
            self.up_blocks.append(nn.Sequential(
                PatchExpanding3D(dims[i], dims[i-1]),
                nn.Conv3d(dims[i-1], dims[i-1], kernel_size=3, padding=1),
                nn.GELU()
            ))
        self.head = nn.Conv3d(dims[0], num_classes, kernel_size=1)

        # create AlphaManager if nas
        if self.nas:
            self.alpha_mgr = AlphaManager(alpha_groups)
        else:
            self.alpha_mgr = None

        # map alpha assignment order to blocks
        # We'll create a flat list of block references to assign in set_alpha()
        self._collect_blocks_for_alpha()

    def _collect_blocks_for_alpha(self):
        # flatten list of blocks in order
        self._nas_blocks = []
        for stg in self.stages:
            for blk in stg.blocks:
                self._nas_blocks.append(blk)

    def set_alpha(self):
        if not self.nas: return
        alphas = self.alpha_mgr()
        ptr = 0
        for blk in self._nas_blocks:
            # assign ffn alpha
            ffn_alpha = alphas[ptr]; ptr += 1
            attn_alpha = alphas[ptr]; ptr += 1
            gate_alpha = alphas[ptr]; ptr += 1

            blk.ffn.alpha = ffn_alpha
            blk.attn.alpha = attn_alpha
            # gate_alpha is vector length 1: set scalar param
            blk.gate = gate_alpha

    def arch_parameters(self):
        if not self.nas: return []
        return list(self.alpha_mgr.parameters())

    def forward(self, x):
        """
        Standard forward (defaults to soft for backward compatibility).
        For Gumbel-Softmax training, use forward_gumbel.
        For validation/architecture evaluation, use forward_soft.
        """
        return self.forward_soft(x)
    
    def forward_soft(self, x):
        """
        Forward using soft mix (weighted average of all candidates).
        Used for validation and architecture evaluation.
        """
        return self._forward_shared(x, mode='soft')
    
    def forward_gumbel(self, x, temp=1.0, hard=False):
        """
        Forward using Gumbel-Softmax sampling (for weight updates).
        - temp: temperature (higher = smoother, lower = more discrete)
        - hard: if True, uses one-hot in forward (ST estimator), else soft weights
        """
        return self._forward_shared(x, mode='gumbel', temp=temp, hard=hard)
    
    def forward_encoder(self, x, mode='soft', temp=1.0, hard=False):
        """
        Forward only through encoder, return bottleneck features (for Barlow Twins).
        Returns: (B, C, D, H, W) bottleneck feature map from the last encoder stage.
        """
        # x: B,C,D,H,W
        B,C,D,H,W = x.shape

        # patch embed
        x = self.patch_embed(x)
        # compute stage resolutions, set into blocks
        res = [(D//2, H//2, W//2)]
        for i in range(1,4):
            res.append((res[-1][0]//2, res[-1][1]//2, res[-1][2]//2))
        # assign resolutions to blocks
        for stg, r in zip(self.stages, res):
            for blk in stg.blocks:
                blk.D, blk.H, blk.W = r

        # assign alpha values to blocks (needed for both soft and gumbel)
        if self.nas:
            self.set_alpha()

        # encoder only - return bottleneck (last stage output)
        out = x
        for stage_idx, stg in enumerate(self.stages):
            # Process through blocks
            for blk in stg.blocks:
                out = blk(out, mode=mode, temp=temp, hard=hard)
            
            # Apply downsampling if present
            if stg.downsample is not None:
                out = stg.downsample(out)
        
        # Return bottleneck feature (last stage output, shape: B, dims[-1], D/16, H/16, W/16)
        return out
    
    def _forward_shared(self, x, mode='soft', temp=1.0, hard=False):
        """
        Shared forward implementation supporting soft and gumbel modes.
        """
        # x: B,C,D,H,W
        B,C,D,H,W = x.shape

        # patch embed
        x = self.patch_embed(x)
        # compute stage resolutions, set into blocks
        res = [(D//2, H//2, W//2)]
        for i in range(1,4):
            res.append((res[-1][0]//2, res[-1][1]//2, res[-1][2]//2))
        # assign resolutions to blocks
        for stg, r in zip(self.stages, res):
            for blk in stg.blocks:
                blk.D, blk.H, blk.W = r

        # assign alpha values to blocks (needed for both soft and gumbel)
        if self.nas:
            self.set_alpha()

        # encoder
        features = []
        out = x
        for stage_idx, stg in enumerate(self.stages):
            # Process through blocks
            for blk in stg.blocks:
                out = blk(out, mode=mode, temp=temp, hard=hard)
            
            # Store feature BEFORE downsampling for skip connections
            if stage_idx < 3:  # First 3 stages have downsampling
                features.append(out)  # Store before downsampling
            
            # Apply downsampling if present
            if stg.downsample is not None:
                out = stg.downsample(out)
        
        # Store the final stage output (no downsampling)
        features.append(out)

        # decoder
        x = features[-1]
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            skip = features[2-i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = x + skip

        logits = self.head(x)
        
        # Upsample to original input resolution
        input_resolution = (D, H, W)
        if logits.shape[2:] != input_resolution:
            logits = F.interpolate(
                logits,
                size=input_resolution,
                mode="trilinear",
                align_corners=False
            )
        
        return logits
    
    def expected_flops(self, temp=1.0):
        """
        Compute expected FLOPs across all MixedOps and MixedAttentions.
        Returns a scalar tensor for use in loss computation.
        - temp: temperature for computing softmax probabilities
        """
        total_flops = 0.0
        for blk in self._nas_blocks:
            total_flops += blk.ffn.expected_flops(temp=temp)
            total_flops += blk.attn.expected_flops(temp=temp)
        # Convert to tensor on the same device as model parameters
        device = next(self.parameters()).device
        return torch.tensor(total_flops, device=device, dtype=torch.float32)
