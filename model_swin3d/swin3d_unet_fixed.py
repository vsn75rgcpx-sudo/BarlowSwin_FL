"""
swin3d_unet_fixed.py
--------------------
Final fixed-architecture SwinUNet3D for retraining/inference.
Modified to fix Skip Connection dimension mismatch in U-Net.
"""

import torch
import torch.nn as nn
import json
from .layers import PatchMerging3D, PatchExpanding3D, WindowAttention3D, window_partition, window_reverse
from .swin3d_block import LayerNormChannel, DropPath
from .nas_ops import build_op_3d


# ------------------------------------------------------------
# 1. Fixed FFN Block
# ------------------------------------------------------------
class FixedFFN3D(nn.Module):
    def __init__(self, chosen_op):
        super().__init__()
        self.op = chosen_op

    def forward(self, x):
        return self.op(x)


# ------------------------------------------------------------
# 2. Swin3DBlock_Fixed
# ------------------------------------------------------------
class Swin3DBlock_Fixed(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift_size,
            ffn_op,
            drop_path
    ):
        super().__init__()
        self.dim = dim
        self.D, self.H, self.W = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = LayerNormChannel(dim)
        self.norm2 = LayerNormChannel(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.drop_path = DropPath(drop_path)
        self.ffn = FixedFFN3D(ffn_op)
        self.register_buffer("attn_mask", None, persistent=False)

    def get_mask(self, x):
        if self.attn_mask is not None:
            return self.attn_mask
        from .layers import compute_attn_mask
        mask = compute_attn_mask(
            self.D, self.H, self.W,
            self.window_size, self.shift_size,
            x.device
        )
        self.attn_mask = mask
        return mask

    def forward(self, x):
        B, C, D, H, W = x.shape
        shortcut = x
        x = self.norm1(x)
        sd, sh, sw = self.shift_size
        if sd or sh or sw:
            shifted_x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(2, 3, 4))
        else:
            shifted_x = x

        wd, wh, ww = self.window_size
        N = wd * wh * ww
        windows = window_partition(shifted_x, self.window_size)
        windows = windows.view(-1, N, C)
        mask = self.get_mask(x)
        attn_windows = self.attn(windows, mask)
        attn_windows = attn_windows.view(-1, wd, wh, ww, C)
        shifted_x = window_reverse(attn_windows, self.window_size, B, D, H, W)

        if sd or sh or sw:
            x_attn = torch.roll(shifted_x, shifts=(sd, sh, sw), dims=(2, 3, 4))
        else:
            x_attn = shifted_x
        x = shortcut + self.drop_path(x_attn)

        shortcut2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut2 + self.drop_path(x)
        return x


# ------------------------------------------------------------
# 3. SwinStage_Fixed
# ------------------------------------------------------------
class SwinStage_Fixed(nn.Module):
    def __init__(
            self,
            dim,
            resolution,
            depth,
            num_heads,
            window_size,
            drop_path_list,
            ffn_ops,
            keep_list=None,
            downsample=True
    ):
        super().__init__()
        D, H, W = resolution
        self.blocks = nn.ModuleList()

        if keep_list is None:
            keep_list = [True] * depth

        for i in range(depth):
            # If gate_keep is False, we skip this block (Identity)
            if not keep_list[i]:
                self.blocks.append(nn.Identity())
                continue

            shift = (
                0 if i % 2 == 0 else window_size[0] // 2,
                0 if i % 2 == 0 else window_size[1] // 2,
                0 if i % 2 == 0 else window_size[2] // 2,
            )

            blk = Swin3DBlock_Fixed(
                dim=dim,
                input_resolution=(D, H, W),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift,
                ffn_op=ffn_ops[i],
                drop_path=drop_path_list[i]
            )
            self.blocks.append(blk)

        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):
        # NOTE: This forward includes downsample.
        # For U-Net skip connections, we need features BEFORE downsample.
        # So we won't strictly use this forward in SwinUNet3D_Fixed,
        # but iterate blocks manually.
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


# ------------------------------------------------------------
# 4. PatchEmbed3D
# ------------------------------------------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


# ------------------------------------------------------------
# 5. Final model: SwinUNet3D_Fixed
# ------------------------------------------------------------
class SwinUNet3D_Fixed(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            dims,
            depths,
            arch_json,
            window_size=(2, 7, 7)
    ):
        super().__init__()

        # Load architecture
        if isinstance(arch_json, dict):
            # 如果传进来的是字典，直接用
            config = arch_json
        else:
            # 如果传进来的是路径(str)，则读取文件
            with open(arch_json, "r") as f:
                config = json.load(f)

        def map_op_name(class_name):
            name_lower = class_name.lower()
            if "conv3" in name_lower:
                return "conv3"
            elif "conv1" in name_lower:
                return "conv1"
            elif "conv5" in name_lower:
                return "conv5"
            elif "dw" in name_lower or "depthwise" in name_lower:
                return "dw"
            elif "identity" in name_lower:
                return "identity"
            else:
                raise ValueError(f"Unknown operator: {class_name}")

        self.chosen_operators = []
        self.block_keeps = []

        # [修复点]: 这里之前写成了 if "stages" in arch，现改为 config
        if "stages" in config:
            stages_sorted = sorted(config["stages"], key=lambda x: x["stage_id"])
            for stage in stages_sorted:
                stage_ops = []
                stage_keeps = []
                for blk in stage["blocks"]:
                    op_name = blk["ffn_choice"]
                    mapped_name = map_op_name(op_name)
                    stage_ops.append(mapped_name)
                    stage_keeps.append(blk.get("gate_keep", True))

                self.chosen_operators.append(stage_ops)
                self.block_keeps.append(stage_keeps)
        elif "ops" in config: # [修复点]: 同上，改为 config
            flat_ops = config["ops"] # [修复点]: 同上，改为 config
            ptr = 0
            for d in depths:
                s_ops = []
                s_keeps = []
                for _ in range(d):
                    if ptr < len(flat_ops):
                        op_name = flat_ops[ptr]["op"]
                        s_ops.append(map_op_name(op_name))
                        ptr += 1
                    else:
                        s_ops.append("conv3")
                    s_keeps.append(True)
                self.chosen_operators.append(s_ops)
                self.block_keeps.append(s_keeps)
        else:
            # 调试信息
            raise ValueError(f"Invalid JSON format. Keys found: {list(config.keys())}")

        self.patch_embed = PatchEmbed3D(in_channels, dims[0], patch_size=2)
        self.depths = depths
        self.dims = dims
        self.window_size = window_size

        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = torch.linspace(0, 0.1, total_blocks).tolist()
        dpr_ptr = 0

        for stage_id in range(4):
            dim = dims[stage_id]
            depth = depths[stage_id]
            heads = dim // 32

            ops_names = self.chosen_operators[stage_id]
            keeps = self.block_keeps[stage_id]

            real_ops = []
            for op_name in ops_names:
                real_ops.append(build_op_3d(op_name, dim, dim))

            drop_list = dpr[dpr_ptr: dpr_ptr + depth]
            dpr_ptr += depth

            stage = SwinStage_Fixed(
                dim=dim,
                resolution=(1, 1, 1),
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                drop_path_list=drop_list,
                ffn_ops=real_ops,
                keep_list=keeps,
                downsample=(stage_id < 3)
            )
            self.stages.append(stage)

        self.up_blocks = nn.ModuleList()
        for i in range(3, 0, -1):
            up = nn.Sequential(
                PatchExpanding3D(dims[i], dims[i - 1]),
                nn.Conv3d(dims[i - 1], dims[i - 1], kernel_size=3, padding=1),
                nn.GELU()
            )
            self.up_blocks.append(up)

        self.head = nn.Conv3d(dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        input_resolution = (D, H, W)
        x = self.patch_embed(x)
        _, C1, D1, H1, W1 = x.shape

        res = [(D1, H1, W1)]
        for _ in range(3):
            d, h, w = res[-1]
            res.append((d // 2, h // 2, w // 2))

        # Update dynamic resolution in blocks
        for stg, r in zip(self.stages, res):
            for blk in stg.blocks:
                if not isinstance(blk, nn.Identity):
                    blk.D, blk.H, blk.W = r

        # Encoder forward (Manually iterating to catch skip connections correctly)
        feats = []
        out = x
        for stage_idx, stg in enumerate(self.stages):
            # 1. Run all blocks in the stage
            for blk in stg.blocks:
                out = blk(out)

            # 2. Store feature map for Skip Connection (BEFORE downsampling)
            if stage_idx < 3:
                feats.append(out)

            # 3. Apply Downsample (if exists)
            if stg.downsample is not None:
                out = stg.downsample(out)

        # Last stage output
        feats.append(out)

        # Decoder forward
        x = feats[-1]
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            skip = feats[2 - i]

            # Interpolate if shapes don't match (e.g. odd dimensions)
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

            x = x + skip

        logits = self.head(x)

        # Final upsample to input resolution
        if logits.shape[2:] != input_resolution:
            logits = torch.nn.functional.interpolate(
                logits,
                size=input_resolution,
                mode="trilinear",
                align_corners=False
            )

        return logits

    def arch_parameters(self):
        return []