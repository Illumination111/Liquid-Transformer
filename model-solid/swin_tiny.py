"""Swin Transformer Tiny adapted to native 32x32 CIFAR images."""

from __future__ import annotations

import torch
from common import DropPath, Mlp
from torch import nn


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size**2, channels)


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    batch = int(windows.shape[0] / (height * width / window_size**2))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        table_size = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        ).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index", relative_coords.sum(-1), persistent=False
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens
            )
            attn = (attn + mask.unsqueeze(0).unsqueeze(2)).view(
                -1, self.num_heads, tokens, tokens
            )
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: int,
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = min(window_size, input_resolution)
        self.shift_size = 0 if input_resolution <= window_size else shift_size
        if input_resolution % self.window_size:
            raise ValueError("input resolution must be divisible by window size")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, self.window_size, num_heads, True, attn_drop, drop
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path)
        self.register_buffer("attn_mask", self._make_mask(), persistent=False)

    def _make_mask(self) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        size = self.input_resolution
        mask = torch.zeros((1, size, size, 1))
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for height_slice in slices:
            for width_slice in slices:
                mask[:, height_slice, width_slice, :] = count
                count += 1
        mask_windows = window_partition(mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != size * size:
            raise ValueError(f"expected {size * size} tokens, got {tokens}")
        shortcut = x
        x = self.norm1(x).view(batch, size, size, channels)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = window_partition(x, self.window_size)
        x = self.attn(x, self.attn_mask)
        x = window_reverse(x, self.window_size, size, size)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.reshape(batch, tokens, channels)
        x = shortcut + self.drop_path1(x)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: int, dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = self.input_resolution
        batch, tokens, channels = x.shape
        if size % 2 or tokens != size * size:
            raise ValueError("PatchMerging requires an even square token grid")
        x = x.view(batch, size, size, channels)
        x = torch.cat(
            (x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]),
            dim=-1,
        ).view(batch, -1, 4 * channels)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_paths: list[float],
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim,
                    input_resolution,
                    num_heads,
                    window_size,
                    0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio,
                    drop,
                    attn_drop,
                    drop_paths[index],
                )
                for index in range(depth)
            ]
        )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.downsample(x) if self.downsample is not None else x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 4, embed_dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class SwinTiny(nn.Module):
    """Swin-Tiny with the canonical [2, 2, 6, 2] hierarchy.

    CIFAR's 32x32 images use a 4x4 patch embedding and 4x4 attention windows,
    producing stage resolutions 8, 4, 2 and 1.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 100,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        total_depth = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_depth).tolist()
        resolution = image_size // patch_size
        offset = 0
        layers = []
        for index, depth in enumerate(depths):
            layers.append(
                BasicLayer(
                    embed_dim * 2**index,
                    resolution,
                    depth,
                    num_heads[index],
                    window_size,
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    drop_paths[offset : offset + depth],
                    index < len(depths) - 1,
                )
            )
            offset += depth
            if index < len(depths) - 1:
                resolution //= 2
        self.layers = nn.ModuleList(layers)
        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def swin_tiny(num_classes: int = 100, **kwargs: object) -> SwinTiny:
    return SwinTiny(num_classes=num_classes, **kwargs)
