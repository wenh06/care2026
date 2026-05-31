"""
Shared 3-D convolutional building blocks for VNet and UNet3D.

Conventions:
- All tensors: (B, C, H, W, D)
- norm: "batch" (BatchNorm3d) or "instance" (InstanceNorm3d)
- activation: "relu", "leaky_relu", "mish", "elu", "prelu"
"""

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "ConvNormAct",
    "ResBlock3D",
    "DownBlock3D",
    "ECAGate3D",
    "WindowedMHSA3D",
    "BottleneckTransformer3D",
    "NestedUpBlock3D",
    "UpBlock3D",
]

NormType = Literal["batch", "instance", "none"]
ActType = Literal["relu", "leaky_relu", "mish", "elu", "prelu"]


def _get_norm(norm: NormType, channels: int) -> Optional[nn.Module]:
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    elif norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    elif norm == "none":
        return None
    else:
        raise ValueError(f"Unknown norm type: {norm}")


def _get_act(activation: ActType) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
        "mish": nn.Mish(inplace=True),
        "elu": nn.ELU(inplace=True),
        "prelu": nn.PReLU(),
    }
    if activation not in mapping:
        raise ValueError(f"Unknown activation: {activation}")
    return mapping[activation]


class ConvNormAct(nn.Sequential):
    """Conv3d → Norm → Activation block.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int, default 3
    stride : int, default 1
    padding : int or str, default "same"
    norm : NormType, default "instance"
    activation : ActType, default "mish"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = "same",
        norm: NormType = "instance",
        activation: ActType = "mish",
    ) -> None:
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=(norm == "none"),
            ),
        ]
        norm_layer = _get_norm(norm, out_channels)
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(_get_act(activation))
        super().__init__(*layers)


class ResBlock3D(nn.Module):
    """Two ConvNormAct layers with a skip connection.

    If in_channels != out_channels, a 1×1×1 projection is used.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int, default 3
    norm : NormType, default "instance"
    activation : ActType, default "mish"
    dropout : float, default 0.0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: NormType = "instance",
        activation: ActType = "mish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, norm=norm, activation=activation)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, norm=norm, activation=activation)
        self.skip = nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.dropout(self.conv2(self.conv1(x)))
        return out + identity


class DownBlock3D(nn.Module):
    """Strided-conv downsample (×2) + residual block.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int, default 3
    norm : NormType, default "instance"
    activation : ActType, default "mish"
    dropout : float, default 0.0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: NormType = "instance",
        activation: ActType = "mish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.down = nn.Conv3d(in_channels, in_channels, 2, stride=2, padding=0, bias=False)
        self.block = ResBlock3D(
            in_channels, out_channels, kernel_size=kernel_size, norm=norm, activation=activation, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(dim < 2 for dim in x.shape[2:]):
            pad = []
            for dim in reversed(x.shape[2:]):
                pad.extend([0, max(0, 2 - dim)])
            x = nn.functional.pad(x, pad)
        return self.block(self.down(x))


class ECAGate3D(nn.Module):
    """Efficient Channel Attention gate for 3-D feature maps (ECA-Net).

    Computes per-channel attention weights via global-average-pooling
    followed by a 1-D convolution of kernel size *k* (auto-sized from
    ``channels``). Adds negligible parameters compared to SE blocks.

    Reference
    ---------
    Wang et al., ECA-Net: Efficient channel attention for deep
    convolutional neural networks. CVPR 2020.

    Parameters
    ----------
    channels : int
        Number of input / output channels.
    gamma : int, default 2
    b : int, default 1
        ECA hyper-parameters for automatic kernel sizing:
        k = ⌈log2(C) / γ + b / γ⌉, rounded up to nearest odd number.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        import math

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate *x* with channel-wise attention weights."""
        # (B, C, H, W, D) → (B, C, 1, 1, 1)
        y = self.gap(x)
        # (B, C, 1, 1, 1) → (B, 1, C) for 1-D conv
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        # (B, 1, C) → (B, C, 1, 1, 1)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)


class WindowedMHSA3D(nn.Module):
    """Multi-head self-attention within non-overlapping 3-D windows.

    The input volume is partitioned into windows of ``window_size``
    (zero-padded if necessary). Full MHSA is applied within each
    window independently, then the results are reassembled.

    With the typical VNet bottleneck shapes (32×32×5 at Stage 2,
    18×18×5 at Stage 1) and window_size=(8,8,5) the attention matrix
    is only 320×320 per head—well within memory budget.

    Parameters
    ----------
    channels : int
    num_heads : int, default 8
    window_size : (wH, wW, wD), default (8, 8, 5)
    dropout : float, default 0.0
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (8, 8, 5),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    @staticmethod
    def _partition(x: torch.Tensor, ws: Tuple[int, int, int]) -> torch.Tensor:
        """(B, H, W, D, C) → (nW·B, wH·wW·wD, C)."""
        B, H, W, D, C = x.shape
        wH, wW, wD = ws
        x = x.view(B, H // wH, wH, W // wW, wW, D // wD, wD, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wH * wW * wD, C)

    @staticmethod
    def _unpartition(windows: torch.Tensor, ws: Tuple[int, int, int], shape: Tuple[int, int, int]) -> torch.Tensor:
        """(nW·B, wH·wW·wD, C) → (B, H, W, D, C)."""
        H, W, D = shape
        wH, wW, wD = ws
        C = windows.shape[-1]
        B = windows.shape[0] // ((H // wH) * (W // wW) * (D // wD))
        x = windows.view(B, H // wH, W // wW, D // wD, wH, wW, wD, C)
        return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W, D) → (B, C, H, W, D)."""
        _, C, H, W, D = x.shape
        wH, wW, wD = self.window_size

        # Pad so each spatial dim is divisible by its window size
        pH = (wH - H % wH) % wH
        pW = (wW - W % wW) % wW
        pD = (wD - D % wD) % wD
        if pH or pW or pD:
            x = torch.nn.functional.pad(x, (0, pD, 0, pW, 0, pH))
        _, _, Hp, Wp, Dp = x.shape

        # (B, C, H, W, D) → (B, H, W, D, C)
        x_in = x.permute(0, 2, 3, 4, 1)
        shortcut = x_in
        x_in = self.norm(x_in)

        wins = self._partition(x_in, (wH, wW, wD))  # (nW·B, N, C)
        N = wH * wW * wD
        qkv = self.qkv(wins).reshape(-1, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(-1, N, C)
        out = self.proj_drop(self.proj(out))

        x_out = self._unpartition(out, (wH, wW, wD), (Hp, Wp, Dp))
        x_out = x_out + shortcut

        if pH or pW or pD:
            x_out = x_out[:, :H, :W, :D, :]

        return x_out.permute(0, 4, 1, 2, 3).contiguous()


class BottleneckTransformer3D(nn.Module):
    """Windowed self-attention + feed-forward block for the VNet bottleneck.

    Stacks one :class:`WindowedMHSA3D` and one position-wise FFN (both with
    pre-norm and residual connection), replacing or augmenting the deepest
    convolutional block.

    Parameters
    ----------
    channels : int
    num_heads : int, default 8
    window_size : (wH, wW, wD), default (8, 8, 5)
    mlp_ratio : float, default 4.0
    dropout : float, default 0.0
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (8, 8, 5),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = WindowedMHSA3D(channels, num_heads, window_size, dropout)
        mlp_dim = int(channels * mlp_ratio)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W, D) → (B, C, H, W, D)."""
        x = self.attn(x)
        # FFN in channel-last layout
        x_in = x.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        x_in = x_in + self.ffn(self.norm(x_in))
        return x_in.permute(0, 4, 1, 2, 3).contiguous()


class NestedUpBlock3D(nn.Module):
    """Upsample (×2) + multi-skip concat + residual block for Nested UNet++.

    Supports an arbitrary number of skip tensors: all skips are channel-
    projected to ``out_channels // 2`` via a 1×1 conv, then concatenated with
    the upsampled ``x`` (also projected to ``out_channels // 2``).

    Parameters
    ----------
    in_channels : int
        Channels from the deeper path.
    total_skip_channels : int
        Sum of channels of **all** skip tensors that will be concatenated.
    out_channels : int
        Output channels (must be even).
    kernel_size : int, default 3
    norm : NormType, default "instance"
    activation : ActType, default "mish"
    dropout : float, default 0.0
    """

    def __init__(
        self,
        in_channels: int,
        total_skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: NormType = "instance",
        activation: ActType = "mish",
        dropout: float = 0.0,
        use_eca: bool = False,
    ) -> None:
        super().__init__()
        half = out_channels // 2
        # Upsample + project to out_channels // 2
        self.up = nn.ConvTranspose3d(in_channels, half, kernel_size=2, stride=2, bias=False)
        self.up_bn = _get_norm(norm, half) or nn.Identity()
        self.up_act = _get_act(activation)
        # Compress all concatenated skips to out_channels // 2 (1×1 if already right size)
        if total_skip_channels == half:
            self.skip_proj: nn.Module = nn.Identity()
        else:
            skip_norm = _get_norm(norm, half) or nn.Identity()
            self.skip_proj = nn.Sequential(
                nn.Conv3d(total_skip_channels, half, kernel_size=1, bias=False),
                skip_norm,
                _get_act(activation),
            )
        # Residual block on [up_out ; skip_proj_out]
        self.block = ResBlock3D(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.eca: nn.Module = ECAGate3D(total_skip_channels) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, h, w, d) — deeper-path features
        skips : list of tensors to concatenate along channel dim
        """
        x_up = self.up_act(self.up_bn(self.up(x)))
        skip_cat = torch.cat(skips, dim=1)
        skip_cat = self.eca(skip_cat)
        # Align spatial dimensions
        if x_up.shape[2:] != skip_cat.shape[2:]:
            diffs = [s - x_up.shape[i + 2] for i, s in enumerate(skip_cat.shape[2:])]
            pad = []
            for d in reversed(diffs):
                p = d // 2
                pad.extend([p, d - p])
            x_up = nn.functional.pad(x_up, pad)
        return self.block(torch.cat([x_up, self.skip_proj(skip_cat)], dim=1))


class UpBlock3D(nn.Module):
    """Transposed-conv upsample (×2) + skip concat + residual block.

    Parameters
    ----------
    in_channels : int
        Channels from the deeper path.
    skip_channels : int
        Channels from the skip connection.
    out_channels : int
    kernel_size : int, default 3
    norm : NormType, default "instance"
    activation : ActType, default "mish"
    dropout : float, default 0.0
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: NormType = "instance",
        activation: ActType = "mish",
        dropout: float = 0.0,
        use_eca: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2, bias=False)
        self.block = ResBlock3D(
            in_channels + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        self.eca: nn.Module = ECAGate3D(skip_channels) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.eca(skip)
        # pad x if spatial sizes don't match exactly (can happen with odd input dims)
        if x.shape[2:] != skip.shape[2:]:
            diffs = [s - x.shape[i + 2] for i, s in enumerate(skip.shape[2:])]
            pad = []
            for d in reversed(diffs):
                p = d // 2
                pad.extend([p, d - p])
            x = nn.functional.pad(x, pad)
        return self.block(torch.cat([x, skip], dim=1))
