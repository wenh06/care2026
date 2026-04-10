"""
Shared 3-D convolutional building blocks for VNet and UNet3D.

Conventions:
- All tensors: (B, C, H, W, D)
- norm: "batch" (BatchNorm3d) or "instance" (InstanceNorm3d)
- activation: "relu", "leaky_relu", "mish", "elu", "prelu"
"""

from typing import Literal, Optional, Union, List

import torch
import torch.nn as nn

__all__ = [
    "ConvNormAct",
    "ResBlock3D",
    "DownBlock3D",
    "UpBlock3D",
    "NestedUpBlock3D",
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
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        )
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
        self.block = ResBlock3D(in_channels, out_channels, kernel_size=kernel_size, norm=norm, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


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

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, h, w, d) — deeper-path features
        skips : list of tensors to concatenate along channel dim
        """
        x_up = self.up_act(self.up_bn(self.up(x)))
        skip_cat = torch.cat(skips, dim=1)
        # Align spatial dimensions (odd input sizes may cause ±1 mismatch)
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

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad x if spatial sizes don't match exactly (can happen with odd input dims)
        if x.shape[2:] != skip.shape[2:]:
            diffs = [s - x.shape[i + 2] for i, s in enumerate(skip.shape[2:])]
            pad = []
            for d in reversed(diffs):
                p = d // 2
                pad.extend([p, d - p])
            x = nn.functional.pad(x, pad)
        return self.block(torch.cat([x, skip], dim=1))
