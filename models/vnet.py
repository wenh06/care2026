"""
3-D segmentation networks for CARE2026.

Both classes share a common encoder (_SegEncoder3D):
- DualHeadVNet : shared encoder → 2 independent decoders (LA cavity + scar, MRI Tasks 1 & 2)
- VNet         : shared encoder → 1 decoder (CT Task 3 / CPS)

Instance Norm for MRI (domain generalisation), Batch Norm for CT.
Architecture is fully driven by __DEFAULT_CONFIG__ in each class.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils import SizeMixin

from .layers import ConvNormAct, DownBlock3D, UpBlock3D

__all__ = ["DualHeadVNet", "VNet"]


class _SegEncoder3D(nn.Module):
    """Shared encoder for DualHeadVNet and UNet3D.

    stem → ModuleList[DownBlock3D]
    Stores skip feature channels in ``_enc_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        norm: str,
        activation: str,
        input_conv: CFG,
        down_conv: CFG,
    ) -> None:
        super().__init__()
        ic, dc = input_conv, down_conv
        self.stem = ConvNormAct(in_channels, ic.channels, kernel_size=ic.kernel_size, norm=norm, activation=activation)
        enc_in = [ic.channels] + list(dc.channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                DownBlock3D(
                    in_channels=enc_in[i],
                    out_channels=dc.channels[i],
                    kernel_size=dc.kernel_size[i],
                    norm=norm,
                    activation=activation,
                    dropout=dc.dropout[i],
                )
                for i in range(len(dc.channels))
            ]
        )
        self._enc_channels: List[int] = [ic.channels] + list(dc.channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return list of skip tensors [stem_out, down0_out, ..., bottleneck_out]."""
        skips = [self.stem(x)]
        for block in self.down_blocks:
            skips.append(block(skips[-1]))
        return skips


def _make_decoder(
    enc_channels: List[int],
    up_conv: CFG,
    norm: str,
    activation: str,
) -> nn.ModuleList:
    """Build a ModuleList of UpBlock3D for one decoder."""
    blocks = []
    for i in range(len(up_conv.channels)):
        in_ch = enc_channels[-1] if i == 0 else up_conv.channels[i - 1]
        skip_ch = enc_channels[-(i + 2)]
        blocks.append(
            UpBlock3D(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=up_conv.channels[i],
                kernel_size=up_conv.kernel_size[i],
                norm=norm,
                activation=activation,
                dropout=up_conv.dropout[i],
            )
        )
    return nn.ModuleList(blocks)


class DualHeadVNet(nn.Module, SizeMixin):
    """Dual-head V-Net for simultaneous LA cavity + scar segmentation.

    Architecture is fully driven by ``config``; falls back to
    ``__DEFAULT_CONFIG__`` for any key not supplied.

    Parameters
    ----------
    config : CFG, optional
        Architecture overrides (merged on top of ``__DEFAULT_CONFIG__``).

    Encoder channel sizes are derived from the config as:
        enc_channels = [input_conv.channels] + down_conv.channels

    Decoder up-block i uses:
        in_channels  = enc_channels[-1] if i == 0 else up_conv.channels[i-1]
        skip_channels = enc_channels[-(i+2)]
        out_channels = up_conv.channels[i]
    """

    __name__ = "DualHeadVNet"

    __DEFAULT_CONFIG__ = CFG(
        in_channels=1,
        norm="instance",
        activation="mish",
        input_conv=CFG(channels=16, kernel_size=5),
        down_conv=CFG(
            channels=[32, 64, 128, 256],
            kernel_size=[3, 3, 3, 3],
            dropout=[0.0, 0.0, 0.3, 0.3],
        ),
        up_conv=CFG(
            channels=[128, 64, 32, 16],
            kernel_size=[3, 3, 3, 3],
            dropout=[0.0, 0.0, 0.0, 0.0],
        ),
        output_conv=CFG(kernel_size=1),
        heads=CFG(
            la=CFG(out_channels=2),
            scar=CFG(out_channels=2),
        ),
    )

    def __init__(self, config: Optional[CFG] = None) -> None:
        super().__init__()
        self.__config = deepcopy(self.__DEFAULT_CONFIG__)
        if config is not None:
            self.__config.update(deepcopy(config))
        self._check_config()

        norm = self.__config.norm
        act = self.__config.activation

        self.encoder = _SegEncoder3D(
            in_channels=self.__config.in_channels,
            norm=norm,
            activation=act,
            input_conv=self.__config.input_conv,
            down_conv=self.__config.down_conv,
        )

        self.la_up_blocks = _make_decoder(self.encoder._enc_channels, self.__config.up_conv, norm, act)
        self.scar_up_blocks = _make_decoder(self.encoder._enc_channels, self.__config.up_conv, norm, act)

        uc = self.__config.up_conv
        self.la_out = nn.Conv3d(uc.channels[-1], self.__config.heads.la.out_channels, kernel_size=1)
        self.scar_out = nn.Conv3d(uc.channels[-1], self.__config.heads.scar.out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, H, W, D)

        Returns
        -------
        tuple of (la_logits, scar_logits), each shape (B, 2, H, W, D)
        """
        # Encoder: collect skip features
        skips = self.encoder(x)

        # Decoder (LA head)
        out_la = skips[-1]
        for i, block in enumerate(self.la_up_blocks):
            out_la = block(out_la, skips[-(i + 2)])

        # Decoder (scar head)
        out_scar = skips[-1]
        for i, block in enumerate(self.scar_up_blocks):
            out_scar = block(out_scar, skips[-(i + 2)])

        return self.la_out(out_la), self.scar_out(out_scar)

    # ------------------------------------------------------------------
    def _check_config(self) -> None:
        dc = self.__config.down_conv
        uc = self.__config.up_conv
        assert len(dc.channels) == len(dc.kernel_size) == len(dc.dropout), (
            "down_conv: channels, kernel_size, dropout must have equal length"
        )
        assert len(uc.channels) == len(uc.kernel_size) == len(uc.dropout), (
            "up_conv: channels, kernel_size, dropout must have equal length"
        )
        assert len(dc.channels) == len(uc.channels), "down_conv and up_conv must have the same depth"

    @property
    def config(self) -> CFG:
        return self.__config


class VNet(nn.Module, SizeMixin):
    """Standard single-head V-Net for multi-class segmentation (CT Task 3 / CPS).

    Used as both model1 and model2 in Cross Pseudo Supervision (Chen et al., 2021).
    Architecture is fully driven by ``config``; falls back to
    ``__DEFAULT_CONFIG__`` for any key not supplied.

    Parameters
    ----------
    config : CFG, optional
        Architecture overrides (merged on top of ``__DEFAULT_CONFIG__``).
    """

    __name__ = "VNet"

    __DEFAULT_CONFIG__ = CFG(
        in_channels=1,
        num_classes=4,
        norm="batch",
        activation="relu",
        input_conv=CFG(channels=16, kernel_size=3),
        down_conv=CFG(
            channels=[32, 64, 128, 256],
            kernel_size=[3, 3, 3, 3],
            dropout=[0.0, 0.0, 0.0, 0.2],
        ),
        up_conv=CFG(
            channels=[128, 64, 32, 16],
            kernel_size=[3, 3, 3, 3],
            dropout=[0.0, 0.0, 0.0, 0.0],
        ),
        output_conv=CFG(kernel_size=1),
    )

    def __init__(self, config: Optional[CFG] = None) -> None:
        super().__init__()
        self.__config = deepcopy(self.__DEFAULT_CONFIG__)
        if config is not None:
            self.__config.update(deepcopy(config))
        self._check_config()

        norm = self.__config.norm
        act = self.__config.activation

        self.encoder = _SegEncoder3D(
            in_channels=self.__config.in_channels,
            norm=norm,
            activation=act,
            input_conv=self.__config.input_conv,
            down_conv=self.__config.down_conv,
        )
        self.up_blocks = _make_decoder(self.encoder._enc_channels, self.__config.up_conv, norm, act)
        self.out_conv = nn.Conv3d(self.__config.up_conv.channels[-1], self.__config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, H, W, D)

        Returns
        -------
        torch.Tensor, shape (B, num_classes, H, W, D)
        """
        skips = self.encoder(x)
        out = skips[-1]
        for i, block in enumerate(self.up_blocks):
            out = block(out, skips[-(i + 2)])
        return self.out_conv(out)

    def _check_config(self) -> None:
        dc = self.__config.down_conv
        uc = self.__config.up_conv
        assert len(dc.channels) == len(dc.kernel_size) == len(dc.dropout), (
            "down_conv: channels, kernel_size, dropout must have equal length"
        )
        assert len(uc.channels) == len(uc.kernel_size) == len(uc.dropout), (
            "up_conv: channels, kernel_size, dropout must have equal length"
        )
        assert len(dc.channels) == len(uc.channels), "down_conv and up_conv must have the same depth"

    @property
    def config(self) -> CFG:
        return self.__config

