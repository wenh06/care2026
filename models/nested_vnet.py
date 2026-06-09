"""
Nested V-Net (UNet++ / VNet++) for 3-D segmentation with deep supervision.

Dense skip connections: each decoder node receives features from
ALL shallower encoder levels at the same resolution, plus the
upsampled output from the node one level deeper.

Reference: Zhou et al., "UNet++: A Nested U-Net Architecture for
Medical Image Segmentation", DLMIA 2018.
"""

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils import SizeMixin

from .layers import BottleneckTransformer3D, NestedUpBlock3D
from .vnet import _SegEncoder3D, _upsample_to

__all__ = ["NestedVNet"]


class NestedVNet(nn.Module, SizeMixin):
    """UNet++ with dense skip connections and deep supervision.

    Architecture layout for a 4-level network::

        X⁰⁰  →  X⁰¹  →  X⁰²  →  X⁰³  →  X⁰⁴   (full res,  DS head)
            X¹⁰  →  X¹¹  →  X¹²  →  X¹³          (1/2 res, DS head)
                X²⁰  →  X²¹  →  X²²               (1/4 res, DS head)
                    X³⁰  →  X³¹                    (1/8 res, DS head)
                        X⁴⁰                         (bottleneck)

    * Column 0: encoder outputs (Xⁱ⁰ = enc[i]).
    * Column j>0: upsampled Xⁱ⁺¹⁽ʲ⁻¹⁾ + all prior nodes in row i.
    * Deep supervision heads at the rightmost node of each row.
    """

    __name__ = "NestedVNet"

    __DEFAULT_CONFIG__ = CFG(
        in_channels=1,
        num_classes=4,
        norm="batch",
        activation="relu",
        use_eca_skip=False,
        deep_supervision=True,
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
        bottleneck_transformer=None,
    )

    def __init__(self, config: Optional[CFG] = None) -> None:
        super().__init__()
        self.__config = deepcopy(self.__DEFAULT_CONFIG__)
        if config is not None:
            self.__config.update(deepcopy(config))
        self._check_config()

        norm = self.__config.norm
        act = self.__config.activation
        use_eca = bool(self.__config.get("use_eca_skip", False))
        enc_cfg = self.__config.down_conv
        up_cfg = self.__config.up_conv
        num_levels = len(enc_cfg.channels)  # e.g. 4

        # ---- shared encoder -----------------------------------------------------
        bt_cfg = self.__config.get("bottleneck_transformer", None)
        bt = None
        if bt_cfg:
            bt = BottleneckTransformer3D(
                channels=enc_cfg.channels[-1],
                num_heads=bt_cfg.get("num_heads", 8),
                window_size=tuple(bt_cfg.get("window_size", [8, 8, 5])),
                mlp_ratio=float(bt_cfg.get("mlp_ratio", 4.0)),
                dropout=float(bt_cfg.get("dropout", 0.0)),
            )
        self.encoder = _SegEncoder3D(
            in_channels=self.__config.in_channels,
            norm=norm,
            activation=act,
            input_conv=self.__config.input_conv,
            down_conv=enc_cfg,
            bottleneck_transformer=bt,
        )
        enc_ch = self.encoder._enc_channels  # [stem, d0, d1, ..., bottleneck]
        # decoder channels: out_chan[i] for row i (0=shallowest)
        out_chan = list(up_cfg.channels[::-1])  # e.g. [16, 32, 64, 128]

        # ---- UNet++ decoder nodes -----------------------------------------------
        # x_blocks[i] stores nn.ModuleList for row i (i from shallow to deep).
        # Only rows 0..(num_levels-1) have decoder nodes (bottleneck row has none).
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_levels):
            row = nn.ModuleList()
            for j in range(1, num_levels - i + 1):
                # in_channels: from node (i+1, j-1)
                in_ch = enc_ch[i + 1] if j == 1 else out_chan[i + 1]
                # skip channels: enc[i] + (j-1) * out_chan[i]
                total_skip = enc_ch[i] + (j - 1) * out_chan[i]
                block = NestedUpBlock3D(
                    in_channels=in_ch,
                    total_skip_channels=total_skip,
                    out_channels=out_chan[i],
                    kernel_size=int(up_cfg.kernel_size[num_levels - 1 - i]),
                    norm=norm,
                    activation=act,
                    dropout=float(up_cfg.dropout[num_levels - 1 - i]),
                    use_eca=use_eca,
                )
                row.append(block)
            self.decoder_blocks.append(row)

        # ---- deep supervision heads ---------------------------------------------
        # One per decoder row (skip bottleneck row)
        self.ds_heads = nn.ModuleList(
            [nn.Conv3d(out_chan[i], self.__config.num_classes, kernel_size=1) for i in range(num_levels)]
        )
        self.do_ds = bool(self.__config.get("deep_supervision", True))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Returns
        -------
        list of tensors at each decoder resolution, ordered coarse → fine.
        Use ``outputs[-1]`` for full-resolution logits.
        """
        enc = self.encoder(x)  # [stem, d0, d1, d2, bottleneck]; 5 tensors
        num_levels = len(enc) - 1  # e.g. 4

        # x_blocks[i][j] stores the tensor at row i, column j
        # Initialize column 0 from encoder
        x_blocks: List[List[Optional[torch.Tensor]]] = [[enc[i]] + [None] * (num_levels - i) for i in range(num_levels + 1)]

        # Compute decoder nodes column by column
        for j in range(1, num_levels + 1):  # columns 1..num_levels
            for i in range(num_levels - j + 1):  # rows that have this column
                up = _upsample_to(x_blocks[i + 1][j - 1], x_blocks[i][0])
                skips = [x_blocks[i][k] for k in range(j)]  # columns 0..j-1
                x_blocks[i][j] = self.decoder_blocks[i][j - 1](up, skips)

        # Deep supervision: rightmost node of each decoder row
        # Order: coarsest → finest (consistent with nnUNet convention)
        ds = [self.ds_heads[i](x_blocks[i][num_levels - i]) for i in range(num_levels - 1, -1, -1)]
        if self.do_ds:
            return ds
        return [ds[-1]]

    def _check_config(self) -> None:
        dc = self.__config.down_conv
        uc = self.__config.up_conv
        assert (
            len(dc.channels) == len(dc.kernel_size) == len(dc.dropout)
        ), "down_conv: channels, kernel_size, dropout must have equal length"
        assert len(dc.channels) == len(uc.channels), "down_conv and up_conv must have the same depth"

    @property
    def config(self) -> CFG:
        return self.__config
