"""
Nested (UNet++ style) 3D VNet for volumetric medical image segmentation.

DualHeadNestedVNet shares the same _SegEncoder3D as DualHeadVNet but replaces
each decoder path with a nested (UNet++) decoder that propagates dense skip
connections across all encoder levels, enabling deep supervision.

References
----------
Zhou et al., UNet++: A Nested U-Net Architecture for Medical Image Segmentation
https://arxiv.org/abs/1807.10165
"""

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils import SizeMixin

from .layers import NestedUpBlock3D
from .vnet import _SegEncoder3D

__all__ = ["DualHeadNestedVNet"]


class _NestedDecoder(nn.Module):
    """UNet++ decoder for one segmentation head.

    Builds a triangular grid of decoder nodes ``tc[row][col]``:

    - ``tc[row][0]``  : encoder output at depth *row* (passed in as ``skips``)
    - ``tc[row][col]`` (col > 0): output of the nested up-block at (row, col)

    The skip connection for node ``(idx, i+1)`` is the concatenation of all
    earlier nodes at the same spatial resolution::

        skip = cat([ tc[j][j - (idx - i - 1)]  for j in range(idx-i-1, idx) ])

    Parameters
    ----------
    enc_channels : list[int]
        Channel counts ``[stem_ch, down0_ch, ..., bottleneck_ch]``
        (length = n_levels + 1).
    up_conv : CFG
        ``channels`` / ``kernel_size`` / ``dropout`` lists of length *n_levels*.
        ``channels`` should be in *descending* order, e.g. [128, 64, 32, 16].
    norm : str
    activation : str
    num_classes : int
    deep_supervision : bool
        If True, return one logit tensor per decoder level (coarse→fine).
        If False, return only the finest-resolution logit tensor.
    """

    def __init__(
        self,
        enc_channels: List[int],
        up_conv: CFG,
        norm: str,
        activation: str,
        num_classes: int,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        n_levels = len(enc_channels) - 1
        up_ch = list(up_conv.channels)  # e.g. [128, 64, 32, 16]

        # ------------------------------------------------------------------
        # Precompute node channel table
        #   node_ch[row][col] = number of channels at grid position (row, col)
        #   row=0..n_levels, col=0..row
        # ------------------------------------------------------------------
        node_ch: List[List[Optional[int]]] = [[None] * (row + 1) for row in range(n_levels + 1)]
        for row in range(n_levels + 1):
            node_ch[row][0] = enc_channels[row]
        # up_tr[lv] processes rows at depth lv+1; its channel slice is up_ch[-(1+lv):]
        for lv in range(n_levels):
            channels_slice = up_ch[-(1 + lv) :]
            idx = lv + 1
            for i, ch in enumerate(channels_slice):
                node_ch[idx][i + 1] = ch

        # ------------------------------------------------------------------
        # Build up_blocks[lv][i]  (indexed as: lv = encoder depth 0..n_levels-1)
        # ------------------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        for lv in range(n_levels):
            idx = lv + 1
            channels_slice = up_ch[-(1 + lv) :]  # channels for this column
            lv_blocks: List[nn.Module] = []
            in_ch = enc_channels[idx]  # input to first block in this column
            for i, out_ch in enumerate(channels_slice):
                # Skip connection: cat over all nodes at same spatial scale
                skip_ch = sum(
                    node_ch[j][j - (idx - i - 1)]  # type: ignore[index]
                    for j in range(idx - i - 1, idx)
                )
                ks_idx = n_levels - 1 - lv + i  # maps (lv, i) → up_conv array index
                lv_blocks.append(
                    NestedUpBlock3D(
                        in_channels=in_ch,
                        total_skip_channels=skip_ch,
                        out_channels=out_ch,
                        kernel_size=up_conv.kernel_size[ks_idx],
                        norm=norm,
                        activation=activation,
                        dropout=up_conv.dropout[ks_idx],
                    )
                )
                in_ch = out_ch
            self.up_blocks.append(nn.ModuleList(lv_blocks))

        # ------------------------------------------------------------------
        # Output convolutions (one per level for deep supervision)
        # ------------------------------------------------------------------
        terminal_ch = up_ch[-1]  # all terminal nodes output this many channels
        self.out_convs = nn.ModuleList(
            [nn.Conv3d(terminal_ch, num_classes, kernel_size=1) for _ in range(n_levels)]
        )
        self.n_levels = n_levels
        self.deep_supervision = deep_supervision

    def forward(
        self, skips: List[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        skips : list of tensors  [stem, down0, ..., bottleneck]
                (length = n_levels + 1)

        Returns
        -------
        Without deep supervision: single tensor (finest resolution)
        With deep supervision   : list of tensors, coarse to fine
        """
        # tc[row][col]: row = 0..n_levels, col = 0..row
        tc: List[List[Optional[torch.Tensor]]] = [
            [None] * (row + 1) for row in range(self.n_levels + 1)
        ]
        for idx in range(self.n_levels + 1):
            tc[idx][0] = skips[idx]

        outputs: List[torch.Tensor] = []
        for idx in range(1, self.n_levels + 1):
            lv = idx - 1
            for i, block in enumerate(self.up_blocks[lv]):  # type: ignore[index]
                skip_tensors = [
                    tc[j][j - (idx - i - 1)]  # type: ignore[index]
                    for j in range(idx - i - 1, idx)
                ]
                tc[idx][i + 1] = block(tc[idx][i], skip_tensors)  # type: ignore[arg-type]
            # Terminal node at this level → output conv
            terminal = tc[idx][-1]
            if self.deep_supervision or idx == self.n_levels:
                outputs.append(self.out_convs[lv](terminal))

        return outputs if self.deep_supervision else outputs[0]


class DualHeadNestedVNet(nn.Module, SizeMixin):
    """Dual-head Nested V-Net (UNet++) for LA cavity + scar segmentation.

    Shares the same :class:`~.vnet._SegEncoder3D` encoder as
    :class:`~.vnet.DualHeadVNet`, but replaces each decoder with a nested
    (UNet++) path that aggregates dense skip connections from **all** encoder
    levels, enabling deep supervision.

    With ``deep_supervision=True`` (default), ``forward`` returns a pair of
    *lists* ``(la_logits_list, scar_logits_list)``, each with ``n_levels``
    tensors from coarsest to finest resolution.  The Trainer should sum the
    losses across all supervision levels (weighted or equal).

    With ``deep_supervision=False``, ``forward`` returns a pair of single
    tensors identical in shape to :class:`~.vnet.DualHeadVNet`.

    Parameters
    ----------
    config : CFG, optional
        Architecture overrides merged on top of ``__DEFAULT_CONFIG__``.
    """

    __name__ = "DualHeadNestedVNet"

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
        deep_supervision=True,
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
        enc_ch = self.encoder._enc_channels
        up_conv = self.__config.up_conv
        ds = self.__config.deep_supervision

        self.la_decoder = _NestedDecoder(enc_ch, up_conv, norm, act, self.__config.heads.la.out_channels, ds)
        self.scar_decoder = _NestedDecoder(enc_ch, up_conv, norm, act, self.__config.heads.scar.out_channels, ds)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
    ]:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W, D)

        Returns
        -------
        (la_out, scar_out)

        Without deep supervision: each is a tensor of shape (B, 2, H, W, D).
        With deep supervision:    each is a list of tensors coarse→fine.
        """
        skips = self.encoder(x)
        return self.la_decoder(skips), self.scar_decoder(skips)

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

