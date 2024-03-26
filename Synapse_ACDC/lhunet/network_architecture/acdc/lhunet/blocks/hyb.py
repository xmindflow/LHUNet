from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

from ..modules.vit.transformers import (
    TransformerBlock,
    TransformerBlock_3D_LKA,
    TransformerBlock_LKA_Channel,
    TransformerBlock_SE,
    TransformerBlock_Deform_LKA_Channel,
    TransformerBlock_Deform_LKA_Channel_sequential,
    TransformerBlock_3D_LKA_3D_conv,
    TransformerBlock_LKA_Channel_norm,
    TransformerBlock_Deform_LKA_Spatial_sequential,
    TransformerBlock_Deform_LKA_Spatial,
    TransformerBlock_3D_single_deform_LKA,
    TransformerBlock_Deform_LKA_Channel_V2,
    TransformerBlock_Deform_LKA_Spatial_V2,
    TransformerBlock_3D_single_deform_LKA_V2,
)

from ..modules.vit.blocks import (
    # TransformerBlock_LKA3D_571,
    # TransformerBlock_LKA3D_5731,
    # TransformerBlock_DLKA3D_single,
    TransformerBlock_DLKA3D_SpatialSequential,
    TransformerBlock_DLKA3D_ChannelSequential,
    TransformerBlock_3D_ChannelAtt_ONLY,
    TransformerBlock_LKA3D_SpatialParallel,
    TransformerBlock_DLKA3D_SpatialParallel,
    TransformerBlock_LKA3D_ChannelNormParallel,
    TransformerBlock_DLKA3D_ChannelParallel,
    # TransformerBlock_LKA3D_ChannelParallel_tempsphead,
    # TransformerBlock_3D_EPA,
    # TransformerBlock_3D_EA,
    # TransformerBlock_3D_SE,
)

from ..modules.vit.new import (
    HybAttn_DLKA3D_Parallel_SpatialViT,
    HybAttn_DLKA3D_Parallel_ChannelViT,
)

from .base import BaseBlock, get_conv_layer
from .cnn import get_cnn_block


__all__ = ["HybridEncoder", "HybridDecoder"]


class TransformerBlock_Deform_LKA_SC_sequential(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dlka_s = TransformerBlock_Deform_LKA_Spatial_sequential(*args, **kwargs)
        self.dlka_c = TransformerBlock_Deform_LKA_Channel_sequential(*args, **kwargs)

    def forward(self, x):
        s_a = self.dlka_s(x)
        c_a = self.dlka_c(x)
        sc_a = s_a + c_a
        return sc_a
        return F.layer_norm(sc_a, normalized_shape=sc_a.shape[2:])


def get_vit_block(code):
    if code == "c":
        return TransformerBlock_Deform_LKA_Channel_V2
    elif code == "s":
        return TransformerBlock_Deform_LKA_Spatial_V2
    # elif code == 'C': return TransformerBlock_Deform_LKA_Channel_sequential
    # elif code == 'S': return TransformerBlock_Deform_LKA_Spatial_sequential
    elif code == "R":
        return TransformerBlock_3D_single_deform_LKA
    elif code == "B":
        return TransformerBlock_Deform_LKA_SC_sequential

    # new blocks
    elif code == "L":
        return TransformerBlock_3D_ChannelAtt_ONLY
    elif code == "Z":
        return TransformerBlock_DLKA3D_ChannelSequential
    elif code == "X":
        return TransformerBlock_DLKA3D_SpatialSequential

    elif code == "w":
        return TransformerBlock_LKA3D_SpatialParallel
    elif code == "W":
        return TransformerBlock_DLKA3D_SpatialParallel
    elif code == "u":
        return TransformerBlock_LKA3D_ChannelNormParallel
    elif code == "U":
        return TransformerBlock_DLKA3D_ChannelParallel

    elif code == "S":
        return HybAttn_DLKA3D_Parallel_SpatialViT
    elif code == "C":
        return HybAttn_DLKA3D_Parallel_ChannelViT

    else:
        raise NotImplementedError(f"Not implemented cnn-block for code:<{code}>")


class BaseHybridBlock:
    def combine(self, tensors: list[torch.Tensor]):
        if self.res_mode == "sum":
            # return torch.sum(torch.stack(tensors))
            res = torch.zeros_like(tensors[0])
            for t in tensors:
                res = res + t
            return res
        # elif self.res_mode=="cat":
        #     res = torch.concatenate(tensors, dim=1)
        #     return res
        else:
            raise NotImplementedError(
                f"Not implemented combining mode for <{self.res}>"
            )


class HybridEncoder(BaseBlock, BaseHybridBlock):
    def __init__(
        self,
        in_channels: int,
        features: list[int],
        cnn_kernel_sizes: list[int | tuple],
        cnn_strides: list[int | tuple],
        cnn_maxpools: list[bool],
        cnn_dropouts: list[float] | float,
        vit_input_sizes: list[int],
        vit_proj_sizes: list[int],
        vit_repeats: list[int],
        vit_num_heads: list[int],
        vit_dropouts: list[float] | float,
        spatial_dims=3,
        cnn_blocks="b",
        vit_blocks="c",
        arch_mode="sequential",  # sequential, residual, parallel, collective
        res_mode="sum",  # "sum" or "cat"
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        *args: Any,
        **kwds: Any,
    ) -> Any:
        super().__init__()

        # >>> checking
        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(vit_dropouts, list):
            vit_dropouts = [vit_dropouts for _ in features]
        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]
        # <<< checking

        self.arch_mode = arch_mode.lower()
        self.res_mode = res_mode.lower()

        self.downs, self.vits, self.convs = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )
        infos = zip(
            cnn_blocks,
            vit_blocks,
            io_channles,
            cnn_kernel_sizes,
            cnn_strides,
            cnn_maxpools,
            cnn_dropouts,
            vit_input_sizes,
            vit_proj_sizes,
            vit_repeats,
            vit_num_heads,
            vit_dropouts,
        )
        for (
            c_blkc,
            t_blkc,
            (ich, och),
            c_ks,
            c_st,
            c_mp,
            c_do,
            t_is,
            t_ps,
            t_rp,
            t_nh,
            t_do,
        ) in infos:
            self.downs.append(
                nn.Sequential(
                    get_cnn_block(code=c_blkc)(
                        spatial_dims=spatial_dims,
                        in_channels=ich,
                        out_channels=och,
                        kernel_size=c_ks,
                        stride=1 if c_mp else c_st,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=c_do,
                    ),
                    # nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if c_mp else nn.Identity()
                    nn.MaxPool3d(kernel_size=c_st, stride=c_st)
                    if c_mp
                    else nn.Identity(),
                )
            )

            self.vits.append(
                nn.Sequential(
                    *[
                        get_vit_block(code=t_blkc)(
                            input_size=t_is,
                            hidden_size=och,
                            proj_size=t_ps,
                            num_heads=t_nh,
                            dropout_rate=t_do,
                            pos_embed=True,
                        )
                        for _ in range(t_rp)
                    ]
                )
            )

            self.convs.append(
                get_cnn_block(code=c_blkc)(
                    spatial_dims=spatial_dims,
                    in_channels=och,
                    out_channels=och,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    act_name=act_name,
                    dropout=c_do,
                )
            )

        self.apply(self._init_weights)

    def forward(self, x):
        layer_features = []
        for down, vit, conv in zip(self.downs, self.vits, self.convs):
            x = down(x)
            if self.arch_mode == "sequential":
                x = conv(vit(x))
            elif self.arch_mode == "residual":
                x = self.combine([x, vit(x)])
                x = conv(x)
            elif self.arch_mode == "parallel":
                x = self.combine([x, vit(x), conv(x)])
            elif self.arch_mode == "collective":
                x_v = vit(x)
                x = conv(self.combine([x, x_v]))
                x = self.combine([x, x_v])
            else:
                raise NotImplementedError("Not implementer Arch. for Hybrid Encoder!")

            layer_features.append(x.clone())
        return x, layer_features


class HybridDecoder(BaseBlock, BaseHybridBlock):
    def __init__(
        self,
        in_channels,
        skip_channels,
        features,
        cnn_kernel_sizes: list[int | tuple],
        cnn_dropouts: list[float] | float,
        vit_input_sizes: list[int],
        vit_proj_sizes: list[int],
        vit_repeats: list[int],
        vit_num_heads: list[int],
        vit_dropouts: list[float] | float,
        tcv_kernel_sizes,
        tcv_strides,
        tcv_bias=False,
        spatial_dims=3,
        cnn_blocks="b",
        vit_blocks="c",
        skip_mode="sum",
        arch_mode="sequential",  # sequential, residual, parallel, collective
        res_mode="sum",  # "sum" or "cat"
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        *args: Any,
        **kwds: Any,
    ) -> Any:
        super().__init__()

        # >>> checking
        if len(cnn_blocks) == 1:
            cnn_blocks *= len(cnn_kernel_sizes)
        if len(vit_blocks) == 1:
            vit_blocks *= len(vit_input_sizes)
        assert isinstance(cnn_kernel_sizes, list), "cnn_kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert (
            len(cnn_blocks) == len(cnn_kernel_sizes) == len(features)
        ), "cnn_blocks, cnn_kernel_sizes, and features must have the same length"
        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(vit_dropouts, list):
            vit_dropouts = [vit_dropouts for _ in features]
        assert (
            len(cnn_kernel_sizes) == len(tcv_strides) == len(features)
        ), "kernel_sizes, features, and tcv_strides must have the same length"
        # <<< checking

        self.skip_mode = skip_mode.lower()
        self.arch_mode = arch_mode.lower()
        self.res_mode = res_mode.lower()

        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]

        self.ups, self.convs, self.vits, self.convs_o = [
            nn.ModuleList() for _ in range(4)
        ]
        info = zip(
            cnn_blocks,
            vit_blocks,
            io_channles,
            skip_channels,
            tcv_kernel_sizes,
            tcv_strides,
            cnn_kernel_sizes,
            cnn_dropouts,
            vit_input_sizes,
            vit_proj_sizes,
            vit_repeats,
            vit_num_heads,
            vit_dropouts,
        )

        for (
            c_blkc,
            t_blkc,
            (ich, och),
            skch,
            tcv_ks,
            tcv_st,
            c_ks,
            c_do,
            t_is,
            t_ps,
            t_rp,
            t_nh,
            t_do,
        ) in info:
            self.ups.append(
                get_conv_layer(
                    spatial_dims=spatial_dims,
                    in_channels=ich,
                    out_channels=och,
                    kernel_size=tcv_ks,
                    stride=tcv_st,
                    dropout=0,
                    bias=tcv_bias,
                    conv_only=True,
                    is_transposed=True,
                )
            )

            vit_chs = och + skch if self.skip_mode == "cat" else och
            self.vits.append(
                nn.Sequential(
                    *[
                        get_vit_block(code=t_blkc)(
                            input_size=t_is,
                            hidden_size=och
                            if self.arch_mode in ["sequential-lite", "collective"]
                            else vit_chs,
                            proj_size=t_ps,
                            num_heads=t_nh,
                            dropout_rate=t_do,
                            pos_embed=True,
                        )
                        for _ in range(t_rp)
                    ]
                )
            )

            self.convs.append(
                get_cnn_block(code=c_blkc)(
                    spatial_dims=spatial_dims,
                    in_channels=och + skch if self.skip_mode == "cat" else och,
                    out_channels=och + skch if self.skip_mode == "cat" else och,
                    kernel_size=c_ks,
                    stride=1,
                    dropout=c_do,
                    norm_name=norm_name,
                    act_name=act_name,
                )
                if self.arch_mode == "parallel"
                else nn.Identity()
            )

            self.convs_o.append(
                get_cnn_block(code=c_blkc)(
                    spatial_dims=spatial_dims,
                    in_channels=och + skch if self.skip_mode == "cat" else och,
                    out_channels=och,
                    kernel_size=3,
                    stride=1,
                    dropout=c_do,
                    norm_name=norm_name,
                    act_name=act_name,
                )
            )

        self.apply(self._init_weights)

    def forward(self, x, skips: list, return_outs=False):
        outs = []
        for up, conv, vit, conv_o in zip(self.ups, self.convs, self.vits, self.convs_o):
            x = up(x)
            if self.skip_mode == "sum":
                x = x + skips.pop()
            else:
                x = torch.cat((x, skips.pop()), dim=1)

            if self.arch_mode == "sequential":
                x = conv_o(vit(x))
            elif self.arch_mode == "residual":
                x = self.combine([x, vit(x)])
                x = conv_o(x)
            elif self.arch_mode == "parallel":
                x = self.combine([x, vit(x), conv(x)])
                x = conv_o(x)
            elif self.arch_mode == "sequential-lite":
                x = vit(conv_o(x))
            elif self.arch_mode == "collective":
                x = conv_o(x)
                x = self.combine([x, vit(x)])
            else:
                raise NotImplementedError("Not implementer Arch. for Hybrid Decoder!")

            if return_outs:
                outs.append(x.clone())
        return (x, outs) if return_outs else x
