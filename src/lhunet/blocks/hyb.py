from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

from ..modules.vit.new import HybAttnBlock
from .base import BaseBlock, get_conv_layer
from .cnn import get_cnn_block


__all__ = ["HybridEncoder", "HybridDecoder"]


class BaseHybridBlock:
    def combine(self, tensors: list[torch.Tensor]):
        if self.res_mode == "sum":

            res = torch.zeros_like(tensors[0])
            for t in tensors:
                res = res + t
            return res
        else:
            raise NotImplementedError(
                f"Not implBaseBlockemented combining mode for <{self.res}>"
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
        # ======================================== NEW
        att_cnn_blocks: list[str],
        att_vit_blocks: list[str],
        # ========================================
        spatial_dims=3,
        cnn_blocks="b",
        arch_mode="sequential",  # sequential, residual, parallel, collective
        res_mode="sum",  # "sum" or "cat"
        norm_name="batch",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        use_rb=False,
        use_r=True,
        # *args: Any,
        # **kwds: Any,
    ) -> Any:
        super().__init__()

        # >>> checking
        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(vit_dropouts, list):
            vit_dropouts = [vit_dropouts for _ in features]

        if len(cnn_blocks) == 1:
            cnn_blocks *= len(cnn_kernel_sizes)
        if len(att_vit_blocks) == 1:
            att_vit_blocks *= len(vit_input_sizes)

        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]
        # <<< checking

        self.arch_mode = arch_mode.lower()
        self.res_mode = res_mode.lower()

        self.downs, self.attns, self.convs = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )
        infos = zip(
            cnn_blocks,
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
            # ======================================== NEW
            att_cnn_blocks,
            att_vit_blocks,
            # ========================================
        )
        for (
            c_blkc,
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
            # ======================================== NEW
            acb,
            avb,
            # ========================================
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
                    (
                        nn.MaxPool3d(kernel_size=c_st, stride=c_st)
                        if c_mp
                        else nn.Identity()
                    ),
                )
            )
            self.attns.append(
                nn.Sequential(
                    *[
                        HybAttnBlock(
                            input_size=t_is,
                            dim=och,
                            proj_size=t_ps,
                            num_heads=t_nh,
                            dropout_rate=t_do,
                            # pos_embed=True,
                            # ======================================== NEW
                            cnn_block_code=acb,
                            vit_block_code=avb,
                            use_rb=use_rb,
                            use_r=use_r,
                            # ========================================
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
        for down, attn, conv in zip(self.downs, self.attns, self.convs):
            # print("Encode - HybridEncode, x.shape:", x.shape)
            x = down(x)

            if self.arch_mode.lower == "attconv":  # residual
                x = conv(attn(x) + x)
            elif self.arch_mode == "convatt":
                y = conv(x)
                x = attn(y) + y

            elif self.arch_mode == "sequential":
                x = conv(attn(x))
            elif self.arch_mode == "residual":
                x = self.combine([x, attn(x)])
                x = conv(x)
            elif self.arch_mode == "parallel":
                x = self.combine([x, attn(x), conv(x)])
            elif self.arch_mode == "collective":
                x_a = attn(x)
                x = conv(self.combine([x, x_a]))
                x = self.combine([x, x_a])

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
        # ======================================== NEW
        att_cnn_blocks: list[str],
        att_vit_blocks: list[str],
        # ========================================
        tcv_kernel_sizes,
        tcv_strides,
        tcv_bias=False,
        spatial_dims=3,
        cnn_blocks="b",
        skip_mode="sum",
        arch_mode="sequential",  # sequential, residual, parallel, collective
        res_mode="sum",  # "sum" or "cat"
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        use_rb=False,
        use_r=True,
        # *args: Any,
        # **kwds: Any,
    ) -> Any:
        super().__init__()

        # >>> checking
        if len(cnn_blocks) == 1:
            cnn_blocks *= len(cnn_kernel_sizes)
        if len(att_vit_blocks) == 1:
            att_vit_blocks *= len(vit_input_sizes)
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

        self.ups, self.convs, self.attns, self.convs_o = [
            nn.ModuleList() for _ in range(4)
        ]
        info = zip(
            cnn_blocks,
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
            vit_dropouts,  # ======================================== NEW
            att_cnn_blocks,
            att_vit_blocks,
            # =================================
        )
        for (
            c_blkc,
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
            # ======================================== NEW
            acb,
            avb,
            # =================================
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
            self.attns.append(
                nn.Sequential(
                    *[
                        HybAttnBlock(
                            input_size=t_is,
                            dim=(
                                och
                                if self.arch_mode in ["sequential-lite", "collective"]
                                else vit_chs
                            ),
                            proj_size=t_ps,
                            num_heads=t_nh,
                            dropout_rate=t_do,
                            # pos_embed=True,
                            # ======================================== NEW
                            cnn_block_code=acb,
                            vit_block_code=avb,
                            use_rb=use_rb,
                            use_r=use_r,
                            # ========================================
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
        for up, conv, attn, conv_o in zip(
            self.ups, self.convs, self.attns, self.convs_o
        ):
            # print("Decode - HybridDecode, x.shape:", x.shape)

            x = up(x)
            if self.skip_mode == "sum":
                x = x + skips.pop()
            else:
                x = torch.cat((x, skips.pop()), dim=1)

            if self.arch_mode.lower == "attconv":  # residual
                x = conv_o(attn(x) + x)
            elif self.arch_mode == "convatt":  # collective
                x = conv_o(x)
                x = attn(x) + x

            elif self.arch_mode == "sequential":
                x = conv_o(attn(x))
            elif self.arch_mode == "residual":
                x = self.combine([x, attn(x)])
                x = conv_o(x)
            elif self.arch_mode == "parallel":
                x = self.combine([x, attn(x), conv(x)])
                x = conv_o(x)
            elif self.arch_mode == "sequential-lite":
                x = attn(conv_o(x))
            elif self.arch_mode == "collective":
                x = conv_o(x)
                x = self.combine([x, attn(x)])
            else:
                raise NotImplementedError("Not implementer Arch. for Hybrid Decoder!")

            if return_outs:
                outs.append(x.clone())
        return (x, outs) if return_outs else x
