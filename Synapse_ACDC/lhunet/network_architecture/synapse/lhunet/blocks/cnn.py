from typing import Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

from monai.networks.layers.utils import get_act_layer, get_norm_layer

from .base import BaseBlock, get_conv_layer, get_padding
from ..modules.deform_conv import DeformConvPack


__all__ = ["CNNEncoder", "CNNDecoder", "get_cnn_block"]


class DCNNBlock(BaseBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu",
            {"inplace": True, "negative_slope": 0.01},
        ),
        dropout: float | None = None,
    ):
        super().__init__()
        self.dconv = DeformConvPack(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=get_padding(kernel_size, stride),
        )
        # nn.BatchNorm3d(out_channels),
        # nn.PReLU(),
        self.norm = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )
        self.lrelu = get_act_layer(name=act_name)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout, inplace=False)

        self.apply(self._init_weights)

    def forward(self, inp):
        out = self.dconv(inp)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class UnetResBlock(BaseBlock):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu",
            {"inplace": True, "negative_slope": 0.01},
        ),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )
        self.norm2 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                conv_only=True,
            )
            self.norm3 = get_norm_layer(
                name=norm_name, spatial_dims=spatial_dims, channels=out_channels
            )

        self.apply(self._init_weights)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(BaseBlock):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu",
            {"inplace": True, "negative_slope": 0.01},
        ),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            conv_only=True,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )
        self.norm2 = get_norm_layer(
            name=norm_name, spatial_dims=spatial_dims, channels=out_channels
        )

        self.apply(self._init_weights)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(BaseBlock):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = (
            "leakyrelu",
            {"inplace": True, "negative_slope": 0.01},
        ),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

        self.apply(self._init_weights)

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(BaseBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dropout=dropout,
            bias=True,
            conv_only=True,
        )

        self.apply(self._init_weights)

    def forward(self, inp):
        return self.conv(inp)


# =================================================


def get_cnn_block(code):
    if code.lower() == "n":
        return UnetResBlock
    elif code.lower() == "d":
        return DCNNBlock
    elif code.lower() == "b":
        return UnetBasicBlock
    else:
        raise NotImplementedError(f"Not implemented cnn-block for code:<{code}>")


class CNNEncoder(BaseBlock):
    def __init__(
        self,
        in_channels,
        kernel_sizes,
        features,
        strides,
        maxpools,
        dropouts,
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        blocks: str = "n",
        spatial_dims=3,
    ) -> Any:
        super().__init__()

        # >>> checking
        if len(blocks) == 1:
            blocks *= len(kernel_sizes)
        assert isinstance(kernel_sizes, list), "kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert isinstance(strides, list), "strides must be a list"
        assert (
            len(blocks) == len(kernel_sizes) == len(strides) == len(features)
        ), "blocks, kernel_sizes, features, and strides must have the same length"
        if not isinstance(dropouts, list):
            dropouts = [dropouts for _ in features]
        in_out_channles = [in_channels] + features
        in_out_channles = [
            (i, o) for i, o in zip(in_out_channles[:-1], in_out_channles[1:])
        ]
        # <<< checking

        self.encoder_blocks = nn.ModuleList()
        for blkc, (ich, och), ks, st, mp, do in zip(
            blocks, in_out_channles, kernel_sizes, strides, maxpools, dropouts
        ):
            encoder = get_cnn_block(code=blkc)(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                kernel_size=ks,
                stride=1 if mp else st,
                norm_name=norm_name,
                act_name=act_name,
                dropout=do,
            )
            if mp:
                # maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
                maxpool = nn.MaxPool3d(kernel_size=st, stride=st)
                self.encoder_blocks.append(nn.Sequential(encoder, maxpool))
            else:
                self.encoder_blocks.append(encoder)

        self.apply(self._init_weights)

    def forward(self, x):
        layer_features = []
        for block in self.encoder_blocks:
            x = block(x)
            layer_features.append(x.clone())
        return x, layer_features


class CNNDecoder(BaseBlock):
    def __init__(
        self,
        in_channels,
        skip_channels,
        features,
        kernel_sizes,
        dropouts,
        tcv_kernel_sizes,
        tcv_strides,
        tcv_bias=False,
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        blocks: str = "n",
        spatial_dims=3,
    ) -> Any:
        super().__init__()
        if len(blocks) == 1:
            blocks *= len(kernel_sizes)
        assert isinstance(kernel_sizes, list), "kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert (
            len(blocks) == len(kernel_sizes) == len(features)
        ), "blocks, kernel_sizes, and features must have the same length"
        if not isinstance(dropouts, list):
            dropouts = [dropouts for _ in features]
        assert (
            len(kernel_sizes) == len(tcv_strides) == len(features)
        ), "kernel_sizes, features, and tcv_strides must have the same length"

        in_out_channles = [in_channels] + features
        in_out_channles = [
            (i, o) for i, o in zip(in_out_channles[:-1], in_out_channles[1:])
        ]

        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        info = zip(
            blocks,
            in_out_channles,
            skip_channels,
            kernel_sizes,
            dropouts,
            tcv_kernel_sizes,
            tcv_strides,
        )
        for blkc, (ich, och), skch, ks, dpo, tcvks, tcvst in info:
            transp_conv = get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                kernel_size=tcvks,
                stride=tcvst,
                dropout=dpo,
                bias=tcv_bias,
                conv_only=True,
                is_transposed=True,
            )
            self.ups.append(transp_conv)

            conv_block = get_cnn_block(code=blkc)(
                spatial_dims=spatial_dims,
                in_channels=och + skch,
                out_channels=och,
                kernel_size=ks,
                stride=1,
                dropout=dpo,
                norm_name=norm_name,
                act_name=act_name,
            )
            self.convs.append(conv_block)

        self.apply(self._init_weights)

    def forward(self, x, skips: list, return_outs=False, skip_sum=False):
        outs = []
        for up, conv in zip(self.ups, self.convs):
            # print(f"x: {x.shape}, skip: {skips[-1].shape}")
            x = up(x)

            if skip_sum:
                x = x + skips.pop()
            else:
                x = torch.cat((x, skips.pop()), dim=1)

            x = conv(x)
            if return_outs:
                outs.append(x.clone())
        return (x, outs) if return_outs else x
