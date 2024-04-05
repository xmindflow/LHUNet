import numpy as np
import torch
from torch import nn
from ..blocks.cnn import UnetResBlock, UnetOutBlock
from ..blocks import *


__all__ = ["LHUNet"]


from ....neural_network import SegmentationNetwork

class LHUNet(SegmentationNetwork):
    def __init__(
        self,
        spatial_shapes,
        do_ds=False,
        in_channels=4,
        out_channels=3,
        # encoder params
        cnn_kernel_sizes=[5, 3],
        cnn_features=[32, 64],
        cnn_strides=[2, 2],
        cnn_maxpools=[0, 1],
        cnn_dropouts=0.1,
        cnn_blocks="n",
        hyb_kernel_sizes=[3, 3, 3],
        hyb_features=[128, 256, 512],
        hyb_strides=[2, 2, 2],
        hyb_maxpools=[0, 0, 0],
        hyb_cnn_dropouts=0.1,
        hyb_tf_proj_sizes=[32, 64, 64],
        hyb_tf_repeats=[3, 3, 3],
        hyb_tf_num_heads=[4, 4, 4],
        hyb_tf_dropouts=0.15,
        hyb_cnn_blocks="n",
        hyb_vit_blocks="s",
        # hyb_vit_sandwich=False,
        hyb_skip_mode="sum",
        hyb_arch_mode="sequential",  # sequential, residual, parallel, collective
        hyb_res_mode="sum",  # "sum" or "cat"
        # decoder params
        dec_hyb_tcv_kernel_sizes=[5, 5, 5],
        dec_cnn_tcv_kernel_sizes=[5, 5, 5],
        dec_tcv_bias=False,
        dec_cnn_blocks=None,
        dec_hyb_tcv_bias=False,
        dec_hyb_kernel_sizes=None,
        dec_hyb_features=None,
        dec_hyb_cnn_dropouts=None,
        dec_hyb_tf_proj_sizes=None,
        dec_hyb_tf_repeats=None,
        dec_hyb_tf_num_heads=None,
        dec_hyb_tf_dropouts=None,
        dec_cnn_kernel_sizes=None,
        dec_cnn_features=None,
        dec_cnn_dropouts=None,
        dec_hyb_cnn_blocks=None,
        dec_hyb_vit_blocks=None,
        # dec_hyb_vit_sandwich=None,
        dec_hyb_skip_mode=None,
        dec_hyb_arch_mode=None,
        dec_hyb_res_mode=None,
    ):
        super().__init__()
        self.do_ds = do_ds

        # ------------------------------------- Vars Prepration --------------------------------
        spatial_dims = len(spatial_shapes)
        init_features = cnn_features[0] // 2
        enc_cnn_in_channels = in_channels
        enc_cnn_out_channels = cnn_features[-1]
        enc_hyb_in_channels = enc_cnn_out_channels
        enc_hyb_out_channels = hyb_features[-1]

        # check dropouts
        cnn_dropouts = (
            [cnn_dropouts for _ in spatial_shapes]
            if not isinstance(cnn_dropouts, list)
            else cnn_dropouts
        )
        hyb_cnn_dropouts = (
            [hyb_cnn_dropouts for _ in spatial_shapes]
            if not isinstance(hyb_cnn_dropouts, list)
            else hyb_cnn_dropouts
        )
        hyb_tf_dropouts = (
            [hyb_tf_dropouts for _ in spatial_shapes]
            if not isinstance(hyb_tf_dropouts, list)
            else hyb_tf_dropouts
        )

        # check strides
        cnn_strides = [
            [st for _ in spatial_shapes] if not isinstance(st, list) else st
            for st in cnn_strides
        ]
        hyb_strides = [
            [st for _ in spatial_shapes] if not isinstance(st, list) else st
            for st in hyb_strides
        ]

        # check dec params
        dec_hyb_skip_channels = hyb_features[::-1][1:] + cnn_features[::-1]
        dec_cnn_skip_channels = cnn_features[::-1][1:] + [init_features]
        if not dec_hyb_features:
            dec_hyb_features = hyb_features[::-1][1:] + [enc_hyb_in_channels]
        if not dec_cnn_features:
            dec_cnn_features = cnn_features[::-1][1:] + [init_features]

        if not dec_hyb_kernel_sizes:
            dec_hyb_kernel_sizes = hyb_kernel_sizes[::-1]
        if not dec_hyb_cnn_dropouts:
            dec_hyb_cnn_dropouts = hyb_cnn_dropouts[::-1]
        if not dec_hyb_tf_proj_sizes:
            dec_hyb_tf_proj_sizes = hyb_tf_proj_sizes[::-1]
        if not dec_hyb_tf_repeats:
            dec_hyb_tf_repeats = hyb_tf_repeats[::-1]
        if not dec_hyb_tf_num_heads:
            dec_hyb_tf_num_heads = hyb_tf_num_heads[::-1]
        if not dec_hyb_tf_dropouts:
            dec_hyb_tf_dropouts = hyb_tf_dropouts[::-1]
        if not dec_cnn_kernel_sizes:
            dec_cnn_kernel_sizes = cnn_kernel_sizes[::-1]

        if not dec_cnn_dropouts:
            dec_cnn_dropouts = cnn_dropouts[::-1]

        if not dec_cnn_blocks:
            dec_cnn_blocks = cnn_blocks[::-1]
        if not dec_hyb_cnn_blocks:
            dec_hyb_cnn_blocks = hyb_cnn_blocks[::-1]
        if not dec_hyb_vit_blocks:
            dec_hyb_vit_blocks = hyb_vit_blocks[::-1]
        # if not dec_hyb_vit_sandwich:
        #     dec_hyb_vit_sandwich = hyb_vit_sandwich
        if not dec_hyb_skip_mode:
            dec_hyb_skip_mode = hyb_skip_mode
        if not dec_hyb_arch_mode:
            dec_hyb_arch_mode = hyb_arch_mode
        if not dec_hyb_res_mode:
            dec_hyb_res_mode = hyb_res_mode

        # calculate spatial_shapes in encoder and decoder diferent layers
        enc_spatial_shaps = [spatial_shapes]
        for stride in cnn_strides + hyb_strides:
            enc_spatial_shaps.append(
                [int(np.ceil(ss / st)) for ss, st in zip(enc_spatial_shaps[-1], stride)]
            )
        dec_spatial_shaps = [enc_spatial_shaps[-2]]
        for stride in hyb_strides[::-1] + cnn_strides[::-1]:
            dec_spatial_shaps.append(
                [int(np.ceil(ss * st)) for ss, st in zip(dec_spatial_shaps[-1], stride)]
            )
       
        enc_cnn_spatial_shaps = enc_spatial_shaps[: len(cnn_kernel_sizes)]
        enc_hyb_spatial_shaps = enc_spatial_shaps[
            len(cnn_kernel_sizes) + 1 :
        ]  # we need output channels of cnn before tf
        dec_hyb_spatial_shaps = dec_spatial_shaps[: len(hyb_kernel_sizes)]
        dec_cnn_spatial_shaps = dec_spatial_shaps[
            len(hyb_kernel_sizes) :
        ]  # we need input channels of block before cnn

        # calc hyb_tf_input_sizes corresponding cnn_strides and hyb_strides
        enc_hyb_tf_input_sizes = [
            np.prod(ss, dtype=int) for ss in enc_hyb_spatial_shaps
        ]
        dec_hyb_tf_input_sizes = [
            np.prod(ss, dtype=int) for ss in dec_hyb_spatial_shaps
        ]

        # ------------------------------------- Initialization --------------------------------
        self.init = nn.Sequential(
            nn.Conv3d(in_channels, init_features, 1),
            nn.PReLU(),
            nn.BatchNorm3d(init_features),
        )

        # ------------------------------------- Encoder --------------------------------
        self.cnn_encoder = CNNEncoder(
            in_channels=init_features,
            kernel_sizes=cnn_kernel_sizes,
            features=cnn_features,
            strides=cnn_strides,
            maxpools=cnn_maxpools,
            dropouts=cnn_dropouts,
            blocks=cnn_blocks,
            spatial_dims=spatial_dims,
            norm_name="batch",  
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        )

        self.hyb_encoder = HybridEncoder(
            in_channels=cnn_features[-1],
            features=hyb_features,
            cnn_kernel_sizes=hyb_kernel_sizes,
            cnn_strides=hyb_strides,
            cnn_maxpools=hyb_maxpools,
            cnn_dropouts=hyb_cnn_dropouts,
            vit_input_sizes=enc_hyb_tf_input_sizes,
            vit_proj_sizes=hyb_tf_proj_sizes,
            vit_repeats=hyb_tf_repeats,
            vit_num_heads=hyb_tf_num_heads,
            vit_dropouts=hyb_tf_dropouts,
            spatial_dims=spatial_dims,
            cnn_blocks=hyb_cnn_blocks,
            vit_blocks=hyb_vit_blocks,
            # vit_sandwich=hyb_vit_sandwich,
            skip_mode=hyb_skip_mode,
            arch_mode=hyb_arch_mode,
            res_mode=hyb_res_mode,
            norm_name="batch",  
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        )

        # ------------------------------------- Decoder --------------------------------
        self.hyb_decoder = HybridDecoder(
            in_channels=enc_hyb_out_channels,
            features=dec_hyb_features,
            skip_channels=dec_hyb_skip_channels,
            tcv_kernel_sizes=dec_hyb_tcv_kernel_sizes,
            tcv_strides=hyb_strides[::-1],
            tcv_bias=dec_hyb_tcv_bias,
            cnn_kernel_sizes=dec_hyb_kernel_sizes,
            cnn_dropouts=dec_hyb_cnn_dropouts,
            vit_input_sizes=dec_hyb_tf_input_sizes,
            vit_proj_sizes=dec_hyb_tf_proj_sizes,
            vit_repeats=dec_hyb_tf_repeats,
            vit_num_heads=dec_hyb_tf_num_heads,
            vit_dropouts=dec_hyb_tf_dropouts,
            # return_outs=self.use_ds,
            spatial_dims=spatial_dims,
            cnn_blocks=dec_hyb_cnn_blocks,
            vit_blocks=dec_hyb_vit_blocks,
            # vit_sandwich=dec_hyb_vit_sandwich,
            skip_mode=dec_hyb_skip_mode,
            arch_mode=dec_hyb_arch_mode,
            res_mode=dec_hyb_res_mode,
            norm_name="batch",  
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        )

        self.cnn_decoder = CNNDecoder(
            in_channels=dec_hyb_features[-1],
            skip_channels=dec_cnn_skip_channels,
            features=dec_cnn_features,
            kernel_sizes=dec_cnn_kernel_sizes,
            dropouts=dec_cnn_dropouts,
            tcv_kernel_sizes=dec_cnn_tcv_kernel_sizes,
            tcv_strides=cnn_strides[::-1],
            tcv_bias=dec_tcv_bias,
            # return_outs=self.use_ds,
            spatial_dims=spatial_dims,
            blocks=dec_cnn_blocks,
            norm_name="batch",  
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        )

        # -------------------------------- OUT --------------------------------
        self.out_skip = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=dec_cnn_features[-1],
            kernel_size=5,
            stride=1,
            norm_name="batch",
        )
        self.out = nn.Sequential(
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=dec_cnn_features[-1] * 2,
                out_channels=dec_cnn_features[-1],
                kernel_size=5,
                stride=1,
                norm_name="batch",
            ),
            UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=dec_cnn_features[-1],
                out_channels=out_channels,
                dropout=0,
            ),
        )

        self.num_classes = out_channels

    def forward(self, x):
        in_x = x.clone()
        x = self.init(x)
        r = x.clone()

        x, cnn_skips = self.cnn_encoder(x)
        x, hyb_skips = self.hyb_encoder(x)

        x = self.hyb_decoder(x, [cnn_skips[-1]] + hyb_skips[:-1])
        x = self.cnn_decoder(x, [r] + cnn_skips[:-1])

        x = torch.concatenate([x, self.out_skip(in_x)], dim=1)
        x = self.out(x)

        return x
