from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from typing import Union, Tuple, List
from torch import nn
from nnunetv2.model_sharing.lhunet.models.lhunet import LHUNet


class lhunetSynapseTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        args: dict = None,
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, device, args
        )
        self.enable_deep_supervision = False
        self.initial_lr = 0.003
        ####### Dirty hack to make sure the batch and patch size are set correctly for nnUNetTrainer #######
        self.configuration_manager.configuration["patch_size"] = [64, 128, 128]
        self.configuration_manager.configuration["batch_size"] = 2
        self.batch_size = 2
        print(f"batch size: {self.configuration_manager.batch_size}")
        print(f"patch size: {self.configuration_manager.patch_size}")

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return LHUNet(
            spatial_shapes=[64, 128, 128],
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            do_ds=enable_deep_supervision,
            # encoder params
            cnn_kernel_sizes=[3, 3],
            cnn_features=[12, 16],
            cnn_strides=[[1, 2, 2], 2],
            cnn_maxpools=[True, True],
            cnn_dropouts=0.0,
            cnn_blocks="nn",  # n= resunet, d= deformconv, b= basicunet,
            hyb_kernel_sizes=[3, 3, 3],
            hyb_features=[32, 64, 128],
            hyb_strides=[2, 2, 2],
            hyb_maxpools=[True, True, True],
            hyb_cnn_dropouts=0.0,
            hyb_tf_proj_sizes=[64, 32, 0],
            hyb_tf_repeats=[1, 1, 1],
            hyb_tf_num_heads=[8, 8, 16],
            hyb_tf_dropouts=0.0,  ############################ YOUSEF HERE
            hyb_cnn_blocks="nnn",  # n= resunet, d= deformconv, b= basicunet,
            # hyb_vit_blocks="SSC",  # s= dlka_special_v2, S= dlka_sp_seq, c= dlka_channel_v2, C= dlka_ch_seq, ######################## new
            # hyb_vit_sandwich= False,
            hyb_skip_mode="cat",  # "sum" or "cat",
            hyb_arch_mode="residual",  # sequential, residual, parallel, collective,
            hyb_res_mode="sum",  # "sum" or "cat",
            # decoder params
            dec_hyb_tcv_kernel_sizes=[5, 5, 5],
            dec_cnn_tcv_kernel_sizes=[5, 7],
            dec_cnn_blocks=None,
            dec_tcv_bias=False,
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
            # dec_hyb_vit_sandwich= None,
            dec_hyb_skip_mode=None,
            dec_hyb_arch_mode="collective",  # sequential, residual, parallel, collective, sequential-lite,
            dec_hyb_res_mode=None,
            ############################## NEW ################################
            hyb_att_cnn_blocks="ddd",  # d= dlka, l= lka, i= identity, a= cnn-attention | item 1
            hyb_att_vit_blocks="ssc",  # s= spatial-attention, c= channel-attention
            dec_hyb_att_cnn_blocks=None,
            dec_hyb_att_vit_blocks=None,
            use_rb=True,
            use_r=True,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        pass

   