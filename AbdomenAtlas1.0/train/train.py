import torch
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from dataset.dataloader_bdmap import get_loader

from tensorboardX import SummaryWriter

from model.network_architecture.synapse.lhunet.models.v7 import LHUNet
from monai.losses import DiceLoss
from optimizers.custom_decay import CustomDecayLR

torch.multiprocessing.set_sharing_strategy("file_system")


def train(args, train_loader, model, optimizer, loss_seg_dice, loss_seg_ce):
    model.train()
    loss_ave_dice = 0
    loss_ave_ce = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = (
            batch["image"].to(args.device),
            batch["label"].float().to(args.device),
            batch["name"],
        )
        assert x.shape == y.shape # x: (B, C, w, h, D), y: (B, C, w, h, D)
        logit_map = model(x)
        temp_loss_dice = loss_seg_dice.forward(logit_map, y)
        temp_loss_ce = loss_seg_ce.forward(logit_map, y.squeeze(1).long())
        loss = temp_loss_dice + temp_loss_ce
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, ce_loss=%2.5f)"
            % (
                args.epoch,
                step,
                len(train_loader),
                temp_loss_dice.item(),
                temp_loss_ce.item()
            )
        )
        loss_ave_dice += temp_loss_dice.item()
        loss_ave_ce += temp_loss_ce.item()
        torch.cuda.empty_cache()
    print(
        "Epoch=%d: ave_dice_loss=%2.5f, ave_ce_loss=%2.5f"
        % (
            args.epoch,
            loss_ave_dice / len(epoch_iterator),
            loss_ave_ce / len(epoch_iterator),
        )
    )

    return loss_ave_dice / len(epoch_iterator), loss_ave_ce / len(epoch_iterator)


def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

   
    if args.backbone == "lhunet":
        model = LHUNet(
            spatial_shapes=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.num_class,
            do_ds=False,
            # encoder params
            cnn_kernel_sizes=[3, 3],
            cnn_features=[12, 16],
            cnn_strides=[2, 2],
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
            hyb_tf_dropouts=0.1,
            hyb_cnn_blocks="nnn",  # n= resunet, d= deformconv, b= basicunet,
            hyb_vit_blocks="SSC",  # s= dlka_special_v2, S= dlka_sp_seq, c= dlka_channel_v2, C= dlka_ch_seq,
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
        )
    ### load model from checkpoint
    

    model.to(args.device)
    model.train()

    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])
        
    loss_seg_dice= DiceLoss(include_background=True, to_onehot_y=True, softmax=True, smooth_nr=0).to(args.device)
    loss_seg_ce = torch.nn.CrossEntropyLoss().to(args.device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.99,
        nesterov=True,
    )

    scheduler = CustomDecayLR(optimizer, max_epochs=args.max_epoch)

    torch.backends.cudnn.benchmark = True 
    
    if args.continue_training:
        if args.continue_path is None:
            raise ValueError("Please provide the path to the checkpoint")
        checkpoint = torch.load(args.continue_path, map_location='cpu')
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.epoch = checkpoint["epoch"]
        print("Model loaded successfully at epoch:", args.epoch) 
        model.to(args.device)      

    train_loader, train_sampler, train_ds = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join("out", args.log_name))

    if args.cache_dataset:
        train_ds.start()
        
    
    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
            
        loss_dice, loss_ce = train(
            args, train_loader, model, optimizer, loss_seg_dice, loss_seg_ce
        )
        
        scheduler.step()
        if args.cache_dataset:
            train_ds.update_cache()
        
        if rank == 0:
            writer.add_scalar("train_ce_loss", loss_ce, args.epoch)
            writer.add_scalar("train_dice_loss", loss_dice, args.epoch)
            writer.add_scalar("lr", scheduler.get_lr(), args.epoch)
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": args.epoch,
            }
            save_dir = os.path.join("out", args.log_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, args.backbone + ".pth")
            torch.save(checkpoint, save_path)
            print("Model saved successfully at epoch:", args.epoch)

        args.epoch += 1
        
    if args.cache_dataset:
        train_ds.shutdown()
        
    dist.destroy_process_group()
    


def main():
    parser = argparse.ArgumentParser()

    ## for distributed training
    parser.add_argument(
        "--dist",
        dest="dist",
        type=bool,
        default=False,
        help="distributed training or not",
    )
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--num_workers", default=12, type=int, help="workers numebr for DataLoader"
    )

    ## logging
    parser.add_argument(
        "--log_name",
        default="AbdomenAtlas1.0.unet",
        help="The path resume from checkpoint",
    )

    ## hyperparameter
    parser.add_argument("--epoch", default=0)
    parser.add_argument(
        "--max_epoch", default=800, type=int, help="Number of training epoches"
    )
    parser.add_argument(
        "--warmup_epoch", default=20, type=int, help="number of warmup epochs"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-5, help="Weight Decay")
    parser.add_argument(
        "--backbone", default="unet", help="model backbone, unet backbone by default"
    )

    ## dataset
    parser.add_argument("--dataset_list", nargs="+", default=["AbdomenAtlas1.0"])
    parser.add_argument(
        "--data_root_path",
        default="/data2/wenxuan/AbdomenAtlas1.0",
        help="data root path",
    )
    parser.add_argument(
        "--data_txt_path", default="./dataset/dataset_list/", help="data txt path"
    )
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")
    parser.add_argument(
        "--a_min", default=-175, type=float, help="a_min in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--a_max", default=250, type=float, help="a_max in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged"
    )
    parser.add_argument(
        "--space_x", default=1.5, type=float, help="spacing in x direction"
    )
    parser.add_argument(
        "--space_y", default=1.5, type=float, help="spacing in y direction"
    )
    parser.add_argument(
        "--space_z", default=1.5, type=float, help="spacing in z direction"
    )
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument(
        "--num_samples", default=2, type=int, help="sample number in each ct"
    )
    parser.add_argument(
        "--cache_dataset",
        action="store_true",
        default=False,
        help="whether use cache dataset",
    )
    parser.add_argument(
        "--cache_rate",
        default=1.0,
        type=float,
        help="the percentage of cached data in total",
    )
    parser.add_argument(
        "--cache_num", default=500, type=int, help="the number of cached data"
    )
    parser.add_argument("--num_class", default=10, type=int, help="number of class")
    
    parser.add_argument("--continue_training", default=False, type=bool, help="continue training or not")
    
    parser.add_argument("--continue_path", default=None, help="The path resume from checkpoint")

    args = parser.parse_args()

    process(args=args)


if __name__ == "__main__":
    main()
