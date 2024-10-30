#!/usr/bin/python3

import csv
import json
import warnings

import torch

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils.data.data_module import CifarDataModule, CoreDataModule

# DieHardNET packages
from utils.utils import build_model, get_parser, parse_args, validate

# Suppress the annoying warning for non-empty checkpoint directory
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def software_fault_injection(args, net, datamodule):

    # Pytorch-Lightning Trainer
    # trainer = pl.Trainer(
    #     max_epochs=args.epochs,
    #     devices=[int(args.device)],
    #     callbacks=callbacks,
    #     logger=wandb_logger,
    #     deterministic=True,
    #     benchmark=True,
    #     accelerator="gpu",
    #     strategy="auto",
    #     sync_batchnorm=True,
    #     gradient_clip_val=args.clip,
    # )

    # if args.ckpt:
    #     args.ckpt = "ckpt/" + args.ckpt
    #     net = load_fi_weights(net, args.ckpt)
    if args.mode == "train" or args.mode == "training":
        pass
        # trainer.fit(net, datamodule, ckpt_path=args.ckpt)
    elif args.mode == "validation" or args.mode == "validate":
        noisy_loss, loss, noisy_acc, acc, noisy_miou, miou = validate(
            net, datamodule, args
        )
        # trainer.validate(net, datamodule, ckpt_path=None)
    else:
        print(
            'ERROR: select a suitable mode "train/training" or "validation/validate".'
        )

    return noisy_loss, loss, noisy_acc, acc, noisy_miou, miou


def main():
    parser, config_parser = get_parser()
    args = parse_args(parser, config_parser)

    ### Initialization ###

    # Set random seed
    # pl.seed_everything(args.seed, workers=True)
    torch.manual_seed(args.seed)

    augs = {
        "rand_aug": args.rand_aug,
        "rand_erasing": args.rand_erasing,
        "mixup_cutmix": args.mixup_cutmix,
        "jitter": args.jitter,
        "label_smooth": args.label_smooth,
    }
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        datamodule = CifarDataModule(
            args.dataset, args.data_dir, args.batch_size, 1, augs
        )
    elif args.dataset == "sentinel":
        datamodule = CoreDataModule(args)

    # Build model (Resnet only up to now)
    args.optim_params = (
        {
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "lr": args.lr,
            "wd": args.wd,
        }
        if args.optim_params is None
        else args.optim_params
    )

    # args.stats = {...}

    net = build_model(args)
    net = net.cuda().eval()

    # W&B logger
    if args.wandb:
        wandb_logger = WandbLogger(project="asi", name=args.name, id=args.name)
        wandb_logger.log_hyperparams(args)
        wandb_logger.watch(net, log_graph=False)
    else:
        wandb_logger = None

    # Callbacks
    # ckpt_callback = ModelCheckpoint(
    #     "ckpt/",
    #     filename=args.name + "-{epoch:02d}-{val_acc:.2f}",
    #     save_last=True,
    # )
    # callbacks = [ckpt_callback]

    ### Fault Injection Test ###

    reader = csv.reader(open(f"cfg/{args.name}_layers_info.csv", mode="r"))
    layers = {i: row[1] for i, row in enumerate(reader)}

    if args.inject_index == -1:
        # inject all layers
        noisy_loss, loss, noisy_acc, acc, noisy_miou, miou = software_fault_injection(
            args, net, datamodule
        )
        print(
            f"All Layers - Noisy Loss: {noisy_loss:.1e}, Loss: {loss:.1e}, Noisy Acc: {noisy_acc:.2f}, Acc: {acc:.2f}, Noisy mIoU: {noisy_miou:.2f}, mIoU: {miou:.2f}"
        )
        return

    # inject specific layer
    while args.inject_index < len(layers):
        try:
            results = json.load(open(f"ckpt/{args.exp}_eval.json", "r"))
            args.inject_index = len(results)
            if args.inject_index == len(layers):
                print("All layers have been tested.")
                break
        except:
            results = {}

        noisy_loss, loss, noisy_acc, acc, noisy_miou, miou = software_fault_injection(
            args, net, datamodule
        )
        print(
            f"Layer {args.inject_index} ({layers[args.inject_index]}): Noisy Loss: {noisy_loss:.1e}, Loss: {loss:.1e}, Noisy Acc: {noisy_acc:.2f}, Acc: {acc:.2f}, Noisy mIoU: {noisy_miou:.2f}, mIoU: {miou:.2f}"
        )
        results[args.inject_index] = (
            float(noisy_loss.cpu().numpy()),
            float(loss.cpu().numpy()),
            float(noisy_acc.cpu().numpy()),
            float(acc.cpu().numpy()),
            float(noisy_miou),
            float(miou),
        )
        json.dump(results, open(f"ckpt/{args.exp}_eval.json", "w"))


if __name__ == "__main__":
    main()
