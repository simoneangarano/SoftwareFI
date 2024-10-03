#!/usr/bin/python3

import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# DieHardNET packages
from pytorch_scripts.utils import *
from pytorch_scripts.data_module import CifarDataModule

# Suppress the annoying warning for non-empty checkpoint directory
warnings.filterwarnings("ignore")

config_parser = parser = argparse.ArgumentParser(
    description="Configuration", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments.",
)

parser = argparse.ArgumentParser(description="PyTorch Training")


# General
parser.add_argument("--name", default="test", help="Experiment name.")
parser.add_argument(
    "--mode", default="train", help="Mode: train/training or validation/validate."
)
parser.add_argument(
    "--ckpt", default=None, help="Pass the name of a checkpoint to resume training."
)
parser.add_argument(
    "--dataset", default="tinyimagenet", help="Dataset name: cifar10 or cifar100."
)
parser.add_argument("--data_dir", default="./data", help="Path to dataset.")
parser.add_argument("--device", default=0, help="Device number.")

# Optimization
parser.add_argument("--loss", default="bce", help="Loss: bce, ce or sce.")
parser.add_argument("--clip", default=None, help="Gradient clipping value.")
parser.add_argument("--epochs", default=150, help="Number of epochs.")
parser.add_argument("--batch_size", default=128, help="Batch Size")
parser.add_argument("--lr", default=1e-1, help="Learning rate.")
parser.add_argument("--optimizer", default="sgd", help="Optimizer name: adamw or sgd.")

# Model
parser.add_argument(
    "--model", default="ghostnetv2", help="Network name. Resnets only for now."
)
parser.add_argument(
    "--order",
    default="bn-relu",
    help="Order of activation and normalization: bn-relu or relu-bn.",
)
parser.add_argument(
    "--affine",
    default=True,
    help="Whether to use Affine transform after normalization or not.",
)
parser.add_argument(
    "--activation", default="relu", help="Non-linear activation: relu or relu6."
)
parser.add_argument("--nan", default=False, help="Whether to convert NaNs to 0 or not.")

# Injection
parser.add_argument(
    "--error_model", default="random", help="Optimizer name: adamw or sgd."
)
parser.add_argument(
    "--inject_p", default=0.1, help="Probability of noise injection at training time."
)
parser.add_argument(
    "--inject_epoch", default=0, help="How many epochs before starting the injection."
)

# Augmentations and Regularisations
parser.add_argument("--wd", default=1e-4, help="Weight Decay.")
parser.add_argument(
    "--rand_aug", type=str, default=None, help="RandAugment magnitude and std."
)
parser.add_argument(
    "--rand_erasing", type=float, default=0.0, help="Random Erasing propability."
)
parser.add_argument(
    "--mixup_cutmix",
    type=bool,
    default=False,
    help="Whether to use mixup/cutmix or not.",
)
parser.add_argument("--jitter", type=float, default=0.0, help="Color jitter.")
parser.add_argument("--label_smooth", type=float, default=0.0, help="Label Smoothing.")


# Others
parser.add_argument("--seed", default=0, help="Random seed for reproducibility.")
parser.add_argument(
    "--comment",
    default="ResNet trained with original settings but the scheduler.",
    help="Optional comment.",
)


def main():
    args = parse_args(parser, config_parser)

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    augs = {
        "rand_aug": args.rand_aug,
        "rand_erasing": args.rand_erasing,
        "mixup_cutmix": args.mixup_cutmix,
        "jitter": args.jitter,
        "label_smooth": args.label_smooth,
    }
    cifar = CifarDataModule(args.dataset, args.data_dir, args.batch_size, 1, augs)

    # Build model (Resnet only up to now)
    optim_params = {
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
    }
    classes = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}
    n_classes = classes[args.dataset]
    net = build_model(
        args.model,
        n_classes,
        optim_params,
        args.loss,
        args.error_model,
        args.inject_p,
        args.inject_epoch,
        args.order,
        args.activation,
        args.nan,
        args.affine,
    )

    # W&B logger
    # wandb_logger = None
    wandb_logger = WandbLogger(project="asi", name=args.name, id=args.name)
    wandb_logger.log_hyperparams(args)
    wandb_logger.watch(net, log_graph=False)

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        "checkpoints/",
        filename=args.name + "-{epoch:02d}-{val_acc:.2f}",
        save_last=True,
    )
    callbacks = [ckpt_callback]

    # Pytorch-Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=[int(args.device)],
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=True,
        benchmark=True,
        accelerator="gpu",
        strategy="auto",
        sync_batchnorm=True,
        gradient_clip_val=args.clip,
    )

    if args.ckpt:
        # args.ckpt = '~/Dropbox/DieHardNet/Checkpoints/' + args.ckpt
        args.ckpt = "checkpoints/" + args.ckpt
    if args.mode == "train" or args.mode == "training":
        trainer.fit(net, cifar, ckpt_path=args.ckpt)
    elif args.mode == "validation" or args.mode == "validate":
        trainer.validate(net, cifar, ckpt_path=args.ckpt)
    else:
        print(
            'ERROR: select a suitable mode "train/training" or "validation/validate".'
        )


if __name__ == "__main__":
    main()
