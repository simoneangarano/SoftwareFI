import yaml, argparse
from tqdm import tqdm
from timm.data import create_loader, FastCollateMixup

from .models.LightningModelWrapper import ModelWrapper
from .models.hard_resnet import *
from .models.hard_densenet import *
from .models.ghostnetv2 import ghostnetv2, GhostNetSS, SegmentationHeadGhostBN


def build_model(
    model="hard_resnet20",
    n_classes=10,
    optim_params={},
    loss="bce",
    error_model="random",
    inject_p=0.1,
    inject_epoch=0,
    order="relu-bn",
    activation="relu",
    nan=False,
    affine=True,
    ckpt=None,
):
    if model == "hard_resnet20":
        net = hard_resnet20(
            n_classes,
            error_model,
            inject_p,
            inject_epoch,
            order,
            activation,
            nan,
            affine,
        )
    elif model == "hard_resnet32":
        net = hard_resnet32(
            n_classes,
            error_model,
            inject_p,
            inject_epoch,
            order,
            activation,
            nan,
            affine,
        )
    elif model == "hard_resnet44":
        net = hard_resnet44(
            n_classes,
            error_model,
            inject_p,
            inject_epoch,
            order,
            activation,
            nan,
            affine,
        )
    elif model == "hard_resnet56":
        net = hard_resnet56(
            n_classes,
            error_model,
            inject_p,
            inject_epoch,
            order,
            activation,
            nan,
            affine,
        )
    elif model == "densenet100":
        net = densenet100(n_classes)
    elif model == "ghostnetv2":
        if n_classes == 0:
            net = ghostnetv2(
                # num_classes=n_classes,
                error_model=error_model,
                inject_p=inject_p,
                inject_epoch=inject_epoch,
                ckpt=ckpt,
                activation=activation,
            )
        else:
            backbone = ghostnetv2(
                # num_classes=n_classes,
                error_model=error_model,
                inject_p=inject_p,
                inject_epoch=inject_epoch,
                ckpt=None,
                activation=activation,
            )
            head = SegmentationHeadGhostBN(num_classes=n_classes, activation=activation)
            net = GhostNetSS(backbone, head, ckpt=ckpt)
    else:
        model = "hard_resnet20"
        net = hard_resnet20(
            n_classes,
            error_model,
            inject_p,
            inject_epoch,
            order,
            activation,
            nan,
            affine,
        )

    print(f"\n==> {model} built.")
    return ModelWrapper(net, n_classes, optim_params, loss)


def get_loader(
    data,
    batch_size=128,
    workers=4,
    n_classes=1000,
    stats=None,
    mixup_cutmix=True,
    rand_erasing=0.0,
    label_smooth=0.1,
    rand_aug="rand-m9-mstd0.5-inc1",
    jitter=0.0,
    size=32,
):
    if mixup_cutmix:
        mixup_alpha = 0.8
        cutmix_alpha = 1.0
        prob = 1.0
        switch_prob = 0.5
    else:
        mixup_alpha = 0.0
        cutmix_alpha = 0.0
        prob = 0.0
        switch_prob = 0.0
    collate = FastCollateMixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=None,
        prob=prob,
        switch_prob=switch_prob,
        mode="batch",
        label_smoothing=label_smooth,
        num_classes=n_classes,
    )
    return create_loader(
        data,
        input_size=(3, size, size),
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=False,
        re_prob=rand_erasing,  # RandErasing
        re_mode="pixel",
        re_count=1,
        re_split=False,
        scale=[0.75, 1.0],
        ratio=[3.0 / 4.0, 4.0 / 3.0],
        hflip=0.5,
        vflip=0,
        color_jitter=jitter,
        auto_augment=rand_aug,
        num_aug_splits=0,
        interpolation="random",
        mean=stats[0],
        std=stats[1],
        num_workers=workers,
        distributed=False,
        collate_fn=collate,
        pin_memory=True,
        use_multi_epochs_loader=False,
        fp16=False,
    )


def get_parser():
    config_parser = parser = argparse.ArgumentParser(
        description="Configuration", add_help=False
    )
    parser.add_argument(
        "-c",
        "--config",
        default="cfg/ghostnetv2_clouds.yaml",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments.",
    )

    parser = argparse.ArgumentParser(description="PyTorch Training")

    # General
    parser.add_argument("--name", default="test", help="Experiment name.")
    parser.add_argument(
        "--mode",
        default="validation",
        help="Mode: train/training or validation/validate.",
    )
    parser.add_argument(
        "--ckpt",
        default="ckpt/GN_SSL_280.pt",
        help="Pass the name of a checkpoint to resume training.",
    )
    parser.add_argument(
        "--dataset", default="sentinel", help="Dataset name: cifar10 or cifar100."
    )
    parser.add_argument("--num_classes", default=0, help="Number of classes.")
    parser.add_argument("--data_dir", default="./data", help="Path to dataset.")
    parser.add_argument("--device", default=0, help="Device number.")

    # Optimization
    parser.add_argument("--loss", default="mse", help="Loss: bce, ce or sce.")
    parser.add_argument("--clip", default=None, help="Gradient clipping value.")
    parser.add_argument("--epochs", default=150, help="Number of epochs.")
    parser.add_argument("--batch_size", default=128, help="Batch Size")
    parser.add_argument("--lr", default=1e-1, help="Learning rate.")
    parser.add_argument(
        "--optimizer", default="sgd", help="Optimizer name: adamw or sgd."
    )

    # Model
    parser.add_argument("--model", default="ghostnetv2", help="Network name.")
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
    parser.add_argument(
        "--nan", default=False, help="Whether to convert NaNs to 0 or not."
    )

    # Injection
    parser.add_argument(
        "--error_model", default="random", help="Optimizer name: adamw or sgd."
    )
    parser.add_argument(
        "--inject_p",
        default=0.001,
        help="Probability of noise injection at training time.",
    )
    parser.add_argument(
        "--inject_epoch",
        default=0,
        help="How many epochs before starting the injection.",
    )

    parser.add_argument(
        "--inject_index",
        default=0,
        help="Injection index: specific layer.",
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
    parser.add_argument(
        "--label_smooth", type=float, default=0.0, help="Label Smoothing."
    )

    # Others
    parser.add_argument("--seed", default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--comment",
        default="ResNet trained with original settings but the scheduler.",
        help="Optional comment.",
    )

    return parser, config_parser


def parse_args(parser, config_parser, args=None):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args(args)
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    print("\n==> Config parsed:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print()

    return args


@torch.no_grad()
def validate(net: ModelWrapper, datamodule, args):
    total_loss, total_noisy_loss = 0, 0
    total_acc, total_noisy_acc = 0, 0
    for batch_idx, batch in tqdm(enumerate(datamodule.val_dataloader())):
        batch = [b.cuda() for b in batch]
        noisy_loss, loss, noisy_acc, acc = net.validation_step(
            batch, batch_idx, True, inject_index=args.inject_index
        )
        total_loss += loss
        total_noisy_loss += noisy_loss
        total_acc += acc
        total_noisy_acc += noisy_acc
    return (
        total_noisy_loss / len(datamodule.val_dataloader()),
        total_loss / len(datamodule.val_dataloader()),
        total_noisy_acc / len(datamodule.val_dataloader()),
        total_acc / len(datamodule.val_dataloader()),
    )
