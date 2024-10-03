import yaml
from timm.data import create_loader, FastCollateMixup

from .LightningModelWrapper import ModelWrapper
from .hard_resnet import *
from .hard_densenet import *
from .ghostnetv2 import ghostnetv2


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
        net = ghostnetv2(
            num_classes=n_classes,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )
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
    n_classes=100,
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


def parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args
