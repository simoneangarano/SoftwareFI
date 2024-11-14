import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from timm.data import FastCollateMixup, create_loader
from tqdm import tqdm

from .models.ghostnetv2 import GhostNetSS, SegmentationHeadGhostBN, ghostnetv2
from .models.hard_densenet import densenet100
from .models.hard_resnet import (
    hard_resnet20,
    hard_resnet32,
    hard_resnet44,
    hard_resnet56,
)
from .models.LightningModelWrapper import ModelWrapper


### Configuration ###


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
    parser.add_argument("--name", default="test")
    parser.add_argument("--mode", default="validation")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--dataset", default="sentinel")
    parser.add_argument("--num_classes", default=0)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device", default=0)

    # Optimization
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--batch_size", default=128)

    # Model
    parser.add_argument("--model", default="ghostnetv2")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--nan", default=False)

    # Injection
    parser.add_argument("--error_model", default="random")
    parser.add_argument("--inject_p", default=1.0)
    parser.add_argument("--inject_epoch", default=0)
    parser.add_argument("--inject_index", default=0)

    # Others
    parser.add_argument("--seed", default=0)
    parser.add_argument("--comment", default="_")

    return parser, config_parser


def parse_args(parser, config_parser, args=None, verbose=True):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args(args)
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    args.name = f"{args.model}_{args.task}"
    args.exp = f"{args.name}_{args.activation}"
    if args.inject:
        args.exp += f"_{args.inject_p:.0e}"
        if args.inject_first:
            args.exp += "_first"
    if args.stats:
        args.exp += "_stats"
        if args.clip:
            args.exp += "_clip"

    if verbose:
        print("\n==> Config parsed:")
        for k, v in sorted(vars(args).items()):
            print(f"{k}: {v}")
        print()

    # save config
    with open(f"log/{args.exp}.yaml", "w") as f:
        yaml.dump(vars(args), f)

    return args


### Data ###


def get_loader(
    data,
    batch_size=128,
    workers=4,
    num_classes=1000,
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
        num_classes=num_classes,
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


### Model ###


def build_model(args):
    if args.model == "hard_resnet20":
        net = hard_resnet20(args)
    elif args.model == "hard_resnet32":
        net = hard_resnet32(args)
    elif args.model == "hard_resnet44":
        net = hard_resnet44(args)
    elif args.model == "hard_resnet56":
        net = hard_resnet56(args)
    elif args.model == "densenet100":
        net = densenet100(args)
    elif args.model == "ghostnetv2":
        if args.num_classes == 0:
            net = ghostnetv2(args)
        else:
            backbone = ghostnetv2(args)
            head = SegmentationHeadGhostBN(args)
            net = GhostNetSS(backbone, head, args)
    else:
        args.model = "hard_resnet20"
        net = hard_resnet20(args)

    print(f"\n==> {args.model} built.")
    return ModelWrapper(net, args)


def get_layer_type(model, lname):
    for name, layer in model.named_modules():
        if name == lname[6:]:
            return str(type(layer)).split(".")[-1][:-2]


### Validation ###


@torch.no_grad()
def validate(net: ModelWrapper, datamodule, args):
    total = {
        "loss": 0.0,
        "noisy_loss": 0.0,
        "acc": 0.0,
        "noisy_acc": 0.0,
        "miou": 0.0,
        "noisy_miou": 0.0,
        "bacc": 0.0,
        "noisy_bacc": 0.0,
    }
    for _, batch in tqdm(enumerate(datamodule.val_dataloader())):
        batch = [b.cuda() for b in batch]
        noisy_metrics, metrics = net.validation_step(
            batch, inject_index=args.inject_index
        )
        total["loss"] += metrics["loss"]
        total["noisy_loss"] += noisy_metrics["loss"]
        total["acc"] += metrics["acc"]
        total["noisy_acc"] += noisy_metrics["acc"]
        total["miou"] += metrics["miou"]
        total["noisy_miou"] += noisy_metrics["miou"]
        total["bacc"] += metrics["bacc"]
        total["noisy_bacc"] += noisy_metrics["bacc"]

    return {key: val / len(datamodule.val_dataloader()) for key, val in total.items()}


class RunningStats(object):
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, num=0.0, mean=None, var=None, min=None, max=None, clip=1e9):
        self.num = num  # number of samples
        self.mean = mean  # mean
        self.var = var  # sum of squared differences from the mean
        self.min = min  # min value
        self.max = max  # max value
        self.clip = clip  # clip values

    def clear(self):
        self.num = 0.0

    def push(self, x, per_dim=True):
        # process input
        if per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        x = np.clip(x, -self.clip, self.clip)
        self.num += 1
        if self.num == 1:
            self.mean = x
            self.var = 0.0
            self.min = x
            self.max = x
        else:
            prev_m = self.mean.copy()
            self.mean += (x - self.mean) / self.num
            self.var += (x - prev_m) * (x - self.mean)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.num + other.num
            prod_ns = self.num * other.num
            delta2 = (other.mean - self.mean) ** 2.0
            return RunningStats(
                sum_ns,
                (self.mean * self.num + other.mean * other.num) / sum_ns,
                self.var + other.var + delta2 * prod_ns / sum_ns,
            )
        else:
            self.push(other)
            return self

    def get_stats(self):
        return self._mean, self._std, self._min, self._max

    @property
    def _mean(self):
        return float(self.mean.mean()) if self.num else 0.0

    @property
    def _var(self):
        return float(self.var.mean() / (self.num - 1)) if self.num > 1 else 0.0

    @property
    def _std(self):
        return float(np.sqrt(self._var))

    @property
    def _min(self):
        return float(self.min.min()) if self.num else 0.0

    @property
    def _max(self):
        return float(self.max.max()) if self.num else 0.0

    def __repr__(self):
        return "<RunningMean(mean={: 2.4f}, std={: 2.4f}, min={: 2.4f}, max={: 2.4f})>".format(
            self._mean, self._std, self._min, self._max
        )

    def __str__(self):
        return "mean={: 2.4f}, std={: 2.4f}, min={: 2.4f}, max={: 2.4f}".format(
            self._mean, self._std, self._min, self._max
        )


### Visualization ###


def plot_results(results, layers, metric):
    if metric == "all":
        for m in ["loss", "acc", "miou", "bacc"]:
            plot_results(results, layers, m)
        return
    m_ids = {m: 2 * i for i, m in enumerate(["loss", "acc", "miou", "bacc"])}

    x = [int(i) for i, _ in results.items()]
    y = [metrics[m_ids[metric]] for _, metrics in results.items()]
    plt.rcParams['figure.figsize'] = [15, 7]
    plt.plot(x, y, label=metric, color="tab:grey", alpha=0.3)

    for i, metrics in results.items():
        noisy_loss, loss, noisy_acc, acc, noisy_miou, miou, noisy_bacc, bacc = (
            metrics  # NOQA F841
        )
        layer_type = layers[int(i)]["layer_type"]
        layer_color = (
            "tab:orange"
            if layer_type == "Conv2d"
            else "tab:blue" if layer_type == "BatchNorm2d" else "tab:red"
        )
        plt.scatter(i, eval(f"noisy_{metric}"), color=layer_color, label=layer_type)
    plt.hlines(eval(metric), 0, i, color="tab:green", label="Original", linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel(metric)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")
    plt.xticks(range(0, len(results.keys()), 15))
    if metric == "loss":
        plt.yscale("log")
    plt.show()


def plot_stats(results, fresults=None, ltype=None, ylim=None, log=False, alpha=1.0):
    ax = plt.gca()
    results.plot(
        ax=ax,
        kind="line",
        xlabel="Layer" if ltype is None else ltype,
        ylabel="Activation",
        figsize=(20, 10),
        style="--",
        color=["C0", "C1", "C2", "C3"],
        ylim=ylim,
        alpha=alpha,
    )
    if fresults is not None:
        fresults.plot(
            ax=ax,
            kind="line",
            xlabel="Layer" if ltype is None else ltype,
            ylabel="Activation",
            figsize=(20, 10),
            style="-",
            color=["C0", "C1", "C2", "C3"],
            ylim=ylim,
            alpha=alpha,
        )
    if log:
        plt.yscale("symlog")
    plt.show()


def plot_intermediate_stats(
    results, fresults=None, per_layer=False, ylim=None, log=False, alpha=1.0
):
    if per_layer:
        for ltype in results["layer_type"].unique():
            plot_stats(
                results[results["layer_type"] == ltype],
                (
                    fresults[fresults["layer_type"] == ltype]
                    if fresults is not None
                    else None
                ),
                ltype=ltype,
                ylim=ylim,
                log=log,
                alpha=alpha,
            )
    else:
        plot_stats(results, fresults, ylim=ylim, log=log, alpha=alpha)


### Utils ###
