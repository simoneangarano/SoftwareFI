#!/usr/bin/python3

import csv
import json
import warnings

import random
import numpy as np
import torch

from pytorch_lightning.loggers import WandbLogger
from utils.data.data_module import CoreDataModule

# DieHardNET packages
from utils.utils import build_model, get_parser, parse_args, validate

# Suppress the annoying warning for non-empty checkpoint directory
# warnings.filterwarnings("ignore")
# torch.set_float32_matmul_precision("high")
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def software_fault_injection(args, net, datamodule):

    if args.mode == "train" or args.mode == "training":
        pass
    elif args.mode == "validation" or args.mode == "validate":
        results = validate(net, datamodule, args)
    else:
        print(
            'ERROR: select a suitable mode "train/training" or "validation/validate".'
        )

    return results


def main():
    parser, config_parser = get_parser()
    args = parse_args(parser, config_parser)

    ### Initialization ###

    # Set random seed
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # random.seed(0)
    # np.random.seed(0)

    if args.dataset == "sentinel":
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

    # results = json.load(open(f"ckpt/{args.exp}_stats.json", "r"))
    # args.stats = results

    net = build_model(args)
    net = net.cuda().eval()

    # W&B logger
    if args.wandb:
        wandb_logger = WandbLogger(project="asi", name=args.name, id=args.name)
        wandb_logger.log_hyperparams(args)
        wandb_logger.watch(net, log_graph=False)
    else:
        wandb_logger = None

    ### Fault Injection Test ###
    reader = csv.reader(open(f"cfg/{args.name}_layers_info.csv", mode="r"))
    layers = {i: row[1] for i, row in enumerate(reader)}

    if args.inject_index == -1:
        # inject all layers
        results = software_fault_injection(args, net, datamodule)
        print(
            f'All Layers - Noisy Loss: {results["noisy_loss"]:.1e}, Loss: {results["loss"]:.1e},\n'
            + f'Noisy Acc: {results["noisy_acc"]:.2f}, Acc: {results["acc"]:.2f},\n'
            + f'Noisy mIoU: {results["noisy_miou"]:.2f}, mIoU: {results["miou"]:.2f},\n'
            + f'Noisy F1: {results["noisy_f1"]:.2f}, F1: {results["f1"]:.2f}'
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

        results = software_fault_injection(args, net, datamodule)
        print(
            f'Layer {args.inject_index} ({layers[args.inject_index]}): Noisy Loss: {results["noisy_loss"]:.1e}, Loss: {results["loss"]:.1e},\n'
            + f'Noisy Acc: {results["noisy_acc"]:.2f}, Acc: {results["acc"]:.2f},\n'
            + f'Noisy mIoU: {results["noisy_miou"]:.2f}, mIoU: {results["miou"]:.2f},\n'
            + f'Noisy F1: {results["noisy_f1"]:.2f}, F1: {results["f1"]:.2f}'
        )
        results[args.inject_index] = (
            float(results["noisy_loss"].cpu().numpy()),
            float(results["loss"].cpu().numpy()),
            float(results["noisy_acc"]),
            float(results["acc"]),
            float(results["noisy_miou"]),
            float(results["miou"]),
            float(results["noisy_f1"]),
            float(results["f1"]),
        )
        json.dump(results, open(f"ckpt/{args.exp}_eval.json", "w"))


if __name__ == "__main__":
    main()
