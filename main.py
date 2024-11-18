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
warnings.filterwarnings("ignore")


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
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "sentinel":
        datamodule = CoreDataModule(args)

    # Build model (Resnet only up to now)
    # args.optim_params = (
    #     {
    #         "optimizer": args.optimizer,
    #         "epochs": args.epochs,
    #         "lr": args.lr,
    #         "wd": args.wd,
    #     }
    #     if args.optim_params is None
    #     else args.optim_params
    # )

    # Load Stats
    if args.stats:
        results = json.load(open(f"ckpt/{args.name}_{args.activation}_stats.json", "r"))
        args.stats = results

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
            f'All Layers: \nNoisy Loss: {results["noisy_loss"]:.1e}, Loss: {results["loss"]:.1e},\n'
            + f'Noisy Acc: {results["noisy_acc"]:.2f}, Acc: {results["acc"]:.2f},\n'
            + f'Noisy mIoU: {results["noisy_miou"]:.2f}, mIoU: {results["miou"]:.2f},\n'
            + f'Noisy BAcc: {results["noisy_bacc"]:.2f}, BAcc: {results["bacc"]:.2f}'
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

        res = software_fault_injection(args, net, datamodule)
        print(
            f'Layer {args.inject_index} ({layers[args.inject_index]}): \nNoisy Loss: {res["noisy_loss"]:.1e}, Loss: {res["loss"]:.1e},\n'
            + f'Noisy Acc: {res["noisy_acc"]:.2f}, Acc: {res["acc"]:.2f},\n'
            + f'Noisy mIoU: {res["noisy_miou"]:.2f}, mIoU: {res["miou"]:.2f},\n'
            + f'Noisy BAcc: {res["noisy_bacc"]:.2f}, BAcc: {res["bacc"]:.2f}\n'
        )
        results[args.inject_index] = (
            float(res["noisy_loss"].cpu().numpy()),
            float(res["loss"].cpu().numpy()),
            float(res["noisy_acc"]),
            float(res["acc"]),
            float(res["noisy_miou"]),
            float(res["miou"]),
            float(res["noisy_bacc"]),
            float(res["bacc"]),
        )
        json.dump(results, open(f"ckpt/{args.exp}_eval.json", "w"))


if __name__ == "__main__":
    main()
