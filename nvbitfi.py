#!/usr/bin/python3
import argparse
import time

import configparser
import pandas as pd
import torch
import torchvision
import yaml

from pytorch_scripts.utils import build_model, parse_args


def load_cifar100(data_dir: str, transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load CIFAR 100 from <data dir>"""
    # Get a dataset
    test_set = torchvision.datasets.cifar.CIFAR100(root=data_dir, download=True, train=False,
                                                   transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    return test_loader


def load_cifar10(data_dir: str, transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load CIFAR 10 from <data dir>"""
    # Get a dataset
    test_set = torchvision.datasets.cifar.CIFAR10(root=data_dir, download=True, train=False,
                                                  transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    return test_loader


def load_ptl_model(args):
    # Build model (Resnet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
    n_classes = 10 if args.dataset == 'cifar10' else 100

    ptl_model = build_model(model=args.model, n_classes=n_classes, optim_params=optim_params,
                            loss=args.loss, inject_p=args.inject_p,
                            order=args.order, activation=args.activation, affine=args.affine)
    checkpoint = torch.load(args.ckpt)
    ptl_model.load_state_dict(checkpoint['state_dict'])
    return ptl_model.model


def perform_fault_injection_for_a_model(args):
    img_index = 0
    gold_path = args.goldpath
    generate = args.generate
    model = load_ptl_model(args=args)
    model.eval()
    model = model.to("cuda")
    load_data = load_cifar100
    if args.dataset == "cifar10":
        load_data = load_cifar10

    test_loader = load_data(data_dir="data",
                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                      torchvision.transforms.Normalize(
                                                                          mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]))
    # Get only the first 10 images
    images = list()
    for i, (image, label) in enumerate(test_loader):
        images.append((image, label))
        if i > 10:
            break
    gold_top1_label = gold_top1_prob = None
    gold_probabilities = torch.empty(0)
    if generate is False:
        gold_probabilities = torch.load(gold_path)
        gold_top1_label = int(torch.topk(gold_probabilities, k=1).indices.squeeze(0))
        gold_top1_prob = torch.softmax(gold_probabilities, dim=1)[0, gold_top1_label].item()

    total_time = time.time()
    with torch.no_grad():
        image, label = images[img_index]
        image_gpu = image.to("cuda")
        # Golden execution
        model_time = time.time()
        dnn_output = model(image_gpu, inject=False)
        model_time = time.time() - model_time

        probabilities = dnn_output.to("cpu")
        top1_label = int(torch.topk(probabilities, k=1).indices.squeeze(0))
        top1_prob = torch.softmax(probabilities, dim=1)[0, top1_label].item()

        cmp_gold_prob = torch.flatten(gold_probabilities)
        cmp_out_prob = torch.flatten(probabilities)
        # cmp_gold_prob[7] = 333333
        if generate is False and torch.any(torch.not_equal(cmp_gold_prob, cmp_out_prob)):
            print("SDC detected")
            for i, (g, f) in enumerate(zip(cmp_gold_prob, cmp_out_prob)):
                if g != f:
                    print(f"{i} e:{g} r:{f}")
            if gold_top1_label != top1_label:
                print("Critical SDC detected. "
                      f"e_label:{gold_top1_label} r_label:{top1_label} "
                      f"e_prob:{gold_top1_prob} r_prob:{top1_prob}")
    total_time = time.time() - total_time
    print(f"TOTAL TIME:{total_time:.2f} MODEL TIME:{model_time:.2f}")

    if generate is True:
        torch.save(probabilities, gold_path)


def main() -> None:
    config_parser = argparse.ArgumentParser(description='Criticality eval', add_help=False)
    config_parser.add_argument('--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments.')
    args, remaining_argv = config_parser.parse_known_args()

    defaults = {"option": "default"}

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        defaults.update(**cfg)

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[config_parser]
    )
    parser.set_defaults(**defaults)
    parser.add_argument('--generate', default=False, action="store_true",
                        help="Set this flag to generate the golds and reprogram the board")
    parser.add_argument('--goldpath', default="gold.pt", help="Gold path to save/load the gold file")
    args = parser.parse_args(remaining_argv)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print()
    perform_fault_injection_for_a_model(args)


if __name__ == '__main__':
    main()
