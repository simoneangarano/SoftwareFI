import argparse
import os
import re

import torch

from pytorch_scripts.utils import build_model, parse_args


def get_max_values(name):
    filenames = next(os.walk("checkpoints/"), (None, None, []))[2]  # [] if no file
    pattern = fr'{name}-epoch=(\d+)-val_acc=(\S+).ckpt'

    for file in filenames:
        m = re.match(pattern, file)
        if m:
            return m.groups()
    return None, None


def main():
    config_parser = parser = argparse.ArgumentParser(description='Configuration', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments.')

    args, args_text = parse_args(parser, config_parser)
    print(args_text)
    # Build model (Resnet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
    n_classes = 10 if args.dataset == 'cifar10' else 100

    ptl_model = build_model(args.model, n_classes, optim_params, args.loss, args.inject_p,
                            args.inject_epoch, args.order, args.activation, args.affine)

    #  filename=args.name + '-{epoch:02d}-{val_acc:.2f}'
    max_epoch, max_val_acc = get_max_values(name=args.name)
    checkpoint_base_name = f"{args.name}-epoch={max_epoch}-val_acc={max_val_acc}"
    model_path = f"checkpoints/{checkpoint_base_name}.ckpt"
    serialized_model = f"data/{checkpoint_base_name}.ts"
    print(serialized_model)
    ptl_model.load_from_checkpoint(checkpoint_path=model_path, strict=False, model=args.model, n_classes=n_classes,
                                   optim=optim_params, loss=args.loss)
    # print(ptl_model)
    torch.save(ptl_model, serialized_model)


main()
