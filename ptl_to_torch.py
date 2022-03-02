import argparse

import torch

from pytorch_scripts.utils import build_model, parse_args


def main():
    config_parser = parser = argparse.ArgumentParser(description='PytorchLighting to Pytorch', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments.')

    args = parse_args(parser, config_parser)

    # Build model (Resnet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
    n_classes = 10 if args.dataset == 'cifar10' else 100

    ptl_model = build_model(args.model, n_classes, optim_params, args.loss, args.inject_p,
                            args.order, args.activation, args.affine)

    model_path = "checkpoints/" + args.ckpt
    serialized_model = "checkpoints/" + args.ckpt.replace("ckpt", "ts")
    ptl_model.load_from_checkpoint(checkpoint_path=model_path, strict=False, model=args.model, n_classes=n_classes,
                                   optim=optim_params, loss=args.loss)
    # print(ptl_model)
    torch.save(ptl_model, serialized_model)


main()
