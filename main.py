import argparse

# import torch
# torch.autograd.set_detect_anomaly(True)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# DieHardNET packages
from pytorch_scripts.utils import build_model, CifarDataModule, _parse_args
from hg_noise_injector.hans_gruber import HansGruberNI


config_parser = parser = argparse.ArgumentParser(description='Configuration', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--name', default='c10_resnet20_injection_test_0', help='Experiment name.')
parser.add_argument('--mode', default='validate', help='train or validation/validate')
parser.add_argument('--ckpt', default='c10_resnet20_base-epoch=155-val_acc=0.92.ckpt', help='Pass the name of a checkpoint to resume training.')
parser.add_argument('--dataset', default='cifar10', help='Dataset name: cifar10 or cifar100.')
parser.add_argument('--data_dir', default='./data', help='Path to dataset.')
parser.add_argument('--num_gpus', default=1, help='Number of GPUs used.')
parser.add_argument('--model', default='resnet20', help='Network name. Resnets only for now.')
parser.add_argument('--epochs', default=160, help='Number of epochs.')
parser.add_argument('--batch_size', default=128, help='Batch Size')
parser.add_argument('--lr', default=1e-1, help='Learning rate.')
parser.add_argument('--wd', default=1e-4, help='Weight Decay.')
parser.add_argument('--optimizer', default='sgd', help='Optimizer name, adamw or sgd.')
parser.add_argument('--seed', default=0, help='Random Seed for reproducibility.')
parser.add_argument('--comment', default='ResNet trained with original settings but the scheduler',
                    help='Optional comment.')


def main():
    args, args_text = _parse_args(parser, config_parser)

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    print('==> Loading dataset..')
    cifar = CifarDataModule(args.dataset, args.data_dir, args.batch_size, args.num_gpus)

    # Build model (Resnet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
    n_classes = 10 if args.dataset == 'cifar10' else 100
    net = build_model(args.model, n_classes, optim_params)

    # W&B logger
    wandb_logger = WandbLogger(project="NeutronRobustness", name=args.name, id=args.name, entity="neutronstrikesback")
    wandb_logger.log_hyperparams(args)
    wandb_logger.watch(net, log_graph=False)

    # Callbacks
    ckpt_callback = ModelCheckpoint('checkpoints/', filename=args.name + '-{epoch:02d}-{val_acc:.2f}',
                                    mode='max', monitor='val_acc')
    callbacks = [ckpt_callback]

    # Pytorch-Lightning Trainer
    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.num_gpus, callbacks=callbacks, logger=wandb_logger,
                         deterministic=True, benchmark=True, accelerator='gpu', strategy="dp", sync_batchnorm=True)

    if args.ckpt:
        args.ckpt = 'checkpoints/' + args.ckpt
    if args.mode == 'train' or args.mode == 'training':
        trainer.fit(net, cifar, ckpt_path=args.ckpt)
    elif args.mode == 'validation' or args.mode == 'validate':
        trainer.validate(net, cifar, ckpt_path=args.ckpt)
    else:
        print('ERROR: select a suitable mode "train/training" or "validation/validate".')


if __name__ == '__main__':
    main()
