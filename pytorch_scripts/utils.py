import yaml

import pytorch_lightning
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from .resnetCIFAR import *
from .LightningModelWrapper import ModelWrapper


class CifarDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, dataset='cifar10', data_dir='data', batch_size=128, num_gpus=1):
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.n_classes = None
        self.train_trans = None
        self.test_trans = None
        self.train_data = None
        self.test_data = None

    def prepare_data(self):

        if self.dataset == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.n_classes = 10
        elif self.dataset == 'cifar100':
            normalize = transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            self.n_classes = 100

        self.train_trans = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(), normalize])
        self.test_trans = transforms.Compose([transforms.ToTensor(), normalize])

        if self.dataset == 'cifar10':
            CIFAR10(root=self.data_dir, train=True, download=True)
            CIFAR10(root=self.data_dir, train=False, download=True)
        elif self.dataset == 'cifar100':
            CIFAR100(root=self.data_dir, train=True, download=True)
            CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.dataset == 'cifar10':
            self.train_data = CIFAR10(root=self.data_dir, train=True, transform=self.train_trans, download=True)
            self.test_data = CIFAR10(root=self.data_dir, train=False, transform=self.test_trans, download=True)
        elif self.dataset == 'cifar100':
            self.train_data = CIFAR100(root=self.data_dir, train=True, transform=self.train_trans, download=True)
            self.test_data = CIFAR100(root=self.data_dir, train=False, transform=self.test_trans, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size * self.num_gpus, num_workers=4 * self.num_gpus,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=200 * self.num_gpus, num_workers=4 * self.num_gpus)


def build_model(model='resnet20', n_classes=10, optim_params={}, loss='bce', inject_p=0.1):
    if model == 'resnet20':
        net = resnet20(n_classes, inject_p)
    elif model == 'resnet32':
        net = resnet32(n_classes, inject_p)
    elif model == 'resnet44':
        net = resnet44(n_classes, inject_p)
    elif model == 'resnet56':
        net = resnet56(n_classes), inject_p
    else:
        model = 'resnet20'
        net = resnet20(n_classes, inject_p)

    print(f'\n    {model} built.')
    return ModelWrapper(net, n_classes, optim_params, loss)


def _parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
