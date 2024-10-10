import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

import numpy as np
import mlstac
import pandas as pd
from .tiny_imagenet import TinyImageNet
from utils.utils import get_loader, build_model


class CifarDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset="cifar10", data_dir="data", batch_size=128, num_gpus=1, augs={}
    ):
        super().__init__()
        print(f"==> Loading {dataset} dataset..")
        # self.save_hyperparameters()
        self.dataset = dataset
        self.data_dir = data_dir
        self.size = 32 if "cifar" in dataset else 64  # TinyImagenet is 64x64
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.n_classes = None
        self.train_trans = None
        self.test_trans = None
        self.train_data = None
        self.test_data = None
        self.mixup_cutmix = augs["mixup_cutmix"]
        self.jitter = augs["jitter"]
        self.rand_aug = augs["rand_aug"]
        self.rand_erasing = augs["rand_erasing"]
        self.label_smooth = augs["label_smooth"]

        # Due to deprecation and future removal
        self.prepare_data_per_node = False

    def prepare_data(self):

        if self.dataset == "cifar10":
            CIFAR10(root=self.data_dir, train=True, download=True)
            CIFAR10(root=self.data_dir, train=False, download=True)
        elif self.dataset == "cifar100":
            CIFAR100(root=self.data_dir, train=True, download=True)
            CIFAR100(root=self.data_dir, train=False, download=True)
        elif self.dataset == "tinyimagenet":
            TinyImageNet(root=self.data_dir, split="train", download=True)
            TinyImageNet(root=self.data_dir, split="val", download=True)

    def setup(self, stage=None):
        if self.dataset == "cifar10":
            self.stats = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        elif self.dataset == "cifar100":
            self.stats = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
        elif self.dataset == "tinyimagenet":
            self.stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(self.stats[0], self.stats[1])
        self.test_trans = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        if self.dataset == "cifar10":
            self.train_data = CIFAR10(
                root=self.data_dir, train=True, transform=None, download=False
            )
            self.test_data = CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_trans,
                download=False,
            )
            self.n_classes = 10
        elif self.dataset == "cifar100":
            self.train_data = CIFAR100(
                root=self.data_dir, train=True, transform=None, download=False
            )
            self.test_data = CIFAR100(
                root=self.data_dir,
                train=False,
                transform=self.test_trans,
                download=False,
            )
            self.n_classes = 100
        elif self.dataset == "tinyimagenet":
            self.train_data = TinyImageNet(
                root=self.data_dir, split="train", download=False, transform=None
            )
            self.test_data = TinyImageNet(
                root=self.data_dir,
                split="val",
                download=False,
                transform=self.test_trans,
            )
            self.n_classes = 200

    def train_dataloader(self):
        return get_loader(
            self.train_data,
            self.batch_size // self.num_gpus,
            4 * self.num_gpus,
            self.n_classes,
            self.stats,
            self.mixup_cutmix,
            rand_erasing=self.rand_erasing,
            jitter=self.jitter,
            rand_aug=self.rand_aug,
            label_smooth=self.label_smooth,
            size=self.size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=200 * self.num_gpus,
            num_workers=4 * self.num_gpus,
            shuffle=False, pin_memory=True,
        )


class CoreDataset(torch.utils.data.DataLoader):
    def __init__(self, subset: pd.DataFrame, args=None):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset
        self.args = args
        if self.args is not None and self.args.num_classes == 0:
            # unsuperivised feature comparison
            self.model = build_model(
                model=args.model,
                n_classes=args.num_classes,
                optim_params={},
                loss=args.loss,
                error_model=args.error_model,
                inject_p=0,
                inject_epoch=args.inject_epoch,
                order=args.order,
                activation="relu",
                nan=args.nan,
                affine=args.affine,
                ckpt=args.ckpt,
            )
            self.model.eval()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        # Retrieve the data from HuggingFace
        sample = mlstac.get_data(dataset=self.subset.iloc[index], quiet=True).squeeze()

        # Load the Sentinel-2 all bands
        # We set <0:32> to make it faster and run in CPU
        X = sample[[1, 2, 3, 7, 10], ...].astype(np.float32) / 10_000

        # Load the target
        if self.args is not None and self.args.num_classes == 0:
            y, _ = self.model(torch.tensor(X[None, ...]), inject=False)
            y = y.squeeze().detach().cpu().numpy()
        else:
            y = sample[13, ...].astype(np.int64)

        return X, y


class CoreDataModule(pl.LightningDataModule):
    def __init__(self, args=None, batch_size: int = 4):
        super().__init__()

        # Load the metadata from the MLSTAC Collection file
        metadata = mlstac.load(snippet="isp-uv-es/CloudSEN12Plus").metadata

        # Split the metadata into train, validation and test sets
        self.train_dataset = metadata[
            (metadata["split"] == "test")
            & (metadata["label_type"] == "high")
            & (metadata["proj_shape"] == 509)
        ]
        self.validation_dataset = metadata[
            (metadata["split"] == "validation")
            & (metadata["label_type"] == "high")
            & (metadata["proj_shape"] == 509)
        ]
        self.test_dataset = metadata[
            (metadata["split"] == "test")
            & (metadata["label_type"] == "high")
            & (metadata["proj_shape"] == 509)
        ]

        # Define the batch_size
        self.batch_size = batch_size
        self.args = args

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.train_dataset, args=self.args),
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True, pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.validation_dataset, args=self.args),
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=False, pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset, args=self.args),
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=False, pin_memory=True,
        )
