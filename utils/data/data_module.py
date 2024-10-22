import os
import rasterio
import numpy as np
import mlstac
import pandas as pd

import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

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
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=200 * self.num_gpus,
            num_workers=4 * self.num_gpus,
            shuffle=False,
            pin_memory=True,
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
                activation=args.activation,
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
            y[y == 2] = 1
            y[y == 3] = 2
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
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.validation_dataset, args=self.args),
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset, args=self.args),
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )


# CloudSen12 dataset
class CloudSen12(torch.utils.data.Dataset):

    def __init__(
        self,
        path: str = "/media/SSD2/inzerillo/datasets/cloudsen12",
        normalization=True,
        split=None,
        aggregate_labels=False,
    ):

        self.normalization = normalization

        self.path = path

        self.folders = os.listdir(self.path)

        self.aggregate_labels = aggregate_labels

        # split could be "train", "val" or "test"
        self.split = split

        if split is not None:
            # CSV with metadata
            # path/cloudsen12_metadata.csv
            metadata_path = path + "/cloudsen12_metadata.csv"

            # Read the metadata
            metadata = pd.read_csv(metadata_path)

            # Keep only "roi_id", "s2_id_gee", "test" fields
            metadata = metadata[["roi_id", "test", "s2_id_gee", "label_type"]]

            metadata = metadata[metadata["label_type"] == "high"]

            # Train split -> test = 0
            # Val split -> test = 1
            # Test split -> test = 2
            if split == "train":
                metadata = metadata[metadata["test"] == 0]
            elif split == "val":
                metadata = metadata[metadata["test"] == 1]
            elif split == "test":
                metadata = metadata[metadata["test"] == 2]

            # Delete from metadata the row with "roi_id" that is not present in self.folders
            metadata = metadata[metadata["roi_id"].isin(self.folders)]

            self.folders = metadata["roi_id"] + "/" + metadata["s2_id_gee"]

            # Delete duplicates from the list
            self.folders = list(dict.fromkeys(self.folders))

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):

        # Get image and label paths
        folder_path = os.path.join(self.path, self.folders[index])
        img_path = folder_path + "/S2L1C.tif"
        label_path = folder_path + "/labels/manual_hq.tif"

        # Open the images using rasterio
        with rasterio.open(img_path) as img:
            b02 = img.read(2)  # Band 2 = Blue
            b03 = img.read(3)  # Band 3 = Green
            b04 = img.read(4)  # Band 4 = Red
            b08 = img.read(8)  # Band 8 = NIR
            b11 = img.read(11)  # Band 11 = SWIR1

            # Stack the bands
            img_image = np.stack([b02, b03, b04, b08, b11], axis=0).astype(np.float32)

        with rasterio.open(label_path) as label:
            # 0 = Background
            # 1 = Thick Clouds
            # 2 = Thin Clouds
            # 3 = Shadows
            label_image = label.read().astype(np.float32)

            if self.aggregate_labels:
                # 0 = Background
                # 1 = Clouds (Thick + Thin)
                # 2 = Shadows
                label_image[label_image == 2] = 1
                label_image[label_image == 3] = 2

        # From numpy to torch tensor
        img_tensor = torch.from_numpy(img_image)
        label_tensor = torch.from_numpy(label_image)

        # Normalize
        if self.normalization:
            img_tensor = (img_tensor - torch.min(img_tensor)) / (
                torch.max(img_tensor) - torch.min(img_tensor)
            )

        # Original image size is 509 x 509
        # Padding from 509 x 509 to 512 x 512
        # Padding with zeros
        img_tensor = torch.cat((torch.zeros((5, 1, 509)), img_tensor), dim=1)
        img_tensor = torch.cat((img_tensor, torch.zeros((5, 2, 509))), dim=1)
        img_tensor = torch.cat((torch.zeros((5, 512, 1)), img_tensor), dim=2)
        img_tensor = torch.cat((img_tensor, torch.zeros((5, 512, 2))), dim=2)

        label_tensor = torch.cat((torch.zeros((1, 1, 509)), label_tensor), dim=1)
        label_tensor = torch.cat((label_tensor, torch.zeros((1, 2, 509))), dim=1)
        label_tensor = torch.cat((torch.zeros((1, 512, 1)), label_tensor), dim=2)
        label_tensor = torch.cat((label_tensor, torch.zeros((1, 512, 2))), dim=2)

        return img_tensor, label_tensor
