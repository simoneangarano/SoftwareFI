import os

import mlstac
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio
import torch

from utils.utils import build_model


class CoreDataset(torch.utils.data.DataLoader):
    def __init__(self, subset: pd.DataFrame, args=None):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset
        self.args = args
        if self.args is not None and self.args.num_classes == 0:
            # unsuperivised feature comparison
            self.model = build_model(args).cuda()
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
    def __init__(self, args=None):
        super().__init__()

        # Load the metadata from the MLSTAC Collection file
        metadata = mlstac.load(snippet="data/main.json").metadata

        # Split the metadata into train, validation and test sets
        self.train_dataset = metadata[
            (metadata["split"] == "test")
            & (metadata["label_type"] == "high")
            & (metadata["proj_shape"] == 509)
        ]
        self.validation_dataset = metadata[
            (metadata["split"] == "test")
            & (metadata["label_type"] == "high")
            & (metadata["proj_shape"] == 509)
        ]
        self.test_dataset = metadata[
            (metadata["split"] == "test")
            & (metadata["label_type"] == "high")
            & (metadata["proj_shape"] == 509)
        ]

        self.args = args

        self.gen = torch.Generator()
        self.gen.manual_seed(args.seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.train_dataset, args=self.args),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=self.args.pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=self.gen,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.validation_dataset, args=self.args),
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last,
            worker_init_fn=seed_worker,
            generator=self.gen,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset, args=self.args),
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=self.args.pin_memory,
            drop_last=self.args.drop_last,
            worker_init_fn=seed_worker,
            generator=self.gen,
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


def seed_worker(worker_id):
    worker_seed = torch.seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
