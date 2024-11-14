import mlstac
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
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


def seed_worker(worker_id):
    worker_seed = torch.seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
