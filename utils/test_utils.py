import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import mlstac


def load_fi_weights(model, filename, verbose=False):
    count = 0
    new_dict = {}
    weights = torch.load(filename, weights_only=True)
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if verbose:
            print(name, param.data.shape)
        if "dummy" in name:
            if verbose:
                print("-\n")
            continue
        new_name = name.replace("conv.weight", "weight").replace("conv.bias", "bias")
        new_name = new_name.replace("linear.weight", "weight").replace(
            "linear.bias", "bias"
        )
        new_name = new_name.replace(".layers", "")
        new_weights = weights[new_name]
        if verbose:
            print(new_name, new_weights.shape, "\n")
        count += 1
        if param.data.shape != new_weights.shape:
            raise ValueError(
                f"Shape mismatch: {param.data.shape} != {new_weights.shape}"
            )
        new_dict[name] = new_weights

    print(f"Loaded {count} weights")
    model.load_state_dict(new_dict, strict=False)
    return model


class CoreDataset(torch.utils.data.DataLoader):
    def __init__(self, subset: pd.DataFrame):
        subset.reset_index(drop=True, inplace=True)
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int):
        # Retrieve the data from HuggingFace
        sample = mlstac.get_data(dataset=self.subset.iloc[index], quiet=True).squeeze()

        # Load the Sentinel-2 all bands
        # We set <0:32> to make it faster and run in CPU
        X = sample[[1, 2, 3, 7, 10], ...].astype(np.float32) / 10_000

        # Load the target
        y = sample[13, ...].astype(np.int64)

        return X, y


class CoreDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4):
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

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.train_dataset),
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.validation_dataset),
            num_workers=4,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=CoreDataset(self.test_dataset),
            num_workers=4,
            batch_size=self.batch_size,
        )


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        _, intermediates = self.model(
            x,
            inject=inject,
            current_epoch=current_epoch,
            counter=counter,
            inject_index=inject_index,
        )
        return intermediates[-1]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        x, y = batch
        f = self(x, inject=False)
        f_hat = self(x)
        loss = self.loss(f, f_hat)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
