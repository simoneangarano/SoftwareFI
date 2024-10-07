import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import mlstac


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
