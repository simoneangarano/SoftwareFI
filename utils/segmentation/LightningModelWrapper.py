import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR, StepLR

from utils.segmentation.losses import FocalLoss
from utils.segmentation.stream_metrics import StreamSegMetrics


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, num_classes, optim, loss, freeze=False, inject_p=0.0):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.num_classes = num_classes
        self.optim = optim
        self.freeze = freeze

        self.real_p = inject_p

        self.clean_metrics = StreamSegMetrics(self.num_classes)
        self.noisy_metrics = StreamSegMetrics(self.num_classes)

        if loss == "ce":
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
        elif loss == "focal":
            self.criterion = FocalLoss(ignore_index=255, size_average=True)

        self.save_hyperparameters("model", "num_classes", "optim", "loss")

    def forward(self, x):
        return self.model(x)  # ['out']

    def configure_optimizers(self):
        if self.optim["optimizer"] == "sgd":
            optimizer = SGD(
                params=[
                    {
                        "params": self.model.model.backbone.parameters(),
                        "lr": 0.1 * self.optim["lr"],
                    },
                    {
                        "params": self.model.model.classifier.parameters(),
                        "lr": self.optim["lr"],
                    },
                ],
                lr=self.optim["lr"],
                momentum=0.9,
                weight_decay=self.optim["wd"],
            )

        if self.optim["scheduler"] == "poly":
            scheduler = PolynomialLR(optimizer, self.optim["epochs"], power=0.9)
        elif self.optim["scheduler"] == "step":
            scheduler = StepLR(
                optimizer, step_size=self.optim["epochs"] // 3, gamma=0.1
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        self.model.to_be_injected = True

        x, y = batch

        outputs = self(x)
        loss = self.criterion(outputs, y).mean()

        # self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx == 0:
            self.clean_metrics.reset()
            self.clean_loss = [0, 0]

        x, y = val_batch

        # clean
        outputs = self(x)
        self.clean_loss[1] += x.size(0)
        self.clean_loss[0] += self.criterion(outputs, y).mean().item()
        _, preds = torch.max(outputs, 1)

        self.clean_metrics.update(y.cpu().numpy(), preds.cpu().numpy())

    def on_validation_epoch_end(self):
        clean_miou = self.clean_metrics.get_results()["Mean IoU"]
        clean_loss = self.clean_loss[0] / self.clean_loss[1]
        # self.log("val_loss", clean_loss)
        # self.log("val_miou", clean_miou)

    def test_step(self, val_batch, batch_idx):
        if batch_idx == 0:
            self.clean_metrics.reset()
            self.noisy_metrics.reset()
            self.clean_loss = [0, 0]
            self.noisy_loss = [0, 0]

            self.model.p = 1.0

        x, y = val_batch

        # clean
        self.model.to_be_injected = False

        outputs = self(x)
        self.clean_loss[1] += x.size(0)
        self.clean_loss[0] += self.criterion(outputs, y).mean().item()
        _, preds = torch.max(outputs, 1)

        self.clean_metrics.update(y.cpu().numpy(), preds.cpu().numpy())

        # noisy
        self.model.to_be_injected = True

        outputs = self(x)
        self.noisy_loss[1] += x.size(0)
        self.noisy_loss[0] += self.criterion(outputs, y).mean().item()
        _, preds = torch.max(outputs, 1)

        self.noisy_metrics.update(y.cpu().numpy(), preds.cpu().numpy())

    def on_test_epoch_end(self):
        self.model.p = self.real_p

        clean_miou = self.clean_metrics.get_results()["Mean IoU"]
        noisy_miou = self.noisy_metrics.get_results()["Mean IoU"]

        clean_loss = self.clean_loss[0] / self.clean_loss[1]
        noisy_loss = self.noisy_loss[0] / self.noisy_loss[1]

        # self.check_criticality(gold=clean_vals, faulty=noisy_vals)

        # self.log("test_loss", clean_loss)
        # self.log("test_miou", clean_miou)
        # self.log("noisy_test_loss", noisy_loss)
        # self.log("noisy_test_miou", noisy_miou)

    def on_train_epoch_start(self):
        self.model.current_epoch = self.current_epoch
        lr = self.optimizers().param_groups[0]["lr"]
        self.epoch_log("lr", lr)

    def epoch_log(self, name, value, prog_bar=True):
        pass
        # self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)

    def check_criticality(self, gold: tuple, faulty: tuple):
        gold_vals, gold_preds = gold
        fault_vals, fault_preds = faulty
        # Magic number to define what is a zero
        err_lambda = 1e-4
        # Check if the sum of diffs are
        value_diff_pct = (
            torch.sum(torch.abs(gold_vals - fault_vals) > err_lambda)
            / gold_vals.shape[0]
        )
        preds_diff_pct = torch.sum(gold_preds != fault_preds) / gold_vals.shape[0]
        self.epoch_log("value_diff_pct", value_diff_pct)
        self.epoch_log("preds_diff_pct", preds_diff_pct)
