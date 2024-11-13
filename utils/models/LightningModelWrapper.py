import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.segmentation.stream_metrics import StreamSegMetrics


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, args):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.num_classes = args.num_classes
        self.optim = None

        if args.loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.use_one_hot = True
        elif args.loss == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.use_one_hot = False
        elif args.loss == "sce":
            self.criterion = SymmetricCELoss()
            self.use_one_hot = True
        elif args.loss == "mse":
            self.criterion = nn.MSELoss()
            self.use_one_hot = False

        self.metrics = StreamSegMetrics(self.num_classes, ignore_index=None)
        # self.save_hyperparameters("model", "num_classes", "optim", "loss")

    def forward(self, x, inject=True, inject_index=0):
        return self.model(
            x,
            inject=inject,
            current_epoch=self.current_epoch,
            inject_index=inject_index,
        )

    def configure_optimizers(self):
        if self.optim["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.optim["lr"],
                weight_decay=self.optim["wd"],
                momentum=0.9,
            )

        elif self.optim["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.optim["lr"], weight_decay=self.optim["wd"]
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.optim["epochs"], eta_min=1e-4
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_metrics(self, batch, inject, inject_index=0):
        x, y = batch
        metrics = {"y": y}

        # forward
        outputs = self(x, inject=inject, inject_index=inject_index)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Loss
        if self.use_one_hot and not self.training:
            metrics["loss"] = self.criterion(outputs, get_one_hot(y, self.num_classes))
        else:
            metrics["loss"] = self.criterion(outputs, y)

        # Accuracy and Mean IoU
        if not self.training and self.num_classes > 0:
            metrics["probs"], metrics["preds"] = torch.max(outputs, 1)

            self.metrics.update(y.cpu().numpy(), metrics["preds"].cpu().numpy())
            results = self.metrics.get_results()
            self.metrics.reset()

            metrics["acc"] = results["Overall Acc"]
            metrics["miou"] = results["Mean IoU"]
            metrics["bacc"] = results["Mean Acc"]

        return metrics

    def training_step(self, train_batch, _, inject=True, inject_index=0):
        metrics = self.get_metrics(
            train_batch, inject=inject, inject_index=inject_index
        )

        self.epoch_log("train_loss", metrics["loss"])
        self.epoch_log("train_acc", metrics["acc"])
        self.epoch_log("train_miou", metrics["miou"])
        self.epoch_log("train_bacc", metrics["bacc"])
        return metrics

    def check_criticality(self, gold: tuple, faulty: tuple):
        gold_vals, gold_preds = gold
        fault_vals, fault_preds = faulty
        # Magic number to define whats is an zero
        err_lambda = 1e-4
        # Check if the sum of diffs are
        value_diff_pct = (
            torch.sum(torch.abs(gold_vals - fault_vals) > err_lambda)
            / gold_vals.shape[0]
        )
        preds_diff_pct = torch.sum(gold_preds != fault_preds) / gold_vals.shape[0]
        self.epoch_log("value_diff_pct", value_diff_pct)
        self.epoch_log("preds_diff_pct", preds_diff_pct)

    @torch.no_grad()
    def validation_step(self, val_batch, inject_index=0, check_criticality=False):
        metrics = self.get_metrics(val_batch, inject=False, inject_index=inject_index)
        noisy_metrics = self.get_metrics(
            val_batch, inject=True, inject_index=inject_index
        )

        # Test the accuracy
        # self.epoch_log("val_loss", loss)
        # self.epoch_log("val_acc", acc)
        # self.epoch_log("val_miou", miou)
        # self.epoch_log("noisy_val_loss", noisy_loss)
        # self.epoch_log("noisy_val_acc", noisy_acc)
        # self.epoch_log("noisy_val_miou", noisy_miou)
        if check_criticality and self.num_classes > 0:
            self.check_criticality(
                gold=(metrics["probs"], metrics["preds"]),
                faulty=(noisy_metrics["probs"], noisy_metrics["preds"]),
            )

        return noisy_metrics, metrics

    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]["lr"]
        self.epoch_log("lr", lr)

    def epoch_log(self, name, value, prog_bar=True):
        pass
        # self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)


def get_one_hot(target, num_classes=10, device="cuda"):
    one_hot = torch.zeros(target.shape[0], num_classes, device=device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.0)
    return one_hot


class SymmetricCELoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(SymmetricCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nnlloss = nn.NLLLoss()
        self.Softmax = nn.Softmax()

    def negative_log_likelihood(self, inputs, targets):
        return -torch.sum(targets * torch.log(inputs + 1e-6)) / inputs.shape[0]

    def forward(self, inputs, targets):
        inputs = self.Softmax(inputs)
        # standard crossEntropy
        ce = self.negative_log_likelihood(inputs, targets)
        # reverse crossEntropy
        rce = self.negative_log_likelihood(targets, inputs)
        return ce * self.alpha + rce * self.beta
