import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, n_classes, optim):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.n_classes = n_classes
        self.optim = optim

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optim['optimizer'] == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.optim['lr'], weight_decay=self.optim['wd'],
                            momentum=0.9)

        elif self.optim['optimizer'] == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.optim['lr'], weight_decay=self.optim['wd'])

        scheduler = CosineAnnealingLR(optimizer, self.optim['epochs'], eta_min=5e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_metrics(self, batch):
        x, y = batch

        # forward
        outputs = self(x)
        loss = self.criterion(outputs, get_one_hot(y, self.n_classes))
        _, preds = torch.max(outputs, 1)
        mask_correct_class = preds == y
        incorrect_classes = x[mask_correct_class], mask_correct_class
        acc = torch.sum(mask_correct_class) / x.shape[0]
        return loss, acc, incorrect_classes

    def training_step(self, train_batch, batch_idx):
        loss, acc, _ = self.get_metrics(train_batch)

        self.epoch_log('train_loss', loss)
        self.epoch_log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, acc, incorrect_class = self.get_metrics(val_batch)

        self.epoch_log('val_loss', loss)
        self.epoch_log('val_acc', acc)
        self.epoch_log('incorrect_vals', incorrect_class[0])
        return loss

    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.epoch_log('lr', lr)

    def epoch_log(self, name, value, prog_bar=True):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)


def get_one_hot(target, n_classes=10, device='cuda'):
    one_hot = torch.zeros(target.shape[0], n_classes, device=device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot
