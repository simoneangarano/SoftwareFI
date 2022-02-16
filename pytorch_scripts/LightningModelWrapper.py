import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, n_classes, optim, loss):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.n_classes = n_classes
        self.optim = optim

        if loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
            self.use_one_hot = True
        elif loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
            self.use_one_hot = False
        elif loss == 'sce':
            self.criterion = SymmetricCELoss()
            self.use_one_hot = True

    def forward(self, x, inject=True):
        return self.model(x, inject, self.current_epoch)

    def configure_optimizers(self):
        if self.optim['optimizer'] == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.optim['lr'], weight_decay=self.optim['wd'],
                            momentum=0.9)

        elif self.optim['optimizer'] == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.optim['lr'], weight_decay=self.optim['wd'])

        scheduler = CosineAnnealingLR(optimizer, self.optim['epochs'], eta_min=5e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_metrics(self, batch, inject=True):
        x, y = batch

        # forward
        outputs = self(x, inject)
        # loss
        if self.use_one_hot:
            # bce or sce
            loss = self.criterion(outputs, get_one_hot(y, self.n_classes))
        else:
            # ce
            loss = self.criterion(outputs, y)
        # accuracy
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == y) / x.shape[0]
        return loss, acc

    def training_step(self, train_batch, batch_idx):
        loss, acc = self.get_metrics(train_batch)

        self.epoch_log('train_loss', loss)
        self.epoch_log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, acc = self.get_metrics(val_batch, False)
        noisy_loss, noisy_acc = self.get_metrics(val_batch)

        self.epoch_log('val_loss', loss)
        self.epoch_log('val_acc', acc)
        self.epoch_log('noisy_val_loss', noisy_loss)
        self.epoch_log('noisy_val_acc', noisy_acc)
        return noisy_loss

    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.epoch_log('lr', lr)

    def epoch_log(self, name, value, prog_bar=True):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)


def get_one_hot(target, n_classes=10, device='cuda'):
    one_hot = torch.zeros(target.shape[0], n_classes, device=device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class SymmetricCELoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(SymmetricCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nnlloss = nn.NLLLoss()
        self.Softmax = nn.Softmax()

    def negative_log_likelihood(self, inputs, targets):
        return - torch.sum(targets * torch.log(inputs + 1e-6)) / inputs.shape[0]

    def forward(self, inputs, targets):
        inputs = self.Softmax(inputs)
        # standard crossEntropy
        ce = self.negative_log_likelihood(inputs, targets)
        # reverse crossEntropy
        rce = self.negative_log_likelihood(targets, inputs)
        return ce * self.alpha + rce * self.beta
