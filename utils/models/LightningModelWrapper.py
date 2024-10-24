import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, n_classes=0, optim=None, loss="mse"):
        super(ModelWrapper, self).__init__()

        self.model = model
        self.n_classes = n_classes
        self.optim = optim

        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.use_one_hot = True
        elif loss == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.use_one_hot = False
        elif loss == "sce":
            self.criterion = SymmetricCELoss()
            self.use_one_hot = True
        elif loss == "mse":
            self.criterion = nn.MSELoss()
            self.use_one_hot = False

        self.miou = iouCalc(validClasses=range(n_classes))

        # self.save_hyperparameters("model", "n_classes", "optim", "loss")

    def forward(self, x, inject=True, inject_index=0):
        return self.model(x, inject, self.current_epoch, inject_index=inject_index)

    def configure_optimizers(self):
        if self.optim["optimizer"] == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=self.optim["lr"],
                weight_decay=self.optim["wd"],
                momentum=0.9,
            )

        elif self.optim["optimizer"] == "adamw":
            optimizer = AdamW(
                self.parameters(), lr=self.optim["lr"], weight_decay=self.optim["wd"]
            )

        scheduler = CosineAnnealingLR(optimizer, self.optim["epochs"], eta_min=1e-4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_metrics(self, batch, inject=True, inject_index=0):
        x, y = batch

        # forward
        outputs = self(x, inject, inject_index=inject_index)
        if self.n_classes == 0:
            outputs = outputs[0]

        # Loss
        if self.use_one_hot and not self.training:
            # bce or sce
            loss = self.criterion(outputs, get_one_hot(y, self.n_classes))
        else:
            # ce
            loss = self.criterion(outputs, y)
        if loss.isnan():
            print("NaN detected")
        elif loss > 1e4:
            print("Large Loss detected")

        # Accuracy
        if not self.training and self.n_classes > 0:
            probs, preds = torch.max(outputs, 1)
            acc = torch.mean((preds == y).float())
        else:
            acc, probs, preds = 0, 0, 0

        # Mean IoU
        if not self.training and self.n_classes > 0:
            self.miou.evaluateBatch(preds, y)
            miou = self.miou.outputScores()
        else:
            miou = 0

        return loss, acc, miou, (probs, preds)

    def training_step(self, train_batch, _, inject=True, inject_index=0):
        loss, acc, _ = self.get_metrics(
            train_batch, inject=inject, inject_index=inject_index
        )

        self.epoch_log("train_loss", loss)
        self.epoch_log("train_acc", acc)
        return loss

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

    def validation_step(self, val_batch, _, check_criticality=True, inject_index=0):
        # loss, acc, clean_vals = 0, 0, 0
        loss, acc, miou, clean_vals = self.get_metrics(
            val_batch, False, inject_index=inject_index
        )  ############# Comment it out
        noisy_loss, noisy_acc, noisy_miou, noisy_vals = self.get_metrics(
            val_batch, True, inject_index=inject_index
        )

        # Test the accuracy
        self.epoch_log("val_loss", loss)
        self.epoch_log("val_acc", acc)
        self.epoch_log("val_miou", miou)
        self.epoch_log("noisy_val_loss", noisy_loss)
        self.epoch_log("noisy_val_acc", noisy_acc)
        self.epoch_log("noisy_val_miou", noisy_miou)
        if check_criticality and self.n_classes > 0:
            self.check_criticality(gold=clean_vals, faulty=noisy_vals)

        return noisy_loss, loss, noisy_acc, acc, noisy_miou, miou

    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]["lr"]
        self.epoch_log("lr", lr)

    def epoch_log(self, name, value, prog_bar=True):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)


def get_one_hot(target, n_classes=10, device="cuda"):
    one_hot = torch.zeros(target.shape[0], n_classes, device=device)
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


class iouCalc:

    def __init__(self, validClasses, classLabels=None, voidClass=None):
        if classLabels is None:
            classLabels = list(map(str, validClasses))
        assert len(classLabels) == len(
            validClasses
        ), "Number of class ids and names must be equal"
        self.classLabels = classLabels
        self.validClasses = validClasses
        self.voidClass = voidClass
        self.evalClasses = [l for l in validClasses if l != voidClass]

        self.perImageStats = []
        self.nbPixels = 0
        self.confMatrix = np.zeros(
            shape=(len(self.validClasses), len(self.validClasses)), dtype=np.ulonglong
        )

        # Init IoU log files
        self.headerStr = "epoch, "
        for label in self.classLabels:
            if label.lower() != "void":
                self.headerStr += label + ", "

    def clear(self):
        self.perImageStats = []
        self.nbPixels = 0
        self.confMatrix = np.zeros(
            shape=(len(self.validClasses), len(self.validClasses)), dtype=np.ulonglong
        )

    def getIouScoreForLabel(self, label):
        # Calculate and return IOU score for a particular label (train_id)
        if label == self.voidClass:
            return float("nan")

        # the number of true positive pixels for this label
        # the entry on the diagonal of the confusion matrix
        tp = np.longlong(self.confMatrix[label, label])

        # the number of false negative pixels for this label
        # the row sum of the matching row in the confusion matrix
        # minus the diagonal entry
        fn = np.longlong(self.confMatrix[label, :].sum()) - tp

        # the number of false positive pixels for this labels
        # Only pixels that are not on a pixel with ground truth label that is ignored
        # The column sum of the corresponding column in the confusion matrix
        # without the ignored rows and without the actual label of interest
        notIgnored = [
            l for l in self.validClasses if not l == self.voidClass and not l == label
        ]
        fp = np.longlong(self.confMatrix[notIgnored, label].sum())

        # the denominator of the IOU score
        denom = tp + fp + fn
        if denom == 0:
            return float("nan")

        # return IOU
        return float(tp) / denom

    def evaluateBatch(self, predictionBatch, groundTruthBatch):
        # Calculate IoU scores for single batch
        assert predictionBatch.size(0) == groundTruthBatch.size(
            0
        ), "Number of predictions and labels in batch disagree."

        # Load batch to CPU and convert to numpy arrays
        predictionBatch = predictionBatch.cpu().numpy()
        groundTruthBatch = groundTruthBatch.cpu().numpy()

        for i in range(predictionBatch.shape[0]):
            predictionImg = predictionBatch[i, :, :]
            groundTruthImg = groundTruthBatch[i, :, :]

            # Check for equal image sizes
            assert (
                predictionImg.shape == groundTruthImg.shape
            ), "Image shapes do not match."
            assert (
                len(predictionImg.shape) == 2
            ), "Predicted image has multiple channels."

            imgWidth = predictionImg.shape[0]
            imgHeight = predictionImg.shape[1]
            nbPixels = imgWidth * imgHeight

            # Evaluate images
            encoding_value = (
                max(groundTruthImg.max(), predictionImg.max()).astype(np.int32) + 1
            )
            encoded = (groundTruthImg.astype(np.int32) * encoding_value) + predictionImg

            values, cnt = np.unique(encoded, return_counts=True)

            for value, c in zip(values, cnt):
                pred_id = value % encoding_value
                gt_id = int((value - pred_id) / encoding_value)
                if not gt_id in self.validClasses:
                    print("Unknown label with id {:}".format(gt_id))
                self.confMatrix[gt_id][pred_id] += c

            # Calculate pixel accuracy
            notIgnoredPixels = np.in1d(
                groundTruthImg, self.evalClasses, invert=True
            ).reshape(groundTruthImg.shape)
            erroneousPixels = np.logical_and(
                notIgnoredPixels, (predictionImg != groundTruthImg)
            )
            nbNotIgnoredPixels = np.count_nonzero(notIgnoredPixels)
            nbErroneousPixels = np.count_nonzero(erroneousPixels)
            self.perImageStats.append([nbNotIgnoredPixels, nbErroneousPixels])

            self.nbPixels += nbPixels

        return

    def outputScores(self, verbose=False):
        # Output scores over dataset
        assert (
            self.confMatrix.sum() == self.nbPixels
        ), "Number of analyzed pixels and entries in confusion matrix disagree: confMatrix {}, pixels {}".format(
            self.confMatrix.sum(), self.nbPixels
        )

        # Calculate IOU scores on class level from matrix
        classScoreList = []

        for c in self.evalClasses:
            iouScore = self.getIouScoreForLabel(c)
            classScoreList.append(iouScore)
        miou = self.getScoreAverage(classScoreList)

        return miou

    def getScoreAverage(self, scoreList):
        validScores = 0
        scoreSum = 0.0
        for score in scoreList:
            if not np.isnan(score):
                validScores += 1
                scoreSum += score
        if validScores == 0:
            return float("nan")
        return scoreSum / validScores
