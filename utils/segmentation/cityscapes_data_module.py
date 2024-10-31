import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .cityscapes import Cityscapes
from .transforms import ExtCompose, ExtNormalize, ExtToTensor
from .utils import get_loader


def get_data_dir(data_dir, dataset):
    assert dataset == "cityscapes"
    return os.path.join(data_dir, "Cityscapes")


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset="cityscapes",
        data_dir="data",
        batch_size=128,
        num_gpus=1,
        num_workers=None,
        size=256,
        fp16=True,
        augs={},
    ):
        super().__init__()
        print(f"==> Loading {dataset} dataset..")
        self.save_hyperparameters()
        self.dataset = dataset
        self.fp16 = fp16
        self.data_dir = get_data_dir(data_dir, dataset)
        self.size = size
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.num_classes = None
        self.train_trans = None
        self.test_trans = None
        self.train_data = None
        self.test_data = None
        self.mixup_cutmix = augs["mixup_cutmix"]
        self.jitter = augs["jitter"]
        self.rand_aug = augs["rand_aug"]
        self.rand_erasing = augs["rand_erasing"]
        self.label_smooth = augs["label_smooth"]
        self.rcc = augs["rcc"]
        self.num_workers = num_workers or 8

        # Due to deprecation and future removal
        self.prepare_data_per_node = False

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.test_trans = ExtCompose(
            [
                ExtToTensor(),
                ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_data = Cityscapes(
            root=self.data_dir, split="train", mode="fine", transform=None
        )
        self.test_data = Cityscapes(
            root=self.data_dir, split="val", mode="fine", transform=self.test_trans
        )
        self.num_classes = 19

    def train_dataloader(self):
        return get_loader(
            self.dataset,
            self.train_data,
            self.batch_size // self.num_gpus,
            self.num_workers,
            self.num_classes,
            self.stats,
            self.mixup_cutmix,
            rand_erasing=self.rand_erasing,
            jitter=self.jitter,
            rand_aug=self.rand_aug,
            label_smooth=self.label_smooth,
            rcc=self.rcc,
            size=self.size,
            fp16=self.fp16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
