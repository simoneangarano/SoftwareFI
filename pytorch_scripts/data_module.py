import pytorch_lightning
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from .tiny_imagenet import TinyImageNet

from pytorch_scripts.utils import get_loader


class CifarDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self, dataset="cifar10", data_dir="data", batch_size=128, num_gpus=1, augs={}
    ):
        super().__init__()
        print(f"==> Loading {dataset} dataset..")
        # self.save_hyperparameters()
        self.dataset = dataset
        self.data_dir = data_dir
        self.size = 32 if "cifar" in dataset else 64  # TinyImagenet is 64x64
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.n_classes = None
        self.train_trans = None
        self.test_trans = None
        self.train_data = None
        self.test_data = None
        self.mixup_cutmix = augs["mixup_cutmix"]
        self.jitter = augs["jitter"]
        self.rand_aug = augs["rand_aug"]
        self.rand_erasing = augs["rand_erasing"]
        self.label_smooth = augs["label_smooth"]

        # Due to deprecation and future removal
        self.prepare_data_per_node = False

    def prepare_data(self):

        if self.dataset == "cifar10":
            CIFAR10(root=self.data_dir, train=True, download=True)
            CIFAR10(root=self.data_dir, train=False, download=True)
        elif self.dataset == "cifar100":
            CIFAR100(root=self.data_dir, train=True, download=True)
            CIFAR100(root=self.data_dir, train=False, download=True)
        elif self.dataset == "tinyimagenet":
            TinyImageNet(root=self.data_dir, split="train", download=True)
            TinyImageNet(root=self.data_dir, split="val", download=True)

    def setup(self, stage=None):
        if self.dataset == "cifar10":
            self.stats = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        elif self.dataset == "cifar100":
            self.stats = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
        elif self.dataset == "tinyimagenet":
            self.stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(self.stats[0], self.stats[1])
        self.test_trans = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        if self.dataset == "cifar10":
            self.train_data = CIFAR10(
                root=self.data_dir, train=True, transform=None, download=False
            )
            self.test_data = CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_trans,
                download=False,
            )
            self.n_classes = 10
        elif self.dataset == "cifar100":
            self.train_data = CIFAR100(
                root=self.data_dir, train=True, transform=None, download=False
            )
            self.test_data = CIFAR100(
                root=self.data_dir,
                train=False,
                transform=self.test_trans,
                download=False,
            )
            self.n_classes = 100
        elif self.dataset == "tinyimagenet":
            self.train_data = TinyImageNet(
                root=self.data_dir, split="train", download=False, transform=None
            )
            self.test_data = TinyImageNet(
                root=self.data_dir,
                split="val",
                download=False,
                transform=self.test_trans,
            )
            self.n_classes = 200

    def train_dataloader(self):
        return get_loader(
            self.train_data,
            self.batch_size // self.num_gpus,
            4 * self.num_gpus,
            self.n_classes,
            self.stats,
            self.mixup_cutmix,
            rand_erasing=self.rand_erasing,
            jitter=self.jitter,
            rand_aug=self.rand_aug,
            label_smooth=self.label_smooth,
            size=self.size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=200 * self.num_gpus,
            num_workers=4 * self.num_gpus,
        )
