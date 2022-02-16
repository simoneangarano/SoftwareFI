import pytorch_lightning
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


class CifarDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, dataset='cifar10', data_dir='data', batch_size=128, num_gpus=1):
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.n_classes = None
        self.train_trans = None
        self.test_trans = None
        self.train_data = None
        self.test_data = None

    def prepare_data(self):

        if self.dataset == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.n_classes = 10
        elif self.dataset == 'cifar100':
            normalize = transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            self.n_classes = 100

        self.train_trans = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor(), normalize])
        self.test_trans = transforms.Compose([transforms.ToTensor(), normalize])

        if self.dataset == 'cifar10':
            CIFAR10(root=self.data_dir, train=True, download=True)
            CIFAR10(root=self.data_dir, train=False, download=True)
        elif self.dataset == 'cifar100':
            CIFAR100(root=self.data_dir, train=True, download=True)
            CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.dataset == 'cifar10':
            self.train_data = CIFAR10(root=self.data_dir, train=True, transform=self.train_trans, download=True)
            self.test_data = CIFAR10(root=self.data_dir, train=False, transform=self.test_trans, download=True)
        elif self.dataset == 'cifar100':
            self.train_data = CIFAR100(root=self.data_dir, train=True, transform=self.train_trans, download=True)
            self.test_data = CIFAR100(root=self.data_dir, train=False, transform=self.test_trans, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size * self.num_gpus, num_workers=4 * self.num_gpus,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=200 * self.num_gpus, num_workers=4 * self.num_gpus)
