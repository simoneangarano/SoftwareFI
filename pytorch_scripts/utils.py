from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from resnetCIFAR import *
from LightningModelWrapper import ModelWrapper


def get_dataset(dataset='cifar10', data_dir='data', batch_size=128, num_gpus=1):

    if dataset == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        n_classes = 10
    elif dataset == 'cifar100':
        normalize = transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
        n_classes = 100

    train_trans = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(), normalize])
    test_trans = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'cifar10':
        train_data = CIFAR10(root=data_dir, train=True, transform=train_trans, download=True)
        test_data = CIFAR10(root=data_dir, train=False, transform=test_trans, download=False)
    elif dataset == 'cifar100':
        train_data = CIFAR100(root=data_dir, train=True, transform=train_trans, download=True)
        test_data = CIFAR100(root=data_dir, train=False, transform=test_trans, download=False)

    train_data = DataLoader(train_data, batch_size=batch_size * num_gpus, shuffle=True,
                            num_workers=4 * num_gpus, drop_last=True)
    test_data = DataLoader(test_data, batch_size=250 * num_gpus, shuffle=False,
                           num_workers=4 * num_gpus, drop_last=False)

    return train_data, test_data, n_classes


def build_model(model='resnet20', n_classes=10, optim_params={}):
    if model == 'resnet20':
        net = resnet20(n_classes)
    elif model == 'resnet32':
        net = resnet32(n_classes)
    elif model == 'resnet44':
        net = resnet44(n_classes)
    elif model == 'resnet56':
        net = resnet56(n_classes)
    else:
        model = 'resnet20'
        net = resnet20(n_classes)

    print(f'\n    {model} built.')
    return ModelWrapper(net, n_classes, optim_params)
