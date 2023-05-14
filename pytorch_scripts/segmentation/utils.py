import yaml
import torch
import torch.nn as nn
from .transforms import ExtCompose, ExtRandomCrop, ExtColorJitter, ExtRandomHorizontalFlip, ExtToTensor, ExtNormalize
from torch.utils.data import DataLoader

from .LightningModelWrapper import ModelWrapper

def build_model(model=None, n_classes=10, optim_params={}, loss='bce', error_model='random', inject_p=0.0, inject_epoch=0,
                clip=False, nan=False, freeze=False, pretrained=False, activation='max'):

    if model == 'deeplab':
        from .deeplabv3_custom.deeplab import deeplabv3_resnet101
        net = deeplabv3_resnet101(n_classes, pretrained=pretrained)
    
    elif model == 'deeplab_relumax':
        from .deeplabv3_custom.deeplab_robust import deeplabv3_resnet101
        net = deeplabv3_resnet101(n_classes, pretrained=pretrained, activation=activation)

    # Hook clipping and NaNs
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if clip:
                m.register_forward_hook(lambda module, input, output : torch.clip(output, -6, 6))
            if nan:
                m.register_forward_hook(lambda module, input, output : torch.nan_to_num(output, 0.0))

    print(f'\n==> {model} built.')
    return ModelWrapper(net, n_classes, optim_params, loss, freeze, inject_p)


def get_loader(dataset_name, data, batch_size=128, workers=4, n_classes=100, stats=None, mixup_cutmix=True, rand_erasing=0.0,
               label_smooth=0.1, rand_aug='rand-m9-mstd0.5-inc1', jitter=0.0, rcc=0.75, size=32, fp16=True):
    # Segmentation
    assert dataset_name == 'cityscapes'

    # Setup: https://arxiv.org/pdf/1706.05587.pdf (CVPR 2017)
    train_transform = ExtCompose([
        ExtRandomCrop(size=(size, size)),
        ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        ExtRandomHorizontalFlip(),
        ExtToTensor(),
        ExtNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    data.transform = train_transform
    return DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=True)

def parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args

