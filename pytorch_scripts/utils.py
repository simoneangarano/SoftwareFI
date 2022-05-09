import yaml

from .LightningModelWrapper import ModelWrapper
from .hard_resnet import *
from .hard_densenet import *


def build_model(model='hard_resnet20', n_classes=10, optim_params={}, loss='bce', error_model='random', inject_p=0.1, inject_epoch=0,
                order='relu-bn', activation='relu', affine=True):
    if model == 'hard_resnet20':
        net = hard_resnet20(n_classes, error_model, inject_p, inject_epoch, order, activation, affine)
    elif model == 'hard_resnet32':
        net = hard_resnet32(n_classes, error_model, inject_p, inject_epoch, order, activation, affine)
    elif model == 'hard_resnet44':
        net = hard_resnet44(n_classes, error_model, inject_p, inject_epoch, order, activation, affine)
    elif model == 'densenet100':
        net = densenet100(n_classes)
    else:
        model = 'hard_resnet20'
        net = hard_resnet20(n_classes, error_model, inject_p, inject_epoch, order, activation, affine)

    print(f'\n==> {model} built.')
    return ModelWrapper(net, n_classes, optim_params, loss)


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
