import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# from torchvision.models import resnet101
try:
    from utils.segmentation.deeplabv3_custom.activations import RobustActivation
    from utils.segmentation.deeplabv3_custom.resnet_relumax import resnet101
except ModuleNotFoundError:
    from activations import RobustActivation
    from resnet_relumax import resnet101

from collections import OrderedDict
from typing import Dict, List

from torch import Tensor


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {"return_layers"}

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DeepLabHead(nn.Sequential):
    def __init__(
        self, in_channels: int, num_classes: int, activation: str = "max"
    ) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36], activation=activation),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            RobustActivation(nn.ReLU(), activation),  #!
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        activation: str = "max",
    ) -> None:
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            RobustActivation(nn.ReLU(), activation),  #!
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "max"
    ) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            RobustActivation(nn.ReLU(), activation),  #!
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: List[int],
        out_channels: int = 256,
        activation: str = "max",
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                RobustActivation(nn.ReLU(), activation),
            )  #!
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(
                ASPPConv(in_channels, out_channels, rate, activation=activation)
            )

        modules.append(ASPPPooling(in_channels, out_channels, activation=activation))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            RobustActivation(nn.ReLU(), activation),  #!
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3(nn.Module):

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        # result["out"] = x
        result = x

        return result


def deeplabv3_resnet101(num_classes=19, pretrained=True, activation="max"):
    if pretrained:
        num_classes = 21

    return_layers = {"layer4": "out"}
    backbone = resnet101(
        replace_stride_with_dilation=[False, True, True], activation=activation
    )
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = DeepLabHead(2048, num_classes, activation=activation)
    model = DeepLabV3(backbone, classifier)

    if pretrained:
        model.load_state_dict(
            DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.get_state_dict(
                progress=True
            ),
            strict=False,
        )

    return model
