import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.hg_noise_injector.hans_gruber import HansGruberNI

###############
# GhostNetV2  #
###############


class NaNReLU(nn.Module):
    def __init__(self, nan=True, act="relu6", inplace=False):
        super().__init__()

        if act == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act == "relu6":
            self.act = F.relu6
        self.nan = nan

    def forward(self, x):
        if self.nan:
            return torch.nan_to_num(self.act(x), 0.0)
        else:
            return self.act(x)


class ConvInjector(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
        bias=False,
        **kwargs,
    ):
        super(ConvInjector, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs
        )
        self.injector = HansGruberNI(error_model, p=inject_p, inject_epoch=inject_epoch)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        x = self.conv(x)
        if inject and counter == inject_index:
            x = self.injector(x, inject, current_epoch)
        counter += 1
        return x, counter, inject_index


class BNInjector(nn.Module):
    def __init__(
        self,
        out_channels,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
        **kwargs,
    ):
        super(BNInjector, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.injector = HansGruberNI(error_model, p=inject_p, inject_epoch=inject_epoch)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        x = self.bn(x)
        if inject and counter == inject_index:
            x = self.injector(x, inject, current_epoch)
        counter += 1
        return x, counter, inject_index


class LinearInjector(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        error_model,
        inject_p=0.01,
        inject_epoch=0,
        **kwargs,
    ):
        super(LinearInjector, self).__init__()
        self.linear = (
            nn.Linear(in_channels, out_channels, **kwargs)
            if out_channels > 0
            else nn.Identity()
        )
        self.injector = HansGruberNI(error_model, p=inject_p, inject_epoch=inject_epoch)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        x = self.linear(x)
        if counter == inject_index and isinstance(self.linear, nn.Linear):
            x = self.injector(x, inject, current_epoch)
        return x


class SequentialInjector(nn.Module):
    def __init__(self, *args):
        super(SequentialInjector, self).__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        for layer in self.layers:
            if any(isinstance(layer, t) for t in INJECTION_LAYERS):
                x, counter, inject_index = layer(
                    x, inject, current_epoch, counter, inject_index
                )
            elif any(isinstance(layer, t) for t in [LinearInjector]):
                x = layer(x, inject, current_epoch, counter, inject_index)
            else:
                x = layer(x)
        return x, counter, inject_index


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False, activation="relu"):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return NaNReLU(act=activation)(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=NaNReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = ConvInjector(
            in_chs,
            reduced_chs,
            1,
            bias=True,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )
        self.act1 = act_layer(inplace=True)
        self.conv_expand = ConvInjector(
            reduced_chs,
            in_chs,
            1,
            bias=True,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        x_se = self.avg_pool(x)
        x_se, counter, inject_index = self.conv_reduce(
            x_se, inject, current_epoch, counter, inject_index
        )
        x_se = self.act1(x_se)
        x_se, counter, inject_index = self.conv_expand(
            x_se, inject, current_epoch, counter, inject_index
        )
        x = x * self.gate_fn(x_se)
        return x, counter, inject_index


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        act_layer=NaNReLU,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
    ):
        super(ConvBnAct, self).__init__()
        self.conv = ConvInjector(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            kernel_size // 2,
            bias=False,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )
        self.bn1 = BNInjector(
            out_chs, error_model="random", inject_p=0.01, inject_epoch=0
        )
        self.act1 = act_layer(inplace=True)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):

        x, counter, inject_index = self.conv(
            x, inject, current_epoch, counter, inject_index
        )
        x, counter, inject_index = self.bn1(
            x, inject, current_epoch, counter, inject_index
        )
        x = self.act1(x)

        return x, counter, inject_index


class GhostModuleV2(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        relu=True,
        activation="relu",
        mode=None,
        args=None,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
    ):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ["original"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = SequentialInjector(
                ConvInjector(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(
                    init_channels, error_model="random", inject_p=0.01, inject_epoch=0
                ),
                NaNReLU(act=activation, inplace=True) if relu else SequentialInjector(),
            )
            self.cheap_operation = SequentialInjector(
                ConvInjector(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(
                    new_channels, error_model="random", inject_p=0.01, inject_epoch=0
                ),
                NaNReLU(act=activation, inplace=True) if relu else SequentialInjector(),
            )
        elif self.mode in ["attn"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = SequentialInjector(
                ConvInjector(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(
                    init_channels, error_model="random", inject_p=0.01, inject_epoch=0
                ),
                NaNReLU(act=activation, inplace=True) if relu else SequentialInjector(),
            )
            self.cheap_operation = SequentialInjector(
                ConvInjector(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(
                    new_channels, error_model="random", inject_p=0.01, inject_epoch=0
                ),
                NaNReLU(act=activation, inplace=True) if relu else SequentialInjector(),
            )
            self.short_conv = SequentialInjector(
                ConvInjector(
                    inp,
                    oup,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(oup, error_model="random", inject_p=0.01, inject_epoch=0),
                ConvInjector(
                    oup,
                    oup,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=oup,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(oup, error_model="random", inject_p=0.01, inject_epoch=0),
                ConvInjector(
                    oup,
                    oup,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=oup,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(oup, error_model="random", inject_p=0.01, inject_epoch=0),
            )

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        if self.mode in ["original"]:
            x1, counter, inject_index = self.primary_conv(
                x, inject, current_epoch, counter, inject_index
            )
            x2, counter, inject_index = self.cheap_operation(
                x1, inject, current_epoch, counter, inject_index
            )
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :], counter, inject_index
        elif self.mode in ["attn"]:
            res, counter, inject_index = self.short_conv(
                F.avg_pool2d(x, kernel_size=2, stride=2),
                inject,
                current_epoch,
                counter,
                inject_index,
            )
            x1, counter, inject_index = self.primary_conv(
                x, inject, current_epoch, counter, inject_index
            )
            x2, counter, inject_index = self.cheap_operation(
                x1, inject, current_epoch, counter, inject_index
            )
            out = torch.cat([x1, x2], dim=1)
            return (
                out[:, : self.oup, :, :]
                * F.interpolate(
                    self.gate_fn(res),
                    size=(out.shape[-2], out.shape[-1]),
                    mode="nearest",
                ),
                counter,
                inject_index,
            )


class GhostBottleneckV2(nn.Module):

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        act_layer=NaNReLU,
        se_ratio=0.0,
        layer_id=None,
        args=None,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
    ):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(
                in_chs,
                mid_chs,
                relu=True,
                mode="original",
                args=args,
                error_model=error_model,
                inject_p=inject_p,
                inject_epoch=inject_epoch,
            )
        else:
            self.ghost1 = GhostModuleV2(
                in_chs,
                mid_chs,
                relu=True,
                mode="attn",
                args=args,
                error_model=error_model,
                inject_p=inject_p,
                inject_epoch=inject_epoch,
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = ConvInjector(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
                error_model=error_model,
                inject_p=inject_p,
                inject_epoch=inject_epoch,
            )
            self.bn_dw = BNInjector(
                mid_chs, error_model="random", inject_p=0.01, inject_epoch=0
            )

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(
                mid_chs,
                se_ratio=se_ratio,
                error_model=error_model,
                inject_p=inject_p,
                inject_epoch=inject_epoch,
            )
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            mid_chs,
            out_chs,
            relu=False,
            mode="original",
            args=args,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = SequentialInjector()
        else:
            self.shortcut = SequentialInjector(
                ConvInjector(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(in_chs, error_model="random", inject_p=0.01, inject_epoch=0),
                ConvInjector(
                    in_chs,
                    out_chs,
                    1,
                    stride=1,
                    padding=0,
                    bias=False,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                ),
                BNInjector(
                    out_chs, error_model="random", inject_p=0.01, inject_epoch=0
                ),
            )

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        residual = x
        x, counter, inject_index = self.ghost1(
            x, inject, current_epoch, counter, inject_index
        )
        if self.stride > 1:
            x, counter, inject_index = self.conv_dw(
                x, inject, current_epoch, counter, inject_index
            )
            x, counter, inject_index = self.bn_dw(
                x, inject, current_epoch, counter, inject_index
            )
        if self.se is not None:
            x, counter, inject_index = self.se(
                x, inject, current_epoch, counter, inject_index
            )
        x, counter, inject_index = self.ghost2(
            x, inject, current_epoch, counter, inject_index
        )
        x_short, counter, inject_index = self.shortcut(
            residual, inject, current_epoch, counter, inject_index
        )
        x = x + x_short

        return x, counter, inject_index


INJECTION_LAYERS = [
    ConvInjector,
    SqueezeExcite,
    GhostModuleV2,
    GhostBottleneckV2,
    SequentialInjector,
    ConvBnAct,
    BNInjector,
]


class GhostNetV2(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        block=GhostBottleneckV2,
        activation="relu",
        args=None,
        error_model="random",
        inject_p=0.01,
        inject_epoch=0,
    ):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)

        # 5 channels input
        # 5 spectral bands: B02, B03, B04, B08, B11
        self.conv_stem = ConvInjector(
            5,
            output_channel,
            3,
            2,
            1,
            bias=False,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )
        self.bn1 = BNInjector(
            output_channel, error_model="random", inject_p=0.01, inject_epoch=0
        )
        self.act1 = NaNReLU(act=activation, inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        self.stages = []
        # block = block
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(
                        block(
                            input_channel,
                            hidden_channel,
                            output_channel,
                            k,
                            s,
                            se_ratio=se_ratio,
                            layer_id=layer_id,
                            args=args,
                            error_model=error_model,
                            inject_p=inject_p,
                            inject_epoch=inject_epoch,
                        )
                    )
                input_channel = output_channel
                layer_id += 1
            self.stages.append(SequentialInjector(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        self.stages.append(
            SequentialInjector(
                ConvBnAct(
                    input_channel,
                    output_channel,
                    1,
                    error_model=error_model,
                    inject_p=inject_p,
                    inject_epoch=inject_epoch,
                )
            )
        )
        input_channel = output_channel

        self.blocks = SequentialInjector(*self.stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = ConvInjector(
            input_channel,
            output_channel,
            1,
            1,
            0,
            bias=True,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )
        self.act2 = NaNReLU(act=activation, inplace=True)
        self.classifier = LinearInjector(
            output_channel,
            num_classes,
            error_model=error_model,
            inject_p=inject_p,
            inject_epoch=inject_epoch,
        )

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        intermediates = []  # List to store intermediate features

        x, counter, inject_index = self.conv_stem(
            x, inject, current_epoch, counter, inject_index
        )
        x, counter, inject_index = self.bn1(
            x, inject, current_epoch, counter, inject_index
        )
        x = self.act1(x)

        for block in self.stages:
            x, counter, inject_index = block(
                x, inject, current_epoch, counter, inject_index
            )
            intermediates.append(x)  # Store the intermediate features

        x = self.global_pool(x)
        x, counter, inject_index = self.conv_head(
            x, inject, current_epoch, counter, inject_index
        )
        x = self.act2(x)

        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x, inject, current_epoch, counter, inject_index)

        return x, intermediates


def cfgs_standard():

    cfgs = [  # Each line is a GhostBottleneckV2 block (16 blocks in total)
        # k, t, c, SE, s
        [[3, 16, 16, 0, 1]],
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        [[3, 240, 80, 0, 2]],
        [
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
        ],
        [[5, 672, 160, 0.25, 2]],
        [
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
        ],
    ]

    return cfgs


def ghostnetv2(
    width=1.6,
    num_classes=1000,
    error_model="random",
    inject_p=0.01,
    inject_epoch=0,
    activation="relu",
    ckpt=None,
):
    cfgs = cfgs_standard()
    model = GhostNetV2(
        cfgs,
        num_classes=num_classes,
        width=width,
        error_model=error_model,
        inject_p=inject_p,
        inject_epoch=inject_epoch,
        activation=activation,
    )
    if ckpt is not None:
        model = load_fi_weights(model, ckpt)
    return model


def load_fi_weights(model, filename, verbose=False):
    count = 0
    new_dict = {}
    weights = torch.load(filename, weights_only=True)
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if verbose:
            print(name, param.data.shape)
        if "dummy" in name:
            if verbose:
                print("-\n")
            continue
        new_name = name.replace("conv.weight", "weight").replace("conv.bias", "bias")
        new_name = new_name.replace("bn.weight", "weight").replace("bn.bias", "bias")
        new_name = (
            new_name.replace("bn.running_mean", "running_mean")
            .replace("bn.running_var", "running_var")
            .replace("bn.num_batches_tracked", "num_batches_tracked")
        )
        new_name = new_name.replace("linear.weight", "weight").replace(
            "linear.bias", "bias"
        )
        new_name = new_name.replace(".layers", "")
        new_weights = weights[new_name]
        if verbose:
            print(new_name, new_weights.shape, "\n")
        count += 1
        if param.data.shape != new_weights.shape:
            raise ValueError(
                f"Shape mismatch: {param.data.shape} != {new_weights.shape}"
            )
        new_dict[name] = new_weights

    print(f"Loaded {count} weights")
    model.load_state_dict(new_dict, strict=False)
    return model


###############
#  Seg Head   #
###############


class SegmentationHeadGhostBN(nn.Module):

    def __init__(self, num_classes, width, se_ratio=0.0):

        super().__init__()

        self.num_classes = num_classes

        # Number of input channels for 3 feature maps used from the backbone
        in_channels = [
            960,
            112,
            40,
        ]  # From standard GhostNetV2 configuration (layers: 9, 6, 4)
        in_channels = [
            math.ceil(c * width) for c in in_channels
        ]  # Apply width multiplier (round up)

        # Upsampling layers
        self.up2 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.up8 = nn.Upsample(mode="bilinear", scale_factor=8)

        self.conv_block_960 = GhostBottleneckV2(
            in_channels[0],
            in_channels[0] // 4,
            num_classes,
            3,
            layer_id=0,
            se_ratio=se_ratio,
        )
        self.conv_block_112 = GhostBottleneckV2(
            in_channels[1],
            in_channels[1] // 4,
            num_classes,
            3,
            layer_id=0,
            se_ratio=se_ratio,
        )
        self.conv_block_40 = GhostBottleneckV2(
            in_channels[2],
            in_channels[2] // 4,
            num_classes,
            3,
            layer_id=0,
            se_ratio=se_ratio,
        )

    def forward(self, tensors):
        """
        Forward pass through the module.

        This method takes a list of tensors, each of which is the output of another neural network.

        Arguments:
            tensors (list of torch.Tensor): A list containing tensors which are the features of the backbone.
                Each tensor in the list should have the following dimensions:
                - Output n.0 shape: torch.Size([batch_size, 16 * width, H / 2, W / 2])
                - Output n.1 shape: torch.Size([batch_size, 24 * width, H / 4, W / 4])
                - Output n.2 shape: torch.Size([batch_size, 24 * width, H / 4, W / 4])
                - Output n.3 shape: torch.Size([batch_size, 40 * width, H / 8, W / 8])
                - Output n.4 shape: torch.Size([batch_size, 40 * width, H / 8, W / 8])
                - Output n.5 shape: torch.Size([batch_size, 80 * width, H / 16, W / 16])
                - Output n.6 shape: torch.Size([batch_size, 112 * width, H / 16, W / 16])
                - Output n.7 shape: torch.Size([batch_size, 160 * width, H / 32, W / 32])
                - Output n.8 shape: torch.Size([batch_size, 160 * width, H / 32, W / 32])
                - Output n.9 shape: torch.Size([batch_size, 960 * width, H / 32, W / 32])

        Returns:
            torch.Tensor: The final tensor obtained after applying the operations on the input tensors.
        """

        # Extract the feature maps from the backbone
        out_9 = tensors[9]  # Sequential 2-10: [B, C * width, H/32, W/32]
        out_6 = tensors[6]  # Sequential 2-7: [B, C * width, H/16, W/16]
        out_4 = tensors[4]  # Sequential 2-5: [B, C * width, H/8, H/8]

        out_9 = self.conv_block_960(out_9)  # [..., H/32, W/32]
        x_2upsampled_pred = self.up2(out_9)  # [..., H/16, W/16]

        out_6 = self.conv_block_112(out_6)  # [..., H/16, W/16]

        x = x_2upsampled_pred + out_6

        x_2upsampled_pred = self.up2(x)  # [..., H/8, W/8]

        out_4 = self.conv_block_40(out_4)  # [..., H/8, W/8]

        x = x_2upsampled_pred + out_4

        x = self.up8(x)  # [..., H, W]

        return x


def __main__():

    model = ghostnetv2()
    x = torch.randn(1, 5, 224, 224)
    y, intermediates = model(x)
    print(y.shape)
    print(len(intermediates))


if __name__ == "__main__":
    __main__()
