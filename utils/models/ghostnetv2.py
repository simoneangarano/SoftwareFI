import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hg_noise_injector.hans_gruber import HansGruberNI


###############
# GhostNetV2  #
###############


class NaNAct(nn.Module):
    def __init__(
        self,
        act=None,
        inplace=False,
        args=None,
    ):
        super().__init__()
        if act is None:
            act = args.activation
        if act == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act == "relu6":
            self.act = nn.ReLU6(inplace=inplace)
        elif act == "relumax":
            self.act = ReLUMax(inplace=inplace)
        elif act == "sigmoid":
            self.act = torch.sigmoid
        elif act == "hard_sigmoid":
            self.act = HardSigmoid(inplace=inplace)

        self.nan = args.nan

    def forward(self, x):
        x = self.act(x)
        if self.nan:
            return torch.nan_to_num(x, 0.0)
        # if hasattr(self, "stats"):
        #     x[x > self.stats[3] * 10] = self.stats[3]  # or 0?
        #     x[x < self.stats[2] * 10] = self.stats[3]  # or 0?
        return x


class ConvInjector(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        args=None,
        **kwargs,
    ):
        super(ConvInjector, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        if args.inject:
            self.injector = HansGruberNI(args)
        else:
            self.injector = SequentialInjector()

        self.inject_first = args.inject_first

    def forward(self, x, fwargs):
        if not self.inject_first:
            x = self.conv(x)

        if fwargs["inj"] and (
            (fwargs["cnt"] == fwargs["idx"]) or (fwargs["idx"] == -1)
        ):
            x, fwargs = self.injector(x, fwargs)
        fwargs["cnt"] += 1

        if self.inject_first:
            x = self.conv(x)

        # if hasattr(self, "stats"):
        #     x[x > self.stats[3]*10] = self.stats[3] # or 0?
        #     x[x < self.stats[2]*10] = self.stats[3] # or 0?

        return x, fwargs


class BNInjector(nn.Module):
    def __init__(
        self,
        out_channels,
        args=None,
    ):
        super(BNInjector, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        if args.inject:
            self.injector = HansGruberNI(args)
        else:
            self.injector = SequentialInjector()

        self.inject_first = args.inject_first

    def forward(self, x, fwargs):
        if not self.inject_first:
            x = self.bn(x)

        if fwargs["inj"] and (
            (fwargs["cnt"] == fwargs["idx"]) or (fwargs["idx"] == -1)
        ):
            x, fwargs = self.injector(x, fwargs)
        fwargs["cnt"] += 1

        if self.inject_first:
            x = self.bn(x)

        # if hasattr(self, "stats"):
        #     x[x > self.stats[3] * 10] = self.stats[3]  # or 0?
        #     x[x < self.stats[2] * 10] = self.stats[3]  # or 0?

        return x, fwargs


class LinearInjector(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        args=None,
        **kwargs,
    ):
        super(LinearInjector, self).__init__()
        self.linear = (
            nn.Linear(in_channels, out_channels, **kwargs)
            if out_channels > 0
            else nn.Identity()
        )
        if args.inject:
            self.injector = HansGruberNI(args)
        else:
            self.injector = SequentialInjector()

        self.inject_first = args.inject_first

    def forward(self, x, fwargs):
        if not self.inject_first:
            x = self.linear(x)

        if (
            fwargs["inj"]
            and ((fwargs["cnt"] == fwargs["idx"]) or (fwargs["idx"] == -1))
            and isinstance(self.linear, nn.Linear)
        ):
            x, fwargs = self.injector(x, fwargs)
        fwargs["cnt"] += 1

        if self.inject_first:
            x = self.linear(x)

        # if hasattr(self, "stats"):
        #     x[x > self.stats[3] * 10] = self.stats[3]  # or 0?
        #     x[x < self.stats[2] * 10] = self.stats[3]  # or 0?

        return x


class SequentialInjector(nn.Module):
    def __init__(self, *args):
        super(SequentialInjector, self).__init__()
        self.layers = nn.ModuleList(args)
        self.injection_layers = [
            ConvInjector,
            SqueezeExcite,
            GhostModuleV2,
            GhostBottleneckV2,
            SequentialInjector,
            ConvBnAct,
            BNInjector,
        ]

    def forward(self, x, fwargs):
        for layer in self.layers:
            if any(isinstance(layer, t) for t in self.injection_layers):
                x, fwargs = layer(x, fwargs)
            elif isinstance(layer, LinearInjector):
                x = layer(x, fwargs)
            else:
                x = layer(x)
        return x, fwargs


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False, args=None):
        super().__init__()
        self.inplace = inplace
        self.activation = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        if self.inplace:
            return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
        else:
            return self.activation(x + 3.0) / 6.0


class ReLUMax(nn.Module):
    def __init__(self, inplace: bool = False, args=None):
        super().__init__()
        self.activation = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.activation(x)
        # x[x > self.max] = 0.0
        return x


class ClampAvgPool2d(nn.Module):
    def __init__(
        self,
        output_size=None,
        kernel_size=None,
        stride=None,
        padding=0,
        args=None,
    ):
        super(ClampAvgPool2d, self).__init__()
        if output_size is not None:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        elif kernel_size is not None:
            self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise ValueError("output_size or kernel_size must be defined")

    def forward(self, x):
        # if self.max is not None:
        #     x[x > self.max] = self.max  # or 0?
        # if hasattr(self, "stats"):
        #     x[x > self.stats[3] * 10] = self.stats[3]  # or 0?
        #     x[x < self.stats[2] * 10] = self.stats[3]  # or 0?
        return self.avg_pool(x)


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


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        divisor=4,
        args=None,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = NaNAct(act="hard_sigmoid", args=args)
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = ClampAvgPool2d(output_size=1, args=args)
        self.conv_reduce = ConvInjector(
            in_channels=in_chs,
            out_channels=reduced_chs,
            kernel_size=1,
            bias=True,
            args=args,
        )
        self.act1 = NaNAct(inplace=True, args=args)
        self.conv_expand = ConvInjector(
            in_channels=reduced_chs,
            out_channels=in_chs,
            kernel_size=1,
            bias=True,
            args=args,
        )

    def forward(self, x, fwargs):
        x_se = self.avg_pool(x)
        x_se, fwargs = self.conv_reduce(x_se, fwargs)
        x_se = self.act1(x_se)
        x_se, fwargs = self.conv_expand(x_se, fwargs)
        x = x * self.gate_fn(x_se)
        return x, fwargs


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        args=None,
    ):
        super(ConvBnAct, self).__init__()
        self.conv = ConvInjector(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
            args=args,
        )
        self.bn1 = BNInjector(out_channels=out_chs, args=args)
        self.act1 = NaNAct(inplace=True, args=args)

    def forward(self, x, fwargs):
        x, fwargs = self.conv(x, fwargs)
        x, fwargs = self.bn1(x, fwargs)
        x = self.act1(x)  # CHECK (act after bn)

        return x, fwargs


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
        mode=None,
        args=None,
    ):
        super(GhostModuleV2, self).__init__()
        self.mode = mode

        if self.mode in ["original"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = SequentialInjector(
                ConvInjector(
                    in_channels=inp,
                    out_channels=init_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=init_channels, args=args),
                (
                    NaNAct(inplace=True, args=args)  # CHECK (act after bn)
                    if relu
                    else SequentialInjector()
                ),
            )
            self.cheap_operation = SequentialInjector(
                ConvInjector(
                    in_channels=init_channels,
                    out_channels=new_channels,
                    kernel_size=dw_size,
                    stride=1,
                    padding=dw_size // 2,
                    groups=init_channels,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=new_channels, args=args),
                (
                    NaNAct(inplace=True, args=args)  # CHECK (act after bn)
                    if relu
                    else SequentialInjector()
                ),
            )
        elif self.mode in ["attn"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.avg_pool = ClampAvgPool2d(kernel_size=2, stride=2, args=args)
            self.gate_fn = NaNAct(act="sigmoid", args=args)
            self.primary_conv = SequentialInjector(
                ConvInjector(
                    in_channels=inp,
                    out_channels=init_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=init_channels, args=args),
                (
                    NaNAct(inplace=True, args=args)  # CHECK (act after bn)
                    if relu
                    else SequentialInjector()
                ),
            )
            self.cheap_operation = SequentialInjector(
                ConvInjector(
                    in_channels=init_channels,
                    out_channels=new_channels,
                    kernel_size=dw_size,
                    stride=1,
                    padding=dw_size // 2,
                    groups=init_channels,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=new_channels, args=args),
                (
                    NaNAct(inplace=True, args=args)  # CHECK (act after bn)
                    if relu
                    else SequentialInjector()
                ),
            )
            self.short_conv = SequentialInjector(  # CHECK (bn without act)
                ConvInjector(
                    in_channels=inp,
                    out_channels=oup,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=oup, args=args),
                ConvInjector(
                    in_channels=oup,
                    out_channels=oup,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=oup,
                    bias=False,
                    args=args,
                ),
                BNInjector(oup, args=args),
                ConvInjector(
                    in_channels=oup,
                    out_channels=oup,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=oup,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=oup, args=args),
            )

    def forward(self, x, fwargs):
        if self.mode in ["original"]:
            x1, fwargs = self.primary_conv(x, fwargs)
            x2, fwargs = self.cheap_operation(x1, fwargs)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :], fwargs
        elif self.mode in ["attn"]:
            res, fwargs = self.short_conv(self.avg_pool(x), fwargs)
            x1, fwargs = self.primary_conv(x, fwargs)
            x2, fwargs = self.cheap_operation(x1, fwargs)
            out = torch.cat([x1, x2], dim=1)
            return (
                out[:, : self.oup, :, :]
                * F.interpolate(
                    self.gate_fn(res),
                    size=(out.shape[-2], out.shape[-1]),
                    mode="nearest",
                ),
                fwargs,
            )


class GhostBottleneckV2(nn.Module):

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0.0,
        layer_id=None,
        args=None,
    ):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(
                inp=in_chs,
                oup=mid_chs,
                relu=True,
                mode="original",
                args=args,
            )
        else:
            self.ghost1 = GhostModuleV2(
                inp=in_chs,
                oup=mid_chs,
                relu=True,
                mode="attn",
                args=args,
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = ConvInjector(
                in_channels=mid_chs,
                out_channels=mid_chs,
                kernel_size=dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
                args=args,
            )
            self.bn_dw = BNInjector(out_channels=mid_chs, args=args)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(in_chs=mid_chs, se_ratio=se_ratio, args=args)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            inp=mid_chs,
            oup=out_chs,
            relu=False,
            mode="original",
            args=args,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = SequentialInjector()
        else:
            self.shortcut = SequentialInjector(
                ConvInjector(  # CHECK (bn without act)
                    in_channels=in_chs,
                    out_channels=in_chs,
                    kernel_size=dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=in_chs, args=args),
                ConvInjector(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    args=args,
                ),
                BNInjector(out_channels=out_chs, args=args),
            )

    def forward(self, x, fwargs):
        residual = x
        x, fwargs = self.ghost1(x, fwargs)  # activation at the end of the block
        if self.stride > 1:  # no activation at the end of the block
            x, fwargs = self.conv_dw(x, fwargs)
            x, fwargs = self.bn_dw(x, fwargs)  # CHECK (bn without act)
        if self.se is not None:  # leading avgpool
            x, fwargs = self.se(x, fwargs)
        x, fwargs = self.ghost2(x, fwargs)
        x_short, fwargs = self.shortcut(residual, fwargs)
        x = x + x_short

        return x, fwargs


class GhostNetV2(nn.Module):
    def __init__(
        self,
        args,
        cfgs,
        block=GhostBottleneckV2,
        dropout=0.2,
    ):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * args.width, 4)

        # 5 channels input
        # 5 spectral bands: B02, B03, B04, B08, B11
        self.conv_stem = ConvInjector(
            in_channels=5,
            out_channels=output_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            args=args,
        )
        self.bn1 = BNInjector(out_channels=output_channel, args=args)
        self.act1 = NaNAct(inplace=True, args=args)
        input_channel = output_channel

        # building inverted residual blocks
        self.stages = []
        # block = block
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * args.width, 4)
                hidden_channel = _make_divisible(exp_size * args.width, 4)
                if block == GhostBottleneckV2:
                    layers.append(
                        block(
                            in_chs=input_channel,
                            mid_chs=hidden_channel,
                            out_chs=output_channel,
                            dw_kernel_size=k,
                            stride=s,
                            se_ratio=se_ratio,
                            layer_id=layer_id,
                            args=args,
                        )
                    )
                input_channel = output_channel
                layer_id += 1
            self.stages.append(SequentialInjector(*layers))

        output_channel = _make_divisible(exp_size * args.width, 4)
        self.stages.append(
            SequentialInjector(
                ConvBnAct(
                    in_chs=input_channel,
                    out_chs=output_channel,
                    kernel_size=1,
                    args=args,
                )
            )
        )
        # input_channel = output_channel

        self.blocks = SequentialInjector(*self.stages)

        # building last several layers
        # output_channel = 1280
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv_head = ConvInjector(
        #     input_channel,
        #     output_channel,
        #     1,
        #     1,
        #     0,
        #     bias=True,
        #     args=args,
        # )
        # self.act2 = NaNAct(inplace=True, args=args)
        # self.classifier = LinearInjector(
        #     output_channel,
        #     num_classes,
        #     args=args,
        # )

    def forward(self, x, fwargs):
        intermediates = []  # List to store intermediate features

        x, fwargs = self.conv_stem(x, fwargs)
        x, fwargs = self.bn1(x, fwargs)  # CHECK (act after bn)
        x = self.act1(x)

        for block in self.stages:
            x, fwargs = block(x, fwargs)
            intermediates.append(x)  # Store the intermediate features

        # x = self.global_pool(x)
        # x, fwargs = self.conv_head(x, fwargs)
        # x = self.act2(x)
        # x = x.view(x.size(0), -1)
        # if self.dropout > 0.0:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.classifier(x, fwargs)

        return (x, fwargs), intermediates


def cfgs_standard():

    cfgs = [  # Each line is a GhostBottleneckV2 block (16 blocks in total)
        # k, t, c, SE, s
        [[3, 16, 16, 0, 1]],
        [[3, 48, 24, 0, 2]],  # avgpool critical
        [[3, 72, 24, 0, 1]],
        [[5, 72, 40, 0.25, 2]],  # avgpool critical
        [[5, 120, 40, 0.25, 1]],
        [[3, 240, 80, 0, 2]],  # avgpool critical
        [
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
        ],
        [[5, 672, 160, 0.25, 2]],  # avgpool critical
        [
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
        ],
    ]

    return cfgs


def ghostnetv2(args):
    cfgs = cfgs_standard()
    model = GhostNetV2(args, cfgs)
    return model


###############
#  Seg. Head  #
###############


class SegmentationHeadGhostBN(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.num_classes = args.num_classes

        # Number of input channels for 3 feature maps used from the backbone
        in_channels = [960, 112, 40]  # Standard GhostNetV2 cfg (layers: 9, 6, 4)
        in_channels = [
            math.ceil(c * args.width) for c in in_channels
        ]  # Apply width multiplier (round up)

        # Upsampling layers
        self.up2 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.up8 = nn.Upsample(mode="bilinear", scale_factor=8)

        self.conv_block_960 = GhostBottleneckV2(
            in_chs=in_channels[0],
            mid_chs=in_channels[0] // 4,
            out_chs=args.num_classes,
            dw_kernel_size=3,
            layer_id=0,
            se_ratio=args.se_ratio,
            args=args,
        )
        self.conv_block_112 = GhostBottleneckV2(
            in_chs=in_channels[1],
            mid_chs=in_channels[1] // 4,
            out_chs=args.num_classes,
            dw_kernel_size=3,
            layer_id=0,
            se_ratio=args.se_ratio,
            args=args,
        )
        self.conv_block_40 = GhostBottleneckV2(
            in_chs=in_channels[2],
            mid_chs=in_channels[2] // 4,
            out_chs=args.num_classes,
            dw_kernel_size=3,
            layer_id=0,
            se_ratio=args.se_ratio,
            args=args,
        )
        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, tensors, fwargs):
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

        out_9, fwargs = self.conv_block_960(out_9, fwargs)  # [..., H/32, W/32]
        x_2upsampled_pred = self.up2(out_9)  # [..., H/16, W/16]
        out_6, fwargs = self.conv_block_112(out_6, fwargs)  # [..., H/16, W/16]
        x = x_2upsampled_pred + out_6  # [..., H/16, W/16]
        x_2upsampled_pred = self.up2(x)  # [..., H/8, W/8]
        out_4, _ = self.conv_block_40(out_4, fwargs)  # [..., H/8, W/8]
        x = x_2upsampled_pred + out_4  # [..., H/8, W/8]
        x = self.up8(x)  # [..., H, W]

        return x


###############
# GhostNet+SS #
###############


class GhostNetSS(nn.Module):

    def __init__(self, ghostnet, head, args):

        super().__init__()

        self.ghostnet = ghostnet
        self.head = head
        if args.ckpt is not None:
            self = load_fi_weights(self, args.ckpt)
        if args.stats is not None:
            self.apply_stats(args.stats)

    def forward(
        self, tensors, inject=False, current_epoch=0, counter=0, inject_index=-1
    ):
        fwargs = {
            "inj": inject,
            "idx": inject_index,
            "ep": current_epoch,
            "cnt": counter,
            "faulty_idxs": torch.ones(tensors.shape[0]) * -1,
        }
        (_, fwargs), intermediate_features = self.ghostnet(tensors, fwargs)
        outputs = self.head(intermediate_features, fwargs)

        return outputs

    def apply_stats(self, stats):
        # apply stats to each layer using the stats dictionary
        for name, layer in self.named_modules():
            if name in stats.keys():
                layer.stats = stats[name]


def load_fi_weights(model, filename, verbose=False):
    count = 0
    new_dict = {}
    weights = torch.load(filename, weights_only=True)
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if verbose:
            print(name, param.data.shape)
        if (
            "dummy" in name or "conv_head" in name
        ):  # conv_head and dummy layers are not needed
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
    model.load_state_dict(
        new_dict,
        strict=False,
    )
    return model


def __main__():
    model = ghostnetv2()
    x = torch.randn(1, 5, 224, 224)
    y, intermediates = model(x)
    print(y.shape)
    print(len(intermediates))


if __name__ == "__main__":
    __main__()
