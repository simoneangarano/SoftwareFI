import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import math


###############
#  GhostNet   #
###############
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


def hard_sigmoid(x, inplace: bool = False):
    quant = torch.ao.quantization.QuantStub()
    dequant = torch.ao.quantization.DeQuantStub()

    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        x = torch.dequantize(x)
        x = F.relu6(x + 3.0) / 6.0
        quant(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

        self.ff = torch.nn.quantized.FloatFunctional()
        self.quant = torch.ao.quantization.QuantStub()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # x = x * self.gate_fn(x_se)
        self.quant(self.gate_fn(x_se))
        x = self.ff.mul(x, self.quant(self.gate_fn(x_se)))
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)

        return x


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
        self.gate_fn = nn.Sigmoid()

        if self.mode in ["original"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
        elif self.mode in ["attn"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(
                    oup,
                    oup,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=oup,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
                nn.Conv2d(
                    oup,
                    oup,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=oup,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
            )

        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.mode in ["original"]:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :]
        elif self.mode in ["attn"]:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            # return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest')
            return self.ff.mul(
                out[:, : self.oup, :, :],
                F.interpolate(
                    self.gate_fn(res),
                    size=(out.shape[-2], out.shape[-1]),
                    mode="nearest",
                ),
            )


class GhostBottleneckV2(nn.Module):

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        act_layer=nn.ReLU,
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
                in_chs, mid_chs, relu=True, mode="original", args=args
            )
        else:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, relu=True, mode="attn", args=args
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            mid_chs, out_chs, relu=False, mode="original", args=args
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)

        # x += self.shortcut(residual)
        x = self.ff.add(x, self.shortcut(residual))
        return x


class GhostNetV2(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        block=GhostBottleneckV2,
        args=None,
    ):
        super(GhostNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout = dropout

        # QuantStub and DeQuantStub layers to quantize weights
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        # 4 channels input
        self.conv_stem = nn.Conv2d(5, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
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
                        )
                    )
                input_channel = output_channel
                layer_id += 1
            self.stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        self.stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*self.stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        intermediates = []  # List to store intermediate feature maps

        # x = self.quant(x)

        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        for block in self.stages:
            x = block(x)
            intermediates.append(x)  # Store the intermediate feature maps

        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)

        x = x.reshape(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)

        # x = self.dequant(x)

        return x, intermediates


def cfgs_standard():

    cfgs = [
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


###############
#  Seg. Head  #
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

        self.ff = torch.nn.quantized.FloatFunctional()

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

        x = self.ff.add(x_2upsampled_pred, out_6)  # [..., H/16, W/16]

        x_2upsampled_pred = self.up2(x)  # [..., H/8, W/8]

        out_4 = self.conv_block_40(out_4)  # [..., H/8, W/8]

        x = self.ff.add(x_2upsampled_pred, out_4)  # [..., H/8, W/8]

        x = self.up8(x)  # [..., H, W]

        return x


###############
# GhostNet+SS #
###############
class GhostNetSS(nn.Module):

    def __init__(self, ghostnet, head):

        super().__init__()

        self.ghostnet = ghostnet
        self.head = head

    def forward(self, tensors):

        features, intermediate_features = self.ghostnet(tensors)

        outputs = self.head(intermediate_features)

        return outputs
