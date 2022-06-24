'''
Namely, each convolution is wrapped with ConvInjector which is convolution + injector.
Once the network is built, we count the number of convolutions ==> self.n_convs.
At each forward, inject_index = randint(0, self.n_convs) to select which convolution will be affected.
Whenever a convolution is performed, a counter is updated:
   ==> if counter == inject_index:
       ==> injection
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .hg_noise_injector.hans_gruber import HansGruberNI

__all__ = ['HardResNet', 'hard_resnet20', 'hard_resnet32', 'hard_resnet44']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def soft(x):
    return torch.log(1+torch.exp(x))


def relu_s(x):
    #x = torch.clip(x, 0, None)
    #s = soft(x)

    return 2 * torch.sigmoid(.5 * x) * x


class MyActivation(nn.Module):
    def __init__(self):
        super().__init__()

        self.gelu = nn.GELU()

    def forward(self, x):
        return torch.nan_to_num(torch.clip(self.gelu(x), None, 6), 0.0)


class ConvInjector(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=0, error_model='random', inject_p=0.01, inject_epoch=0):
        super(ConvInjector, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding, bias=False)
        self.injector = HansGruberNI(error_model, p=inject_p, inject_epoch=inject_epoch)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        x = self.conv(x)
        if counter == inject_index:
            x = self.injector(x, inject, current_epoch)
        counter += 1
        return x, counter, inject_index


class LinearInjector(nn.Module):
    def __init__(self, inplanes, n_classes, error_model, inject_p=0.01, inject_epoch=0):
        super(LinearInjector, self).__init__()
        self.linear = nn.Linear(inplanes, n_classes)
        self.injector = HansGruberNI(error_model, p=inject_p, inject_epoch=inject_epoch)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        x = self.linear(x)
        if counter == inject_index:
            x = self.injector(x, inject, current_epoch)
        return x


class Shortcut(nn.Module):
    def __init__(self, inplanes, outplanes, stride, affine, error_model,  inject_p, inject_epoch):
        super(Shortcut, self).__init__()
        self.conv = ConvInjector(inplanes, outplanes, kernel_size=1, stride=stride, error_model=error_model,
                                 inject_p=inject_p, inject_epoch=inject_epoch)
        self.bn = nn.BatchNorm2d(outplanes, affine=affine)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        out, counter, inject_index = self.conv(x, inject, current_epoch, counter, inject_index)
        out = self.bn(out)
        return out, counter


# This class is just a fake nn.Sequential but allows us to pass
# more arguments when we perform a forward pass.
class BlockGroup(nn.Module):
    def __init__(self, layers):
        super(BlockGroup, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        for layer in self.layers:
            x, counter = layer(x, inject, current_epoch, counter, inject_index)
        return x, counter


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, order='relu-bn', activation='relu', affine=True,
                 error_model='random', inject_p=0.01, inject_epoch=0):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvInjector(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                  error_model=error_model, inject_p=inject_p, inject_epoch=inject_epoch)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.conv2 = ConvInjector(planes, planes, kernel_size=3, stride=1, padding=1,
                                  error_model=error_model, inject_p=inject_p, inject_epoch=inject_epoch)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        '''if activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'relu6':
            self.relu = nn.ReLU6()'''
        self.relu = MyActivation()
        self.order = order

        self.shortcut = False
        if stride != 1 or in_planes != planes:
            self.shortcut = Shortcut(in_planes, self.expansion * planes, stride, affine, error_model, inject_p, inject_epoch)

    def forward(self, x, inject=True, current_epoch=0, counter=0, inject_index=0):
        shortcut = x
        out, counter, inject_index = self.conv1(x, inject, current_epoch, counter, inject_index)
        if self.order == 'relu-bn':
            out = self.bn1(self.relu(out))
        elif self.order == 'bn-relu':
            out = self.relu(self.bn1(out))
        out, counter, inject_index = self.conv2(out, inject, current_epoch, counter, inject_index)
        out = self.bn2(out)
        if self.shortcut:
            shortcut, counter = self.shortcut(x, inject, current_epoch, counter, inject_index)
        out += shortcut
        out = self.relu(out)
        return out, counter


class HardResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, error_model='random', inject_p=0.1, inject_epoch=0,
                 order='relu-bn', activation='relu', affine=True):
        """ Class that represents the ResNet order """
        super(HardResNet, self).__init__()
        self.order = order
        self.affine = affine
        self.error_model = error_model
        self.inject_p = inject_p
        self.inject_epoch = inject_epoch

        self.in_planes = 16
        self.conv1 = ConvInjector(3, 16, kernel_size=3, stride=1, padding=1,
                                  error_model=error_model, inject_p=inject_p, inject_epoch=inject_epoch)
        self.bn1 = nn.BatchNorm2d(16, affine=affine)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, order=order,
                                       activation=activation, affine=affine)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, order=order,
                                       activation=activation, affine=affine)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, order=order,
                                       activation=activation, affine=affine)
        self.linear = LinearInjector(64, n_classes=num_classes, error_model=error_model, inject_p=inject_p,
                                     inject_epoch=inject_epoch)
        '''if activation == 'relu':
                    self.relu = nn.ReLU()
                elif activation == 'relu6':
                    self.relu = nn.ReLU6()'''
        self.relu = MyActivation()

        self.apply(_weights_init)
        self.n_convs = 0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.n_convs += 1

    def _make_layer(self, block, planes, num_blocks, stride, order, activation, affine):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, order=order, activation=activation,
                                affine=affine, error_model=self.error_model, inject_p=self.inject_p,
                                inject_epoch=self.inject_epoch))
            self.in_planes = planes * block.expansion

        return BlockGroup(layers)

    def forward(self, x, inject=True, current_epoch=0):
        counter, inject_index = 0, torch.randint(0, self.n_convs, size=(1,))
        out, counter, inject_index = self.conv1(x, inject, current_epoch, counter, inject_index)
        if self.order == 'relu-bn':
            out = self.bn1(self.relu(out))
        elif self.order == 'bn-relu':
            out = self.relu(self.bn1(out))
        out, counter = self.layer1(out, inject, current_epoch, counter, inject_index)
        out, counter = self.layer2(out, inject, current_epoch, counter, inject_index)
        out, counter = self.layer3(out, inject, current_epoch, counter, inject_index)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out, inject, current_epoch, counter, inject_index)
        return out


def hard_resnet20(n_classes=10, error_model='random', inject_p=0.1, inject_epoch=0, order='relu-bn', activation='relu', affine=True):
    return HardResNet(BasicBlock, [3, 3, 3], n_classes, error_model, inject_p, inject_epoch, order, activation, affine)


def hard_resnet32(n_classes=10, error_model='random', inject_p=0.1, inject_epoch=0, order='relu-bn', activation='relu', affine=True):
    return HardResNet(BasicBlock, [5, 5, 5], n_classes, error_model, inject_p, inject_epoch, order, activation, affine)


def hard_resnet44(n_classes=10, error_model='random', inject_p=0.1, inject_epoch=0, order='relu-bn', activation='relu', affine=True):
    return HardResNet(BasicBlock, [7, 7, 7], n_classes, error_model, inject_p, inject_epoch, order, activation, affine)