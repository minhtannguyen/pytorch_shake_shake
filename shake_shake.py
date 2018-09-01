# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.shake_shake_function import get_alpha_beta, shake_function


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

class NRelu(nn.Module):
    """
    -max(-x,0)
    Parameters
    ----------
    Input shape: (N, C, W, H)
    Output shape: (N, C * W * H)
    """
    def __init__(self, inplace):
        super(NRelu, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return -F.relu(-x, inplace=self.inplace)

class ResidualPath(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualPath, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1min = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2min = nn.BatchNorm2d(out_channels)
        self.nrelu = NRelu(inplace=False)
        self.is_min = False

    def forward(self, x):
        x = self.nrelu(x) if self.is_min else F.relu(x, inplace=False) 
        x = self.nrelu(self.bn1min(self.conv1(x))) if self.is_min else F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.bn2min(self.conv2(x)) if self.is_min else self.bn2(self.conv2(x))
        return x
    
    def change_state(self, is_min):
        self.is_min = is_min
    
class DownsamplingShortcut(nn.Module):
    def __init__(self, in_channels):
        super(DownsamplingShortcut, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.bnmin = nn.BatchNorm2d(in_channels * 2)
        self.nrelu = NRelu(inplace=False)
        self.is_min = False

    def forward(self, x):
        x = self.nrelu(x) if self.is_min else F.relu(x, inplace=False)
        y1 = F.avg_pool2d(x, kernel_size=1, stride=2, padding=0)
        y1 = self.conv1(y1)

        y2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        y2 = F.avg_pool2d(y2, kernel_size=1, stride=2, padding=0)
        y2 = self.conv2(y2)

        z = torch.cat([y1, y2], dim=1)
        z = self.bnmin(z) if self.is_min else self.bn(z)

        return z
    
    def change_state(self, is_min):
        self.is_min = is_min
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shake_config):
        super(BasicBlock, self).__init__()

        self.shake_config = shake_config

        self.residual_path1 = ResidualPath(in_channels, out_channels, stride)
        self.residual_path2 = ResidualPath(in_channels, out_channels, stride)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('downsample',
                                     DownsamplingShortcut(in_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        self.residual_path1.change_state(False)
        x1 = self.residual_path1(x[0])
        self.residual_path2.change_state(False)
        x2 = self.residual_path2(x[0])
        
        self.residual_path1.change_state(True)
        x1min = self.residual_path1(x[1])
        self.residual_path2.change_state(True)
        x2min = self.residual_path2(x[1])

        if self.training:
            shake_config = self.shake_config
        else:
            shake_config = (False, False, False)

        alpha, beta = get_alpha_beta(x[0].size(0), shake_config, x[0].is_cuda)
        alpha_min, beta_min = get_alpha_beta(x[1].size(0), shake_config, x[1].is_cuda)
        y = shake_function(x1, x2, alpha, beta)
        ymin = shake_function(x1min, x2min, alpha_min, beta_min)
        if self.in_channels != self.out_channels:
            self.shortcut[0].change_state(False)
        xsc = self.shortcut(x[0])
        if self.in_channels != self.out_channels:
            self.shortcut[0].change_state(True)
        xscmin = self.shortcut(x[1])

        return xsc + y, xscmin + ymin

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels']
        depth = config['depth']
        self.shake_config = (config['shake_forward'], config['shake_backward'],
                             config['shake_image'])

        block = BasicBlock
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]
        
        self.nrelu = NRelu(inplace=False)

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(base_channels)

        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, block, stride=2)

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape))[0].view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(
                        in_channels,
                        out_channels,
                        stride=stride,
                        shake_config=self.shake_config))
            else:
                stage.add_module(
                    block_name,
                    block(
                        out_channels,
                        out_channels,
                        stride=1,
                        shake_config=self.shake_config))
        return stage

    def _forward_conv(self, x):
        x = self.bn(self.conv(x))
        xmax = F.relu(x, inplace=False)
        xmin = self.nrelu(x)
        x = [xmax, xmin] 
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        xmax = F.adaptive_avg_pool2d(x[0], output_size=1)
        xmin = F.adaptive_avg_pool2d(x[1], output_size=1)
        return xmax, xmin

    def forward(self, x):
        x, xmin = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        xmin = xmin.view(xmin.size(0), -1)
        x = self.fc(x)
        xmin = self.fc(xmin)
        return x, xmin
