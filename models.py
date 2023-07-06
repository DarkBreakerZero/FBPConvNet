# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=False, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl, eps=1e-4)
        if is_relu is True:
            # self.relu = nn.ReLU(inplace=True)
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(ResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            )

        self.conv_x = None

        if in_chl != out_chl:

            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv_out = self.encode(x)

        if self.conv_x is None:

            res_out = F.relu(conv_out + x)

        else: 

            res_out = F.relu(conv_out + self.conv_x(x))

        return res_out


class EncodeResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(EncodeResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            )

        self.conv_x = None

        if in_chl != out_chl:

            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv_out = self.encode(x)

        if self.conv_x is None:

            res_out = F.relu(conv_out + x)

        else: 

            res_out = F.relu(conv_out + self.conv_x(x))

        down_out = F.max_pool2d(res_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return res_out, down_out

class DecodeResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(DecodeResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            )

        self.conv_x = None

        if in_chl != out_chl:

            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x, skip_x):

        _, _, H, W = skip_x.size()
        
        up_out = F.upsample(x, size=(H, W), mode='bilinear')
        
        conv_out = self.encode(torch.cat([up_out, skip_x], 1))

        if self.conv_x is None:

            res_out = F.relu(conv_out + up_out)

        else: 

            res_out = F.relu(conv_out + self.conv_x(up_out))


        return res_out

class EncodeBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3, bn=False):
        super(EncodeBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=bn),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=bn),
        )
    def forward(self, x):

        conv_out = self.encode(x)

        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out

class CascadedConvBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3, bn=False):
        super(CascadedConvBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=bn),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=bn),
            )

    def forward(self, x):

        conv_out = self.encode(x)

        return conv_out

class DecodeBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3, bn=False):
        super(DecodeBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=bn),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=bn),
        )

    def forward(self, x, skip_x):

        _, _, H, W = skip_x.size()

        up_out = F.upsample(x, size=(H, W), mode='bilinear')

        conv_out = self.encode(torch.cat([up_out, skip_x], 1))

        return conv_out

class FBPConvNet(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=64):
        super(FBPConvNet, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=True)
        self.encoder1 = EncodeBlock2D(model_chl, model_chl, bn=True)
        self.encoder2 = EncodeBlock2D(model_chl, model_chl*2, bn=True)
        self.encoder3 = EncodeBlock2D(model_chl*2, model_chl*4, bn=True)
        self.encoder4 = EncodeBlock2D(model_chl*4, model_chl*8, bn=True)

        self.conv_center = CascadedConvBlock2D(model_chl*8, model_chl*16, bn=True)

        self.decoder4 = DecodeBlock2D(model_chl*16, model_chl*8, bn=True)
        self.decoder3 = DecodeBlock2D(model_chl*8, model_chl*4, bn=True)
        self.decoder2 = DecodeBlock2D(model_chl*4, model_chl*2, bn=True)
        self.decoder1 = DecodeBlock2D(model_chl*2, model_chl, bn=True)

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):

        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)
        res_x2, down_x2 = self.encoder2(down_x1)
        res_x3, down_x3 = self.encoder3(down_x2)
        res_x4, down_x4 = self.encoder4(down_x3)

        x_center = self.conv_center(down_x4)

        out_x4 = self.decoder4(x_center, res_x4)
        out_x3 = self.decoder3(out_x4, res_x3)
        out_x2 = self.decoder2(out_x3, res_x2)
        out_x1 = self.decoder1(out_x2, res_x1)

        out = F.relu(self.conv_end(out_x1) + x)

        return out