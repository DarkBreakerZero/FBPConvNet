# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.autograd as autograd


# input (Tensor)
# pad (tuple)
# mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
# value – fill value for 'constant' padding. Default: 0

class ConvRelu3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_relu=True):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.relu = nn.LeakyReLU(inplace=True)
        if is_relu is False:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvBnRelu3d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=False, is_relu=True):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm3d(out_chl, eps=1e-4)
        if is_relu is True:
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, conv1_bn=True, conv2_bn=True):
        super(StackEncoder, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=conv1_bn),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_bn=conv2_bn),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool3d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class EncoderResBlock3D(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(EncoderResBlock3D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        res_out = x + conv_out

        return res_out


class DeconderResBlock3D(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(DeconderResBlock3D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu3d(in_chl * 2, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu3d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, cat_x, upsample_x):
        conv_out = self.encode(cat_x)
        res_out = upsample_x + conv_out

        return res_out


class CLEAR_UNetL3(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(CLEAR_UNetL3, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.c1 = ConvBnRelu3d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.res1 = EncoderResBlock3D(model_chl, model_chl)
        self.d1 = ConvBnRelu3d(model_chl, model_chl * 2, kernel_size=kernel_size, dilation=1, stride=(1, 2, 2),
                               groups=1)

        self.res2 = EncoderResBlock3D(model_chl * 2, model_chl * 2)
        self.d2 = ConvBnRelu3d(model_chl * 2, model_chl * 4, kernel_size=kernel_size, dilation=1, stride=(1, 2, 2),
                               groups=1)

        self.res3 = EncoderResBlock3D(model_chl * 4, model_chl * 4)
        self.d3 = ConvBnRelu3d(model_chl * 4, model_chl * 8, kernel_size=kernel_size, dilation=1, stride=(1, 2, 2),
                               groups=1)

        self.res4 = EncoderResBlock3D(model_chl * 8, model_chl * 8)

        self.u3 = nn.ConvTranspose3d(model_chl * 8, model_chl * 4, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 1, 1), stride=(1, 2, 2))
        self.ures3 = DeconderResBlock3D(model_chl * 4, model_chl * 4)

        self.u2 = nn.ConvTranspose3d(model_chl * 4, model_chl * 2, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 1, 1), stride=(1, 2, 2))
        self.ures2 = DeconderResBlock3D(model_chl * 2, model_chl * 2)

        self.u1 = nn.ConvTranspose3d(model_chl * 2, model_chl * 1, kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2, output_padding=(0, 1, 1), stride=(1, 2, 2))
        self.ures1 = DeconderResBlock3D(model_chl * 1, model_chl * 1)

        self.out = ConvBnRelu3d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):
        c1 = self.c1(x)
        res1 = self.res1(c1)
        d1 = self.d1(res1)
        # print(c1.size(), res1.size(), d1.size())

        res2 = self.res2(d1)
        d2 = self.d2(res2)
        # print(res2.size(), d2.size())

        res3 = self.res3(d2)
        d3 = self.d3(res3)

        res4 = self.res4(d3)

        u3 = self.u3(res4)
        cat3 = torch.cat([u3, res3], 1)
        ures3 = self.ures3(cat3, u3)

        u2 = self.u2(ures3)
        cat2 = torch.cat([u2, res2], 1)
        ures2 = self.ures2(cat2, u2)

        u1 = self.u1(ures2)
        cat1 = torch.cat([u1, res1], 1)
        ures1 = self.ures1(cat1, u1)

        out = F.relu(self.out(ures1) + x)
        # out = F.leaky_relu(self.out(ures1) + x)

        return out


class StackDecoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDecoder, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu3d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, conv_res):
        _, _, S, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(S, H, W), mode='trilinear')
        conv_out = self.conv(torch.cat([up_out, conv_res], 1))
        return conv_out


class UNet3D(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=64):
        super(UNet3D, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.down1 = StackEncoder(self.in_chl, self.model_chl, kernel_size=3, conv1_bn=False)  # 256
        self.down2 = StackEncoder(self.model_chl, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu3d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu3d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu3d(self.model_chl, self.model_chl, kernel_size=3, stride=1),
                                 ConvBnRelu3d(self.model_chl, 1, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv1, d1 = self.down1(x)  # ;print('down1',down1.size())  #256
        conv2, d2 = self.down2(d1)  # ;print('down2',down2.size())  #128
        conv3, d3 = self.down3(d2)  # ;print('down3',down3.size())  #64
        conv4, d4 = self.down4(d3)  # ;print('down4',down4.size())  #32
        conv5 = self.center(d4)  # ; print('out  ',out.size())
        up4 = self.up4(conv5, conv4)  # ;print('out  ',out.size())
        up3 = self.up3(up4, conv3)  # ;print('out  ',out.size())
        up2 = self.up2(up3, conv2)  # ;print('out  ',out.size())
        up1 = self.up1(up2, conv1)  # ;print('out  ',out.size())
        conv6 = self.end(up1)
        out = F.relu(conv6 + x)
        return out


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl, eps=1e-4)
        if is_relu is True:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackEncoder2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class StackDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDecoder2d, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv_out = self.conv(torch.cat([up_out, conv_res], 1))
        return conv_out


class StackDenseEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseEncoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu2d(in_chl + out_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                  groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.relu(conv3 + x)
        else:
            convx = F.relu(conv3 + self.convx(x))

        down_out = F.max_pool2d(convx, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return convx, down_out


class StackDenseBlock2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDenseBlock2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv3 = ConvBnRelu2d(in_chl + out_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1,
                                  groups=1, is_relu=False)

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], 1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], 1))

        if self.convx is None:
            convx = F.relu(conv3 + x)
        else:
            convx = F.relu(conv3 + self.convx(x))

        return convx


class StackResDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResDecoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1,
                                  is_relu=False)
        self.convx = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                  is_relu=False)

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv1 = self.conv1(torch.cat([up_out, conv_res], 1))
        conv2 = self.conv2(conv1)
        convx = F.relu(conv2 + self.convx(torch.cat([up_out, conv_res], 1)))
        return convx


class UNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(UNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackEncoder2d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = nn.Sequential(ConvBnRelu2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3, stride=1),
                                    ConvBnRelu2d(self.model_chl * 16, self.model_chl * 16, kernel_size=3, stride=1))

        self.up4 = StackDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu2d(self.model_chl, 1, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.relu(x + conv6)
        return res_out


class DenseUNet2d(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=128):
        super(DenseUNet2d, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackDenseEncoder2d(self.model_chl, self.model_chl, kernel_size=3)
        self.down2 = StackDenseEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)
        self.down3 = StackDenseEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)
        self.down4 = StackDenseEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)

        self.center = StackDenseBlock2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(ConvBnRelu2d(self.model_chl, 1, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)
        conv2, d2 = self.down2(d1)
        conv3, d3 = self.down3(d2)
        conv4, d4 = self.down4(d3)
        conv5 = self.center(d4)
        up4 = self.up4(conv5, conv4)
        up3 = self.up3(up4, conv3)
        up2 = self.up2(up3, conv2)
        up1 = self.up1(up2, conv1)
        conv6 = self.end(up1)
        res_out = F.relu(x + conv6)
        return res_out

class GeneratorCLEAR(nn.Module):

    def __init__(self, recon_op, chl=32):
        super(GeneratorCLEAR, self).__init__()
        self.chl = chl
        self.recon_op = recon_op

        self.net = nn.ModuleList()
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))

    def forward(self, proj, mask):

        proj_net = self.net[0](proj)
        proj_wrt = proj_net * (1 - mask) + proj * mask

        img_fbp = self.recon_op.backprojection(self.recon_op.filter_sinogram(proj_wrt)) * 1024
        # print(img_fbp.size())
        img_net = self.net[1](img_fbp)
        # print(img_net.size())
        proj_re = self.recon_op.forward(img_net / 1024)
        # print(proj_re.size())

        return proj_net, img_fbp, img_net, proj_re

class GeneratorCLEARV2(nn.Module):

    def __init__(self, recon_op, chl=32):
        super(GeneratorCLEARV2, self).__init__()
        self.chl = chl
        self.recon_op = recon_op

        self.net = nn.ModuleList()
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))

    def forward(self, proj, mask):

        proj_net = self.net[0](proj)
        proj_wrt = proj_net * (1 - mask) + proj * mask

        img_fbp = self.recon_op.backprojection(self.recon_op.filter_sinogram(proj_wrt)) * 1024
        # print(img_fbp.size())
        img_net = self.net[1](img_fbp)
        # print(img_net.size())
        # proj_re = self.recon_op.forward(img_net / 1024)
        # print(proj_re.size())

        return proj_wrt, img_fbp, img_net

class GeneratorCLEARMouse(nn.Module):

    def __init__(self, recon_op, chl=32):
        super(GeneratorCLEARMouse, self).__init__()
        self.chl = chl
        self.recon_op = recon_op

        self.net = nn.ModuleList()
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))
        self.net = self.net.append(CLEAR_UNetL3(in_chl=1, out_chl=1, model_chl=self.chl))

    def forward(self, proj, mask, img_mask):

        proj_net = self.net[0](proj)
        proj_wrt = proj_net * (1 - mask) + proj * mask

        img_fbp = self.recon_op.backprojection(self.recon_op.filter_sinogram(proj_wrt)) * 1024 * 50 * img_mask
        # print(img_fbp.size())
        img_net = self.net[1](img_fbp)
        # print(img_net.size())
        # proj_re = self.recon_op.forward(img_net / 1024)
        # print(proj_re.size())

        return proj_wrt, img_fbp, img_net

class DiscriminatorCLEAR(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(DiscriminatorCLEAR, self).__init__()
        self.ConvLayers = nn.Sequential(
            ConvBnRelu3d(in_chl, model_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl, model_chl, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1, is_bn=True),

            ConvBnRelu3d(model_chl * 1, model_chl * 2, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl * 2, model_chl * 2, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1,
                         is_bn=True),

            ConvBnRelu3d(model_chl * 2, model_chl * 4, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl * 4, model_chl * 4, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1,
                         is_bn=True),

            ConvBnRelu3d(model_chl * 4, model_chl * 8, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=True),
            ConvBnRelu3d(model_chl * 8, model_chl * 8, kernel_size=3, dilation=1, stride=(1, 2, 2), groups=1,
                         is_bn=True)
        )
        self.FCLayer = nn.Sequential(
            nn.Linear(model_chl * 8, out_chl)
        )

    def forward(self, x):
        out = self.ConvLayers(x)
        # print(out.size())
        out = torch.reshape(F.adaptive_avg_pool3d(out, (1, 1, 1)), [out.shape[0], out.shape[1]])
        # print(out.size())
        out = self.FCLayer(out)
        # print(out.size())

        return out

def compute_gradient_penalty(D, real_samples, fake_samples):
    # print(real_samples.size())
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.ndim == 5:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    else:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class Discriminator2D(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(Discriminator2D, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl
        self.ConvLayers = nn.Sequential(
            ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl, self.model_chl, kernel_size=3, stride=2, is_bn=False, is_relu=False),

            ConvBnRelu2d(self.model_chl, self.model_chl * 2, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 2, self.model_chl * 2, kernel_size=3, stride=2, is_bn=False, is_relu=False),

            ConvBnRelu2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 4, kernel_size=3, stride=2, is_bn=False, is_relu=False),

            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3, stride=1),
            ConvBnRelu2d(self.model_chl * 8, self.model_chl * 8, kernel_size=3, stride=2, is_bn=False, is_relu=False),
        )
        self.FCLayer = nn.Sequential(
            nn.Linear(self.model_chl * 8, self.out_chl)
        )

    def forward(self, x):
        out = self.ConvLayers(x)
        out = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)), [out.shape[0], out.shape[1]])
        out = self.FCLayer(out)

        return out