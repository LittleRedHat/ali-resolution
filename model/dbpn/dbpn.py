# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
__author__ = 'zookeeper'
import torch
import torch.nn as nn
from model.rbpn.base_network import UpBlock, DownBlock, D_UpBlock, ConvBlock, D_DownBlock, UpBlockPix, DownBlockPix


class DBPN_5(nn.Module):
    def __init__(self, config):
        super(DBPN_5, self).__init__()
        num_channels = config.get("num_channels")
        base_filter = config.get("base_filter")
        feat = config.get("feat")
        num_stages = config.get("num_stages")
        scale_factor = config.get("scale_factor")
        self.residual = config.get('residual', False)
        self.scale_factor = scale_factor

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu',
                               norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu',
                               norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1,
                                     1, activation=None, norm=None)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, bicubic = x
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)
        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        if self.residual:
            x = x + bicubic
        return x


class DBPN_7(nn.Module):
    def __init__(self, config):
        super(DBPN_7, self).__init__()
        num_channels = config.get("num_channels")
        base_filter = config.get("base_filter")
        feat = config.get("feat")
        num_stages = config.get("num_stages")
        scale_factor = config.get("scale_factor")
        self.residual = config.get('residual', False)
        self.scale_factor = scale_factor

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu',
                               norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu',
                               norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1,
                                     1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, bicubic = x
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        if self.residual:
            x = x + bicubic
        return x


class DBPN_Pixel_3(nn.Module):
    def __init__(self, config):
        super(DBPN_Pixel_3, self).__init__()

        num_channels = config.get("num_channels")
        base_filter = config.get("base_filter")
        feat = config.get("feat")
        num_stages = config.get("num_stages")
        scale_factor = config.get("scale_factor")
        self.residual = config.get('residual', False)
        self.scale_factor = scale_factor
        self.norm = config.get('norm', None)
        self.act = config.get('act', 'prelu')

        self.kernel = config.get('kernel', 8)
        self.stride = config.get('stride', 4)
        self.padding = config.get('padding', 2)

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation=self.act,
                               norm=self.norm)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation=self.act,
                               norm=self.norm)
        # Back-projection stages
        self.up1 = UpBlockPix(base_filter, self.kernel, self.stride, self.padding, self.scale_factor)
        self.down1 = DownBlockPix(base_filter, self.kernel, self.stride, self.padding, self.scale_factor)
        self.up2 = UpBlockPix(base_filter, self.kernel, self.stride, self.padding, self.scale_factor)
        self.down2 = DownBlockPix(base_filter, self.kernel, self.stride, self.padding, self.scale_factor)
        self.up3 = UpBlockPix(base_filter, self.kernel, self.stride, self.padding, self.scale_factor)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1,
                                     1, activation=None, norm=None)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, bicubic = x
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        h3 = self.up3(self.down2(h2))

        x = self.output_conv(torch.cat((h3, h2, h1), 1))
        if self.residual:
            x = x + bicubic
        return x




class DBPN_3(nn.Module):
    def __init__(self, config):
        super(DBPN_3, self).__init__()

        num_channels = config.get("num_channels")
        base_filter = config.get("base_filter")
        feat = config.get("feat")
        num_stages = config.get("num_stages")
        scale_factor = config.get("scale_factor")
        self.residual = config.get('residual', False)
        self.scale_factor = scale_factor
        self.norm = config.get('norm', None)
        self.act = config.get('act', 'prelu')

        self.kernel = config.get('kernel', 8)
        self.stride = config.get('stride', 4)
        self.padding = config.get('padding', 2)

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation=self.act,
                               norm=self.norm)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation=self.act,
                               norm=self.norm)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, self.kernel, self.stride, self.padding)
        self.down1 = DownBlock(base_filter, self.kernel, self.stride, self.padding)
        self.up2 = UpBlock(base_filter, self.kernel, self.stride, self.padding)
        self.down2 = DownBlock(base_filter, self.kernel, self.stride, self.padding)
        self.up3 = UpBlock(base_filter, self.kernel, self.stride, self.padding)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1,
                                     1, activation=None, norm=None)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, bicubic = x
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        h3 = self.up3(self.down2(h2))

        # print(h1.shape, h2.shape, h3.shape)

        x = self.output_conv(torch.cat((h3, h2, h1), 1))
        if self.residual:
            x = x + bicubic
        return x


class DBPN_2(nn.Module):
    def __init__(self, config):
        super(DBPN_2, self).__init__()

        num_channels = config.get("num_channels")
        base_filter = config.get("base_filter")
        feat = config.get("feat")
        num_stages = config.get("num_stages")
        scale_factor = config.get("scale_factor")

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu',
                               norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu',
                               norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_filter, num_channels, 3, 1,
                                     1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))

        x = self.output(torch.cat((h2, h1), 1))

        return x

