"""
author: Min Seok Lee and Wooseok Shin
"""
import numpy as np
import torch.nn as nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
from util.utils import *
import torch.nn.functional as F
from config import getConfig
from modules.conv_modules import BasicConv2d, DWConv, DWSConv


cfg = getConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Frequency_Edge_Module(nn.Module):
    def __init__(self, radius, channel):
        super(Frequency_Edge_Module, self).__init__()
        self.radius = radius
        self.UAM = UnionAttentionModule(channel, only_channel_tracing=True)

        # DWS + DWConv
        self.DWSConv = DWSConv(channel, channel, kernel=3, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel, channel, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.conv = BasicConv2d(channel, 1, 1)

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        batch, channels, rows, cols = img.shape
        mask = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def forward(self, x):
        """
        Input:
            The first encoder block representation: (B, C, H, W)
        Returns:
            Edge refined representation: X + edge (B, C, H, W)
        """
        x_fft = fft2(x, dim=(-2, -1))
        x_fft = fftshift(x_fft)

        # Mask -> low, high separate
        mask = self.mask_radial(img=x, r=self.radius).to(device)
        high_frequency = x_fft * (1 - mask)
        x_fft = ifftshift(high_frequency)
        x_fft = ifft2(x_fft, dim=(-2, -1))
        x_H = torch.abs(x_fft)

        x_H, _ = self.UAM.Channel_Tracer(x_H)
        edge_maks = self.DWSConv(x_H)
        skip = edge_maks.clone()

        edge_maks = torch.cat([self.DWConv1(edge_maks), self.DWConv2(edge_maks),
                               self.DWConv3(edge_maks), self.DWConv4(edge_maks)], dim=1) + skip
        edge = torch.relu(self.conv(edge_maks))

        x = x + edge  # Feature + Masked Edge information

        return x, edge


class RFB_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_Block, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class UnionAttentionModule(nn.Module):
    def __init__(self, n_channels, only_channel_tracing=False):
        super(UnionAttentionModule, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = cfg.gamma
        self.bn = nn.BatchNorm2d(n_channels)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Dropout3d(self.confidence_ratio)
        )
        self.channel_q = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_k = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_v = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.fc = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                            padding=0, bias=False)

        if only_channel_tracing == False:
            self.spatial_q = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_k = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_v = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def masking(self, x, mask):
        mask = mask.squeeze(3).squeeze(2)
        threshold = torch.quantile(mask, self.confidence_ratio, dim=-1, keepdim=True)
        mask[mask <= threshold] = 0.0
        mask = mask.unsqueeze(2).unsqueeze(3)
        mask = mask.expand(-1, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        masked_x = x * mask

        return masked_x

    def Channel_Tracer(self, x):
        avg_pool = self.GAP(x)
        x_norm = self.norm(avg_pool)

        q = self.channel_q(x_norm).squeeze(-1)
        k = self.channel_k(x_norm).squeeze(-1)
        v = self.channel_v(x_norm).squeeze(-1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        # a*v
        att = torch.matmul(alpha, v).unsqueeze(-1)
        att = self.fc(att)
        att = self.sigmoid(att)

        output = (x * att) + x
        alpha_mask = att.clone()

        return output, alpha_mask

    def forward(self, x):
        X_c, alpha_mask = self.Channel_Tracer(x)
        X_c = self.bn(X_c)
        x_drop = self.masking(X_c, alpha_mask)

        q = self.spatial_q(x_drop).squeeze(1)
        k = self.spatial_k(x_drop).squeeze(1)
        v = self.spatial_v(x_drop).squeeze(1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)

        return output


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel[2], channel[1], 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel[2], channel[0], 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel[1], channel[0], 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel[2], channel[2], 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel[2] + channel[1], channel[2] + channel[1], 3, padding=1)

        self.conv_concat2 = BasicConv2d((channel[2] + channel[1]), (channel[2] + channel[1]), 3, padding=1)
        self.conv_concat3 = BasicConv2d((channel[0] + channel[1] + channel[2]),
                                        (channel[0] + channel[1] + channel[2]), 3, padding=1)

        self.UAM = UnionAttentionModule(channel[0] + channel[1] + channel[2])

    def forward(self, e4, e3, e2):
        e4_1 = e4
        e3_1 = self.conv_upsample1(self.upsample(e4)) * e3
        e2_1 = self.conv_upsample2(self.upsample(self.upsample(e4))) \
               * self.conv_upsample3(self.upsample(e3)) * e2

        e3_2 = torch.cat((e3_1, self.conv_upsample4(self.upsample(e4_1))), 1)
        e3_2 = self.conv_concat2(e3_2)

        e2_2 = torch.cat((e2_1, self.conv_upsample5(self.upsample(e3_2))), 1)
        x = self.conv_concat3(e2_2)

        output = self.UAM(x)

        return output


class ObjectAttention(nn.Module):
    def __init__(self, channel, kernel_size):
        super(ObjectAttention, self).__init__()
        self.channel = channel
        self.DWSConv = DWSConv(channel, channel // 2, kernel=kernel_size, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel // 2, channel // 2, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel // 2, channel // 8, 1),
        )
        self.conv1 = BasicConv2d(channel // 2, 1, 1)

    def forward(self, decoder_map, encoder_map):
        """
        Args:
            decoder_map: decoder representation (B, 1, H, W).
            encoder_map: encoder block output (B, C, H, W).
        Returns:
            decoder representation: (B, 1, H, W)
        """
        mask_bg = -1 * torch.sigmoid(decoder_map) + 1  # Sigmoid & Reverse
        mask_ob = torch.sigmoid(decoder_map)  # object attention
        x = mask_ob.expand(-1, self.channel, -1, -1).mul(encoder_map)

        edge = mask_bg.clone()
        edge[edge > cfg.denoise] = 0
        x = x + (edge * encoder_map)

        x = self.DWSConv(x)
        skip = x.clone()
        x = torch.cat([self.DWConv1(x), self.DWConv2(x), self.DWConv3(x), self.DWConv4(x)], dim=1) + skip
        x = torch.relu(self.conv1(x))

        return x + decoder_map