"""
author: Min Seok Lee and Wooseok Shin
Github repo: https://github.com/Karel911/TRACER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EfficientNet import EfficientNet
from util.effi_utils import get_model_shape
from modules.att_modules import RFB_Block, aggregation, ObjectAttention


class TRACER(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = EfficientNet.from_pretrained(f'efficientnet-b{cfg.arch}', advprop=True)
        self.block_idx, self.channels = get_model_shape()

        # Receptive Field Blocks
        channels = [int(arg_c) for arg_c in cfg.RFB_aggregated_channel]
        self.rfb2 = RFB_Block(self.channels[1], channels[0])
        self.rfb3 = RFB_Block(self.channels[2], channels[1])
        self.rfb4 = RFB_Block(self.channels[3], channels[2])

        # Multi-level aggregation
        self.agg = aggregation(channels)

        # Object Attention
        self.ObjectAttention2 = ObjectAttention(channel=self.channels[1], kernel_size=3)
        self.ObjectAttention1 = ObjectAttention(channel=self.channels[0], kernel_size=3)

    def forward(self, inputs):
        B, C, H, W = inputs.size()

        # EfficientNet backbone Encoder
        x = self.model.initial_conv(inputs)
        features = self.model.get_blocks(x, H, W)

        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)

        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        D_1 = self.ObjectAttention2(D_0, features[1])
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode='bilinear')

        ds_map = F.interpolate(D_1, scale_factor=2, mode='bilinear')
        D_2 = self.ObjectAttention1(ds_map, features[0])
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode='bilinear')

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3

        return torch.sigmoid(final_map), (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2))