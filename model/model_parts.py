import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gussin(v):
    outk = []
    v = v
    for i in range(32):
        for k in range(32):

            out = []
            for x in range(32):
                row = []
                for y in range(32):
                    cord_x = i
                    cord_y = k
                    dis_x = np.abs(x - cord_x)
                    dis_y = np.abs(y - cord_y)
                    dis_add = -(dis_x * dis_x + dis_y * dis_y)
                    dis_add = dis_add / (2 * v * v)
                    dis_add = math.exp(dis_add) / (2 * math.pi * v * v)

                    row.append(dis_add)
                out.append(row)

            outk.append(out)

    out = np.array(outk)
    f = out.sum(-1).sum(-1)
    q = []
    for i in range(1024):
        g = out[i] / f[i]
        q.append(g)
    out = np.array(q)
    return torch.from_numpy(out)


class ConvReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvReluBlock, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convrelu(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.convrelu(x)


class _PartialConv_(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, kernel_size=3, stride=1):
        super(_PartialConv_, self).__init__()
        self.feat_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        feat, mask = input[0], input[1]
        out = self.feat_conv(feat * mask)
        with torch.no_grad():
            out_mask = self.mask_conv(mask)
        if self.feat_conv.bias is not None:
            out_bias = self.feat_conv.bias.view(1, -1, 1, 1).expand_as(out)
        else:
            out_bias = torch.zeros_like(out)
        mask_zeros = (out_mask==0)
        mask_sum = out_mask.masked_fill_(mask_zeros, 1.) # torch.tensor.masked_fill_()
        out = (out - out_bias) / mask_sum + out_bias
        out = out.masked_fill_(mask_zeros, 0.)
        new_mask = torch.ones_like(out)
        new_mask = new_mask.masked_fill_(mask_zeros, 0.)
        out = self.bn(out)
        out = self.relu(out)
        return out, new_mask


class MultuScaleFilling(nn.Module):
    def __init__(self, hid_ch, mode):
        super(MultuScaleFilling, self).__init__()
        if mode == "textures":
            ch_list = [hid_ch, 2*hid_ch, 4*hid_ch]
            self.conv_f1 = nn.Sequential(
                nn.Conv2d(ch_list[0], 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.conv_f2 = nn.Sequential(
                nn.Conv2d(ch_list[1], 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.conv_f3 = nn.Sequential(
                nn.Conv2d(ch_list[2], 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        elif mode == "structures":
            ch_list = [8*hid_ch, 8*hid_ch, 8*hid_ch]
            self.conv_f1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_list[0], 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.conv_f2 = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_list[0], 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            self.conv_f3 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_list[0], 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        self.convrelu = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        sequence_3 = []
        sequence_5 = []
        sequence_7 = []
        for _ in range(5):
            sequence_3 += [_PartialConv_(in_channels=256, out_channels=256, kernel_size=3)]
            sequence_5 += [_PartialConv_(in_channels=256, out_channels=256, kernel_size=5)]
            sequence_7 += [_PartialConv_(in_channels=256, out_channels=256, kernel_size=7)]
        self.partial_convs_3 = nn.Sequential(*sequence_3)
        self.partial_convs_5 = nn.Sequential(*sequence_5)
        self.partial_convs_7 = nn.Sequential(*sequence_7)

    def forward(self, f1, f2, f3, mask):
        x = torch.cat([self.conv_f1(f1), self.conv_f2(f2), self.conv_f3(f3)], dim=1)
        x = self.convrelu(x)
        pc_3, _ = self.partial_convs_3([x, mask])
        pc_5, _ = self.partial_convs_5([x, mask])
        pc_7, _ = self.partial_convs_7([x, mask])
        out = torch.cat([pc_3, pc_5, pc_7], dim=1)
        return out


class _SqueezeExtractionLayer(nn.Module):
    def __init__(self, in_channels=32, ratio=16):
        super(_SqueezeExtractionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() 
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


class BPA(nn.Module):
    def __init__(self, in_channels=512, ratio=16):
        super(BPA, self).__init__()
        self.se_layer = _SqueezeExtractionLayer(in_channels=in_channels, ratio=ratio)
        gus = _gussin(1.5)
        self.gus = torch.unsqueeze(gus, 1).double()
        self.convrelu = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.se_layer(x)
        # spatial step:
        x_spatial = x.expand(h*w, c, h, w)
        gus = self.gus.float().to(device=x_spatial.device)
        x_spatial = (x_spatial * gus).sum(-1).sum(-1).permute(1, 0).reshape(c, h, w)
        # range step:
        x_range = self.sigmoid(x)
        x_range_3x3 = F.unfold(x_range, kernel_size=(3, 3), padding=(1, 1)).permute(2, 1, 0).reshape(h*w, 3, 3, c)
        x_range_1x1 = x_range.permute(2, 3, 0, 1).reshape(h*w, 1, 1, c)
        x_range_1x1 = (x_range_1x1 * x_range_3x3).sum(-1)
        x_range_1x1 = F.softmax(x_range_1x1.reshape(h*w, 3*3), dim=1).reshape(h*w, 3, 3)
        x_range = (x_range_1x1.unsqueeze(1) * x_range_3x3.permute(0, 3, 1, 2)).sum(-1).sum(-1).permute(1, 0).reshape(c, h, w)
        # final cat and conv
        out = self.convrelu(torch.cat([x_range, x_spatial], dim=0).reshape(b, 2*c, h, w))
        return out


class PreDecoder(nn.Module):
    def __init__(self, hid_ch):
        super(PreDecoder, self).__init__()
        self.up_f2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(512, hid_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hid_ch),
            nn.ReLU()
        )
        self.up_f4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 2*hid_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2*hid_ch),
            nn.ReLU()
        )
        self.up_f8 = nn.Sequential(
            nn.Conv2d(512, 4*hid_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4*hid_ch),
            nn.ReLU()
        )
        self.up_f16 = nn.Sequential(
            nn.Conv2d(512, 8*hid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*hid_ch),
            nn.ReLU()
        )
        self.up_f32 = nn.Sequential(
            nn.Conv2d(512, 8*hid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*hid_ch),
            nn.ReLU(),
            nn.Conv2d(8*hid_ch, 8*hid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*hid_ch),
            nn.ReLU()
        )
        self.up_f64 = nn.Sequential(
            nn.Conv2d(512, 8*hid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*hid_ch),
            nn.ReLU(),
            nn.Conv2d(8*hid_ch, 8*hid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*hid_ch),
            nn.ReLU(),
            nn.Conv2d(8*hid_ch, 8*hid_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8*hid_ch),
            nn.ReLU(),
        )

    def forward(self, feat_2, feat_4, feat_8, feat_16, feat_32, feat_64, feat_bpa):
        feat_2 = feat_2 + self.up_f2(feat_bpa)
        feat_4 = feat_4 + self.up_f4(feat_bpa)
        feat_8 = feat_8 + self.up_f8(feat_bpa)
        feat_16 = feat_16 + self.up_f16(feat_bpa)
        feat_32 = feat_32 + self.up_f32(feat_bpa)
        feat_64 = feat_64 + self.up_f64(feat_bpa)
        return feat_2, feat_4, feat_8, feat_16, feat_32, feat_64


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpBlock, self).__init__()
        self.upconvrelu = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upconvrelu(x)

