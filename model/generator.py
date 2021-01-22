import torch
import torch.nn as nn
from .model_parts import ConvReluBlock, ResBlock, MultuScaleFilling, BPA, PreDecoder, UpBlock


class Generator(nn.Module):
    def __init__(self, in_ch=3, hid_ch=64, res_blocks=5):
        super(Generator, self).__init__()
        self.maxpool_mask = nn.MaxPool2d(kernel_size=8) 
        # Encoder
        self.down1 = nn.Conv2d(in_ch, hid_ch, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvReluBlock(in_channels=hid_ch, out_channels=2*hid_ch)
        self.down3 = ConvReluBlock(in_channels=2*hid_ch, out_channels=4*hid_ch)
        self.down4 = ConvReluBlock(in_channels=4*hid_ch, out_channels=8*hid_ch)
        self.down5 = ConvReluBlock(in_channels=8*hid_ch, out_channels=8*hid_ch)
        self.down6 = ConvReluBlock(in_channels=8*hid_ch, out_channels=8*hid_ch)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # Middle part:
        self.middle_resblock = ResBlock(in_channels=8*hid_ch, kernel_size=3, stride=1, padding=1)
        #   # Filling: 
        self.tex_fill = MultuScaleFilling(hid_ch=hid_ch, mode="textures")
        self.str_fill = MultuScaleFilling(hid_ch=hid_ch, mode="structures")
        middle = []
        for _ in range(res_blocks):
            middle += [ResBlock(in_channels=8*hid_ch)]
        self.middle = nn.Sequential(*middle)
        self.middle_conv = ConvReluBlock(768, 256, kernel_size=1, stride=1, padding=0)
        self.middle_fuse = ConvReluBlock(512, 512, kernel_size=1, stride=1, padding=0)
        #   # BPA:
        self.bpa = BPA(in_channels=512)
        self.pre_decoder = PreDecoder(hid_ch)
        # Decoder:
        self.up6 = UpBlock(in_channels=8*hid_ch, out_channels=8*hid_ch, scale_factor=2)
        self.up5 = UpBlock(in_channels=16*hid_ch, out_channels=8*hid_ch, scale_factor=2)
        self.up4 = UpBlock(in_channels=16*hid_ch, out_channels=4*hid_ch, scale_factor=2)
        self.up3 = UpBlock(in_channels=8*hid_ch, out_channels=2*hid_ch, scale_factor=2)
        self.up2 = UpBlock(in_channels=4*hid_ch, out_channels=hid_ch, scale_factor=2)
        self.up1 = UpBlock(in_channels=2*hid_ch, out_channels=3, scale_factor=2, final=True)
        # for train:
        self.textures_out = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.structures_out = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img, mask):
        mask = self.maxpool_mask(mask).expand(1, 256, 32, 32)
        mask = torch.add(torch.neg(mask), 1.)
        # Encoder
        feat_2 = self.down1(img)
        feat_4 = self.down2(self.maxpool(feat_2))
        feat_8 = self.down3(self.maxpool(feat_4))
        feat_16 = self.down4(self.maxpool(feat_8))
        feat_32 = self.down5(self.maxpool(feat_16))
        feat_64 = self.down6(self.maxpool(feat_32))
        # Middle part
        feat_middle = self.middle_resblock(feat_64)
        ##### filling
        feat_tex = self.middle_conv(self.tex_fill(feat_2, feat_4, feat_8, mask))
        feat_str = self.middle_conv(self.str_fill(feat_16, feat_32, feat_64, mask))
        feat_fill = self.middle_fuse(torch.cat([feat_str, feat_tex], dim=1))
        #### bpa
        feat_bpa = self.bpa(feat_fill)
        feat_2, feat_4, feat_8, feat_16, feat_32, feat_64 = self.pre_decoder(feat_2, feat_4, feat_8, feat_16, feat_32, feat_middle, feat_bpa)
        # Decoder
        out = self.up6(feat_64)
        out = torch.cat([out, feat_32], dim=1)
        out = self.up5(out)
        out = torch.cat([out, feat_16], dim=1)
        out = self.up4(out)
        out = torch.cat([out, feat_8], dim=1)
        out = self.up3(out)
        out = torch.cat([out, feat_4], dim=1)
        out = self.up2(out)
        out = torch.cat([out, feat_2], dim=1)
        out = self.up1(out)
        if self.train:
            tex_ = self.textures_out(feat_tex)
            str_ = self.structures_out(feat_str)
            return out, tex_, str_
        else:
            return out


if __name__ == "__main__":
    x = torch.rand([1, 3, 256, 256])
    mask = torch.ones([1, 1, 256, 256])
    model = Generator()
    model.eval()
    out = model(x, mask)
    print("Done!", x.shape)