import argparse

import torch
import torch.nn.functional as F
import torchvision

# import class()
from model.generator import Generator
from model.discriminator import NLayerDiscriminator as Discriminator
from loss_metric.losses import VGG16, PerceptualLoss, StyleLoss, GANLoss

# import def()
from dataloader.dataloader import Loader


# init dataloader
data_loader = Loader()


# init parser
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda:0", help="Device name")

args = parser.parse_args()


# init models
device = torch.device(args.device)
model_G = Generator().to(device)
model_D = Discriminator().to(device)


# init optimizer
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.0002)#, weight_decay=0.0005)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0002)#, weight_decay=0.0005)


# init losses:
criterion_GAN = GANLoss().to(device)

def criterion_L1(x, y, interpolate=False):
    if interpolate == True: 
        y = F.interpolate(y, size=(32, 32), mode="bilinear")
    return F.l1_loss(x, y)

def criterion_L2(x, y, interpolate=False):
    if interpolate == True: 
        y = F.interpolate(y, size=(32, 32), mode="bilinear")
    return F.mse_loss(x, y)

criterion_Pe = PerceptualLoss().to(device)
criterion_Style = StyleLoss().to(device)

def criterion_G(out, out_tex, out_str, gt, str_, pred_fake_G, pred_real_G):
    loss_re = criterion_L1(out, gt, interpolate=False) * 1
    loss_pe = criterion_Pe(out, gt) * 0.1
    loss_style = criterion_Style(out, gt) * 250
    loss_GAN = criterion_GAN(pred_fake_G, pred_real_G, False) * 0.2
    loss_tex = criterion_L1(out_tex, gt, interpolate=True) * 1 
    loss_str = criterion_L1(out_str, str_, interpolate=True) * 1
    return loss_re + loss_pe + loss_style + loss_tex + loss_str + loss_GAN


# init train:
def train():
    iter = 0
    for epoch in range(100):
        for i, data in enumerate(data_loader):
            iter += 1
            img, gt, str_, mask = data
            img, gt, str_, mask = img.to(device), gt.to(device), str_.to(device), mask.to(device)
            outs = model_G(img, mask)
            out, out_tex, out_str = outs
            # Discriminator:
            optimizer_D.zero_grad()
            pred_fake_D = model_D(out.detach())
            pred_true_D = model_D(gt)
            loss_D = criterion_GAN(pred_fake_D, pred_true_D, True)
            loss_D.backward()
            optimizer_D.step()
            # Generator:
            optimizer_G.zero_grad()
            pred_fake_G = model_D(out)
            pred_real_G = model_D(gt)
            loss_G = criterion_G(out, out_tex, out_str, gt, str_, pred_fake_G, pred_real_G)
            loss_G.backward()
            optimizer_G.step()
            print(epoch, iter, loss_G.item(), loss_D.item())
            image_out = torch.cat([img, out, gt], 0)
            grid = torchvision.utils.make_grid(image_out)
            if iter % 10 == 0:
                torchvision.transforms.functional.to_pil_image(grid).save('./saveimg/000.png')
            if iter % 100 == 0:
                torchvision.transforms.functional.to_pil_image(grid).save('./saveimg/img%i.png' % iter)



if __name__ == "__main__":
    train()
