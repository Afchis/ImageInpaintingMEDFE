import argparse

import torch
import torch.nn.functional as F

# import class()
from model.generator import Generator

# import def()
from dataloader.dataloader import Loader


parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, deafult="cuda:0", help="Device name")

args = parser.parse_args()

device = torch.device(args.device)
model = Generator()
data_loader = Loader()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, weight_decay=0.0005)

def l1_loss(x, y):
    return(x - y).mean()


def train():
    for epoch in range(100):
        for i, data in enumerate(data_loader):
            img, gt, str_, mask = data
            img, gt, str_, mask = img.to(device), gt.to(device), str_.to(device), mask.to(device)
            outs = model(img, mask)
            out, tex_out, str_out = outs
            out_loss = F.l1_loss(out, gt)
            tex_loss = F.l1_loss(tex_out, torch.nn.functional.interpolate(gt, size=(32, 32), mode='bilinear', align_corners=False))
            str_loss = F.l1_loss(str_out, torch.nn.functional.interpolate(str_, size=(32, 32), mode='bilinear', align_corners=False))
            loss = out_loss + tex_loss + str_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.item())


if __name__ == "__main__":
    train()
