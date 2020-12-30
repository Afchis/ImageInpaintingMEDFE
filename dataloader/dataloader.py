import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image


class DataTrain(Dataset):
	def __init__(self):
		super().__init__()
		self.data_path = "./dataloader/data/"
		self.img_names = sorted(os.listdir(self.data_path + "gt/"))
		self.str_names = sorted(os.listdir(self.data_path + "st/"))
		self.msk_names = sorted(os.listdir(self.data_path + "mask/"))
		self.img2tensor = transforms.Compose([
			transforms.Resize((256, 256), interpolation=0),
			transforms.ToTensor()
		])

	def __len__(self):
		return len(self.img_names)

	def __getitem__(self, idx):
		gt = self.img2tensor(Image.open(self.data_path + "gt/" + self.img_names[idx]))
		str_ = self.img2tensor(Image.open(self.data_path + "st/" + self.str_names[idx]))
		mask = self.img2tensor(Image.open(self.data_path + "mask/" + self.msk_names[idx])) # do random!!!!
		img = gt * mask
		return img, gt, str_, mask
		

def Loader():
    print("Initiate DataLoader")
    train_dataset = DataTrain()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True)
    return train_loader