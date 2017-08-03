#-*- coding: utf8 -*-
import os
import torch.utils.data as data
from torchvision.datasets.folder import default_loader


class ImageTestFolder(data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        imgs = os.listdir(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.imgs)
