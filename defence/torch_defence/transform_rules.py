#-*- coding: utf8 -*-
import random
from PIL import Image
import torchvision.transforms as transforms
from .transforms import TenCropPick, SpatialPick, normalize, ColorJitter


def imagenet_like():

    crop_size = 299#224

    train_transformations = transforms.Compose([
        transforms.RandomSizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        lambda img: img if random.random() < 0.5 else img.transpose(Image.FLIP_TOP_BOTTOM),
        transforms.ToTensor(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        normalize,
    ])

    val_transformations = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    test_transformation = transforms.Compose([
        #TenCropPick(224),
        SpatialPick(),
        #transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    return {'train': train_transformations, 'val': val_transformations, 'test': test_transformation}
