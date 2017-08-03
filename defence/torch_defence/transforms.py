#-*- coding: utf8 -*-
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img



class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        transforms = []
        if brightness != 0:
            transforms.append(Brightness(brightness))
        if contrast != 0:
            transforms.append(Contrast(contrast))
        if saturation != 0:
            transforms.append(Saturation(saturation))

        RandomOrder.__init__(self, transforms)


class TenCropPick(object):
    """Four corner patches and center crop from image and its horizontal reflection.
    Pick one of the crop specified in the constructor
    """
    def __init__(self, size, index=0):
        self.size = size
        self.__index = index

    def __init_functions(self, w, h):
        funcs = []
        transp = lambda x: x.transpose(Image.FLIP_LEFT_RIGHT)
        funcs.append(lambda _img: CenterCrop(self.size)(_img))
        funcs.append(lambda _img: _img.crop((0, 0, self.size, self.size)))
        funcs.append(lambda _img: _img.crop((w - self.size, 0, w, self.size)))
        funcs.append(lambda _img: _img.crop((0, h - self.size, self.size, h)))
        funcs.append(lambda _img: _img.crop((w - self.size, h - self.size, w, h)))
        funcs.append(lambda _img: CenterCrop(self.size)(transp(_img)))
        funcs.append(lambda _img: transp(_img).crop((0, 0, self.size, self.size)))
        funcs.append(lambda _img: transp(_img).crop((w - self.size, 0, w, self.size)))
        funcs.append(lambda _img: transp(_img).crop((0, h - self.size, self.size, h)))
        funcs.append(lambda _img: transp(_img).crop((w - self.size, h - self.size, w, h)))
        return funcs

    def setter(self, value):
        if value < 0 or value > 9:
            raise ValueError('out of bounds [0,9]')
        self.__index = value

    index = property(fset=setter)

    def __call__(self, img):
        w, h = img.size
        func = self.__init_functions(w, h)[self.__index]

        return func(img)

    def __len__(self):
        return 10


def rotate_or_flip(img, number):
    if number <= 3:
        return img.rotate(90 * number)
    elif number == 4:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img.transpose(Image.FLIP_TOP_BOTTOM)


class SpatialPick(object):
    def __init__(self, index=0):
        self.__index = index

    def setter(self, value):
        max_value = self.__len__() - 1
        if value < 0 or value > max_value:
            raise ValueError('out of bounds [0,%d]' % max_value)
        self.__index = value

    def __len__(self):
        return 6

    index = property(fset=setter)

    def __call__(self, img):
        return rotate_or_flip(img, self.__index)
