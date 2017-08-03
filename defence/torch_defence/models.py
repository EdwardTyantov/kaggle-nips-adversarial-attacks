#-*- coding: utf8 -*-
import torch
from torchvision.models.densenet import densenet201
from torchvision.models.inception import inception_v3


def densenet201_():
    model = densenet201(pretrained=False)
    return model


def inception_v3_():
    model = inception_v3(pretrained=False)
    return model


def factory(name):
    model_func = globals().get(name, None)
    if model_func is None:
        raise AttributeError("Model %s doesn't exist" % (name,))

    model = model_func()
    return model