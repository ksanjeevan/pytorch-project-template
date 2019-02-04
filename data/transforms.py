

import numpy as np

import torch
from torchvision import transforms


class ImageTransforms(object):

    def __init__(self, name, size, scale, ratio, colorjitter):
        self.transfs = {
            'val': transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=colorjitter[0], 
                    contrast=colorjitter[1], 
                    saturation=colorjitter[2]),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }[name]


    def apply(self, data, target):
        return self.transfs(data), target
