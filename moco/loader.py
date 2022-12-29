# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torchvision.datasets as datasets
from torchvision.datasets.folder import ImageFolder, default_loader
import os
import torch
import torch.utils.data as data

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TwoDiffTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, strong_transform):
        self.base_transform = base_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.strong_transform(x)
        return [q, k]



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index


class SubSetInstance(data.Subset):

    def __getitem__(self, index):

        sample, _ = self.dataset[self.indices[index]]
        return sample, index


class VTAB(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, mode=None, is_individual_prompt=False,
                 **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        self.train = train
        train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root, img_name), label))
        self.targets = torch.Tensor([s[1] for s in self.samples])
       
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

