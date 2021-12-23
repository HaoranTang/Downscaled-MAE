import sys
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from torchvision.datasets import ImageFolder
import argparse

def build_dataset(args):

    if args.dataset == 'cifar10':
        trainset =  CIFAR10(root='cifar10', train=True, transform=DataAugmentationForMAE(args, True), download=True)
        testset =  CIFAR10(root='cifar10', train=False, transform=DataAugmentationForMAE(args, False), download=True)
    elif args.dataset == 'stl10':
        trainset =  STL10(root='stl10', split='train', transform=DataAugmentationForMAE(args, True), download=True)
        testset =  STL10(root='stl10', split='test', transform=DataAugmentationForMAE(args, False), download=True)
    elif args.dataset == 'miniImageNet':
        trainset =  miniImageNet(root='mini_imagenet_mem', transform=DataAugmentationForMAE(args, True))
        testset =  miniImageNet(root='mini_imagenet_test', transform=DataAugmentationForMAE(args, False))
    else:
        print('unsupported dataset: ' + args.dataset)
        sys.exit(1)
    return trainset, testset

class miniImageNet(ImageFolder):
    def __init__(self, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.my_transform = transform

    def __getitem__(self, index):
        img,tgt = ImageFolder.__getitem__(self, index)
        return self.my_transform(img), tgt

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

class DataAugmentationForMAE(object):
    def __init__(self, args, train=True):
        if args.dataset == 'cifar10':
            if train:
                self.transform = train_transform_cifar10
            else:
                self.transform = test_transform_cifar10
        elif args.dataset == 'stl10':
            if train:
                self.transform = train_transform_stl10
            else:
                self.transform = test_transform_stl10
        elif args.dataset == 'miniImageNet':
            if train:
                self.transform = train_transform_mini
            else:
                self.transform = test_transform_mini
            
        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

train_transform_cifar10 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform_stl10 = transforms.Compose([
    transforms.RandomResizedCrop(96),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_transform_mini = transforms.Compose([
    transforms.RandomResizedCrop(84),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform_mini = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
