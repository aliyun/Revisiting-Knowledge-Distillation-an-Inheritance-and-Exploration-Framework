#! /usr/bin/env python
import torch
import torchvision
from torchvision import transforms as T
import numpy as np


def cifar100(data_dir='./data/cifar100/', padding_mode='reflect', rotation=0):
    root = data_dir
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([129.3, 124.1, 112.4]) / 255.0,
                    np.array([68.2, 65.4, 70.4]) / 255.0),
    ])
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode=padding_mode),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=rotation),
        test_transform
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=False,
        transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=False,
        transform=test_transform)
    return train_set, test_set


def cifar10(data_dir='./data/cifar10/'):
    root = data_dir
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_transform = T.Compose([
        T.Pad(4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32),
        test_transform
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=test_transform)
    return train_set, test_set

