import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CIFAR10C(datasets.VisionDataset):
    """
    CIFAR-10-C dataset class to handle corrupted CIFAR-10 images.
    
    Args:
        root (string): Root directory of dataset where ``CIFAR-10-C/*`` will exist.
        corruption_type (string): Type of corruption to load (e.g., 'brightness', 'contrast').
        transform (callable, optional): A function/transform that takes in an PIL image
          and returns a transformed version. E.g, ``transforms.ToTensor()``
    """
    def __init__(self, root='./data', corruption_type='brightness', severity=1, transform=None):
        super(CIFAR10C, self).__init__(root, transform=transform)
        self.corruption_type = corruption_type
        self.severity = severity
        self.data, self.targets = self.load_cifar10c()

    def load_cifar10c(self):
        # Path to the specific corruption type
        file_path = os.path.join(self.root, f'{self.corruption_type}.npy')
        # print("File path check")
        # print(file_path)
        labels_path = os.path.join(self.root, 'labels.npy')

        # Load the corrupted data and labels
        data = np.load(file_path)
        targets = np.load(labels_path)

        # Each corruption file contains 5 levels of severity, we pick the required one
        data = data[(self.severity - 1) * 10000:self.severity * 10000]
        targets = targets[(self.severity - 1) * 10000:self.severity * 10000]

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)