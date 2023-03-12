import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize,  Normalize
from PIL import Image

# TODO: special transform for any task
def get_transform(args):
    transform = Compose([
        Resize(args.image_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return transform

class P3Dataset(Dataset):
    def __init__(self, root, labels=None, transform=None):
        self.filenames = sorted(os.listdir(root))
        self.labels = labels
        self.root = root
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, idx):
        imagePath = os.path.join(self.root, self.filenames[idx])
        # if img is gray scale (n_ch = 1) -> convert to n_ch = 3
        image = Image.open(imagePath).convert("RGB") 
        if self.transform is not None:
            image = self.transform(image)
    
        if self.labels is not None:
        #torch.LongTensor(4) means input a (random) vector with dimension = 4
            label = torch.from_numpy(
                np.array(self.labels[idx])
            )
            return image, label
        else:
            return image
        

    def __len__(self):
        return self.len

class P3TestDataset(Dataset):
    def __init__(self, root, labels=None, transform=None):
        self.filenames = sorted(os.listdir(root))
        self.labels = labels
        self.root = root
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        imagePath = os.path.join(self.root, img_name)
        # if img is gray scale (n_ch = 1) -> convert to n_ch = 3
        image = Image.open(imagePath).convert("RGB") 
        if self.transform is not None:
            image = self.transform(image)

        return img_name, image
        

    def __len__(self):
        return self.len