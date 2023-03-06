import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from PIL import Image


def get_transform(args):
    transform = Compose([
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(), # HWC->CHW, divide by 255
            Lambda(lambda t: (t * 2) - 1)  
        ])
    return transform

# reverse transform
def get_reverse_transform(args):
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage()
    ])
    return reverse_transform

class P2Dataset(Dataset):
    def __init__(self, root, labels, transform=None):
        self.filenames = sorted(os.listdir(root))
        self.labels = labels
        self.root = root
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, idx):
        imagePath = os.path.join(self.root, self.filenames[idx])
        image = Image.open(imagePath)
        if self.transform is not None:
            image = self.transform(image)
        
        #torch.LongTensor(4) means input a (random) vector with dimension = 4
        label = torch.from_numpy(
            np.array(self.labels[idx])
        )
        return image, label

    def __len__(self):
        return self.len