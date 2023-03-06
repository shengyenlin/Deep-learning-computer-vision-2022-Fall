import os
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from PIL import Image

path_label2id = './p2/label2id.json'

class MiniDataset(Dataset):
    def __init__(self, root, transform=None):
        self.filenames = sorted(os.listdir(root))
        self.root = root
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, idx):
        imagePath = os.path.join(self.root, self.filenames[idx])
        image = Image.open(imagePath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.len

class OfficeDataset(Dataset):
    def __init__(self, img_root, df_path, transform=None, mode='train'):
        self.img_root = img_root
        self.df = pd.read_csv(df_path)
        self.transform = transform
        self.len = self.df.shape[0]
        self.mode = mode

        with open(path_label2id) as json_file:
            self.label2id = json.load(json_file)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        id = self.df.iloc[idx]['id']
        imagePath = os.path.join(self.img_root, img_name)
        img = Image.open(imagePath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.mode != 'test':
            label = self.df.iloc[idx]['label']
            label = torch.from_numpy(
                np.array(self.label2id[label])
            )
            return img, label
        else:
            return id, img_name, img

    def __len__(self):
        return self.len

