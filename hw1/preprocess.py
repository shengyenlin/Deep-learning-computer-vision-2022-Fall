import os 
import imageio.v2 as imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image


class P1TestDataset(Dataset):
    def __init__(self, img_path, root, transform=None):
        self.root = root
        self.img_name = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def read_img(self, idx):
        path = os.path.join(self.root, self.img_name[idx])
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        return img
    
    def do_transform(self, img):
        return self.transform(img)

    def __getitem__(self, idx):
        img = self.read_img(idx)
        if self.transform is not None:
            img = self.do_transform(img)
        return img

class P1_TransformsModelB:
    def __init__(self):
        self.random_erase = 0.1
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        
        self.transform_P1_modelB = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(degrees = (-20, 20)),
                transforms.ToTensor(),
                self.normalize,
                # transforms.RandomErasing(p=self.random_erase),
            ]),
            'val':  transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ])
        }

class P2Dataset(Dataset):
    def __init__(
        self, path, isSegFormer=False, doEnsemble=False,
        feature_extractor=None, transform_seg=None, transform_vgg=None
        ):
        self.path = path
        self.data = None #3x512x512
        self.labels = None
        self.load_image_p2(path)
        self.transform_seg = transform_seg
        self.transform_vgg = transform_vgg
        self.isSegFormer = isSegFormer
        self.feature_extractor = feature_extractor
        self.doEnsemble = doEnsemble

    def __len__(self):
       return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        img_seg = self.read_img(idx)
        label = self.read_mask(idx)
        label = torch.from_numpy(label).type(torch.LongTensor)
        
        if self.isSegFormer:
            encoded_inputs = self.feature_extractor(
                img_seg, label, return_tensors='pt'
            )
            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
                encoded_inputs['labels'] = label
            return encoded_inputs
        # if self.doEnsemble:
        #     return encoded_inputs, img_vgg, label
        # else:
        #     return img_vgg, label

    def read_img(self, idx):
        path = os.path.join(self.path, self.data[idx])
        img = Image.open(path)
        if self.transform_seg is not None:
            img_seg = self.transform_seg(img)
        if self.transform_vgg is not None:
            img_vgg = self.transform_vgg(img)

        if self.doEnsemble:
            return img_seg, img_vgg
        else:
            return img_seg

    def read_mask(self, idx):
        path = os.path.join(self.path, self.labels[idx])
        mask = imageio.imread(path) #512x512x3
        #not np.empty
        masks = np.zeros((mask.shape[0],mask.shape[1]))

        #load mask
        mask = (mask >= 128).astype(int)
        #512x512x3 -> 512x512
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown 
        return masks

    def load_image_p2(self, path):
        data, labels = [], []
        path = os.listdir(path)
        path = sorted(path)
        for img_name in path:
            file_name = img_name.split('.')
            #data
            if file_name[1] == 'jpg':
                data.append(img_name)
            #label
            elif file_name[1] == 'png':
                labels.append(img_name)
        self.data = data
        self.labels = labels

class P2TestDataset(Dataset):
    def __init__(
        self, root, isSegFormer=False, 
        feature_extractor=None, transform=None
        ):
        self.root = root
        self.data_path = None #3x512x512
        self.load_image(self.root)
        self.transform = transform
        self.isSegFormer = isSegFormer
        self.feature_extractor = feature_extractor

    def __len__(self):
       return len(self.data_path)
    
    def __getitem__(self, idx):
        out_img_name, img = self.read_img(idx)
        if self.isSegFormer:
            encoded_inputs = self.feature_extractor(
                img, return_tensors='pt'
            )
            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
            return out_img_name, encoded_inputs
        else:
            return out_img_name, img

    def read_img(self, idx):
        pic_name = self.data_path[idx]
        path = os.path.join(self.root, pic_name)
        img_prefix = pic_name.split('.')[0]
        out_img_name = img_prefix + '.png'
        img = imageio.imread(path) #If using color gitter, change to Image.open()
        if self.transform is not None:
            img = self.transform(img)
        return out_img_name, img

    def load_image(self, path):
        path = os.listdir(path)
        path = sorted(path)
        paths = []
        for path_tmp in path:
            if path_tmp.split('.')[1] == 'jpg':
                paths.append(path_tmp)
        self.data_path = sorted(paths)

#TODO: randomcrop and filp to 256
class P2_TransformsModelA:
    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        
        self.transform = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ]),
            'val':  transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
                self.normalize,
            ])
        }