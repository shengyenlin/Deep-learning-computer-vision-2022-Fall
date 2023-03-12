import os
import json

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    img_id, img, cap_id, cap_id_y, cap_attn_mask = zip(*batch)
    # id = [BOS, ...]
    seq_len = [len(x) for x in cap_id]
    max_len = max(seq_len)
    cap_id_pad = pad_sequence(list(cap_id), batch_first=True, padding_value=0) #id: 0 = "[PAD]"
    cap_id_y_pad = pad_sequence(list(cap_id_y), batch_first=True, padding_value=0) #id: 0 = "[PAD]"
    cap_attn_mask_pad = pad_sequence(list(cap_attn_mask), batch_first=True, padding_value=1) #attn: 1 = no attention
    img = torch.stack(list(img))
    img_id = list(img_id)
    data = {'img_id': img_id, 'img': img, 'cap_id': cap_id_pad, 'cap_id_y': cap_id_y_pad, 'cap_attn_mask': cap_attn_mask_pad, 'max_len': max_len}
    return data

class P2Dataset(Dataset):
    def __init__(self, img_root, text_root, tokenizer, seq_max_len, transform=None, dy_pad=False):
        self.img_names = sorted(os.listdir(img_root))
        self.img_root = img_root
        self.transform = transform
        self.text_root = text_root
        self.tokenizer = tokenizer
        self.seq_max_len = seq_max_len

        f = open(self.text_root)
        self.data = json.load(f)
        self.len = len(self.data) 
        self.data_ids = list(self.data.keys())

        self.dy_pad = dy_pad

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        file_name = self.data[data_id]['file_name']

        # file name
        img_id = file_name.split(".")[0]

        # img 
        imagePath = os.path.join(self.img_root, file_name)
        img = Image.open(imagePath).convert("RGB") # if img is gray scale (n_ch = 1) -> convert to n_ch = 3
        if self.transform is not None:
            img = self.transform(img)
        
        # text
        cap = self.data[data_id]['caption']
        #print(data_id, img_id, cap)
        text_encode = self.tokenizer.encode(cap)
        # id = [BOS, ..., EOS]
        cap_id, cap_attn_mask = text_encode.ids, text_encode.attention_mask
        
        # id = [BOS, ...]
        # y = [..., EOS]
        cap_id, cap_id_y = cap_id[:-1], cap_id[1:]  
        cap_id = torch.from_numpy(np.array(cap_id))
        cap_id_y = torch.from_numpy(np.array(cap_id_y))
        cap_attn_mask = ~torch.from_numpy(np.array(cap_attn_mask[:-1])).bool()
        
        if self.dy_pad:
            return img_id, img, cap_id, cap_id_y, cap_attn_mask
        else:
        # pad to same length
            len_seq = len(cap_id)
            cap_id = F.pad(cap_id, (0, self.seq_max_len-len_seq), mode='constant', value=0)
            cap_id_y = F.pad(cap_id_y, (0, self.seq_max_len-len_seq), mode='constant', value=0)
            cap_attn_mask = F.pad(cap_attn_mask, (0, self.seq_max_len-len_seq), mode='constant', value=1)
            data = {'img_id':img_id, 'img':img, 'cap_id': cap_id, 'cap_id_y': cap_id_y, 'cap_attn_mask': cap_attn_mask}
            return data 

    def __len__(self):
        return self.len

class P2TestDataset(Dataset):
    def __init__(self, img_root, transform=None):
        self.img_names = sorted(os.listdir(img_root))
        self.img_root = img_root
        self.transform = transform
        self.len = len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        image_path = os.path.join(self.img_root, img_name)
        img_prefix = img_name.split(".")
        img_prefix = img_prefix[0]

        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img_prefix, img

    def __len__(self):
        return self.len