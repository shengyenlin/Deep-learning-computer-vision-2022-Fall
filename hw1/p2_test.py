import os
import gc
import time
import random
import sys

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import SegformerFeatureExtractor

from PIL import Image
import numpy as np

from preprocess import P2TestDataset
from model import get_p2_model

IMG_PATH = sys.argv[1]
PRED_PATH = sys.argv[2]
MODEL_PATH = 'p2_model.pth'
NUM_CLASS = 7

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

cls_color =np.array([
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 255],
    [0, 0, 0]
]).astype("uint8")

def main():
    #load model feature extractor
    segformer_type = 'b4'
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        f"nvidia/segformer-{segformer_type}-finetuned-ade-512-512",
        num_labels = NUM_CLASS,
        ignore_mismatched_sizes=True, 
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    #prepare dataset and data loader
    test_dataset = P2TestDataset(
        IMG_PATH, isSegFormer=True, 
        feature_extractor=feature_extractor, 
        transform=transform
    )

    batch_size = 12
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    #load model
    model_name = 'segformer'
    model = get_p2_model(
        model_name, 
        NUM_CLASS,
        seg_type = segformer_type,
        mode = 'test')
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(
        state_dict['model']
    )
    model = model.to(device)

    preds = []
    out_imgs = []
    model.eval()
    with torch.no_grad():
        for out_img_name, batch in test_loader:
            img = batch['pixel_values']
            img = img.to(device)
            out = model(img)
            logits = out[0] #(12, 7, 128, 128)
            # print(logits.size())
            upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=(512, 512), # (height, width)
                    mode='bilinear',
                    align_corners=False
                )

            predict = upsampled_logits.argmax(dim=1)
            predict = predict.detach().cpu().numpy()

            preds.extend(predict)
            out_imgs.extend(out_img_name)

        preds = cls_color[preds]
        for i in range(len(preds)):
            out_path = os.path.join(
                PRED_PATH, out_imgs[i]
            )

            im = Image.fromarray(preds[i])
            im.save(out_path)

if __name__ == '__main__':
    main()
