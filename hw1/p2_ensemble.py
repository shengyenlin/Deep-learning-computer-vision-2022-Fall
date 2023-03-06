import os
import gc
import time
import random
from tqdm import tqdm

import numpy as np

import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms, models, datasets
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from PIL import Image


from utils import train_P2_segform, valid_P2_segform, save_checkpoint, FocalLoss, Interpolate
from preprocess import P2_TransformsModelA, P2Dataset
from model import get_p2_model
from mean_iou_evaluate import mean_iou_score
from viz_mask import viz_data

IMG_FILE = 'hw1_data/p2_data'
TRAIN_PATH = 'hw1_data/p2_data/train'
VAL_PATH = 'hw1_data/p2_data/validation'
TRAIN_NUM_PIC = 2000
VAL_NUM_PIC = 257
NUM_CLASS = 7
MODEL_SEG_PATH = './models/best_p2/segformer_best_7422.pth'
MODEL_VGG_PATH = './models/best_p2/VGG16FCN8.pth'

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def main():
    seg_type = 'b1'
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
    f"nvidia/segformer-{seg_type}-finetuned-ade-512-512",
    num_labels = NUM_CLASS,
    ignore_mismatched_sizes=True, 
    )
    transform_seg = transforms.Compose(
        [transforms.ToTensor()]
    )
    trans = P2_TransformsModelA()
    transform_vgg = trans.transform['train']

    val_dataset = P2Dataset(
        VAL_PATH, isSegFormer=True, doEnsemble=True,
        feature_extractor=feature_extractor, 
        transform_seg=transform_seg, transform_vgg=transform_vgg
    )

    batch_size = 12
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    models =[]
    model_name = 'segformer'
    model = get_p2_model(
        model_name, seg_type = seg_type, 
        mode = 'test', num_class = NUM_CLASS
        )
    state_dict = torch.load(MODEL_SEG_PATH)
    model.load_state_dict(
        state_dict['model']
    )
    model = model.to(device)
    models.append(model)

    model_name = 'VGG16_FCN8'
    model = get_p2_model(model_name, NUM_CLASS)
    state_dict = torch.load(MODEL_VGG_PATH)
    model.load_state_dict(
        state_dict['model']
    )
    model = model.to(device)
    models.append(model)
    # print(models[0].__class__.__name__)
    # print(models[1].__class__.__name__)

    preds = []
    labels = []

    softmax = torch.nn.Softmax(dim=1)
    is_first_mini_batch = True
    with torch.no_grad():
        for seg_batch, x, label in tqdm(val_loader, leave=False, colour='green'):
            x_seg = seg_batch['pixel_values']
            x, x_seg, label = x.to(device), x_seg.to(device), label.to(device)
            for model in models:
                model.eval()
                if model.__class__.__name__ == 'SegFormer':        
                    out = model(x_seg)
                    logits = out[0]
                    upsampled_logits = nn.functional.interpolate(
                            logits,
                            size=label.shape[-2:], # (height, width)
                            mode='bilinear',
                            align_corners=False
                        )
                    predict = softmax(upsampled_logits)
                    pred_accu = predict.detach().cpu().numpy()

                else:
                    out = model(x)
                    predict = softmax(out)
                    predict = predict.detach().cpu().numpy()
                    pred_accu = np.add(pred_accu, predict)
                    
            pred_accu = np.argmax(pred_accu, axis=1)

            label = label.detach().cpu().numpy()

            if is_first_mini_batch:
                iou_matrix = pred_accu
                label_matrix = label
                is_first_mini_batch = False
            else:
                iou_matrix = np.concatenate(
                    [iou_matrix, pred_accu], axis = 0
                )
                label_matrix = np.concatenate(
                    [label_matrix, label], axis = 0
                )

        valid_iou = mean_iou_score(iou_matrix, label_matrix)
        print(valid_iou)

if __name__ == '__main__':
    main()