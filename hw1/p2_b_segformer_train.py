import os
import gc
import time
import random
from tqdm import tqdm

import numpy as np

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

SEED = 5566
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
#device = 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
print(f"Train on {device}.")

IMG_FILE = 'hw1_data/p2_data'
TRAIN_PATH = 'hw1_data/p2_data/train'
VAL_PATH = 'hw1_data/p2_data/validation'
TRAIN_NUM_PIC = 2000
VAL_NUM_PIC = 257
NUM_CLASS = 7


def main():
    customized_transform = P2_TransformsModelA()
    seg_type = 'b4'
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        f"nvidia/segformer-{seg_type}-finetuned-ade-512-512",
        num_labels = NUM_CLASS,
        ignore_mismatched_sizes=True, 
    )
    #make sure no loss is computed for the background class
    #feature_extractor.reduce_labels = True
    #the size of images that SegFormer receives as training input
    # feature_extractor.size = 512

    normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
    train_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.ToTensor()
        ])
    val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_dataset = P2Dataset(
        TRAIN_PATH, isSegFormer=True, 
        feature_extractor=feature_extractor, transform_seg=train_transform
        )
    val_dataset = P2Dataset(
        VAL_PATH, isSegFormer=True, 
        feature_extractor=feature_extractor, transform_seg=val_transform
        )

    loss_fn = nn.CrossEntropyLoss()
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True,drop_last=False
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    ) 

    model_name = 'segformer'

    print(f"Training {model_name}_{seg_type}...")
    # summary(model, (3,512,512), device=device)
    lrs = [1e-04]
    weight_decays = [0]
    for lr in lrs:
        for weight_decay in weight_decays:
            print(f"lr = {lr}, weight_decay = {weight_decay}")
            #####Change model#####
            model_name = 'segformer'
            model = get_p2_model(
                model_name, NUM_CLASS,
                seg_type = seg_type
                )
            model = model.to(device)
            #####Change model#####
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
                )   
            #input: [0.5, 0.2, 0.6] 
            #target: [1, 0, 1]

            num_epoch = 100

            iou_record = []
            best_valid_iou = 0
            best_valid_loss = 999999
            best_epoch = 0
            no_update_cnt = 0
            best_model = None
            patience = 5

            x = time.time()
            for epoch in range(num_epoch):
                train_loss = train_P2_segform(train_loader, model, loss_fn, optimizer, device)
                valid_iou, valid_loss = valid_P2_segform(val_loader, model, loss_fn, device)
                iou_record.append(valid_iou)

                if valid_iou > best_valid_iou:
                    best_valid_loss = valid_loss
                    best_valid_iou = valid_iou
                    best_epoch = epoch
                    no_update_cnt = 0
                    best_model = model

                else:
                    no_update_cnt += 1
                y = time.time()
                time_train = (y - x) / 60
                print(f"Epoch {epoch}: {round(time_train, 2)} min elapsed, train loss: {round(train_loss, 3)}, valid iou: {round(valid_iou * 100, 2)}%, valid loss: {round(valid_loss, 3)}")
                #Early stop
                if no_update_cnt > patience:
                    break

            best_valid_iou = round(best_valid_iou, 2)
            model_name = f"models/p2/segb4_final"
            save_checkpoint(
                best_model, optimizer, 
                round(best_valid_iou, 2), prefix=model_name
            )  
            print('########################################################')
            print("Finish model tuning")
            print(f"Best epoch is {best_epoch}, iou: {best_valid_iou}, Loss: {best_valid_loss}")
            print('########################################################')
            del model
            del best_model


if __name__ == '__main__':
    main()