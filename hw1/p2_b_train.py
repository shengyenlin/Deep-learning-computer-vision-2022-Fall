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

from PIL import Image


from utils import train_P2, valid_P2, save_checkpoint, FocalLoss
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

    train_dataset = P2Dataset(
        TRAIN_PATH, transform=customized_transform.transform['train']
        )
    val_dataset = P2Dataset(
        VAL_PATH,  transform=customized_transform.transform['val']
        )

    batch_size = 18
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    ) 

    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = FocalLoss(NUM_CLASS)
    loss_fn = loss_fn.to(device)
    model_name = 'VGG16_FCN8'
    print(f"Training {model_name}...")
    # summary(model, (3,512,512), device=device)
    lrs = [0.0005, 0.0001, 0.00005]
    weight_decays = [0, 0.01, 0.001, 0.0001]
    for lr in lrs:
        for weight_decay in weight_decays:
            if weight_decay == 0:
                continue
            print(f"lr = {lr}, weight_decay = {weight_decay}")
            #####Change model#####
            model = get_p2_model(model_name, NUM_CLASS)
            model = model.to(device)
            #####Change model#####
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
                )   
            #input: [0.5, 0.2, 0.6] 
            #target: [1, 0, 1]

            num_epoch = 200

            iou_record = []
            best_valid_iou = 0
            best_valid_loss = 999999
            best_epoch = 0
            no_update_cnt = 0
            best_model = None
            patience = 5

            x = time.time()
            for epoch in range(num_epoch):
                train_loss = train_P2(train_loader, model, loss_fn, optimizer, device)
                valid_iou, valid_loss = valid_P2(val_loader, model, loss_fn, device)
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
                print(f"Epoch {epoch+1}: {round(time_train, 2)} min elapsed, train loss: {round(train_loss, 3)}, valid iou: {round(valid_iou * 100, 2)}%, valid loss: {round(valid_loss, 3)}")
                #Early stop
                if no_update_cnt > patience:
                    break
            best_valid_iou = round(best_valid_iou, 2)
            model_name = f"models/p2/1005_{model_name}_{lr}_weight_decay_{weight_decay}_iou_{best_valid_iou}"
            save_checkpoint(
                best_model, optimizer, 
                round(best_valid_iou, 2), prefix=model_name
            )  
            print('########################################################')
            print("Finish model tuning")
            print(f"Best epoch is {best_epoch}, Iou: {best_valid_iou}, Loss: {best_valid_loss}")
            print('########################################################')
            del model
            del best_model
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()