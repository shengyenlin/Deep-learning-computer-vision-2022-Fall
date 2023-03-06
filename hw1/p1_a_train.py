import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from PIL import Image

from collections import Counter
import gc
import time

from preprocess import ValDataset, TrainDataset, TransformsModelB
from utils import Args, train, valid, save_checkpoint

IMG_FILE = 'hw1_data/p1_data'
TRAIN_PATH = 'hw1_data/p1_data/train_50'
VAL_PATH = 'hw1_data/p1_data/val_50'
OUT_PATH = 'hw1_data/val_gt.csv'
NUM_CLASS = 50

#Reproducibility
SEED = 5566
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

def get_data_and_label(img_files, mode):
    imgs, labels = [], []
    for img_name in img_files:
        file_name = img_name.split("_")
        if mode == 'train':
            img = TRAIN_PATH + '/' + img_name
        elif mode == 'val':
            img = VAL_PATH + '/' + img_name
        label = file_name[0]
        imgs.append(img)
        labels.append(int(label))
    return imgs, labels

def main():
    #utils class import
    args = Args()
    data_transforms = Transforms()

    #import data
    train_imgs = os.listdir(TRAIN_PATH)
    train_imgs, train_labels = get_data_and_label(train_imgs, mode='train')
    val_imgs = os.listdir(VAL_PATH)
    val_imgs, val_labels = get_data_and_label(val_imgs, mode='val')
    
    #dataset
    train_dataset = TrainDataset(
        train_imgs, train_labels, data_transforms.transform_P1_modelB
    )
    val_dataset = ValDataset(
        val_imgs, val_labels, data_transforms.transform_P1_modelB
        )

    #dataloder
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )    
   
    #Backbone model
    model = models.resnet152(
        weights = 'ResNet152_Weights.IMAGENET1K_V2'
    )

    #Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASS)

    #setup optimizer and scheduler
    optimizer = args.setup_optimizer(model)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    #start training
    acc_record = []
    best_valid_acc = 0
    best_valid_loss = 999999
    best_epoch = 0
    no_update_cnt = 0
    
    x = time.time()
    for epoch in range(args.num_epoch):
        train_acc, train_loss = train(train_loader, model, loss_fn, optimizer, device)
        valid_acc, valid_loss = valid(valid_loader, model, loss_fn, device)
        acc_record.append(valid_acc)
        if valid_acc > best_valid_acc:
            model_name = "models/0929_modelB"
            save_checkpoint(
                model, round(valid_acc, 2), prefix=model_name
                )
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            best_epoch = epoch
            no_update_cnt = 0
        else:
            no_update_cnt += 1
        y = time.time()
        time_train = (y - x) / 60
        print(f"Epoch {epoch+1}: {round(time_train, 2)} min elapsed, train acc: {round(train_acc * 100, 2)}%, train loss: {round(train_loss, 3)}, valid acc: {round(valid_acc * 100, 2)}%, valid loss: {round(valid_loss, 3)}")
        gc.collect()
        torch.cuda.empty_cache()
        #Early stop
        if no_update_cnt > args.patience:
            break

    print('########################################################')
    print("Finish model tuning")
    print(f"Best epoch is {best_epoch}, Accuracy: {best_valid_acc}, Loss: {best_valid_loss}")
    print('########################################################')
