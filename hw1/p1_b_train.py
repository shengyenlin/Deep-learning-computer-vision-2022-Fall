#sys
import random
import os
import copy
import sys
from collections import Counter
import gc
import time

#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate

#other
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#customized
from preprocess import TransformsModelB
from utils import P1Args, train, valid, save_checkpoint, do_avg_on_list

TRAIN_PATH = 'hw1_data/p1_data/train'
VAL_PATH = 'hw1_data/p1_data/val'
OUT_PATH = 'hw1_data/val_gt.csv'
NUM_CLASS = 50
MODEL_PATH = 'models/'
IMG_PATH = 'train_log_pics/'

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
print(f"Training on {device}.")


def main():
    #utils class import
    args = P1Args()
    data_transforms = TransformsModelB()
    
    #dataset
    train_dataset = datasets.ImageFolder(
        root=TRAIN_PATH, transform=data_transforms.transform_P1_modelB['train']
        )
    
    val_dataset = datasets.ImageFolder(
        root=VAL_PATH,transform=data_transforms.transform_P1_modelB['val']
        )
    

    #dataloder
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )      
   
    #Backbone model
    #Resnet
    model = models.resnet152(
        weights=models.ResNet152_Weights.DEFAULT
    )
    
    #Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASS)

    #Efficiency net
    # model = models.efficientnet_b3(weights='DEFAULT')
    # print(model)


    #setup optimizer and scheduler
    #optim_name = 'Adam'
    # optimizer = args.setup_optimizer(
    #     model, args.optim_name, args.lr, 
    #     args.momentum, args.weight_decay
    #     )
    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9
        )
    loss_fn = nn.CrossEntropyLoss() #perform softmax + NLL loss

    #start training
    train_acc_record = []
    val_acc_record = []
    best_val_acc = 0
    best_val_loss = 999999
    best_epoch = 0
    best_model = None
    no_update_cnt = 0
    x = time.time()
    model = model.to(device)
    
    for epoch_idx in range(args.num_epoch):
        print(f"Epoch {epoch_idx}:")
        train_acc, train_loss = train(train_loader, model, loss_fn, optimizer, device)
        val_acc, val_loss = valid(val_loader, model, loss_fn, device)
       
        train_acc_record.append(train_acc)
        val_acc_record.append(val_acc)

        if val_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch_idx
            no_update_cnt = 0

        else:
            no_update_cnt += 1
        y = time.time()

        time_train = (y - x) / 60
        sys.stdout.write("\033[K") #Clear line
        sys.stdout.write("\033[K") #Clear line
        print(f"Epoch {epoch_idx}: {round(time_train, 2)} min elapsed, train acc: {round(train_acc * 100, 2)}%, train loss: {round(train_loss, 3)}, valid acc: {round(val_acc * 100, 2)}%, valid loss: {round(val_loss, 3)}")
        gc.collect()
        torch.cuda.empty_cache()
        #Early stop
        if no_update_cnt > args.patience:
            break

    #save model
    model_name = MODEL_PATH + args.model_name 
    save_checkpoint(
        best_model, optimizer, round(best_val_acc, 2), prefix=model_name
    )

    #save training plots
    plt_name = args.model_name + ' val acc: ' + str(round(best_val_acc, 2))
    x = [i for i in range(1, best_epoch+1)]
    plt.plot(x, train_acc_record, label = 'train acc')
    plt.plot(x, val_acc_record, label = 'valid acc')
    plt.legend()
    plt.title(plt_name)
    plt.savefig(IMG_PATH + args.model_name)


    print('########################################################')
    print("Finish model tuning")
    print("Params:")
    print(f"Best epoch is {best_epoch}, Accuracy: {best_val_acc}, Loss: {best_val_loss}")
    print('########################################################')   

if __name__ == '__main__':
    main()