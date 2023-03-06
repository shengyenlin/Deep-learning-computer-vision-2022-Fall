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
from sklearn.model_selection import KFold

#customized
from preprocess import TransformsModelB
from utils import  Args, train, valid, save_checkpoint, do_avg_on_list

IMG_FILE = 'hw1_data/p1_data'
TRAIN_PATH = 'hw1_data/p1_data/train'
VAL_PATH = 'hw1_data/p1_data/val'
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
print(f"Training on {device}.")


def main():
    #utils class import
    args = Args()
    data_transforms = TransformsModelB()
    
    #dataset
    train_dataset = datasets.ImageFolder(
        root=TRAIN_PATH, transform=data_transforms.transform_P1_modelB['train']
        )
    
    #use as test dataset
    val_dataset = datasets.ImageFolder(
        root=VAL_PATH,transform=data_transforms.transform_P1_modelB['val']
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )    

    #Do k-fold CV
    kfold = KFold(n_splits=args.k_folds, shuffle=True)
    history = {
        'fold_id': [],
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'best_epoch': []
    }
    for fold, (train_ids, subtrain_ids) in enumerate(kfold.split(train_dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        subtrain_subsampler = SubsetRandomSampler(subtrain_ids)

        #dataloder
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            num_workers=args.num_workers,
            sampler=train_subsampler
        )
        subtrain_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            num_workers=args.num_workers,
            sampler=subtrain_subsampler
        )    
   
        #Backbone model
        model = models.resnet152(
            weights='IMAGENET1K_V1'
        )
        
        #Head
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASS)

    
        #setup optimizer and scheduler
        #optim_name = 'Adam'
        optimizer = args.setup_optimizer(
            model, args.optim_name, args.lr, 
            args.momentum, args.weight_decay
            )

        loss_fn = nn.CrossEntropyLoss()

        #start training
        acc_record = []
        best_train_acc = 0
        best_train_loss = 0
        best_val_acc = 0
        best_val_loss = 999999
        best_epoch = 0
        best_model = None
        no_update_cnt = 0
        
        print(f'Start training..., FOLD {fold}')
        print('--------------------------------')
        x = time.time()
        model = model.to(device)
        for epoch_idx in range(args.num_epoch):
            print(f"Epoch {epoch_idx}:")
            train_acc, train_loss = train(train_loader, model, loss_fn, optimizer, device)
            val_acc, val_loss = valid(subtrain_loader, model, loss_fn, device)
            acc_record.append(val_acc)
            if val_acc > best_val_acc:
                # model_name = f"models/0930_modelB_lr_{lr}"
                # best_model = copy.deepcopy(model)
                best_train_acc = train_acc
                best_train_loss = train_loss
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
        # save_checkpoint(
        #     best_model, optimizer, round(valid_acc, 2), prefix=model_name
        # )
        history['fold_id'].append(fold)
        history['train_acc'].append(best_train_acc)
        history['train_loss'].append(best_train_loss)
        history['val_acc'].append(best_val_acc)
        history['val_loss'].append(best_val_loss)
        history['best_epoch'].append(best_epoch)

        print('########################################################')
        print("Finish model tuning")
        print("Params:")
        print(f"Best epoch is {best_epoch}, Accuracy: {best_val_acc}, Loss: {best_val_loss}")
        print('########################################################')   

    avg_train_acc = do_avg_on_list(history['train_acc'])
    avg_train_loss = do_avg_on_list(history['train_loss'])
    avg_val_acc = do_avg_on_list(history['val_acc'])
    avg_val_loss = do_avg_on_list(history['val_loss'])
    print(f"{args.k_folds}-Fold CV result:")
    print(
        f"avg train acc: {avg_train_acc}, avg train loss: {avg_train_loss}  \
            avg val acc: {avg_val_acc}, avg val loss: {avg_val_loss}"
        )
    print(history)
    # test_acc, test_loss = valid(val_loader, model, loss_fn, device)
    # print(f"Resuls: test acc: {test_acc}, test loss: {test_loss}")

if __name__ == '__main__':
    main()