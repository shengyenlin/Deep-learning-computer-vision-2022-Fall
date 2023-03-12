import math
from inspect import isfunction
from functools import partial
from os import times
import os
import subprocess
from pathlib import Path
from argparse import ArgumentParser, Namespace
import random
import time 
import datetime
from typing import List

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.optim.lr_scheduler import ReduceLROnPlateau

from p3.model import DANN
from p3.dataset import P3Dataset, get_transform

# Set random seed for reproducibility
SEED = 5566
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed:", SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--source_domain",
        type = Path,
        help = "mnistm, svhn, usps",
        default = "mnistm"
    )
    parser.add_argument(
        "--target_domain",
        type = Path,
        help = "mnistm, svhn, usps",
        default = "mnistm"
    )

    parser.add_argument(
        "--data_dir",
        type = Path,
        help = "Directroy in which data stores",
        default = "./hw2_data/digits"
    )

    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/p3/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/p3/",
    )

    # model
    parser.add_argument("--use_dann", type=int, default=0)
    parser.add_argument("--train_on_source", type=int, default=0)
    parser.add_argument("--train_on_target", type=int, default=0)
    parser.add_argument("--use_scheduler", type=int, default=0)

    # data loader
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--optim", type=str, default = 'Adam')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_epoch", type=int, default=100)


    args = parser.parse_args()
    return args

def train_dann_iter(
    model, optimizer,
    sr_iter, tg_iter, lam, 
    loss_fn_class, loss_fn_domain,
    hist
    ):

    # training model using source data
    # domain label: 0
    # get class loss and 
    x_sr = sr_iter.next()
    sr_img, sr_label = x_sr
    model.zero_grad()
    bs = len(sr_img)

    sr_img, sr_label = sr_img.to(device), sr_label.to(device)
    domain_label = torch.zeros(bs).long().to(device)

    class_out, domain_out = model(sr_img, lam)
    err_sr_class = loss_fn_class(class_out, sr_label)
    err_sr_domain = loss_fn_domain(domain_out, domain_label)

    # training model using source data, domain label: 1
    x_tg = tg_iter.next()
    tg_img, _ = x_tg
    bs = len(tg_img)

    tg_img = tg_img.to(device)
    domain_label = torch.ones(bs).long().to(device)

    _, domain_out = model(tg_img, lam)
    err_tg_domain = loss_fn_domain(domain_out, domain_label)

    # calculate loss
    err = err_tg_domain + err_sr_domain + err_sr_class
    err.backward()
    optimizer.step()
    
    hist['train_loss'].append(err)
    hist['err_tg_domain'].append(err_tg_domain)
    hist['err_sr_domain'].append(err_sr_domain)
    hist['err_sr_class'].append(err_sr_class)

def eval_dann(model, val_tg_dl, hist, scheduler = None):
    model.eval()
    lam = 0 #we don't need to compute domain clf in eval
    n_correct = 0
    n_data = len(val_tg_dl.dataset)
    with torch.no_grad():
        for img, label in val_tg_dl:
            img = img.to(device)
            label = label.to(device)
            class_out, _ = model(img, lam)
            pred_class = torch.max(class_out, axis=1)[1]
            n_correct += torch.sum(pred_class==label)

    acc = n_correct / n_data
    hist['val_acc'].append(acc)

    if scheduler is not None:
        scheduler.step(acc)

def update_best_metric(best, hist):
    best['epoch'] = hist['epoch'][-1]
    best['iter'] = hist['iter'][-1]
    best['train_loss'] = hist['train_loss'][-1]
    # best['val_loss'] = hist['val_loss'][-1]
    best['val_acc'] = hist['val_acc'][-1]
    best['err_tg_domain'] = hist['err_tg_domain'][-1]
    best['err_sr_domain'] = hist['err_sr_domain'][-1]
    best['err_sr_class'] = hist['err_sr_class'][-1]


def save_best(model, best, hist, cache_dir):
    if hist['val_acc'][-1] >= best['val_acc']:
        update_best_metric(best, hist)
        filename = 'ep_%d_iter_%d_acc_%.3f_.ckpt' % (
            best['epoch'], best['iter'], best['val_acc']
        )

        path = os.path.join(cache_dir, filename)
        torch.save(
            {
                'model': model.state_dict(),
                'acc': best['val_acc']
            }, path
        )
        best_path = os.path.join(cache_dir, 'best.ckpt')
        try:
            Path(best_path).unlink()
        except:
            pass
        Path(best_path).symlink_to(filename)

def train_normal(img, label, optimizer, model, loss_fn, hist):
    optimizer.zero_grad()
    img, label = img.to(device), label.to(device)
        
    out = model(img)
    loss = loss_fn(out, label)

    loss.backward()
    optimizer.step()

    hist['train_loss'].append(loss)

def eval_normal(val_dl, model, hist):
    model.eval()
    n_correct = 0
    n_data = len(val_dl.dataset)
    with torch.no_grad():
        for img, label in val_dl:
            img = img.to(device)
            label = label.to(device)
            class_out = model(img)
            pred_class = torch.max(class_out, axis=1)[1]
            n_correct += torch.sum(pred_class==label)

    acc = n_correct / n_data
    hist['val_acc'].append(acc)

def save_best_to_txt(cache_dir, best, args_dict):
    with open(Path(cache_dir) / 'best.txt', 'w') as f:
        print(best, file=f)
        print(args_dict, file=f)
    f.close()    

def adjust_learning_rate(optimizer, p):
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr
    return lr

def main(args):
    run_id = int(time.time())
    args_dict = vars(args)
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")

    sr_dir = args.data_dir / args.source_domain
    tg_dir = args.data_dir / args.target_domain
    train_sr_dir = args.data_dir / args.source_domain / 'data' / 'train'
    val_tg_dir = args.data_dir / args.target_domain / 'data' / 'val'
    cache_dir = args.cache_dir / Path(str(args.source_domain) + '_' + str(args.target_domain)) / str(date) /str(run_id)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if args.use_dann:
        train_tg_dir = args.data_dir / args.target_domain / 'data' / 'train'

    # read label, img of source domain and target domain
    train_sr_df = pd.read_csv(
        os.path.join(sr_dir, 'train.csv')
    ) 
    val_tg_df = pd.read_csv(
        os.path.join(tg_dir, 'val.csv')
    )

    # data augmentation
    transform = get_transform(args)

    # dataset and dataloader
    train_sr_ds = P3Dataset(train_sr_dir, train_sr_df['label'].tolist(), transform)
    val_tg_ds = P3Dataset(val_tg_dir, val_tg_df['label'].tolist(), transform)

    train_sr_dl = torch.utils.data.DataLoader(
        train_sr_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_tg_dl = torch.utils.data.DataLoader(
        val_tg_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    if args.use_dann or args.train_on_target:
        train_tg_df = pd.read_csv(
            os.path.join(tg_dir, 'train.csv')
        )
        train_tg_ds = P3Dataset(train_tg_dir, train_tg_df['label'].tolist(), transform)
        train_tg_dl = torch.utils.data.DataLoader(
            train_tg_ds, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )       
    
    # model
    model = DANN(args.use_dann)
    model = model.to(device) 

    # optimizer
    if args.optim == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'max', eps=1e-7, verbose=True)
    else:
        scheduler = None

    loss_fn_class = torch.nn.NLLLoss()
    # loss_fn_domain = torch.nn.BCELoss()
    # loss_fn_class = nn.CrossEntropyLoss()
    loss_fn_domain = nn.CrossEntropyLoss()

    hist = {
        'epoch': [],
        'iter': [],
        'train_loss': [],
        # 'train_sr_acc': [], it's meaning less to calculate per iteration acc
        # 'val_loss': [],
        'val_acc': [],
        'err_tg_domain': [],
        'err_sr_domain': [],
        'err_sr_class': []
    }

    best = {
        'epoch': 0,
        'iter': 0,
        'train_loss': np.inf,
        # 'val_loss': np.inf,
        'val_acc': 0,
        'err_tg_domain': 0,
        'err_sr_domain': 0,
        'err_sr_class': 0
    }

    # train and eval
    x = time.time()
    i = 0
    for epoch in range(args.num_epoch):
        model.train()
        if args.use_dann:
            len_dl = min(len(train_sr_dl), len(train_tg_dl))
            sr_iter = iter(train_sr_dl) #return iterator
            tg_iter = iter(train_tg_dl)

            j = 0
            while j < len_dl:
                hist['epoch'].append(epoch)
                hist['iter'].append(j)
                # TODO: change to boshun's p
                p = float(i + epoch * len_dl) / args.num_epoch / len_dl
                lam = 2. / (1. + np.exp(-10 * p)) - 1    

                lr = adjust_learning_rate(optimizer, p)

                train_dann_iter(model, optimizer, sr_iter, tg_iter, lam, loss_fn_class, loss_fn_domain, hist)
                i+=1
                j+=1
                #print(i, j)
                if i % 50 == 0:
                    eval_dann(model, val_tg_dl, hist, scheduler)
                    save_best(model, best, hist, cache_dir)

                    y = time.time()
                    time_passed = (y-x) / 60
                    print(
                        '[%d/%d][%d/%d] lam: %.4f\ttrain_loss: %.4f (tg_d: %.4f, sr_d: %.4f, sr_cl: %.4f) val_acc:%.4f\t%.2f mins elpased' 
                        % (
                            hist['epoch'][-1], args.num_epoch, j, len_dl, lam,
                            hist['train_loss'][-1], hist['err_tg_domain'][-1], hist['err_sr_domain'][-1], hist['err_sr_class'][-1], 
                            hist['val_acc'][-1], time_passed
                        )
                    )
                    save_best_to_txt(cache_dir, best, args_dict)

        elif args.train_on_source:
            hist['epoch'].append(epoch)
            for img, label in train_sr_dl:
                train_normal(img, label, optimizer, model, loss_fn_class, hist)
            eval_normal(val_tg_dir, model, hist)
            save_best(model, best, hist, cache_dir)
            save_best_to_txt(cache_dir, best, args_dict)
            y = time.time()
            time_passed = (y-x) / 60
            print(
                '[%d/%d]\ttrain_loss: %.4f\tval_acc:%.4f\t%.2f mins elpased'
                % (
                    hist['epoch'][-1], args.num_epoch, time_passed
                )
            )

        elif args.train_on_target:
            hist['epoch'].append(epoch)
            for img, label in train_tg_dl:
                train_normal(img, label, optimizer, model, loss_fn_class, hist)
            eval_normal(val_tg_dir, model, hist)
            save_best(model, best, hist, cache_dir)
            save_best_to_txt(cache_dir, best, args_dict)
            print(
                '[%d/%d]\ttrain_loss: %.4f\tval_acc:%.4f\t%.2f mins elpased'
                % (
                    hist['epoch'][-1], args.num_epoch, time_passed
                )
            )


if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)