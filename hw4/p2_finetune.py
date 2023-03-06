import random
import time
import datetime
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models

from p2.dataset import OfficeDataset
from p2.utils import get_office_tarnsform

# Set random seed for reproducibility
SEED = 5566
print("Random Seed:", SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def save_best(model, best, hist, model_save_path):
    if hist['eval_acc'][-1] >= best['eval_acc']:
            update_best_metric(hist, best)

            filename = 'ep_%d_acc_%.4f' % (best['ep'], best['eval_acc'])
            path = os.path.join(model_save_path, filename)
            torch.save(
                {
                    'model': model.state_dict(),
                }, path
            )

            best_path = os.path.join(model_save_path, 'best.ckpt')
            try:
                Path(best_path).unlink()
            except:
                pass
            Path(best_path).symlink_to(filename)

def plot_acc(hist, cache_dir):
    train_acc = hist['train_acc']
    eval_acc = hist['eval_acc']

    len_train = len(train_acc)
    len_eval = len(eval_acc)
    x_train = np.arange(1, len_train+1, 1)
    x_eval = np.arange(1, len_eval+1, 1)

    plt.plot(x_train, train_acc, label = 'train acc')
    plt.plot(x_eval, eval_acc, label = 'valid acc')
    plt.plot()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('loss plot')
    plt.legend()
    plt.savefig(
        os.path.join(cache_dir, 'acc plot')
    )
    plt.close()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw4_data/office",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./cache/p2_finetune/",
    )

    parser.add_argument(
        "--my_ssl_model_path",
        type=Path
        )
    parser.add_argument(
        "--ta_model_path",
        type=Path,
        default='./hw4_data/pretrain_model_SL.pt'
    )
    # other settings
    parser.add_argument("--use_model", type=str, choices=['TA', 'mySSL'], default='mySSL')
    parser.add_argument("--fix_backbone", type=bool, default=False)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-4)

    # data loader
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=200)

    args = parser.parse_args()
    return args

def train_epoch(train_dl, model, optim, loss_fn, hist):
    model.train()
    train_loss = 0
    correct_cnt = 0
    num_data = 0
    for img, label in tqdm(train_dl, leave=False, colour='green'):
        img, label = img.to(device), label.to(device)
        # (bs, C)
        preds = model(img)
        optim.zero_grad()
        loss = loss_fn(preds, label)
        loss.backward()
        optim.step()
        train_loss += loss.item()

        preds_label = np.argmax(preds.cpu().detach().numpy(), axis=1)
        correct_cnt += sum(preds_label==label.cpu().detach().numpy())
        num_data += img.size(0)

    train_acc = correct_cnt / num_data
    hist['train_loss'].append(train_loss)
    hist['train_acc'].append(train_acc)
    
def eval_epoch(eval_dl, model, loss_fn, hist):
    model.eval()
    eval_loss = 0
    correct_cnt = 0
    num_data = 0
    for img, label in tqdm(eval_dl, leave=False, colour='green'):
        img, label = img.to(device), label.to(device)
        # (bs, C)
        preds = model(img)
        loss = loss_fn(preds, label)
        eval_loss += loss.item()

        preds_label = np.argmax(preds.cpu().detach().numpy(), axis=1)
        correct_cnt += sum(preds_label==label.cpu().detach().numpy())
        num_data += img.size(0)

    # eval_loss = eval_loss / num_data
    eval_acc = correct_cnt / num_data
    hist['eval_loss'].append(eval_loss)
    hist['eval_acc'].append(eval_acc)

def update_best_metric(hist, best):
    best['ep'] = hist['ep'][-1]
    best['train_loss'] = hist['train_loss'][-1]
    best['eval_loss'] = hist['eval_loss'][-1]
    best['train_acc'] = hist['train_acc'][-1]
    best['eval_acc'] = hist['eval_acc'][-1]

def main(args):
    args_dict = vars(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")
    cache_dir = args.cache_dir / str(date) / str(run_id)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    train_dir, val_dir = args.data_dir / 'train', args.data_dir / 'val'
    train_df_path, val_df_path = args.data_dir / 'train.csv', args.data_dir / 'val.csv'

    # get transform (don't need to resize to 128 in transform?)
    transform_train = get_office_tarnsform(mode='train')
    transform_val = get_office_tarnsform(mode='valid')
    
    # load train dataset
    train_ds = OfficeDataset(train_dir, train_df_path, transform_train)
    val_ds = OfficeDataset(val_dir, val_df_path, transform_val)

    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # load unpretrained resnet
    model = models.resnet50(pretrained=False)
    model = model.to(device)

    # model
    if args.use_model == 'mySSL':
        state_dict = torch.load(args.my_ssl_model_path)
        model.load_state_dict(state_dict['model'])
        print("Finish loading my SSL model...")
    elif args.use_model == 'TA':
        state_dict = torch.load(args.ta_model_path)
        model.load_state_dict(state_dict)
        print("Finish loading TA's model...")

    # fixed backbone
    model.fc = nn.Linear(2048, 65)
    if args.fix_backbone:
        for p in model.parameters():
            p.requires_grad = False
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        model.eval()
        model.fc.train()
    
    model = model.to(device)
    
    # optimzer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_fn = nn.CrossEntropyLoss()

    hist = {
        'ep': [],
        'train_loss': [],
        'train_acc': [], 
        'eval_loss': [],
        'eval_acc': []
    }
    best = {
        'ep': 0,
        'train_loss': np.inf,
        'train_acc': 0, 
        'eval_loss': np.inf,
        'eval_acc': 0,
    }

    # training loop
    x = time.time()
    patience = 5
    no_update_cnt = 0
    for n_ep in range(args.num_epoch):
        hist['ep'].append(n_ep)
        train_epoch(train_dl, model, optimizer, loss_fn, hist)
        eval_epoch(val_dl, model, loss_fn, hist)

        y = time.time()
        time_elapsed = (y-x)/60
        plot_acc(hist, cache_dir)
        save_best(model, best, hist, cache_dir)
        with open(Path(cache_dir) / 'best.txt', 'w') as f:
            print(best, file=f)
            print(args_dict, file=f)
        f.close()

        print('[%d/%d] train loss: %.4f\ttrain acc: %.4f\tval loss: %.4f\tval acc: %.4f \t %.4f mins elapsed' 
            % (
                n_ep, args.num_epoch, 
                hist['train_loss'][-1], hist['train_acc'][-1],
                hist['eval_loss'][-1], hist['eval_acc'][-1], 
                time_elapsed
            )
        )
        
        # Early stopping
        if hist['eval_acc'][-1] < best['eval_acc']:
            no_update_cnt += 1
        else:
            no_update_cnt = 0

        if no_update_cnt > patience:
            print("Early stopping!")
            print("best result: \n", best)
            return

if __name__ == '__main__':
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    main(args)