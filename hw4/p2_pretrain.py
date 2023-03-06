import random
import time
import datetime
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from byol_pytorch import BYOL
from torchvision import models

from p2.dataset import MiniDataset
from p2.utils import get_mini_transform

# Set random seed for reproducibility
SEED = 5566
print("Random Seed:", SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def plot_loss(hist, cache_dir):
    train_loss = hist['loss']
    len_train = len(train_loss)
    x_train = np.arange(1, len_train+1, 1)

    plt.plot(x_train, train_loss, label = 'train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss plot')
    plt.legend()
    plt.savefig(
        os.path.join(cache_dir, 'loss plot')
    )
    plt.close()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw4_data/mini/train",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./cache/p2_pretrain/",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=200)

    parser.add_argument("--save_all", action='store_true')

    args = parser.parse_args()
    return args

def update_best_metric(hist, best):
    best['ep'] = hist['ep'][-1]
    best['loss'] = hist['loss'][-1]


def save_best(model, best, hist, model_save_path, save_all=True):
    if hist['loss'][-1] <= best['loss']:
        update_best_metric(hist, best)
        filename = 'ep_%d_loss_%.4f.ckpt' % (best['ep'], best['loss'])
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

    else:
        if save_all:
            filename = 'ep_%d_loss_%.4f' % (hist['ep'][-1], hist['loss'][-1])
            path = os.path.join(model_save_path, filename)          
            torch.save(
                {
                    'model': model.state_dict(),
                }, path
            )

def train_iter(img ,learner, optim):
    img = img.to(device)
    loss = learner(img)
    optim.zero_grad()
    loss.backward()
    optim.step()
    learner.update_moving_average()
    return loss

def update_best(hist, best):
    if hist['loss'][-1] < best['loss']:
        best['ep'] = hist['ep'][-1]
        best['loss'] = hist['loss'][-1]

def main(args):
    args_dict = vars(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")
    cache_dir = args.cache_dir / str(date) / str(run_id)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # get transform (don't need to resize to 128 in transform?)
    transform = get_mini_transform()
    
    # load train dataset
    train_ds = MiniDataset(args.train_dir, transform)
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # load unpretrained resnet
    model = models.resnet50(pretrained=False)
    model = model.to(device)

    # boyl trainer
    learner = BYOL(
        model,
        image_size = 128,
        hidden_layer = 'avgpool',
    )
    learner = learner.to(device)
    
    # optimzer
    optimizer = torch.optim.Adam(learner.parameters(), args.lr)

    hist = {
        'ep': [],
        'loss': []
    }
    best = {
        'ep': 0,
        'loss': np.inf
    }

    print(args.save_all)

    # training loop
    x = time.time()
    for n_ep in range(1, args.num_epoch+1):
        loss_ep = 0
        for img in tqdm(train_dl, leave=False, colour='green'):
            loss_iter = train_iter(img, learner, optimizer)
            loss_ep += loss_iter.detach().item() 
        
        hist['loss'].append(loss_ep), hist['ep'].append(n_ep)
        loss_ep = round(loss_ep, 3)
        y = time.time()
        time_elapsed = (y-x)/60
        plot_loss(hist, cache_dir)
        save_best(model, best, hist, cache_dir, args.save_all)
        print('[%d/%d]\tLoss: %.4f\t%.4f mins elapsed' 
            % (n_ep, args.num_epoch, loss_ep, time_elapsed)
            )

        with open(Path(cache_dir) / 'best.txt', 'w') as f:
            print(best, file=f)
            print(args_dict, file=f)
        f.close()

if __name__ == '__main__':
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    main(args)