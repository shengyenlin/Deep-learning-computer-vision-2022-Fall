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
from tqdm import tqdm

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import requests
import pandas as pd

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
import torchvision.utils as vutils

from p2.utils import precalculate, extract
from p2.dataset import P2Dataset, get_transform, get_reverse_transform
from p2.model import Unet

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

GEN_IMG_PER_CLASS = 100
NUM_CLASS = 10 

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw2_data/digits/mnistm/data/train",
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        help="Directory to the valid dataset.",
        default="./hw2_data/digits/mnistm/data/val",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/p2/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/p2/",
    )

    # model
    # TODO
    parser.add_argument("--timesteps", type=int, default=200) #T in paper
    parser.add_argument(
        "--beta_schedule", type=str, 
        help="linear, cosine, quadratic, sigmoid", 
        default='linear'
        )
    #parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    # 1. Resnet block / convNeXT block (use_convnext)
    # 2. positional embedding (with_time_emb)
    # 3. ulti-head self-attention (quadratic time) / linear attention variant (linear time)
    # 4. different variance schedule
    # 5. norm before / after attention in transformer
    # 6. different loss function
    # 7. When sampling use simplified version or clipping

    # data loader
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--loss_type", type=str, default='l1', help='l1, l2, huber')


    args = parser.parse_args()
    return args

@torch.no_grad()
def p_sample(model, x, label, t, t_index, precal):
    betas_t = extract(precal['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        precal['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = extract(precal['sqrt_recip_alphas'], t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, label) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(precal['posterior_variance'], t, x.shape)
        # Algorithm 2 line 4:
        # use fixed noise? 
        # return model_mean + torch.sqrt(posterior_variance_t) * precal['fixed_noise']
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(label, precal, model, args, shape) -> List[np.array]:
    device = next(model.parameters()).device

    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, args.timesteps)):
    # for i in tqdm(reversed(range(0, args.timesteps)), desc='sampling loop time step', total=args.timesteps):
        img = p_sample(model, img, label, torch.full((b,), i, device=device, dtype=torch.long), i, precal)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(label, precal, model, args, batch_size):
    # List[np.array]
    return p_sample_loop(
        label, precal, model, args, shape=(batch_size, args.channels, args.image_size, args.image_size)
        )

# forward diffusion
def q_sample(x_start, t, precal, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(precal['sqrt_alphas_cumprod'], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        precal['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(
    denoise_model, x_start, label, t, precal, noise=None, loss_type="l1"
    ):
    
    if noise is None:
        noise = torch.randn_like(x_start)

    # 0 -> t, forward diffusion
    x_noisy = q_sample(
        x_start, t, precal, noise=noise
        )
        
    # t -> 0, reverse process by learning the conditional data distribution
    predicted_noise = denoise_model(x_noisy, t, label)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss

def train(img, label, optimizer, model, precal, hist, args):
    optimizer.zero_grad()

    bs = img.shape[0]
    img = img.to(device)
    label = label.to(device)

    # Algorithm 1 line 3: sample t uniformally for every example in the batch
    t = torch.randint(0, args.timesteps, (bs,), device=device).long()

    loss = p_losses(model, img, label, t, precal, loss_type=args.loss_type)
    
    loss.backward()
    optimizer.step()

    hist['loss'].append(loss.item())

def get_pic_idx(i) -> str:
    if i // 10 == 0:
        return f'00{str(i)}'
    elif i // 10 == 10:
        return i
    else:
        return f'0{str(i)}'


def reverse_img(img):
    img = (img + 1) / 2
    img = img * 255.
    return img

def eval(precal, model, hist, img_save_dir, args):
    for i in range(NUM_CLASS):
        class_label = np.repeat(i, GEN_IMG_PER_CLASS)
        class_label_ts = torch.from_numpy(class_label)
        class_label_ts = class_label_ts.to(device)
        class_img_list = sample(class_label_ts, precal,model, args, GEN_IMG_PER_CLASS)

        #[-1]: last
        for j, img in enumerate(class_img_list[-1], start=1):
            img = (img + 1) * 0.5
            # img = reverse_img(img)
            # print(img)
            pic_idx = get_pic_idx(j)
            # plt.imsave(
            #     os.path.join(img_save_dir, f"{class_label[j]}_{pic_idx}.png"),
            #     img.reshape(args.image_size, args.image_size, args.channels)
            # )
            vutils.save_image(
                img,
                #normalize=True,
                fp=os.path.join(img_save_dir, f"{class_label[j-1]}_{pic_idx}.png"),
            )
    
    bash = f"python3 digit_classifier.py --folder {img_save_dir}"
    result = subprocess.check_output(bash, shell=True)
    result = str(result).split('/')
    n_all = int(result[-1][:4])
    result = result[-2].split('=')
    n_correct = int(result[-1][1:])
    acc = n_correct / n_all
    hist['acc'].append(acc)

def update_best_metric(best, hist):
    best['epoch'] = hist['epoch'][-1]
    best['iter'] = hist['iter'][-1]
    best['loss'] = hist['loss'][-1]
    best['acc'] = hist['acc'][-1]

def save_best(model, best, hist, cache_dir):
    if hist['acc'][-1] >= best['acc']:
        update_best_metric(best, hist)

        filename = 'ep_%d_iter_%d_acc_%.3f_.ckpt' % (
            best['epoch'], best['iter'], best['acc']
        )

        path = os.path.join(cache_dir, filename)
        torch.save(
            {
                'model': model.state_dict(),
                'acc': best['acc']
            }, path
        )
        best_path = os.path.join(cache_dir, 'best.ckpt')
        try:
            Path(best_path).unlink()
        except:
            pass
        Path(best_path).symlink_to(filename)

def main(args):
    run_id = int(time.time())
    args_dict = vars(args)
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")

    cache_dir = args.cache_dir / str(date) / str(run_id)
    img_save_dir = os.path.join(cache_dir, 'pics')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.mkdir(img_save_dir)

    betas, alphas_cumprod, alphas_cumprod_prev, \
        sqrt_recip_alphas, sqrt_alphas_cumprod, \
        sqrt_one_minus_alphas_cumprod, posterior_variance = \
        precalculate(args.beta_schedule, args.timesteps) 

    # fixed_noise = torch.randn(
    #     (GEN_IMG_PER_CLASS, args.channels, args.image_size, args.image_size)
    #     )
    # fixed_noise = fixed_noise.to(device)
    precal = {
       'betas': betas,
       'alphas_cumprod': alphas_cumprod,
       'alphas_cumprod_prev': alphas_cumprod_prev,
       'sqrt_recip_alphas': sqrt_recip_alphas,
       'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
       'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
       'posterior_variance': posterior_variance,
       # 'fixed_noise': fixed_noise
    }

    train_df = pd.read_csv('./hw2_data/digits/mnistm/train.csv')
    val_df = pd.read_csv('./hw2_data/digits/mnistm/val.csv')
    
    transform = get_transform(args)
    train_dataset = P2Dataset(args.train_dir, train_df['label'].tolist(), transform)
    val_dataset = P2Dataset(args.val_dir, val_df['label'].tolist(), transform)

    train_dl = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # What is val_dl used for??
    val_dl = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    # We assume that image data consists of integers in {0,1,...,255} scaled linearly to [−1, 1]. 
    # This ensures that the neural network reverse process operates on consistently scaled inputs starting from the standard normal prior p(\mathbf{x}_T )p(x_{T}).
    model = Unet(
            dim=args.image_size,
            channels=args.channels,
            dim_mults=(1, 2, 4,), #or dim_mults=(1, 2, 4, 8)?
        )
        
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    hist = {
        'epoch': [],
        'iter': [],
        'loss': [],
        'acc': []
    }

    best = {
        'epoch': 0,
        'iter': 0,
        'loss': np.inf,
        'acc': 0
    }

    x = time.time()
    for epoch in range(args.num_epoch):
        for i, (img, label) in enumerate(train_dl):
            hist['epoch'].append(epoch)
            hist['iter'].append(i)
            train(img, label, optimizer, model, precal, hist, args)
            
            y = time.time()
            if i % 100 == 0:
                eval(precal, model, hist, img_save_dir, args)
                time_passed = (y-x) / 60
                print(
                    '[%d/%d][%d/%d]\tloss: %.4f\tacc:%.4f\t%.2f mins elpased' 
                    % (
                        hist['epoch'][-1], args.num_epoch, i, len(train_dl),
                        hist['loss'][-1], hist['acc'][-1], time_passed
                    )
                )
                save_best(model, best, hist, cache_dir)
                with open(Path(cache_dir) / 'best.txt', 'w') as f:
                    print(best, file=f)
                    print(args_dict, file=f)
                f.close()           
    

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)