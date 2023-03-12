from __future__ import print_function
import os
import random
import time
import subprocess
import datetime
import gc

#%matplotlib inline
from argparse import ArgumentParser, Namespace
from pathlib import Path
import matplotlib.pyplot as plt

#torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np

from p1.model import DCGAN_Generator, DCGAN_Discriminator, weights_init
from p1.dataset import P1Dataset, transform
from p1.face_recog import face_recog

# Set random seed for reproducibility
SEED = 5566
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed:", SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
NUM_SELECTED = 1000

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
real_label = 1
fake_label = 0


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw2_data/face/train",
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        help="Directory to the valid dataset.",
        default="./hw2_data/face/val",
    )
    parser.add_argument(
        "--img_save_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/p1/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/p1/",
    )

    # data
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--nc", type=int, default=3) #number of input picture channel
    parser.add_argument("--nz", type=int, default=100) #size of z latent vector 

    # model
    parser.add_argument("--ngf", type=int, default=64) # Size of feature maps in generator
    parser.add_argument("--ndf", type=int, default=64) # Size of feature maps in discriminator

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--loss_fn", type=str, default='BCE')
    parser.add_argument("--num_pics_generated", type=int, default=1000)
    parser.add_argument("--add_ms_loss", type=int, default=0)
    parser.add_argument("--noise", type=str, default='StandardNorm')

    args = parser.parse_args()
    return args

def update_best_metric(hist, best):
    best['epoch'] = hist['epoch'][-1]
    best['iter'] = hist['iter'][-1]
    best['D(x)'] = hist['D(x)'][-1]
    best['D(G(z1))'] = hist['D(G(z1))'][-1]
    best['D(G(z2))'] = hist['D(G(z2))'][-1]
    best['G_losses'] = hist['G_losses'][-1]
    best['D_losses'] = hist['D_losses'][-1]
    best['face_reg'] = hist['face_reg'][-1]
    best['fid'] = hist['fid'][-1]

def save_best(netD, netG, best, hist, model_save_path):
    #save fid, face_reg ,each iter model (symlink)

    if hist['fid'][-1] <= best['fid'] or \
        hist['face_reg'][-1] >= best['face_reg'] or \
        (hist['face_reg'][-1] >= 90 and hist['face_reg'][-1] <= 27):
            update_best_metric(hist, best)

            #store best epoch
            filename = 'ep_%d_iter_%d_fid_%.4f_face_reg_%.4f.ckpt' % (
                best['epoch'], best['iter'],
                best['fid'], best['face_reg'] 
            )
            path = os.path.join(model_save_path, filename)
            torch.save(
                {
                    'netD': netD.state_dict(),
                    'netG': netG.state_dict(),
                    'fid': best['fid'],
                    'face_reg': best['face_reg']
                }, path
            )

            best_path = os.path.join(model_save_path, 'best.ckpt')
            try:
            # Remove this file or symbolic link
                Path(best_path).unlink()
            except:
                pass
            # A shortcut to another file
            # i.e. save best new ckpt as best.ckpt
            Path(best_path).symlink_to(filename)

def plot_eval_graph(hist, img_save_path):
    iter_len = len(hist['iter'])
    x = np.arange(0, iter_len, 1)
    D_loss = hist['D_losses']
    G_loss = hist['G_losses']
    plt.plot(x, D_loss, label = 'discriminator loss')
    plt.plot(x, G_loss, label = 'generator loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss plot')
    plt.legend()
    plt.savefig(
        os.path.join(img_save_path, 'loss_plot')
    )
    plt.close()

    D_x = hist['D(x)']
    D_G_z1 = hist['D(G(z1))']
    D_G_z2 = hist['D(G(z2))']
    plt.plot(x, D_x, label = 'D(x)')
    plt.plot(x, D_G_z1, label = 'D(G(z1))')
    plt.plot(x, D_G_z2, label = 'D(G(z2))')
    plt.xlabel('iteration')
    plt.ylabel('score')
    plt.title('Score plot')
    plt.legend()
    plt.savefig(
        os.path.join(img_save_path, 'score_plot')
    )

def train_batch(data, netD, netG, optimizerD, optimizerG, loss_fn, args, hist):
    netD.zero_grad()
    # Format batch
    real = data.to(device)
    b_size = real.size(0)
    label = torch.full(
        (b_size,), real_label, 
        dtype=torch.float, device=args.device
        ) #create labels (1, 1, 1, ..., 1) since ImageFolder randomly create label if the dir strcture is wrong
    # Forward pass real batch through D
    output = netD(real).view(-1) #(bs,)
    # Calculate loss on all-real batch
    errD_real = loss_fn(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    # Compute D(x)
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    # TODO: Use a spherical Z (sample from gaussian distribution)
    if args.noise == 'StandardNorm':
        #(bs, c, h, w)
        noise = torch.randn(b_size, args.nz, 1, 1, device=args.device)
        noise_2 = torch.randn(b_size, args.nz, 1, 1, device=args.device)
    elif args.noise == 'Gaussian':
        pass
    # Generate fake image batch with G
    fake = netG(noise)
    fake_2 = netG(noise_2)

    label.fill_(fake_label) #(0, 0, ..., 0)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = loss_fn(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    hist['D(G(z1))'].append(D_G_z1)
    hist['D(x)'].append(D_x)
    hist['D_losses'].append(errD.item())


    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost, (1, 1, ..., 1)
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = loss_fn(output, label)

    # Add mode seeking loss
    # Ref: https://github.com/HelenMao/MSGAN/blob/master/DCGAN-Mode-Seeking/model.py
    if args.add_ms_loss:
        lz = torch.mean(torch.abs(fake_2 - fake)) / torch.mean(torch.abs(noise_2 - noise))
        eps = 1 * 1e-5
        ms_loss = 1 / (lz + eps)
        errG += ms_loss

    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()
    hist['G_losses'].append(errG.item())
    hist['D(G(z2))'].append(D_G_z2)
    gc.collect()
    torch.cuda.empty_cache()


def generate_pics_with_best_model(model_path, pic_save_path, fixed_noise, args):
    netD, netG = DCGAN_Discriminator(args).to(device), DCGAN_Generator(args).to(device)
    ckpt = torch.load(model_path)
    netD.load_state_dict(ckpt['netD'])
    netG.load_state_dict(ckpt['netG'])

    with torch.no_grad():
        fake = netG(fixed_noise)

    gc.collect()
    torch.cuda.empty_cache()
    grade = netD(fake).view(-1)
    _, indices = torch.sort(grade)
    indices = indices[:NUM_SELECTED]
    fake = fake[indices].detach().cpu()

    vutils.save_image(
        fake,
        padding=5, normalize=True,
        fp=os.path.join(pic_save_path, f"best_grid.png")
    )

    for i, img in enumerate(fake):
        vutils.save_image(
            img,
            normalize=True,
            fp=os.path.join(pic_save_path, f"{i}.png"),
        )

def eval(fixed_noise, netD, netG, epoch, i, img_save_path_ind, img_save_path_grid, hist):
    with torch.no_grad():
        fake = netG(fixed_noise)
    
    gc.collect()
    torch.cuda.empty_cache()
    # calculate grade by net D
    # choose top 1000 pics out of NUM_GEN_IMG
    grade = netD(fake).view(-1)
    _, indices = torch.sort(grade)
    indices = indices[:NUM_SELECTED]
    fake = fake[indices].detach().cpu()

    #save img grid
    vutils.save_image(
        fake,
        padding=5, normalize=True,
        fp=os.path.join(img_save_path_grid, f"epoch_{epoch}_iter_{i}.png")
    )

    #save individual img
    for i, img in enumerate(fake):
        vutils.save_image(
            img,
            normalize=True,
            fp=os.path.join(img_save_path_ind, f"{i}.png"),
        )
    
    face_reg = face_recog(img_save_path_ind)
    batcmd = f"python -W ignore -m pytorch_fid hw2_data/face/val {img_save_path_ind} --device cuda:0 --batch-size=20"
    result = subprocess.check_output(batcmd, shell=True)
    fid = float(str(result)[8:-3])
    #clear tqdm in fid
    # print ("\033[A                             \033[A")
    # print ("\033[A                             \033[A")

    print("\tface reg: %.4f\tfid: %.4f" % (face_reg, fid))

    hist['face_reg'].append(face_reg)
    hist['fid'].append(fid)
    

def main(args):
    #print(args)
    args_dict = vars(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")
    img_save_path = args.img_save_dir / str(date) / str(run_id)
    img_save_path_ind = os.path.join(img_save_path, 'ind')
    img_save_path_grid = os.path.join(img_save_path, 'grid')
    model_save_path = os.path.join(img_save_path, 'models')
    Path(img_save_path).mkdir(parents=True, exist_ok=True)
    os.mkdir(img_save_path_ind)
    os.mkdir(img_save_path_grid)
    os.mkdir(model_save_path)

    train_dataset = P1Dataset(args.train_dir, transform)
    #val_dataset = P1Dataset(args.val_dir, transform)

    train_dl = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    # val_dl = torch.utils.data.DataLoader(
    #     val_dataset, 
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers
    # )

    # TODO: Mode seeking loss
    # Initialize BCELoss function
    if args.loss_fn == 'BCE':
        loss_fn = nn.BCELoss()
    elif args.loss_fn == 'ModeSeekingLoss':
        #TODO
        pass

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    # TODO: Use a spherical Z (sample from gaussian distribution)
    if args.noise == 'StandardNorm':
        #generated 1000 image
        fixed_noise = torch.randn(args.num_pics_generated, args.nz, 1, 1, device=device)
    elif args.noise == 'Gaussian':
        #TODO
        pass

    # model
    netD, netG = DCGAN_Discriminator(args).to(device), DCGAN_Generator(args).to(device)
    netD, netG = netD.apply(weights_init), netG.apply(weights_init)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(
            netD.parameters(), lr=args.lr, 
            betas=(args.beta1, 0.999)
        )
    optimizerG = optim.Adam(
            netG.parameters(), lr=args.lr, 
            betas=(args.beta1, 0.999)
        )

    hist = {
        'epoch': [],
        'iter': [],
        'D(x)': [],
        'D(G(z1))': [],
        'D(G(z2))': [],
        'G_losses': [],
        'D_losses': [],
        'face_reg': [],
        'fid': []
    }

    best = {
        'epoch': 0,
        'iter': 0,
        'D(x)': 0,
        'D(G(z1))': 0,
        'D(G(z2))': 0,
        'G_losses': np.inf,
        'D_losses': np.inf,
        'face_reg': 0,
        'fid': np.inf
    }

    iters = 0
    for epoch in range(args.num_epoch):
        # For each batch in the dataloader
        #for i, data in enumerate(tqdm(train_dl, leave=False, colour='green')):
        for i, data in enumerate(train_dl):
            hist['epoch'].append(epoch)
            hist['iter'].append(i)
            train_batch(data, netD, netG, optimizerD, optimizerG, loss_fn, args, hist)
            
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (hist['epoch'][-1], args.num_epoch, i, len(train_dl),
                        hist['D_losses'][-1], hist['G_losses'][-1], 
                        hist['D(x)'][-1], hist['D(G(z1))'][-1], hist['D(G(z2))'][-1]),
                        end = ' '
                        )

                eval(fixed_noise, netD, netG, epoch, i, img_save_path_ind, img_save_path_grid, hist)
                #store best epoch, show in the end at store as training result
                save_best(netD, netG, best, hist, model_save_path)
                gc.collect()
                torch.cuda.empty_cache()

                # Record every 50 iters
                # TODO: add inception score
                with open(Path(img_save_path) / 'best.txt', 'w') as f:
                    print(best, file=f)
                    print(args_dict, file=f)
                f.close()

                
            iters += 1
            gc.collect()
            torch.cuda.empty_cache()

    #TODO: cmd line: ctrl + L OR clear tqdm output
    print(f"Finish model training, best training result:")
    print(best)
    plot_eval_graph(hist, img_save_path)

    with open(Path(img_save_path) / 'best.txt', 'w') as f:
        print(best, file=f)
        print(args_dict, file=f)
    f.close()

    best_save_path = os.path.join(img_save_path, 'best')
    os.mkdir(best_save_path)
    generate_pics_with_best_model(
        os.path.join(model_save_path, 'best.ckpt'), 
        best_save_path, fixed_noise, args)

if __name__ == '__main__':
    args = parse_args()
    # The parents=True tells the mkdir command to also create any intermediate parent directories 
    # that don't already exist.
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)