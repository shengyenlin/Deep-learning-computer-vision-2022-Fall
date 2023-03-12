import os
import random
import sys

#torch
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils

from p1.model import DCGAN_Generator, DCGAN_Discriminator

# Set random seed for reproducibility
SEED = 5566
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

NUM_PIC_GENERATED = 1000
OUT_PATH = sys.argv[1]
MODEL_PATH = "p1.ckpt"

class MODEL_ARGS:
    def __init__(self):
        self.nz = 300
        self.ngf = 64
        self.ndf = 64
        self.nc = 3

def generate_pics_with_best_model():
    args = MODEL_ARGS()
    netD, netG = DCGAN_Discriminator(args).to(device), DCGAN_Generator(args).to(device)
    ckpt = torch.load(MODEL_PATH)
    netD.load_state_dict(ckpt['netD'])
    netG.load_state_dict(ckpt['netG'])

    noise = torch.randn(NUM_PIC_GENERATED, args.nz, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise).detach().cpu()

    for i, img in enumerate(fake):
        vutils.save_image(
            img,
            normalize=True,
            fp=os.path.join(OUT_PATH, f"{i}.png"),
        )

def main():
    generate_pics_with_best_model()

if __name__ == '__main__':
    main()