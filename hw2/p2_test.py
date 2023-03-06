import os
import random
import sys
import numpy as np

#torch
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils

from p2.utils import precalculate, extract
from p2.model import Unet


# Set random seed for reproducibility
SEED = 34
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

GEN_IMG_PER_CLASS = 100
NUM_CLASS = 10 
OUT_PATH = sys.argv[1]
MODEL_PATH = "p2.ckpt"

def get_pic_idx(i) -> str:
    if i // 10 == 0:
        return f'00{str(i)}'
    elif i // 10 == 10:
        return i
    else:
        return f'0{str(i)}'

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
def p_sample_loop(label, precal, model, args, shape):
    device = next(model.parameters()).device

    b = shape[0]

    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, args.timesteps)):
        img = p_sample(model, img, label, torch.full((b,), i, device=device, dtype=torch.long), i, precal)
        imgs.append(img)
    return imgs

@torch.no_grad()
def sample(label, precal, model, args, batch_size):
    # List[np.array]
    return p_sample_loop(
        label, precal, model, args, shape=(batch_size, args.channels, args.image_size, args.image_size)
        )

class Args:
    def __init__(self):
        self.timesteps = 200
        self.channels = 3
        self.image_size = 28
        self.beta_schedule = "linear"

def main():
    args = Args()

    betas, alphas_cumprod, alphas_cumprod_prev, \
        sqrt_recip_alphas, sqrt_alphas_cumprod, \
        sqrt_one_minus_alphas_cumprod, posterior_variance = \
        precalculate(args.beta_schedule, args.timesteps) 

    precal = {
       'betas': betas,
       'alphas_cumprod': alphas_cumprod,
       'alphas_cumprod_prev': alphas_cumprod_prev,
       'sqrt_recip_alphas': sqrt_recip_alphas,
       'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
       'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
       'posterior_variance': posterior_variance,
    }

    model = Unet(
            dim=args.image_size,
            channels=args.channels,
            dim_mults=(1, 2, 4,), #or dim_mults=(1, 2, 4, 8)?
        )
    
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    for i in range(NUM_CLASS):
        class_label = np.repeat(i, GEN_IMG_PER_CLASS)
        class_label_ts = torch.from_numpy(class_label)
        class_label_ts = class_label_ts.to(device)
        class_img_list = sample(class_label_ts, precal,model, args, GEN_IMG_PER_CLASS)

        # if i == 0:
        #     # save the first img in class 0
        #     steps = [0, 39, 79, 119, 159, 199]
        #     pics_one_one = []
        #     for step in steps:
        #         img = class_img_list[step][i]
        #         img = (img + 1) * 0.5
        #         img = img.unsqueeze(0)
        #         pics_one_one.append(img)
        #     pics_one_one = torch.cat(pics_one_one, axis=0)
        #     vutils.save_image(
        #         pics_one_one,
        #         padding=5, 
        #         #normalize=True,
        #         fp=os.path.join(OUT_PATH, f"grid_one_one.png")
        #     )

        for j, img in enumerate(class_img_list[-1], start=1):
            img = (img + 1) * 0.5
            pic_idx = get_pic_idx(j)
            vutils.save_image(
                img,
                #normalize=True,
                fp=os.path.join(OUT_PATH, f"{class_label[j-1]}_{pic_idx}.png"),
            )
    
if __name__ == '__main__':
    main()
