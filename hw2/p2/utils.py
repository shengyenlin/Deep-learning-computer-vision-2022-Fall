import torch
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    # X -> X_t
def extract(a, t, x_shape):
    # t: (bs, )
    batch_size = t.shape[0]
    out = a.gather(dim=-1, index=t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def precalculate(beta_schedule, timesteps):
    # define beta schedule
    if beta_schedule == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    elif beta_schedule == 'quadratic':
        betas = quadratic_beta_schedule(timesteps)
    elif beta_schedule == 'sigmoid':
        betas = sigmoid_beta_schedule(timesteps)

    # define alphas 
    alphas = 1. - betas #(timesteps,)
    alphas_cumprod = torch.cumprod(alphas, axis=0) #(timesteps,)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) #alpha value of previous step in each slot, (timesteps,)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) 
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    # beta^{tilda}
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return betas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance