# coding=utf-8
from inspect import isfunction
import torch.nn as nn
import torch

def exists(x):
    return x is not None


# 有val时返回val，val为None时返回d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# 上采样（反卷积）
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


def cosine_beta_schedule(timesteps, s=0.008):
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
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
