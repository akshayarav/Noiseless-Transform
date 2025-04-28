import math
import torch.nn.functional as F
import torch

T_STEPS    = 300  
    #   We can reduce this if too many steps slows training, but thats what was in paper
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
sigmas = [math.exp(0.01 * s) for s in range(T_STEPS+1)]  

# 27 is kernel size in paper we can maybe change this if its slow 
def gaussian_kernel(sigma: float, kernel_size=27):
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    row_vec = x.unsqueeze(0)
    row_vec_squared = row_vec**2
    col_vec = x.unsqueeze(1)
    col_vec_squared = col_vec**2
    grid = torch.exp(- (row_vec_squared + col_vec_squared) / (sigma**2))
    kernel = grid / grid.sum()
    return kernel.view(1,1,kernel_size,kernel_size)

kernels_list = [gaussian_kernel(sigmas[s]).to(DEVICE) for s in range(T_STEPS+1)]

def blur_function(x: torch.Tensor, step: int) -> torch.Tensor:
    B,C,H,W = x.shape
    out = []
    for c in range(C):
        # Im gettign pylint error here dont know why
        out.append(F.conv2d(x[:,c:c+1], kernels_list[step], padding=kernels_list[step].shape[-1]//2))
    return torch.cat(out, dim=1)

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

def swirl_function(x: torch.Tensor, step: int) -> torch.Tensor:
    B, C, H, W = x.shape
    out = []
    max_swirl_strength = 3.0
    swirl_strength = (step / T_STEPS) * max_swirl_strength
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    r = torch.sqrt(xx**2 + yy**2)
    theta = torch.atan2(yy, xx) + swirl_strength * r
    xx_new = r * torch.cos(theta)
    yy_new = r * torch.sin(theta)
    grid = torch.stack((xx_new, yy_new), dim=-1)

    for c in range(C):
        out.append(F.grid_sample(x[:,c:c+1], grid.expand(B, -1, -1, -1), mode='bilinear', padding_mode='border', align_corners=True))

    return torch.cat(out, dim=1)
