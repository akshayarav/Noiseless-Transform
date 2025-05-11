import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange
T_STEPS = 100

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, dim)
    def forward(self, t):
        return torch.relu(self.lin(t.unsqueeze(-1)))
def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        
        self.time_proj = nn.Linear(time_emb_dim, dim_out) if time_emb_dim is not None else None

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.conv(x)

        if self.time_proj is not None:
            assert time_emb is not None, "time_emb must be provided if time_emb_dim is used"
            t = self.time_proj(time_emb)
            h = h + t.unsqueeze(-1).unsqueeze(-1)  # shape [B, C, 1, 1]

        return h + self.res_conv(x)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
    
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        return self.final_conv(x)

# class SimpleUNet(nn.Module):
#     def __init__(self, in_channels=3, time_dim=32):
#         super().__init__()
#         self.time_emb = TimeEmbedding(time_dim)
#         self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)   # 64x64 -> 32x32
#         self.down2 = nn.Conv2d(64, 128, 4, 2, 1)            # 32x32 -> 16x16
#         self.down3 = nn.Conv2d(128, 256, 4, 2, 1)           # 16x16 -> 8x8  (new layer!)

#         self.time_mlp = nn.Linear(time_dim, 256)  # Now matching 256 channels at the bottom

#         # --- Decoder (Upsampling path) ---
#         self.up0 = nn.ConvTranspose2d(256, 128, 4, 2, 1)    # 8x8 -> 16x16 (new layer!)
#         self.up1 = nn.ConvTranspose2d(256, 64, 4, 2, 1)     # 16x16 -> 32x32
#         self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1)    # After skip concat
#         self.up2 = nn.ConvTranspose2d(64, in_channels, 4, 2, 1)  # 32x32 -> 64x64

#     def forward(self, x, s):
#         t_emb = self.time_emb(s.float() / T_STEPS)  
#         t_emb = self.time_mlp(t_emb)

#         # --- Down path ---
#         h1 = torch.relu(self.down1(x))  # (B, 64, 32, 32)
#         h2 = torch.relu(self.down2(h1)) # (B, 128, 16, 16)
#         h3 = torch.relu(self.down3(h2) + t_emb.view(-1, t_emb.shape[-1], 1, 1))  # (B, 256, 8, 8)

#         # --- Up path ---
#         h = torch.relu(self.up0(h3))    # (B, 128, 16, 16)

#         # Skip connection with h2
#         h = torch.cat([h, h2], dim=1)   # (B, 256, 16, 16)
#         h = torch.relu(self.up1(h))     # (B, 64, 32, 32)

#         # Skip connection with h1
#         h = torch.cat([h, h1], dim=1)   # (B, 128, 32, 32)
#         h = self.conv1x1(h)             # (B, 64, 32, 32)

#         # Final upsampling
#         out = self.up2(h)               # (B, 3, 64, 64)

#         return out



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# T_STEPS = 100
# class TimeEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.lin = nn.Linear(1, dim)
#     def forward(self, t):
#         return torch.relu(self.lin(t.unsqueeze(-1)))

# class SimpleUNet(nn.Module):
#     def __init__(self, in_channels=3, time_dim=32):
#         super().__init__()
#         self.time_emb = TimeEmbedding(time_dim)
        
#         self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
#         self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
#         self.up1   = nn.ConvTranspose2d(128, 64, 4, 2, 1)
#         self.up2   = nn.ConvTranspose2d(64, in_channels, 4, 2, 1)
#         self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1)
#         self.time_mlp = nn.Linear(time_dim, 128)

#     def forward(self, x, s):
#       t_emb = self.time_emb(s.float() / T_STEPS)  
#       t_emb = self.time_mlp(t_emb)
#       h1 = torch.relu(self.down1(x))
#       h2 = torch.relu(self.down2(h1) + t_emb.view(-1, t_emb.shape[-1], 1, 1))

#       h = torch.relu(self.up1(h2))
#       h = torch.cat([h, h1], dim=1)
#       h = self.conv1x1(h)

#       return self.up2(h)

    


# from tqdm import tqdm
# import random
# from kernel import swirl_function
# import torch.nn as nn

# l1 = nn.L1Loss()
# def training_loop(model, loader, optimizer, device):
#     for epoch in range(20):
#         progressbar = tqdm(loader)
#         for x0, _ in progressbar:
#             x0 = x0.to(device)
#             s = random.randint(1, T_STEPS)
#             xs = swirl_function(x0, s)                    
#             x_pred = model(xs, torch.tensor([s]*x0.size(0), device=device))
#             loss = l1(x_pred, x0)            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             progressbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")
