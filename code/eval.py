import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
import random
import numpy as np

from model import Unet
from kernel import blur_function
from fid_utils import calculate_fid_given_samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_SIZE = 64 #If CIFAR10, change to 32
T_STEPS = 100
CHECKPOINT_PATH = "checkpoint_epoch_13.pt"

REAL_DIR = "fid_direct/real"
RECON_DIR = "fid_direct/reconstructed"
BLUR_DIR = "fid_blurred/blurred"
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(RECON_DIR, exist_ok=True)
os.makedirs(BLUR_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.STL10(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

'''
For CIFAR10:

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
'''
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = Unet(dim=32).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

real_images = []
recon_images = []
blurred_images = []

with torch.no_grad():
    for batch_idx, (x0, _) in enumerate(loader):
        x0 = x0.to(DEVICE)
        s_tensor = torch.full((x0.size(0),), T_STEPS, device=DEVICE)
        x_blurred = blur_function(x0, T_STEPS)
        x_direct = model(x_blurred, s_tensor)
        x0 = torch.clamp(x0, 0., 1.)
        x_blurred = torch.clamp(x_blurred, 0., 1.)
        x_direct = torch.clamp(x_direct, 0., 1.)
        for i in range(x0.size(0)):
            idx = batch_idx * BATCH_SIZE + i
            save_image(x0[i].cpu(), f"{REAL_DIR}/{idx:05d}.png")
            save_image(x_blurred[i].cpu(), f"{BLUR_DIR}/{idx:05d}.png")
            save_image(x_direct[i].cpu(), f"{RECON_DIR}/{idx:05d}.png")
        real_images.append(x0.cpu())
        blurred_images.append(x_blurred.cpu())
        recon_images.append(x_direct.cpu())
real_tensor = torch.cat(real_images, dim=0)
blurred_tensor = torch.cat(blurred_images, dim=0)
recon_tensor = torch.cat(recon_images, dim=0)

fid_direct = calculate_fid_given_samples(
    real=real_tensor,
    gen=recon_tensor,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    dims=2048
)

fid_blurred = calculate_fid_given_samples(
    real=real_tensor,
    gen=blurred_tensor,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    dims=2048
)

print(f"FID Score (full dataset - reconstruction): {fid_direct:.4f}")
print(f"FID Score (full dataset - blurred):       {fid_blurred:.4f}")
