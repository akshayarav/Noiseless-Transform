import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

IMG_SIZE   = 128
BATCH_SIZE = 64
T_STEPS    = 300        
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),               
    transforms.Normalize([0.5]*3, [0.5]*3)  
])

celeba = datasets.CelebA(root="data", split="validation", download=True, transform=transform)
loader = DataLoader(celeba, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print(len(loader))