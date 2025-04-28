import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from kernel import swirl_function
from model import SimpleUNet

BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 20
IMG_SIZE = 64
T_STEPS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(
    root="./small_data",
    train=True,
    download=True,
    transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = SimpleUNet(in_channels=3, time_dim=32).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
l1 = nn.L1Loss()

latest_ckpt = os.path.join(OUTPUT_FOLDER, "unet_latest.pth")
if os.path.isfile(latest_ckpt):
    model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
    print(f"Loaded model from checkpoint: {latest_ckpt}")

def train():
    for epoch in range(EPOCHS):
        model.train()
        for i, (x0, _) in enumerate(loader):
            x0 = x0.to(DEVICE)
            s = torch.randint(1, T_STEPS + 1, (x0.size(0),), device=DEVICE)
            xs = torch.stack([
                swirl_function(img.unsqueeze(0), step.item())[0]
                for img, step in zip(x0, s)
            ], dim=0)

            x_pred = model(xs, s)
            loss = l1(x_pred, x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{i}/{len(loader)}] Loss: {loss.item():.4f}")

        save_generated_images(epoch, loader)
        epoch_ckpt = os.path.join(OUTPUT_FOLDER, f"unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_ckpt)
        torch.save(model.state_dict(), latest_ckpt)
        print(f"Saved checkpoint: {epoch_ckpt}")


def save_generated_images(epoch, data_loader):
    model.eval()
    with torch.no_grad():
        x0, _ = next(iter(data_loader))
        x0 = x0.to(DEVICE)

        s_full = torch.full((x0.size(0),), T_STEPS, device=DEVICE)
        x = torch.stack([
            swirl_function(img.unsqueeze(0), step.item())[0]
            for img, step in zip(x0, s_full)
        ], dim=0)

        for step in reversed(range(1, T_STEPS + 1)):
            s = torch.full((x.size(0),), step, device=DEVICE)
            x = model(x, s)

        save_image(
            x.cpu(),
            os.path.join(OUTPUT_FOLDER, f"sample_epoch_{epoch}.png"),
            nrow=4,
            normalize=True
        )

if __name__ == "__main__":
    train()
