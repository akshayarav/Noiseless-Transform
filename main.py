import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from kernel import swirl_function, blur_function
from model import Unet
import torch.nn as nn

l1 = nn.L1Loss()
BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 20
IMG_SIZE = 64
T_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


transform = transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.STL10(
    root="./data",
    split="train",
    download=True,
    transform=transform  # ✅ use the variable you defined above
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = Unet(32).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)


def train():
    for epoch in range(EPOCHS):
        model.train()
        for i, (x0, _) in enumerate(loader):
            x0 = x0.to(DEVICE)
            s = torch.randint(1, T_STEPS+1, (x0.size(0),), device=DEVICE)
            xs = torch.stack([blur_function(x.unsqueeze(0), step.item())[0] for x, step in zip(x0, s)], dim=0)

            x_pred = model(xs, s)
            loss = l1(x_pred, x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{i}/{len(loader)}] Loss: {loss.item():.4f}")

        save_generated_images(epoch, loader)


def save_generated_images(epoch, data_loader):
    model.eval()
    with torch.no_grad():
        x0, _ = next(iter(data_loader))
        x0 = x0.to(DEVICE)

        # Create degraded input at t = T
        s_full = torch.full((x0.size(0),), T_STEPS, device=DEVICE)
        x_blurred = torch.stack(
            [blur_function(img.unsqueeze(0), step.item())[0] for img, step in zip(x0, s_full)],
            dim=0
        )

        # Save original and degraded (blurred) images
        save_image(x0.cpu(), os.path.join(OUTPUT_FOLDER, f"sample_epoch_{epoch}_original.png"), nrow=4, normalize=True)
        save_image(x_blurred.cpu(), os.path.join(OUTPUT_FOLDER, f"sample_epoch_{epoch}_blurred.png"), nrow=4, normalize=True)

        ### Algorithm 2 — Cold Diffusion (Improved Sampling)
        x_alg2 = x_blurred.clone()
        for s in reversed(range(1, T_STEPS + 1)):
            s_tensor = torch.full((x_alg2.size(0),), s, device=DEVICE)
            x0_hat = model(x_alg2, s_tensor)
            D_s = torch.stack([blur_function(img.unsqueeze(0), s)[0] for img in x0_hat], dim=0)
            D_s_minus_1 = torch.stack([blur_function(img.unsqueeze(0), s - 1)[0] for img in x0_hat], dim=0)
            x_alg2 = x_alg2 - D_s + D_s_minus_1

        save_image(x_alg2.cpu(), os.path.join(OUTPUT_FOLDER, f"sample_epoch_{epoch}_alg2.png"), nrow=4, normalize=True)

        ### Algorithm 1 — Naive Sampling
        x_alg1 = x_blurred.clone()
        for s in reversed(range(1, T_STEPS + 1)):
            s_tensor = torch.full((x_alg1.size(0),), s, device=DEVICE)
            x0_hat = model(x_alg1, s_tensor)
            x_alg1 = torch.stack([blur_function(img.unsqueeze(0), s - 1)[0] for img in x0_hat], dim=0)

        save_image(x_alg1.cpu(), os.path.join(OUTPUT_FOLDER, f"sample_epoch_{epoch}_alg1.png"), nrow=4, normalize=True)

        # Save model weights
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()
