import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

# Constants
T_STEPS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sigmas = [math.exp(0.01 * s) for s in range(T_STEPS + 1)]

# Blur kernel setup
def gaussian_kernel(sigma: float, kernel_size=27):
    x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    row_vec = x.unsqueeze(0)
    row_vec_squared = row_vec**2
    col_vec = x.unsqueeze(1)
    col_vec_squared = col_vec**2
    grid = torch.exp(-(row_vec_squared + col_vec_squared) / (sigma**2))
    kernel = grid / grid.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)

kernels_list = [gaussian_kernel(sigmas[s]).to(DEVICE) for s in range(T_STEPS + 1)]

# Blur function
def blur_function(x: torch.Tensor, step: int) -> torch.Tensor:
    B, C, H, W = x.shape
    out = []
    for c in range(C):
        out.append(F.conv2d(x[:, c:c+1], kernels_list[step], padding=kernels_list[step].shape[-1] // 2))
    return torch.cat(out, dim=1)

# Transform and load CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))  # upscale for better blur visibility
])
dataset = datasets.CelebA(root="./data", download=True, transform=transform)
sample_img, _ = dataset[0]
sample_img = sample_img.unsqueeze(0).to(DEVICE)  # shape: [1, 3, H, W]

# Apply blur
blurred_0 = blur_function(sample_img, 0).cpu().squeeze().permute(1, 2, 0).numpy()
blurred_100 = blur_function(sample_img, 100).cpu().squeeze().permute(1, 2, 0).numpy()
original = sample_img.cpu().squeeze().permute(1, 2, 0).numpy()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(blurred_0)
axes[1].set_title("Blurred at Step 0")
axes[1].axis('off')

axes[2].imshow(blurred_100)
axes[2].set_title("Blurred at Step 100")
axes[2].axis('off')

plt.tight_layout()
plt.show()
