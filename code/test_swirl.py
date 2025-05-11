import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from kernel import swirl_function


def test_swirl_function(swirl_function, step=100):
    # Load a simple test image (like one CIFAR10 image)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    x0, _ = next(iter(loader))  # get one image
    x0 = x0.to("cpu")

    # Apply swirl function
    swirled_img = swirl_function(x0, step)  # Assuming swirl_function returns (output, _)

    # Plot original and swirled side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x0.squeeze(0).permute(1, 2, 0).numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(swirled_img.squeeze(0).permute(1, 2, 0).numpy())
    axes[1].set_title(f"Swirled Image (step={step})")
    axes[1].axis('off')

    plt.show()

test_swirl_function(swirl_function, 1000)
# import matplotlib.pyplot as plt
# import torchvision
# import torchvision.transforms as transforms
# import torch

# # Define transform
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # Resize to 64x64
#     transforms.ToTensor()
# ])

# # Load CIFAR-10 dataset
# dataset = torchvision.datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform
# )

# # Create DataLoader
# loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# # Get two images
# images, labels = next(iter(loader))  # images: (2, 3, 64, 64)

# # Plot the two images
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# for idx in range(2):
#     img = images[idx].permute(1, 2, 0).numpy()  # (64, 64, 3)
#     axes[idx].imshow(img)
#     axes[idx].set_title(f"Label: {labels[idx].item()}")
#     axes[idx].axis('off')
# plt.show()
