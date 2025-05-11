import torch
import torch.nn as nn
from torchvision.models import inception_v3

class InceptionV3(nn.Module):
    BLOCK_INDEX_BY_DIM = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3
    }

    def __init__(self, output_blocks=[3], resize_input=True, normalize_input=True):
        super().__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.blocks = nn.ModuleList()

        from torchvision.models import Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block1))
        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d
        ]
        self.blocks.append(nn.Sequential(*block2))
        block3 = [
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        self.blocks.append(nn.Sequential(*block3))

        self.output_blocks = sorted(output_blocks)

    def forward(self, x):
        outputs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
        return outputs
