import torch
import torch.nn as nn
import torch.nn.functional as F

T_STEPS = 300
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(1, dim)
    def forward(self, t):
        return torch.relu(self.lin(t.unsqueeze(-1)))

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=32):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.up1   = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2   = nn.ConvTranspose2d(64, in_channels, 4, 2, 1)

    def forward(self, x, s):
        t_emb = self.time_emb(s.float() / T_STEPS)  
        h = torch.relu(self.down1(x))
        h = torch.relu(self.down2(h) + t_emb.view(-1, t_emb.shape[-1], 1, 1))
        h = torch.relu(self.up1(h))
        return self.up2(h)  
