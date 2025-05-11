import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
import torch.nn.functional as F



def get_activations(samples: torch.Tensor,
                    model: InceptionV3,
                    batch_size: int = 50,
                    dims: int = 2048,
                    device: str = 'cpu') -> np.ndarray:
    model.eval()
    n = samples.size(0)
    acts = np.empty((n, dims))
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = samples[i:i+batch_size].to(device)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)  # 🔧 Add this line
            pred = model(batch)[0]
            if pred.dim() == 4 and pred.size(2) != 1:
                pred = adaptive_avg_pool2d(pred, (1,1))
            acts[i:i+batch_size] = pred.squeeze(-1).squeeze(-1).cpu().numpy()
    return acts


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Compute Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = eps * np.eye(sigma1.shape[0])
        covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    covmean = np.real(covmean)
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)

def calculate_activation_statistics(samples: torch.Tensor,
                                    model: InceptionV3,
                                    **kwargs) -> tuple[np.ndarray,np.ndarray]:
    acts = get_activations(samples, model, **kwargs)
    return acts.mean(axis=0), np.cov(acts, rowvar=False)

def calculate_fid_given_samples(real: torch.Tensor,
                                gen:  torch.Tensor,
                                batch_size: int = 50,
                                device: str = 'cuda:0',
                                dims: int   = 2048) -> float:
    """Main entry: returns FID(real, gen)."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mu1, sigma1 = calculate_activation_statistics(real, model,
                      batch_size=batch_size, dims=dims, device=device)
    mu2, sigma2 = calculate_activation_statistics(gen, model,
                      batch_size=batch_size, dims=dims, device=device)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)