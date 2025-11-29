import torch
import math
import torch.nn.functional as F

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: scalar or [B] (int indices or normalized floats in [0,1])
    returns [B, dim]
    """
    if t.dim() == 0:
        t = t[None]
    t = t.float()
    device = t.device
    half = dim // 2
    # frequencies from 1..10k
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def exists(x): return x is not None

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).to(dtype=torch.float32)

def linear_beta_schedule(T: int, min_beta=1e-4, max_beta=0.02) -> torch.Tensor:
    return torch.linspace(min_beta, max_beta, T, dtype=torch.float32)

class DiffusionSchedule:
    def __init__(self, min_beta=1e-4, max_beta=0.02, device="cpu"):
        T = 1000
        betas = linear_beta_schedule(T, min_beta, max_beta)
        self.device = device
        self.T = T
        self.betas = betas.to(device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas
            * (1. - F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
            / (1. - self.alphas_cumprod)
        ).to(device)
        self.sigmas = torch.sqrt(self.betas).to(device)
        # move everything to device
        for name in [
            "betas","alphas","alphas_cumprod","sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod","sqrt_recip_alphas"
        ]:
            setattr(self, name, getattr(self, name).to(device))

    def q_sample(self, x0, t_idx, noise=None):
        """
        x_t = sqrt(\bar{alpha}_t) x0 + sqrt(1-\bar{alpha}_t) eps
        t_idx: [B] integer in [0, T-1]
        """
        if noise is None: noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t_idx][:, None, None, None]
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t_idx][:, None, None, None]
        return sqrt_ab * x0 + sqrt_omb * noise, noise