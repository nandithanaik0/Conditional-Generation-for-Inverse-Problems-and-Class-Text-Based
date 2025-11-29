import time
import torch.nn as nn
from typing import Optional
from step2_utils.diff_utils import *

# ---------------------------
# Tiny conditional UNet (MNIST 28x28)
# ---------------------------

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, time_dim, label_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, ch_out))
        self.label_mlp = nn.Sequential(nn.SiLU(), nn.Linear(label_dim, ch_out)) if label_dim>0 else None
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x, t_emb, y_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        add = self.time_mlp(t_emb)
        if exists(y_emb) and self.label_mlp is not None:
            add = add + self.label_mlp(y_emb)
        h = h + add[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class TinyUNet(nn.Module):
    def __init__(self, base_ch=32, time_dim=64, num_classes=10, cond_drop_prob=0.1):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, base_ch)
        label_dim = base_ch

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, base_ch*4), nn.SiLU(), nn.Linear(base_ch*4, time_dim)
        )

        self.inp = nn.Conv2d(1, base_ch, 3, padding=1)
        # down
        self.rb1 = ResBlock(base_ch, base_ch, time_dim, label_dim)
        self.down = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)  # 28->14
        self.rb2 = ResBlock(base_ch*2, base_ch*2, time_dim, label_dim)
        # mid
        self.mid1 = ResBlock(base_ch*2, base_ch*2, time_dim, label_dim)
        self.mid2 = ResBlock(base_ch*2, base_ch*2, time_dim, label_dim)
        # up
        self.up = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)  # 14->28
        self.rb3 = ResBlock(base_ch*2, base_ch, time_dim, label_dim)
        self.out = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x_t, t, y: Optional[torch.Tensor]=None, force_uncond: bool=False):
        """
        x_t: [B,1,28,28]
        t: [B] integer indices or normalized floats
        y: [B] labels (int) or None
        If force_uncond=True, ignore labels (used to get unconditional branch for CFG).
        """
        if t.ndim == 0:
            t = t.expand(x_t.size(0))
        if t.dtype in (torch.long, torch.int32, torch.int64):
            t_norm = (t.float() + 1e-8) / 999.0
        else:
            t_norm = t
        t_emb = self.time_mlp(sinusoidal_time_embedding(t_norm, self.time_dim))

        # classifier-free label dropout during training OR explicit unconditional during sampling
        if (self.training and self.cond_drop_prob>0) or force_uncond or y is None:
            if force_uncond or y is None:
                y_emb = torch.zeros(x_t.size(0), self.label_embed.embedding_dim, device=x_t.device)
            else:
                drop_mask = (torch.rand(y.shape, device=x_t.device) < self.cond_drop_prob)
                y2 = y.clone()
                y2[drop_mask] = 0
                y_emb = self.label_embed(y2)
                y_emb[drop_mask] = 0.0
        else:
            y_emb = self.label_embed(y)

        h0 = self.inp(x_t)
        h1 = self.rb1(h0, t_emb, y_emb)
        d = self.down(h1)
        h2 = self.rb2(d, t_emb, y_emb)
        m = self.mid2(self.mid1(h2, t_emb, y_emb), t_emb, y_emb)
        u = self.up(m)
        u = torch.cat([u, h1], dim=1)
        u = self.rb3(u, t_emb, y_emb)
        out = self.out(u)   # predicts epsilon
        return out

# ---------------------------
# Noise-aware classifier p(y|x_t, t)
# ---------------------------

class NoiseAwareClassifier(nn.Module):
    def __init__(self, time_dim=64, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.fc = nn.Linear(64*7*7, 256)  # 28->14->7; 64*7*7=3136
        self.head = nn.Linear(256, num_classes)
        self.time_dim = time_dim

    def forward(self, x_t, t):
        if t.ndim == 0:
            t = t.expand(x_t.size(0))
        if t.dtype in (torch.long, torch.int32, torch.int64):
            t = (t.float()+1e-8)/999.0
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_feat = self.time_mlp(t_emb)[:, :, None, None]  # [B,64,1,1]

        h = F.silu(self.conv1(x_t))
        h = self.pool(h)                  # 28->14
        h = F.silu(self.conv2(h))
        h = h + t_feat
        h = self.pool(h)                  # 14->7
        h = h.view(h.size(0), -1)
        h = F.silu(self.fc(h))
        return self.head(h)

# ---------------------------
# Training loops
# ---------------------------

def ddpm_loss(model: TinyUNet, sched: DiffusionSchedule, x0, y, device):
    B = x0.size(0)
    t_idx = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
    x_t, eps = sched.q_sample(x0, t_idx)
    pred = model(x_t, t_idx, y)
    return F.mse_loss(pred, eps)

def train_diffusion(model, sched, loader, epochs, device, lr=2e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs+1):
        t0 = time.time()
        loss_avg = 0.0
        for x,y in loader:
            x = (x*2-1).to(device)  # scale to [-1,1]
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ddpm_loss(model, sched, x, y, device)
            loss.backward()
            opt.step()
            loss_avg += loss.item()*x.size(0)
        loss_avg /= len(loader.dataset)
        print(f"[Diffusion] epoch {ep}/{epochs}  loss={loss_avg:.4f}  ({time.time()-t0:.1f}s)")

def train_classifier(clf, sched, loader, epochs, device, lr=2e-4, noise_prob=0.9):
    """
    Train on noisy x_t with random t, to make classifier noise-aware (as in Dhariwal & Nichol).
    """
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    clf.train()
    for ep in range(1, epochs+1):
        t0 = time.time()
        correct = 0
        total = 0
        loss_avg = 0.0
        for x,y in loader:
            x = (x*2-1).to(device)
            y = y.to(device)
            B = x.size(0)
            t_idx = torch.randint(0, sched.T, (B,), device=device)
            if noise_prob < 1.0:
                mask = (torch.rand(B, device=device) > noise_prob)
                t_idx[mask] = torch.randint(0, 10, (mask.sum(),), device=device)
            x_t, _ = sched.q_sample(x, t_idx)
            logits = clf(x_t, t_idx)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_avg += loss.item()*B
            pred = logits.argmax(1)
            correct += (pred==y).sum().item(); total += B
        print(f"[Classifier] epoch {ep}/{epochs}  loss={loss_avg/len(loader.dataset):.4f}  noisy-acc={correct/total:.3f}  ({time.time()-t0:.1f}s)")