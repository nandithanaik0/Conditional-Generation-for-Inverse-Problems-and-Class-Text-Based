import torch
import os
import torch.nn.functional as F
from torchvision import utils as vutils

@torch.no_grad()
def classifier_accuracy_on_samples(clf, imgs, labels, device):
    clf.eval()
    logits = clf((imgs*2-1), torch.zeros(imgs.size(0), device=device, dtype=torch.long))
    preds = logits.argmax(1)
    return (preds==labels).float().mean().item()

@torch.no_grad()
def penultimate_features(clf, imgs, device):
    """Extract penultimate features from your classifier."""
    clf.eval()
    x = (imgs*2-1).to(device)
    t = torch.zeros(x.size(0), dtype=torch.long, device=device)
    h = F.silu(clf.conv1(x)); h = clf.pool(h)
    h = F.silu(clf.conv2(h)); h = clf.pool(h)
    h = h.view(h.size(0), -1)
    h = F.silu(clf.fc(h))
    return h  # [N, D]

@torch.no_grad()
def intraclass_diversity_cosine(clf, imgs, labels, device):
    """
    Mean (1 - cosine similarity) within each class, then averaged over classes.
    Higher = more diverse shapes *within the same label*.
    """
    feats = F.normalize(penultimate_features(clf, imgs, device), dim=1)
    classes = labels.unique()
    vals = []
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() < 2:
            continue
        f = feats[idx]  # [n_c, D]
        sim = f @ f.t()  # [n_c, n_c]
        n = f.size(0)
        mean_cos = (sim.sum() - n) / (n * (n - 1))
        vals.append(1.0 - mean_cos.item())
    return float(sum(vals) / max(len(vals), 1))

def save_grid(imgs, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(imgs, path, nrow=nrow, padding=2)

def save_weights(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ckpt] saved to {path}")

def load_weights(model: torch.nn.Module, path: str, device: torch.device) -> bool:
    if os.path.isfile(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"[ckpt] loaded from {path}")
        return True
    print(f"[ckpt] not found: {path}")
    return False