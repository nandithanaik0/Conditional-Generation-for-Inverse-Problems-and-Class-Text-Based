import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from step2_utils.model_utils import TinyUNet, NoiseAwareClassifier
from step2_utils.diff_utils import DiffusionSchedule
from step2_utils.generic_utils import *

def sample_cg(model, clf, sched, cls_labels, num_steps, guidance_scale, device="cpu"):
    ############################
    # Hint-1: alpha_t = sched.alphas[t_idx], abar_t  = sched.alphas_cumprod[t_idx]
    # Hint-2: model(x_t, t_vec, cls_labels, force_uncond=True) returns the unconditional eps --> ϵθ(xt, t, ∅)
    # Hint-3: Once you obtain the logits from the classifier, log pϕ(y|xt, t) can be found via: logp_y = grad_log_softmax(logits, cls_labels
    ############################
    
    def grad_log_softmax(logits, y):
        return F.log_softmax(logits, dim=1)[range(y.size(0)), y]
    
    model.eval(); clf.eval()
    B = cls_labels.size(0)
    torch.manual_seed(0)
    x_t = torch.randn(B,1,28,28, device=device).requires_grad_(True)
    timesteps = torch.linspace(sched.T-1, 0, num_steps, device=device, dtype=torch.long)
    for i, t_idx in enumerate(timesteps):
        
        ############################
        # Implement classifier guidance (CG) sampling
        ############################
        t_vec = t_idx.repeat(B)  # shape [B]

        alpha_t = sched.alphas[t_idx]                
        abar_t  = sched.alphas_cumprod[t_idx]        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_abar_t = torch.sqrt(1.0 - abar_t)
        coef_eps = (1.0 - alpha_t) / sqrt_one_minus_abar_t

    
        eps_uncond = model(x_t, t_vec, cls_labels, force_uncond=True)

        logits = clf(x_t, t_vec)
        logp_y = grad_log_softmax(logits, cls_labels)
        grad = torch.autograd.grad(logp_y.sum(), x_t)[0]

        eps_tilde = eps_uncond - guidance_scale * sqrt_one_minus_abar_t * grad

        x_prev_mean = (1.0 / sqrt_alpha_t) * (x_t - coef_eps * eps_tilde)
        
        sigma_t = torch.sqrt(sched.posterior_variance[t_idx].clamp(min=1e-20))
        if i < num_steps - 1:
            x_t = (x_prev_mean + sigma_t * torch.randn_like(x_t)).detach().requires_grad_(True)
        else:
            x_t = x_prev_mean.detach()
    return (x_t.clamp(-1,1) + 1)/2

@torch.no_grad()
def sample_cfg(model, sched, cls_labels, num_steps, guidance_scale, device="cpu"):
    
    model.eval()
    B = cls_labels.size(0)
    torch.manual_seed(0)
    x_t = torch.randn(B,1,28,28, device=device)
    timesteps = torch.linspace(sched.T-1, 0, num_steps, device=device, dtype=torch.long)
    for i, t_idx in enumerate(timesteps):
        
        ############################
        # Implement classifier-free guidance (CFG) sampling
        ############################
        t_vec = t_idx.repeat(B)  # shape [B]

        alpha_t = sched.alphas[t_idx]
        abar_t  = sched.alphas_cumprod[t_idx]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_abar_t = torch.sqrt(1.0 - abar_t)
        coef_eps = (1.0 - alpha_t) / sqrt_one_minus_abar_t

        eps_uncond = model(x_t, t_vec, cls_labels, force_uncond=True)  # ϵθ(xt,t,∅)
        eps_cond   = model(x_t, t_vec, cls_labels)                     # ϵθ(xt,t,c)

        eps_tilde = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        x_prev_mean = (1.0 / sqrt_alpha_t) * (x_t - coef_eps * eps_tilde)
        
        sigma_t = torch.sqrt(sched.posterior_variance[t_idx].clamp(min=1e-20))
        if i < num_steps - 1:
            x_t = x_prev_mean + sigma_t * torch.randn_like(x_t)
        else:
            x_t = x_prev_mean

    x = (x_t.clamp(-1,1) + 1)/2
    return x

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--clf_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sample_steps", type=int, default=1000)
    p.add_argument("--cg_scale", type=float, default=0.0)
    p.add_argument("--cfg_scale", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse()
    device = torch.device(args.device)
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    sched = DiffusionSchedule(device=device)

    # model & classifier
    model = TinyUNet(base_ch=32, time_dim=64, num_classes=10, cond_drop_prob=0.1).to(device)
    clf = NoiseAwareClassifier(time_dim=64, num_classes=10).to(device)
    
    # train (or load) diffusion and noise-aware classifier --> This is up to you, you may decide to re-train if you want to
    diff_path = "step2_utils/mnist_diffusion.pt"
    clf_path  = "step2_utils/mnist_classifier.pt"
    _ = load_weights(model, diff_path, device)
    _ = load_weights(clf, clf_path, device)
    #train_diffusion(model, sched, train_loader, epochs=args.epochs, device=device, lr=2e-4)
    #save_weights(model, diff_path)
    #train_classifier(clf, sched, train_loader, epochs=args.clf_epochs, device=device, lr=2e-4, noise_prob=0.9)
    #save_weights(clf, clf_path)

    labels = torch.arange(0,10, device=device).repeat_interleave(10)
    imgs_cg  = sample_cg(model, clf, sched, labels, num_steps=args.sample_steps, guidance_scale=args.cg_scale, device=device)
    imgs_cfg = sample_cfg(model, sched, labels, num_steps=args.sample_steps, guidance_scale=args.cfg_scale, device=device)

    save_grid(imgs_cg,  f"results/step2_results/mnist_cg_{args.cg_scale}.png",  nrow=10)
    save_grid(imgs_cfg, f"results/step2_results/mnist_cfg_{args.cfg_scale}.png", nrow=10)

    # metrics
    acc_cg  = classifier_accuracy_on_samples(clf, imgs_cg,  labels, device)
    acc_cfg = classifier_accuracy_on_samples(clf, imgs_cfg, labels, device)
    div_cg  = intraclass_diversity_cosine(clf, imgs_cg, labels, device)
    div_cfg = intraclass_diversity_cosine(clf, imgs_cfg, labels, device)

    print("\n=== Results ===")
    print(f"Classifier Acc on CG  samples: {acc_cg:.3f} | Diversity: {div_cg:.3f}")
    print(f"Classifier Acc on CFG samples: {acc_cfg:.3f} | Diversity: {div_cfg:.3f}")

if __name__ == "__main__":
    main()
