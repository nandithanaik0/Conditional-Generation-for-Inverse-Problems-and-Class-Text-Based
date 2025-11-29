import torch
import os
from step1_utils.models.unet import create_model
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import step1_utils.utils as utils
from step1_utils.DDIM_sampler import Sampler as DDIM
import argparse
import torchvision.transforms as transforms
from step1_utils.data.dataloader import get_dataset, get_dataloader
from step1_utils.degradations import GaussianNoise, get_degradation
import numpy as np

import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # hyperparameters for path & dataset
        self.parser.add_argument('--out_path', type=str, default='results/step1_results', help='results file directory')
        self.parser.add_argument('--dataset', type=str, default='CelebA_HQ', help='either choose CelebA_HQ or ImageNet')
        self.parser.add_argument('--sigma_y', type=float, default=0.0, help='measurement noise')
        
        # hyperparameters for sampling
        self.parser.add_argument('--diff_timesteps', type=int, default=1000, help='Original number of steps from Ho et al. (2020) which is 1000 - do not change')
        self.parser.add_argument('--desired_timesteps', type=int, default=1000, help='How many steps do you want?')
        self.parser.add_argument('--eta', type=float, default=1.0, help='Should be between [0.0, 1.0]')
        self.parser.add_argument('--schedule', type=str, default="1000", help="regular/irregular schedule to use (jumps)")
        
        # hyperparameters for algos
        self.parser.add_argument('--ps_type', type=str, default="ILVR", help="choose from projection, DPS, DDNM")
        self.parser.add_argument('--degradation', type=str, default='Inpainting', help='SR or Inpainting')
        
        # hyperparameters for the inpainting mask & SR
        self.parser.add_argument('--mask_type', type=str, default="box", help='box or random')
        self.parser.add_argument('--random_amount', type=float, default=0.8, help='how much do you want to mask out?')
        self.parser.add_argument('--box_indices', type=int, default=[30,30,128,128], help='inpainting box indices - (y,x,height,width)')
        self.parser.add_argument('--scale_factor', type=int, default=4, help='SR scale factor')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf
        # self.conf, _ = self.parser.parse_known_args(args=args)
        # return self.conf


class posterior_samplers():
    def __init__(self, conf, sampler_operator, score_model):
        self.conf = conf
        self.sampler_operator = sampler_operator
        self.score_model = score_model
    
    def predict_x0_hat(self, x_t, t, model_output):
        ############################
        # TODO: Implement the function predicting the clean denoised estimate x_{0|t}
        # Similar to HW3, use utils.extract_and_expand() function when necessary
        ############################
        alpha_bar_t = utils.extract_and_expand(self.sampler_operator.alphas_cumprod, t, x_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_hat = (x_t - model_output * sqrt_one_minus_alpha_bar_t) / sqrt_alpha_bar_t
        return utils.clip_denoised(x0_hat)
    
    def sample_ddim(self, x_t, t, x0_hat, model_output):
        ############################
        # TODO: Implement DDIM sampling
        ############################
        with torch.no_grad():
            eps = model_output

            alpha_bar_t = utils.extract_and_expand(self.sampler_operator.alphas_cumprod, t, x_t)
            alpha_bar_prev = utils.extract_and_expand(self.sampler_operator.alphas_cumprod_prev, t, x_t)

            eta = self.conf.eta

            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            )

            mean = torch.sqrt(alpha_bar_prev) * x0_hat + torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)
            ) * eps

            batch = t.shape[0]
            mask = (t > 0).float().view(batch, *([1] * (x_t.dim() - 1)))

            noise = torch.randn_like(x_t)
            x_t_prev = mean + sigma * noise * mask
            return x_t_prev
    
    def q_sample(self, data, t):
        ############################
        # TODO: Implement q(xt−1 | x0) = N (xt−1; √¯αt−1 x0, (1 − ¯αt−1)I)
        # Hint-1: Reparametrization Trick
        # Hint-2: You can get \bar{α}_{t−1} from --> self.sampler_operator.alphas_cumprod_prev
        ############################
        with torch.no_grad():
            alpha_bar_prev = utils.extract_and_expand(self.sampler_operator.alphas_cumprod_prev, t, data)
            mean = torch.sqrt(alpha_bar_prev) * data
            std = torch.sqrt(1.0 - alpha_bar_prev)
            noise = torch.randn_like(data)

            q_xt_x0 = mean + std * noise
            return q_xt_x0
    
    def ilvr(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement ILVR based on the HW PDF description
        
        # Hint-1: You can get the model output similar to HW3:
        # model_output = self.score_model(x_t, model_t)
        # model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        
        # Hint-2: A, A^T or A^\dagger operations can be performed by:
        # A_funcs.A(), A_funcs.At(), A_funcs.A_pinv()
        ############################
        with torch.no_grad():
            model_output = self.score_model(x_t, model_t)
            model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
            x0_hat = self.predict_x0_hat(x_t, t, model_output)
            x_t_prev = self.sample_ddim(x_t, t, x0_hat, model_output)

            # Sample y_{t-1} from q(y_{t-1} | y_0)
            y_t_minus_1 = self.q_sample(measurement, t)

    
            zeta_ilvr = 1.0  # tune this hyperparameter in experiments
            Ax_prev = A_funcs.A(x_t_prev)
            correction = A_funcs.A_pinv(y_t_minus_1 - Ax_prev)
            x_t_prev = x_t_prev + zeta_ilvr * correction
            return x_t_prev
    
    def mcg(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement MCG based on the HW PDF description
        ############################
        x_t_in = x_t.detach().requires_grad_(True)

        # Model prediction and Tweedie estimate
        model_output = self.score_model(x_t_in, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        x0_hat = self.predict_x0_hat(x_t_in, t, model_output)

        # Data-fidelity loss: ||y - A x0_hat||^2
        Ax0 = A_funcs.A(x0_hat)
        loss = torch.mean((Ax0 - measurement) ** 2)

        # ∇_{x_t} || y - A x0_hat ||^2
        grad = torch.autograd.grad(loss, x_t_in, retain_graph=False, create_graph=False)[0]

        # Unconditional DDIM/DDPM step (detach to avoid backprop through sampling)
        x_t_prev = self.sample_ddim(x_t.detach(), t, x0_hat.detach(), model_output.detach())

        # Gradient step along manifold
        zeta_mcg = 1.0  # tune this hyperparameter in experiments
        x_t_prev_tilde = x_t_prev - zeta_mcg * grad

        # Sample y_{t-1} same as ILVR
        y_t_minus_1 = self.q_sample(measurement, t)

        # Projection step using A^T
        Ax_tilde = A_funcs.A(x_t_prev_tilde)
        proj_correction = A_funcs.At(y_t_minus_1 - Ax_tilde)
        x_t_prev = x_t_prev_tilde + proj_correction

        return x_t_prev
    
    def ddnm(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement DDNM based on the HW PDF description
        ############################
        with torch.no_grad():
            model_output = self.score_model(x_t, model_t)
            model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
            x0_hat = self.predict_x0_hat(x_t, t, model_output)

            zeta_ddnm = 1.0 
            residual = measurement - A_funcs.A(x0_hat)
            x0_tilde = x0_hat + zeta_ddnm * A_funcs.A_pinv(residual)

            x_t_prev = self.sample_ddim(x_t, t, x0_tilde, model_output)

            return x_t_prev
    
    def dps(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement DPS based on the HW PDF description
        ############################
        x_t_in = x_t.detach().requires_grad_(True)

        # Model prediction and Tweedie estimate
        model_output = self.score_model(x_t_in, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        x0_hat = self.predict_x0_hat(x_t_in, t, model_output)

        Ax0 = A_funcs.A(x0_hat)
        loss = torch.mean((Ax0 - measurement) ** 2)

        grad = torch.autograd.grad(loss, x_t_in, retain_graph=False, create_graph=False)[0]

        # Unconditional DDIM/DDPM step
        x_t_prev = self.sample_ddim(x_t.detach(), t, x0_hat.detach(), model_output.detach())

        zeta_dps = 1.0  
        x_t_prev = x_t_prev - zeta_dps * grad

        return x_t_prev

def sample_unconditional(dataset_name, device, n_samples=5, timesteps=1000):
    
    conf = Config().parse(args=[
        "--dataset", dataset_name,
        "--desired_timesteps", str(timesteps),
        "--eta", "1.0",              # DDPM variant (η=1)
    ])


    model_config = utils.load_yaml(f"step1_utils/models/{conf.dataset}_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()

    
    sampler_operator = DDIM(conf)
    time_map = sampler_operator.recreate_alphas().to(device)
    ps_ops = posterior_samplers(conf, sampler_operator, score_model)

  
    x_t = utils.get_noise_x_t(device)          # [1, 3, H, W]
    x_t = x_t.expand(n_samples, *x_t.shape[1:])  # [N, 3, H, W]

    pbar = list(range(conf.desired_timesteps))[::-1]

    with torch.no_grad():
        for idx in tqdm(pbar):
            t = torch.full((x_t.shape[0],), idx, device=device, dtype=torch.long)  
            model_t = time_map[t]  
            model_output = score_model(x_t, model_t)
            model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)

            x0_hat = ps_ops.predict_x0_hat(x_t, t, model_output)
            x_t = ps_ops.sample_ddim(x_t, t, x0_hat, model_output)


    x_0 = x_t.detach().cpu()

    x_0 = (x_0 + 1) / 2
    x_0 = torch.clamp(x_0, 0.0, 1.0)

 
    imgs = x_0.permute(0, 2, 3, 1).numpy()
    return imgs

        
def main():
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conf = Config().parse()
    
    print('*' * 60 + f'\nSTARTED DDIM Sampling with eta = \"%.1f\" \n' %conf.eta)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #lpips metrics
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    # Create and config model
    model_config = utils.load_yaml("step1_utils/models/" + conf.dataset + "_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(conf.dataset, f"step1_utils/data/{conf.dataset}/", transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    noiser = GaussianNoise(conf.sigma_y*2.0)
    A_funcs = get_degradation(conf, device)
    
    # Sampling
    for i, ref_img in enumerate(loader):
        start_time = time.time()
        print(f'\nSampling for Image {i+1} has started!')
        sampler_operator = DDIM(conf)
        ref_img = ref_img.to(device)
        measurement = noiser(A_funcs.A(ref_img))
        x_t = utils.get_noise_x_t(device).requires_grad_()
        pbar = (list(range(conf.desired_timesteps))[::-1])
        time_map = sampler_operator.recreate_alphas().to(device)
        ps_ops = posterior_samplers(conf, sampler_operator, score_model)
        
        for idx in tqdm(pbar):
            time = torch.tensor([idx] * x_t.shape[0], device=device)
            if conf.ps_type == "ILVR":
                x_t_prev = ps_ops.ilvr(x_t, time, time_map[time], measurement, A_funcs)    
            elif conf.ps_type == "MCG":
                x_t_prev = ps_ops.mcg(x_t, time, time_map[time], measurement, A_funcs)    
            elif conf.ps_type == "DDNM":
                x_t_prev = ps_ops.ddnm(x_t, time, time_map[time], measurement, A_funcs)         
            elif conf.ps_type == "DPS":
                x_t_prev = ps_ops.dps(x_t, time, time_map[time], measurement, A_funcs)
            else:
                raise ValueError(f"Unknown ps_type: {conf.ps_type}")
            x_t = x_t_prev


        ##for metrics
        ref_np = utils.clear_color(ref_img)   # (H, W, 3) in [0,1]
        rec_np = utils.clear_color(x_t)       # (H, W, 3) in [0,1]

        psnr = peak_signal_noise_ratio(ref_np, rec_np, data_range=1.0)
        ssim = structural_similarity(ref_np, rec_np, channel_axis=2, data_range=1.0)

        # LPIPS 
        with torch.no_grad():
            lpips_val = lpips_fn(ref_img, x_t).item()

        elapsed = time.time() - start_time

        print(
            f"Image {i+1}: "
            f"PSNR={psnr:.2f}, SSIM={ssim:.4f}, "
            f"LPIPS={lpips_val:.4f}, time={elapsed:.2f}s"
        )

        image_filename = f"recon_{i+1}.png"
        
        image_path = os.path.join(conf.out_path, conf.dataset, conf.ps_type, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.imsave(image_path, np.concatenate([utils.clear_color(ref_img), utils.clear_color(A_funcs.A_pinv(measurement).reshape(1,3,256,256)), utils.clear_color(x_t)], axis=1))
    print('\nFINISHED Sampling!\n' + '*' * 60)

if __name__ == '__main__':
    main()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

celeba_imgs = sample_unconditional("CelebA_HQ", device, n_samples=5, timesteps=1000)

# 5 samples from ImageNet ADM
imagenet_imgs = sample_unconditional("ImageNet", device, n_samples=5, timesteps=1000)

print("CelebA imgs shape:", celeba_imgs.shape)    # should be (5, H, W, 3)
print("ImageNet imgs shape:", imagenet_imgs.shape)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for j in range(5):
    axes[0, j].imshow(celeba_imgs[j])
    axes[0, j].axis("off")

    axes[1, j].imshow(imagenet_imgs[j])
    axes[1, j].axis("off")

plt.tight_layout()
plt.savefig("unconditional_2x5_grid.png", dpi=200, bbox_inches="tight")
plt.show()
