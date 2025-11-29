import torch
import step1_utils.utils as utils

class Sampler():
    def __init__(self, conf):
        self.conf = conf
        scale = 1000 / self.conf.diff_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.conf.diff_timesteps, dtype=torch.float64)
        self.alpha_init()
       
    def alpha_init(self):
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))
    
    def recreate_alphas(self):
        use_timesteps = utils.space_timesteps(self.conf.diff_timesteps, self.conf.schedule)
        self.timestep_map = []
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        self.betas = torch.tensor(new_betas)
        self.alpha_init()
        return torch.tensor(self.timestep_map)
    '''
    def predict_xstart(self, x_t, t, model_output):
        coeff1 = torch.sqrt(1/self.alphas_cumprod)
        coeff2 = torch.sqrt(1 - self.alphas_cumprod) / torch.sqrt(self.alphas_cumprod)
        coeff1 = utils.extract_and_expand(coeff1, t, x_t)
        coeff2 = utils.extract_and_expand(coeff2, t, model_output)
        return utils.clip_denoised(coeff1 * x_t - coeff2 * model_output)
    '''
    def sample_ddim(self, x_t, t, x0_hat, model_output):
        sigma = (self.conf.eta * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)) 
                 * torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod_prev)
        posterior_mean_coef2 = torch.sqrt(1.0 - self.alphas_cumprod_prev - sigma**2)
        coef1 = utils.extract_and_expand(posterior_mean_coef1, t, x0_hat)
        coef2 = utils.extract_and_expand(posterior_mean_coef2, t, model_output)
        sample = coef1 * x0_hat + coef2 * model_output
        noise = torch.randn_like(x_t)
        if t != 0:
            sample += utils.extract_and_expand(sigma, t, noise) * noise
        return sample