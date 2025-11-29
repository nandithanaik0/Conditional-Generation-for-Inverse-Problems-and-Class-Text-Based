import numpy as np
import torch
import yaml

def get_noise_x_t(device):
    #torch.manual_seed(0)
    x_t = torch.randn((1,3,256,256), device=device)
    return x_t

def clip_denoised(x):
    return x.clamp(-1, 1)

def extract_and_expand(array, time, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(target.device)[time].float()
    else:
        array = array.to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)

def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target).to(target.device)
    
def normalize_np(img):
    img -= np.min(img)
    img /= np.max(img)
    return img

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.clip(-1,1)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))
    
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
    
def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)