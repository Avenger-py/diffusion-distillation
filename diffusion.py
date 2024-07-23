import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils import show_tensor_image, equally_spaced_items
from config import config


def cosine_schedule(num_timesteps, s=0.008):
    def f(t):
        return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, 0.0001, 0.999)
    return betas

def sample_t_index(val, t):
    out = val[t]
    return out[:, None, None, None]

class Diffusion():
    def __init__(self, timesteps=config.T, device=config.device):
        self.T = timesteps
        self.device = device
        self.betas = cosine_schedule(num_timesteps=self.T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    def forward_diffusion_sample(self, x_0, t):
        """Add noise to input"""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = sample_t_index(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = sample_t_index(self.sqrt_one_minus_alphas_cumprod, t)
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)
    
    def vis_forward_diffusion(self, image):
        """Display forward diffusion"""
        plt.figure(figsize=(15,2))
        plt.axis('off')
        num_images = 10
        stepsize = int(self.T/num_images)

        for idx in range(0, self.T, stepsize):
            t = torch.Tensor([idx]).type(torch.int64)
            plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
            img, noise = self.forward_diffusion_sample(image, t)
            show_tensor_image(img.cpu())
        plt.show()
    
    def get_loss(self, model, image, t, gamma=0.3):
        mse = nn.MSELoss()

        # Add noise to original image for timestep=t
        alpha_t = sample_t_index(self.sqrt_alphas_cumprod, t)
        sigma_t = sample_t_index(self.sqrt_one_minus_alphas_cumprod, t)
        eps = torch.randn_like(image, device=self.device)
        img_T = alpha_t * image.to(self.device) + sigma_t * eps

        # denoised prediction of the model
        img_pred = model(img_T, t)
        # print(img_pred.shape, image.shape)
        # return loss b/w original denoised image and predicted denoised image
        return mse(img_pred , image.to(self.device))


class Sampler:
    def __init__(self, sample_timesteps=config.sample_T, device=config.device):
        self.T = sample_timesteps
        self.device = device
        self.betas = cosine_schedule(num_timesteps=self.T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    # @torch.no_grad()
    # def sample_one_step_ddpm(self, model, x, t):
    #     betas_t = sample_t_index(self.betas, t)
    #     sqrt_one_minus_alphas_cumprod_t = sample_t_index(self.sqrt_one_minus_alphas_cumprod, t)
    #     sqrt_recip_alphas_t = sample_t_index(self.sqrt_recip_alphas, t)

    #     # Call model (current image - noise prediction)
    #     model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    #     if t.item() == 0:
    #         noise = 0
    #     else:
    #         noise = torch.randn_like(x)

    #     return model_mean + torch.sqrt(betas_t) * noise

    def sample_one_step_ddim(self, model, x_t, time_step, prev_time_step, eta=0.0, testing=False):
        x_t = x_t.to(self.device)
        # Generalised sampling, eta = 0.0 gives ddim, eta = 1.0 gives ddpm, Source: equation 12 from https://arxiv.org/pdf/2010.02502
        # For DDIM, eta = 1.0, therefore sigma_t = 0, this makes the sampling process deterministic and the samples are sharper
        alpha_t = sample_t_index(self.alphas_cumprod, time_step).to(self.device)
        alpha_t_prev = sample_t_index(self.alphas_cumprod, prev_time_step).to(self.device)
        x_theta_t = x_t.to(self.device) if testing else model(x_t, time_step)
        
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t, device=self.device)
         
        if time_step == 1:
            # x_t_minus_one = x_theta_t + sigma_t * epsilon_t # sigma_t = 0 for ddim
            x_t_minus_one = torch.sqrt(alpha_t_prev) * x_theta_t + sigma_t * epsilon_t
        else:
            x_t_minus_one = torch.sqrt(alpha_t_prev) * x_theta_t + torch.sqrt((1 - alpha_t_prev - sigma_t ** 2) / (1 - alpha_t)) * (x_t - torch.sqrt(alpha_t)*x_theta_t) + sigma_t*epsilon_t
            
        return x_t_minus_one
    
    
    @torch.no_grad()
    def sample_plot_image(self, model, sampling_steps, testing=False, return_last_img=False, img=None, sampling="DDIM", num_images=10, show=True):
        # assert sampling_steps < self.T
        
        if not testing and img == None:
            img_size = config.img_size
            img = torch.randn((1, 3, img_size, img_size), device=self.device)
        
        if show:
            plt.figure(figsize=(15,2))
            plt.axis('off')

        sampling_steps_list = [(self.T // sampling_steps) * i for i in range(sampling_steps)]

        if sampling == "DDIM":
            eta = 0.0
        if sampling == "DDPM":
            eta = 1.0

        rng = list(range(1, len(sampling_steps_list))[::-1])
        if num_images >= sampling_steps:
            num_images = sampling_steps - 1

        stepsize = len(rng) // num_images
        step_list = equally_spaced_items(rng, num_images)
        
        for i in rng:
            t = torch.full((1,), sampling_steps_list[i], device=self.device, dtype=torch.long)
            t_prev = torch.full((1,), sampling_steps_list[i - 1], device=self.device, dtype=torch.long)
            img = self.sample_one_step_ddim(model, img, t, t_prev, testing=testing, eta=eta)
            
            img = torch.clamp(img, -1.0, 1.0)
            if show:
                if i in step_list:
                    plt.subplot(1, num_images, step_list.index(i) + 1)
                    show_tensor_image(img.detach().cpu())
        if show:
            plt.show()
            
        if return_last_img:
            return img.detach().cpu()


def distill_loss(image, tlist, teacher_model, student_model, teacher_diffusion):    
    t, t_1, t_2 = tlist

    # Foraward diffusion
    alpha_t = sample_t_index(teacher_diffusion.sqrt_alphas_cumprod, t).to(config.device)
    sigma_t = sample_t_index(teacher_diffusion.sqrt_one_minus_alphas_cumprod, t).to(config.device)
    eps = torch.randn_like(image, device=config.device)
    img_T = alpha_t * image.to(config.device) + sigma_t * eps
    
    # 2 steps of reverse diffusion using teacher model
    # t --> t_1
    alpha_t_1 = sample_t_index(teacher_diffusion.sqrt_alphas_cumprod, t_1)
    sigma_t_1 = sample_t_index(teacher_diffusion.sqrt_one_minus_alphas_cumprod, t_1)
    img_T_hat = teacher_model(img_T, t)
    img_T_1 = alpha_t_1 * img_T_hat + (sigma_t_1/sigma_t) * (img_T - alpha_t * img_T_hat)

    # t_1 --> t_2
    alpha_t_2 = sample_t_index(teacher_diffusion.sqrt_alphas_cumprod, t_2)
    sigma_t_2 = sample_t_index(teacher_diffusion.sqrt_one_minus_alphas_cumprod, t_2)
    img_T_1_hat = teacher_model(img_T_1, t_1)
    img_T_2  = alpha_t_2 * img_T_1_hat + (sigma_t_2/sigma_t_1) * (img_T_1 - alpha_t_1 * img_T_1_hat)

    # Target for student model
    img_target = (img_T_2 - (sigma_t_2/sigma_t) * img_T) / (alpha_t_2 - (sigma_t_2/sigma_t) * alpha_t)

    # 1 step of reverse diffusion using student model
    t_s = t
    img_pred = student_model(img_T, t_s).to(config.device)
    
    # Compute loss
    k = alpha_t**2 / sigma_t**2 
    lambda_t = torch.clamp(k, max=1, min=0.01).to(config.device)
    loss = F.mse_loss(img_pred * lambda_t, img_target.to(config.device) * lambda_t)
    return loss
