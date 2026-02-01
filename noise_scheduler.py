import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM_Scheduler(nn.Module):
    """
    Variance scheduler decides the variance of the Gaussian noise that is added to the image at a given timestep. It is linearly spaced scheduled from 0.0001 to 0.02 at t=T
    """
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.num_time_steps = num_time_steps
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False) # amount of noise to be added
        alpha = 1. - self.beta
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # register buffer helper function
        # for more on register_buffer read here: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # Register variance schedule related buffers
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Buffer for diffusion calculations q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # Clipped because posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps=1e-20))

        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))    
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))    
    
    def noise_sample(self, x_start, t, noise=None):
        """
        Sample x_t from the forward diffusion process q(x_t | x_0) at timestep t
        using the closed-form equation:
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        noised = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return noised
    
    def sample_posterior(self, x_start: torch.tensor, x_t: torch.tensor, t: torch.tensor, **kwargs):
        """
        Compute posterior parameters (mean, variance, log variance) of
        q(x_{t-1} | x_t, x_0) used in the reverse diffusion process
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # Extract the value corresponding to the current time from the buffers, and then reshape to (b, 1, 1, 1)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Given a noised image x_t and its noise component `noise`, calculate the unnoised image x_0
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

