import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc
import math
import copy
from .latents import initialize_or_scale
import latent_preview


import torch.nn.functional as F
import torchvision.transforms as T

import functools

from .noise_classes import *

import comfy.model_patcher
import comfy.sample
import comfy.samplers


from .extra_samplers_helpers import get_deis_coeff_list


from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3


def phi(j, neg_h):
  remainder = torch.zeros_like(neg_h)
  
  for k in range(j): 
    remainder += (neg_h)**k / math.factorial(k)
  phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
  
  return phi_j_h
  
  
def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))

def epsilon(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    data = model(x, sigma * s_in, **extra_args)
    eps = (x - data) / (sigma * s_in) 
    return eps, data


def epsilon_res(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    data = model(x, sigma * s_in, **extra_args)
    eps = data - x
    return eps, data

from .helper import slerp

def sample_rk_sphere(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        if sigma == 1.0:
            sigma = torch.full_like(sigma, 0.9999)
        
        t, t_next = t_fn(torch.clamp(sigma, max=0.9999)), t_fn(sigma_next)
        h = t_next - t
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        
        t_down = t_fn(sigma_down)
        
        h = t_down - t
        
        sigma_s = sigma_fn(t + h*c2)
        
        h = -torch.log(sigma_down/sigma)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_down > 0:
            x_2 = torch.exp(-h * c2) * x + h * (a2_1 * denoised)
            
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)
            
            x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_2)
            
            x = slerp(denoised_2, x, (sigma_next/sigma))
            #x = slerp(h*(b1 * denoised + b2 * denoised_2), x, (sigma_next/sigma)**c2)
            #x = slerp((sigma_next-sigma)*(b1 * denoised + b2 * denoised_2), x, (sigma_next/sigma))
            #x = slerp(h*(b1 * denoised + b2 * denoised_2), x, h**c2)
            
            if callback is not None:
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised



from .rk_guide_func import get_collinear, get_orthogonal


def sample_rk_momentum(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    eps_prev, m_t, v_t, m_t_prev, v_t_prev = [torch.zeros_like(x) for _ in range(5)]

    momentum = float(get_extra_options_kv("momentum", "0.0", extra_options))
    beta1    = float(get_extra_options_kv("beta1", "0.0", extra_options))
    beta2    = float(get_extra_options_kv("beta2", "0.0", extra_options))

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        
        h = sigma_down - sigma

        denoised = model(x, sigma * s_in, **extra_args)

        eps = (x - denoised) / sigma
        s_theta = -eps  

        if step == 0:
            eps_prev = eps.clone()

        
        eps_collin = get_collinear (eps, eps_prev)     
        eps_ortho  = get_orthogonal(eps, eps_prev)

        #eps_ortho  = get_orthogonal(eps, eps_collin)
        
        eps_prev_collin = get_collinear (eps_prev, eps)     
        eps_prev_ortho  = get_orthogonal(eps_prev, eps)

        #eps_ortho  = get_orthogonal(eps_prev, eps_collin)

        
        #
        # eps = eps + beta2 * (eps_collin - eps_ortho)
        
        #eps = beta2 * eps + (1-beta2) * (eps_collin + eps_prev_ortho)
        
        eps = (1-beta1) * eps + beta1 * eps_prev

        eps = (1-beta2) * eps + beta2 * (eps_prev_collin + eps_ortho)

            
        x = x + h * eps

        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / (noise.std())
        
        x = alpha_ratio * x + sigma_up * noise
        
        eps_prev = eps
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised





def sample_rk_ralston_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    

    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        
        h = sigma_down - sigma
        
        a2_1 = 2/3
        b1, b2 = 1/4, 3/4
        c1, c2 = 0.0, 2/3
        
        a2_1 = 1.0
        b1, b2 = 1/2, 1/2
        c1, c2 = 0.0, 1.0
        
        sigma2 = sigma + h * c2
        
        #h2 = sigma2 - sigma
        
        sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma2, eta, eta_var, noise_mode)

        h2 = sub_sigma_down - sigma

        denoised = model(x, sigma * s_in, **extra_args)

        eps = (x - denoised) / sigma

        x2_eps = x + h2 * (a2_1 * eps)
        
        x2 = (sub_sigma_down/sigma) * x + (1 - sub_sigma_down/sigma) * denoised
        
        print("x2_eps - x2", torch.norm(x2 - x2_eps).item())
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma2)
        noise = (noise - noise.mean()) / (noise.std())
        
        x2 = sub_alpha_ratio * x2 + sub_sigma_up * noise
        
        

        denoised2 = model(x2, sigma2 * s_in, **extra_args)
        eps2 = (x2 - denoised2) / sigma2
        eps2 = (x - denoised2) / sigma
        
        x_eps = x + h * (b1 * eps + b2 * eps2)
        
        x = (sigma_down/sigma) * x + (1 - sigma_down/sigma) * (0.5 * denoised + 0.5 * denoised2)
        
        print("x_eps - x", torch.norm(x - x_eps).item())
        
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / (noise.std())
        
        x = alpha_ratio * x + sigma_up * noise
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def sample_rk_implicit_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    reverse_weight = float(get_extra_options_kv("reverse_weight", str(0.0), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
                
        h = -torch.log(sigma_down/sigma)
        
        #s = t + h * c2
        #sigma_s = sigma_fn_x(s)
        s2 = -torch.log(sigma) + h * c2
        sigma_2 = torch.exp(-s2)
        #sigma_2 = torch.exp(-h * c2)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)
            return denoised
            
        #denoised = model(x, sigma * s_in, **extra_args)
        eps = (denoised - x)
        x_2 = x + h * (a2_1 * eps)
        
        for i in range(iter):
            denoised = model(x_2, sigma_2 * s_in, **extra_args)
            eps = (denoised - x)
            x_2 = x + h * (a2_1 * eps)

        denoised_2 = model(x_2, sigma_2 * s_in, **extra_args)
        eps_2 = (denoised_2 - x)
        x_next = x + h * (b1 * eps + b2 * eps_2)
        
        for i in range(iter):
            denoised_2 = model(x_next, sigma_2 * s_in, **extra_args)
            eps_2 = (denoised_2 - x)
            x_next = x + h * (b1 * eps + b2 * eps_2)

        x = x_next
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised








def sample_rk_implicit_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    reverse_weight = float(get_extra_options_kv("reverse_weight", str(0.0), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        
        h = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        x_next = x + h * eps
         
        
        for i in range(iter):
            denoised = model(x_next, sigma_next * s_in, **extra_args)
            eps = (x - denoised) / sigma
            x_next_new = x + h * eps
            
            x_reverse_new = (x_next - h*denoised) / (sigma_down/sigma)
            x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
            
            x_next = x_next_new

        x = x_next
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised



from . import res

def default_noise_sampler(x, seed=None):
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    x0_func = lambda x, sigma: model(x, sigma, **extra_args)
    solver_cfg = res.SolverConfig()
    solver_cfg.s_churn = s_churn
    solver_cfg.s_t_max = s_tmax
    solver_cfg.s_t_min = s_tmin
    solver_cfg.s_noise = s_noise
    x = res.differential_equation_solver(x0_func, sigmas, solver_cfg, noise_sampler, callback=callback, disable=disable)(x)
    return x




def sample_rk_implicit_euler_reverse_weight_fail(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    reverse_weight = float(get_extra_options_kv("reverse_weight", str(0.0), extra_options))
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        
        h = sigma_down - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        x_next = x + h * eps
         
        
        for i in range(iter):
            denoised = model(x_next, sigma_next * s_in, **extra_args)
            eps = (x - denoised) / sigma
            x_next_new = x + h * eps
            
            x_reverse_new = (x_next - h*denoised) / (sigma_down/sigma)
            x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
            
            x_next = x_next_new

        x = x_next
        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised








def sample_rk_momentum_adam(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    eps_prev, m_t, v_t, m_t_prev, v_t_prev = [torch.zeros_like(x) for _ in range(5)]

    momentum = float(get_extra_options_kv("momentum", "0.0", extra_options))
    beta1    = float(get_extra_options_kv("beta1", "0.0", extra_options))
    beta2    = float(get_extra_options_kv("beta2", "0.0", extra_options))

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        
        h = sigma_down - sigma

        denoised = model(x, sigma * s_in, **extra_args)

        eps = (x - denoised) / sigma
        s_theta = -eps  

        if step == 0:
            m_t_prev = eps.clone()
            v_t_prev = eps**2
            
        m_t = beta1 * m_t_prev + (1-beta1) * eps
        v_t = beta2 * v_t_prev + (1-beta2) * eps**2
        
        v_t = torch.clamp(v_t, min=1e-8)  #this is probably nonsense for sampling. explosive noise growth

        eps = m_t / torch.sqrt(v_t)
            
        x = x + h * eps

        m_t_prev, v_t_prev = m_t, v_t
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / (noise.std())
        
        x = alpha_ratio * x + sigma_up * noise
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised






def sample_rk_vpsde(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    

    if extra_args is None:
        extra_args = {}

    # If you have some noise_sampler object, otherwise just do:
    if noise_sampler is None:
        def noise_sampler(sigma, sigma_next):
            # Basic Gaussian
            return torch.randn_like(x)
    
    # We'll do an Euler step for each pair (sigma_i, sigma_{i+1})
    for step in trange(len(sigmas)-1, disable=disable):
        sigma_i = sigmas[step]
        sigma_next = sigmas[step+1]

        # 1) Get the model's predicted "denoised" image
        denoised = model(x, sigma_i, **extra_args)

        # 2) Convert that to an epsilon-prediction and the score:
        #    eps = (x - denoised) / sigma_i
        #    score = - eps / sigma_i, but we often just write s_theta = - eps
        eps = (x - denoised) / sigma_i
        s_theta = -eps  # i.e. approximate ∇ log p_t(x)

        # 3) We'll define the discrete "beta_t" as the difference in sigma^2
        #    for a variance-preserving approach:
        beta_t = sigma_i**2 - sigma_next**2
        # This acts like ∫ beta(t) dt from t_i to t_{i+1} in continuous time.

        # 4) The drift has two parts:
        #    (a) -1/2 * beta_t * x    (Ornstein–Uhlenbeck shrink of x)
        #    (b) -    beta_t * s_theta (the "score" drift)
        drift = -0.5 * beta_t * x + (-beta_t) * s_theta

        # 5) The diffusion term is sqrt(beta_t)
        #    We also typically add Gaussian noise here
        noise = noise_sampler(sigma_i, sigma_next)
        noise = (noise - noise.mean()) / (noise.std() + 1e-20)  # optional "normalize"

        # Euler–Maruyama step:
        x = x + drift + torch.sqrt(torch.clamp(beta_t, min=1e-20)) * noise

        # If callback is needed
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma_i, 'sigma_next': sigma_next, 'denoised': denoised})
            


    # after final step, we expect x to be a (nearly) denoised sample
    return x
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #beta_t = sigma_next - sigma
        #beta_t = torch.sqrt(sigma**2 - sigma_next**2)
        #alpha_t = (1 - sigma) / (1 - sigma_next)
        #alpha_t = (1 - sigma**2) / (1 - sigma_next**2)
        #beta_t = 1 - alpha_t
        
        #beta_t = sigma_next / sigma
        
        #beta_t = 1 / (1 - sigma**2)
        #beta_t = sigma**2 - sigma_next**2
        #beta_t = sigma - sigma_next
        
        
        #for DDPM
        #alpha_bar_t = 1 / (1 + sigma**2)
        #alpha_bar_t_next = 1 / (1 + sigma_next**2)
        #alpha_t = alpha_bar_t / alpha_bar_t_next
        #beta_t = 1 - (1 + sigma_next**2) / (1 + sigma**2)
        
        

        #beta_t = sigma_next #maybe???
        
        beta_t = torch.sqrt(sigma**2 - sigma_next**2)
        
        #delta_t = sigma_next - sigma
        delta_t = sigma - sigma_next
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised) / sigma
        s_theta = -eps

        #s_theta = -1 * eps / sigma
        
        #s_theta = -(x - alpha_t * denoised) / (sigma**2)
        
        
        #s_theta = -1 * eps / beta_t
        

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        
        beta_t = torch.sqrt(-2 * sigma * sigma - sigma**2)
        delta_t = -sigma + torch.sqrt(sigma**2 - beta_t**2)
        
        x = x + ((1/2) * beta_t * x   -   beta_t * eps) * delta_t   # +   torch.sqrt(beta_t * delta_t) * noise
        
        
        
        
        #alpha_coeff = 2 - torch.sqrt(1 - beta_t)
        #x_next = alpha_coeff * x   +  (1/2) * beta_t * s_theta   
        #x_next = x_next + torch.sqrt(beta_t) * noise
        
        #x = x_next_noised
        
        #x = x_next #ROCHEURCHOIRCHOEURCHOEURCHOERCUHROCEHURCOEHURCOEHURCOHEURCHOERCUHORECUHRCOEUHRCOHEU
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised



def sample_rk_vpsde_csbw(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        eps = (x - denoised) / sigma
        s_theta = -1 * eps #/ sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        #alpha_coeff = 1 - sigma_next + torch.sqrt(sigma_next**2 - (sigma_next/2)**2)

        #x_next = alpha_coeff * (x + (sigma-(torch.sqrt(sigma_next**2 - (sigma_next/2)**2))/(1 - sigma_next + torch.sqrt(sigma_next**2 - (sigma_next/2)**2))) * eps) + (sigma_next/2) * noise
        #x = x_next
        
        #x = (sigma_next**2 / sigma**2) * x + (1 - (sigma_next**2 / sigma**2)) * denoised
        
        #x = x + (sigma**2 - sigma_next**2) * s_theta + torch.sqrt(  (sigma_next**2 * (sigma**2 - sigma_next**2))   /   sigma**2  ) * noise
        
        
        #x = x + torch.sqrt(  (sigma_next**2 * (sigma**2 - sigma_next**2))   /   sigma**2  ) * noise
        
        
        #x = x + (sigma**2 - sigma_next**2) * (-eps)   + torch.sqrt(  (sigma_next**2 * (sigma**2 - sigma_next**2))   /   sigma**2  ) * noise
        
        #h = sigma - sigma_next
        
        #alpha_t = h**2 - 2*h + 1 
        
        #x = torch.sqrt(alpha_t) * x   +   h * s_theta   +   torch.sqrt(1 - alpha_t) * noise
        
        
        #x = torch.sqrt(sigma_next**2 / sigma**2) * x   +    (1 - torch.sqrt(sigma_next**2 / sigma**2)) * s_theta  # +   torch.sqrt(sigma**2 - sigma_next**2) * noise 
        
        
        #x =  x   +    (1 - torch.sqrt(sigma_next**2 / sigma**2)) * s_theta
        
        #x = x + h * (-eps)   + torch.sqrt(2 *h ) * noise
        
        h = sigma - sigma_next

        sigma_up = sigma_next * eta 
        sigma_signal = 1 - sigma_next
        sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)
        
        sigma_residual = sigma_next * (1 - eta**2)**0.5
        
        eta_root = (1 - eta**2)**0.5
        
        
        
        sigma_residual = sigma_next * eta_root

        alpha_ratio = sigma_signal + sigma_residual
        sigma_down = sigma_residual / alpha_ratio
            
        
        alpha_ratio = (1 - sigma_next)   +   sigma_next * eta_root
        
        
        alpha_ratio = 1  +   sigma_next * eta_root - sigma_next
        
        alpha_ratio = (1  +   sigma_next * (eta_root - 1))


        
        #x = x + (sigma - sigma_down) * s_theta
        
        #x = alpha_ratio * x + sigma_up * noise
        
        
        #x = alpha_ratio * (x + (sigma - sigma_down) * s_theta) + sigma_up * noise
        
        
        #x = x    +    (torch.sqrt(sigma_next**2 - sigma_up**2) - sigma_next) * x    +   (1 + torch.sqrt(sigma_next**2 - sigma_up**2) - sigma_next) * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        
        #x = x    +    (sigma_residual - sigma_next) * x    +   (1 + sigma_residual - sigma_next) * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        #x = x    +    (sigma_residual - sigma_next) * x    +   alpha_ratio * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        #x = x    +    (alpha_ratio - 1) * x    +   alpha_ratio * (sigma - sigma_residual / alpha_ratio) * s_theta    +    sigma_up * noise
        
        
        
        #x = x    +    (alpha_ratio - 1) * x    +   (sigma * alpha_ratio - sigma_residual) * s_theta    +    sigma_up * noise
        
        
        #x = x    +    (alpha_ratio - 1) * x    +   (sigma * alpha_ratio - sigma_residual) * s_theta    +    sigma_up * noise
        
        #x = x    +     sigma_next * (eta_root - 1) * x    +   (sigma *  (1  +   sigma_next * (eta_root - 1)) - sigma_next * eta_root) * s_theta    +    sigma_up * noise
        
        x = x    +     sigma_next * (eta_root - 1) * x    +   (sigma *  (1  +   sigma_next * (eta_root - 1)) - sigma_next * eta_root) * s_theta    +    sigma_next * eta * noise


        

        
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised





def sample_rk_vpsde_idk(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #beta_t = sigma_next - sigma
        #beta_t = sigma**2 - sigma_next**2
        #beta_t = sigma - sigma_next
        alpha_t = (1 - sigma) / (1 - sigma_next)
        #alpha_t = (1 - sigma**2) / (1 - sigma_next**2)
        beta_t = 1 - alpha_t
        
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised
        
        eps = (x - denoised) / sigma
        s_theta = -1 * eps / sigma
        
        #s_theta = -(x - alpha_t * denoised) / (sigma**2)
        
        #s_theta = -eps
        
        #s_theta = -1 * eps / beta_t
        
        alpha_coeff = 2 - torch.sqrt(1 - beta_t)
        x_next = alpha_coeff * x   +   beta_t * s_theta
        


        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x_next_noised = x_next + torch.sqrt(beta_t) * noise
        
        x = x_next_noised
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised






from tqdm import trange

def sample_rk_vpsde_trivial(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):  
      
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(
        x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0
    )
    
    for step in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step + 1]
        
        # Model predicts noise (epsilon_theta)
        denoised = model(x, sigma * s_in, **extra_args)
        eps = (x - denoised) / sigma
        
        if sigma_next == 0:
            return denoised  # Return final denoised image

        # Compute x_t-1 using VP-SDE equation
        scale_factor = sigma_next / sigma  # Scales the update step
        x_pred = x + (sigma_next**2 - sigma**2) * eps  # Drift term
        
        # Add stochastic noise for VPSDE
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()  # Normalize
        x_next = x_pred + scale_factor * noise  # Stochastic step
        
        x = x_next  # Update state

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
    
    return x  # Return final state



def sample_rk_vpsde_trivial_old(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised

        eps = (x - denoised) / sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()

        #x = denoised + sigma_next * noise
        x = x + (sigma_next - sigma) * eps
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised






def sample_rk_vpsde_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="", 
                  cfg1=0, cfg2=0, cfg_cw=1.0, latent_guide=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        

        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_next == 0:
            return denoised

        eps = (x - denoised) / sigma

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()

        x_next = torch.sqrt(sigma_next**2 + 1) / torch.sqrt(sigma**2 + 1)   *   (x - (sigma / torch.sqrt(sigma**2 + 1))*eps )  +  sigma_next * noise
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})


    return denoised





def sample_rk_logit(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
        
    if extra_options_flag("logit", extra_options):
        sigma_fn = lambda t: (t.exp() + 1) ** -1
        t_fn = lambda sigma: ((1-sigma)/sigma).log()
    if extra_options_flag("logsnr", extra_options):
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        if sigma == 1.0:
            sigma = torch.full_like(sigma, 0.9999)
        
        t, t_next = t_fn(torch.clamp(sigma, max=0.9999)), t_fn(sigma_next)
        h = t_next - t
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        
        t_down = t_fn(sigma_down)
        
        h = t_down - t
        
        sigma_s = sigma_fn(t + h*c2)
        
        h = -torch.log(sigma_down/sigma)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        denoised = model(x, sigma * s_in, **extra_args)
        if sigma_down > 0:
            x_2 = torch.exp(-h * c2) * x + h * (a2_1 * denoised)
            
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)
            
            x = torch.exp(-h) * x + h * (b1 * denoised + b2 * denoised_2)
            
            if callback is not None:
                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised





class Zampler_Test:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                     "eta_var": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculate variance-corrected noise amount (overrides eta/noise_mode settings). Cannot be used at very low sigma values; reverts to eta/noise_mode for final steps."}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False, "tooltip": "Fractal noise mode: <0 = extra high frequency noise, >0 = extra low frequency noise, 0 = white noise."}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False, "tooltip": "Fractal noise mode: all that matters is positive vs. negative. Effect unclear."}),
                     "cfg1": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Sample CFG."}),
                     "cfg2": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Unsample CFG."}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                     "noise_mode": (["hard", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                     "iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                     "sub_iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    },
                    "optional": 
                    {
                        "latent_guide": ("LATENT",),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1.0, cfg1=1.0, cfg2=1.0, buffer=0, extra_options="", noise_sampler_type="gaussian", noise_mode="hard",
                    rk_type="dormand-prince", t_fn_formula=None, sigma_fn_formula=None, iter=0, sub_iter=0, latent_guide=None,
                    ):
        sampler_name = "zample"
        
        sampler_name = get_extra_options_kv("sampler_name", "zample", extra_options)

        steps = 10000

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "alpha": alpha, "k": k, "cfg1": cfg1, "cfg2": cfg2, "buffer": buffer, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "rk_type": rk_type, 
                                                         "iter": iter,"sub_iter": sub_iter, "latent_guide": latent_guide, "extra_options": extra_options})
        return (sampler, )







def sample_zample_edit(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
    sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(x.dtype)
    
    y0 = latent_guide = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
    x  = y0.clone()
    
    noise = noise_sampler(sigma=1.0, sigma_next=sigma_min)
    noise = (noise - noise.mean()) / noise.std()
    x_noise = noise
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        #noise = noise_sampler(sigma=sigma, sigma_next=sigma_min)
        #noise = (noise - noise.mean()) / noise.std()
        
        #y = x + eta * (y0 - x)
        #y = y0.clone()
        
        y_adj = sigma * (noise - y0)

        y0_noised = y0 + sigma * (noise - y0)
        eps_y, data_y = epsilon(model, y0_noised, sigma, **extra_args)

        #x_hat     = x  + sigma * (noise - eta * y0 - (1-eta) * x)
        x_hat     = x  + eta * sigma * (noise - y0)
        #x_hat = (x - sigma*y0) + sigma*noise
        eps_x, data_x = epsilon(model, x_hat, sigma, **extra_args)
        
        x_next = x + h * (eps_x - eta * eps_y)

        denoised = data_x
        
        if extra_options_flag("x_hat", extra_options):
            x_next = x + h * eps_x
        elif extra_options_flag("y_noised", extra_options):
            x_next = x + h * eps_y
        if extra_options_flag("denoised_y", extra_options):
            denoised = data_y
        elif extra_options_flag("denoised_x", extra_options):
            denoised = data_x

        x = x_next
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised








def sample_zample_edit2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    y0 = latent_guide['samples'].to(x.device).to(x.dtype)
    x  = y0.clone()
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(x.dtype)
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_min)
        noise = (noise - noise.mean()) / noise.std()
        #x = alpha_ratio * x + noise * s_noise * sigma_up
        
        z_src = (1-sigma)*y0 + sigma*noise
        
        z_tar = x + (z_src - y0)
        
        denoised_tar = model(z_tar, sigma * s_in, **extra_args)
        eps_tar = (z_tar - denoised_tar) / sigma
        
        denoised_src = model(z_src, sigma * s_in, **extra_args)
        eps_src = (z_src - denoised_src) / sigma
        
        x = x + h * (eps_tar - eps_src)
        
        if extra_options_flag("denoised_tar", extra_options):
            denoised = denoised_tar
        else:
            denoised = denoised_src

        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised









def sample_zample_inversion(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    uncond = [0]
    uncond[0] = torch.full_like(x, 0.0)
    def post_cfg_function(args):
        uncond[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = cfg_fn(denoised, uncond[0], cfg1)
        eps = (x - denoised) / sigma
        
        if sigma_down > 0:
            #x_down = torch.sqrt(sigma_down) * (x - torch.sqrt(1-sigma) * eps) / torch.sqrt(sigma)   +   torch.sqrt(1 - sigma_down) * eps
            x_down = x + h * eps
            
            denoised_down = model(x_down, sigma_down * s_in, **extra_args)
            denoised_down = cfg_fn(denoised_down, uncond[0], cfg2)
            eps_down = (x_down - denoised_down) / sigma
            
            #x_next = torch.sqrt(sigma / sigma_down) * x_down   +   torch.sqrt(sigma) * (torch.sqrt(1/sigma - 1) - torch.sqrt(1/sigma_down - 1)) * eps_down
            
            x_next = x_down + (sigma_next - sigma_down) * eps_down

            x = x_next
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def cfg_fn(cond, uncond, scale):
    return cond + scale * (cond - uncond) 


def sample_zample_paper(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    uncond = [0]
    uncond[0] = torch.full_like(x, 0.0)
    def post_cfg_function(args):
        uncond[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = cfg_fn(denoised, uncond[0], cfg1)
        eps = (x - denoised) / sigma
        
        if sigma_down > 0:
            #x_down = torch.sqrt(sigma_down) * (x - torch.sqrt(1-sigma) * eps) / torch.sqrt(sigma)   +   torch.sqrt(1 - sigma_down) * eps
            x_down = torch.sqrt(sigma) * (x - torch.sqrt(1-sigma_down) * eps) / torch.sqrt(sigma_down)   +   torch.sqrt(1 - sigma) * eps

            #x_down = x + h * eps
            
            denoised_down = model(x_down, sigma_down * s_in, **extra_args)
            denoised_down = cfg_fn(denoised_down, uncond[0], cfg2)
            eps_down = (x_down - denoised_down) / sigma
            
            #x_next = torch.sqrt(sigma / sigma_down) * x_down   +   torch.sqrt(sigma) * (torch.sqrt(1/sigma - 1) - torch.sqrt(1/sigma_down - 1)) * eps_down
            x_next = torch.sqrt(sigma_down / sigma) * x_down   +   torch.sqrt(sigma_down) * (torch.sqrt(1/sigma_down - 1) - torch.sqrt(1/sigma - 1)) * eps_down

            #x_next = x_down + (sigma_next - sigma_down) * eps_down

            x = x_next
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

    return denoised




def sample_zsample(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfg1=1.0, cfg2=1.0, iter=0, sub_iter=0, reverse_weight=0.0, extra_options="",  latent_guide=None,):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    uncond = [0]
    uncond[0] = torch.full_like(x, 0.0)
    def post_cfg_function(args):
        uncond[0] = args["uncond_denoised"]
        return args["denoised"]
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    #sigma_fn = lambda t: t.neg().exp()
    #t_fn = lambda sigma: sigma.log().neg()
    
    sigma_fn = lambda t: t
    t_fn = lambda sigma: sigma

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h)
        t_down = t_fn(sigma_down)
        h = t_down - t
        
        
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = cfg_fn(denoised, uncond[0], cfg1)
        eps = (x - denoised) / sigma
        
        if sigma_down > 0:
            #x_down = torch.sqrt(sigma_down) * (x - torch.sqrt(1-sigma) * eps) / torch.sqrt(sigma)   +   torch.sqrt(1 - sigma_down) * eps
            x_down = x + h * eps
            
            denoised_down = model(x_down, sigma_down * s_in, **extra_args)
            denoised_down = cfg_fn(denoised_down, uncond[0], cfg2)
            eps_down = (x_down - denoised_down) / sigma
            
            #x_next = torch.sqrt(sigma / sigma_down) * x_down   +   torch.sqrt(sigma) * (torch.sqrt(1/sigma - 1) - torch.sqrt(1/sigma_down - 1)) * eps_down
            
            x_next = x_down + (sigma_next - sigma_down) * eps_down

            x = x_next
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

            #noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            #noise = (noise - noise.mean()) / noise.std()
            #x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised







def sample_rk_test3(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    x_2 = x + h * (11/200) * k1
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * (2/3), **extra_args)
    
    xp  =  ((1/4)*k1 + (1/4)*k2)
    xpp =  ((1/4)*k1 + (3/4)*k2)
    #xp = k1.clone() #* (sigmas[1] - sigmas[0])
    #xpp = k1.clone() * (sigmas[1] - sigmas[0])
    #xpp = k1.clone() * (sigmas[1] - sigmas[0])

    h = -torch.log(sigmas[1] / sigmas[0])
    k1, data1 = epsilon_res(model, x, sigmas[0], **extra_args)
    a2_1 = c2 * phi(1, -h*c2)
    x_2 = x + h * a2_1 * k1
    k2, data2 = epsilon_res(model, x_2, sigmas[0] + h * c2, **extra_args)

    xp, xpp = (k1.clone() for _ in range(2))
    xpp = (k2 - k1) 
    
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        #h = sigma_down - sigma
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, -torch.log(sigma_next/sigma))
        h = -torch.log(sigma_down/sigma)
        
        a2_1 = c2 * phi(1, -h*c2)
        b1 =        phi(1, -h) - phi(2, -h)/c2
        b2 =        phi(2, -h)/c2
                
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:
            
            
            
            #k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            #x_2 = x   +   h*(2/3)*xp   +   ((h**2)/2)*((2/3)**2) * xpp   +   (h**3) * ((11/200)*k1)
            #k2, data2 = epsilon(model, x_2, sigma + h*(2/3), **extra_args)
            #x   =   x + h*xp + ((h**2)/2)*xpp + (h**3) * ((1/8)*k1 + (1/24)*k2)
            
            k1, data1 = epsilon_res(model, x, sigma, **extra_args)
            x_2 = x   +   h*c2*xp   +   ((h**2)/2)*(c2**2) * xpp   +   (h**3) * (a2_1*k1)
            k2, data2 = epsilon_res(model, x_2, sigma + h*c2, **extra_args)
            x   =   x + h*xp + ((h**2)/2)*xpp + (h**3) * (b1*k1 + b2*k2)
            
            #xp = k1.clone()
            #xpp = (k2 - k1) 
            
            xp  =  xp + h*xpp + h**2 * (b1*k1 + b2*k2)
            xpp = xpp + h * (b1*k1 + b2*k2)
            
            #xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
            
            #xpp = (k2 - k1) / h
            #xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)


            
            denoised = data2
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised


from .helper import get_extra_options_kv, extra_options_flag






def sample_rk_test6th(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    a2_1 = 7/120 - (3 * 15**0.5)/200
    
    a3_1 = -1/96 + (15**0.5)/480
    a3_2 = 1/32 - (15**0.5)/480
    
    a4_1 = -1/600 + (15**0.5)/600
    a4_2 = (15**0.5)/50
    a4_3 = 3/50 - (15**0.5)/150
    
    b1 = 0
    b2 = 1/18 + (15**0.5)/72
    b3 = 1/18
    b4 = 1/18 - (15**0.5)/72
    
    b1p = 0
    b2p = 5/36 + (15**0.5)/36
    b3p = 2/9
    b4p = 5/36 - (15**0.5)/36
    
    b1pp = 0
    b2pp = 5/18
    b3pp = 4/9
    b4pp = 5/18
    
    c2 = 1/2 - (15**0.5)/10
    c3 = 1/2
    c4 = 1/2 + (15**0.5)/10
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    
    x_2 = x + h *(a2_1*k1)
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * c2, **extra_args)
    
    x_3 = x + h * (a3_1*k1 + a3_2*k2)
    k3, data3 = epsilon(model, x_3, sigmas[0] + h * c3, **extra_args)
    
    x_4 = x + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)
    k4, data4 = epsilon(model, x_4, sigmas[0] + h * c4, **extra_args)

    xp1, xp2, xp3, xp4 = k1, k2, k3, k4
    
    xpp1 = (k2 - k1) / (h*c2)
    xpp2 = (k3 - k1) / (h*c3)
    xpp3 = (k4 - k1) / (h*c4)
    
    xp = k1
    xpp = (k4 - k1) / (h)
    
    x = x + h * ((1/4)*k1 + (1/4)*k2 + (1/4)*k3 + (1/4)*k4)
    
    if not extra_options_flag("disable_proper_derivs_init", extra_options):
        xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3 + b4p*k4)
        xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3 + b4pp*k4)
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        h = sigma_down - sigma
        
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:            
            k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            #x_2 = x + h *(a2_1*k1)
            x_2 = x + h*c2*xp + ((h**2)/2)*(c2**2)*xpp + (h**3) * (a2_1*k1)
            k2, data2 = epsilon(model, x_2, sigma + h * c2, **extra_args)
            
            #x_3 = x + h * (a3_1*k1 + a3_2*k2)
            x_3 = x + h*c3*xp + ((h**2)/2)*(c3**2)*xpp + (h**3) * (a3_1*k1 + a3_2*k2)
            k3, data3 = epsilon(model, x_3, sigma + h * c3, **extra_args)
            
            #x_4 = x + h * (a4_1*k1 + a4_2*k2 + a4_3*k3)
            x_3 = x + h*c4*xp + ((h**2)/2)*(c4**2)*xpp + (h**3) * (a4_1*k1 + a4_2*k2 + a4_3*k3)
            k4, data4 = epsilon(model, x_4, sigma + h * c4, **extra_args)
            
            x = x + h*xp + ((h**2)/2)*xpp + (h**3) * (b1*k1 + b2*k2 + b3*k3 + b4*k4)

            
            if not extra_options_flag("enable_proper_derivs", extra_options):
                xp = k1.clone()
                #xp = h * (b1p * k1 + b2p * k2 + b3p * k3 + b4p * k4)
                xpp = (k4 - k1) / (h)
                xp1, xp2, xp3, xp4 = k1, k2, k3, k4
    
                xpp1 = (k2 - k1) / (h*c2)
                xpp2 = (k3 - k1) / (h*c3)
                xpp3 = (k4 - k1) / (h*c4)
            
            if not extra_options_flag("disable_proper_derivs", extra_options):
                xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3 + b4p*k4)
                xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3 + b4pp*k4)

            denoised = data4
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised



def sample_rk_test3rd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    a2_1 = 1/48
    
    a3_1 = 1/12
    a3_2 = 1/12

    
    b1 = 1/12
    b2 = 1/12
    b3 = 0
    
    b1p = 1/6
    b2p = 1/3
    b3p = 0
    
    b1pp = 1/6
    b2pp = 2/3
    b3pp = 1/6
    
    c2 = 1/2
    c3 = 1
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    
    x_2 = x + h *(a2_1*k1)
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * c2, **extra_args)
    
    x_3 = x + h * (a3_1*k1 + a3_2*k2)
    k3, data3 = epsilon(model, x_3, sigmas[0] + h * c3, **extra_args)

    xp1, xp2, xp3 = k1, k2, k3
    
    xpp1 = (k2 - k1) / (h*c2)
    xpp2 = (k3 - k1) / (h*c3)
    
    xp = k1
    xpp = (k3 - k1) / (h)
    
    x = x + h * ((1/3)*k1 + (1/3)*k2 + (1/3)*k3)
    
    if not extra_options_flag("disable_proper_derivs_init", extra_options):
        xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3)
        xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3)
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        h = sigma_down - sigma
        
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:            
            k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            #x_2 = x + h *(a2_1*k1)
            x_2 = x + h*c2*xp + ((h**2)/2)*(c2**2)*xpp + (h**3) * (a2_1*k1)
            k2, data2 = epsilon(model, x_2, sigma + h * c2, **extra_args)
            
            #x_3 = x + h * (a3_1*k1 + a3_2*k2)
            x_3 = x + h*c3*xp + ((h**2)/2)*(c3**2)*xpp + (h**3) * (a3_1*k1 + a3_2*k2)
            k3, data3 = epsilon(model, x_3, sigma + h * c3, **extra_args)

            x = x + h*xp + ((h**2)/2)*xpp + (h**3) * (b1*k1 + b2*k2 + b3*k3)

            
            if not extra_options_flag("enable_proper_derivs", extra_options):
                xp = k1.clone()
                #xp = h * (b1p * k1 + b2p * k2 + b3p * k3 + b4p * k4)
                xpp = (k3 - k1) / (h)
                xp1, xp2, xp3 = k1, k2, k3
    
                xpp1 = (k2 - k1) / (h*c2)
                xpp2 = (k3 - k1) / (h*c3)
            
            if not extra_options_flag("disable_proper_derivs", extra_options):
                xp  =  xp + h*xpp + h**2 * (b1p*k1 + b2p*k2 + b3p*k3)
                xpp = xpp + h * (b1pp*k1 + b2pp*k2 + b3pp*k3)

            denoised = data3
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised







def sample_rk_test(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    
    if extra_options_flag("logit", extra_options) or extra_options_flag("logsnr", extra_options):
        return sample_rk_logit(model, x, sigmas, extra_args, callback, disable, noise_sampler, noise_sampler_type, noise_mode, rk_type, 
              sigma_fn_formula, t_fn_formula,
                  eta, eta_var, s_noise, alpha, k, scale, c2, c3, buffer, cfgpp, iter, sub_iter, reverse_weight, extra_options)
    
    if extra_options_flag("sixth_order", extra_options):
        return sample_rk_test6th(model, x, sigmas, extra_args, callback, disable, noise_sampler, noise_sampler_type, noise_mode, rk_type, 
              sigma_fn_formula, t_fn_formula,
                  eta, eta_var, s_noise, alpha, k, scale, c2, c3, buffer, cfgpp, iter, sub_iter, reverse_weight, extra_options)
    
    if extra_options_flag("third", extra_options):
        return sample_rk_test3rd(model, x, sigmas, extra_args, callback, disable, noise_sampler, noise_sampler_type, noise_mode, rk_type, 
              sigma_fn_formula, t_fn_formula,
                  eta, eta_var, s_noise, alpha, k, scale, c2, c3, buffer, cfgpp, iter, sub_iter, reverse_weight, extra_options)
    
    
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=torch.initial_seed()+1, sigma_min=0.0, sigma_max=1.0)
    
    h = sigmas[1] - sigmas[0]
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    x_2 = x + h * (11/200) * k1
    k2, data2 = epsilon(model, x_2, sigmas[0] + h * (2/3), **extra_args)
    
    xp  =  ((1/4)*k1 + (1/4)*k2)
    xpp =  ((1/4)*k1 + (3/4)*k2)

    xp, xpp = (k1.clone() for _ in range(2))
    xpp = (k2 - k1) / (h*(2/3))
    
    x = x + h * ((1/2)*k1 + (1/2)*k2)
    #xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
    #xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)
    
    if not extra_options_flag("disable_proper_derivs_init", extra_options):
        xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
        xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)
    
    for step in trange(1, len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, sigma_next-sigma )
        h = sigma_down - sigma
        
        if sigma_down == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:
        
            k1, data1 = epsilon(model, x, sigma, **extra_args)
            
            x_2 = x   +   h*(2/3)*xp   +   ((h**2)/2)*((2/3)**2) * xpp   +   ((h**3)) * ((11/200)*k1)
            k2, data2 = epsilon(model, x_2, sigma + h*(2/3), **extra_args)
            x   =   x + h*xp + ((h**2)/2)*xpp + ((h**3)) * ((1/8)*k1 + (1/24)*k2)
            
            
            if not extra_options_flag("enable_proper_derivs", extra_options):
                xp = k1.clone()
                xpp = (k2 - k1) / (h*(2/3))
            
            if not extra_options_flag("disable_proper_derivs", extra_options):
                xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
                
                #xpp = (k2 - k1) / h
                xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)

            denoised = data2
            
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})

        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up

    return denoised





def sample_rk_test2(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", 
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.5, eta_var=0.0, s_noise=1., alpha=-1.0, k=1.0, scale=0.1, c2=0.5, c3=1.0, buffer=0, cfgpp=0.5, iter=0, sub_iter=0, reverse_weight=0.0, extra_options=""):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    seed = torch.initial_seed() + 1
    
    sigma_min, sigma_max = model.inner_model.inner_model.model_sampling.sigma_min, model.inner_model.inner_model.model_sampling.sigma_max
    
    t_fn = lambda sigma: sigma
    sigma_fn = lambda t: T
    h_fn = lambda sigma, sigma_down: sigma_down - sigma
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=0.0, sigma_max=1.0)
    
    if noise_sampler_type == "fractal":
        noise_sampler.alpha = alpha
        noise_sampler.k = k
        noise_sampler.scale = scale
        
    a = [[0,0], [11/200, 0]]
    b = [
        [1/8, 1/24],
        [1/4, 1/4], 
        [1/4, 3/4],
    ]
    c = [0, 2/3]

    xp, xpp = (torch.zeros_like(x) for _ in range(2))
    k1, data1 = epsilon(model, x, sigmas[0], **extra_args)
    xp, xpp = k1.clone(), k1.clone()
    #xp, xpp = (x.clone() for _ in range(2))
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        #h_orig = t_fn(sigma_next)-t_fn(sigma)
        #sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h_orig)
        #sigma_down = sigma_next
        #t_down, t = t_fn(sigma_down), t_fn(sigma)
        #h = h_fn(sigma, sigma_down)
        h = sigma_next - sigma
                
        if sigma_next == 0:
            denoised = model(x, sigma * s_in, **extra_args)

        else:
            k1, data1 = epsilon_res(model, x, sigma, **extra_args)
            
            x_2 = x   +   h*(2/3)*xp   +   ((h**2)/2)*((2/3)**2) * xpp   +   (h**3) * ((11/200)*k1)
            k2, data2 = epsilon(model, x_2, sigma + h*(2/3), **extra_args)
            

            k1, data1 = epsilon(model, x, sigma, **extra_args)
            x   =   x + h*xp + ((h**2)/2)*xpp + (h**3) * ((1/8)*k1 + (1/24)*k2)
            
            xp  =  xp + h*xpp + h**2 * ((1/4)*k1 + (1/4)*k2)
            xpp = xpp + h * ((1/4)*k1 + (3/4)*k2)
            
            denoised = data2
            
         
        if callback is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
        """noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        x = alpha_ratio * x + noise * s_noise * sigma_up"""

    return denoised












class SamplerRK_Test:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                     "eta_var": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculate variance-corrected noise amount (overrides eta/noise_mode settings). Cannot be used at very low sigma values; reverts to eta/noise_mode for final steps."}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False, "tooltip": "Fractal noise mode: <0 = extra high frequency noise, >0 = extra low frequency noise, 0 = white noise."}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False, "tooltip": "Fractal noise mode: all that matters is positive vs. negative. Effect unclear."}),
                     "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "CFG++ scale. Replaces CFG."}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                     "noise_mode": (["hard", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                     "iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                     "sub_iter": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit refinement steps to run after each explicit step. Currently only working with CFG."}),
                     #"t_fn_formula": ("STRING", {"default": "1/((sigma).exp()+1)", "multiline": True}),
                     #"sigma_fn_formula": ("STRING", {"default": "((1-t)/t).log()", "multiline": True}),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    },
                    "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, s_noise=1.0, alpha=-1.0, k=1.0, cfgpp=0.0, buffer=0, extra_options="", noise_sampler_type="gaussian", noise_mode="hard", rk_type="dormand-prince", t_fn_formula=None, sigma_fn_formula=None, iter=0, sub_iter=0,
                    ):
        sampler_name = "rk_test"

        steps = 10000

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "alpha": alpha, "k": k, "cfgpp": cfgpp, "buffer": buffer, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "rk_type": rk_type, 
                                                         "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "iter": iter,"sub_iter": sub_iter, "extra_options": extra_options})
        return (sampler, )





class UltraSharkSamplerRBTest:  
    # for use with https://github.com/ClownsharkBatwing/UltraCascade
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_is_latent": ("BOOLEAN", {"default": False}),
                "noise_type": (NOISE_GENERATOR_NAMES, ),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": 0.01}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": 0.01}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),               
                "guide_type": (['residual', 'weighted'], ),
                "guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
            },
            "optional": {
                "latent_noise": ("LATENT", ),
                "guide": ("LATENT",),
                "guide_weights": ("SIGMAS",),
                "style": ("CONDITIONING", ),
                "img_style": ("CONDITIONING", ),
            }
        }

    RETURN_TYPES = ("LATENT","LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output", "latent_batch")

    FUNCTION = "main"

    CATEGORY = "RES4LYF/samplers/ultracascade"
    DESCRIPTION = "For use with Stable Cascade and UltraCascade."
    
    def main(self, model, add_noise, noise_is_latent, noise_type, noise_seed, cfg, alpha, k, positive, negative, sampler, 
               sigmas, guide_type, guide_weight, latent_image, latent_noise=None, guide=None, guide_weights=None, style=None, img_style=None): 

            if model.model.model_config.unet_config.get('stable_cascade_stage') == 'up':
                model = model.clone()
                x_lr = guide['samples'] if guide is not None else None
                guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)#("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                #model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
                #model.model.diffusion_model.set_guide_type(guide_type=guide_type)
                #model.model.diffusion_model.set_x_lr(x_lr=x_lr)
                patch = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("ultracascade", {}).get("main")
                if patch is not None:
                    patch.update(x_lr=x_lr, guide_weights=guide_weights, guide_type=guide_type)
                else:
                    model.model.diffusion_model.set_sigmas_schedule(sigmas_schedule=sigmas)
                    model.model.diffusion_model.set_sigmas_prev(sigmas_prev=sigmas[:1])
                    model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
                    model.model.diffusion_model.set_guide_type(guide_type=guide_type)
                    model.model.diffusion_model.set_x_lr(x_lr=x_lr)
                
            elif model.model.model_config.unet_config['stable_cascade_stage'] == 'b':
                c_pos, c_neg = [], []
                for t in positive:
                    d_pos = t[1].copy()
                    d_neg = t[1].copy()
                    
                    d_pos['stable_cascade_prior'] = guide['samples']

                    pooled_output = d_neg.get("pooled_output", None)
                    if pooled_output is not None:
                        d_neg["pooled_output"] = torch.zeros_like(pooled_output)
                    
                    c_pos.append([t[0], d_pos])            
                    c_neg.append([torch.zeros_like(t[0]), d_neg])
                positive = c_pos
                negative = c_neg
                
            if style is not None:
                model.set_model_patch(style, 'style_cond')
            if img_style is not None:
                model.set_model_patch(img_style,'img_style_cond')
        
        
            # 1, 768      clip_style[0][0][1]['unclip_conditioning'][0]['clip_vision_output'].image_embeds.shape
            # 1, 1280     clip_style[0][0][1]['pooled_output'].shape 
            # 1, 77, 1280 clip_style[0][0][0].shape
        
        
            latent = latent_image
            latent_image = latent["samples"]
            torch.manual_seed(noise_seed)

            if not add_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
            elif latent_noise is None:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds, alpha, k)
            else:
                noise = latent_noise["samples"]#.to(torch.float64)

            if noise_is_latent:
                noise += latent_image.cpu()
                noise.sub_(noise.mean()).div_(noise.std())

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
            disable_pbar = False

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, 
                                                 noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, 
                                                 seed=noise_seed)

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
                
            return (out, out_denoised)


