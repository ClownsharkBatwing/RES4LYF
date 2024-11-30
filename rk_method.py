import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc
import math
import copy
import re
from typing import Optional

import torch.nn.functional as F
import torchvision.transforms as T

import functools

from .noise_classes import *

import comfy.model_patcher
import comfy.supported_models


from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3
from .rk_coefficients import *
from .phi_functions import *


class RK_Method:
    def __init__(self, model, name="", method="explicit", dynamic_method=False, device='cuda', dtype=torch.float64):
        self.model = model
        self.model_sampling = model.inner_model.inner_model.model_sampling
        self.device = device
        self.dtype = dtype
        
        self.method = method
        self.dynamic_method = dynamic_method
        
        self.stages = 0
        self.name = name
        self.ab = None
        self.a = None
        self.b = None
        self.c = None
        self.denoised = None
        self.uncond = None
        
        self.rows = 0
        self.cols = 0
        
        self.y0 = None
        self.y0_inv = None
        
        self.sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(dtype)
        self.sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(dtype)
        
        self.noise_sampler = None
        
        self.h_prev = None
        self.h_prev2 = None
        self.multistep_stages = 0
        
        #self.UNSAMPLE = False
        
    @staticmethod
    def is_exponential(rk_type):
        if rk_type.startswith(("res", "dpmpp", "ddim", "rk_exp")):
            return True
        else:
            return False

    @staticmethod
    def create(model, rk_type, device='cuda', dtype=torch.float64, name="", method="explicit"):
        if RK_Method.is_exponential(rk_type):
            return RK_Method_Exponential(model, name, method, device, dtype)
        else:
            return RK_Method_Linear(model, name, method, device, dtype)
                
    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")
    
    def model_epsilon(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        x0 = self.model(x, sigma * s_in, **extra_args)
        #return x0 ###################################THIS WORKS ONLY WITH THE MODEL SAMPLING PATCH
        eps = (x - x0) / (sigma * s_in) 
        return eps, x0
    
    def model_denoised(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        x0 = self.model(x, sigma * s_in, **extra_args)
        return x0
    
    @staticmethod
    def phi(j, neg_h):
        remainder = torch.zeros_like(neg_h)
        for k in range(j): 
            remainder += (neg_h)**k / math.factorial(k)
        phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
        return phi_j_h
    
    @staticmethod
    def calculate_gamma(c2, c3):
        return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))
    
    def init_noise_sampler(self, x, noise_seed, noise_sampler_type, alpha, k=1., scale=0.1):
        seed = torch.initial_seed()+1 if noise_seed == -1 else noise_seed
        if noise_sampler_type == "fractal":
            self.noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler.alpha = alpha
            self.noise_sampler.k = k
            self.noise_sampler.scale = scale
        else:
            self.noise_sampler = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            
    def add_noise_pre(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None):
        if isinstance(self.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard":
            return self.add_noise(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
        
    def add_noise_post(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None):
        if isinstance(self.model_sampling, comfy.model_sampling.CONST) == True  or (isinstance(self.model_sampling, comfy.model_sampling.CONST) == False and noise_mode != "hard"):
            return self.add_noise(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
    
    def add_noise(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t):

        if sigma_next > 0.0:
            noise = self.noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = torch.nan_to_num((noise - noise.mean()) / noise.std(), 0.0)

            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            return alpha_ratio * x + noise * sigma_up * s_noise
        
        else:
            return x
    
    def ab_sum(self, ab, row, columns, ki, ki_u, y0, y0_inv):
        ks, ks_u, ys, ys_inv = torch.zeros_like(ki[0]), torch.zeros_like(ki[0]), torch.zeros_like(ki[0]), torch.zeros_like(ki[0])
        for col in range(columns):
            ks     += ab[row][col] * ki  [col]
            ks_u   += ab[row][col] * ki_u[col]
            ys     += ab[row][col] * y0
            ys_inv += ab[row][col] * y0_inv
        return ks, ks_u, ys, ys_inv
    
    def prepare_sigmas(self, sigmas):
        if sigmas[0] == 0.0:      #remove padding used to avoid need for model patch with noise inversion
            UNSAMPLE = True
            sigmas = sigmas[1:-1]
        else: 
            UNSAMPLE = False
            
        if hasattr(self.model, "sigmas"):
            self.model.sigmas = sigmas
            
        return sigmas, UNSAMPLE
    
    
    def set_coeff(self, rk_type, h, c1=0.0, c2=0.5, c3=1.0, stepcount=0, sigmas=None, sigma=None, sigma_down=None):
        if rk_type == "default": 
            return
        #if self.a is None or self.dynamic_method == True:
        sigma = sigmas[stepcount]
        sigma_next = sigmas[stepcount+1]
        
        a, b, ci, multistep_stages, FSAL = get_rk_methods(rk_type, h, c1, c2, c3, self.h_prev, self.h_prev2, stepcount, sigmas, sigma, sigma_next, sigma_down)
        
        self.multistep_stages = multistep_stages
        
        self.a = torch.tensor(a, dtype=h.dtype, device=h.device)
        self.a = self.a.view(*self.a.shape, 1, 1, 1, 1)
        
        self.b = torch.tensor(b, dtype=h.dtype, device=h.device)
        self.b = self.b.view(*self.b.shape, 1, 1, 1, 1)
        
        self.c = torch.tensor(ci, dtype=h.dtype, device=h.device)
        self.rows = self.a.shape[0]
        self.cols = self.a.shape[1]
            
    def a_k_sum(self, k, row):
        if len(k.shape) == 4:
            ks = k * self.a[row].sum(dim=0)
        ks = (k[0:self.cols] * self.a[row]).sum(dim=0)
        return ks
    
    def b_k_sum(self, k, row):
        if len(k.shape) == 4:
            ks = k * self.b[row].sum(dim=0)
        ks = (k[0:self.cols] * self.b[row]).sum(dim=0)
        return ks



    def init_guides(self, x, latent_guide, latent_guide_inv, mask, sigmas, UNSAMPLE):
        y0, y0_inv = torch.zeros_like(x), torch.zeros_like(x)
        
        if latent_guide is not None:
            if sigmas[0] > sigmas[1]:
                y0 = latent_guide = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
            elif UNSAMPLE and mask is not None:
                x = (1-mask) * x + mask * self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
            else:
                x = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)

        if latent_guide_inv is not None:
            if sigmas[0] > sigmas[1]:
                y0_inv = latent_guide_inv = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)
            elif UNSAMPLE and mask is not None:
                x = mask * x + (1-mask) * self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)
            else:
                x = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_INV AFTER SETTING TO LG above!
                
        if UNSAMPLE and sigmas[0] < sigmas[1]: #sigma_next > sigma:
            y0 = self.noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min)
            y0 = (y0 - y0.mean()) / y0.std()
            y0_inv = self.noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min)
            y0_inv = (y0_inv - y0_inv.mean()) / y0_inv.std()
            
        return x, y0, y0_inv



    def init_cfgpp(self, model, x, cfgpp, extra_args):
        self.uncond = [torch.full_like(x, 0.0)]
        if cfgpp != 0.0:
            def post_cfg_function(args):
                self.uncond[0] = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
        #TODO: complete this method
        



class RK_Method_Exponential(RK_Method):
    def __init__(self, model, name="", method="explicit", device='cuda', dtype=torch.float64):
        super().__init__(model, name, method, device, dtype) 
        self.exponential = True
        self.eps_pred = True
        
    @staticmethod
    def alpha_fn(neg_h):
        return torch.exp(neg_h)

    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def t_fn(sigma):
        return sigma.log().neg()
    
    @staticmethod
    def h_fn(sigma_down, sigma):
        return -torch.log(sigma_down/sigma)

    def __call__(self, x_0, x, sigma, h, **extra_args):

        denoised = self.model_denoised(x, sigma, **extra_args)
        epsilon = denoised - x_0
        
        if self.uncond == None:
            self.uncond = torch.zeros_like(x)
        denoised_u = self.uncond.clone()
        if torch.all(denoised_u == 0):
            epsilon_u = torch.zeros_like(x_0)
        else:
            epsilon_u = denoised_u - x_0
            
        self.h_prev2 = self.h_prev
        self.h_prev = h
        return epsilon, denoised
    
    def data_to_vel(self, x, data, sigma):
        return data - x
    
    def get_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, t_i=None, extra_options=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = t_i if t_i is not None else sigma_cur

        if extra_options is not None:
            if re.search(r"\bpower_unsample\b", extra_options) or re.search(r"\bpower_resample\b", extra_options):
                if sigma_down is None:
                    return y - x_0
                else:
                    if sigma_down > sigma:
                        return (x_0 - y) * sigma_cur
                    else:
                        return (y - x_0) * sigma_cur
            else:
                if sigma_down is None:
                    return (y - x_0) / sigma_cur
                else:
                    if sigma_down > sigma:
                        return (x_0 - y) / sigma_cur
                    else:
                        return (y - x_0) / sigma_cur



class RK_Method_Linear(RK_Method):
    def __init__(self, model, name="", method="explicit", device='cuda', dtype=torch.float64):
        super().__init__(model, name, method, device, dtype) 
        self.expanential = False
        self.eps_pred = True
        
    @staticmethod
    def alpha_fn(neg_h):
        return torch.ones_like(neg_h)

    @staticmethod
    def sigma_fn(t):
        return t

    @staticmethod
    def t_fn(sigma):
        return sigma
    
    @staticmethod
    def h_fn(sigma_down, sigma):
        return sigma_down - sigma
    
    def __call__(self, x_0, x, sigma, h, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        
        epsilon, denoised = self.model_epsilon(x, sigma, **extra_args)
        
        if self.uncond == None:
            self.uncond = torch.zeros_like(x)
        denoised_u = self.uncond.clone()
        if torch.all(denoised_u == 0):
            epsilon_u = torch.zeros_like(x_0)
        else:
            epsilon_u  = (x_0 - denoised_u) / (sigma * s_in)
            
        self.h_prev2 = self.h_prev
        self.h_prev = h
        return epsilon, denoised

    def data_to_vel(self, x, data, sigma):
        return (data - x) / sigma
    
    def get_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, t_i=None, extra_options=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = t_i if t_i is not None else sigma_cur

        if sigma_down is None:
            return (x - y) / sigma_cur
        else:
            if sigma_down > sigma:
                return (y - x) / sigma_cur
            else:
                return (x - y) / sigma_cur



