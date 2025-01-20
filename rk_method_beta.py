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

import itertools 

from .rk_coefficients_beta import *
from .phi_functions import *



class RK_Method_Beta:
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
        self.u = None
        self.v = None
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
        
        self.cfg_cw = 1.0

        
    @staticmethod
    def is_exponential(rk_type):
        #if rk_type.startswith(("res", "dpmpp", "ddim", "irk_exp_diag_2s"   )): 
        if rk_type.startswith(("res", "dpmpp", "ddim", "pec", "etdrk"   )): 
            return True
        else:
            return False

    @staticmethod
    def create(model, rk_type, device='cuda', dtype=torch.float64, name="", method="explicit"):
        if RK_Method_Beta.is_exponential(rk_type):
            return RK_Method_Exponential(model, name, method, device, dtype)
        else:
            return RK_Method_Linear(model, name, method, device, dtype)
                
    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")
    
    def model_epsilon(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        denoised = self.model(x, sigma * s_in, **extra_args)
        denoised = self.calc_cfg_channelwise(denoised)

        #return x0 ###################################THIS WORKS ONLY WITH THE MODEL SAMPLING PATCH
        eps = (x - denoised) / (sigma * s_in).view(x.shape[0], 1, 1, 1)
        return eps, denoised
    
    def model_denoised(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        denoised = self.model(x, sigma * s_in, **extra_args)
        denoised = self.calc_cfg_channelwise(denoised)
        return denoised
    


    def init_noise_sampler(self, x, noise_seed, noise_sampler_type, alpha, k=1., scale=0.1):
        seed = torch.initial_seed()+1 if noise_seed == -1 else noise_seed
        print("Noise seed set to: ", seed)
        if noise_sampler_type == "fractal":
            self.noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler.alpha = alpha
            self.noise_sampler.k = k
            self.noise_sampler.scale = scale
        else:
            self.noise_sampler = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            
    def add_noise_pre(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None):
        if isinstance(self.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard": 
            return self.add_noise(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
        
    def add_noise_post(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None):
        if isinstance(self.model_sampling, comfy.model_sampling.CONST) == True   or   (isinstance(self.model_sampling, comfy.model_sampling.CONST) == False and noise_mode != "hard"):
            return self.add_noise(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
    
    def add_noise(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t):

        if sigma_next > 0.0:
            noise = self.noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = torch.nan_to_num((noise - noise.mean()) / noise.std(), 0.0)

            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            return alpha_ratio * x + noise * sigma_up * s_noise
        
        else:
            return x


    def set_coeff(self, rk_type, h, c1=0.0, c2=0.5, c3=1.0, step=0, sigmas=None, sigma_down=None, extra_options=None):
        if rk_type == "default": 
            return

        sigma = sigmas[step]
        sigma_next = sigmas[step+1]
        
        h_prev = []
        a, b, u, v, ci, multistep_stages, hybrid_stages, FSAL = get_rk_methods_beta(rk_type, h, c1, c2, c3, h_prev, step, sigmas, sigma, sigma_next, sigma_down, extra_options)
        
        self.multistep_stages = multistep_stages
        self.hybrid_stages    = hybrid_stages
        
        self.a = torch.tensor(a, dtype=h.dtype, device=h.device)
        self.a = self.a.view(*self.a.shape, 1, 1, 1, 1, 1)
        
        self.b = torch.tensor(b, dtype=h.dtype, device=h.device)
        self.b = self.b.view(*self.b.shape, 1, 1, 1, 1, 1)
        
        if u is not None and v is not None:
            self.u = torch.tensor(u, dtype=h.dtype, device=h.device)
            self.u = self.u.view(*self.u.shape, 1, 1, 1, 1, 1)
            
            self.v = torch.tensor(v, dtype=h.dtype, device=h.device)
            self.v = self.v.view(*self.v.shape, 1, 1, 1, 1, 1)
        
        self.c = torch.tensor(ci, dtype=h.dtype, device=h.device)
        self.rows = self.a.shape[0]
        self.cols = self.a.shape[1]


    def a_k_sum(self, k, row):
        if len(k.shape) == 4:
            a_coeff = self.a[row].squeeze(-1)
            ks = k * a_coeff.sum(dim=0)
        elif len(k.shape) == 5:
            a_coeff = self.a[row].squeeze(-1)
            ks = (k[0:self.cols] * a_coeff).sum(dim=0)
        elif len(k.shape) == 6:
            a_coeff = self.a[row]
            ks = (k[0:self.cols] * a_coeff).sum(dim=0)
        else:
            raise ValueError(f"Unexpected k shape: {k.shape}")
        return ks

    def b_k_sum(self, k, row):
        if len(k.shape) == 4:
            b_coeff = self.b[row].squeeze(-1)
            ks = k * b_coeff.sum(dim=0)
        elif len(k.shape) == 5:
            b_coeff = self.b[row].squeeze(-1)
            ks = (k[0:self.cols] * b_coeff).sum(dim=0)
        elif len(k.shape) == 6:
            b_coeff = self.b[row]
            ks = (k[0:self.cols] * b_coeff).sum(dim=0)
        else:
            raise ValueError(f"Unexpected k shape: {k.shape}")
        return ks

    def u_k_sum(self, k, row):
        if self.u is None:
            return 0
        if len(k.shape) == 4:
            u_coeff = self.u[row].squeeze(-1)
            ks = k * u_coeff.sum(dim=0)
        elif len(k.shape) == 5:
            u_coeff = self.u[row].squeeze(-1)
            ks = (k[0:self.cols] * u_coeff).sum(dim=0)
        elif len(k.shape) == 6:
            u_coeff = self.u[row]
            ks = (k[0:self.cols] * u_coeff).sum(dim=0)
        else:
            raise ValueError(f"Unexpected k shape: {k.shape}")
        return ks

    def v_k_sum(self, k, row):
        if self.v is None:
            return 0
        if len(k.shape) == 4:
            v_coeff = self.v[row].squeeze(-1)
            ks = k * v_coeff.sum(dim=0)
        elif len(k.shape) == 5:
            v_coeff = self.v[row].squeeze(-1)
            ks = (k[0:self.cols] * v_coeff).sum(dim=0)
        elif len(k.shape) == 6:
            v_coeff = self.v[row]
            ks = (k[0:self.cols] * v_coeff).sum(dim=0)
        else:
            raise ValueError(f"Unexpected k shape: {k.shape}")
        return ks


    def init_cfg_channelwise(self, x, cfg_cw=1.0, **extra_args):
        self.uncond = [torch.full_like(x, 0.0)]
        self.cfg_cw = cfg_cw
        if cfg_cw != 1.0:
            def post_cfg_function(args):
                self.uncond[0] = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
        return extra_args
            
            
    def calc_cfg_channelwise(self, denoised):
        if self.cfg_cw != 1.0:            
            avg = 0
            for b, c in itertools.product(range(denoised.shape[0]), range(denoised.shape[1])):
                avg     += torch.norm(denoised[b][c] - self.uncond[0][b][c])
            avg  /= denoised.shape[1]
            
            for b, c in itertools.product(range(denoised.shape[0]), range(denoised.shape[1])):
                ratio     = torch.nan_to_num(torch.norm(denoised[b][c] - self.uncond[0][b][c])   /   avg,     0)
                denoised_new = self.uncond[0] + ratio * self.cfg_cw * (denoised - self.uncond[0])
            return denoised_new
        else:
            return denoised
        
        

class RK_Method_Exponential(RK_Method_Beta):
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

    def __call__(self, x_0, x, sigma, **extra_args):

        denoised = self.model_denoised(x, sigma, **extra_args)
        epsilon = denoised - x_0
        
        return epsilon, denoised
    
    def data_to_vel(self, x, data, sigma):
        return data - x
    
    def get_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, unsample_resample_scale=None, extra_options=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = unsample_resample_scale if unsample_resample_scale is not None else sigma_cur

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



class RK_Method_Linear(RK_Method_Beta):
    def __init__(self, model, name="", method="explicit", device='cuda', dtype=torch.float64):
        super().__init__(model, name, method, device, dtype) 
        self.expanential = False
        self.eps_pred = True
        
    #@staticmethod
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
    
    def __call__(self, x_0, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        
        epsilon, denoised = self.model_epsilon(x, sigma, **extra_args)
            
        return epsilon, denoised

    def data_to_vel(self, x, data, sigma):
        return (data - x) / sigma
    
    def get_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, unsample_resample_scale=None, extra_options=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = unsample_resample_scale if unsample_resample_scale is not None else sigma_cur

        if sigma_down is None:
            return (x - y) / sigma_cur
        else:
            if sigma_down > sigma:
                return (y - x) / sigma_cur
            else:
                return (x - y) / sigma_cur



