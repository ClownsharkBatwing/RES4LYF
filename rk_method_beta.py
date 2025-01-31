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

from .helper import get_orthogonal, get_collinear, get_extra_options_list, has_nested_attr

MAX_STEPS = 10000


class RK_Method_Beta:
    def __init__(self, model, rk_type, device='cuda', dtype=torch.float64):
        self.model = model
        self.model_sampling = model.inner_model.inner_model.model_sampling
        self.device = device
        self.dtype = dtype
                
        self.rk_type = rk_type
             
        if rk_type in IRK_SAMPLER_NAMES_BETA:
            self.IMPLICIT = True
        else:
            self.IMPLICIT = False
            
        if RK_Method_Beta.is_exponential(rk_type):
            self.EXPONENTIAL = True
        else:
            self.EXPONENTIAL = False
        
        self.stages = 0
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
        
        self.noise_sampler  = None
        self.noise_sampler2 = None
        
        self.multistep_stages = 0
        
        self.cfg_cw = 1.0

    @staticmethod
    def is_exponential(rk_type):
        if rk_type.startswith(("res", "dpmpp", "ddim", "pec", "etdrk", "lawson", "abnorsett"   )): 
            return True
        else:
            return False

    @staticmethod
    def create(model, rk_type, device='cuda', dtype=torch.float64):
        if RK_Method_Beta.is_exponential(rk_type):
            return RK_Method_Exponential(model, rk_type, device, dtype)
        else:
            return RK_Method_Linear(model, rk_type, device, dtype)
                
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


    def set_coeff(self, rk_type, h, c1=0.0, c2=0.5, c3=1.0, step=0, sigmas=None, sigma_down=None, extra_options=None):

        self.rk_type = rk_type

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


    def reorder_tableau(self, indices):
        if indices[0]:
            self.a    = self.a   [indices]
            self.b[0] = self.b[0][indices]
            self.c    = self.c   [indices]
            self.c = torch.cat((self.c, self.c[-1:])) 
        return

    def update_substep(self, x_0, x_, eps_, eps_prev_, row, row_offset, h, h_new, h_new_orig, sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, \
                       SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, extra_options, SUBSTEP=True, ):
        if row < self.rows - row_offset   and   self.multistep_stages == 0:
            if self.IMPLICIT and not extra_options_flag("guide_fully_pseudoimplicit_use_post_substep_eta", extra_options): 
                x_[row+row_offset] = x_row_down = x_0 + h     * (self.a_k_sum(eps_, row + row_offset) + self.u_k_sum(eps_prev_, row + row_offset))
            else:
                x_[row+row_offset] = x_row_down = x_0 + h_new * (self.a_k_sum(eps_, row + row_offset) + self.u_k_sum(eps_prev_, row + row_offset))
                x_[row+row_offset] = NS.add_noise_post(x_[row+row_offset], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP=True)
            eps_row_down = x_[row+row_offset] - x_row_down
            
            if SYNC_MEAN_CW:
                x_row_down_tmp = x_0 + h_new_orig * (self.a_k_sum(eps_, row + row_offset) + self.u_k_sum(eps_prev_, row + row_offset))
                x_row_tmp = x_row_down_tmp + eps_row_down
                for c in range(x_[0].shape[-3]):
                    x_[row+row_offset][..., c, :, :] = x_[row+row_offset][..., c, :, :] - x_[row+row_offset][..., c, :, :].mean() + x_row_tmp[..., c, :, :].mean()
                
        else: 
            if self.IMPLICIT and not extra_options_flag("guide_fully_pseudoimplicit_use_post_substep_eta", extra_options): 
                x_[row+1] = x_row_down = x_0 + h     * (self.b_k_sum(eps_, 0) + self.v_k_sum(eps_prev_, 0))
            else:
                x_[row+1] = x_row_down = x_0 + h_new * (self.b_k_sum(eps_, 0) + self.v_k_sum(eps_prev_, 0))
                x_[row+1] = NS.add_noise_post(x_[row+1], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP=True)
            eps_row_down = x_[row+1] - x_row_down
            
            if SYNC_MEAN_CW:
                x_row_down_tmp = x_0 + h_new_orig * (self.b_k_sum(eps_, 0) + self.v_k_sum(eps_prev_, 0))
                x_row_tmp = x_row_down_tmp + eps_row_down
                for c in range(x_[0].shape[-3]):
                    x_[row+1][..., c, :, :] = x_[row+1][..., c, :, :] - x_[row+1][..., c, :, :].mean() + x_row_tmp[..., c, :, :].mean()
        return x_

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
        

    @staticmethod
    def calculate_res_2m_step(x_0, denoised_, sigma_down, sigmas, step,):
        if denoised_[2].sum() == 0:
            return None
        
        sigma = sigmas[step]
        sigma_prev = sigmas[step-1]
        
        h_prev = -torch.log(sigma/sigma_prev)
        h = -torch.log(sigma_down/sigma)

        c1 = 0
        c2 = (-h_prev / h).item()

        ci = [c1,c2]
        φ = Phi(h, ci, analytic_solution=True)

        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        eps_2 = denoised_[1] - x_0
        eps_1 = denoised_[0] - x_0

        h_a_k_sum = h * (b1 * eps_1 + b2 * eps_2)
        
        x = torch.exp(-h) * x_0 + h_a_k_sum
        
        denoised = x_0 + (sigma / (sigma - sigma_down)) * h_a_k_sum

        return x, denoised


    @staticmethod
    def calculate_res_3m_step(x_0, denoised_, sigma_down, sigmas, step,):
        if denoised_[3].sum() == 0:
            return None
        
        sigma       = sigmas[step]
        sigma_prev  = sigmas[step-1]
        sigma_prev2 = sigmas[step-2]

        h       = -torch.log(sigma_down/sigma)
        h_prev  = -torch.log(sigma/sigma_prev)
        h_prev2 = -torch.log(sigma/sigma_prev2)

        c1 = 0
        c2 = (-h_prev  / h).item()
        c3 = (-h_prev2 / h).item()

        ci = [c1,c2,c3]
        φ = Phi(h, ci, analytic_solution=True)
        
        gamma = (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))

        b3 = (1 / (gamma * c2 + c3)) * φ(2, -h)      
        b2 = gamma * b3 
        b1 = φ(1, -h) - b2 - b3    
        
        eps_3 = denoised_[2] - x_0
        eps_2 = denoised_[1] - x_0
        eps_1 = denoised_[0] - x_0

        h_a_k_sum = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        x = torch.exp(-h) * x_0 + h_a_k_sum
        
        denoised = x_0 + (sigma / (sigma - sigma_down)) * h_a_k_sum

        return x, denoised

    def swap_rk_type_at_step_or_threshold(self, x_0, data_prev_, sigma_down, sigmas, step, RK, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print):
        if rk_swap_type == "":
            if self.EXPONENTIAL:
                rk_swap_type = "res_3m" 
            else:
                rk_swap_type = "deis_3m"
            
        if step > rk_swap_step:
            print("Switching rk_type to:", rk_swap_type)
            self.rk_type = rk_swap_type
        if step > 2 and sigmas[step+1] > 0 and self.rk_type != rk_swap_type and rk_swap_threshold > 0:
            x_res_2m, denoised_res_2m = RK.calculate_res_2m_step(x_0, data_prev_, sigma_down, sigmas, step)
            x_res_3m, denoised_res_3m = RK.calculate_res_3m_step(x_0, data_prev_, sigma_down, sigmas, step)
            if rk_swap_print:
                print("res_3m - res_2m:", torch.norm(denoised_res_3m - denoised_res_2m).item())
            if rk_swap_threshold > torch.norm(denoised_res_2m - denoised_res_3m):
                print("Switching rk_type to:", rk_swap_type, "at step:", step)
                self.rk_type = rk_swap_type
                
        return self.rk_type




    def newton_iter(self, x_0, x_, eps_, eps_prev_, data_, s_, row, h, sigmas, step, newton_name, extra_options):
        
        newton_iter_name = "newton_iter_" + newton_name
        
        default_anchor_x_all = False
        if newton_name == "lying":
            default_anchor_x_all = True
        
        newton_iter                 =   int(get_extra_options_kv(newton_iter_name,                    str("100"),   extra_options))
        newton_iter_skip_last_steps =   int(get_extra_options_kv(newton_iter_name + "_skip_last_steps", str("0"),   extra_options))
        newton_iter_mixing_rate     = float(get_extra_options_kv(newton_iter_name + "_mixing_rate",    str("1.0"),  extra_options))
        
        newton_iter_anchor          =   int(get_extra_options_kv(newton_iter_name + "_anchor",          str("0"),   extra_options))
        newton_iter_anchor_x_all    =  bool(get_extra_options_kv(newton_iter_name + "_anchor_x_all",   str(default_anchor_x_all),       extra_options))
        newton_iter_type            =       get_extra_options_kv(newton_iter_name + "_type",      "from_epsilon",   extra_options)
        newton_iter_sequence        =       get_extra_options_kv(newton_iter_name + "_sequence",        "double",   extra_options)
        
        row_b_offset = 0
        if extra_options_flag(newton_iter_name + "_include_row_b", extra_options):
            row_b_offset = 1
        
        if step >= len(sigmas)-1-newton_iter_skip_last_steps   or   sigmas[step+1] == 0   or   not self.IMPLICIT:
            return x_, eps_
        
        sigma = sigmas[step]
        
        start, stop = 0, self.rows+row_b_offset
        if newton_name   == "pre":
            start = row
        elif newton_name == "post":
            start = row + 1
            
        if newton_iter_anchor >= 0:
            eps_anchor = eps_[newton_iter_anchor].clone()
            
        if newton_iter_anchor_x_all:
            x_orig_ = x_.clone()
            
        for n_iter in range(newton_iter):
            for r in range(start, stop):
                if newton_iter_anchor >= 0:
                    eps_[newton_iter_anchor] = eps_anchor.clone()
                if newton_iter_anchor_x_all:
                    x_ = x_orig_.clone()
                x_tmp, eps_tmp = x_[r].clone(), eps_[r].clone()
                
                seq_start, seq_stop = r, r+1
                
                if newton_iter_sequence == "double":
                    seq_start, seq_stop = start, stop
                    
                for r_ in range(seq_start, seq_stop):
                    if r_ < self.rows:
                        x_[r_] = x_0 + h * (self.a_k_sum(eps_, r_) + self.u_k_sum(eps_prev_, r_))
                    else:
                        x_[r_] = x_0 + h * (self.b_k_sum(eps_, 0) + self.v_k_sum(eps_prev_, 0))

                for r_ in range(seq_start, seq_stop):
                    if newton_iter_type == "from_data":
                        data_[r_] = get_data_from_step(x_0, x_[r_], sigma, s_[r_])  
                        eps_ [r_] = get_epsilon_simple(x_0, data_[r_], s_[r_], self.rk_type)
                    elif newton_iter_type == "from_step":
                        eps_[r_] = get_epsilon_from_step(x_0, x_[r_], sigma, s_[r_])
                    elif newton_iter_type == "from_alt":
                        eps_[r_] = x_0/sigma - x_[r_]/s_[r_]
                    elif newton_iter_type == "from_epsilon":
                        eps_ [r_] = get_epsilon_simple(x_[r_], data_[r_], s_[r_], self.rk_type)
                    
                    if extra_options_flag(newton_iter_name + "_opt", extra_options):
                        opt_timing, opt_type, opt_subtype = get_extra_options_list(newton_iter_name+"_opt", "", extra_options).split(",")
                        
                        opt_start, opt_stop = 0, self.rows+row_b_offset
                        if    opt_timing == "early":
                            opt_stop  = row + 1
                        elif  opt_timing == "late":
                            opt_start = row + 1

                        for r2 in range(opt_start, opt_stop): 
                            if r_ != r2:
                                if   opt_subtype == "a":
                                    eps_a = eps_[r2]
                                    eps_b = eps_[r_]
                                elif opt_subtype == "b":
                                    eps_a = eps_[r_]
                                    eps_b = eps_[r2]
                                
                                if   opt_type == "ortho":
                                    eps_ [r_] = get_orthogonal(eps_a, eps_b)
                                elif opt_type == "collin":
                                    eps_ [r_] = get_collinear(eps_a, eps_b)
                                elif opt_type == "proj":
                                    eps_ [r_] = get_collinear(eps_a, eps_b) + get_orthogonal(eps_b, eps_a)
                                    
                    x_  [r_] =   x_tmp + newton_iter_mixing_rate * (x_  [r_] -   x_tmp)
                    eps_[r_] = eps_tmp + newton_iter_mixing_rate * (eps_[r_] - eps_tmp)
                    
                if newton_iter_sequence == "double":
                    break
        
        return x_, eps_




class RK_Method_Exponential(RK_Method_Beta):
    def __init__(self, model, rk_type, device='cuda', dtype=torch.float64):
        super().__init__(model, rk_type, device, dtype) 
        
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

    def __call__(self, x, sub_sigma, x_0, sigma, **extra_args):
        denoised = self.model_denoised(x, sub_sigma, **extra_args)
        epsilon = denoised - x_0
        #print("MODEL SUB_SIGMA: ", round(float(sub_sigma),3), round(float(sigma),3))

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
    def __init__(self, model, rk_type, device='cuda', dtype=torch.float64):
        super().__init__(model, rk_type, device, dtype) 
        
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
    
    def __call__(self, x, sub_sigma, x_0, sigma, **extra_args):
        denoised = self.model_denoised(x, sub_sigma, **extra_args)
        epsilon = (x_0 - denoised) / sigma
        #print("MODEL SUB_SIGMA: ", round(float(sub_sigma),3), round(float(sigma),3))

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






def get_epsilon_simple(x_0, denoised, sigma, rk_type):
    if RK_Method_Beta.is_exponential(rk_type):
        eps = denoised - x_0
    else:
        eps = (x_0 - denoised) / sigma
    return eps

def get_data_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (x - x_next) / h








class RK_NoiseSampler:
    def __init__(self, model, device='cuda', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        
        if has_nested_attr(model, "inner_model.inner_model.model_sampling"):
            model_sampling = model.inner_model.inner_model.model_sampling
        elif has_nested_attr(model, "model.model_sampling"):
            model_sampling = model.model.model_sampling
        
        self.CONST = isinstance(model_sampling, comfy.model_sampling.CONST)

        self.sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(dtype)
        self.sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(dtype)
        
        self.noise_sampler  = None
        self.noise_sampler2 = None

    def init_noise_sampler(self, x, noise_seed, noise_sampler_type, noise_sampler_type2, alpha, alpha2, k=1., k2=1., scale=0.1, scale2=0.1):
        if noise_seed < 0:
            seed = torch.initial_seed()+1 
            print("SDE noise seed: ", seed, " (set via torch.initial_seed()+1)")
        else:
            seed = noise_seed
            print("SDE noise seed: ", seed)
            
        seed2 = seed + MAX_STEPS #for substep noise generation. offset needed to ensure seeds are not reused
            
        if noise_sampler_type == "fractal":
            self.noise_sampler        = NOISE_GENERATOR_CLASSES.get(noise_sampler_type )(x=x, seed=seed,  sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler.alpha  = alpha
            self.noise_sampler.k      = k
            self.noise_sampler.scale  = scale
        if noise_sampler_type2 == "fractal":
            self.noise_sampler2       = NOISE_GENERATOR_CLASSES.get(noise_sampler_type2)(x=x, seed=seed2, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler2.alpha = alpha2
            self.noise_sampler2.k     = k2
            self.noise_sampler2.scale = scale2
        else:
            self.noise_sampler  = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type )(x=x, seed=seed,  sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler2 = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type2)(x=x, seed=seed2, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            
    def add_noise_pre(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, CONSERVE_MEAN_CW=True, SDE_NOISE_EXTERNAL=False, sde_noise_t=None, SUBSTEP=False, ):
        if not self.CONST and noise_mode == "hard": 
            return self.add_noise(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SUBSTEP, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
        
    def add_noise_post(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, CONSERVE_MEAN_CW=True, SDE_NOISE_EXTERNAL=False, sde_noise_t=None, SUBSTEP=False, ):
        if self.CONST   or   (not self.CONST and noise_mode != "hard"):
            return self.add_noise(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP, )
        else:
            return x
    
    def add_noise(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP, ):

        if sigma_next > 0.0 and sigma_up > 0.0:
            if not SUBSTEP:
                noise = self.noise_sampler (sigma=sigma, sigma_next=sigma_next)
            else:
                noise = self.noise_sampler2(sigma=sigma, sigma_next=sigma_next)
                
            noise = torch.nan_to_num((noise - noise.mean()) / noise.std(), 0.0)

            noise_ortho = get_orthogonal(noise, x)
            noise_ortho = noise_ortho / noise_ortho.std()
            
            noise = noise_ortho

            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            x_next = alpha_ratio * x + noise * sigma_up * s_noise
            
            if CONSERVE_MEAN_CW:
                for c in range(x.shape[-3]):
                    x_next[..., c, :, :] = x_next[..., c, :, :] - x_next[..., c, :, :].mean() + x[..., c, :, :].mean()
            
            return x_next
        
        else:
            return x

