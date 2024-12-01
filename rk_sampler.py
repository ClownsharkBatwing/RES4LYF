import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re

from tqdm.auto import trange
import gc

import comfy.model_patcher

from .noise_classes import *
from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3

from .rk_method import RK_Method
from .rk_guide_func import *

from .latents import normalize_latent
from .helper import get_extra_options_kv



def sample_res_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_2m", eta=0.0, )

def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_2s", eta=0.0, )

def sample_res_3s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_3s", eta=0.0, )

def sample_res_5s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_5s", eta=0.0, )



def sample_res_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_2m", eta=0.5, )

def sample_res_2s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_2s", eta=0.5, )

def sample_res_3s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_3s", eta=0.5, )

def sample_res_5s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="res_5s", eta=0.5, )



def sample_deis_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="deis_2m", eta=0.0, )

def sample_deis_3m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="deis_3m", eta=0.0, )

def sample_deis_4m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="deis_4m", eta=0.0, )



def sample_deis_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="deis_2m", eta=0.5, )

def sample_deis_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="deis_3m", eta=0.5, )

def sample_deis_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="linear", noise_seed=-1, rk_type="deis_4m", eta=0.5, )



def normalize_inputs(x, y0, y0_inv, guide_mode,  extra_options):
    
    if guide_mode == "epsilon_guide_mean_std_from_bkg":
        y0 = normalize_latent(y0, y0_inv)
        
    input_norm = get_extra_options_kv("input_norm", "", extra_options)
    input_std = float(get_extra_options_kv("input_std", "1.0", extra_options))
    
    if input_norm == "input_ch_mean_set_std_to":
        x = normalize_latent(x, set_std=input_std)

    if input_norm == "input_ch_set_std_to":
        x = normalize_latent(x, set_std=input_std, mean=False)
            
    if input_norm == "input_mean_set_std_to":
        x = normalize_latent(x, set_std=input_std, channelwise=False)
        
    if input_norm == "input_std_set_std_to":
        x = normalize_latent(x, set_std=input_std, mean=False, channelwise=False)
    
    return x, y0, y0_inv


def prepare_step_to_sigma_zero(rk, irk, rk_type, irk_type, model, x, extra_options, alpha, k, noise_sampler_type):
    rk_type_final_step = f"ralston_{rk_type[-2:]}" if rk_type[-2:] in {"2s", "3s"} else "ralston_3s"
    rk_type_final_step = f"deis_2m" if rk_type[-2:] in {"2m", "3m", "4m"} else rk_type_final_step
    rk_type_final_step = f"euler" if rk_type in {"ddim"} else rk_type_final_step
    rk_type_final_step = get_extra_options_kv("rk_type_final_step", rk_type_final_step, extra_options)
    rk = RK_Method.create(model, rk_type_final_step, x.device)
    rk.init_noise_sampler(x, torch.initial_seed() + 1, noise_sampler_type, alpha=alpha, k=k)

    if irk_type == rk_type:
        irk_type_final_step = f"gauss-legendre_{rk_type[-2:]}" if rk_type[-2:] in {"2s", "3s", "4s", "5s"} else "gauss-legendre_2s"
        irk_type_final_step = f"deis_2m" if rk_type[-2:] in {"2m", "3m", "4m"} else irk_type_final_step
        irk_type_final_step = get_extra_options_kv("irk_type_final_step", irk_type_final_step, extra_options)
        irk = RK_Method.create(model, irk_type_final_step, x.device)
        irk.init_noise_sampler(x, torch.initial_seed() + 100, noise_sampler_type, alpha=alpha, k=k)
    else:
        irk_type_final_step = irk_type

    eta, eta_var = 0, 0
    return rk, irk, rk_type_final_step, irk_type_final_step, eta, eta_var


@torch.no_grad()
def sample_rk(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="use_explicit",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, scale=0.1, c1=0.0, c2=0.5, c3=1.0, MULTISTEP=False, cfgpp=0.0, implicit_steps=0, reverse_weight=0.0, exp_mode=False,
                  latent_guide=None, latent_guide_inv=None, latent_guide_weights=None, latent_guide_weights_inv=None, guide_mode="blend", unsampler_type="linear",
                  GARBAGE_COLLECT=False, mask=None, mask_inv=None, LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,sde_noise=None,
                  extra_options="",
                  etas=None, s_noises=None, momentums=None,
                  ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    SDE_NOISE_EXTERNAL = False
    if sde_noise is not None:
        if len(sde_noise) > 0 and sigmas[1] > sigmas[2]:
            SDE_NOISE_EXTERNAL = True

    
    if latent_guide_weights is not None:
        lgw = latent_guide_weights.to(x.device)
    else:
        lgw = torch.full_like(sigmas, 0.)
    if latent_guide_weights_inv is not None:
        lgw_inv = latent_guide_weights_inv.to(x.device)
    else:
        lgw_inv = torch.full_like(sigmas, 0.)
    
    if sigmas_override is not None:
        sigmas = sigmas_override.clone()
    sigmas = sigmas.clone() * d_noise
    
    rk = RK_Method.create(model, rk_type, x.device)
    rk.init_noise_sampler(x, noise_seed, noise_sampler_type, alpha=alpha, k=k)

    irk_type = implicit_sampler_name if implicit_sampler_name != "use_explicit" else rk_type
    irk = RK_Method.create(model, irk_type, x.device)
    irk.init_noise_sampler(x, noise_seed+100, noise_sampler_type, alpha=alpha, k=k)

    sigmas, UNSAMPLE = rk.prepare_sigmas(sigmas)
    mask, LGW_MASK_RESCALE_MIN = prepare_mask(x, mask, LGW_MASK_RESCALE_MIN)
    if mask_inv is not None:
        mask_inv, LGW_MASK_RESCALE_MIN = prepare_mask(x, mask_inv, LGW_MASK_RESCALE_MIN)
    elif sigmas[0] < sigmas[1]:
        mask_inv = (1-mask)
    
    x, y0, y0_inv = rk.init_guides(x, latent_guide, latent_guide_inv, mask, sigmas, UNSAMPLE)
    x, y0, y0_inv = normalize_inputs(x, y0, y0_inv, guide_mode, extra_options)
        
    if SDE_NOISE_EXTERNAL:
        sigma_up_total = torch.zeros_like(sigmas[0])
        for i in range(len(sde_noise)-1):
            sigma_up_total += sigmas[i+1]
        eta = eta / sigma_up_total

    uncond = [torch.full_like(x, 0.0)]
    if cfgpp != 0.0:
        def post_cfg_function(args):
            uncond[0] = args["uncond_denoised"]
            return args["denoised"]
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)  

    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        unsample_resample_scale = unsample_resample_scales[step] if unsample_resample_scales is not None else None
        eta = eta_var = etas[step] if etas is not None else eta
        s_noise = s_noises[step] if s_noises is not None else s_noise
        
        if sigma_next == 0:
            rk, irk, rk_type, irk_type, eta, eta_var = prepare_step_to_sigma_zero(rk, irk, rk_type, irk_type, model, x, extra_options, alpha, k, noise_sampler_type)

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, rk.h_fn(sigma_next,sigma) )
        h     =  rk.h_fn(sigma_down, sigma)
        h_irk = irk.h_fn(sigma_down, sigma)
        
        c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn, t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula)
        
        rk. set_coeff(rk_type, h, c1, c2, c3, step, sigmas, sigma, sigma_down)
        irk.set_coeff(irk_type, h_irk, c1, c2, c3, step, sigmas, sigma, sigma_down)
        
        if step == 0:
            x_, data_, data_u, eps_ = (torch.zeros(max(rk.rows, irk.rows) + 2, *x.shape, dtype=x.dtype, device=x.device) for step in range(4))
        
        s_       = [(  rk.sigma_fn( rk.t_fn(sigma) +     h*c_)) * s_in for c_ in   rk.c]
        s_irk_rk = [(  rk.sigma_fn( rk.t_fn(sigma) +     h*c_)) * s_in for c_ in  irk.c]
        s_irk    = [( irk.sigma_fn(irk.t_fn(sigma) + h_irk*c_)) * s_in for c_ in  irk.c]

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
        x_[0] = rk.add_noise_pre(x, y0, lgw[step], sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t) #y0, lgw, sigma_down are currently unused
        
        x_0 = x_[0].clone()
        
        for ms in range(rk.multistep_stages):
            if RK_Method.is_exponential(rk_type):
                eps_ [rk.multistep_stages - ms] = data_ [rk.multistep_stages - ms] - x_0
            else:
                eps_ [rk.multistep_stages - ms] = (x_0 - data_ [rk.multistep_stages - ms]) / sigma
            
        lgw_mask, lgw_mask_inv = prepare_weighted_masks(mask, mask_inv, lgw[step], lgw_inv[step], latent_guide, latent_guide_inv, LGW_MASK_RESCALE_MIN)        

        if implicit_steps == 0: 
            for row in range(rk.rows - rk.multistep_stages):
                x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row)
                eps_[row], data_[row] = rk(x_0, x_[row+1], s_[row], h, **extra_args)       #MODEL CALL
                eps_ = process_guides_substep(x_0, x_, eps_, data_, row, y0, y0_inv, lgw[step], lgw_inv[step], lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, guide_mode, latent_guide_inv, UNSAMPLE, extra_options)
            
            x = x_0 + h * rk.b_k_sum(eps_, 0)
                    
            denoised = x + (sigma / (sigma - sigma_down)) *  h * rk.b_k_sum(eps_, 0) 
            eps = x - denoised
            x = process_guides_poststep(x, denoised, eps, y0, y0_inv, mask, lgw_mask, lgw_mask_inv, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options)

        elif any(irk_type.startswith(prefix) for prefix in {"crouzeix", "irk_exp_diag"}):
            for row in range(irk.rows - irk.multistep_stages):
                s_tmp = s_irk[row-1] if row >= 1 else sigma
                eps_[row], data_[row] = irk(x_0, x_[row], s_tmp, h, **extra_args) 
                for diag_iter in range(implicit_steps+1):
                    x_[row+1] = x_0 + h * irk.a_k_sum(eps_, row)
                    eps_[row], data_[row] = irk(x_0, x_[row+1], s_irk[row], h, **extra_args)       #MODEL CALL
                    eps_ = process_guides_substep(x_0, x_, eps_, data_, row, y0, y0_inv, lgw[step], lgw_inv[step], lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_irk, unsample_resample_scale, irk, irk_type, guide_mode, latent_guide_inv, UNSAMPLE, extra_options)
                    
            x = x_0 + h * irk.b_k_sum(eps_, 0) 
            
            denoised = x + (sigma / (sigma - sigma_down)) *  h * irk.b_k_sum(eps_, 0) 
            eps = x - denoised
            x = process_guides_poststep(x, denoised, eps, y0, y0_inv, mask, lgw_mask, lgw_mask_inv, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options)

        else:
            s2 = s_irk_rk[:]
            s2.append(sigma.unsqueeze(dim=0))
            s_all = torch.sort(torch.stack(s2, dim=0).squeeze(dim=1).unique(), descending=True)[0]
            sigmas_and = torch.cat( (sigmas[0:step], s_all), dim=0)
            
            eps_ [0], data_ [0] = torch.zeros_like(eps_ [0]), torch.zeros_like(data_[0])
            eps_list = []
            
            x_mid = x
            for i in range(len(s_all)-1):
                x_mid, eps_, data_ = get_explicit_rk_step(rk, rk_type, x_mid, y0, y0_inv, lgw[step], lgw_inv[step], mask, lgw_mask, lgw_mask_inv, step, s_all[i], s_all[i+1], eta, eta_var, s_noise, noise_mode, c2, c3, step+i, sigmas_and, x_, eps_, data_, unsample_resample_scale, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options, **extra_args)
                eps_list.append(eps_[0])
                eps_ [0], data_ [0] = torch.zeros_like(eps_ [0]), torch.zeros_like(data_[0])
                
            if torch.allclose(s_all[-1], sigma_down, atol=1e-8):
                eps_down, data_down = rk(x_0, x_mid, sigma_down, h, **extra_args)
                eps_list.append(eps_down)
                
            s_all = [s for s in s_all if s in s_irk_rk]

            eps_list = [eps_list[s_all.index(s)].clone() for s in s_irk_rk]
            eps2_ = torch.stack(eps_list, dim=0)

            for implicit_iter in range(implicit_steps):
                for row in range(irk.rows):
                    x_[row+1] = x_0 + h_irk * irk.a_k_sum(eps2_, row)
                    eps2_[row], data_[row] = irk(x_0, x_[row+1], s_irk[row], h, **extra_args)
                    eps2_ = process_guides_substep(x_0, x_, eps2_, data_, row, y0, y0_inv, lgw[step], lgw_inv[step], lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_irk, unsample_resample_scale, irk, irk_type, guide_mode, latent_guide_inv, UNSAMPLE, extra_options)
                x = x_0 + h_irk * irk.b_k_sum(eps2_, 0)
                denoised = x + (sigma / (sigma - sigma_down)) *  h * irk.b_k_sum(eps2_, 0) 
                eps = x - denoised
                x = process_guides_poststep(x, denoised, eps, y0, y0_inv, mask, lgw_mask, lgw_mask_inv, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options)
                
                
                
        callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data_[0]}) if callback is not None else None

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
        x = rk.add_noise_post(x, y0, lgw[step], sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)    #y0, lgw, sigma_down are currently unused
        
        for ms in range(rk.multistep_stages):
            eps_ [rk.multistep_stages - ms] = eps_ [rk.multistep_stages - ms - 1]
            data_[rk.multistep_stages - ms] = data_[rk.multistep_stages - ms - 1]
        eps_ [0] = torch.zeros_like(eps_ [0])
        data_[0] = torch.zeros_like(data_[0])
        
    return x



def get_explicit_rk_step(rk, rk_type, x, y0, y0_inv, lgw, lgw_inv, mask, lgw_mask, lgw_mask_inv, step, sigma, sigma_next, eta, eta_var, s_noise, noise_mode, c2, c3, stepcount, sigmas, x_, eps_, data_, unsample_resample_scale, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options, **extra_args):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(rk.model, sigma, sigma_next, eta, eta_var, noise_mode, rk.h_fn(sigma_next,sigma) )
    h = rk.h_fn(sigma_down, sigma)
    c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn)
    
    rk.set_coeff(rk_type, h, c2=c2, c3=c3, stepcount=stepcount, sigmas=sigmas, sigma_down=sigma_down)

    s_ = [(sigma + h * c_) * s_in for c_ in rk.c]
    x_[0] = rk.add_noise_pre(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode)
    
    x_0 = x_[0].clone()
    
    for ms in range(rk.multistep_stages):
        if RK_Method.is_exponential(rk_type):
            eps_ [rk.multistep_stages - ms] = data_ [rk.multistep_stages - ms] - x_0
        else:
            eps_ [rk.multistep_stages - ms] = (x_0 - data_ [rk.multistep_stages - ms]) / sigma
        
    for row in range(rk.rows - rk.multistep_stages):
        x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row)
        eps_[row], data_[row] = rk(x_0, x_[row+1], s_[row], h, **extra_args)
        
        eps_ = process_guides_substep(x_0, x_, eps_, data_, row, y0, y0_inv, lgw, lgw_inv, lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, guide_mode, latent_guide_inv, UNSAMPLE, extra_options)
        
    x = x_0 + h * rk.b_k_sum(eps_, 0)
    
    denoised = x + (sigma / (sigma - sigma_down)) *  h * rk.b_k_sum(eps_, 0) 
    eps = x - denoised
    x = process_guides_poststep(x, denoised, eps, y0, y0_inv, mask, lgw_mask, lgw_mask_inv, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options)

    x = rk.add_noise_post(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode)

    for ms in range(rk.multistep_stages):
        eps_ [rk.multistep_stages - ms] = eps_ [rk.multistep_stages - ms - 1]
        data_[rk.multistep_stages - ms] = data_[rk.multistep_stages - ms - 1]

    return x, eps_, data_


