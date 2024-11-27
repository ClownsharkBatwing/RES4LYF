import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re


from tqdm.auto import trange
import math
import copy
import gc

import comfy.model_patcher

from .noise_classes import *
from .extra_samplers_helpers import get_deis_coeff_list
from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3
from .latents import hard_light_blend


def get_epsilon(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    x0 = model(x, sigma * s_in, **extra_args)
    eps = (x - x0) / (sigma * s_in) 
    return eps

def get_denoised(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    x0 = model(x, sigma * s_in, **extra_args)
    return x0



def __phi(j, neg_h):
  remainder = torch.zeros_like(neg_h)
  
  for k in range(j): 
    remainder += (neg_h)**k / math.factorial(k)
  phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
  
  return phi_j_h
  
  
def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))




from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple


def _gamma(n: int,) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(s: int, x: float, gamma_s: Optional[int] = None) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
  if s is a positive integer,
  Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
  """
  if gamma_s is None:
    gamma_s = _gamma(s)

  sum_: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    quotient: float = numerator/denom
    sum_ += quotient
  incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
  return incomplete_gamma_



def phi(j: int, neg_h: float, ):
  """
  For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = _gamma(j)
  incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)
  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)
  return phi_





def get_irk_explicit_sigmas(model, x, sigmas, eta, eta_var, noise_mode, c1, c2, c3, rk, irk, rk_type, implicit_sampler_name, t_fn_formula="", sigma_fn_formula=""):
    s_in = x.new_ones([x.shape[0]])
    irk_sigmas = torch.empty_like(sigmas)
    
    eta, eta_var = 0, 0
    
    for _ in range(len(sigmas)-1):
        sigma, sigma_next = sigmas[_], sigmas[_+1]
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, rk.h_fn(sigma_next,sigma) )
        h     =  rk.h_fn(sigma_down, sigma)
        h_irk = irk.h_fn(sigma_down, sigma)
        
        c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn, t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula)
        
        rk. set_coeff(rk_type, h, c1, c2, c3, _, sigmas, sigma, sigma_down)
        irk.set_coeff(implicit_sampler_name, h_irk, c1, c2, c3, _, sigmas, sigma, sigma_down)
        
        s_irk    = [(  irk.sigma_fn(irk.t_fn(sigma) + h*c_)) * s_in for c_ in  irk.c]
        
        s_irk.append(sigma)
        s_all = sorted(set(s_irk), reverse=True)
        s_irk = s_irk[:-1]

        s_all[0] = s_all[0].unsqueeze(dim=0)
        s_all_sigmas = torch.stack(s_all, dim=0).squeeze(dim=1)
        
        
        irk_sigmas = torch.cat((irk_sigmas, s_all_sigmas), dim=0)
        
    return irk_sigmas





from .rk_method import RK_Method, RK_Method_Linear, RK_Method_Exponential


@torch.no_grad()
def sample_rk_beta(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="brownian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="default",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, scale=0.1, c1=0.0, c2=0.5, c3=1.0, MULTISTEP=False, cfgpp=0.0, implicit_steps=0, reverse_weight=0.0, exp_mode=False,
                  latent_guide=None, latent_guide_inv=None, latent_guide_weights=None, latent_guide_weights_inv=None, guide_mode="blend", unsampler_type="linear",
                  GARBAGE_COLLECT=False, mask=None, LGW_MASK_RESCALE_MIN=True, sigmas_override=None, t_is=None,sde_noise=None,
                  input_std=1.0, input_normalization="channels", extra_options="",
                  etas=None, s_noises=None, momentums=None,
                  ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    if len(sde_noise) > 0 and sigmas[1] > sigmas[2]:
        SDE_NOISE_EXTERNAL = True
    else:
        SDE_NOISE_EXTERNAL = False
    
    lgw = latent_guide_weights.to(x.device)
    lgw_inv = latent_guide_weights_inv.to(x.device)
    
    if sigmas_override is not None:
        sigmas = sigmas_override.clone()
    sigmas = sigmas.clone() * d_noise
    
    if rk_type.startswith("dpmpp") or rk_type.startswith("res") or rk_type.startswith("rk_exp") or rk_type.startswith("ddim"):
        rk  = RK_Method_Exponential(model, "midpoint_2s", "explicit", x.device)
    else:
        rk  = RK_Method_Linear(model, "midpoint_2s", "explicit", x.device)
    rk.init_noise_sampler(x, noise_seed, noise_sampler_type, alpha=alpha, k=k)

    irk = RK_Method_Linear(model, "crouzeix_2s", "implicit", x.device)
    irk.init_noise_sampler(x, noise_seed+1, noise_sampler_type, alpha=alpha, k=k)

    sigmas, UNSAMPLE = rk.prepare_sigmas(sigmas)
    mask, LGW_MASK_RESCALE_MIN = rk.prepare_mask(x, mask, LGW_MASK_RESCALE_MIN)

    x, y0, y0_inv = rk.init_guides(x, latent_guide, latent_guide_inv, mask, sigmas, UNSAMPLE)
    
    if guide_mode == "epsilon_match_mean_std":
        ks3 = torch.zeros_like(x)
        for n in range(y0_inv.shape[1]):
            ks3[0][n] = (y0[0][n] - y0[0][n].mean()) / y0[0][n].std()
            ks3[0][n] = (ks3[0][n] * y0_inv[0][n].std()) + y0_inv[0][n].mean()
        y0 = ks3

    x_guide_maybe = x
    
    if input_normalization == "channels_mean_std":
        for i in range(x.shape[1]):
            x[0][i] = (x[0][i] - x[0][i].mean()) * (input_std / x[0][i].std())
    if input_normalization == "channels_std":
        for i in range(x.shape[1]):
            x[0][i] = (x[0][i]) * (input_std / x[0][i].std())
    if input_normalization == "mean_std":
        x = (x - x.mean()) * (input_std / x.std())
    if input_normalization == "std":
        x = x * (input_std / x.std())
    if input_normalization == "process_latent_in":
        x = model.inner_model.inner_model.process_latent_in(x).clone().to(x.device)
        
    sigma_up_total = torch.zeros_like(sigmas[0])
    if SDE_NOISE_EXTERNAL:
        for i3 in range(len(sde_noise)-1):
            #sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigmas[i3], sigmas[i3+1], eta, eta, noise_mode, rk.h_fn(sigmas[i3+1],sigmas[i3]) )
            sigma_up_total += sigmas[i3+1]
        eta = eta / sigma_up_total

    uncond = [torch.full_like(x, 0.0)]
    if cfgpp != 0.0:
        def post_cfg_function(args):
            uncond[0] = args["uncond_denoised"]
            return args["denoised"]
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)  

    for _ in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[_], sigmas[_+1]
        t_i = t_is[_] if t_is is not None else None
        
        if sigma_next == 0.0:
            rk_type = "euler"
            rk  = RK_Method_Linear(model, "midpoint_2s", "explicit", x.device)
            rk.init_noise_sampler(x, noise_seed, noise_sampler_type, alpha=alpha, k=k)
            implicit_steps = 0
            eta, eta_var = 0, 0

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, rk.h_fn(sigma_next,sigma) )
        h     =  rk.h_fn(sigma_down, sigma)
        h_irk = irk.h_fn(sigma_down, sigma)
        
        c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn, t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula)
        
        rk. set_coeff(rk_type, h, c1, c2, c3, _, sigmas, sigma, sigma_down)
        irk.set_coeff(implicit_sampler_name, h_irk, c1, c2, c3, _, sigmas, sigma, sigma_down)
        
        if _ == 0:
            x_, data_, data_u, eps_ = (torch.zeros(max(rk.rows, irk.rows) + 2, *x.shape, dtype=x.dtype, device=x.device) for _ in range(4))
        
        s_       = [(  rk.sigma_fn( rk.t_fn(sigma) +     h*c_)) * s_in for c_ in   rk.c]
        s_irk_rk = [(  rk.sigma_fn( rk.t_fn(sigma) +     h*c_)) * s_in for c_ in  irk.c]
        s_irk    = [( irk.sigma_fn(irk.t_fn(sigma) + h_irk*c_)) * s_in for c_ in  irk.c]

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if _ >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[_]
        x_[0] = rk.add_noise_pre(x, y0, lgw[_], sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t) #y0, lgw, sigma_down are currently unused
        
        x_0 = x_[0].clone()
        
        for ms in range(rk.multistep_stages):
            if rk_type.startswith("dpmpp") or rk_type.startswith("res") or rk_type.startswith("rk_exp"):
                eps_ [rk.multistep_stages - ms] = data_ [rk.multistep_stages - ms] - x_0
            else:
                eps_ [rk.multistep_stages - ms] = (x_0 - data_ [rk.multistep_stages - ms]) / sigma
            
        if LGW_MASK_RESCALE_MIN: 
            lgw_mask = mask * (1 - lgw[_]) + lgw[_]
            #lgw_mask_inv = (1-mask) * (1 - lgw[_]) + lgw[_]
            lgw_mask_inv = (1-mask) * (1 - lgw_inv[_]) + lgw_inv[_]
        else:
            if latent_guide is not None:
                lgw_mask = mask * lgw[_]
            else:
                lgw_mask = torch.zeros_like(mask)
            if latent_guide_inv is not None:
                #lgw_mask_inv = (1-mask) * lgw[_]   
                lgw_mask_inv = (1-mask) * lgw_inv[_]   
            else:
                lgw_mask_inv = torch.zeros_like(mask)
            
        if implicit_steps == 0:
            for row in range(rk.rows - rk.multistep_stages):
                x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row)

                eps_[row], data_[row] = rk(x_0, x_[row+1], s_[row], h, **extra_args)
                
                if latent_guide_inv is None:
                    y0_tmp = y0
                    y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0

                elif latent_guide_inv is not None:
                    y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
                    y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv

                if "data" in guide_mode:
                    x_[row+1] = y0_tmp + eps_[row]

                elif "epsilon" in guide_mode:
                    if sigma > sigma_next:
                         
                        if re.search(r"\bdisable_lgw_scaling\b", extra_options): # and lgw[_] > 0:
                            if rk_type.startswith("dpmpp") or rk_type.startswith("res") or rk_type.startswith("rk_exp"):
                                eps_m   = y0     - x_0
                                eps_inv = y0_inv - x_0
                            else:
                                eps_m   = (x_[row+1] - y0)     / (s_[row] * s_in)
                                eps_inv = (x_[row+1] - y0_inv) / (s_[row] * s_in)
                                
                            eps_[row] = eps_[row]      +     lgw_mask * (eps_m - eps_[row])    +    lgw_mask_inv * (eps_inv - eps_[row])
                                
                        elif re.search(r"tol\s*=\s*([\d.]+)", extra_options) and (lgw[_] > 0 or lgw_inv[_] > 0):
                            tol_value = float(re.search(r"tol\s*=\s*([\d.]+)", extra_options).group(1))                    
                            for i4 in range(x.shape[1]):
                                current_diff     = torch.norm(data_[row][0][i4] - y0    [0][i4]) 
                                current_diff_inv = torch.norm(data_[row][0][i4] - y0_inv[0][i4]) 
                                
                                lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                                lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                                
                                lgw_tmp     = min(lgw    [_], lgw_scaled)
                                lgw_tmp_inv = min(lgw_inv[_], lgw_scaled_inv)

                                lgw_mask_clamp = torch.clamp(lgw_mask, max=lgw_tmp)
                                lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)

                                if rk_type.startswith("dpmpp") or rk_type.startswith("res") or rk_type.startswith("rk_exp"):
                                    eps_row     = y0    [0][i4] - x_0[0][i4]
                                    eps_row_inv = y0_inv[0][i4] - x_0[0][i4]
                                else:
                                    eps_row     = (x_[row+1][0][i4] - y0    [0][i4]) / (s_[row] * s_in)
                                    eps_row_inv = (x_[row+1][0][i4] - y0_inv[0][i4]) / (s_[row] * s_in)
                                    
                                eps_[row][0][i4] = eps_[row][0][i4] + lgw_mask_clamp[0][i4] * (eps_row - eps_[row][0][i4]) + lgw_mask_clamp_inv[0][i4] * (eps_row_inv - eps_[row][0][i4])
                                
                        elif (lgw[_] > 0 or lgw_inv[_] > 0):
                            avg, avg_inv = 0, 0
                            for i4 in range(x.shape[1]):
                                avg     += torch.norm(data_[row][0][i4] - y0    [0][i4])
                                avg_inv += torch.norm(data_[row][0][i4] - y0_inv[0][i4])
                            avg /= x.shape[1]
                            avg_inv /= x.shape[1]
                            
                            for i4 in range(x.shape[1]):
                                ratio     = torch.nan_to_num(torch.norm(data_[row][0][i4] - y0    [0][i4])   /   avg,     0)
                                ratio_inv = torch.nan_to_num(torch.norm(data_[row][0][i4] - y0_inv[0][i4])   /   avg_inv, 0)
                                
                                if rk_type.startswith("dpmpp") or rk_type.startswith("res") or rk_type.startswith("rk_exp"):
                                    eps_row     = y0    [0][i4] - x_0[0][i4]
                                    eps_row_inv = y0_inv[0][i4] - x_0[0][i4]
                                else:
                                    eps_row     = (x_[row+1][0][i4] - y0    [0][i4]) / (s_[row] * s_in)
                                    eps_row_inv = (x_[row+1][0][i4] - y0_inv[0][i4]) / (s_[row] * s_in)
                                                                    
                                eps_[row][0][i4] = eps_[row][0][i4]      +     ratio * lgw_mask[0][i4] * (eps_row - eps_[row][0][i4])    +    ratio_inv * lgw_mask_inv[0][i4] * (eps_row_inv - eps_[row][0][i4])


                    else:
                        y0_tmp = (1-lgw[_]) * data_[row]    +   lgw[_] * x_guide_maybe
                        x_plus1 = y0_tmp + eps_[row]
                        eps_[row] = (y0 - x_plus1)   / (s_[row] * s_in)

                elif (UNSAMPLE or "resampler" in guide_mode) and lgw[_] > 0:
                    y0_tmp = y0
                    if latent_guide_inv is not None:
                        y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
                        y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
                        
                    cvf = rk.get_epsilon(x_0, x_[row+1], y0, sigma, s_[row], sigma_down, t_i)
                    if UNSAMPLE and sigma > sigma_next and latent_guide_inv is not None:
                        cvf_inv = rk.get_epsilon(x_0, x_[row+1], y0_inv, sigma, s_[row], sigma_down, t_i)      
                        cvf = (1-lgw_mask)     * eps_[row] + lgw_mask     * cvf
                        cvf = (1-lgw_mask_inv) * cvf       + lgw_mask_inv * cvf_inv

                    if re.search(r"tol\s*=\s*([\d.]+)", extra_options):
                        tol_value = float(re.search(r"tol\s*=\s*([\d.]+)", extra_options).group(1))                    
                        for i4 in range(x.shape[1]):
                            current_diff = torch.norm(data_[row][0][i4] - y0_tmp[0][i4]) 
                            lgw_tmp = min(lgw[_], 1-(tol_value/current_diff))
                            eps_[row][0][i4] = eps_[row][0][i4]  + lgw_tmp * (cvf[0][i4]  - eps_[row][0][i4] )
                    elif re.search(r"\bdisable_lgw_scaling\b", extra_options):
                        eps_[row] = eps_[row] + lgw[_] * (cvf - eps_[row])
                    else:
                        avg = 0
                        for i4 in range(x.shape[1]):
                            avg += torch.norm(data_[row][0][i4] - y0_tmp[0][i4])
                        avg /= x.shape[1]
                        
                        for i4 in range(x.shape[1]):
                            ratio = torch.norm(data_[row][0][i4] - y0_tmp[0][i4])   /   avg
                            lgw_tmp = lgw[_] * ratio
                            eps_[row][0][i4] = eps_[row][0][i4]  + lgw_tmp * (cvf[0][i4]  - eps_[row][0][i4] )
                    
            x = x_0 + h * rk.b_k_sum(eps_, 0) 
            
            #denoised = x_0 + (sigma / (sigma - sigma_down)) *  h * rk.b_k_sum(eps_, 0) 
            #ps = x_0 - denoised
            
            denoised = x + (sigma / (sigma - sigma_down)) *  h * rk.b_k_sum(eps_, 0) 
            eps = x - denoised
            
            if guide_mode == "epsilon_mean_std":
                #denoised_mask     = denoised[mask == 0]
                #denoised_mask_inv = denoised[mask == 1]
                #denoised_mask_intermediate = denoised[(mask > 0) & (mask < 1)]
                
                denoised_masked     = denoised * ((mask==1)*mask)
                denoised_masked_inv = denoised * ((mask==0)*(1-mask))
                denoised_masked_intermediate = denoised - denoised_masked - denoised_masked_inv
                #denoised_masked_intermediate = denoised * (((mask > 0) & (mask < 1))*mask)
                
                ks3 = torch.zeros_like(x)
                for n in range(denoised.shape[1]):
                    denoised_mask     = denoised[0][n][mask[0][n] == 1]
                    denoised_mask_inv = denoised[0][n][mask[0][n] == 0]
                    
                    
                    ks3[0][n] = (denoised_masked[0][n] - denoised_mask.mean()) / denoised_mask.std()
                    #ks3[0][n] = (ks3[0][n] * y0_inv[0][n].std()) + y0_inv[0][n].mean()
                    ks3[0][n] = (ks3[0][n] * denoised_mask_inv.std()) + denoised_mask_inv.mean()
                    
                #denoised_masked = ks3
                x = 0.005 * ks3 + (1-0.005) * denoised_masked           + denoised_masked_intermediate +  denoised_masked_inv + eps
                #x = lgw[_] * ks3 + (1-lgw[_]) * denoised_masked           + denoised_masked_intermediate +  denoised_masked_inv + eps
                #x = denoised_masked_intermediate + lgw[_] * ks3 + (1-lgw[_]) * denoised_masked_intermediate +  denoised_masked_inv + eps
            
            
            if UNSAMPLE == False and latent_guide is not None and lgw[_] > 0:
                y0_tmp = y0
                if latent_guide_inv is not None:
                    y0_tmp = (1-lgw_mask) * denoised + lgw_mask * y0
                    y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
                if guide_mode == "hard_light":
                    blend = hard_light_blend(y0, denoised)
                    denoised = (1-lgw_mask) * denoised + lgw_mask * blend
                    if latent_guide_inv is not None:
                        blend_inv = hard_light_blend(y0_inv, denoised)
                        denoised = (1-lgw_mask_inv) * denoised + lgw_mask_inv * blend_inv
                    x = denoised + eps
                elif guide_mode == "blend":
                    denoised = (1-lgw_mask) * denoised + lgw_mask * y0
                    denoised = (1-lgw_mask_inv) * denoised + lgw_mask_inv * y0_inv
                    x = denoised + eps

                elif guide_mode == "mean_std":
                    ks2 = torch.zeros_like(x)
                    for n in range(y0.shape[1]):
                        ks2[0][n] = (denoised[0][n] - denoised[0][n].mean()) / denoised[0][n].std()
                        ks2[0][n] = (ks2[0][n] * y0[0][n].std()) + y0[0][n].mean()
                        
                    ks3 = torch.zeros_like(x)
                    for n in range(y0_inv.shape[1]):
                        ks3[0][n] = (denoised[0][n] - denoised[0][n].mean()) / denoised[0][n].std()
                        ks3[0][n] = (ks3[0][n] * y0_inv[0][n].std()) + y0_inv[0][n].mean()
                        
                    denoised = (1 - lgw_mask) * denoised   +   lgw_mask * ks2
                    denoised = (1 - lgw_mask_inv) * denoised   +   lgw_mask_inv * ks3
                    x = denoised + eps
                    
                elif guide_mode == "mean":
                    ks2 = torch.zeros_like(x)
                    for n in range(y0.shape[1]):
                        ks2[0][n] = (denoised[0][n] - denoised[0][n].mean())
                        ks2[0][n] = (ks2[0][n]) + y0[0][n].mean()
                        
                    ks3 = torch.zeros_like(x)
                    for n in range(y0_inv.shape[1]):
                        ks3[0][n] = (denoised[0][n] - denoised[0][n].mean())
                        ks3[0][n] = (ks3[0][n]) + y0_inv[0][n].mean()
                        
                    denoised = (1 - lgw_mask) * denoised   +   lgw_mask * ks2
                    denoised = (1 - lgw_mask_inv) * denoised   +   lgw_mask_inv * ks3
                    x = denoised + eps
                    
                elif guide_mode == "std":
                    ks2 = torch.zeros_like(x)
                    for n in range(y0.shape[1]):
                        ks2[0][n] = (denoised[0][n]) / denoised[0][n].std()
                        ks2[0][n] = (ks2[0][n] * y0[0][n].std())
                        
                    ks3 = torch.zeros_like(x)
                    for n in range(y0_inv.shape[1]):
                        ks3[0][n] = (denoised[0][n]) / denoised[0][n].std()
                        ks3[0][n] = (ks3[0][n] * y0_inv[0][n].std())
                        
                    denoised = (1 - lgw_mask) * denoised   +   lgw_mask * ks2
                    denoised = (1 - lgw_mask_inv) * denoised   +   lgw_mask_inv * ks3
                    x = denoised + eps
                    #if latent_guide_inv is not None:
                    #    blend_inv = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
                
            


        else:
            s2 = s_irk_rk[:]
            s2.append(sigma.unsqueeze(dim=0))
            s_all = torch.sort(torch.stack(s2, dim=0).squeeze(dim=1).unique(), descending=True)[0]
            sigmas_and = torch.cat( (sigmas[0:_], s_all), dim=0)
            
            eps_ [0] = torch.zeros_like(eps_ [0])
            data_ [0] = torch.zeros_like(data_[0])
            eps_list = []
            
            x_mid = x
            for i in range(len(s_all)-1):
                x_mid, eps_mid, denoised_mid, eps_, data_ = get_explicit_rk_step(rk, rk_type, x_mid, y0, lgw, s_all[i], s_all[i+1], eta, eta_var, s_noise, noise_mode, c2, c3, _+i, sigmas_and, x_, eps_, data_, **extra_args)
                eps_list.append(eps_[0])

                eps_ [0] = torch.zeros_like(eps_ [0])
                data_[0] = torch.zeros_like(data_[0])
                
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
                    
                    cvf = irk.get_epsilon(x_0, x_[row+1], y0, sigma, s_irk[row], sigma_down)
                    eps2_[row] = eps2_[row] + lgw[_] * (cvf - eps2_[row])
                x = x_0 + h_irk * irk.b_k_sum(eps2_, 0)
            
        #print("x stats: ", x.std(), x.mean(), x.abs().mean(), x.max())
        #callback({'x': x, 'i': _, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised}) if callback is not None else None

        callback({'x': x, 'i': _, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data_[0]}) if callback is not None else None

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if _ >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[_]
        x = rk.add_noise_post(x, y0, lgw[_], sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)    #y0, lgw, sigma_down are currently unused
        
        for ms in range(rk.multistep_stages):
            eps_ [rk.multistep_stages - ms] = eps_ [rk.multistep_stages - ms - 1]
            data_[rk.multistep_stages - ms] = data_[rk.multistep_stages - ms - 1]
        eps_ [0] = torch.zeros_like(eps_ [0])
        data_[0] = torch.zeros_like(data_[0])
        
    #print("x stats (end): ", x.std(), x.mean(), x.abs().mean(), x.max())
    return x



def get_explicit_rk_step(rk, rk_type, x, y0, lgw, sigma, sigma_next, eta, eta_var, s_noise, noise_mode, c2, c3, stepcount, sigmas, x_, eps_, data_, **extra_args):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(rk.model, sigma, sigma_next, eta, eta_var, noise_mode, rk.h_fn(sigma_next,sigma) )
    h = rk.h_fn(sigma_down, sigma)
    c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn)
    
    rk.set_coeff(rk_type, h, c2=c2, c3=c3, stepcount=stepcount, sigmas=sigmas, sigma_down=sigma_down)

    s_ = [(sigma + h * c_) * s_in for c_ in rk.c]
    #x_, eps_, data_, data_u_ = (torch.zeros(rk.rows + 2, *x.shape, dtype=x.dtype, device=x.device) for _ in range(4))
    x_[0] = rk.add_noise_pre(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode)
    
    x_0 = x_[0].clone()
    
    for ms in range(rk.multistep_stages):
        if rk_type.startswith("dpmpp") or rk_type.startswith("res") or rk_type.startswith("rk_exp"):
            eps_ [rk.multistep_stages - ms] = data_ [rk.multistep_stages - ms] - x_0
        else:
            eps_ [rk.multistep_stages - ms] = (x_0 - data_ [rk.multistep_stages - ms]) / sigma
        
    for row in range(rk.rows - rk.multistep_stages):
        x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row)
        eps_[row], data_[row] = rk(x_0, x_[row+1], s_[row], h, **extra_args)
        
        cvf = rk.get_epsilon(x_0, x_[row+1], y0, sigma, s_[row], sigma_down)
        eps_[row] = eps_[row] + lgw[stepcount] * (cvf - eps_[row])
    x = x_0 + h * rk.b_k_sum(eps_, 0)

    denoised = rk.b_k_sum(data_, 0) / sum(rk.b[0])
    eps = rk.b_k_sum(eps_, 0) / sum(rk.b[0])
    
    x = rk.add_noise_post(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode)

    for ms in range(rk.multistep_stages):
        eps_ [rk.multistep_stages - ms] = eps_ [rk.multistep_stages - ms - 1]
        data_[rk.multistep_stages - ms] = data_[rk.multistep_stages - ms - 1]

    return x, eps, denoised, eps_, data_






