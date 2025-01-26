import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re
import copy

from tqdm.auto import trange
import gc

import comfy.model_patcher

from .noise_classes import *
from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_alpha_ratio_from_sigma_down, get_res4lyf_half_step3

from .rk_method_beta import RK_Method_Beta
from .rk_guide_func_beta import *

from .latents import normalize_latent, initialize_or_scale, latent_normalize_channels
from .helper import get_extra_options_kv, extra_options_flag, get_cosine_similarity
from .sigmas import get_sigmas

from .rk_coefficients_beta import IRK_SAMPLER_NAMES_BETA
from .phi_functions import Phi


PRINT_DEBUG=False


def prepare_sigmas(model, sigmas):
    if sigmas[0] == 0.0:      #remove padding used to prevent comfy from adding noise to the latent (for unsampling, etc.)
        UNSAMPLE = True
        sigmas = sigmas[1:-1]
    else: 
        UNSAMPLE = False
        
    if hasattr(model, "sigmas"):
        model.sigmas = sigmas
        
    return sigmas, UNSAMPLE


def prepare_step_to_sigma_zero(rk, rk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=1.0, **extra_args):
    rk_type_final_step = f"euler" if rk_type in {"euler"} else rk_type
    rk_type_final_step = f"ralston_{rk_type[-2:]}" if rk_type[-2:] in {"2s", "3s"} else "ralston_3s"
    rk_type_final_step = f"deis_2m" if rk_type[-2:] in {"2m", "3m", "4m"} else rk_type_final_step
    rk_type_final_step = f"euler" if rk_type in {"ddim", "euler"} else rk_type_final_step
    rk_type_final_step = get_extra_options_kv("rk_type_final_step", rk_type_final_step, extra_options)
    rk = RK_Method_Beta.create(model, rk_type_final_step, x.device)
    rk.init_noise_sampler(x, torch.initial_seed() + 1, noise_sampler_type, alpha=alpha, k=k)
    extra_args =  rk.init_cfg_channelwise(x, cfg_cw, **extra_args)

    eta, eta_var = 0, 0
    return rk, rk_type_final_step, eta, eta_var, extra_args

def get_epsilon(x_0, denoised, sigma, rk_type):
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

def debug_cuda_cleanup(doSync=False, doEmpty=False, doGC=False) -> None:
    if doSync:
        torch.cuda.synchronize()
    if doEmpty:
        torch.cuda.empty_cache()
    if doGC:
        import gc
        gc.collect()

@torch.no_grad()
def sample_rk_beta(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="explicit_full",
                   sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, c1=0.0, c2=0.5, c3=1.0, implicit_steps_diag=0, implicit_steps_full=0, 
                  LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, etas_substep=None, s_noises=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights_grp=None, eta_substep=0.0, noise_mode_sde_substep="hard",
                  ):
    
    cuda_sync_a_flag = extra_options_flag("cuda_sync_a", extra_options)
    cuda_empty_a_flag = extra_options_flag("cuda_empty_a", extra_options)
    cuda_gc_a_flag = extra_options_flag("cuda_gc_a", extra_options)
    cuda_sync_b_flag = extra_options_flag("cuda_sync_b", extra_options)
    cuda_empty_b_flag = extra_options_flag("cuda_empty_b", extra_options)
    cuda_gc_b_flag = extra_options_flag("cuda_gc_b", extra_options)

    extra_args = {} if extra_args is None else extra_args
    
    if noise_seed < 0:
        noise_seed = torch.initial_seed()+1 
        print("Set noise_seed to: ", noise_seed, " using torch.initial_seed()+1")

    c1 = c1_ = float(get_extra_options_kv("c1", str(c1), extra_options))
    c2 = c2_ = float(get_extra_options_kv("c2", str(c2), extra_options))
    c3 = c3_ = float(get_extra_options_kv("c3", str(c3), extra_options))
    
    newton_iter_post = int(get_extra_options_kv("newton_iter_post", str("0"), extra_options))
    newton_iter_pre = int(get_extra_options_kv("newton_iter_pre", str("0"), extra_options))
    newton_iter_mixing_rate = float(get_extra_options_kv("newton_iter_mixing_rate", str("0.5"), extra_options))

    guide_skip_steps = int(get_extra_options_kv("guide_skip_steps", 0, extra_options))        
    eta_substep_cutoff_step = int(get_extra_options_kv("eta_substep_cutoff_step", '10000', extra_options))
    default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "float64", extra_options), torch.float64)   
    cfg_cw = float(get_extra_options_kv("cfg_cw", str(cfg_cw), extra_options))
    
    MODEL_SAMPLING = model.inner_model.inner_model.model_sampling
    
    #s_in, s_one = x.new_ones([x.shape[0]]), x.new_ones([1])
    max_steps=10000
    
    if sigmas_override is not None:
        sigmas = sigmas_override.clone()
    sigmas = sigmas.clone() * d_noise
    sigmas, UNSAMPLE = prepare_sigmas(model, sigmas)
    
    SDE_NOISE_EXTERNAL = False
    if sde_noise is not None:
        if len(sde_noise) > 0 and sigmas[1] > sigmas[2]:
            SDE_NOISE_EXTERNAL = True
            sigma_up_total = torch.zeros_like(sigmas[0])
            for i in range(len(sde_noise)-1):
                sigma_up_total += sigmas[i+1]
            eta = eta / sigma_up_total

    if implicit_sampler_name not in ("use_explicit", "none"): # and implicit_steps_full + implicit_steps_diag > 0:
        rk_type = implicit_sampler_name
    print("rk_type: ", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0

    rk = RK_Method_Beta.create(model,  rk_type, x.device)
    rk. init_noise_sampler(x, noise_seed, noise_sampler_type, alpha=alpha, k=k)
    extra_args = rk.init_cfg_channelwise(x, cfg_cw, **extra_args)


    LG = LatentGuide(guides, x, model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options, frame_weights_grp=frame_weights_grp)
    x = LG.init_guides(x, rk.noise_sampler)
    
    y0, y0_inv = LG.y0, LG.y0_inv

    denoised = torch.zeros_like(x)
    denoised_prev = torch.zeros_like(x)
    denoised_prev2 = torch.zeros_like(x).to('cpu')
    eps = torch.zeros_like(x)
    data_prev = torch.zeros_like(x).to('cpu')
    s_prev = None
    
    torch.cuda.empty_cache()
    gc.collect()

    for step in trange(len(sigmas)-1, disable=disable):

        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        unsample_resample_scale = float(unsample_resample_scales[step]) if unsample_resample_scales is not None else None
        
        if regional_conditioning_weights is not None:
            extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = regional_conditioning_weights[step]
            extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = regional_conditioning_floors [step]
        else:
            extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
            extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0
        
        eta = eta_var = etas[step] if etas is not None else eta
        eta_substep = eta_var_substep = etas_substep[step] if etas_substep is not None else eta_substep
        eta_substep = 0.0 if step >= eta_substep_cutoff_step else eta_substep
        s_noise = s_noises[step] if s_noises is not None else s_noise
        
        if sigma_next == 0:
            rk, rk_type, eta, eta_var, extra_args = prepare_step_to_sigma_zero(rk, rk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=cfg_cw, **extra_args)
        if step == len(sigmas)-2:
            print("cut noise at step: ", step)
            eta = 0.0

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, extra_options)
        h = h_new = rk.h_fn(sigma_down, sigma)
        h_no_eta = rk.h_fn(sigma_next, sigma)

        rk. set_coeff(rk_type, h, c1, c2, c3, step, sigmas, sigma_down, extra_options)

        s_        = [(rk.sigma_fn(rk.t_fn(sigma) +        h*c_)) * x.new_ones([1]) for c_ in rk.c]
        #s_no_eta_ = [(rk.sigma_fn(rk.t_fn(sigma) + h_no_eta*c_)) * x.new_ones([1]) for c_ in rk.c]

        recycled_stages = max(rk.multistep_stages, rk.hybrid_stages)
        if step == 0 or step == guide_skip_steps:
            #x_, data_, denoised_, eps_ = (torch.zeros(rk.rows+2, *x.shape, dtype=x.dtype, device=x.device) for step in range(4))
            x_ = torch.zeros(rk.rows+2, *x.shape, dtype=x.dtype, device=x.device)
            data_ = torch.zeros(min(3, rk.rows+2), *x.shape, dtype=x.dtype, device=x.device)  # Only 3 rows needed max
            denoised_ = torch.zeros(recycled_stages+1, *x.shape, dtype=x.dtype, device=x.device)  # Only recycled_stages+1 needed
            eps_ = torch.zeros(rk.rows+2, *x.shape, dtype=x.dtype, device=x.device)

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
        
        x_[0] = x
        if sigma_up > 0:
            x_[0] = rk.add_noise_pre(x_[0], sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t) #y0, lgw, sigma_down are currently unused
        
        x_0 = x_[0].clone()
        
        for ms in range(recycled_stages):
            eps_ [recycled_stages - ms] = get_epsilon(x_0, denoised_ [recycled_stages - ms], sigma, rk_type)
                
        eps_prev_ = eps_.clone()

        if rk_type in IRK_SAMPLER_NAMES_BETA:
            if extra_options_flag("implicit_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
                eps_[0], data_[0] = eps.clone(), denoised.clone()
                eps_[0] = get_epsilon(x_0, denoised, sigma, rk_type)
                """if denoised_prev2.sum() != 0:
                    sratio = sigma - s_[0]
                    
                    data_[0] = denoised + sratio * (denoised - denoised_prev2)"""

                # RES_2M GUESS HERE
                if denoised_prev2.sum() != 0 and extra_options_flag("implicit_res_2m_denoised_guess", extra_options):
                    data_prev = data_prev.to(x.device)

                    h_prev1 = -torch.log(sigmas[step]/sigmas[step-1])
                    h_curr = -torch.log(s_[0]/sigma)
                                        
                    c1,c2 = 0, 1/2
                    c2 = (-h_prev1 / h_curr).item()

                    ci = [c1,c2]
                    #h = -torch.log(sigma_next/sigma)
                    φ = Phi(h_curr, ci, analytic_solution=True)
                    
                    a2_1 = c2 * φ(1,2)
                    b2 = φ(2)/c2
                    b1 = φ(1) - b2
                    
                    print("denoised prev c2: ", (-h_prev1 / h_curr).item(), h_curr.item(), b1.item(), b2.item())
                    
                    
                    
                    h_prev1 = -torch.log(s_prev/s_prev2)
                    h_curr = -torch.log(s_[0]/s_prev)

                    c1,c2 = 0, 1/2
                    c2 = (-h_prev1 / h_curr).item()

                    ci = [c1,c2]
                    #h = -torch.log(sigma_next/sigma)
                    φ = Phi(h_curr, ci)
                    
                    a2_1 = c2 * φ(1,2)
                    b2 = φ(2)/c2
                    b1 = φ(1) - b2
                    
                    print("data prev c2: ", (-h_prev1 / h_curr).item(), h_curr.item(), b1.item(), b2.item())
                
                    #x_2 = torch.exp(-h * c2) * x_0 + h * (a2_1 * denoised_prev2)
                    if c2 < -0.25 or c2 > 0.25:
                        x_[0] = torch.exp(-h_curr) * x_prev + h_curr * (b1 * data_prev2 + b2 * data_prev)
                        data_[0] = get_data_from_step(x_prev, x_[0], s_prev, s_[0])  
                        #data_[0] = get_data_from_step(x_0, x_[0], sigma, s_[0]) 
                        eps_[0] = get_epsilon(x_[0], data_[0], s_[0], rk_type)
                    
                    
                    h_prev1 = -torch.log(s_prev/s_prev2)
                    h_curr = -torch.log(s_[1]/s_prev)
                    
                    c1,c2 = 0, 1/2
                    c2 = (-h_prev1 / h_curr).item()

                    ci = [c1,c2]
                    #h = -torch.log(sigma_next/sigma)
                    φ = Phi(h_curr, ci)
                    
                    a2_1 = c2 * φ(1,2)
                    b2 = φ(2)/c2
                    b1 = φ(1) - b2
                    
                    print("data prev c2 substep: ", (-h_prev1 / h_curr).item(), h_curr.item(), b1.item(), b2.item())
                
                    #x_2 = torch.exp(-h * c2) * x_0 + h * (a2_1 * denoised_prev2)
                    if c2 < -0.25 or c2 > 0.25:
                        x_[1] = torch.exp(-h_curr) * x_prev + h_curr * (b1 * data_prev2 + b2 * data_prev)
                        data_[1] = get_data_from_step(x_prev, x_[1], s_prev, s_[1])  
                        #data_[1] = get_data_from_step(x_0, x_[1], sigma, s_[1]) 
                        eps_[1] = get_epsilon(x_[1], data_[1], s_[1], rk_type)

                    data_prev = data_prev.to('cpu')

            else:
                eps_[0], data_[0] = rk(x_[0], sigma, x_0, sigma, **extra_args) 
                
            for r in range(rk.rows):
                eps_ [r] = eps_ [0].clone() * sigma / s_[r]
                data_[r] = data_[0].clone()

            newton_iter_ynyt = int(get_extra_options_kv("newton_iter_ynyt", str("0"), extra_options))
            for n_iter_ynyt in range(newton_iter_ynyt):
                for r in range(rk.rows):
                    x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                for r in range(rk.rows):
                    eps_[r] = get_epsilon_from_step(x_0, x_[r], sigma, s_[r])
            
            eps_0  = eps_ [0].clone().to('cpu')
            data_0 = data_[0].clone().to('cpu') 

            newton_iter_init = int(get_extra_options_kv("newton_iter_init", str("0"), extra_options))
            if step >= len(sigmas)-6:
                newton_iter_init = 0
            for n_iter_init in range(newton_iter_init):
                for r in range(0, rk.rows+1): #+1):
                    eps_[0] = eps_0.clone().to(x.device)
                    x_tmp, eps_tmp = x_[r].clone(), eps_[r].clone()
                    if r < rk.rows:
                        x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                    else:
                        x_[r] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                    if sigma == s_[r]:
                        continue
                    #x_[r] = x_[r] + newton_iter_mixing_rate * (x_tmp - x_[r])
                    data_[r] = get_data_from_step(x_0, x_[r], sigma, s_[r])  
                    eps_ [r] = get_epsilon(x_0, data_[r], s_[r], rk_type)
                    
                    #for r2 in range(0, rk.rows+1): 
                    #    if r != r2:
                    #        eps_ [r] = get_orthogonal(eps_[r2], eps_[r])
                    
                    x_[r] = x_tmp + newton_iter_mixing_rate * (x_[r] - x_tmp)
                    eps_[r] = eps_tmp + newton_iter_mixing_rate * (eps_[r] - eps_tmp)
                    
            newton_iter_alt_init = int(get_extra_options_kv("newton_iter_alt_init", str("0"), extra_options))
            if step == len(sigmas)-4:
                newton_iter_alt_init = 0
            for n_iter_alt_init in range(newton_iter_alt_init):
                for r in range(rk.rows+1): #+1):
                    for blah in range(100):
                        eps_[0] = eps_0.clone()
                        if r < rk.rows:
                            x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                        else:
                            x_[r] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))

                        data_[r] = get_data_from_step(x_0, x_[r], sigma, s_[r])  
                        eps_ [r] = get_epsilon(x_0, data_[r], s_[r], rk_type)


        if extra_options_flag("guide_fully_pseudoimplicit_power_substep", extra_options) or extra_options_flag("inject_fully_pseudoimplicit_power_substep", extra_options) or extra_options_flag("guide_relaxing_fully_pseudoimplicit_power_substep", extra_options):
            inj_proj_weight = inject_fully_pseudoimplicit_power_substep_projection = float(get_extra_options_kv("inject_fully_pseudoimplicit_power_substep_projection", "1.0", extra_options))
            inj_weight = inject_fully_pseudoimplicit_power_substep = float(get_extra_options_kv("inject_fully_pseudoimplicit_power_substep", "1.0", extra_options))
            inj_w = max(inj_weight, inj_proj_weight)
            
            x_lying_ = x_.clone()
            s_lying_ = []
            eps_lying_ = eps_.clone()
            for r in range(rk.rows):
                eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, r, rk_type)
                eps_substep_guide = LG.mask * eps_substep_guide + (1-LG.mask) * eps_substep_guide_inv
                
                maxmin_ratio = (s_[r] - rk.sigma_min) / s_[r]
                fully_sub_sigma_2 = s_[r] - maxmin_ratio * (s_[r] * LG.lgw[step])
                s_lying_.append(fully_sub_sigma_2)
                
                if extra_options_flag("guide_fully_pseudoimplicit_power_substep_projection", extra_options) or extra_options_flag("inject_fully_pseudoimplicit_power_substep_projection", extra_options):
                    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_lying_, r, rk_type)
                    eps_row_lerp = eps_[r]   +   LG.mask * (eps_row-eps_[r])   +   (1-LG.mask) * (eps_row_inv-eps_[r])
                    eps_collinear_eps_lerp = get_collinear(eps_[r], eps_row_lerp)
                    eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[r])
                    eps_sum = eps_collinear_eps_lerp + eps_lerp_ortho_eps
                    lgw_mask, lgw_mask_inv = LG.get_masks_for_step(step)
                    eps_substep_guide = eps_[r] + inj_w*inj_w*lgw_mask * (eps_sum - eps_[r]) + inj_w*inj_w*lgw_mask_inv * (eps_sum - eps_[r])

                if not extra_options_flag("implicit_disable_preupdate", extra_options) and rk_type in IRK_SAMPLER_NAMES_BETA: 
                    if r < rk.rows - rk.multistep_stages:
                        x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                    else:
                        x_[r] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                x_lying_[r] = x_[r] + rk.h_fn(fully_sub_sigma_2, s_[r]) * eps_substep_guide
                data_tmp = get_data_from_step(x_0, x_lying_[r], sigma, s_lying_[r])
                if rk.h_fn(s_lying_[r], sigma).abs() > h.abs() / 100:           # if this ratio is too great, bad results
                    eps_lying_[r] = get_epsilon(x_0, data_tmp, sigma, rk_type)
                
            eps_ = eps_lying_
            
            if extra_options_flag("lying_scaling_post", extra_options):
                for r in range(rk.rows):
                    eps_ [r] = eps_lying_ [0].clone() * sigma / s_[r]
                    #data_[r] = data_[0].clone()
            
            eps_0_lying = eps_[0].clone()
            newton_iter_lying_init = int(get_extra_options_kv("newton_iter_lying_init", str("0"), extra_options))
            for n_iter_lying_init in range(newton_iter_lying_init):
                for r in range(0, rk.rows+1): #+1):
                    eps_[0] = eps_0_lying
                    x_tmp, eps_tmp = x_[r].clone(), eps_[r].clone()

                    if r < rk.rows:
                        x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                    else:
                        x_[r] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                    if sigma == s_[r]:
                        continue
                    #x_[r] = x_[r] + newton_iter_mixing_rate * (x_tmp - x_[r])
                    data_[r] = get_data_from_step(x_0, x_[r], sigma, s_[r])
                    eps_ [r] = get_epsilon(x_0, data_[r], s_[r], rk_type)
                    
                    for r2 in range(0, rk.rows+1): 
                        if r != r2:
                            #eps_ [r2] = get_orthogonal(eps_[r2], eps_[r])
                            eps_ [r] = get_orthogonal(eps_[r2], eps_[r])
                    #eps_ [r] = get_collinear(eps_[r], eps_0_lying)
                    
                    x_[r] = x_tmp + newton_iter_mixing_rate * (x_[r] - x_tmp)
                    eps_[r] = eps_tmp + newton_iter_mixing_rate * (eps_[r] - eps_tmp)



        row_offset = 1 if rk.a[0].sum() == 0 and rk_type not in IRK_SAMPLER_NAMES_BETA else 0          
        for full_iter in range(implicit_steps_full + 1):

            for row in range(rk.rows - rk.multistep_stages - row_offset + 1):
                for diag_iter in range(implicit_steps_diag+1):
                    
                    sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio = 0., s_[row], s_[row+row_offset+rk.multistep_stages], s_[row+row_offset+rk.multistep_stages], 1.
                    h_new = h.clone()

                    if row < rk.rows   and   s_[row+row_offset+rk.multistep_stages] > 0:
                        if   diag_iter > 0 and diag_iter == implicit_steps_diag and extra_options_flag("implicit_substep_skip_final_eta", extra_options):
                            pass
                        elif diag_iter > 0 and                                      extra_options_flag("implicit_substep_only_first_eta", extra_options):
                            pass
                        elif full_iter > 0 and full_iter == implicit_steps_full and extra_options_flag("implicit_step_skip_final_eta", extra_options):
                            pass
                        elif full_iter > 0 and                                      extra_options_flag("implicit_step_only_first_eta", extra_options):
                            pass
                        elif full_iter > 0 and                                      extra_options_flag("implicit_step_only_first_all_eta", extra_options):
                            sigma_down = sigma_next
                            sigma_up *= 0
                            alpha_ratio /= alpha_ratio
                            h_new = h = h_no_eta
                        elif (row < rk.rows-row_offset-rk.multistep_stages   or   diag_iter < implicit_steps_diag):
                            sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row], s_[row+row_offset+rk.multistep_stages], eta_substep, eta_var_substep, noise_mode_sde_substep)
                            
                        elif (row < rk.rows-rk.multistep_stages   or   diag_iter < implicit_steps_diag)   and not   extra_options_flag("substep_eta_skip_final", extra_options):
                            sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row], s_[row+row_offset+rk.multistep_stages], eta_substep, eta_var_substep, noise_mode_sde_substep)

                    h_new = h * rk.h_fn(sub_sigma_down, sigma) / rk.h_fn(sub_sigma_next, sigma) 

                    if extra_options_flag("guide_pseudoimplicit_power_substep", extra_options):
                        eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                        eps_substep_guide = LG.mask * eps_substep_guide + (1-LG.mask) * eps_substep_guide_inv
                        maxmin_ratio = (sub_sigma - rk.sigma_min) / sub_sigma
                        sub_sigma_2 = sub_sigma - maxmin_ratio * (sub_sigma * LG.lgw[step])
                        #s_2_ = s_.clone()
                        s_2_ = copy.deepcopy(s_)
                        s_2_[row] = sub_sigma_2
                        if extra_options_flag("guide_pseudoimplicit_power_substep_projection", extra_options):
                            eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_2_, row, rk_type)
                            eps_row_lerp = eps_[row]   +   LG.mask * (eps_row-eps_[row])   +   (1-LG.mask) * (eps_row_inv-eps_[row])

                            eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                            eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                            eps_sum = eps_collinear_eps_lerp + eps_lerp_ortho_eps
                            
                            lgw_mask, lgw_mask_inv = LG.get_masks_for_step(step)
                            eps_substep_guide = eps_[row] + lgw_mask * (eps_sum - eps_[row]) + lgw_mask_inv * (eps_sum - eps_[row])
                                
                        if not extra_options_flag("implicit_disable_preupdate", extra_options) and rk_type in IRK_SAMPLER_NAMES_BETA: 
                            if row < rk.rows - rk.multistep_stages:
                                x_[row] = x_0 + h * (rk.a_k_sum(eps_, row) + rk.u_k_sum(eps_prev_, row))
                            else:
                                x_[row] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                        x_row_tmp = x_[row] + rk.h_fn(sub_sigma_2, sub_sigma) * eps_substep_guide



                    # MODEL CALL
                    if row < rk.rows: # A-tableau still

                        if extra_options_flag("guide_pseudoimplicit_power_substep", extra_options): 
                            x_tmp = x_row_tmp
                            s_tmp = sub_sigma_2

                        elif extra_options_flag("guide_fully_pseudoimplicit_power_substep", extra_options): 
                            x_tmp = x_lying_[row]
                            s_tmp = s_lying_[row]

                        elif extra_options_flag("guide_relaxing_fully_pseudoimplicit_power_substep", extra_options) and full_iter == 0: 
                            x_tmp = x_lying_[row]
                            s_tmp = s_lying_[row]

                        elif full_iter > 0 and row_offset == 1 and row == 0: # explicit full implicit
                            x_tmp = x
                            s_tmp = sigma_next

                        elif diag_iter > 0:
                            x_tmp = x_[row+row_offset]
                            s_tmp = s_[row+row_offset+rk.multistep_stages]
                            if rk_type in IRK_SAMPLER_NAMES_BETA   and   not extra_options_flag("implicit_disable_diagonal_preupdate", extra_options): 
                                 x_tmp = x_[row+row_offset] = x_0 + h * (rk.a_k_sum(eps_, row+row_offset) + rk.u_k_sum(eps_prev_, row+row_offset))

                        else:
                            x_tmp = x_[row]
                            s_tmp = s_[row]
                            if rk_type in IRK_SAMPLER_NAMES_BETA   and   not extra_options_flag("implicit_disable_full_preupdate", extra_options): 
                                x_tmp = x_[row] = x_0 + h * (rk.a_k_sum(eps_, row) + rk.u_k_sum(eps_prev_, row))

                        if rk_type in IRK_SAMPLER_NAMES_BETA   and   extra_options_flag("implicit_recycle_first_model_call_at_start", extra_options)   and   row == 0:   #and   s_[0] == sigma:
                            eps_ [0] = eps_0
                            data_[0] = data_0

                        elif rk_type in IRK_SAMPLER_NAMES_BETA   and   extra_options_flag("implicit_lazy_recycle_first_model_call_at_start", extra_options)   and   row == 0:   #and   s_[0] == sigma:
                            pass
                        else:
                            if s_tmp == 0:
                                break
                            if sigma_next > 0:
                                for n_iter_pre in range(newton_iter_pre):
                                    for r in range(row, rk.rows+1):
                                        if r < rk.rows:
                                            x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                                        else:
                                            x_[r] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                                        data_[r] = get_data_from_step(x_0, x_[r], sigma, s_[r])
                                        eps_[r] = get_epsilon(x_0, data_[r], s_[r], rk_type)
                            eps_[row], data_[row] = rk(x_tmp, s_tmp, x_0, sigma, **extra_args) 



                        # GUIDE 
                        if not extra_options_flag("guide_disable_regular_substep", extra_options):
                            if not extra_options_flag("disable_guides_eps_substep", extra_options):
                                eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options)
                            if not extra_options_flag("disable_guides_eps_prev_substep", extra_options):
                                eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options)

                        if sigma_next > 0:
                            if step == len(sigmas)-4:
                                newton_iter_post = 0
                            eps_orig = eps_.clone()
                            for n_iter_post in range(newton_iter_post):
                                #for r in range(row+1):
                                #    eps_[r] = eps_orig[r]
                                for r in range(row+1, rk.rows+1):
                                    #if r < row+1:
                                    #    eps_[r] = eps_orig[r]
                                    if r < rk.rows:
                                        x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                                    else:
                                        x_[r] = x_0 + h * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                                    data_[r] = get_data_from_step(x_0, x_[r], sigma, s_[r])
                                    eps_[r] = get_epsilon(x_0, data_[r], s_[r], rk_type)
                                    #eps_[r] = get_epsilon_from_step(x_0, x_[r], sigma, s_[r])
                                    
                            newton_yter_post = int(get_extra_options_kv("newton_yter_post", str("0"), extra_options))
                            for n_yter_post in range(newton_yter_post):
                                for r in range(row+1, rk.rows):
                                    x_[r] = x_0 + h * (rk.a_k_sum(eps_, r) + rk.u_k_sum(eps_prev_, r))
                                for r in range(row+1, rk.rows):
                                    eps_[r] = get_epsilon_from_step(x_0, x_[r], sigma, s_[r])


                    # UPDATE
                    if row < rk.rows - row_offset   and   rk.multistep_stages == 0:
                        x_[row+row_offset] = x_0 + h_new * (rk.a_k_sum(eps_, row + row_offset) + rk.u_k_sum(eps_prev_, row + row_offset))
                        x_[row+row_offset] = rk.add_noise_post(x_[row+row_offset], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t)
                        if PRINT_DEBUG:
                            print("A: step,row,h,h_new: \n", step, row+row_offset, round(float(h.item()),3), round(float(h_new.item()),3))
                            print("A: sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio: \n", round(float(sub_sigma_up),3), round(float(sub_sigma),3), round(float(sub_sigma_next),3), round(float(sub_sigma_down),3), round(float(sub_alpha_ratio),3))
                    else: 
                        if PRINT_DEBUG:
                            print("B: step,h,h_new: \n", step, round(float(h.item()),3), round(float(h_new.item()),3))
                            print("B: sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio: \n", round(float(sub_sigma_up),3), round(float(sub_sigma),3), round(float(sub_sigma_next),3), round(float(sub_sigma_down),3), round(float(sub_alpha_ratio),3))
                        x_[row+1] = x_0 + h_new * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                        #x_down = x_[row+1]
                        x_[row+1] = rk.add_noise_post(x_[row+1], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t)
            
            denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h_new) * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
            eps = get_epsilon(x_0, denoised, sigma_next, rk_type)
            #eps = x_0 - denoised
            
            x = x_[rk.rows - rk.multistep_stages - row_offset + 1]
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)
            
            preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options)

            if isinstance(MODEL_SAMPLING, comfy.model_sampling.CONST) == True   or   (isinstance(MODEL_SAMPLING, comfy.model_sampling.CONST) == False and noise_mode != "hard"):
                if sigma_up > 0:
                    x = rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)

        denoised_[0] = data_[0]
        for ms in range(recycled_stages):
            denoised_[recycled_stages - ms] = denoised_[recycled_stages - ms - 1]
            
        denoised_prev2 = denoised_prev.to('cpu')
        denoised_prev = denoised.to('cpu')
        
        data_prev2 = data_prev
        s_prev2 = s_prev
        #s_down_prev = sigma_down
        
        
        data_prev = data_[rk.rows-1]
        s_prev = s_[rk.rows-1]
        x_prev = x_[rk.rows-1]
        
        #eps_prev_lost = get_epsilon_from_step(x_0, x, sigma, sigma_next)
        
        #h_prev_last_data = rk.h_fn(s_[rk.rows-1], sigma_next)

        debug_cuda_cleanup(cuda_sync_b_flag, cuda_empty_b_flag, cuda_gc_b_flag)
        
    preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options, FINAL_STEP=True)
    return x









def preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options, FINAL_STEP=False):
    
    if FINAL_STEP:
        denoised_callback = denoised
        
    elif extra_options_flag("eps_substep_preview", extra_options):
        row_callback = int(get_extra_options_kv("eps_substep_preview", "0", extra_options))
        denoised_callback = eps_[row_callback]
        
    elif extra_options_flag("denoised_substep_preview", extra_options):
        row_callback = int(get_extra_options_kv("denoised_substep_preview", "0", extra_options))
        denoised_callback = data_[row_callback]
        
    elif extra_options_flag("x_substep_preview", extra_options):
        row_callback = int(get_extra_options_kv("x_substep_preview", "0", extra_options))
        denoised_callback = x_[row_callback]
        
    elif extra_options_flag("eps_preview", extra_options):
        denoised_callback = eps
        
    elif extra_options_flag("denoised_preview", extra_options):
        denoised_callback = denoised
        
    elif extra_options_flag("x_preview", extra_options):
        denoised_callback = x
        
    else:
        denoised_callback = data_[0]
        
    callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_callback.to(torch.float32)}) if callback is not None else None
    
    return




