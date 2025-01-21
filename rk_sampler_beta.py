import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re
import copy

from tqdm.auto import trange
import gc

import comfy.model_patcher

from .noise_classes import *
from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3

from .rk_method_beta import RK_Method_Beta
from .rk_guide_func_beta import *

from .latents import normalize_latent, initialize_or_scale, latent_normalize_channels
from .helper import get_extra_options_kv, extra_options_flag, get_cosine_similarity
from .sigmas import get_sigmas

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
    rk_type_final_step = f"euler" if rk_type in {"ddim"} else rk_type_final_step
    rk_type_final_step = get_extra_options_kv("rk_type_final_step", rk_type_final_step, extra_options)
    rk = RK_Method_Beta.create(model, rk_type_final_step, x.device)
    rk.init_noise_sampler(x, torch.initial_seed() + 1, noise_sampler_type, alpha=alpha, k=k)
    extra_args =  rk.init_cfg_channelwise(x, cfg_cw, **extra_args)

    eta, eta_var = 0, 0
    return rk, rk_type_final_step, eta, eta_var, extra_args



@torch.no_grad()
def sample_rk_beta_orig(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="explicit_full",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, scale=0.1, c1=0.0, c2=0.5, c3=1.0, implicit_steps_diag=0, implicit_steps_full=0, reverse_weight=0.0,
                  latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, latent_guide_weight_inv=0.0, latent_guide_weights=None, latent_guide_weights_inv=None, guide_mode="", 
                  GARBAGE_COLLECT=False, mask=None, mask_inv=None, LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, etas_substep=None, s_noises=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights=None, eta_substep=0.0, noise_mode_sde_substep="hard", guide_cossim_cutoff_=1.0, guide_bkg_cossim_cutoff_=1.0,
                  ):
    extra_args = {} if extra_args is None else extra_args


    c1 = c1_ = float(get_extra_options_kv("c1", str(c1), extra_options))
    c2 = c2_ = float(get_extra_options_kv("c2", str(c2), extra_options))
    c3 = c3_ = float(get_extra_options_kv("c3", str(c3), extra_options))
    
    guide_skip_steps = int(get_extra_options_kv("guide_skip_steps", 0, extra_options))        

    cfg_cw = float(get_extra_options_kv("cfg_cw", str(cfg_cw), extra_options))
    
    MODEL_SAMPLING = model.inner_model.inner_model.model_sampling
    
    s_in, s_one = x.new_ones([x.shape[0]]), x.new_ones([1])
    default_dtype = torch.float64
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

    if implicit_sampler_name not in ("none", "use_explicit") and implicit_steps_full + implicit_steps_diag > 0:
        rk_type = implicit_sampler_name
    print("rk_type: ", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0

    rk       = RK_Method_Beta.create(model,  rk_type, x.device)
    extra_args =  rk.init_cfg_channelwise(x, cfg_cw, **extra_args)
    rk. init_noise_sampler(x, noise_seed,     noise_sampler_type, alpha=alpha, k=k)

    if frame_weights is not None:
        frame_weights = initialize_or_scale(frame_weights, 1.0, max_steps).to(default_dtype)
        frame_weights = F.pad(frame_weights, (0, max_steps), value=0.0)

    LG = LatentGuide(guides, x, model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options)
    x = LG.init_guides(x, rk.noise_sampler)
    
    y0, y0_inv = LG.y0, LG.y0_inv

    denoised, denoised_prev, eps, eps_prev = [torch.zeros_like(x) for _ in range(4)]
    
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
        s_noise = s_noises[step] if s_noises is not None else s_noise
        
        if sigma_next == 0:
            rk, rk_type, eta, eta_var, extra_args = prepare_step_to_sigma_zero(rk, rk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=cfg_cw, **extra_args)
        if step == len(sigmas)-2:
            print("cut noise at step: ", step)
            eta = 0.0

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, extra_options)
        h = h_new = rk.h_fn(sigma_down, sigma)

        rk. set_coeff(rk_type, h, c1, c2, c3, step, sigmas, sigma_down, extra_options)

        s_ = [(rk.sigma_fn(rk.t_fn(sigma) + h*c_)) * s_one for c_ in rk.c]


        if step == 0 or step == guide_skip_steps:
            x_, data_, denoised_, eps_ = (torch.zeros(rk.rows+2, *x.shape, dtype=x.dtype, device=x.device) for step in range(4))

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
        
        recycled_stages = max(rk.multistep_stages, rk.hybrid_stages)
        for ms in range(recycled_stages):
            if RK_Method_Beta.is_exponential(rk_type):
                eps_ [recycled_stages - ms] = -(x_0 - denoised_ [recycled_stages - ms])
            else:
                eps_ [recycled_stages - ms] =  (x_0 - denoised_ [recycled_stages - ms]) / sigma
                
        eps_prev_ = eps_.clone()

        row_offset = 1 if rk.a[0].sum() == 0 else 0          
        for full_iter in range(implicit_steps_full + 1):
            for row in range(rk.rows - rk.multistep_stages - row_offset + 1):
                for diag_iter in range(implicit_steps_diag+1):
                    
                    sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio = 0., s_[row], s_[row+row_offset+rk.multistep_stages], s_[row+row_offset+rk.multistep_stages], 1.
                    h_new = h.clone()

                    if row < rk.rows   and   s_[row+row_offset+rk.multistep_stages] > 0:
                        if   diag_iter > 0 and diag_iter == implicit_steps_diag and extra_options_flag("implicit_substep_skip_final_eta", extra_options):
                            pass
                        elif diag_iter > 0 and extra_options_flag("implicit_substep_only_first_eta", extra_options):
                            pass
                        elif full_iter > 0 and full_iter == implicit_steps_full and extra_options_flag("implicit_step_skip_final_eta", extra_options):
                            pass
                        elif full_iter > 0 and extra_options_flag("implicit_step_only_first_eta", extra_options):
                            pass
                        elif row < rk.rows-row_offset-rk.multistep_stages   or   diag_iter < implicit_steps_diag:
                            sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row], s_[row+row_offset+rk.multistep_stages], eta_substep, eta_var_substep, noise_mode_sde_substep)
                            h_new = h * rk.h_fn(sub_sigma_down, sigma) / rk.h_fn(sub_sigma_next, sigma) 
                        
                    h_new = (rk.h_fn(sub_sigma_down, sigma) / rk.c[row+row_offset+rk.multistep_stages])[0]

                    # MODEL CALL
                    if row < rk.rows: # A-tableau still
                        if full_iter > 0 and row_offset == 1 and row == 0 and sigma_next > 0: # explicit full implicit
                            if sigma_next == 0:
                                break
                            eps_[row], data_[row] = rk(x, sigma_next, x_0, sigma, **extra_args)   
                        elif diag_iter > 0:
                            if s_[row+row_offset+rk.multistep_stages] == 0:
                                break
                            eps_[row], data_[row] = rk(x_[row+row_offset], s_[row+row_offset+rk.multistep_stages], x_0, sigma, **extra_args)  
                        else:
                            eps_[row], data_[row] = rk(x_[row], s_[row], x_0, sigma, **extra_args) 

                        # GUIDE 
                        if not extra_options_flag("disable_guides_eps_substep", extra_options):
                            eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)
                        if not extra_options_flag("disable_guides_eps_prev_substep", extra_options):
                            eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)

                    # UPDATE
                    if row < rk.rows - row_offset   and   rk.multistep_stages == 0:
                        x_[row+row_offset] = x_0 + h_new * (rk.a_k_sum(eps_, row + row_offset) + rk.u_k_sum(eps_prev_, row + row_offset))
                        x_[row+row_offset] = rk.add_noise_post(x_[row+row_offset], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t)
                    else: 
                        x_[row+1] = x_0 + h_new * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
                        x_[row+1] = rk.add_noise_post(x_[row+1], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t)
            
            denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h_new) * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
            eps = x_0 - denoised
            
            x = x_[rk.rows - rk.multistep_stages - row_offset + 1]
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)
            
            preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options)

            sde_noise_t = None
            if SDE_NOISE_EXTERNAL:
                if step >= len(sde_noise):
                    SDE_NOISE_EXTERNAL=False
                else:
                    sde_noise_t = sde_noise[step]
                    
            if isinstance(MODEL_SAMPLING, comfy.model_sampling.CONST) == True   or   (isinstance(MODEL_SAMPLING, comfy.model_sampling.CONST) == False and noise_mode != "hard"):
                if sigma_up > 0:
                    x = rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)

        denoised_[0] = denoised
        for ms in range(recycled_stages):
            denoised_[recycled_stages - ms] = denoised_[recycled_stages - ms - 1]
        
    preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options, FINAL_STEP=True)
    return x







@torch.no_grad()
def sample_rk_beta(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="explicit_full",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, scale=0.1, c1=0.0, c2=0.5, c3=1.0, implicit_steps_diag=0, implicit_steps_full=0, reverse_weight=0.0,
                  latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, latent_guide_weight_inv=0.0, latent_guide_weights=None, latent_guide_weights_inv=None, guide_mode="", 
                  GARBAGE_COLLECT=False, mask=None, mask_inv=None, LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, etas_substep=None, s_noises=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights=None, eta_substep=0.0, noise_mode_sde_substep="hard", guide_cossim_cutoff_=1.0, guide_bkg_cossim_cutoff_=1.0,
                  ):
    extra_args = {} if extra_args is None else extra_args

    c1 = c1_ = float(get_extra_options_kv("c1", str(c1), extra_options))
    c2 = c2_ = float(get_extra_options_kv("c2", str(c2), extra_options))
    c3 = c3_ = float(get_extra_options_kv("c3", str(c3), extra_options))
    
    guide_skip_steps = int(get_extra_options_kv("guide_skip_steps", 0, extra_options))        

    cfg_cw = float(get_extra_options_kv("cfg_cw", str(cfg_cw), extra_options))
    
    MODEL_SAMPLING = model.inner_model.inner_model.model_sampling
    
    s_in, s_one = x.new_ones([x.shape[0]]), x.new_ones([1])
    default_dtype = torch.float64
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

    if implicit_sampler_name not in ("none", "use_explicit") and implicit_steps_full + implicit_steps_diag > 0:
        rk_type = implicit_sampler_name
    print("rk_type: ", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0

    rk       = RK_Method_Beta.create(model,  rk_type, x.device)
    extra_args =  rk.init_cfg_channelwise(x, cfg_cw, **extra_args)
    rk. init_noise_sampler(x, noise_seed,     noise_sampler_type, alpha=alpha, k=k)

    if frame_weights is not None:
        frame_weights = initialize_or_scale(frame_weights, 1.0, max_steps).to(default_dtype)
        frame_weights = F.pad(frame_weights, (0, max_steps), value=0.0)

    LG = LatentGuide(guides, x, model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options)
    x = LG.init_guides(x, rk.noise_sampler)
    
    y0, y0_inv = LG.y0, LG.y0_inv

    denoised, denoised_prev, eps, eps_prev = [torch.zeros_like(x) for _ in range(4)]
    
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
        s_noise = s_noises[step] if s_noises is not None else s_noise
        
        if sigma_next == 0:
            rk, rk_type, eta, eta_var, extra_args = prepare_step_to_sigma_zero(rk, rk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=cfg_cw, **extra_args)
        if step == len(sigmas)-2:
            print("cut noise at step: ", step)
            eta = 0.0

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, extra_options)
        h = h_new = rk.h_fn(sigma_down, sigma)

        rk. set_coeff(rk_type, h, c1, c2, c3, step, sigmas, sigma_down, extra_options)

        s_ = [(rk.sigma_fn(rk.t_fn(sigma) + h*c_)) * s_one for c_ in rk.c]


        if step == 0 or step == guide_skip_steps:
            x_, data_, denoised_, eps_ = (torch.zeros(rk.rows+2, *x.shape, dtype=x.dtype, device=x.device) for step in range(4))

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
        
        recycled_stages = max(rk.multistep_stages, rk.hybrid_stages)
        for ms in range(recycled_stages):
            if RK_Method_Beta.is_exponential(rk_type):
                eps_ [recycled_stages - ms] = -(x_0 - denoised_ [recycled_stages - ms])
            else:
                eps_ [recycled_stages - ms] =  (x_0 - denoised_ [recycled_stages - ms]) / sigma
                
        eps_prev_ = eps_.clone()

        row_offset = 1 if rk.a[0].sum() == 0 else 0          
        for full_iter in range(implicit_steps_full + 1):

            for row in range(rk.rows - rk.multistep_stages - row_offset + 1):
                for diag_iter in range(implicit_steps_diag+1):
                    
                    sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio = 0., s_[row], s_[row+row_offset+rk.multistep_stages], s_[row+row_offset+rk.multistep_stages], 1.
                    h_new = h.clone()

                    if row < rk.rows   and   s_[row+row_offset+rk.multistep_stages] > 0:
                        if   diag_iter > 0 and diag_iter == implicit_steps_diag and extra_options_flag("implicit_substep_skip_final_eta", extra_options):
                            pass
                        elif diag_iter > 0 and extra_options_flag("implicit_substep_only_first_eta", extra_options):
                            pass
                        elif full_iter > 0 and full_iter == implicit_steps_full and extra_options_flag("implicit_step_skip_final_eta", extra_options):
                            pass
                        elif full_iter > 0 and extra_options_flag("implicit_step_only_first_eta", extra_options):
                            pass
                        elif row < rk.rows-rk.multistep_stages   or   diag_iter < implicit_steps_diag:
                            sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row], s_[row+row_offset+rk.multistep_stages], eta_substep, eta_var_substep, noise_mode_sde_substep)
                    
                    h_new = h * rk.h_fn(sub_sigma_down, sigma) / rk.h_fn(sub_sigma_next, sigma) 

                    if extra_options_flag("guide_pseudoimplicit_power_substep", extra_options):
                        eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                        
                        maxmin_ratio = (sub_sigma - rk.sigma_min) / sub_sigma
                        sub_sigma_2 = sub_sigma - maxmin_ratio * (sub_sigma * LG.lgw[step])
                        x_row_tmp = x_[row] + rk.h_fn(sub_sigma_2, sub_sigma) * eps_substep_guide
                        
                    if extra_options_flag("guide_pseudoimplicit_eps_proj_substep", extra_options):
                        
                        if step == 0:
                            eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                        else:
                            eps_tmp_, x_tmp_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)
                            eps_substep_guide = eps_tmp_[row]

                        maxmin_ratio = (sub_sigma - rk.sigma_min) / sub_sigma
                        sub_sigma_2 = sub_sigma - maxmin_ratio * (sub_sigma * LG.lgw[step])
                        x_row_tmp = x_[row] + rk.h_fn(sub_sigma_2, sub_sigma) * eps_substep_guide
                        
                    if extra_options_flag("guide_pseudoimplicit_substep", extra_options):
                        eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                        sub_s2 = -torch.log(sub_sigma) + h_new * LG.lgw[step]
                        sub_sigma_2 = torch.exp(-sub_s2)
                        x_row_tmp = x_[row] + rk.h_fn(sub_sigma_2, sub_sigma) * eps_substep_guide
                        
                    # MODEL CALL
                    if row < rk.rows: # A-tableau still
                        if full_iter > 0 and row_offset == 1 and row == 0: # explicit full implicit
                            if sigma_next == 0:
                                break
                            eps_[row], data_[row] = rk(x, sigma_next, x_0, sigma, **extra_args)   
                        elif full_iter == 0 and row_offset == 1 and (extra_options_flag("guide_pseudoimplicit_substep", extra_options) or extra_options_flag("guide_pseudoimplicit_power_substep", extra_options) or extra_options_flag("guide_pseudoimplicit_eps_proj_substep", extra_options)): 
                            if sub_sigma_2 == 0:
                                break
                            eps_[row], data_[row] = rk(x_row_tmp, sub_sigma_2, x_0, sigma, **extra_args)
                        elif diag_iter > 0:
                            if s_[row+row_offset+rk.multistep_stages] == 0:
                                break
                            eps_[row], data_[row] = rk(x_[row+row_offset], s_[row+row_offset+rk.multistep_stages], x_0, sigma, **extra_args)  
                        else:
                            eps_[row], data_[row] = rk(x_[row], s_[row], x_0, sigma, **extra_args) 

                        # GUIDE 
                        if not extra_options_flag("guide_disable_regular_substep", extra_options):
                            if not extra_options_flag("disable_guides_eps_substep", extra_options):
                                eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)
                            if not extra_options_flag("disable_guides_eps_prev_substep", extra_options):
                                eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)

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
                        x_[row+1] = rk.add_noise_post(x_[row+1], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t)
            
            denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h_new) * (rk.b_k_sum(eps_, 0) + rk.v_k_sum(eps_prev_, 0))
            eps = x_0 - denoised
            
            x = x_[rk.rows - rk.multistep_stages - row_offset + 1]
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)
            
            preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options)

            sde_noise_t = None
            if SDE_NOISE_EXTERNAL:
                if step >= len(sde_noise):
                    SDE_NOISE_EXTERNAL=False
                else:
                    sde_noise_t = sde_noise[step]
                    
            if isinstance(MODEL_SAMPLING, comfy.model_sampling.CONST) == True   or   (isinstance(MODEL_SAMPLING, comfy.model_sampling.CONST) == False and noise_mode != "hard"):
                if sigma_up > 0:
                    x = rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)

        denoised_[0] = denoised
        for ms in range(recycled_stages):
            denoised_[recycled_stages - ms] = denoised_[recycled_stages - ms - 1]
        
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




