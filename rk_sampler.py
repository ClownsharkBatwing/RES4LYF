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

from .rk_method import RK_Method
from .rk_guide_func import *

from .latents import normalize_latent, initialize_or_scale, latent_normalize_channels
from .helper import get_extra_options_kv, extra_options_flag
from .sigmas import get_sigmas

PRINT_DEBUG=False

def get_cosine_similarity(a, b):
    if a.dim() == 5 and b.dim() == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)

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

def prepare_sigmas(model, sigmas):
    if sigmas[0] == 0.0:      #remove padding used to prevent comfy from adding noise to the latent (for unsampling, etc.)
        UNSAMPLE = True
        sigmas = sigmas[1:-1]
    else: 
        UNSAMPLE = False
        
    if hasattr(model, "sigmas"):
        model.sigmas = sigmas
        
    return sigmas, UNSAMPLE


def prepare_step_to_sigma_zero(rk, irk, rk_type, irk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=1.0, **extra_args):
    rk_type_final_step = f"ralston_{rk_type[-2:]}" if rk_type[-2:] in {"2s", "3s"} else "ralston_3s"
    rk_type_final_step = f"deis_2m" if rk_type[-2:] in {"2m", "3m", "4m"} else rk_type_final_step
    rk_type_final_step = f"euler" if rk_type in {"ddim"} else rk_type_final_step
    rk_type_final_step = get_extra_options_kv("rk_type_final_step", rk_type_final_step, extra_options)
    rk = RK_Method.create(model, rk_type_final_step, x.device)
    rk.init_noise_sampler(x, torch.initial_seed() + 1, noise_sampler_type, alpha=alpha, k=k)
    extra_args =  rk.init_cfg_channelwise(x, cfg_cw, **extra_args)

    if any(element >= 1 for element in irk.c):
        irk_type_final_step = f"gauss-legendre_{rk_type[-2:]}" if rk_type[-2:] in {"2s", "3s", "4s", "5s"} else "gauss-legendre_2s"
        irk_type_final_step = f"deis_2m" if rk_type[-2:] in {"2m", "3m", "4m"} else irk_type_final_step
        irk_type_final_step = get_extra_options_kv("irk_type_final_step", irk_type_final_step, extra_options)
        irk = RK_Method.create(model, irk_type_final_step, x.device)
        irk.init_noise_sampler(x, torch.initial_seed() + 100, noise_sampler_type, alpha=alpha, k=k)
        extra_args =  irk.init_cfg_channelwise(x, cfg_cw, **extra_args)
    else:
        irk_type_final_step = irk_type

    eta, eta_var = 0, 0
    return rk, irk, rk_type_final_step, irk_type_final_step, eta, eta_var, extra_args



@torch.no_grad()
def sample_rk(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="use_explicit",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, scale=0.1, c1=0.0, c2=0.5, c3=1.0, implicit_steps=0, reverse_weight=0.0,
                  latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, latent_guide_weight_inv=0.0, latent_guide_weights=None, latent_guide_weights_inv=None, guide_mode="", 
                  GARBAGE_COLLECT=False, mask=None, mask_inv=None, LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, s_noises=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights=None, eta_substep=0.0, noise_mode_sde_substep="hard", guide_cossim_cutoff_=1.0, guide_bkg_cossim_cutoff_=1.0,
                  ):
    extra_args = {} if extra_args is None else extra_args

    noise_cossim_iterations         = int(get_extra_options_kv("noise_cossim_iterations",         "1",          extra_options))
    noise_substep_cossim_iterations = int(get_extra_options_kv("noise_substep_cossim_iterations", "1",          extra_options))
    NOISE_COSSIM_MODE               =     get_extra_options_kv("noise_cossim_mode",               "orthogonal", extra_options)
    NOISE_COSSIM_SOURCE             =     get_extra_options_kv("noise_cossim_source",             "x_eps_data_xinit_orthogonal",       extra_options)
    NOISE_SUBSTEP_COSSIM_MODE       =     get_extra_options_kv("noise_substep_cossim_mode",       "orthogonal", extra_options)
    NOISE_SUBSTEP_COSSIM_SOURCE     =     get_extra_options_kv("noise_substep_cossim_source",     "x_eps_data_xinit_orthogonal",       extra_options)
    SUBSTEP_SKIP_LAST               =     get_extra_options_kv("substep_skip_last",               "false",      extra_options) == "true" 
    noise_cossim_tile_size          = int(get_extra_options_kv("noise_cossim_tile",               "2",          extra_options))
    noise_substep_cossim_tile_size  = int(get_extra_options_kv("noise_substep_cossim_tile",       "2",          extra_options))
    
    substep_eta           = float(get_extra_options_kv("substep_eta",           str(eta_substep),  extra_options))
    substep_noise_scaling = float(get_extra_options_kv("substep_noise_scaling", "0.0",  extra_options))
    substep_noise_mode    =       get_extra_options_kv("substep_noise_mode",    noise_mode_sde_substep, extra_options)
    
    substep_eta_start_step = int(get_extra_options_kv("substep_noise_start_step",  "-1", extra_options))
    substep_eta_final_step = int(get_extra_options_kv("substep_noise_final_step", "-1", extra_options))
    
    noise_substep_cossim_max_iter  =   int(get_extra_options_kv("noise_substep_cossim_max_iter",  "50",   extra_options))
    noise_cossim_max_iter          =   int(get_extra_options_kv("noise_cossim_max_iter",          "50",   extra_options))
    noise_substep_cossim_max_score = float(get_extra_options_kv("noise_substep_cossim_max_score", "1e-7", extra_options))
    noise_cossim_max_score         = float(get_extra_options_kv("noise_cossim_max_score",         "1e-7", extra_options))
    
    guide_skip_steps = int(get_extra_options_kv("guide_skip_steps", 0, extra_options))        

    cfg_cw = float(get_extra_options_kv("cfg_cw", str(cfg_cw), extra_options))
    
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



    irk_type = implicit_sampler_name if implicit_sampler_name != "use_explicit" else rk_type
    
    rk_type = "euler" if implicit_sampler_name == "use_explicit" else rk_type
    rk_type = get_extra_options_kv("rk_type", rk_type, extra_options)

    rk       = RK_Method.create(model,  rk_type, x.device)
    irk      = RK_Method.create(model, irk_type, x.device)

    extra_args = irk.init_cfg_channelwise(x, cfg_cw, **extra_args)
    extra_args =  rk.init_cfg_channelwise(x, cfg_cw, **extra_args)

    rk. init_noise_sampler(x, noise_seed,     noise_sampler_type, alpha=alpha, k=k)
    irk.init_noise_sampler(x, noise_seed+100, noise_sampler_type, alpha=alpha, k=k)



    if frame_weights is not None:
        frame_weights = initialize_or_scale(frame_weights, 1.0, max_steps).to(default_dtype)
        frame_weights = F.pad(frame_weights, (0, max_steps), value=0.0)

    LG = LatentGuide(guides, x, model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options)
    x = LG.init_guides(x, rk.noise_sampler)
    
    y0, y0_inv = LG.y0, LG.y0_inv
    lgw, lgw_inv = LG.lgw, LG.lgw_inv
    guide_mode = LG.guide_mode



    denoised, denoised_prev, eps, eps_prev = [torch.zeros_like(x) for _ in range(4)]
    prev_noises = []
    x_init = x.clone()
    
    
    
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
        s_noise = s_noises[step] if s_noises is not None else s_noise
        
  
        if sigma_next == 0:
            rk, irk, rk_type, irk_type, eta, eta_var, extra_args = prepare_step_to_sigma_zero(rk, irk, rk_type, irk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=cfg_cw, **extra_args)

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        h     =  rk.h_fn(sigma_down, sigma)
        h_irk = irk.h_fn(sigma_down, sigma)
        
        c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn, t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula)
        
        rk. set_coeff(rk_type,  h,     c1, c2, c3, step, sigmas, sigma, sigma_down)
        irk.set_coeff(irk_type, h_irk, c1, c2, c3, step, sigmas, sigma, sigma_down)
        
        if step == 0 or step == guide_skip_steps:
            x_, data_, data_u, eps_ = (torch.zeros(max(rk.rows, irk.rows) + 2, *x.shape, dtype=x.dtype, device=x.device) for step in range(4))

        s_       = [(  rk.sigma_fn( rk.t_fn(sigma) +     h*c_)) * s_one for c_ in   rk.c]
        s_irk_rk = [(  rk.sigma_fn( rk.t_fn(sigma) +     h*c_)) * s_one for c_ in  irk.c]
        s_irk    = [( irk.sigma_fn(irk.t_fn(sigma) + h_irk*c_)) * s_one for c_ in  irk.c]
        
        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
                

        x_prenoise = x.clone()
        x_[0] = rk.add_noise_pre(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t) #y0, lgw, sigma_down are currently unused
        
        x_0 = x_[0].clone()
        
        for ms in range(rk.multistep_stages):
            if RK_Method.is_exponential(rk_type):
                eps_ [rk.multistep_stages - ms] = -(x_0 - data_ [rk.multistep_stages - ms])
            else:
                eps_ [rk.multistep_stages - ms] =  (x_0 - data_ [rk.multistep_stages - ms]) / sigma
                
                
                
        lgw_mask, lgw_mask_inv = LG.lgw_masks[step], LG.lgw_masks_inv[step]        



        rk_new       = RK_Method.create(model,  rk_type, x.device)

        if implicit_steps == 0: 
            for row in range(rk.rows - rk.multistep_stages):
                
                sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio = 0, s_[row], s_[row+1], s_[row+1], 1
                if step > substep_eta_start_step and s_[row+1] <= s_[row]:
                    sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row], s_[row+1], substep_eta, eta_var, substep_noise_mode)
                    
                if (substep_eta_final_step < 0 and step == len(sigmas)-1+substep_eta_final_step)   or   (substep_eta_final_step > 0 and step > substep_eta_final_step):
                    sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = 0, s_[row], s_[row+1], 1
                    
                if row > 0 and not extra_options_flag("disable_rough_noise", extra_options): # and s_[row-1] >= s_[row]:
                    sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row-1], s_[row], substep_eta, eta_var, substep_noise_mode)
                    sub_sigma_next = s_[row]
                    
                    
                if row > 0 and substep_eta > 0 and row < rk.rows and ((SUBSTEP_SKIP_LAST == False) or (row < rk.rows - rk.multistep_stages - 1))   and   (sub_sigma_down > 0) and sigma_next > 0:
                    if extra_options_flag("linear_scaling_reg", extra_options):
                        substep_noise_scaling_ratio = (sub_sigma_down - sigma) / (sub_sigma_next - sigma)
                    elif extra_options_flag("linear_scaling_inv", extra_options):
                        substep_noise_scaling_ratio = (sub_sigma_next - sigma) / (sub_sigma_down - sigma)
                    elif extra_options_flag("h_ratio_reg", extra_options):
                        substep_noise_scaling_ratio = rk.h_fn(sub_sigma_down, sigma) / rk.h_fn(sub_sigma_next, sigma)
                    elif extra_options_flag("h_ratio_inv", extra_options):
                        substep_noise_scaling_ratio = rk.h_fn(sub_sigma_next, sigma) / rk.h_fn(sub_sigma_down, sigma)
                    elif extra_options_flag("h_ratio_substep_reg", extra_options):
                        substep_noise_scaling_ratio = rk.h_fn(sub_sigma_down, sub_sigma) / rk.h_fn(sub_sigma_next, sub_sigma)
                    else:
                        substep_noise_scaling_ratio = s_[row+1]/sub_sigma_down
                    eps_[row-1] *= 1 + substep_noise_scaling*(substep_noise_scaling_ratio-1)

                h_new = h.clone()
                if row > 0 and sub_sigma_up > 0:
                    if extra_options_flag("h_new_substep_reg", extra_options):
                        h_new = h_new * rk.h_fn(sub_sigma_down, sub_sigma) / rk.h_fn(sub_sigma_next, sub_sigma)
                    else:
                        if extra_options_flag("substep_eta_c_row_plus_one", extra_options):
                            h_new = (rk.h_fn(sub_sigma_down, sigma) / rk.c[row+1])[0]  
                        else:
                            h_new = (rk.h_fn(sub_sigma_down, sigma) / rk.c[row])[0]   #used to be rk.c[row+1]
                    
                rk_new. set_coeff(rk_type, h_new, c1, c2, c3, step, sigmas, sigma, sigma_down)


                # UPDATE
                if substep_eta > 0 and not extra_options_flag("disable_tableau_scaling_h_only", extra_options):
                    x_[row+1] = x_0 + h_new * rk.a_k_sum(eps_, row)
                    
                elif substep_eta > 0 and extra_options_flag("enable_tableau_scaling_full", extra_options):
                    x_[row+1] = x_0 + h_new * rk_new.a_k_sum(eps_, row)
                elif substep_eta > 0 and extra_options_flag("enable_tableau_extra_linear_denoise", extra_options):
                    x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row) + (sub_sigma_down - sub_sigma_next) * rk.a_k_sum(eps_, row)
                else:
                    x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row)     


                # NOISE ADD
                if (row > 0) and (sub_sigma_up > 0) and ((SUBSTEP_SKIP_LAST == False) or (row < rk.rows - rk.multistep_stages - 1)):
                    data_tmp = denoised_prev if data_[row-1].sum() == 0 else data_[row-1]
                    eps_tmp  = eps_prev      if  eps_[row-1].sum() == 0 else eps_ [row-1]
                    Osde = NoiseStepHandlerOSDE(x_[row+1], eps_tmp, data_tmp, x_init, y0, y0_inv)
                    if Osde.check_cossim_source(NOISE_SUBSTEP_COSSIM_SOURCE):
                        noise = rk.noise_sampler(sigma=sub_sigma, sigma_next=sub_sigma_next) 
                        noise_osde = Osde.get_ortho_noise(noise, prev_noises, max_iter=noise_substep_cossim_max_iter, max_score=noise_substep_cossim_max_score, NOISE_COSSIM_SOURCE=NOISE_SUBSTEP_COSSIM_SOURCE)
                        x_[row+1] = sub_alpha_ratio * x_[row+1] + sub_sigma_up * noise_osde * s_noise
                    elif extra_options_flag("noise_substep_cossim", extra_options):
                        x_[row+1] = handle_tiled_etc_noise_steps(x_0, x_[row+1], x_prenoise, x_init, eps_tmp, data_tmp, y0, y0_inv, row, 
                            rk_type, rk, sub_sigma_up, s_[row-1], s_[row], sub_alpha_ratio, s_noise, substep_noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t,
                            NOISE_SUBSTEP_COSSIM_SOURCE, NOISE_SUBSTEP_COSSIM_MODE, noise_substep_cossim_tile_size, noise_substep_cossim_iterations,
                            extra_options)
                    else:
                        x_[row+1] = rk.add_noise_post(x_[row+1], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise, substep_noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)
                
                #MODEL CALL
                if step < guide_skip_steps:
                    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                    #eps_[row] = lgw_mask * eps_row   +   lgw_mask_inv * eps_row_inv
                    eps_[row] = eps_row
                else:
                    eps_[row], data_[row] = rk(x_0, x_[row+1], s_[row], h, **extra_args)       
                
                eps_, x_ = LG.process_guides_substep(x_0, x_, eps_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)


            x = x_0 + h * rk.b_k_sum(eps_, 0)
                    
            denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h) * rk.b_k_sum(eps_, 0) 
            eps = x - denoised
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)



        elif any(irk_type.startswith(prefix) for prefix in {"crouzeix", "irk_exp_diag", "pareschi_russo", "kraaijevanger_spijker", "qin_zhang"}):
            for row in range(irk.rows - irk.multistep_stages):
                s_tmp = s_irk[row-1] if row >= 1 else sigma
                eps_[row], data_[row] = irk(x_0, x_[row], s_tmp, h_irk, **extra_args) 
                for diag_iter in range(implicit_steps+1):
                    x_[row+1] = x_0 + h_irk * irk.a_k_sum(eps_, row)
                    eps_[row], data_[row] = irk(x_0, x_[row+1], s_irk[row], h_irk, **extra_args)       #MODEL CALL
                    eps_, x_ = LG.process_guides_substep(x_0, x_, eps_, data_, row, step, sigma, sigma_next, sigma_down, s_irk, unsample_resample_scale, irk, irk_type, extra_options, frame_weights)
                    
            x = x_0 + h_irk * irk.b_k_sum(eps_, 0) 
            
            denoised = x_0 + (sigma / (sigma - sigma_down)) *  h_irk * irk.b_k_sum(eps_, 0) 
            eps = x - denoised
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)



        else:
            s2 = s_irk_rk[:]
            s2.append(sigma.unsqueeze(dim=0))
            s_all = torch.sort(torch.stack(s2, dim=0).squeeze(dim=1).unique(), descending=True)[0]
            sigmas_and = torch.cat( (sigmas[0:step], s_all), dim=0)
            
            data_[0].zero_()
            eps_ [0].zero_()
            eps_list = []
            
            if extra_options_flag("fast_implicit_guess",  extra_options):
                if denoised.sum() == 0:
                    if extra_options_flag("fast_implicit_guess_use_guide",  extra_options):
                        data_s = y0
                        eps_s = x_0 - data_s
                    else:
                        eps_s, data_s = rk(x_0, x_0, sigma, h, **extra_args)
                else:
                    eps_s, data_s = eps, denoised
                for i in range(len(s_all)-1):
                    eps_list.append(eps_s * s_all[i]/sigma)
                if torch.allclose(s_all[-1], sigma_down, atol=1e-8):
                    eps_list.append(eps_s * sigma_down/sigma)
            else:
                x_mid = x
                for i in range(len(s_all)-1):
                    x_mid, eps_, data_ = get_explicit_rk_step(rk, rk_type, x_mid, LG, step, s_all[i], s_all[i+1], eta, eta_var, s_noise, noise_mode, c2, c3, step+i, sigmas_and, x_, eps_, data_, unsample_resample_scale, extra_options, frame_weights, 
                                                              x_init, x_prenoise, NOISE_COSSIM_SOURCE, NOISE_COSSIM_MODE, noise_cossim_max_iter, noise_cossim_max_score, noise_cossim_tile_size, noise_cossim_iterations,SDE_NOISE_EXTERNAL,sde_noise_t,
                                                              **extra_args)

                    eps_list.append(eps_[0])
                    data_[0].zero_()
                    eps_ [0].zero_()
                    
                if torch.allclose(s_all[-1], sigma_down, atol=1e-8):
                    eps_down, data_down = rk(x_0, x_mid, sigma_down, h, **extra_args) #should h_irk = h? going to change it for now.
                    eps_list.append(eps_down)

            s_all = [s for s in s_all if s in s_irk_rk]

            eps_list = [eps_list[s_all.index(s)].clone() for s in s_irk_rk]
            eps2_ = torch.stack(eps_list, dim=0)

            for implicit_iter in range(implicit_steps):
                for row in range(irk.rows):
                    x_[row+1] = x_0 + h_irk * irk.a_k_sum(eps2_, row)
                    eps2_[row], data_[row] = irk(x_0, x_[row+1], s_irk[row], h_irk, **extra_args)
                    eps_, x_ = LG.process_guides_substep(x_0, x_, eps2_, data_, row, step, sigma, sigma_next, sigma_down, s_irk, unsample_resample_scale, irk, irk_type, extra_options, frame_weights)
                x = x_0 + h_irk * irk.b_k_sum(eps2_, 0)
                denoised = x_0 + (sigma / (sigma - sigma_down)) *  h_irk * irk.b_k_sum(eps2_, 0) 
                eps = x - denoised
                x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)



        if extra_options_flag("eps_preview", extra_options) == False:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data_[0].to(torch.float32)}) if callback is not None else None
        elif latent_guide is not None:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': eps_[0]}) if callback is not None else None
            #callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': ((x_0 - y0) / sigma).to(torch.float32)}) if callback is not None else None
        else:
            callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': ((x_0 - data_[0]) / sigma).to(torch.float32)}) if callback is not None else None

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
                
        if sigma_up > 0:
            Osde = NoiseStepHandlerOSDE(x, eps, denoised, x_init, y0, y0_inv)
            if Osde.check_cossim_source(NOISE_COSSIM_SOURCE):
                noise = rk.noise_sampler(sigma=sigma, sigma_next=sigma_next)
                noise_osde = Osde.get_ortho_noise(noise, prev_noises, max_iter=noise_cossim_max_iter, max_score=noise_cossim_max_score, NOISE_COSSIM_SOURCE=NOISE_COSSIM_SOURCE)
                x = alpha_ratio * x + sigma_up * noise_osde * s_noise
            elif extra_options_flag("noise_cossim", extra_options):
                x = handle_tiled_etc_noise_steps(x_0, x, x_prenoise, x_init, eps, denoised, y0, y0_inv, step, 
                                 rk_type, rk, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t,
                                 NOISE_COSSIM_SOURCE, NOISE_COSSIM_MODE, noise_cossim_tile_size, noise_cossim_iterations,
                                 extra_options)
            else:
                x = rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)

        if PRINT_DEBUG:
            print("Data vs. y0 cossim score: ", get_cosine_similarity(data_[0], y0).item())

        for ms in range(rk.multistep_stages):
            eps_ [rk.multistep_stages - ms] = eps_ [rk.multistep_stages - ms - 1]
            data_[rk.multistep_stages - ms] = data_[rk.multistep_stages - ms - 1]
        eps_ [0] = torch.zeros_like(eps_ [0])
        data_[0] = torch.zeros_like(data_[0])
        
        if PRINT_DEBUG:
            print("Denoised vs. y0 cossim score: ", get_cosine_similarity(denoised, y0).item())
        denoised_prev = denoised
        eps_prev = eps
        
    callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised.to(torch.float32)}) if callback is not None else None
    return x



def get_explicit_rk_step(rk, rk_type, x, LG, step, sigma, sigma_next, eta, eta_var, s_noise, noise_mode, c2, c3, stepcount, sigmas, x_, eps_, data_, unsample_resample_scale, extra_options, frame_weights, 
                         x_init, x_prenoise, NOISE_COSSIM_SOURCE, NOISE_COSSIM_MODE, noise_cossim_max_iter, noise_cossim_max_score, noise_cossim_tile_size, noise_cossim_iterations,SDE_NOISE_EXTERNAL,sde_noise_t,
                         **extra_args):

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    eta = get_extra_options_kv("implicit_substep_eta", eta, extra_options)

    sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(rk.model, sigma, sigma_next, eta, eta_var, noise_mode)
    h = rk.h_fn(sigma_down, sigma)
    c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=rk.t_fn, sigma_fn=rk.sigma_fn)
    
    rk.set_coeff(rk_type, h, c2=c2, c3=c3, stepcount=stepcount, sigmas=sigmas, sigma_down=sigma_down)

    s_ = [(sigma + h * c_) * s_in for c_ in rk.c]
    x_[0] = rk.add_noise_pre(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode)
    
    x_0 = x_[0].clone()
    
    for ms in range(rk.multistep_stages):
        if RK_Method.is_exponential(rk_type):
            eps_ [rk.multistep_stages - ms] = data_ [rk.multistep_stages - ms] - x_0
        else:
            eps_ [rk.multistep_stages - ms] = (x_0 - data_ [rk.multistep_stages - ms]) / sigma
        
    for row in range(rk.rows - rk.multistep_stages):
        x_[row+1] = x_0 + h * rk.a_k_sum(eps_, row)
        eps_[row], data_[row] = rk(x_0, x_[row+1], s_[row], h, **extra_args)
        
        eps_, x_ = LG.process_guides_substep(x_0, x_, eps_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights)        
        
    x = x_0 + h * rk.b_k_sum(eps_, 0)
    
    denoised = x_0 + (sigma / (sigma - sigma_down)) *  h * rk.b_k_sum(eps_, 0) 
    eps = x - denoised
    
    y0 = LG.y0
    if LG.y0.shape[0] > 1:
        y0 = LG.y0[min(step, LG.y0.shape[0]-1)].unsqueeze(0)  
        
    x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)

    #x = rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode)
    
    if sigma_up > 0:
        Osde = NoiseStepHandlerOSDE(x, eps, denoised, x_init, y0, LG.y0_inv)
        if Osde.check_cossim_source(NOISE_COSSIM_SOURCE):
            noise = rk.noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise_osde = Osde.get_ortho_noise(noise, [], max_iter=noise_cossim_max_iter, max_score=noise_cossim_max_score, NOISE_COSSIM_SOURCE=NOISE_COSSIM_SOURCE)
            x = alpha_ratio * x + sigma_up * noise_osde * s_noise
        elif extra_options_flag("noise_cossim", extra_options):
            x = handle_tiled_etc_noise_steps(x_0, x, x_prenoise, x_init, eps, denoised, y0, LG.y0_inv, step, 
                                rk_type, rk, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t,
                                NOISE_COSSIM_SOURCE, NOISE_COSSIM_MODE, noise_cossim_tile_size, noise_cossim_iterations,
                                extra_options)
        else:
            x = rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)

    for ms in range(rk.multistep_stages):
        eps_ [rk.multistep_stages - ms] = eps_ [rk.multistep_stages - ms - 1]
        data_[rk.multistep_stages - ms] = data_[rk.multistep_stages - ms - 1]

    return x, eps_, data_








def sample_res_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", eta=0.0, )
def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2s", eta=0.0, )
def sample_res_3s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_3s", eta=0.0, )
def sample_res_5s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_5s", eta=0.0, )
def sample_res_6s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_6s", eta=0.0, )

def sample_res_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", eta=0.5, eta_substep=0.5, )
def sample_res_2s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2s", eta=0.5, eta_substep=0.5, )
def sample_res_3s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_3s", eta=0.5, eta_substep=0.5, )
def sample_res_5s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_5s", eta=0.5, eta_substep=0.5, )
def sample_res_6s_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_6s", eta=0.5, eta_substep=0.5, )

def sample_deis_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_2m", eta=0.0, )
def sample_deis_3m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_3m", eta=0.0, )
def sample_deis_4m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_4m", eta=0.0, )

def sample_deis_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_2m", eta=0.5, )
def sample_deis_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_3m", eta=0.5, )
def sample_deis_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_4m", eta=0.5, )

