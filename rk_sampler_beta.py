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

from .rk_method_beta import RK_Method_Beta, RK_NoiseSampler
from .rk_guide_func_beta import *

from .latents import normalize_latent, initialize_or_scale, latent_normalize_channels
from .helper import get_extra_options_kv, extra_options_flag, get_cosine_similarity, get_pearson_similarity
from .sigmas import get_sigmas

from .rk_coefficients_beta import IRK_SAMPLER_NAMES_BETA
from .phi_functions import Phi


PRINT_DEBUG=False

#from .settings import GlobalSettings

#print(GlobalSettings().extra_options["key"]) 


def prepare_sigmas(model, sigmas):
    if sigmas[0] == 0.0:      #remove padding used to prevent comfy from adding noise to the latent (for unsampling, etc.)
        UNSAMPLE = True
        sigmas = sigmas[1:-1]
    else: 
        UNSAMPLE = False
        
    if hasattr(model, "sigmas"):
        model.sigmas = sigmas
        
    return sigmas, UNSAMPLE


def prepare_step_to_sigma_zero(RK, rk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=1.0, **extra_args):
    if rk_type in IRK_SAMPLER_NAMES_BETA:
        if RK.c[-2] == 1.0 and not rk_type.startswith("gauss-legendre"):
            rk_type_final_step = "gauss-legendre_2s"
            #rk_type_final_step = f"gauss-legendre_{rk_type[-2:]}" if rk_type[-2:] in {"2s", "3s", "4s"} else "ralston_3s"
        else:
            rk_type_final_step = rk_type
    elif rk_type in {"euler", "ddim"}:
        rk_type_final_step = "euler"
    elif rk_type[-2:] in {"2m", "3m", "4m"}:
        rk_type_final_step = "deis_" + rk_type[-2:]
    else:
        rk_type_final_step = "ralston_" + rk_type[-2:] if rk_type[-2:] in {"2s", "3s", "4s"} else "ralston_3s"
    
    rk_type_final_step = get_extra_options_kv("rk_type_final_step", rk_type_final_step, extra_options)
    RK = RK_Method_Beta.create(model, rk_type_final_step, x.device)
    extra_args =  RK.init_cfg_channelwise(x, cfg_cw, **extra_args)

    eta = 0
    print("rk_type set to:", rk_type_final_step, " for step to zero.")
    return RK, rk_type_final_step, eta, extra_args



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




@torch.no_grad()
def sample_rk_beta(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian",  noise_sampler_type_substep="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="explicit_full",
                  eta=0.0, eta_var=0.0, s_noise=1., s_noise_substep=1., d_noise=1., alpha=-1.0, alpha_substep=-1.0, k=1.0, k_substep=1.0, c1=0.0, c2=0.5, c3=1.0, implicit_steps_diag=0, implicit_steps_full=0, 
                  LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, etas_substep=None, s_noises=None, s_noises_substep=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights=None, eta_substep=0.0, noise_mode_sde_substep="hard",
                  ):
    extra_args = {} if extra_args is None else extra_args
    default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "float64", extra_options), torch.float64)
    
    MAX_STEPS=10000

    if noise_seed < 0:
        noise_seed = torch.initial_seed()+1 
        print("Set noise_seed to:", noise_seed, " using torch.initial_seed()+1")

    c1 = float(get_extra_options_kv("c1", str(c1), extra_options))
    c2 = float(get_extra_options_kv("c2", str(c2), extra_options))
    c3 = float(get_extra_options_kv("c3", str(c3), extra_options))

    guide_skip_steps  =   int(get_extra_options_kv("guide_skip_steps", 0,          extra_options))   
    rk_swap_step      =   int(get_extra_options_kv("rk_swap_step", str(MAX_STEPS), extra_options))
    rk_swap_print     =         extra_options_flag("rk_swap_print", extra_options)
    rk_swap_threshold = float(get_extra_options_kv("rk_swap_threshold", "0.0",     extra_options))
    rk_swap_type      =       get_extra_options_kv("rk_swap_type",         "",     extra_options)   
    CONSERVE_MEAN_CW  =         extra_options_flag("eta_conserve_mean_cw",         extra_options)  
    SYNC_MEAN_CW  =         not extra_options_flag("eta_sync_mean_cw_disable",     extra_options)  
    
    noise_boost_substep  = float(get_extra_options_kv("noise_boost_substep", "0.0", extra_options))
    noise_boost_step = float(get_extra_options_kv("noise_boost_step",    "0.0", extra_options))
    
    reorder_tableau_indices = get_extra_options_list("reorder_tableau_indices", "", extra_options).split(",")
    if reorder_tableau_indices[0]:
        reorder_tableau_indices = [int(reorder_tableau_indices[_]) for _ in range(len(reorder_tableau_indices))]

    pseudoimplicit_step_weights = get_extra_options_list("pseudoimplicit_step_weights", "", extra_options).split(",")
    if pseudoimplicit_step_weights[0]:
        pseudoimplicit_step_weights = [float(pseudoimplicit_step_weights[_]) for _ in range(len(pseudoimplicit_step_weights))]
    else:
        pseudoimplicit_step_weights = [1. for _ in range(max(implicit_steps_diag, implicit_steps_full)+1)]

    cfg_cw = float(get_extra_options_kv("cfg_cw", str(cfg_cw), extra_options))
    
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
    print("rk_type:", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0

    RK = RK_Method_Beta.create(model,  rk_type, x.device, x.dtype)
    NS = RK_NoiseSampler(model, x.device, x.dtype)
    NS.init_noise_sampler(x, noise_seed, noise_sampler_type, noise_sampler_type_substep, alpha, alpha_substep, k, k_substep)
    
    extra_args = RK.init_cfg_channelwise(x, cfg_cw, **extra_args)

    if frame_weights is not None:
        frame_weights = initialize_or_scale(frame_weights, 1.0, MAX_STEPS).to(default_dtype)
        frame_weights = F.pad(frame_weights, (0, MAX_STEPS), value=0.0)

    LG = LatentGuide(guides, x, model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options)
    x = LG.init_guides(x, RK.noise_sampler)
    
    y0, y0_inv = LG.y0, LG.y0_inv

    denoised, denoised_prev, denoised_prev2, eps = [torch.zeros_like(x) for _ in range(4)]
    
    
    # BEGIN SAMPLING LOOP
    for step in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        unsample_resample_scale = float(unsample_resample_scales[step]) if unsample_resample_scales is not None else None
        
        if regional_conditioning_weights is not None:
            extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = regional_conditioning_weights[step]
            extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = regional_conditioning_floors [step]
        else:
            extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
            extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0
        
        eta             = etas            [step] if etas             is not None else eta
        eta_substep     = etas_substep    [step] if etas_substep     is not None else eta_substep
        s_noise         = s_noises        [step] if s_noises         is not None else s_noise
        s_noise_substep = s_noises_substep[step] if s_noises_substep is not None else s_noise_substep
        
        if sigma_next == 0:
            RK, rk_type, eta, extra_args = prepare_step_to_sigma_zero(RK, rk_type, model, x, extra_options, alpha, k, noise_sampler_type, cfg_cw=cfg_cw, **extra_args)
        if step == len(sigmas)-2:
            print("Cut noise at step:", step)
            eta = 0.0

        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)
        
        h = RK.h_fn(sigma_down, sigma)
        h_no_eta  = RK.h_fn(sigma_next, sigma)
        h = h + noise_boost_step * (h_no_eta - h)

        RK.set_coeff(rk_type, h, c1, c2, c3, step, sigmas, sigma_down, extra_options)
        if sigma_next > 0 and RK.rk_type in IRK_SAMPLER_NAMES_BETA:
            RK.reorder_tableau(reorder_tableau_indices)
        row_offset = 1 if RK.a[0].sum() == 0 and rk_type not in IRK_SAMPLER_NAMES_BETA else 0   

        s_ = [(RK.sigma_fn(RK.t_fn(sigma) + h*c_)) * x.new_ones([1]) for c_ in RK.c]

        if step == 0 or step == guide_skip_steps:
            x_, data_, eps_ = (torch.zeros(RK.rows+2, *x.shape, dtype=x.dtype, device=x.device) for _ in range(3))
            data_prev_ = torch.zeros(max(RK.rows+2, 4), *x.shape, dtype=x.dtype, device=x.device)
            recycled_stages = len(data_prev_)-1

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
        
        x_[0] = NS.add_noise_pre(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t)
        x_0 = x_[0].clone()



        # RECYCLE STAGES FOR MULTISTEP
        for ms in range(len(eps_)):
            eps_[ms] = get_epsilon(x_0, data_prev_[ms], sigma, rk_type)

        eps_prev_ = eps_.clone()



        if rk_type in IRK_SAMPLER_NAMES_BETA:
            if extra_options_flag("implicit_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
                eps_[0], data_[0] = eps.clone(), denoised.clone()
                eps_[0] = get_epsilon(x_0, denoised, sigma, rk_type)
                if denoised_prev2.sum() != 0:
                    sratio = sigma - s_[0]
                    data_[0] = denoised + sratio * (denoised - denoised_prev2)
            else:
                eps_[0], data_[0] = RK(x_[0], sigma, x_0, sigma, **extra_args) 
                
            for r in range(RK.rows):
                eps_ [r] = eps_ [0].clone() * sigma / s_[r]
                data_[r] = data_[0].clone()

            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "init", extra_options)

       
        for full_iter in range(implicit_steps_full + 1):
            
            if rk_type in IRK_SAMPLER_NAMES_BETA and extra_options_flag("fully_implicit_newton_iter", extra_options):
                x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "init", extra_options)
            
            if extra_options_flag("BLARGHWTF", extra_options) and sigma_next > 0 and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0):
                x_lying_ = x_.clone()
                eps_lying_ = eps_.clone()
                s_lying_ = []
                maxmin_ratio = (sigma_down - RK.sigma_min) / sigma_down
                sigma_pseudo_down = sigma_down - maxmin_ratio * (sigma_down * LG.lgw[step])
                if full_iter > 0:
                    fully_sigma = sigma_down - maxmin_ratio * (sigma_down * LG.lgw_inv[step])
                
                fully_h = RK.h_fn(sigma_pseudo_down, sigma)
                s_lying_ = [(RK.sigma_fn(RK.t_fn(sigma) + fully_h*c_)) * x.new_ones([1]) for c_ in RK.c]
                
                for r in range(RK.rows):
                    eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_lying_, r, row_offset, rk_type)
                    eps_substep_guide = LG.mask * eps_substep_guide + (1-LG.mask) * eps_substep_guide_inv
                    eps_lying_[r] = eps_substep_guide
                
                x_lying_, eps_lying_ = RK.newton_iter(x_0, x_lying_, eps_lying_, eps_prev_, data_, s_lying_, 0, fully_h, sigmas, step, "lying", extra_options)
                if not extra_options_flag("pseudoimplicit_keep_eps", extra_options):
                    eps_ = eps_lying_


            elif extra_options_flag("BOOP_BOOP1", extra_options) and sigma_next > 0:
                x_lying_ = x_.clone()
                s_lying_ = []
                eps_lying_ = eps_.clone()
                
                maxmin_ratio_new = (sigma_down - RK.sigma_min) / sigma_down
                sigma_pseudo_down = sigma_down - maxmin_ratio_new * (sigma_down * LG.lgw_inv[step])
                h_2 = RK.h_fn(sigma_pseudo_down, sigma)
                s_2 = [(RK.sigma_fn(RK.t_fn(sigma) + h_2*c_)) * x.new_ones([1]) for c_ in RK.c]
                
                for r in range(RK.rows):
                    s_lying_.append(s_2[r])
                    
                    sub_sigma_next = s_2[r]
                    sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_2[r], s_2[r], eta_substep, noise_mode_sde_substep)
                    h_2_new = h_2 * RK.h_fn(sub_sigma_down, sigma) / RK.h_fn(sub_sigma_next, sigma) 
                    h_2_new_orig = h_2_new.clone()
                    h_2_new = h_2_new + noise_boost_substep * (h_2 - h_2_new)
                    
                    if RK.IMPLICIT:
                        x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, r, row_offset, h_2, h_2_new, h_2_new_orig, sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, NS, \
                            SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, extra_options)

                    if full_iter > 0:
                        #eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_[r], x_, data_[r], data_[r], s_2, r, row_offset, rk_type)
                        eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_[r], x_, denoised, denoised, s_2, r, row_offset, rk_type)
                    else:
                        eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_[r], x_, y0,       y0_inv,   s_2, r, row_offset, rk_type)
                    eps_substep_guide = LG.mask * eps_substep_guide + (1-LG.mask) * eps_substep_guide_inv

                    x_lying_[r] = x_[r] #x_[r] #+ RK.h_fn(sigma_pseudo_down, s_2[r]) * eps_substep_guide
                    
                    data_lying = x_0 + RK.h_fn(0, sigma) * eps_substep_guide
                    eps_lying_[r] = get_epsilon(x_0, data_lying, s_2[r], rk_type)
                    
                eps_ = eps_lying_
                
                x_lying_, eps_ = RK.newton_iter(x_0, x_lying_, eps_, eps_prev_, data_, s_2, 0, h_2, sigmas, step, "lying", extra_options)



            # PREPARE FULLY PSEUDOIMPLICIT GUIDES
            elif LG.guide_mode in {"fully_pseudoimplicit", "fully_pseudoimplicit_projection"} and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0) and sigma_next > 0:
                x_lying_ = x_.clone()
                s_lying_ = []
                eps_lying_ = eps_.clone()
                for r in range(RK.rows):
                    
                    sub_sigma_next = s_[r]
                    sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[r], s_[r], eta_substep, noise_mode_sde_substep)
                    
                    maxmin_ratio = (sub_sigma - RK.sigma_min) / sub_sigma
                    fully_sub_sigma_2 = sub_sigma - maxmin_ratio * (sub_sigma * pseudoimplicit_step_weights[full_iter] * LG.lgw[step])
                    s_lying_.append(fully_sub_sigma_2)

                    h_new = h * RK.h_fn(sub_sigma_down, sigma) / RK.h_fn(sub_sigma_next, sigma) 
                    h_new_orig = h_new.clone()
                    h_new = h_new + noise_boost_substep * (h - h_new)
                    
                    if RK.IMPLICIT:
                        x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, r, row_offset, h, h_new, h_new_orig, sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, NS, \
                            SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, extra_options, IMPLICIT_PREDICTOR=True, )
                    
                    eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, r, row_offset, rk_type)
                    eps_substep_guide = LG.mask * eps_substep_guide + (1-LG.mask) * eps_substep_guide_inv

                    if LG.guide_mode == "fully_pseudoimplicit_projection":
                        eps_substep_guide = get_masked_epsilon_projection(x_0, x_, eps_, y0, y0_inv, s_, r, row_offset, rk_type, LG, step)
                        #eps_substep_guide = get_masked_epsilon_projection(x_0, x_, eps_, y0, y0_inv, s_lying_, r, row_offset, rk_type, LG, step)

                    x_lying_[r] = x_[r] + RK.h_fn(fully_sub_sigma_2, sub_sigma) * eps_substep_guide
                    
                    data_lying = x_[r] + RK.h_fn(0, s_[r]) * eps_substep_guide
                    eps_lying_[r] = get_epsilon(x_0, data_lying, s_[r], rk_type)
                    
                if not extra_options_flag("pseudoimplicit_disable_eps_lying", extra_options):
                    eps_ = eps_lying_
                
                if not extra_options_flag("pseudoimplicit_disable_newton_iter", extra_options):
                    x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "lying", extra_options)



            # TABLEAU LOOP
            for row in range(RK.rows - RK.multistep_stages - row_offset + 1):
                for diag_iter in range(implicit_steps_diag+1):
                    
                    # PREPARE SIGMAS, STEP SIZE
                    sub_sigma_up, sub_sigma, sub_sigma_next, sub_sigma_down, sub_alpha_ratio = 0., s_[row], s_[row+row_offset+RK.multistep_stages], s_[row+row_offset+RK.multistep_stages], 1.


                    if row < RK.rows   and   s_[row+row_offset+RK.multistep_stages] > 0:
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
                        elif (row < RK.rows-row_offset-RK.multistep_stages   or   diag_iter < implicit_steps_diag)   or   extra_options_flag("substep_eta_use_final", extra_options):
                            sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = get_res4lyf_step_with_model(model, s_[row], s_[row+row_offset+RK.multistep_stages], eta_substep, noise_mode_sde_substep)
                            
                    h_new = h * RK.h_fn(sub_sigma_down, sigma) / RK.h_fn(sub_sigma_next, sigma) 
                    h_new_orig = h_new.clone()
                    h_new = h_new + noise_boost_substep * (h - h_new)

                    # PREPARE PSEUDOIMPLICIT GUIDES
                    if LG.guide_mode in {"pseudoimplicit", "pseudoimplicit_projection"}: #  and (full_iter > 0 or diag_iter > 0):
                        maxmin_ratio = (sub_sigma - RK.sigma_min) / sub_sigma
                        
                        if extra_options_flag("guide_pseudoimplicit_power_substep_flip_maxmin_scaling", extra_options):
                            maxmin_ratio *= (RK.rows-row) / RK.rows
                        elif extra_options_flag("guide_pseudoimplicit_power_substep_maxmin_scaling", extra_options):
                            maxmin_ratio *= row / RK.rows
                        
                        sub_sigma_2 = sub_sigma - maxmin_ratio * (sub_sigma * pseudoimplicit_step_weights[full_iter] * LG.lgw[step])
                        s_2_ = copy.deepcopy(s_)
                        s_2_[row] = sub_sigma_2

                        if RK.IMPLICIT:
                            x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, row_offset, h, h_new, h_new_orig, sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, NS, \
                                SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, extra_options, IMPLICIT_PREDICTOR=True, )

                        eps_substep_guide, eps_substep_guide_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, row_offset, rk_type)
                        eps_substep_guide = LG.mask * eps_substep_guide + (1-LG.mask) * eps_substep_guide_inv

                        if LG.guide_mode == "pseudoimplicit_projection":
                            eps_substep_guide = get_masked_epsilon_projection(x_0, x_, eps_, y0, y0_inv, s_, row, row_offset, rk_type, LG, step)
                            #eps_substep_guide = get_masked_epsilon_projection(x_0, x_, eps_, y0, y0_inv, s_2_, row, row_offset, rk_type, LG, step)

                        x_row_tmp = x_[row] + RK.h_fn(sub_sigma_2, sub_sigma) * eps_substep_guide

                    # MODEL CALL
                    if row < RK.rows: # A-tableau still

                        # PREPARE MODEL CALL
                        if (LG.guide_mode in {"fully_pseudoimplicit", "fully_pseudoimplicit_projection"} or extra_options_flag("BLARGHWTF", extra_options) or extra_options_flag("BOOP_BOOP1", extra_options)) and LG.lgw[step] > 0 and LG.lgw_inv[step] > 0: # and full_iter == 0 and diag_iter == 0: 
                            x_tmp = x_lying_[row]
                            s_tmp = s_lying_[row]
                            
                        elif LG.guide_mode in {"pseudoimplicit", "pseudoimplicit_projection"} and LG.lgw[step] > 0 and LG.lgw_inv[step] > 0: # or (extra_options_flag("BLARGHWTF", extra_options) and (full_iter > 0 or diag_iter > 0)):
                            x_tmp = x_row_tmp
                            s_tmp = sub_sigma_2

                        # Fully implicit iteration (explicit only)
                        elif full_iter > 0 and row_offset == 1 and row == 0: # explicit full implicit
                            x_tmp = x
                            s_tmp = sigma_next

                        # All others
                        else:
                            if diag_iter > 0: # Diagonally implicit iteration (explicit or implicit)
                                x_tmp = x_[row+row_offset]
                                s_tmp = s_[row+row_offset+RK.multistep_stages]
                            else:
                                x_tmp = x_[row]
                                s_tmp = s_[row]

                            if rk_type in IRK_SAMPLER_NAMES_BETA: 
                                if not extra_options_flag("implicit_guide_preproc_disable", extra_options):
                                    eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options, frame_weights)
                                    eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options, frame_weights)
                                if row == 0 and not extra_options_flag("substep_eta_use_final", extra_options):
                                    x_tmp = x_[row+row_offset] = x_0 + h     * (RK.a_k_sum(eps_, row+row_offset) + RK.u_k_sum(eps_prev_, row+row_offset))
                                else:
                                    x_tmp = x_[row+row_offset] = x_0 + h_new * (RK.a_k_sum(eps_, row+row_offset) + RK.u_k_sum(eps_prev_, row+row_offset))
                                    x_tmp = x_[row+row_offset] = NS.add_noise_post(x_[row+row_offset], sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP=True)

                        # MAKE MODEL CALL
                        if not RK.IMPLICIT   or not   row == 0   or not   extra_options_flag("implicit_lazy_recycle_first_model_call_at_start", extra_options):
                            if s_tmp == 0:
                                break
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, row, h, sigmas, step, "pre", extra_options)
                            eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma, **extra_args) 

                        # GUIDE 
                        if not extra_options_flag("guide_disable_regular_substep", extra_options):
                            if not extra_options_flag("disable_guides_eps_substep", extra_options):
                                eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options, frame_weights)
                            if not extra_options_flag("disable_guides_eps_prev_substep", extra_options):
                                eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options, frame_weights)

                        if (full_iter == 0 and diag_iter == 0)   or   extra_options_flag("newton_iter_post_use_on_implicit_steps", extra_options):
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, row, h, sigmas, step, "post", extra_options)

                    # UPDATE
                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, row_offset, h, h_new, h_new_orig, sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, NS, \
                                           SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, extra_options)


            denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h_new) * (RK.b_k_sum(eps_, 0) + RK.v_k_sum(eps_prev_, 0))
            eps = get_epsilon(x_0, denoised, sigma_next, rk_type)
            
            x = x_[RK.rows - RK.multistep_stages - row_offset + 1]
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)
            
            preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options)

            x = NS.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)

        data_prev_[0] = data_[0]
        for ms in range(recycled_stages):
            data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
        
        rk_type = RK.swap_rk_type_at_step_or_threshold(x_0, data_prev_, sigma_down, sigmas, step, RK, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print)
        

        denoised_prev2 = denoised_prev
        denoised_prev = denoised
                
    if not (UNSAMPLE and sigmas[1] > sigmas[0]) and not extra_options_flag("preview_last_step_always", extra_options):
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






