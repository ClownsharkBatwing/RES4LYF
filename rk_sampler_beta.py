import torch
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import itertools

from tqdm.auto import trange
import gc

from .noise_classes import *
from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_alpha_ratio_from_sigma_down

from .rk_method_beta import RK_Method_Beta, RK_NoiseSampler, vpsde_noise_add
from .rk_guide_func_beta import *

from .latents import normalize_latent, initialize_or_scale, latent_normalize_channels
from .helper import get_extra_options_kv, extra_options_flag, get_cosine_similarity, get_pearson_similarity, lagrange_interpolation
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


def get_epsilon2(x_0, x, denoised, sigma, rk_type):
    if RK_Method_Beta.is_exponential(rk_type):
        eps = denoised - x_0
    else:
        eps = (x - denoised) / sigma
    return eps


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




def init_implicit_sampling(RK, x_0, x_, eps_, eps_prev_, data_, eps, denoised, denoised_prev, denoised_prev2, step, sigmas, h, s_, extra_options):
    sigma = sigmas[step]
    if extra_options_flag("implicit_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
        eps_[0], data_[0] = eps.clone(), denoised.clone()
        eps_[0] = get_epsilon(x_0, denoised, sigma, RK.rk_type)
        if denoised_prev2.sum() != 0:
            sratio = sigma - s_[0]
            data_[0] = denoised + sratio * (denoised - denoised_prev2)
            
    elif extra_options_flag("implicit_full_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
        eps_[0], data_[0] = eps.clone(), denoised.clone()
        eps_[0] = get_epsilon(x_0, denoised, sigma, RK.rk_type)
        if denoised_prev2.sum() != 0:
            for r in range(RK.rows):
                sratio = sigma - s_[r]
                data_[r] = denoised + sratio * (denoised - denoised_prev2)
                eps_[r] = get_epsilon(x_0, data_[r], s_[r], RK.rk_type)

    elif extra_options_flag("implicit_lagrange_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
        if denoised_prev2.sum() != 0:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            w = h / h_prev
            substeps_prev = len(RK.C[:-1])
            
            for r in range(RK.rows):
                sratio = sigma - s_[r]
                data_[r] = lagrange_interpolation([0,1], [denoised_prev2, denoised], 1 + w*RK.C[r]).squeeze(0) + denoised_prev2 - denoised
                eps_[r]  = get_epsilon(x_0, data_[r], s_[r], RK.rk_type)      
                
            if extra_options_flag("implicit_lagrange_skip_model_call_at_start_0_only", extra_options):
                for r in range(RK.rows):
                    eps_ [r] = eps_ [0].clone() * s_[0] / s_[r]
                    data_[r] = denoised.clone()

        else:
            eps_[0], data_[0] = eps.clone(), denoised.clone()
            eps_[0] = get_epsilon(x_0, denoised, sigma, RK.rk_type)      

    elif extra_options_flag("implicit_lagrange_init", extra_options) and denoised.sum() + eps.sum() != 0:
        sigma_prev = sigmas[step-1]
        h_prev = sigma - sigma_prev
        w = h / h_prev
        substeps_prev = len(RK.C[:-1])

        z_prev_ = eps_.clone()
        for r in range (substeps_prev):
            z_prev_[r] = h * RK.zum(r, eps_) # u,v not implemented for lagrange guess for implicit
        zi_1 = lagrange_interpolation(RK.C[:-1], z_prev_[:substeps_prev], RK.C[0]).squeeze(0) # + x_prev - x_0"""
        x_[0] = x_0 + zi_1
        
    else:
        
        eps_[0], data_[0] = RK(x_[0], sigma, x_0, sigma)
    
    if not extra_options_flag("implicit_lagrange_init", extra_options) and not extra_options_flag("radaucycle", extra_options) and not extra_options_flag("implicit_full_skip_model_call_at_start", extra_options) and not extra_options_flag("implicit_lagrange_skip_model_call_at_start", extra_options):
        for r in range(RK.rows):
            eps_ [r] = eps_ [0].clone() * sigma / s_[r]
            data_[r] = data_[0].clone()

    x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "init", extra_options)
    return x_, eps_, data_






@torch.no_grad()
def sample_rk_beta(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian",  noise_sampler_type_substep="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="explicit_full",
                  eta=0.0, eta_var=0.0, s_noise=1., s_noise_substep=1., d_noise=1., alpha=-1.0, alpha_substep=-1.0, k=1.0, k_substep=1.0, c1=0.0, c2=0.5, c3=1.0, implicit_steps_diag=0, implicit_steps_full=0, 
                  LGW_MASK_RESCALE_MIN=True, sigmas_override=None, unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, etas_substep=None, s_noises=None, s_noises_substep=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights_grp=None, eta_substep=0.0, noise_mode_sde_substep="hard",
                  noise_boost_step=0.0, noise_boost_substep=0.0,
                  ):
    extra_args = {} if extra_args is None else extra_args
    default_device = x.device
    default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "", extra_options), x.dtype)
    x      = x.to(default_dtype)
    sigmas = sigmas.to(default_dtype)
    MAX_STEPS=10000

    if noise_seed < 0:
        noise_seed = torch.initial_seed()+1 
        print("Set noise_seed to:", noise_seed, " using torch.initial_seed()+1")

    c1 = float(get_extra_options_kv("c1", str(c1), extra_options))
    c2 = float(get_extra_options_kv("c2", str(c2), extra_options))
    c3 = float(get_extra_options_kv("c3", str(c3), extra_options))
    
    overstep_eta  = float(get_extra_options_kv("overstep_eta",   "0.0", extra_options))
    overstep_mode =       get_extra_options_kv("overstep_mode", "hard", extra_options)

    guide_skip_steps  =   int(get_extra_options_kv("guide_skip_steps", 0,          extra_options))   
    rk_swap_step      =   int(get_extra_options_kv("rk_swap_step", str(MAX_STEPS), extra_options))
    rk_swap_print     =         extra_options_flag("rk_swap_print", extra_options)
    rk_swap_threshold = float(get_extra_options_kv("rk_swap_threshold", "0.0",     extra_options))
    rk_swap_type      =       get_extra_options_kv("rk_swap_type",         "",     extra_options)   
    CONSERVE_MEAN_CW  =         extra_options_flag("eta_conserve_mean_cw",         extra_options)  
    SYNC_MEAN_CW  =         not extra_options_flag("eta_sync_mean_cw_disable",     extra_options)  
    LOCK_H_SCALE      =         extra_options_flag("lock_h_scale",                 extra_options)  
    
    DOWN_SUBSTEP      =         extra_options_flag("down_substep",                 extra_options)  
    DOWN_STEP         =         extra_options_flag("down_step",                    extra_options)  
    
    brownian_sub_start = get_extra_options_kv("brownian_sub_start", "sub_sigma",    extra_options)
    brownian_sub_stop  = get_extra_options_kv("brownian_sub_stop",  "sub_sigma_next", extra_options)
    brownian_main_start = get_extra_options_kv("brownian_main_start", "sigma",    extra_options)
    brownian_main_stop  = get_extra_options_kv("brownian_main_stop",  "sigma_next", extra_options)
    
    reorder_tableau_indices = get_extra_options_list("reorder_tableau_indices", "", extra_options).split(",")
    if reorder_tableau_indices[0]:
        reorder_tableau_indices = [int(reorder_tableau_indices[_]) for _ in range(len(reorder_tableau_indices))]

    pseudoimplicit_step_weights = get_extra_options_list("pseudoimplicit_step_weights", "", extra_options).split(",")
    if pseudoimplicit_step_weights[0]:
        pseudoimplicit_step_weights = [float(pseudoimplicit_step_weights[_]) for _ in range(len(pseudoimplicit_step_weights))]
    else:
        pseudoimplicit_step_weights = [1. for _ in range(max(implicit_steps_diag, implicit_steps_full)+1)]

    pseudoimplicit_row_weights = get_extra_options_list("pseudoimplicit_row_weights", "", extra_options).split(",")
    if pseudoimplicit_row_weights[0]:
        pseudoimplicit_row_weights = [float(pseudoimplicit_row_weights[_]) for _ in range(len(pseudoimplicit_row_weights))]
    else:
        pseudoimplicit_row_weights = [1. for _ in range(100)]

    cfg_cw = float(get_extra_options_kv("cfg_cw", str(cfg_cw), extra_options))



    # SETUP SIGMAS
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
            etas = torch.full_like(sigmas, eta / sigma_up_total)
            
    sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(default_dtype).to(default_device)
    if sigmas[-1] == 0:
        if sigmas[-2] < sigma_min:
            sigmas[-2] = sigma_min
        elif sigmas[-2] > sigma_min:
            sigmas = torch.cat((sigmas[:-1], sigma_min.unsqueeze(0), sigmas[-1:]))

    # SETUP SAMPLER
    if implicit_sampler_name not in ("use_explicit", "none"):
        rk_type = implicit_sampler_name
    print("rk_type:", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0

    RK = RK_Method_Beta.create(model, rk_type, device=default_device, dtype=default_dtype)
    RK.extra_args = RK.init_cfg_channelwise(x, cfg_cw, **extra_args)
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0
    
    NS = RK_NoiseSampler(model, sigmas, device=default_device, dtype=default_dtype)
    NS.init_noise_samplers(x, noise_seed, noise_sampler_type, noise_sampler_type_substep, noise_mode, noise_mode_sde_substep, alpha, alpha_substep, k, k_substep)
    NS.LOCK_H_SCALE = LOCK_H_SCALE
    

    # SETUP GUIDES
    LG = LatentGuide(model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options, device=default_device, dtype=default_dtype, frame_weights_grp=frame_weights_grp)
    x = LG.init_guides(x, guides, NS.noise_sampler)

    data_           = None
    eps_            = None
    eps             = torch.zeros_like(x)
    denoised        = torch.zeros_like(x)
    denoised_prev   = torch.zeros_like(x).to('cpu')
    denoised_prev2  = torch.zeros_like(x)
    x_          = None
    eps_prev_   = None

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # BEGIN SAMPLING LOOP
    num_steps = len(sigmas)-2 if sigmas[-1] == 0 else len(sigmas)-1
    for step in trange(num_steps, disable=disable):
        sigma, sigma_next = sigmas[step], sigmas[step+1]
                
        if regional_conditioning_weights is not None:
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = regional_conditioning_weights[step]
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = regional_conditioning_floors [step]
        
        unsample_resample_scale = float(unsample_resample_scales[step]) if unsample_resample_scales is not None else None
        eta             = etas            [step] if etas             is not None else eta
        eta_substep     = etas_substep    [step] if etas_substep     is not None else eta_substep
        s_noise         = s_noises        [step] if s_noises         is not None else s_noise
        s_noise_substep = s_noises_substep[step] if s_noises_substep is not None else s_noise_substep

        sigma_up_overstep, sigma_overstep, sigma_down_overstep, alpha_ratio_overstep = NS.get_sde_step(sigma, sigma_next, overstep_eta, overstep_mode, DOWN=DOWN_STEP)
        sigma_up, sigma, sigma_down, alpha_ratio = NS.get_sde_step(sigma, sigma_down_overstep, eta, DOWN=DOWN_STEP)
        
        real_sigma_down = sigma_down.clone()
        if extra_options_flag("lock_h_scale", extra_options):
            sigma_down = sigma_down_overstep

        h         = RK.h_fn(sigma_down, sigma)
        h_no_eta  = RK.h_fn(sigma_next, sigma)
        h = h + noise_boost_step * (h_no_eta - h)

        RK.set_coeff(rk_type, h, c1, c2, c3, step, sigmas, sigma_down, extra_options)
        if RK.IMPLICIT:
            RK.reorder_tableau(reorder_tableau_indices)
        row_offset = 1 if not RK.IMPLICIT and RK.A[0].sum() == 0 else 0   

        s_ = [(RK.sigma_fn(RK.t_fn(sigma) + h*c_)) * x.new_ones([1]) for c_ in RK.C]

        recycled_stages = max(RK.multistep_stages, RK.hybrid_stages)
        if step == 0 or step == guide_skip_steps:
            x_, data_, eps_ = (torch.zeros(    RK.rows+2,     *x.shape, dtype=default_dtype, device=x.device) for _ in range(3))
            data_prev_      =  torch.zeros(max(RK.rows+2, 4), *x.shape, dtype=default_dtype, device=x.device)
            recycled_stages = len(data_prev_)-1

        sde_noise_t = None
        if SDE_NOISE_EXTERNAL:
            if step >= len(sde_noise):
                SDE_NOISE_EXTERNAL=False
            else:
                sde_noise_t = sde_noise[step]
        
        x_[0] = NS.add_noise_pre(x, x, sigma_up, sigma, sigma_next, real_sigma_down, alpha_ratio, s_noise, noise_mode, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t)
        x_0 = x_[0].clone()

        # RECYCLE STAGES FOR MULTISTEP
        for ms in range(len(eps_)):
            eps_[ms] = get_epsilon(x_0, data_prev_[ms], sigma, rk_type)
        eps_prev_ = eps_.clone()

        # INITIALIZE IMPLICIT SAMPLING
        if RK.IMPLICIT:
            LG.to(LG.offload_device)
            x_, eps_, data_ = init_implicit_sampling(RK, x_0, x_, eps_, eps_prev_, data_, eps, denoised, denoised_prev, denoised_prev2, step, sigmas, h, s_, extra_options)
            LG.to(LG.device)

        # BEGIN FULLY IMPLICIT LOOP
        for full_iter in range(implicit_steps_full + 1):
            
            if RK.IMPLICIT: # and extra_options_flag("fully_implicit_newton_iter", extra_options):
                x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "init", extra_options)

            # PREPARE FULLY PSEUDOIMPLICIT GUIDES
            x_, eps_ = LG.prepare_fully_pseudoimplicit_guides_substep(model, x_0, x_, eps_, eps_prev_, data_, 0, row_offset, h, step, sigmas, s_, s_noise_substep, noise_mode_sde_substep, NS, \
                                                    noise_boost_substep, eta_substep, SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, RK, rk_type, pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, extra_options)

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
                            sub_sigma_up, sub_sigma, sub_sigma_down, sub_alpha_ratio = NS.get_sde_substep(s_[row], s_[row+row_offset+RK.multistep_stages], eta_substep, DOWN=DOWN_SUBSTEP)
                            
                    real_sub_sigma_down = sub_sigma_down.clone()
                    if extra_options_flag("lock_h_scale", extra_options):
                        sub_sigma_down = sub_sigma_next
                    
                    h_new = h * RK.h_fn(sub_sigma_down, sigma) / RK.h_fn(sub_sigma_next, sigma) 
                    h_new_orig = h_new.clone()
                    h_new = h_new + noise_boost_substep * (h - h_new)

                    # PREPARE PSEUDOIMPLICIT GUIDES
                    x_row_pseudoimplicit, sub_sigma_pseudoimplicit = LG.process_pseudoimplicit_guides_substep(model, x_0, x_, eps_, eps_prev_, data_, row, row_offset, h, h_new, h_new_orig, step, sigmas, s_, real_sub_sigma_down, sub_sigma_up, sub_sigma, sub_sigma_next, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, NS, \
                                                    noise_boost_substep, eta_substep, SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, RK, rk_type, pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, extra_options)

                    x_[row] = NS.add_noise_pre(x_0, x_[row], sub_sigma_up, sub_sigma, sub_sigma_next, real_sub_sigma_down, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t)

                    # MODEL CALL
                    if row < RK.rows: # A-tableau still

                        # PREPARE MODEL CALL
                        if LG.guide_mode in {"pseudoimplicit", "pseudoimplicit_projection", "fully_pseudoimplicit", "fully_pseudoimplicit_projection"} and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0) and x_row_pseudoimplicit is not None:
                            x_tmp =     x_row_pseudoimplicit 
                            s_tmp = sub_sigma_pseudoimplicit 
                            
                        # Fully implicit iteration (explicit only)                   # or... Fully implicit iteration (implicit only... not standard) 
                        elif (full_iter > 0 and row_offset == 1 and row == 0)   or   (full_iter > 0 and row_offset == 0 and row == 0 and extra_options_flag("fully_implicit_update_x", extra_options)):
                            if extra_options_flag("fully_explicit_pogostick_eta", extra_options): 
                                super_alpha_ratio, super_sigma_up, super_sigma_down = get_alpha_ratio_from_sigma_down(sigma_next, sigma, eta)
                                x = super_alpha_ratio * x + super_sigma_up * NS.noise_sampler(sigma=sigma, sigma_next=sigma_next)
                                
                                x_tmp = x
                                s_tmp = sigma
                            else:
                                x_tmp = x
                                s_tmp = sigma_next
                            
                        # All others
                        else:
                            if diag_iter > 0: # Diagonally implicit iteration (explicit or implicit)
                                x_tmp = x_[row+row_offset]
                                s_tmp = s_[row+row_offset+RK.multistep_stages]
                            else:
                                x_tmp = x_[row]
                                s_tmp = sub_sigma #s_[row]

                            if RK.IMPLICIT: 
                                if not extra_options_flag("implicit_guide_preproc_disable", extra_options):
                                    eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options)
                                    eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options)
                                if row == 0 and (extra_options_flag("implicit_lagrange_init", extra_options)  or   extra_options_flag("radaucycle", extra_options)):
                                    pass
                                elif row == 0 and not extra_options_flag("substep_eta_use_final", extra_options):
                                    x_tmp = x_[row+row_offset] = x_0 + h     * RK.zum(row+row_offset, eps_, eps_prev_)
                                else:
                                    x_tmp = x_[row+row_offset] = x_0 + h_new * RK.zum(row+row_offset, eps_, eps_prev_)
                                    x_tmp = x_[row+row_offset] = NS.add_noise_post(x_0, x_[row+row_offset], sub_sigma_up, sigma, sub_sigma, sub_sigma_next, real_sub_sigma_down, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP=True)

                        # MAKE MODEL CALL
                        if RK.IMPLICIT   and   row == 0   and   (extra_options_flag("implicit_lazy_recycle_first_model_call_at_start", extra_options)   or   extra_options_flag("radaucycle", extra_options)):
                            pass
                        else: 
                            if s_tmp == 0:
                                break
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, row, h, sigmas, step, "pre", extra_options) # will this do anything? not x_tmp

                            LG.to(LG.offload_device)
                            eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                            LG.to(LG.device)

                            if extra_options_flag("preview_substeps", extra_options):
                                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data_[row].to(torch.float32)}) if callback is not None else None

                        # GUIDE 
                        if not extra_options_flag("disable_guides_eps_substep", extra_options):
                            eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options)
                        if not extra_options_flag("disable_guides_eps_prev_substep", extra_options):
                            eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, RK, rk_type, extra_options)

                        if (full_iter == 0 and diag_iter == 0)   or   extra_options_flag("newton_iter_post_use_on_implicit_steps", extra_options):
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, row, h, sigmas, step, "post", extra_options)

                    # UPDATE
                    s_dict = {"sigma": sigma, "sigma_next": sigma_next, "sigma_down": sigma_down, "sigma_up": sigma_up, "sub_sigma": sub_sigma, "sub_sigma_next": sub_sigma_next, "sub_sigma_up": sub_sigma_up, "sub_sigma_down": sub_sigma_down}
                    if row < RK.rows - row_offset   and   RK.multistep_stages == 0:
                        
                        if brownian_sub_stop == "proportional" or brownian_sub_start == "proportional":
                            sigma_brownian = sigma - row * (sigma - sigma_next) / RK.rows
                            sigma_next_brownian = sigma - (row+1) * (sigma - sigma_next) / RK.rows
                        else:
                            sigma_brownian = s_dict[brownian_sub_start]
                            sigma_next_brownian = s_dict[brownian_sub_stop]
                        
                    else:
                        sigma_brownian = s_dict[brownian_main_start]
                        sigma_next_brownian = s_dict[brownian_main_stop]
                        
                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, row_offset, h, h_new, h_new_orig, sigma, real_sub_sigma_down, sub_sigma_up, sigma_brownian, sigma_next_brownian, sub_alpha_ratio, s_noise_substep, noise_mode_sde_substep, NS, \
                                           SYNC_MEAN_CW, CONSERVE_MEAN_CW, SDE_NOISE_EXTERNAL, sde_noise_t, extra_options)
                    
                    if extra_options_flag("use_bong", extra_options) and step < sigmas.shape[0]-3:
                        x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, s_, row, row_offset, h, extra_options)

            denoised = x_0 + ((sigma / (sigma - sigma_down)) *  h_new) * RK.zum(RK.rows, eps_, eps_prev_)
            eps = get_epsilon(x_0, denoised, sigma_next, rk_type)
            
            x = x_[RK.rows - RK.multistep_stages - row_offset + 1]
            
            if extra_options_flag("overstep", extra_options):
                eps = (x_0 - x) / (sigma - sigma_down_overstep)
                denoised = x_0 - sigma * eps
                x = denoised + sigma_down * eps
            
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)
            
            preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options)

            x = NS.add_noise_post(x_0, x, sigma_up, sigma, sigma, sigma_next, real_sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)
            
            if overstep_eta != 0.0:
                noise = NS.noise_sampler(sigma=sigma, sigma_next=sigma_next)
                x = alpha_ratio_overstep * x + sigma_up_overstep * noise

        data_prev_[0] = data_[0]
        for ms in range(recycled_stages):
            data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
        
        rk_type = RK.swap_rk_type_at_step_or_threshold(x_0, data_prev_, sigma_down, sigmas, step, RK, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print)

        denoised_prev2 = denoised_prev.to(denoised_prev2.device)
        denoised_prev  = denoised.to(denoised_prev.device)

    if sigmas[-1] == 0 and sigmas[-2] == sigma_min:
        eps, denoised = RK(x, sigma_min, x, sigma_min)
        x = denoised

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





