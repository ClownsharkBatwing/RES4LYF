import torch
from tqdm.auto import trange
import gc

from .noise_classes import *
from .rk_method_beta import RK_Method_Beta, RK_NoiseSampler
from .rk_guide_func_beta import *
from .helper import get_extra_options_kv, extra_options_flag, lagrange_interpolation

MAX_STEPS=10000
PRINT_DEBUG=False



def init_implicit_sampling(RK, x_0, x_, eps_, eps_prev_, data_, eps, denoised, denoised_prev, denoised_prev2, step, sigmas, h, s_, extra_options):
    
    sigma = sigmas[step]
    if extra_options_flag("implicit_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
        eps_[0], data_[0] = eps.clone(), denoised.clone()
        eps_[0] = RK.get_epsilon_anchored(x_0, denoised, sigma)
        if denoised_prev2.sum() != 0:
            sratio = sigma - s_[0]
            data_[0] = denoised + sratio * (denoised - denoised_prev2)
            
    elif extra_options_flag("implicit_full_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
        eps_[0], data_[0] = eps.clone(), denoised.clone()
        eps_[0] = RK.get_epsilon_anchored(x_0, denoised, sigma)
        if denoised_prev2.sum() != 0:
            for r in range(RK.rows):
                sratio = sigma - s_[r]
                data_[r] = denoised + sratio * (denoised - denoised_prev2)
                eps_[r] = RK.get_epsilon_anchored(x_0, data_[r], s_[r])

    elif extra_options_flag("implicit_lagrange_skip_model_call_at_start", extra_options) and denoised.sum() + eps.sum() != 0:
        if denoised_prev2.sum() != 0:
            sigma_prev = sigmas[step-1]
            h_prev = sigma - sigma_prev
            w = h / h_prev
            substeps_prev = len(RK.C[:-1])
            
            for r in range(RK.rows):
                sratio = sigma - s_[r]
                data_[r] = lagrange_interpolation([0,1], [denoised_prev2, denoised], 1 + w*RK.C[r]).squeeze(0) + denoised_prev2 - denoised
                eps_[r]  = RK.get_epsilon_anchored(x_0, data_[r], s_[r])      
                
            if extra_options_flag("implicit_lagrange_skip_model_call_at_start_0_only", extra_options):
                for r in range(RK.rows):
                    eps_ [r] = eps_ [0].clone() * s_[0] / s_[r]
                    data_[r] = denoised.clone()

        else:
            eps_[0], data_[0] = eps.clone(), denoised.clone()
            eps_[0] = RK.get_epsilon_anchored(x_0, denoised, sigma)      

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
def sample_rk_beta(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian",  noise_sampler_type_substep="gaussian", noise_mode_sde="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="explicit_full",
                  eta=0.0, s_noise=1., s_noise_substep=1., d_noise=1., alpha=-1.0, alpha_substep=-1.0, k=1.0, k_substep=1.0, c1=0.0, c2=0.5, c3=1.0, implicit_steps_diag=0, implicit_steps_full=0, 
                  LGW_MASK_RESCALE_MIN=True, sigmas_override=None, sampler_mode="standard", unsample_resample_scales=None,regional_conditioning_weights=None, sde_noise=[],
                  extra_options="",
                  etas=None, etas_substep=None, s_noises=None, s_noises_substep=None, momentums=None, guides=None, cfgpp=0.0, cfg_cw = 1.0,regional_conditioning_floors=None, frame_weights_grp=None, eta_substep=0.0, noise_mode_sde_substep="hard",
                  noise_boost_step=0.0, noise_boost_substep=0.0, overshoot=0.0, overshoot_substep=0.0, overshoot_mode="hard", overshoot_mode_substep="hard", BONGMATH=True,
                  ):
    extra_args = {} if extra_args is None else extra_args
    default_device = x.device
    default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "", extra_options), x.dtype)
    x      = x.to(default_dtype)
    sigmas = sigmas.to(default_dtype)

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

    # SETUP SAMPLER
    if implicit_sampler_name not in ("use_explicit", "none"):
        rk_type = implicit_sampler_name
    print("rk_type:", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0
        
    RK = RK_Method_Beta.create(model, rk_type, device=default_device, dtype=default_dtype, extra_options=extra_options)
    RK.extra_args = RK.init_cfg_channelwise(x, cfg_cw, **extra_args)
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0
    
    # SETUP SIGMAS
    NS = RK_NoiseSampler(RK, model, device=default_device, dtype=default_dtype, extra_options=extra_options)
    sigmas, UNSAMPLE = NS.prepare_sigmas(sigmas, sigmas_override, d_noise, sampler_mode)
    
    SDE_NOISE_EXTERNAL = False
    if sde_noise is not None:
        if len(sde_noise) > 0 and sigmas[1] > sigmas[2]:
            SDE_NOISE_EXTERNAL = True
            sigma_up_total = torch.zeros_like(sigmas[0])
            for i in range(len(sde_noise)-1):
                sigma_up_total += sigmas[i+1]
            etas = torch.full_like(sigmas, eta / sigma_up_total)
    
    NS.init_noise_samplers(x, noise_seed, noise_sampler_type, noise_sampler_type_substep, noise_mode_sde, noise_mode_sde_substep, overshoot_mode, overshoot_mode_substep, noise_boost_step, noise_boost_substep, alpha, alpha_substep, k, k_substep)

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
        
        NS.set_sde_step(sigma, sigma_next, eta, overshoot, s_noise)
        RK.set_coeff(rk_type, NS.h, c1, c2, c3, step, sigmas, NS.sigma_down)
        NS.set_substep_list(RK)

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
        
        x_[0] = x.clone()
        # PRENOISE METHOD HERE!
        x_0 = x_[0].clone()

        # RECYCLE STAGES FOR MULTISTEP
        for ms in range(len(eps_)):
            eps_[ms] = RK.get_epsilon_anchored(x_0, data_prev_[ms], sigma)
        eps_prev_ = eps_.clone()

        # INITIALIZE IMPLICIT SAMPLING
        if RK.IMPLICIT:
            LG.to(LG.offload_device)
            x_, eps_, data_ = init_implicit_sampling(RK, x_0, x_, eps_, eps_prev_, data_, eps, denoised, denoised_prev, denoised_prev2, step, sigmas, NS.h, NS.s_, extra_options)
            LG.to(LG.device)

        # BEGIN FULLY IMPLICIT LOOP
        for full_iter in range(implicit_steps_full + 1):
            
            if RK.IMPLICIT:
                x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, 0, NS.h, sigmas, step, "init", extra_options)

            # PREPARE FULLY PSEUDOIMPLICIT GUIDES
            x_0, x_, eps_ = LG.prepare_fully_pseudoimplicit_guides_substep(x_0, x_, eps_, eps_prev_, data_, 0, step, sigmas, eta_substep, overshoot_substep, s_noise_substep, NS, RK, pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, BONGMATH, extra_options)

            # TABLEAU LOOP
            for row in range(RK.rows - RK.multistep_stages - RK.row_offset + 1):
                for diag_iter in range(implicit_steps_diag+1):
                    
                    NS.set_sde_substep(row, RK.multistep_stages, eta_substep, overshoot_substep, s_noise_substep, full_iter, diag_iter, implicit_steps_full, implicit_steps_diag)

                    # PREPARE PSEUDOIMPLICIT GUIDES
                    x_0, x_row_pseudoimplicit, sub_sigma_pseudoimplicit = LG.process_pseudoimplicit_guides_substep(x_0, x_, eps_, eps_prev_, data_, row, step, sigmas, NS, RK, pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, BONGMATH, extra_options)
                    
                    if BONGMATH and step < sigmas.shape[0]-1 and not extra_options_flag("disable_pseudobongmath", extra_options) and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0):
                        x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h_new, extra_options)
                    
                    # PRENOISE METHOD HERE!
                    
                    # A-TABLEAU
                    if row < RK.rows:

                        # PREPARE MODEL CALL
                        if LG.guide_mode in {"pseudoimplicit", "pseudoimplicit_projection", "fully_pseudoimplicit", "fully_pseudoimplicit_projection","fully_pseudoimplicit_cw", "fully_pseudoimplicit_projection_cw"} and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0) and x_row_pseudoimplicit is not None:
                            x_tmp =     x_row_pseudoimplicit 
                            s_tmp = sub_sigma_pseudoimplicit 
                            
                        # Fully implicit iteration (explicit only)                   # or... Fully implicit iteration (implicit only... not standard) 
                        elif (full_iter > 0 and RK.row_offset == 1 and row == 0)   or   (full_iter > 0 and RK.row_offset == 0 and row == 0 and extra_options_flag("fully_implicit_update_x", extra_options)):
                            if extra_options_flag("fully_explicit_pogostick_eta", extra_options): 
                                super_alpha_ratio, super_sigma_down, super_sigma_up = NS.get_sde_coeff(sigma, sigma_next, None, eta)
                                x = super_alpha_ratio * x + super_sigma_up * NS.noise_sampler(sigma=sigma, sigma_next=sigma_next)
                                
                                x_tmp = x
                                s_tmp = sigma
                            else:
                                x_tmp = x
                                s_tmp = sigma_next
                        
                        # All others
                        else:
                            if diag_iter > 0: # Diagonally implicit iteration (explicit or implicit)
                                x_tmp =    x_[row+RK.row_offset]
                                s_tmp = NS.s_[row+RK.row_offset+RK.multistep_stages]
                            else:
                                x_tmp = x_[row]
                                s_tmp = NS.sub_sigma 

                            if RK.IMPLICIT: 
                                if not extra_options_flag("implicit_guide_preproc_disable", extra_options):
                                    eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, NS.sigma_down, NS.s_, unsample_resample_scale, RK, extra_options)
                                    eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, NS.sigma_down, NS.s_, unsample_resample_scale, RK, extra_options)
                                if row == 0 and (extra_options_flag("implicit_lagrange_init", extra_options)  or   extra_options_flag("radaucycle", extra_options)):
                                    pass
                                else:
                                    x_tmp = x_[row+RK.row_offset] = x_0 + NS.h_new * RK.zum(row+RK.row_offset, eps_, eps_prev_)
                                    if row > 0:
                                        x_tmp = x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset])
                                    if BONGMATH and step < sigmas.shape[0]-1:
                                        x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, extra_options)
                                        x_tmp = x_[row+RK.row_offset]

                        # MODEL CALL
                        if RK.IMPLICIT   and   row == 0   and   (extra_options_flag("implicit_lazy_recycle_first_model_call_at_start", extra_options)   or   extra_options_flag("radaucycle", extra_options)):
                            pass
                        else: 
                            if s_tmp == 0:
                                break
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "pre", extra_options) # will this do anything? not x_tmp

                            LG.to(LG.offload_device)
                            eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                            LG.to(LG.device)

                            if extra_options_flag("preview_substeps", extra_options):
                                callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': data_[row].to(torch.float32)}) if callback is not None else None

                        # GUIDE 
                        if not extra_options_flag("disable_guides_eps_substep", extra_options):
                            eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, NS.sigma, NS.sigma_next, NS.sigma_down, NS.s_, unsample_resample_scale, RK, extra_options)
                        if not extra_options_flag("disable_guides_eps_prev_substep", extra_options):
                            eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, NS.sigma, NS.sigma_next, NS.sigma_down, NS.s_, unsample_resample_scale, RK, extra_options)
                            
                        if (full_iter == 0 and diag_iter == 0)   or   extra_options_flag("newton_iter_post_use_on_implicit_steps", extra_options):
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "post", extra_options)

                    # UPDATE
                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, RK.row_offset, NS.h_new, NS.h_new_orig, extra_options)
                    
                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                    if not RK.IMPLICIT and NS.noise_mode_sde_substep != "hard_sq":
                        x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset])

                    if BONGMATH and step < sigmas.shape[0]-1:
                        x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, extra_options)

            denoised = x_0 + ((sigma / (sigma - NS.sigma_down)) *  NS.h_new) * RK.zum(RK.rows, eps_, eps_prev_)
            eps = RK.get_epsilon_anchored(x_0, denoised, sigma_next)
            
            x = x_[RK.rows - RK.multistep_stages - RK.row_offset + 1]

            x = NS.rebound_overshoot_step(x_0, x)
            x = LG.process_guides_poststep(x, denoised, eps, step, extra_options)
            
            preview_callback(x, eps, denoised, x_, eps_, data_, step, sigma, sigma_next, callback, extra_options)
            
            x = NS.swap_noise_step(x_0, x)

        data_prev_[0] = data_[0]
        for ms in range(recycled_stages):
            data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
        
        rk_type = RK.swap_rk_type_at_step_or_threshold(x_0, data_prev_, NS.sigma_down, sigmas, step, RK, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print)

        denoised_prev2 = denoised_prev.to(denoised_prev2.device)
        denoised_prev  = denoised.to(denoised_prev.device)

    if sigmas[-1] == 0 and sigmas[-2] == NS.sigma_min:
        eps, denoised = RK(x, NS.sigma_min, x, NS.sigma_min)
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





