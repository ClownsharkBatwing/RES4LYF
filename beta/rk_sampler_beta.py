import torch
from torch import Tensor
from tqdm.auto import trange
import gc
from typing import Optional, Callable, Tuple, Dict, Any, Union

from ..res4lyf              import RESplain
from ..helper               import ExtraOptions
from ..latents              import lagrange_interpolation, get_collinear, get_orthogonal

from .rk_method_beta        import RK_Method_Beta
from .rk_noise_sampler_beta import RK_NoiseSampler
from .rk_guide_func_beta    import LatentGuide
from .phi_functions         import Phi
from .constants             import MAX_STEPS, GUIDE_MODE_NAMES_PSEUDOIMPLICIT


def init_implicit_sampling(
        RK             : RK_Method_Beta,
        x_0            : Tensor,
        x_             : Tensor,
        eps_           : Tensor,
        eps_prev_      : Tensor,
        data_          : Tensor,
        eps            : Tensor,
        denoised       : Tensor,
        denoised_prev2 : Tensor,
        step           : int,
        sigmas         : Tensor,
        h              : Tensor,
        s_             : Tensor,
        EO             : ExtraOptions,
        ):
    
    sigma = sigmas[step]
    if EO("implicit_skip_model_call_at_start") and denoised.sum() + eps.sum() != 0:
        if denoised_prev2.sum() == 0:
            eps_ [0] = eps.clone()
            data_[0] = denoised.clone()
            eps_ [0] = RK.get_epsilon_anchored(x_0, denoised, sigma)
        else:
            sratio = sigma - s_[0]
            data_[0] = denoised + sratio * (denoised - denoised_prev2)
            
    elif EO("implicit_full_skip_model_call_at_start") and denoised.sum() + eps.sum() != 0:
        if denoised_prev2.sum() == 0:
            eps_ [0] = eps.clone()
            data_[0] = denoised.clone()
            eps_ [0] = RK.get_epsilon_anchored(x_0, denoised, sigma)
        else:
            for r in range(RK.rows):
                sratio = sigma - s_[r]
                data_[r] = denoised + sratio * (denoised - denoised_prev2)
                eps_ [r] = RK.get_epsilon_anchored(x_0, data_[r], s_[r])

    elif EO("implicit_lagrange_skip_model_call_at_start") and denoised.sum() + eps.sum() != 0:
        if denoised_prev2.sum() == 0:
            eps_ [0] = eps.clone()
            data_[0] = denoised.clone()
            eps_ [0] = RK.get_epsilon_anchored(x_0, denoised, sigma)   
        else:
            sigma_prev    = sigmas[step-1]
            h_prev        = sigma - sigma_prev
            w             = h / h_prev
            substeps_prev = len(RK.C[:-1])
            
            for r in range(RK.rows):
                sratio = sigma - s_[r]
                data_[r] = lagrange_interpolation([0,1], [denoised_prev2, denoised], 1 + w*RK.C[r]).squeeze(0) + denoised_prev2 - denoised
                eps_ [r] = RK.get_epsilon_anchored(x_0, data_[r], s_[r])      
                
            if EO("implicit_lagrange_skip_model_call_at_start_0_only"):
                for r in range(RK.rows):
                    eps_ [r] = eps_ [0].clone() * s_[0] / s_[r]
                    data_[r] = denoised.clone()


    elif EO("implicit_lagrange_init") and denoised.sum() + eps.sum() != 0:
        sigma_prev    = sigmas[step-1]
        h_prev        = sigma - sigma_prev
        w             = h / h_prev
        substeps_prev = len(RK.C[:-1])

        z_prev_ = eps_.clone()
        for r in range (substeps_prev):
            z_prev_[r] = h * RK.zum(r, eps_) # u,v not implemented for lagrange guess for implicit
        zi_1  = lagrange_interpolation(RK.C[:-1], z_prev_[:substeps_prev], RK.C[0]).squeeze(0) # + x_prev - x_0"""
        x_[0] = x_0 + zi_1
        
    else:
        
        eps_[0], data_[0] = RK(x_[0], sigma, x_0, sigma)

    if not EO(("implicit_lagrange_init", "radaucycle", "implicit_full_skip_model_call_at_start", "implicit_lagrange_skip_model_call_at_start")):
        for r in range(RK.rows):
            eps_ [r] = eps_ [0].clone() * sigma / s_[r]
            data_[r] = data_[0].clone()

    x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "init")
    return x_, eps_, data_




@torch.no_grad()
def sample_rk_beta(
        model,
        x                             : Tensor,
        sigmas                        : Tensor,
        sigmas_override               : Optional[Tensor]   = None,
        
        extra_args                    : Optional[Tensor]   = None,
        callback                      : Optional[Callable] = None,
        disable                       : bool               = None,
        
        sampler_mode                  : str                = "standard",

        rk_type                       : str                = "res_2m",
        implicit_sampler_name         : str                = "explicit_full",

        c1                            : float              =  0.0,
        c2                            : float              =  0.5,
        c3                            : float              =  1.0,

        noise_sampler_type            : str                = "gaussian",
        noise_sampler_type_substep    : str                = "gaussian",
        noise_mode_sde                : str                = "hard",
        noise_mode_sde_substep        : str                = "hard",

        eta                           : float              =  0.0,
        eta_substep                   : float              =  0.0,




        noise_scaling_substep         : float = 0.0,
        noise_scaling_type            : str   = "sampler",
        noise_scaling_mode            : str   = "linear",
        noise_scaling_eta             : float = 0.0,
        noise_scaling_cycles          : int   = 1,
        
        noise_boost_step              : float = 0.0,
        noise_boost_substep           : float = 0.0,
        noise_boost_normalize         : bool  = True,
        noise_anchor                  : float = 1.0,
        
        s_noise                       : float = 1.0,
        s_noise_substep               : float = 1.0,
        d_noise                       : float = 1.0,
        d_noise_start_step            : int   = 0,
        d_noise_inv                   : float                  = 1.0,
        d_noise_inv_start_step        : int                    = 0,
        
        
        
        alpha                         : float              = -1.0,
        alpha_substep                 : float              = -1.0,
        k                             : float              =  1.0,
        k_substep                     : float              =  1.0,
        
        momentum                      : float              =  0.0,


        overshoot_mode                : str                = "hard",
        overshoot_mode_substep        : str                = "hard",
        overshoot                     : float              =  0.0,
        overshoot_substep             : float              =  0.0,

        implicit_type                 : str                = "predictor-corrector",
        implicit_type_substeps        : str                = "predictor-corrector",
        
        implicit_steps_diag           : int                =  0,
        implicit_steps_full           : int                =  0,

        etas                          : Optional[Tensor]   = None,
        etas_substep                  : Optional[Tensor]   = None,
        s_noises                      : Optional[Tensor]   = None,
        s_noises_substep              : Optional[Tensor]   = None,
        
        momentums                     : Optional[Tensor]   = None,

        regional_conditioning_weights : Optional[Tensor]   = None,
        regional_conditioning_floors  : Optional[Tensor]   = None,
                
        LGW_MASK_RESCALE_MIN          : bool               = True,
        guides                        : Optional[Tuple[Any, ...]]    = None,
        epsilon_scales                : Optional[Tensor]   = None,
        frame_weights_grp             : Optional[Tuple[Tensor, Tensor]] = None,

        sde_noise                     : list    [Tensor]   = [],

        noise_seed                    : int                = -1,

        cfgpp                         : float              = 0.0,
        cfg_cw                        : float              = 1.0,

        BONGMATH                      : bool               = True,

        state_info                    : Optional[dict[str, Any]] = None,
        state_info_out                : Optional[dict[str, Any]] = None,
        
        rk_swap_type                  : str                = "",
        rk_swap_step                  : int                = MAX_STEPS,
        rk_swap_threshold             : float              = 0.0,
        rk_swap_print                 : bool               = False,
        
        steps_to_run                  : int                = -1,
        
        batch_num                     : int                = 0,

        extra_options                 : str                = "",
        ):
    
    EO             = ExtraOptions(extra_options)
    default_dtype  = EO("default_dtype", torch.float64)
    
    extra_args     = {} if extra_args     is None else extra_args
    model_device   = x.device
    work_device    = 'cpu' if EO("work_device_cpu") else model_device

    state_info     = {} if state_info     is None else state_info
    state_info_out = {} if state_info_out is None else state_info_out

    if 'raw_x' in state_info and sampler_mode in {"resample", "unsample"}:
        x = state_info['raw_x'].clone()
        RESplain("Continuing from raw latent from previous sampler.", debug=False)
    

    
    start_step = 0
    if 'end_step' in state_info and (sampler_mode == "resample" or sampler_mode == "unsample"):

        if state_info['end_step'] != 0 and state_info['end_step'] != -1 and state_info['end_step'] < len(state_info['sigmas'])-1 :   #incomplete run in previous sampler node
            
            if state_info['sampler_mode'] in {"standard","resample"} and sampler_mode == "unsample":
                sigmas = torch.flip(state_info['sigmas'], dims=[0])
                start_step = (len(sigmas)-1) - (state_info['end_step']-1)
                
            if state_info['sampler_mode'] == "unsample"              and sampler_mode == "resample":
                sigmas = torch.flip(state_info['sigmas'], dims=[0])
                start_step = (len(sigmas)-1) - (state_info['end_step']-1)
        elif state_info['sampler_mode'] == "unsample" and sampler_mode == "resample":
            start_step = 0
        
        if state_info['sampler_mode'] in {"standard", "resample"}    and sampler_mode == "resample":
            start_step = state_info['end_step'] if state_info['end_step'] != -1 else 0
            if start_step > 0:
                sigmas = state_info['sigmas'].clone()
            


    x      = x     .to(dtype=default_dtype, device=work_device)
    sigmas = sigmas.to(dtype=default_dtype, device=work_device)

    c1                          = EO("c1"                         , c1)
    c2                          = EO("c2"                         , c2)
    c3                          = EO("c3"                         , c3)
    
    cfg_cw                      = EO("cfg_cw"                     , cfg_cw)
    
    noise_seed                  = EO("noise_seed"                 , noise_seed)
    noise_seed_substep          = EO("noise_seed_substep"         , noise_seed + MAX_STEPS)
    
    pseudoimplicit_row_weights  = EO("pseudoimplicit_row_weights" , [1. for _ in range(100)])
    pseudoimplicit_step_weights = EO("pseudoimplicit_step_weights", [1. for _ in range(max(implicit_steps_diag, implicit_steps_full)+1)])


    # SETUP SAMPLER
    if implicit_sampler_name not in ("use_explicit", "none"):
        rk_type = implicit_sampler_name
    RESplain("rk_type:", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0
        
    RK            = RK_Method_Beta.create(model, rk_type, noise_anchor, noise_boost_normalize, model_device=model_device, work_device=work_device, dtype=default_dtype, extra_options=extra_options)
    RK.extra_args = RK.init_cfg_channelwise(x, cfg_cw, **extra_args)
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0
    
    # SETUP SIGMAS
    NS               = RK_NoiseSampler(RK, model, device=work_device, dtype=default_dtype, extra_options=extra_options)
    sigmas, UNSAMPLE = NS.prepare_sigmas(sigmas, sigmas_override, d_noise, d_noise_start_step, sampler_mode)
    
    if noise_scaling_mode != "linear" and noise_scaling_type == "model_d":
        #sigmas = sigmas.clone()
        #scaling_cycles = EO("scaling_cycles", 1)
        for _ in range(noise_scaling_cycles):
            for i in range(len(sigmas)-1):
                lying_su, lying_sigma, lying_sd, lying_alpha_ratio = NS.get_sde_step(sigmas[i], sigmas[i+1], noise_scaling_eta, noise_scaling_mode)
                sigmas[i+1] = lying_sd
                
        sigmas = (1+noise_scaling_substep) * sigmas.clone()
        sigmas, UNSAMPLE = NS.prepare_sigmas(sigmas, sigmas_override, d_noise, sampler_mode)
    #elif noise_scaling_mode == "linear" and noise_scaling_type == "model_d":
    #    sigmas = (1-noise_scaling_eta) * sigmas.clone()
    #    sigmas, UNSAMPLE = NS.prepare_sigmas(sigmas, sigmas_override, d_noise, sampler_mode)

    
    SDE_NOISE_EXTERNAL = False
    if sde_noise is not None:
        if len(sde_noise) > 0 and sigmas[1] > sigmas[2]:
            SDE_NOISE_EXTERNAL = True
            sigma_up_total = torch.zeros_like(sigmas[0])
            for i in range(len(sde_noise)-1):
                sigma_up_total += sigmas[i+1]
            etas = torch.full_like(sigmas, eta / sigma_up_total)
    
    if 'last_rng' in state_info and sampler_mode in {"resample", "unsample"} and noise_seed < 0:
        last_rng         = state_info['last_rng'].clone()
        last_rng_substep = state_info['last_rng_substep'].clone()
    else:
        last_rng         = None
        last_rng_substep = None
    
    NS.init_noise_samplers(x, noise_seed, noise_seed_substep, noise_sampler_type, noise_sampler_type_substep, noise_mode_sde, noise_mode_sde_substep, \
                            overshoot_mode, overshoot_mode_substep, noise_boost_step, noise_boost_substep, alpha, alpha_substep, k, k_substep, \
                            last_rng=last_rng, last_rng_substep=last_rng_substep,)

    #if 'last_rng' in state_info and sampler_mode in {"resample", "unsample"} and noise_seed < 0:
    #    NS.noise_sampler.generator.set_state (state_info['last_rng'].clone())
    #    NS.noise_sampler2.generator.set_state(state_info['last_rng_substep'].clone())

    data_               = None
    eps_                = None
    eps                 = torch.zeros_like(x, dtype=default_dtype, device=work_device)
    denoised            = torch.zeros_like(x, dtype=default_dtype, device=work_device)
    denoised_prev       = torch.zeros_like(x, dtype=default_dtype, device=work_device)
    denoised_prev2      = torch.zeros_like(x, dtype=default_dtype, device=work_device)
    x_                  = None
    eps_prev_           = None
    denoised_data_prev  = None
    denoised_data_prev2 = None
    h_prev              = None
    

    # SETUP GUIDES
    LG = LatentGuide(model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options, device=work_device, dtype=default_dtype, frame_weights_grp=frame_weights_grp)
    x = LG.init_guides(x, RK.IMPLICIT, guides, NS.noise_sampler, batch_num)
    if torch.norm(LG.mask - torch.ones_like(LG.mask)) != 0   and   (LG.y0.sum() == 0 or LG.y0_inv.sum() == 0):
        SKIP_PSEUDO = True
        RESplain("skipping pseudo...")
        if   LG.y0    .sum() == 0:
            SKIP_PSEUDO_Y = "y0"
        elif LG.y0_inv.sum() == 0:
            SKIP_PSEUDO_Y = "y0_inv"
    else:
        SKIP_PSEUDO = False
        
    if   LG.y0.sum()     != 0 and LG.y0_inv.sum() != 0:
        denoised_prev = LG.mask * LG.y0 + (1-LG.mask) * LG.y0_inv
    elif LG.y0.sum()     != 0:
        denoised_prev = LG.y0
    elif LG.y0_inv.sum() != 0:
        denoised_prev = LG.y0_inv
        
    if EO("pseudo_mix_strength"):
        orig_y0     = LG.y0.clone()
        orig_y0_inv = LG.y0_inv.clone()

    gc.collect()

    # BEGIN SAMPLING LOOP    
    num_steps = len(sigmas[start_step:])-2 if sigmas[-1] == 0 else len(sigmas[start_step:])-1
    
    #steps_to_run -= 1
    if steps_to_run >= 0:
        current_steps =              min(num_steps, steps_to_run)
        num_steps     = start_step + min(num_steps, steps_to_run)
    else:
        current_steps =              num_steps
        num_steps     = start_step + num_steps
        
    #if sampler_mode == "unsample":
    #    current_steps = (len(sigmas)-1) - current_steps
    
    INIT_SAMPLE_LOOP = True
    step = start_step
    sigma, sigma_next, data_prev_ = None, None, None
    
    if (num_steps-1) == len(sigmas)-2 and sigmas[-1] == 0 and sigmas[-2] == NS.sigma_min:
        progress_bar = trange(current_steps+1, disable=disable)
    else:
        progress_bar = trange(current_steps, disable=disable)
    #progress_bar = trange(len(sigmas)-1, disable=disable)
    
    #pbar_step = len(sigmas)-1 - step if sampler_mode == "unsample" else step
    #progress_bar.n = pbar_step
    #progress_bar.refresh()
    while step < num_steps:
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        if regional_conditioning_weights is not None:
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = regional_conditioning_weights[step]
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = regional_conditioning_floors [step]
        
        epsilon_scale   = float(epsilon_scales[step]) if epsilon_scales   is not None else None
        eta             = etas                [step]  if etas             is not None else eta
        eta_substep     = etas_substep        [step]  if etas_substep     is not None else eta_substep
        s_noise         = s_noises            [step]  if s_noises         is not None else s_noise
        s_noise_substep = s_noises_substep    [step]  if s_noises_substep is not None else s_noise_substep
        
        NS.set_sde_step(sigma, sigma_next, eta, overshoot, s_noise)
        RK.set_coeff(rk_type, NS.h, c1, c2, c3, step, sigmas, NS.sigma_down)
        NS.set_substep_list(RK)

        if (noise_scaling_eta > 0 or noise_scaling_substep != 0) and noise_scaling_type != "model_d":
            lying_s_ = NS.s_
            #if noise_scaling_type == "sampler":
            if noise_scaling_mode == "linear":
                lying_s_ *= (1-noise_scaling_eta)
            else:
                lying_su, lying_sigma, lying_sd, lying_alpha_ratio = NS.get_sde_step(sigma, NS.sigma_down, noise_scaling_eta, noise_scaling_mode)
                for _ in range(noise_scaling_cycles-1):
                    lying_su, lying_sigma, lying_sd, lying_alpha_ratio = NS.get_sde_step(sigma, lying_sd, noise_scaling_eta, noise_scaling_mode)
                
                lying_s_ = NS.get_substep_list(RK, sigma, RK.h_fn(lying_sd, lying_sigma))
            if noise_scaling_type == "model":
                lying_s_ = lying_s_ * (1+noise_scaling_substep)
                NS.s_ = lying_s_
        #if noise_scaling_type == "sampler_substep":
        #    sub_lying_su, sub_lying_sigma, sub_lying_sd, sub_lying_alpha_ratio = NS.get_sde_substep(NS.s_[row], NS.s_[row+RK.row_offset+RK.multistep_stages], noise_scaling_eta, noise_scaling_mode)
        #    #lying_s_[row]   = sub_lying_sigma
        #    lying_s_[row+1] = sub_lying_sd

        rk_swap_stages = 3 if rk_swap_type != "" else 0
        recycled_stages = max(rk_swap_stages, RK.multistep_stages, RK.hybrid_stages)
        
        if INIT_SAMPLE_LOOP:
            INIT_SAMPLE_LOOP = False
            x_, data_, eps_, eps_prev_  = (torch.zeros(    RK.rows+2,     *x.shape, dtype=default_dtype, device=work_device) for _ in range(4))
            
            data_prev_ = state_info.get('data_prev_')
            if data_prev_ is not None:
                data_prev_ = state_info['data_prev_'].clone()
                #data_prev_ = state_info['data_prev_'][len(eps_):].clone()
            else:
                #data_prev_ =  torch.zeros(max(RK.rows+2, 4), *x.shape, dtype=default_dtype, device=work_device)
                data_prev_ =  torch.zeros(4, *x.shape, dtype=default_dtype, device=work_device) # multistep max is 4m... so 4 needed
            
            recycled_stages = len(data_prev_)-1
            
        if RK.rows+2 > x_.shape[0]:
            row_gap = RK.rows+2 - x_.shape[0]
            x_gap_, data_gap_, eps_gap_, eps_prev_gap_  = (torch.zeros(row_gap,     *x.shape, dtype=default_dtype, device=work_device) for _ in range(4))
            x_        = torch.cat((x_       ,x_gap_)       , dim=0)
            data_     = torch.cat((data_    ,data_gap_)    , dim=0)
            eps_      = torch.cat((eps_     ,eps_gap_)     , dim=0)
            eps_prev_ = torch.cat((eps_prev_,eps_prev_gap_), dim=0)

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
        if RK.multistep_stages > 0 or RK.hybrid_stages > 0:
            for ms in range(min(len(data_prev_), len(eps_))):
                eps_[ms] = RK.get_epsilon_anchored(x_0, data_prev_[ms], sigma)
            eps_prev_ = eps_.clone()

        # INITIALIZE IMPLICIT SAMPLING
        if RK.IMPLICIT:
            x_, eps_, data_ = init_implicit_sampling(RK, x_0, x_, eps_, eps_prev_, data_, eps, denoised, denoised_prev2, step, sigmas, NS.h, NS.s_, EO)

        implicit_steps_total = (implicit_steps_full + 1) * (implicit_steps_diag + 1)

        # BEGIN FULLY IMPLICIT LOOP
        for full_iter in range(implicit_steps_full + 1):

            if RK.IMPLICIT:
                x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, 0, NS.h, sigmas, step, "init")

            # PREPARE FULLY PSEUDOIMPLICIT GUIDES
            if step > 0 or not SKIP_PSEUDO:
                if full_iter > 0 and EO("fully_implicit_reupdate_x"):
                    x_[0] = NS.sigma_from_to(x_0, x, sigma, sigma_next, NS.s_[0])
                    x_0   = NS.sigma_from_to(x_0, x, sigma, sigma_next, sigma)
                
                if EO("fully_pseudo_init") and full_iter == 0:
                    guide_mode_tmp = LG.guide_mode
                    LG.guide_mode = "fully_" + LG.guide_mode
                x_0, x_, eps_ = LG.prepare_fully_pseudoimplicit_guides_substep(x_0, x_, eps_, eps_prev_, data_, denoised_prev, 0, step, sigmas, eta_substep, overshoot_substep, s_noise_substep, \
                                                                                NS, RK, pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, BONGMATH)
                if EO("fully_pseudo_init") and full_iter == 0:
                    LG.guide_mode = guide_mode_tmp

            # TABLEAU LOOP
            for row in range(RK.rows - RK.multistep_stages - RK.row_offset + 1):
                for diag_iter in range(implicit_steps_diag+1):
                    
                    if noise_sampler_type_substep == "brownian" and (full_iter > 0 or diag_iter > 0):
                        eta_substep = 0.
                    
                    NS.set_sde_substep(row, RK.multistep_stages, eta_substep, overshoot_substep, s_noise_substep, full_iter, diag_iter, implicit_steps_full, implicit_steps_diag)

                    # PRENOISE METHOD HERE!
                    
                    # A-TABLEAU
                    if row < RK.rows:

                        # PREPARE PSEUDOIMPLICIT GUIDES
                        if step > 0 or not SKIP_PSEUDO:
                            x_0, x_, eps_, x_row_pseudoimplicit, sub_sigma_pseudoimplicit = LG.process_pseudoimplicit_guides_substep(x_0, x_, eps_, eps_prev_, data_, denoised_prev, row, step, sigmas, NS, RK, \
                                                                                                                        pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, BONGMATH)
                        

                        # PREPARE MODEL CALL
                        if LG.guide_mode in GUIDE_MODE_NAMES_PSEUDOIMPLICIT and (step > 0 or not SKIP_PSEUDO) and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0) and x_row_pseudoimplicit is not None:

                            x_tmp =     x_row_pseudoimplicit 
                            s_tmp = sub_sigma_pseudoimplicit 

                        # Fully implicit iteration (explicit only)                   # or... Fully implicit iteration (implicit only... not standard) 
                        elif (full_iter > 0 and RK.row_offset == 1 and row == 0)   or   (full_iter > 0 and RK.row_offset == 0 and row == 0 and EO("fully_implicit_update_x")):
                            if EO("fully_explicit_pogostick_eta"): 
                                super_alpha_ratio, super_sigma_down, super_sigma_up = NS.get_sde_coeff(sigma, sigma_next, None, eta)
                                x = super_alpha_ratio * x + super_sigma_up * NS.noise_sampler(sigma=sigma_next, sigma_next=sigma)
                                
                                x_tmp = x
                                s_tmp = sigma
                            elif EO("enable_fully_explicit_lagrange_rebound1"):
                                substeps_prev = len(RK.C[:-1]) 
                                x_tmp = lagrange_interpolation(RK.C[1:-1], x_[1:substeps_prev], RK.C[0]).squeeze(0)
                                
                            elif EO("enable_fully_explicit_lagrange_rebound2"):
                                substeps_prev = len(RK.C[:-1]) 
                                x_tmp = lagrange_interpolation(RK.C[1:], x_[1:substeps_prev+1], RK.C[0]).squeeze(0)

                            elif EO("enable_fully_explicit_rebound1"):  # 17630, faded dots, just crap
                                eps_tmp, denoised_tmp = RK(x, sigma_next, x, sigma_next)
                                eps_tmp = (x - denoised_tmp) / sigma_next
                                x_[0] = denoised_tmp + sigma * eps_tmp
                                
                                x_0 =   x_[0]
                                x_tmp = x_[0]
                                s_tmp = sigma
                                
                            elif implicit_type == "rebound": 
                                eps_tmp, denoised_tmp = RK(x, sigma_next, x_0, sigma)
                                eps_tmp = (x - denoised_tmp) / sigma_next
                                x = denoised_tmp + sigma * eps_tmp
                                
                                x_tmp = x
                                s_tmp = sigma
                                
                            elif implicit_type == "retro-eta" and (NS.sub_sigma_up > 0 or NS.sub_sigma_up_eta > 0): 
                                x_tmp = NS.sigma_from_to(x_0, x, sigma, sigma_next, sigma)
                                s_tmp = sigma
                                
                            elif implicit_type == "bongmath" and (NS.sub_sigma_up > 0 or NS.sub_sigma_up_eta > 0): 
                                if BONGMATH:
                                    x_tmp =    x_[row]
                                    s_tmp = NS.s_[row]
                                else:
                                    x_tmp = NS.sigma_from_to(x_0, x, sigma, sigma_next, sigma)
                                    s_tmp = sigma
                                
                            else:
                                x_tmp = x
                                s_tmp = sigma_next
                        
                        # All others
                        else:
                            # three potential toggle options: force rebound/model call, force PC style, force pogostick style
                            if diag_iter > 0: # Diagonally implicit iteration (explicit or implicit)
                                if EO("diag_explicit_pogostick_eta"): 
                                    super_alpha_ratio, super_sigma_down, super_sigma_up = NS.get_sde_coeff(NS.s_[row], NS.s_[row+RK.row_offset+RK.multistep_stages], None, eta)
                                    x_[row+RK.row_offset] = super_alpha_ratio * x_[row+RK.row_offset] + super_sigma_up * NS.noise_sampler(sigma=NS.s_[row+RK.row_offset+RK.multistep_stages], sigma_next=NS.s_[row])
                                    
                                    x_tmp = x_[row+RK.row_offset]
                                    s_tmp = sigma
                                
                                elif implicit_type_substeps == "rebound":
                                    eps_[row], data_[row] = RK(x_[row+RK.row_offset], NS.s_[row+RK.row_offset+RK.multistep_stages], x_0, sigma)
                                    
                                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, RK.row_offset, NS.h_new, NS.h_new_orig)
                                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])

                                    x_[row+RK.row_offset] = NS.sigma_from_to(x_0,    x_[row+RK.row_offset],    sigma,    NS.s_[row+RK.row_offset+RK.multistep_stages],    NS.s_[row])
                                    x_tmp = x_[row+RK.row_offset]
                                    s_tmp = NS.s_[row]
                                    
                                elif implicit_type_substeps == "retro-eta" and (NS.sub_sigma_up > 0 or NS.sub_sigma_up_eta > 0):
                                    x_tmp = NS.sigma_from_to(x_0,   x_[row+RK.row_offset],    sigma,    NS.s_[row+RK.row_offset+RK.multistep_stages],    NS.s_[row])
                                    s_tmp = NS.s_[row]
                                    
                                elif implicit_type_substeps == "bongmath" and (NS.sub_sigma_up > 0 or NS.sub_sigma_up_eta > 0) and not EO("disable_diag_explicit_bongmath_rebound"): 
                                    if BONGMATH:
                                        x_tmp = x_[row]
                                        s_tmp = NS.s_[row]
                                    else:
                                        x_tmp = NS.sigma_from_to(x_0, x_[row+RK.row_offset], sigma, NS.s_[row+RK.row_offset+RK.multistep_stages], NS.s_[row])
                                        s_tmp = NS.s_[row]
                                    
                                else:
                                    x_tmp =    x_[row+RK.row_offset]
                                    s_tmp = NS.s_[row+RK.row_offset+RK.multistep_stages]
                            else:
                                x_tmp = x_[row]
                                s_tmp = NS.sub_sigma 

                            if RK.IMPLICIT: 
                                if not EO("disable_implicit_guide_preproc"):
                                    eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                                    eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                                if row == 0 and (EO("implicit_lagrange_init")  or   EO("radaucycle")):
                                    pass
                                else:
                                    x_[row+RK.row_offset] = x_0 + NS.h_new * RK.zum(row+RK.row_offset, eps_, eps_prev_)
                                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                                    if row > 0:
                                        x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset])
                                        if BONGMATH and step < sigmas.shape[0]-1 and not EO("disable_implicit_prebong"):
                                            x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step)     # TRY WITH h_new ??
                                    x_tmp = x_[row+RK.row_offset]



                        # MODEL CALL
                        if RK.IMPLICIT   and   row == 0   and   (EO("implicit_lazy_recycle_first_model_call_at_start")   or   EO("radaucycle")  or RK.C[0] == 0.0):
                            pass
                        else: 
                            if s_tmp == 0:
                                break
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "pre") # will this do anything? not x_tmp

                            eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                            
                            #data_[row] = data_[row] - EO("momentum", 0.0) * (data_prev_[0] - data_[row])  #negative!
                            data_[row] = data_[row] - momentum * (data_prev_[0] - data_[row])  #negative!

                            eps_[row]  = RK.get_epsilon(x_0, x_tmp, data_[row], sigma, s_tmp)

                            if row < RK.rows and noise_scaling_substep != 0 and noise_scaling_type in {"sampler", "sampler_substep"}:
                                if noise_scaling_type == "sampler_substep":
                                    if noise_scaling_mode == "linear":
                                        lying_s_[row+1] = NS.s_[row+1] * (1-noise_scaling_eta)
                                    else:
                                        sub_lying_su, sub_lying_sigma, sub_lying_sd, sub_lying_alpha_ratio = NS.get_sde_substep(NS.s_[row], NS.s_[row+RK.row_offset+RK.multistep_stages], noise_scaling_eta, noise_scaling_mode)
                                        for _ in range(noise_scaling_cycles-1):
                                            sub_lying_su, sub_lying_sigma, sub_lying_sd, sub_lying_alpha_ratio = NS.get_sde_substep(NS.s_[row], sub_lying_sd, noise_scaling_eta, noise_scaling_mode)
                                        lying_s_[row+1] = sub_lying_sd
                                substep_noise_scaling_ratio = NS.s_[row+1]/lying_s_[row+1]
                                eps_[row] *= 1 + noise_scaling_substep*(substep_noise_scaling_ratio-1)

                            """if eta_substep > 0 and row < RK.rows: # and ((row < rk.rows - rk.multistep_stages - 1))   and   (sub_sigma_down > 0) and sigma_next > 0:
                                
                                substep_noise_scaling_ratio = NS.s_[row+1]/NS.sub_sigma_down_eta
                                eps_[row] *= 1 + noise_scaling_substep*(substep_noise_scaling_ratio-1)
                                
                            if eta_substep > 0 and row < RK.rows and EO("lying_substep"):
                                substep_noise_scaling_ratio = lying_s_[row+1]/NS.sub_sigma_down
                                eps_[row] *= 1 + EO("lying_substep",0.0)*(substep_noise_scaling_ratio-1)
                                
                            
                            if overshoot_substep > 0 and row < RK.rows:    
                                substep_noise_scaling_ratio = NS.s_[row+1]/NS.sub_sigma_down
                                eps_[row] *= 1 + EO("substep_overshoot_scaling",1.0)*(substep_noise_scaling_ratio-1)"""
                                
                            
                            if eta_substep > 0 and row < RK.rows and EO("substep_noise_scaling"):    
                                substep_noise_scaling_ratio = NS.s_[row+1]/NS.sub_sigma_down_eta
                                eps_[row] *= 1 + EO("substep_noise_scaling",0.0)*(substep_noise_scaling_ratio-1)

                        if EO("bong2m") and RK.multistep_stages > 0 and step < len(sigmas)-4:
                            h_no_eta       = -torch.log(sigmas[step+1]/sigmas[step])
                            h_prev1_no_eta = -torch.log(sigmas[step]  /sigmas[step-1])
                            c2_prev = (-h_prev1_no_eta / h_no_eta).item()
                            eps_prev = denoised_data_prev - x_0
                            
                            φ = Phi(h_prev, [0.,c2_prev])
                            a2_1 = c2_prev * φ(1,2)
                            for i in range(100):
                                x_prev = x_0 - h_prev * (a2_1 * eps_prev)
                                eps_prev = denoised_data_prev - x_prev
                                
                            eps_[1] = eps_prev
                            
                        if EO("bong3m") and RK.multistep_stages > 0 and step < len(sigmas)-10:
                            h_no_eta       = -torch.log(sigmas[step+1]/sigmas[step])
                            h_prev1_no_eta = -torch.log(sigmas[step]  /sigmas[step-1])
                            h_prev2_no_eta = -torch.log(sigmas[step]  /sigmas[step-2])
                            c2_prev        = (-h_prev1_no_eta / h_no_eta).item()
                            c3_prev        = (-h_prev2_no_eta / h_no_eta).item()      
                            
                            eps_prev2 = denoised_data_prev2 - x_0
                            eps_prev  = denoised_data_prev  - x_0
                            
                            φ = Phi(h_prev1_no_eta, [0.,c2_prev, c3_prev])
                            a2_1 = c2_prev * φ(1,2)
                            for i in range(100):
                                x_prev = x_0 - h_prev1_no_eta * (a2_1 * eps_prev)
                                eps_prev = denoised_data_prev2 - x_prev
                                
                            eps_[1] = eps_prev
                            
                            φ = Phi(h_prev2_no_eta, [0.,c3_prev, c3_prev])
                            
                            def calculate_gamma(c2_prev, c3_prev):
                                return (3*(c3_prev**3) - 2*c3_prev) / (c2_prev*(2 - 3*c2_prev))
                            gamma = calculate_gamma(c2_prev, c3_prev)
                            
                            a2_1 = c2_prev * φ(1,2)
                            a3_2 = gamma * c2_prev * φ(2,2) + (c3_prev ** 2 / c2_prev) * φ(2, 3)
                            a3_1 = c3_prev * φ(1,3) - a3_2
                            
                            for i in range(100):
                                x_prev2 = x_0     - h_prev2_no_eta * (a3_1 * eps_prev + a3_2 * eps_prev2)
                                x_prev  = x_prev2 + h_prev2_no_eta * (a2_1 * eps_prev)
                                
                                eps_prev2 = denoised_data_prev - x_prev2
                                eps_prev  = denoised_data_prev2 - x_prev
                                
                            eps_[2] = eps_prev2



                        # GUIDE 
                        #if not UNSAMPLE:
                        if not EO("disable_guides_eps_substep"):
                            eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, NS.sigma, NS.sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                        if not EO("disable_guides_eps_prev_substep"):
                            eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, NS.sigma, NS.sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)

                        if (full_iter == 0 and diag_iter == 0)   or   EO("newton_iter_post_use_on_implicit_steps"):
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "post")



                    # UPDATE
                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, RK.row_offset, NS.h_new, NS.h_new_orig)

                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                    
                    if not RK.IMPLICIT and NS.noise_mode_sde_substep != "hard_sq":
                        x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset])

                    if BONGMATH and NS.s_[row] > RK.sigma_min and NS.h < RK.sigma_max/2   and   (diag_iter == implicit_steps_diag or EO("enable_diag_explicit_bongmath_all"))   and not EO("disable_terminal_bongmath"):
                        if step == 0 and UNSAMPLE:
                            pass
                        elif full_iter == implicit_steps_full or not EO("disable_fully_explicit_bongmath_except_final"):
                            x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step)
                            
                    #progress_bar.update( round(1 / implicit_steps_total, 2) )
                    
                    #step_update = round(1 / implicit_steps_total, 2)
                    #progress_bar.update(float(f"{step_update:.2f}")) 

            x_next = x_[RK.rows - RK.multistep_stages - RK.row_offset + 1]
            x_next = NS.rebound_overshoot_step(x_0, x_next)
            
            eps = (x_0 - x_next) / (sigma - sigma_next)
            denoised = x_0 - sigma * eps
            
            x_next = LG.process_guides_poststep(x_next, denoised, eps, step)
            x      = NS.swap_noise_step(x_0, x_next)
            
            callback_step = len(sigmas)-1 - step if sampler_mode == "unsample" else step
            preview_callback(x, eps, denoised, x_, eps_, data_, callback_step, sigma, sigma_next, callback, EO)
            
            h_prev = NS.h
            x_prev = x_0
            
            denoised_prev2 = denoised_prev
            denoised_prev  = denoised



        data_prev_[0] = data_[0]
        for ms in range(recycled_stages):
            data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
        
        rk_type = RK.swap_rk_type_at_step_or_threshold(x_0, data_prev_, NS, sigmas, step, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print)
        if step > rk_swap_step:
            implicit_steps_full = 0
            implicit_steps_diag = 0

        
        denoised_data_prev2 = denoised_data_prev
        denoised_data_prev = data_[0]
        
        if SKIP_PSEUDO:
            if SKIP_PSEUDO_Y == "y0":
                LG.y0 = denoised
                LG.HAS_LATENT_GUIDE_INV = True
            else:
                LG.y0_inv = denoised
                LG.HAS_LATENT_GUIDE_INV = True
                
        if EO("pseudo_mix_strength"):
            pseudo_mix_strength = EO("pseudo_mix_strength", 0.0)
            LG.y0     = orig_y0     + pseudo_mix_strength * (denoised - orig_y0)
            LG.y0_inv = orig_y0_inv + pseudo_mix_strength * (denoised - orig_y0_inv)
            
        #if sampler_mode == "unsample":
        #    progress_bar.n -= 1 
        #    progress_bar.refresh() 
        #else:
        #    progress_bar.update(1)
        progress_bar.update(1)  #THIS WAS HERE
        step += 1
        
        if EO("skip_step", -1) == step:
            step += 1

        if d_noise_start_step     == step:
            sigmas = sigmas.clone() * d_noise
        if d_noise_inv_start_step == step:
            sigmas = sigmas.clone() / d_noise_inv
        # end sampling loop

    #progress_bar.close()

    if step == len(sigmas)-2 and sigmas[-1] == 0 and sigmas[-2] == NS.sigma_min:
        eps, denoised = RK(x, NS.sigma_min, x, NS.sigma_min)
        x = denoised
        #progress_bar.update(1)

    eps      = eps     .to(model_device)
    denoised = denoised.to(model_device)
    x        = x       .to(model_device)
    
    progress_bar.close()

    #if not (UNSAMPLE and sigmas[1] > sigmas[0]) and not EO("preview_last_step_always"):
    if not (UNSAMPLE and sigmas[1] > sigmas[0]) and not EO("preview_last_step_always") and sigma is not None:
        callback_step = len(sigmas)-1 - step if sampler_mode == "unsample" else step
        preview_callback(x, eps, denoised, x_, eps_, data_, callback_step, sigma, sigma_next, callback, EO, FINAL_STEP=True)

    state_info_out['raw_x']             = x#.clone()
    state_info_out['data_prev_']        = data_prev_#.clone()
    state_info_out['end_step']          = step
    state_info_out['sigmas']            = sigmas.clone()
    state_info_out['sampler_mode']      = sampler_mode
    state_info_out['last_rng']          = NS.noise_sampler .generator.get_state().clone()
    state_info_out['last_rng_substep']  = NS.noise_sampler2.generator.get_state().clone()
    
    gc.collect()

    return x



def preview_callback(
                    x          : Tensor,
                    eps        : Tensor,
                    denoised   : Tensor,
                    x_         : Tensor,
                    eps_       : Tensor,
                    data_      : Tensor,
                    step       : int,
                    sigma      : Tensor,
                    sigma_next : Tensor,
                    callback   : Callable,
                    EO         : ExtraOptions,
                    FINAL_STEP : bool = False):
    
    if FINAL_STEP:
        denoised_callback = denoised
        
    elif EO("eps_substep_preview"):
        row_callback = EO("eps_substep_preview", 0)
        denoised_callback = eps_[row_callback]
        
    elif EO("denoised_substep_preview"):
        row_callback = EO("denoised_substep_preview", 0)
        denoised_callback = data_[row_callback]
        
    elif EO("x_substep_preview"):
        row_callback = EO("x_substep_preview", 0)
        denoised_callback = x_[row_callback]
        
    elif EO("eps_preview"):
        denoised_callback = eps
        
    elif EO("denoised_preview"):
        denoised_callback = denoised
        
    elif EO("x_preview"):
        denoised_callback = x
        
    else:
        denoised_callback = data_[0]
        
    callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_callback.to(torch.float32)}) if callback is not None else None
    
    return





