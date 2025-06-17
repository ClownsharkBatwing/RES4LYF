import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm.auto import trange
import gc
from typing import Optional, Callable, Tuple, List, Dict, Any, Union
import math
import copy

from comfy.model_sampling import EPS
import comfy

from ..res4lyf              import RESplain
from ..helper               import ExtraOptions, FrameWeightsManager
from ..latents              import lagrange_interpolation, get_collinear, get_orthogonal, get_cosine_similarity, get_pearson_similarity, get_slerp_weight_for_cossim, get_slerp_ratio, slerp_tensor, get_edge_mask, normalize_zscore, compute_slerp_ratio_for_target, find_slerp_ratio_grid

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
        SYNC_GUIDE_ACTIVE,
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

    x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, s_, 0, h, sigmas, step, "init", SYNC_GUIDE_ACTIVE)
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
        implicit_sampler_name         : str                = "use_explicit",

        c1                            : float              =  0.0,
        c2                            : float              =  0.5,
        c3                            : float              =  1.0,

        noise_sampler_type            : str                = "gaussian",
        noise_sampler_type_substep    : str                = "gaussian",
        noise_mode_sde                : str                = "hard",
        noise_mode_sde_substep        : str                = "hard",

        eta                           : float              =  0.5,
        eta_substep                   : float              =  0.5,




        noise_scaling_weight          : float              = 0.0,
        noise_scaling_type            : str                = "sampler",
        noise_scaling_mode            : str                = "linear",
        noise_scaling_eta             : float              = 0.0,
        noise_scaling_cycles          : int                = 1,
        
        noise_scaling_weights         : Optional[Tensor]   = None,
        noise_scaling_etas            : Optional[Tensor]   = None,
        
        noise_boost_step              : float              = 0.0,
        noise_boost_substep           : float              = 0.0,
        noise_boost_normalize         : bool               = True,
        noise_anchor                  : float              = 1.0,
        
        s_noise                       : float              = 1.0,
        s_noise_substep               : float              = 1.0,
        d_noise                       : float              = 1.0,
        d_noise_start_step            : int                = 0,
        d_noise_inv                   : float              = 1.0,
        d_noise_inv_start_step        : int                = 0,
        
        
        
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
        narcissism_start_step         : int                = 0,
        narcissism_end_step           : int                = 5,
                
        LGW_MASK_RESCALE_MIN          : bool                          = True,
        guides                        : Optional[Tuple[Any, ...]]     = None,
        epsilon_scales                : Optional[Tensor]              = None,
        frame_weights_mgr             : Optional[FrameWeightsManager] = None,

        sde_noise                     : list    [Tensor]   = [],

        noise_seed                    : int                = -1,

        cfgpp                         : float              = 0.0,
        cfg_cw                        : float              = 1.0,

        BONGMATH                      : bool               = True,
        unsample_bongmath             = None,

        state_info                    : Optional[dict[str, Any]] = None,
        state_info_out                : Optional[dict[str, Any]] = None,
        
        rk_swap_type                  : str                = "",
        rk_swap_step                  : int                = MAX_STEPS,
        rk_swap_threshold             : float              = 0.0,
        rk_swap_print                 : bool               = False,
        
        steps_to_run                  : int                = -1,
        start_at_step                 : int                = -1,
        tile_sizes                    : Optional[List[Tuple[int,int]]] = None,
        
        flow_sync_eps                 : float              = 0.0,
        
        sde_mask                      : Optional[Tensor]   = None,
        
        batch_num                     : int                = 0,
        
        extra_options                 : str                = "",
        
        AttnMask   = None,
        RegContext = None,
        RegParam   = None,
        
        AttnMask_neg   = None,
        RegContext_neg = None,
        RegParam_neg   = None,
        ):
    
    if sampler_mode == "NULL":
        return x
    
    EO             = ExtraOptions(extra_options)
    default_dtype  = EO("default_dtype", torch.float64)
    
    extra_args     = {} if extra_args     is None else extra_args
    model_device   = model.inner_model.inner_model.device #x.device
    work_device    = 'cpu' if EO("work_device_cpu") else model_device

    state_info     = {} if state_info     is None else state_info
    state_info_out = {} if state_info_out is None else state_info_out
    
    VE_MODEL = isinstance(model.inner_model.inner_model.model_sampling, EPS)
    
    RENOISE = False
    if 'raw_x' in state_info and sampler_mode in {"resample", "unsample"}:
        if x.shape == state_info['raw_x'].shape:
            x = state_info['raw_x'].to(work_device) #clone()
        else:
            denoised = comfy.utils.bislerp(state_info['denoised'], x.shape[-1], x.shape[-2])
            x = denoised.to(x)
            RENOISE = True
        RESplain("Continuing from raw latent from previous sampler.", debug=False)
    
    
    start_step = 0
    if 'end_step' in state_info and (sampler_mode == "resample" or sampler_mode == "unsample"):

        if state_info['completed'] != True and state_info['end_step'] != 0 and state_info['end_step'] != -1 and state_info['end_step'] < len(state_info['sigmas'])-1 :   #incomplete run in previous sampler node
            
            if state_info['sampler_mode'] in {"standard","resample"} and sampler_mode == "unsample" and sigmas[2] < sigmas[1]:
                sigmas = torch.flip(state_info['sigmas'], dims=[0])
                start_step = (len(sigmas)-1) - (state_info['end_step']) #-1) #removed -1 at the end here. correct?
                
            if state_info['sampler_mode'] == "unsample"              and sampler_mode == "resample" and sigmas[2] > sigmas[1]:
                sigmas = torch.flip(state_info['sigmas'], dims=[0])
                start_step = (len(sigmas)-1) - state_info['end_step'] #-1)
        elif state_info['sampler_mode'] == "unsample" and sampler_mode == "resample":
            start_step = 0
        
        if state_info['sampler_mode'] in {"standard", "resample"}    and sampler_mode == "resample":
            start_step = state_info['end_step'] if state_info['end_step'] != -1 else 0
            if start_step > 0:
                sigmas = state_info['sigmas'].clone()
            
    if sde_mask is not None:
        from .rk_guide_func_beta import prepare_mask
        sde_mask, _ = prepare_mask(x, sde_mask, LGW_MASK_RESCALE_MIN)
        sde_mask = sde_mask.to(x.device).to(x.dtype)
    

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

    noise_scaling_cycles = EO("noise_scaling_cycles", 1)
    noise_boost_step     = EO("noise_boost_step",     0.0)
    noise_boost_substep  = EO("noise_boost_substep",  0.0)
    
    # SETUP SAMPLER
    if implicit_sampler_name not in ("use_explicit", "none"):
        rk_type = implicit_sampler_name
    RESplain("rk_type:", rk_type)
    if implicit_sampler_name == "none":
        implicit_steps_diag = implicit_steps_full = 0
        
    RK            = RK_Method_Beta.create(model, rk_type, VE_MODEL, noise_anchor, noise_boost_normalize, model_device=model_device, work_device=work_device, dtype=default_dtype, extra_options=extra_options)
    RK.extra_args = RK.init_cfg_channelwise(x, cfg_cw, **extra_args)
    RK.tile_sizes = tile_sizes
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0
    
    RK.unsample_bongmath = BONGMATH if unsample_bongmath is None else unsample_bongmath # allow turning off bongmath for unsampling with cycles


    # SETUP SIGMAS
    sigmas_orig = sigmas.clone()
    NS               = RK_NoiseSampler(RK, model, device=work_device, dtype=default_dtype, extra_options=extra_options)
    sigmas, UNSAMPLE = NS.prepare_sigmas(sigmas, sigmas_override, d_noise, d_noise_start_step, sampler_mode)
    if UNSAMPLE and sigmas_orig[0] == 0.0 and sigmas_orig[0] != sigmas[0] and sigmas[1] < sigmas[2]:
        sigmas = torch.cat([torch.full_like(sigmas[0], 0.0).unsqueeze(0), sigmas])
        if start_step == 0:
            start_step  = 1
        else:
            start_step -= 1
    
    if sampler_mode in {"resample", "unsample"}:
        state_info_sigma_next = state_info.get('sigma_next', -1)
        state_info_start_step = (sigmas == state_info_sigma_next).nonzero().flatten()
        if state_info_start_step.shape[0] > 0:
            start_step = state_info_start_step.item()
            
    start_step = start_at_step if start_at_step >= 0 else start_step

    
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
    eps_y2x_            = None
    eps_x2y_            = None
    eps_y_              = None
    eps_prev_y_         = None
    data_y_             = None
    yt_                 = None
    yt_0                = None
    eps_yt_             = None
    eps_x_              = None 
    data_y_             = None
    data_x_             = None
    
    y0_bongflow         = state_info.get('y0_bongflow')
    y0_bongflow_orig    = state_info.get('y0_bongflow_orig')
    noise_bongflow      = state_info.get('noise_bongflow')
    y0_standard_guide   = state_info.get('y0_standard_guide')
    y0_inv_standard_guide = state_info.get('y0_inv_standard_guide')
    
    data_prev_y_        = state_info.get('data_prev_y_')
    data_prev_x_        = state_info.get('data_prev_x_')
    data_prev_x2y_      = state_info.get('data_prev_x2y_')

    # BEGIN SAMPLING LOOP    
    num_steps = len(sigmas[start_step:])-2 if sigmas[-1] == 0 else len(sigmas[start_step:])-1
    
    if steps_to_run >= 0:
        current_steps =              min(num_steps, steps_to_run)
        num_steps     = start_step + min(num_steps, steps_to_run)
    else:
        current_steps =              num_steps
        num_steps     = start_step + num_steps
    #current_steps = current_steps + 1 if sigmas[-1] == 0 and steps_to_run < 0 and UNSAMPLE else current_steps
    
    INIT_SAMPLE_LOOP = True
    step = start_step
    sigma, sigma_next, data_prev_ = None, None, None
    
    if (num_steps-1) == len(sigmas)-2 and sigmas[-1] == 0 and sigmas[-2] == NS.sigma_min:
        progress_bar = trange(current_steps+1, disable=disable)
    else:
        progress_bar = trange(current_steps, disable=disable)
    

    # SETUP GUIDES
    LG = LatentGuide(model, sigmas, UNSAMPLE, VE_MODEL, LGW_MASK_RESCALE_MIN, extra_options, device=work_device, dtype=default_dtype, frame_weights_mgr=frame_weights_mgr)

    guide_inversion_y0     = state_info.get('guide_inversion_y0')
    guide_inversion_y0_inv = state_info.get('guide_inversion_y0_inv')

    x = LG.init_guides(x, RK.IMPLICIT, guides, NS.noise_sampler, batch_num, sigmas[step], guide_inversion_y0, guide_inversion_y0_inv)
    LG.y0     = y0_standard_guide     if y0_standard_guide     is not None else LG.y0
    LG.y0_inv = y0_inv_standard_guide if y0_inv_standard_guide is not None else LG.y0_inv
    if (LG.mask != 1.0).any()   and  ((LG.y0 == 0).all() or (LG.y0_inv == 0).all()) : #  and   not LG.guide_mode.startswith("flow"):  # (LG.y0.sum() == 0 or LG.y0_inv.sum() == 0):
        SKIP_PSEUDO = True
        RESplain("skipping pseudo...")
        if   LG.y0    .sum() == 0:
            SKIP_PSEUDO_Y = "y0"
        elif LG.y0_inv.sum() == 0:
            SKIP_PSEUDO_Y = "y0_inv"
    else:
        SKIP_PSEUDO = False
    if guides is not None and guides.get('guide_mode', '') != "inversion" or sampler_mode != "unsample":  #do not set denoised_prev to noise guide with inversion!
        if   LG.y0.sum()     != 0 and LG.y0_inv.sum() != 0:
            denoised_prev = LG.mask * LG.y0 + (1-LG.mask) * LG.y0_inv         
        elif LG.y0.sum()     != 0:
            denoised_prev = LG.y0
        elif LG.y0_inv.sum() != 0:
            denoised_prev = LG.y0_inv
    data_cached = None
        
    if EO("pseudo_mix_strength"):
        orig_y0     = LG.y0.clone()
        orig_y0_inv = LG.y0_inv.clone()
    
    #gc.collect()
    BASE_STARTED = False
    INV_STARTED  = False
    FLOW_STARTED = False
    FLOW_STOPPED = False
    noise_xt, noise_yt = None, None
    FLOW_RESUMED = False
    if state_info.get('FLOW_STARTED', False) and not state_info.get('FLOW_STOPPED', False):
        FLOW_RESUMED = True
        y0 = state_info['y0'].to(work_device) 
        data_cached = state_info['data_cached'].to(work_device) 
        data_x_prev_ = state_info['data_x_prev_'].to(work_device) 
    if EO("flow_use_init_noise") or EO("flow_use_smart_noise"):
        x_init = x.clone()

    #progress_bar = trange(len(sigmas)-1-start_step, disable=disable)
        
    if AttnMask is not None:
        RK.update_transformer_options({'AttnMask'  : AttnMask})
        RK.update_transformer_options({'RegContext': RegContext})

    if AttnMask_neg is not None:
        RK.update_transformer_options({'AttnMask_neg'  : AttnMask_neg})
        RK.update_transformer_options({'RegContext_neg': RegContext_neg})
        
    if EO("y0_to_transformer_options"):
        RK.update_transformer_options({'y0':  LG.y0.clone()})
    
    if EO("y0_inv_to_transformer_options"):
        RK.update_transformer_options({'y0_inv':  LG.y0_inv.clone()})
        for block in model.inner_model.inner_model.diffusion_model.double_stream_blocks:
            for attr in ["txt_q_cache", "txt_k_cache", "txt_v_cache", "img_q_cache", "img_k_cache", "img_v_cache"]:
                if hasattr(block.block.attn1, attr):
                    delattr(block.block.attn1, attr)

        for block in model.inner_model.inner_model.diffusion_model.single_stream_blocks:
            block.block.attn1.EO = EO 
            for attr in ["txt_q_cache", "txt_k_cache", "txt_v_cache", "img_q_cache", "img_k_cache", "img_v_cache"]:
                if hasattr(block.block.attn1, attr):
                    delattr(block.block.attn1, attr)

    RK.update_transformer_options({'ExtraOptions': copy.deepcopy(EO)})
    if EO("update_cross_attn"):
        update_cross_attn = {
            'src_llama_start': EO('src_llama_start', 0),
            'src_llama_end':   EO('src_llama_end', 0),
            'src_t5_start':    EO('src_t5_start', 0),
            'src_t5_end':      EO('src_t5_end', 0),

            'tgt_llama_start': EO('tgt_llama_start', 0),
            'tgt_llama_end':   EO('tgt_llama_end', 0),
            'tgt_t5_start':    EO('tgt_t5_start', 0),
            'tgt_t5_end':      EO('tgt_t5_end', 0),
            'skip_cross_attn': EO('skip_cross_attn', False),
            
            'update_q':        EO('update_q', False),
            'update_k':        EO('update_k', True),
            'update_v':        EO('update_v', True),
            
            
            'lamb':  EO('lamb', 0.01),
            'erase': EO('erase', 10.0),
        }
        RK.update_transformer_options({'update_cross_attn':  update_cross_attn})
    else:
        RK.update_transformer_options({'update_cross_attn':  None})

    if LG.HAS_LATENT_GUIDE_ADAIN:
        RK.update_transformer_options({'blocks_adain_cache': []})
    if LG.HAS_LATENT_GUIDE_ATTNINJ:
        RK.update_transformer_options({'blocks_attninj_cache': []})
    if LG.HAS_LATENT_GUIDE_STYLE_POS:
        if LG.HAS_LATENT_GUIDE and y0_standard_guide is None:
            y0_cache = LG.y0.clone().cpu()
            RK.update_transformer_options({'y0_standard_guide': LG.y0})
        
    sigmas_scheduled = sigmas.clone() # store for return in state_info_out
    
    if EO("sigma_restarts"):
        sigma_restarts = 1 + EO("sigma_restarts", 0)
        sigmas = sigmas[step:num_steps+1].repeat(sigma_restarts)
        step = 0
        num_steps = 2 * sigma_restarts - 1
        
    if RENOISE:      # TODO: adapt for noise inversion somehow
        if VE_MODEL:
            x = x + sigmas[step] * NS.noise_sampler(sigma=sigmas[step], sigma_next=sigmas[step+1])
        else:
            x = (1 - sigmas[step]) * x + sigmas[step] * NS.noise_sampler(sigma=sigmas[step], sigma_next=sigmas[step+1])
                
    if guides is not None:
        RK.update_transformer_options({"freqsep_lowpass_method": guides.get("freqsep_lowpass_method")})
        RK.update_transformer_options({"freqsep_sigma":          guides.get("freqsep_sigma")})
        RK.update_transformer_options({"freqsep_kernel_size":    guides.get("freqsep_kernel_size")})
        RK.update_transformer_options({"freqsep_inner_kernel_size":    guides.get("freqsep_inner_kernel_size")})
        RK.update_transformer_options({"freqsep_stride":    guides.get("freqsep_stride")})

        
        RK.update_transformer_options({"freqsep_lowpass_weight": guides.get("freqsep_lowpass_weight")})
        RK.update_transformer_options({"freqsep_highpass_weight":guides.get("freqsep_highpass_weight")})
        RK.update_transformer_options({"freqsep_mask":           guides.get("freqsep_mask")})

    # BEGIN SAMPLING LOOP
                
    while step < num_steps:
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        if sigma_next > sigma:
            step_sched = torch.where(torch.flip(sigmas, dims=[0]) == sigma)[0][0].item()
        else:
            step_sched = step
            
        SYNC_GUIDE_ACTIVE = LG.guide_mode.startswith("sync") and (LG.lgw[step_sched] != 0 or LG.lgw_inv[step_sched] != 0 or LG.lgw_sync[step_sched] != 0 or LG.lgw_sync_inv[step_sched] != 0)
        
        if LG.HAS_LATENT_GUIDE_ADAIN:
            if LG.lgw_adain[step_sched] == 0.0:
                RK.update_transformer_options({'y0_adain': None})
                RK.update_transformer_options({'blocks_adain': {}})
            else:
                RK.update_transformer_options({'y0_adain': LG.y0_adain.clone()})
                if 'blocks_adain_mmdit' in guides:
                    blocks_adain = {
                        "double_weights": [val * LG.lgw_adain[step_sched] for val in guides['blocks_adain_mmdit']['double_weights']],
                        "single_weights": [val * LG.lgw_adain[step_sched] for val in guides['blocks_adain_mmdit']['single_weights']],
                        "double_blocks" : guides['blocks_adain_mmdit']['double_blocks'],
                        "single_blocks" : guides['blocks_adain_mmdit']['single_blocks'],
                    }
                RK.update_transformer_options({'blocks_adain': blocks_adain})
        
        if LG.HAS_LATENT_GUIDE_ATTNINJ:
            if LG.lgw_attninj[step_sched] == 0.0:
                RK.update_transformer_options({'y0_attninj': None})
                RK.update_transformer_options({'blocks_attninj'    : {}})
                RK.update_transformer_options({'blocks_attninj_qkv': {}})
            else:
                RK.update_transformer_options({'y0_attninj': LG.y0_attninj.clone()})
                if 'blocks_attninj_mmdit' in guides:
                    blocks_attninj = {
                        "double_weights": [val * LG.lgw_attninj[step_sched] for val in guides['blocks_attninj_mmdit']['double_weights']],
                        "single_weights": [val * LG.lgw_attninj[step_sched] for val in guides['blocks_attninj_mmdit']['single_weights']],
                        "double_blocks" : guides['blocks_attninj_mmdit']['double_blocks'],
                        "single_blocks" : guides['blocks_attninj_mmdit']['single_blocks'],
                    }
                RK.update_transformer_options({'blocks_attninj'    : blocks_attninj})
                RK.update_transformer_options({'blocks_attninj_qkv': guides['blocks_attninj_qkv']})

        if LG.HAS_LATENT_GUIDE_STYLE_POS:
            if LG.lgw_style_pos[step_sched] == 0.0:
                RK.update_transformer_options({'y0_style_pos':        None})
                RK.update_transformer_options({'y0_style_pos_weight': 0.0})
                RK.update_transformer_options({'y0_style_pos_synweight': 0.0})
                RK.update_transformer_options({'y0_style_pos_mask': None})
            else:
                RK.update_transformer_options({'y0_style_pos':        LG.y0_style_pos.clone()})
                RK.update_transformer_options({'y0_style_pos_weight': LG.lgw_style_pos[step_sched]})
                RK.update_transformer_options({'y0_style_pos_synweight': guides['synweight_style_pos']})
                RK.update_transformer_options({'y0_style_pos_mask': LG.mask_style_pos})
                RK.update_transformer_options({'y0_style_pos_mask_edge': guides.get('mask_edge_style_pos')})
                RK.update_transformer_options({'y0_style_method': guides['style_method']})
                RK.update_transformer_options({'y0_style_tile_height': guides.get('style_tile_height')})
                RK.update_transformer_options({'y0_style_tile_width': guides.get('style_tile_width')})
                RK.update_transformer_options({'y0_style_tile_padding': guides.get('style_tile_padding')})
                
                if EO("style_edge_width"):
                    RK.update_transformer
                
                #if LG.HAS_LATENT_GUIDE:
                #    y0_cache = LG.y0.clone().cpu()
                #    RK.update_transformer_options({'y0_standard_guide': LG.y0})
                    
                if LG.HAS_LATENT_GUIDE_INV and y0_inv_standard_guide is None:
                    y0_inv_cache = LG.y0_inv.clone().cpu()
                    RK.update_transformer_options({'y0_inv_standard_guide': LG.y0_inv})
                    

        if LG.HAS_LATENT_GUIDE_STYLE_NEG:
            if LG.lgw_style_neg[step_sched] == 0.0:
                RK.update_transformer_options({'y0_style_neg':        None})
                RK.update_transformer_options({'y0_style_neg_weight': 0.0})
                RK.update_transformer_options({'y0_style_neg_synweight': 0.0})
                RK.update_transformer_options({'y0_style_neg_mask': None})
            else:
                RK.update_transformer_options({'y0_style_neg':        LG.y0_style_neg.clone()})
                RK.update_transformer_options({'y0_style_neg_weight': LG.lgw_style_neg[step_sched]})
                RK.update_transformer_options({'y0_style_neg_synweight': guides['synweight_style_neg']})
                RK.update_transformer_options({'y0_style_neg_mask': LG.mask_style_neg})
                RK.update_transformer_options({'y0_style_neg_mask_edge': guides.get('mask_edge_style_neg')})
                RK.update_transformer_options({'y0_style_method': guides['style_method']})
                RK.update_transformer_options({'y0_style_tile_height': guides.get('style_tile_height')})
                RK.update_transformer_options({'y0_style_tile_width': guides.get('style_tile_width')})
                RK.update_transformer_options({'y0_style_tile_padding': guides.get('style_tile_padding')})

        if AttnMask_neg is not None:
            RK.update_transformer_options({'regional_conditioning_weight_neg': RegParam_neg.weights[step_sched]})
            RK.update_transformer_options({'regional_conditioning_floor_neg':  RegParam_neg.floors[step_sched]})

        if AttnMask is not None:
            RK.update_transformer_options({'regional_conditioning_weight': RegParam.weights[step_sched]})
            RK.update_transformer_options({'regional_conditioning_floor':  RegParam.floors[step_sched]})

        elif regional_conditioning_weights is not None:
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = regional_conditioning_weights[step_sched]
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = regional_conditioning_floors [step_sched]
        
        epsilon_scale        = float(epsilon_scales [step_sched]) if epsilon_scales        is not None else None
        eta                  = etas                 [step_sched]  if etas                  is not None else eta
        eta_substep          = etas_substep         [step_sched]  if etas_substep          is not None else eta_substep
        s_noise              = s_noises             [step_sched]  if s_noises              is not None else s_noise
        s_noise_substep      = s_noises_substep     [step_sched]  if s_noises_substep      is not None else s_noise_substep
        noise_scaling_eta    = noise_scaling_etas   [step_sched]  if noise_scaling_etas    is not None else noise_scaling_eta
        noise_scaling_weight = noise_scaling_weights[step_sched]  if noise_scaling_weights is not None else noise_scaling_weight
        
        NS.set_sde_step(sigma, sigma_next, eta, overshoot, s_noise)
        RK.set_coeff(rk_type, NS.h, c1, c2, c3, step, sigmas, NS.sigma_down)
        NS.set_substep_list(RK)

        if (noise_scaling_eta > 0 or noise_scaling_weight != 0) and noise_scaling_type != "model_d":
            if noise_scaling_type == "model_alpha":
                VP_OVERRIDE=True
            else:
                VP_OVERRIDE=None
            if noise_scaling_type in {"sampler", "model", "model_alpha"}:
                if noise_scaling_type == "model_alpha":
                    sigma_divisor = NS.sigma_max
                else:
                    sigma_divisor = 1.0
                
                if RK.multistep_stages > 0:
                    lying_su, lying_sigma, lying_sd, lying_alpha_ratio = NS.get_sde_step(NS.s_[1]/sigma_divisor, NS.s_[0]/sigma_divisor, noise_scaling_eta, noise_scaling_mode, VP_OVERRIDE=VP_OVERRIDE)
                    
                else:
                    lying_su, lying_sigma, lying_sd, lying_alpha_ratio = NS.get_sde_step(sigma/sigma_divisor, NS.sigma_down/sigma_divisor, noise_scaling_eta, noise_scaling_mode, VP_OVERRIDE=VP_OVERRIDE)
                    for _ in range(noise_scaling_cycles-1):
                        lying_su, lying_sigma, lying_sd, lying_alpha_ratio = NS.get_sde_step(sigma/sigma_divisor, lying_sd/sigma_divisor, noise_scaling_eta, noise_scaling_mode, VP_OVERRIDE=VP_OVERRIDE)
                lying_s_ = NS.get_substep_list(RK, sigma, RK.h_fn(lying_sd, lying_sigma))
                lying_s_ = NS.s_ + noise_scaling_weight * (lying_s_ - NS.s_)
            else:
                lying_s_ = NS.s_.clone()
        

        rk_swap_stages = 3 if rk_swap_type != "" else 0
        data_prev_len = len(data_prev_)-1 if data_prev_ is not None else 3
        recycled_stages = max(rk_swap_stages, RK.multistep_stages, RK.hybrid_stages, data_prev_len)
        
        if INIT_SAMPLE_LOOP:
            INIT_SAMPLE_LOOP = False
            x_, data_, eps_, eps_prev_ = (torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device) for _ in range(4))
            
            if sampler_mode in {"unsample", "resample"}:
                data_prev_ = state_info.get('data_prev_')
                if data_prev_ is not None:
                    if x.shape == state_info['raw_x'].shape:
                        data_prev_ = state_info['data_prev_'].clone().to(dtype=default_dtype, device=work_device)
                    else:
                        data_prev_ = torch.stack([comfy.utils.bislerp(data_prev_item, x.shape[-1], x.shape[-2]) for data_prev_item in state_info['data_prev_']])
                        data_prev_ = data_prev_.to(x)
                else:
                    data_prev_ =  torch.zeros(4, *x.shape, dtype=default_dtype, device=work_device) # multistep max is 4m... so 4 needed
            else:
                data_prev_ =  torch.zeros(4, *x.shape, dtype=default_dtype, device=work_device) # multistep max is 4m... so 4 needed
            
            recycled_stages = len(data_prev_)-1
        
        if RK.rows+2 > x_.shape[0]:
            row_gap = RK.rows+2 - x_.shape[0]
            x_gap_, data_gap_, eps_gap_, eps_prev_gap_ = (torch.zeros(row_gap, *x.shape, dtype=default_dtype, device=work_device) for _ in range(4))
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
        x_0   = x_[0].clone()
        if EO("guide_step_cutoff") or EO("guide_step_min"):
            x_0_orig = x_0.clone()
        
        # RECYCLE STAGES FOR MULTISTEP
        if RK.multistep_stages > 0 or RK.hybrid_stages > 0:
            if SYNC_GUIDE_ACTIVE:
                lgw_mask_, lgw_mask_inv_ = LG.get_masks_for_step(step)
                lgw_mask_sync_, lgw_mask_sync_inv_ = LG.get_masks_for_step(step, lgw_type="sync")
                #weight_mask     = (1-(lgw_mask_+lgw_mask_inv_)) * torch.ones_like(lgw_mask_)   +   (lgw_mask_+lgw_mask_inv_) * (lgw_mask_+lgw_mask_inv_)
                #weight_mask_inv = (1-(lgw_mask_+lgw_mask_inv_)) * torch.ones_like(lgw_mask_)   +   (lgw_mask_+lgw_mask_inv_) * (1-(lgw_mask_+lgw_mask_inv_))
                
                weight_mask = lgw_mask_+lgw_mask_inv_
                if LG.SYNC_SEPARATE:
                    sync_mask = lgw_mask_sync_+lgw_mask_sync_inv_
                else:
                    sync_mask = 1.
                            
                if VE_MODEL:
                    yt_0 = y0_bongflow + sigma * noise_bongflow
                else:
                    yt_0 = (1-sigma) * y0_bongflow  + sigma * noise_bongflow
                for ms in range(min(len(data_prev_), len(eps_))):
                    eps_x = RK.get_epsilon_anchored(x_0,  data_prev_x_[ms], sigma)
                    eps_y = RK.get_epsilon_anchored(yt_0, data_prev_y_[ms], sigma)
                    eps_x2y = RK.get_epsilon_anchored(yt_0, data_prev_y_[ms], sigma)

                    if RK.EXPONENTIAL:
                        if VE_MODEL:
                            eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_y + sigma*(-noise_bongflow))
                            if EO("sync_x2y"):
                                eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_x2y + sigma*(-noise_bongflow))
                        else:
                            eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_y + sigma*(y0_bongflow-noise_bongflow))
                            if EO("sync_x2y"):
                                eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_x2y + sigma*(y0_bongflow-noise_bongflow))
                    else:
                        if VE_MODEL:
                            eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_y + (noise_bongflow))
                            if EO("sync_x2y"):
                                eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_x2y + (noise_bongflow))
                        else:
                            eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_y + (noise_bongflow-y0_bongflow))
                            if EO("sync_x2y"):
                                eps_[ms] = sync_mask * eps_x  +  (1-sync_mask) * eps_x2y  +  weight_mask * (-eps_x2y + (noise_bongflow-y0_bongflow))

                    #if RK.EXPONENTIAL:
                    #    if VE_MODEL:
                    #        eps_[ms] = sync_mask * weight_mask_inv * (eps_x - weight_mask * eps_y) +  weight_mask * sigma*(-noise_bongflow)
                    #    else:
                    #        #eps_[ms] = (lgw_mask_sync_+lgw_mask_sync_inv_) * (1-(lgw_mask_+lgw_mask_inv_)) * (eps_x - (lgw_mask_+lgw_mask_inv_) * eps_y) +  (lgw_mask_+lgw_mask_inv_) * sigma*(y0_bongflow-noise_bongflow)
                    #        eps_[ms] = sync_mask * weight_mask_inv * (eps_x - weight_mask * eps_y) +  weight_mask * sigma*(y0_bongflow-noise_bongflow)
                    #else:
                    #    if VE_MODEL:
                    #        eps_[ms] = sync_mask * weight_mask_inv * (eps_x - weight_mask * eps_y) +  weight_mask *       (noise_bongflow)
                    #    else:
                    #        #eps_[ms] = (lgw_mask_sync_+lgw_mask_sync_inv_) * (1-(lgw_mask_+lgw_mask_inv_)) * (eps_x - (lgw_mask_+lgw_mask_inv_) * eps_y) +  (lgw_mask_+lgw_mask_inv_) *       (noise_bongflow-y0_bongflow)
                    #        eps_[ms] = sync_mask * weight_mask_inv * (eps_x - weight_mask * eps_y) +  weight_mask *       (noise_bongflow-y0_bongflow)
                eps_prev_ = eps_.clone()
            
            else:
                for ms in range(min(len(data_prev_), len(eps_))):
                    eps_[ms] = RK.get_epsilon_anchored(x_0, data_prev_[ms], sigma)
                eps_prev_ = eps_.clone()



        # INITIALIZE IMPLICIT SAMPLING
        if RK.IMPLICIT:
            x_, eps_, data_ = init_implicit_sampling(RK, x_0, x_, eps_, eps_prev_, data_, eps, denoised, denoised_prev2, step, sigmas, NS.h, NS.s_, EO, SYNC_GUIDE_ACTIVE)

        implicit_steps_total = (implicit_steps_full + 1) * (implicit_steps_diag + 1)

        # BEGIN FULLY IMPLICIT LOOP
        cossim_counter = 0
        adaptive_lgw = LG.lgw.clone()
        full_iter = 0
        while full_iter < implicit_steps_full+1:

            if RK.IMPLICIT:
                x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, 0, NS.h, sigmas, step, "init", SYNC_GUIDE_ACTIVE)

            # PREPARE FULLY PSEUDOIMPLICIT GUIDES
            if step > 0 or not SKIP_PSEUDO:
                if full_iter > 0 and EO("fully_implicit_reupdate_x"):
                    x_[0] = NS.sigma_from_to(x_0, x, sigma, sigma_next, NS.s_[0])
                    x_0   = NS.sigma_from_to(x_0, x, sigma, sigma_next, sigma)
                
                if EO("fully_pseudo_init") and full_iter == 0:
                    guide_mode_tmp = LG.guide_mode
                    LG.guide_mode = "fully_" + LG.guide_mode
                x_0, x_, eps_ = LG.prepare_fully_pseudoimplicit_guides_substep(x_0, x_, eps_, eps_prev_, data_, denoised_prev, 0, step, step_sched, sigmas, eta_substep, overshoot_substep, s_noise_substep, \
                                                                                NS, RK, pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, BONGMATH)
                if EO("fully_pseudo_init") and full_iter == 0:
                    LG.guide_mode = guide_mode_tmp

            # TABLEAU LOOP
            for row in range(RK.rows - RK.multistep_stages - RK.row_offset + 1):
                diag_iter = 0
                while diag_iter < implicit_steps_diag+1:
                    

                    if noise_sampler_type_substep == "brownian" and (full_iter > 0 or diag_iter > 0):
                        eta_substep = 0.
                    
                    NS.set_sde_substep(row, RK.multistep_stages, eta_substep, overshoot_substep, s_noise_substep, full_iter, diag_iter, implicit_steps_full, implicit_steps_diag)

                    # PRENOISE METHOD HERE!
                    
                    # A-TABLEAU
                    if row < RK.rows:

                        # PREPARE PSEUDOIMPLICIT GUIDES
                        if step > 0 or not SKIP_PSEUDO:
                            x_0, x_, eps_, x_row_pseudoimplicit, sub_sigma_pseudoimplicit = LG.process_pseudoimplicit_guides_substep(x_0, x_, eps_, eps_prev_, data_, denoised_prev, row, step, step_sched, sigmas, NS, RK, \
                                                                                                                        pseudoimplicit_row_weights, pseudoimplicit_step_weights, full_iter, BONGMATH)
                        
                        # PREPARE MODEL CALL
                        if LG.guide_mode in GUIDE_MODE_NAMES_PSEUDOIMPLICIT and (step > 0 or not SKIP_PSEUDO) and (LG.lgw[step_sched] > 0 or LG.lgw_inv[step_sched] > 0) and x_row_pseudoimplicit is not None:

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
                                
                            elif implicit_type == "rebound":          # TODO: ADAPT REBOUND IMPLICIT TO WORK WITH FLOW GUIDE MODE
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
                                        x_tmp =    x_[row]
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
                                    eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step_sched, sigma, sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                                    eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step_sched, sigma, sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                                if row == 0 and (EO("implicit_lagrange_init")  or   EO("radaucycle")):
                                    pass
                                else:
                                    x_[row+RK.row_offset] = x_0 + NS.h_new * RK.zum(row+RK.row_offset, eps_, eps_prev_)
                                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                                    if row > 0:
                                        if not LG.guide_mode.startswith("flow") or (LG.lgw[step_sched] == 0 and LG.lgw[step+1] == 0   and   LG.lgw_inv[step_sched] == 0 and LG.lgw_inv[step+1] == 0):
                                            #x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset])
                                            
                                            x_row_tmp = NS.swap_noise_substep(x_0, x_[row+RK.row_offset], mask=sde_mask, guide=LG.y0)
                                            
                                            if SYNC_GUIDE_ACTIVE:
                                                noise_bongflow_new = (x_row_tmp - x_[row+RK.row_offset]) / s_tmp + noise_bongflow
                                                yt_[row+RK.row_offset] += s_tmp * (noise_bongflow_new - noise_bongflow)
                                                x_0 += sigma * (noise_bongflow_new - noise_bongflow)
                                                if not EO("disable_i_bong"):
                                                    for i_bong in range(len(NS.s_)):
                                                        x_[i_bong] += NS.s_[i_bong] * (noise_bongflow_new - noise_bongflow)
                                                noise_bongflow = noise_bongflow_new
                                            
                                            x_[row+RK.row_offset] = x_row_tmp
                                        
                                        if SYNC_GUIDE_ACTIVE:
                                            #yt_[:NS.s_.shape[0], 0] = y0_bongflow + NS.s_.view(-1, 1, 1, 1) * (noise_bongflow - y0_bongflow)
                                            #yt_0   = y0_bongflow + sigma * (noise_bongflow - y0_bongflow)
                                            if VE_MODEL:
                                                yt_[:NS.s_.shape[0], 0] = y0_bongflow + NS.s_.view(-1, 1, 1, 1) * (noise_bongflow)
                                                yt_0   = y0_bongflow + sigma * (noise_bongflow)
                                            else:
                                                yt_[:NS.s_.shape[0], 0] = y0_bongflow + NS.s_.view(-1, 1, 1, 1) * (noise_bongflow - y0_bongflow)
                                                yt_0   = y0_bongflow + sigma * (noise_bongflow - y0_bongflow)
                                            
                                            if RK.EXPONENTIAL:
                                                eps_y_ = data_y_ - yt_0 # yt_        # watch out for fuckery with size of tableau being smaller later in a chained sampler
                                            else:
                                                if BONGMATH:
                                                    eps_y_[:NS.s_.shape[0]] = (yt_[:NS.s_.shape[0]] - data_y_[:NS.s_.shape[0]]) / NS.s_.view(-1,1,1,1,1) 
                                                    #eps_y_[:NS.s_.shape[0]] = (yt_0.repeat(NS.s_.shape[0], 1,1,1,1) - data_y_[:NS.s_.shape[0]]) / sigma 
                                                else:
                                                    eps_y_[:NS.s_.shape[0]] = (yt_0.repeat(NS.s_.shape[0], 1,1,1,1) - data_y_[:NS.s_.shape[0]]) / sigma    # calc exact to c0 node
                                            if not BONGMATH:
                                                if RK.EXPONENTIAL:
                                                    eps_x_ = data_x_ - x_0 
                                                else:
                                                    eps_x_ = (x_0 - data_x_) / sigma
                                            
                                            weight_mask = lgw_mask_+lgw_mask_inv_
                                            if LG.SYNC_SEPARATE:
                                                sync_mask = lgw_mask_sync_+lgw_mask_sync_inv_
                                            else:
                                                sync_mask = 1.
                                            
                                            
                                            for ms in range(len(eps_)):
                                                if RK.EXPONENTIAL:
                                                    if VE_MODEL:         # ZERO IS THIS                      # ONE IS THIS
                                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + sigma*(-noise_bongflow))
                                                        if EO("sync_x2y"):
                                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + sigma*(-noise_bongflow))
                                                    else:
                                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + sigma*(y0_bongflow-noise_bongflow))
                                                        if EO("sync_x2y"):
                                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + sigma*(y0_bongflow-noise_bongflow))
                                                else:
                                                    if VE_MODEL:
                                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + (noise_bongflow))
                                                        if EO("sync_x2y"):
                                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + (noise_bongflow))
                                                    else:
                                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + (noise_bongflow-y0_bongflow))
                                                        if EO("sync_x2y"):
                                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + (noise_bongflow-y0_bongflow))

                                            
                                        if BONGMATH and step < sigmas.shape[0]-1 and sigma > 0.03 and not EO("disable_implicit_prebong"):
                                            BONGMATH_Y = SYNC_GUIDE_ACTIVE
                                            
                                            x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step, step_sched,
                                                                        BONGMATH_Y, y0_bongflow, noise_bongflow, eps_x_, eps_y_, data_x_, data_y_, LG)     # TRY WITH h_new ??
                                            #                            BONGMATH_Y, y0_bongflow, noise_bongflow, eps_x_, eps_y_, eps_x2y_, data_x_, LG)     # TRY WITH h_new ??
                                    x_tmp = x_[row+RK.row_offset]

                        lying_eps_row_factor = 1.0
                        # MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL
                        if RK.IMPLICIT   and   row == 0   and   (EO("implicit_lazy_recycle_first_model_call_at_start")   or   EO("radaucycle")  or RK.C[0] == 0.0):
                            pass
                        else: 
                            if s_tmp == 0:
                                break
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "pre", SYNC_GUIDE_ACTIVE) # will this do anything? not x_tmp

                            # DETAIL BOOST
                            if noise_scaling_type == "model_alpha" and noise_scaling_weight != 0 and noise_scaling_eta > 0:
                                s_tmp = s_tmp + noise_scaling_weight * (s_tmp * lying_alpha_ratio   -   s_tmp)
                            if noise_scaling_type == "model"       and noise_scaling_weight != 0 and noise_scaling_eta > 0:
                                s_tmp = lying_s_[row]
                                if RK.multistep_stages > 0:
                                    s_tmp = lying_sd

                            # SYNC GUIDE ---------------------------
                            if LG.guide_mode.startswith("sync") and (LG.lgw[step_sched] == 0 and LG.lgw_inv[step_sched] == 0 and LG.lgw_sync[step_sched] == 0 and LG.lgw_sync_inv[step_sched] == 0):
                                data_cached = None
                            elif SYNC_GUIDE_ACTIVE:
                                lgw_mask_,         lgw_mask_inv_         = LG.get_masks_for_step(step_sched)
                                lgw_mask_sync_,    lgw_mask_sync_inv_    = LG.get_masks_for_step(step_sched, lgw_type="sync")
                                lgw_mask_drift_x_, lgw_mask_drift_x_inv_ = LG.get_masks_for_step(step_sched, lgw_type="drift_x")
                                lgw_mask_drift_y_, lgw_mask_drift_y_inv_ = LG.get_masks_for_step(step_sched, lgw_type="drift_y")
                                lgw_mask_lure_x_,  lgw_mask_lure_x_inv_  = LG.get_masks_for_step(step_sched, lgw_type="lure_x")
                                lgw_mask_lure_y_,  lgw_mask_lure_y_inv_  = LG.get_masks_for_step(step_sched, lgw_type="lure_y")
                                
                                weight_mask  = lgw_mask_         + lgw_mask_inv_
                                sync_mask    = lgw_mask_sync_    + lgw_mask_sync_inv_


                                drift_x_mask = lgw_mask_drift_x_ + lgw_mask_drift_x_inv_
                                drift_y_mask = lgw_mask_drift_y_ + lgw_mask_drift_y_inv_
                                lure_x_mask  = lgw_mask_lure_x_  + lgw_mask_lure_x_inv_
                                lure_y_mask  = lgw_mask_lure_y_  + lgw_mask_lure_y_inv_
                                
                                if eps_x_ is None:
                                    eps_x_       = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    data_x_      = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    eps_y2x_     = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    eps_x2y_     = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    eps_yt_      = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    eps_y_       = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    eps_prev_y_  = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    data_y_      = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    yt_          = torch.zeros(RK.rows+2, *x.shape, dtype=default_dtype, device=work_device)
                                    
                                    RUN_X_0_COPY = False
                                    if noise_bongflow is None:
                                        RUN_X_0_COPY = True
                                        data_prev_x_ = torch.zeros(4, *x.shape, dtype=default_dtype, device=work_device)
                                        data_prev_y_ = torch.zeros(4, *x.shape, dtype=default_dtype, device=work_device)
                                        
                                        noise_bongflow = normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=NS.sigma_min), channelwise=True, inplace=True)

                                        _, _ = RK(noise_bongflow, s_tmp/s_tmp, noise_bongflow, sigma/sigma, transformer_options={'latent_type': 'xt'})

                                        if RK.extra_args['model_options']['transformer_options'].get('y0_standard_guide') is not None:
                                            if hasattr(model.inner_model.inner_model.diffusion_model, "y0_standard_guide"):
                                                LG.y0 = y0_standard_guide = model.inner_model.inner_model.diffusion_model.y0_standard_guide.clone()
                                                del model.inner_model.inner_model.diffusion_model.y0_standard_guide
                                                RK.extra_args['model_options']['transformer_options']['y0_standard_guide'] = None
                                        
                                        if RK.extra_args['model_options']['transformer_options'].get('y0_inv_standard_guide') is not None:
                                            if hasattr(model.inner_model.inner_model.diffusion_model, "y0_inv_standard_guide"):
                                                LG.y0_inv = y0_inv_standard_guide = model.inner_model.inner_model.diffusion_model.y0_inv_standard_guide.clone() # RK.extra_args['model_options']['transformer_options'].get('y0_standard_guide')
                                                del model.inner_model.inner_model.diffusion_model.y0_inv_standard_guide
                                                RK.extra_args['model_options']['transformer_options']['y0_inv_standard_guide'] = None

                                        y0_bongflow = LG.HAS_LATENT_GUIDE * LG.mask * LG.y0   +   LG.HAS_LATENT_GUIDE_INV * LG.mask_inv * LG.y0_inv  #LG.y0.clone()
                                    
                                    if VE_MODEL:
                                        yt_0 = y0_bongflow + sigma * noise_bongflow
                                        yt   = y0_bongflow + s_tmp * noise_bongflow
                                    else:
                                        yt_0 = (1-sigma) * y0_bongflow  + sigma * noise_bongflow
                                        yt   = (1-s_tmp) * y0_bongflow  + s_tmp * noise_bongflow
                                        
                                    yt_[row] = yt
                                    
                                    if RUN_X_0_COPY:
                                        x_0 = yt_0.clone()
                                        x_tmp = x_[row] = yt.clone()
                                else:

                                        
                                    y0_bongflow_orig = y0_bongflow.clone() if y0_bongflow_orig is None else y0_bongflow_orig
                                    y0_bongflow = y0_bongflow + LG.drift_x_data  * drift_x_mask * (data_x           - y0_bongflow) \
                                                              + LG.drift_x_sync  * drift_x_mask * (data_barf        - y0_bongflow) \
                                                              + LG.drift_y_data  * drift_y_mask * (data_y           - y0_bongflow) \
                                                              + LG.drift_y_sync  * drift_y_mask * (data_barf_y      - y0_bongflow) \
                                                              + LG.drift_y_guide * drift_y_mask * (y0_bongflow_orig - y0_bongflow)
                                    
                                    if torch.norm(y0_bongflow_orig - y0_bongflow) != 0 and EO("enable_y0_bongflow_update"):
                                        RK.update_transformer_options({'y0_style_pos': y0_bongflow.clone()})
                                    
                                    if not EO("skip_yt"):
                                        if VE_MODEL:
                                            yt_0 = y0_bongflow + sigma * noise_bongflow
                                            yt   = y0_bongflow + s_tmp * noise_bongflow
                                        else:
                                            yt_0 = (1-sigma) * y0_bongflow  + sigma * noise_bongflow
                                            yt   = (1-s_tmp) * y0_bongflow  + s_tmp * noise_bongflow

                                        yt_[row] = yt

                                if ((LG.lgw[step_sched].item() in {1,0} and LG.lgw_inv[step_sched].item() in {1,0} and LG.lgw[step_sched] == 1-LG.lgw_sync[step_sched] and LG.lgw_inv[step_sched] == 1-LG.lgw_sync_inv[step_sched]) or EO("sync_speed_mode")) and not EO("disable_sync_speed_mode"):
                                    data_y = y0_bongflow.clone()
                                    if RK.EXPONENTIAL:
                                        eps_y = data_y - yt_0
                                    elif BONGMATH:
                                        eps_y = (yt_[row] - data_y) / s_tmp
                                    else:
                                        eps_y = (yt_0 - data_y) / sigma
                                else:
                                    eps_y, data_y = RK(yt_[row], s_tmp, yt_0,  sigma, transformer_options={'latent_type': 'yt'})
                                    
                                eps_x, data_x = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'xt'})
                                
                                
                                for sync_lure_iter in range(LG.sync_lure_iter):
                                    if LG.sync_lure_sequence == "x -> y":

                                        if lure_x_mask.abs().sum() > 0:
                                            if VE_MODEL:
                                                x_tmp = x_tmp + lure_x_mask * (data_y - data_x) 
                                                #x_tmp = x_tmp + (data_y - data_x) 
                                            else:
                                                x_tmp = x_tmp + lure_x_mask * (NS.sigma_max - s_tmp) * (data_y - data_x) 
                                                #x_tmp = x_tmp + (NS.sigma_max - s_tmp) * (data_y - data_x) 
                                            eps_x_lure, data_x_lure = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'xt'})
                                            eps_x  = eps_x  + lure_x_mask * (eps_x_lure  - eps_x)
                                            data_x = data_x + lure_x_mask * (data_x_lure - data_x)
                                            
                                        if lure_y_mask.abs().sum() > 0:                      
                                            y_tmp = yt_[row].clone()
                                            if VE_MODEL:
                                                y_tmp = y_tmp + lure_y_mask * (data_x - data_y) 
                                                #y_tmp = y_tmp + (data_x - data_y) 
                                            else:
                                                y_tmp = y_tmp + lure_y_mask * (NS.sigma_max - s_tmp) * (data_x - data_y) 
                                                #y_tmp = y_tmp + (NS.sigma_max - s_tmp) * (data_x - data_y) 
                                            eps_y_lure, data_y_lure = RK(y_tmp, s_tmp, yt_0, sigma, transformer_options={'latent_type': 'yt'})
                                            eps_y  = eps_y  + lure_y_mask * (eps_y_lure  - eps_y)
                                            data_y = data_y + lure_y_mask * (data_y_lure - data_y)
                                        
                                    elif LG.sync_lure_sequence == "y -> x":
                                            
                                        if lure_y_mask.abs().sum() > 0:                      
                                            y_tmp = yt_[row].clone()
                                            if VE_MODEL:
                                                y_tmp = y_tmp + lure_y_mask * (data_x - data_y) 
                                                #y_tmp = y_tmp + (data_x - data_y) 
                                            else:
                                                y_tmp = y_tmp + lure_y_mask * (NS.sigma_max - s_tmp) * (data_x - data_y) 
                                                #y_tmp = y_tmp + (NS.sigma_max - s_tmp) * (data_x - data_y) 
                                            eps_y_lure, data_y_lure = RK(y_tmp, s_tmp, yt_0, sigma, transformer_options={'latent_type': 'yt'})
                                            eps_y  = eps_y  + lure_y_mask * (eps_y_lure  - eps_y)
                                            data_y = data_y + lure_y_mask * (data_y_lure - data_y)
                                        
                                        if lure_x_mask.abs().sum() > 0:
                                            if VE_MODEL:
                                                x_tmp = x_tmp + lure_x_mask * (data_y - data_x) 
                                                #x_tmp = x_tmp + (data_y - data_x) 
                                            else:
                                                x_tmp = x_tmp + lure_x_mask * (NS.sigma_max - s_tmp) * (data_y - data_x) 
                                                #x_tmp = x_tmp + (NS.sigma_max - s_tmp) * (data_y - data_x) 
                                            eps_x_lure, data_x_lure = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'xt'})
                                            eps_x  = eps_x  + lure_x_mask * (eps_x_lure  - eps_x)
                                            data_x = data_x + lure_x_mask * (data_x_lure - data_x)

                                    elif LG.sync_lure_sequence == "xy -> xy":
                                        data_x_orig, data_y_orig = data_x.clone(), data_y.clone()
                                        
                                        if lure_x_mask.abs().sum() > 0:
                                            if VE_MODEL:
                                                x_tmp = x_tmp + lure_x_mask * (data_y_orig - data_x_orig) 
                                                #x_tmp = x_tmp + (data_y_orig - data_x_orig) 
                                            else:
                                                x_tmp = x_tmp + lure_x_mask * (NS.sigma_max - s_tmp) * (data_y_orig - data_x_orig) 
                                                #x_tmp = x_tmp + (NS.sigma_max - s_tmp) * (data_y_orig - data_x_orig) 
                                            eps_x_lure, data_x_lure = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'xt'})
                                            eps_x  = eps_x  + lure_x_mask * (eps_x_lure  - eps_x)
                                            data_x = data_x + lure_x_mask * (data_x_lure - data_x)
                                        
                                        if lure_y_mask.abs().sum() > 0:                      
                                            y_tmp = yt_[row].clone()
                                            if VE_MODEL:
                                                y_tmp = y_tmp + lure_y_mask * (data_x_orig - data_y_orig) 
                                                #y_tmp = y_tmp + (data_x_orig - data_y_orig) 
                                            else:
                                                y_tmp = y_tmp + lure_y_mask * (NS.sigma_max - s_tmp) * (data_x_orig - data_y_orig) 
                                                #y_tmp = y_tmp + (NS.sigma_max - s_tmp) * (data_x_orig - data_y_orig) 
                                            eps_y_lure, data_y_lure = RK(y_tmp, s_tmp, yt_0, sigma, transformer_options={'latent_type': 'yt'})
                                            eps_y  = eps_y  + lure_y_mask * (eps_y_lure  - eps_y)
                                            data_y = data_y + lure_y_mask * (data_y_lure - data_y)
                                        
                                        
                                
                                if EO("sync_proj_y"):
                                    d_collinear_d_lerp = get_collinear(eps_x, eps_y)  
                                    d_lerp_ortho_d     = get_orthogonal(eps_y, eps_x)  
                                    eps_y             = d_collinear_d_lerp + d_lerp_ortho_d
                                    
                                if EO("sync_proj_y2"):
                                    d_collinear_d_lerp = get_collinear(eps_y, eps_x)  
                                    d_lerp_ortho_d     = get_orthogonal(eps_x, eps_y)  
                                    eps_y             = d_collinear_d_lerp + d_lerp_ortho_d
                                    
                                if EO("sync_proj_x"):
                                    d_collinear_d_lerp = get_collinear(eps_y, eps_x)  
                                    d_lerp_ortho_d     = get_orthogonal(eps_x, eps_y)  
                                    eps_x             = d_collinear_d_lerp + d_lerp_ortho_d

                                if EO("sync_proj_x2"):
                                    d_collinear_d_lerp = get_collinear(eps_x, eps_y)  
                                    d_lerp_ortho_d     = get_orthogonal(eps_y, eps_x)  
                                    eps_x             = d_collinear_d_lerp + d_lerp_ortho_d

                                if RK.EXPONENTIAL:
                                    eps_x2y = data_y - x_0
                                elif BONGMATH:
                                    eps_x2y = (x_[row] - data_y) / s_tmp
                                else:
                                    eps_x2y = (x_0 - data_y) / sigma
                                eps_x2y_[row] = eps_x2y
                                
                                if RK.EXPONENTIAL:
                                    eps_y2x = data_y - x_0
                                elif BONGMATH:
                                    eps_y2x = (x_[row] - data_y) / s_tmp
                                else:
                                    eps_y2x = (x_0 - data_y) / sigma
                                eps_y2x_[row] = eps_y2x
                                
                                if sigma_next > sigma and EO("sync_unsample"):
                                    eps_x   = RK.get_guide_epsilon(x_0,   x_[row], data_x, sigma, NS.s_[row], NS.sigma_down, None)
                                    eps_x2y = RK.get_guide_epsilon(x_0,   x_[row], data_y, sigma, NS.s_[row], NS.sigma_down, None)
                                    eps_y   = RK.get_guide_epsilon(yt_0, yt_[row], data_y, sigma, NS.s_[row], NS.sigma_down, None)
                                if sigma_next > sigma and EO("sync_unsample2"):
                                    eps_x   = RK.get_guide_epsilon(x_0,   x_[row], data_y, sigma, NS.s_[row], NS.sigma_down, None)
                                    eps_x2y = RK.get_guide_epsilon(x_0,   x_[row], data_y, sigma, NS.s_[row], NS.sigma_down, None)
                                    eps_y   = RK.get_guide_epsilon(yt_0, yt_[row], data_y, sigma, NS.s_[row], NS.sigma_down, None)

                                eps_x *= EO("eps_x_mult", 1.0)
                                eps_y *= EO("eps_y_mult", 1.0)
                                eps_x2y *= EO("eps_x2y_mult", 1.0)
                                eps_y2x *= EO("eps_y2x_mult", 1.0)
                                
                                if RK.EXPONENTIAL:
                                    if VE_MODEL:         # ZERO IS THIS                      # ONE IS THIS 
                                        eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (-eps_y + sigma*(-noise_bongflow)) 
                                        if EO("sync_x2y"):
                                            eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (-eps_x2y + sigma*(-noise_bongflow)) 
                                    else:
                                        eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (-eps_y + sigma*(y0_bongflow-noise_bongflow))   #+   lure_x_mask * sigma*(data_y - data_x) 
                                        if EO("sync_x2y"):
                                            eps_[row]  = sync_mask * eps_x   -   (1-sync_mask) * eps_x2y   +   weight_mask * (-eps_x2y + sigma*(y0_bongflow-noise_bongflow)) 
                                        eps_yt_[row]  = sync_mask * eps_y   +   (1-sync_mask) * eps_y2x   +   weight_mask * (-eps_x + sigma*(y0_bongflow-noise_bongflow))         # differentiate guide as well toward the x pred?
                                else:
                                    if VE_MODEL:
                                        eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (noise_bongflow - eps_y)
                                        if EO("sync_x2y"):
                                            eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (noise_bongflow - eps_x2y)
                                    else:
                                        eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (noise_bongflow - eps_y - y0_bongflow)
                                        if EO("sync_x2y"):
                                            eps_[row]  = sync_mask * eps_x   +   (1-sync_mask) * eps_x2y   +   weight_mask * (noise_bongflow - eps_x2y - y0_bongflow)
                                        eps_yt_[row]  = sync_mask * eps_y   +   (1-sync_mask) * eps_y2x   +   weight_mask * (noise_bongflow - eps_x - y0_bongflow)         # differentiate guide as well toward the x pred?

                                if VE_MODEL:
                                    data_[row] = x_0   +   sync_mask * NS.h * eps_x   +   (1-sync_mask) * NS.h * eps_x2y   -   weight_mask * (sigma*(eps_y + noise_bongflow))   # -   lure_x_mask * (sigma*(eps_y + eps_x)) 
                                    data_barf_y = yt_0   +   sync_mask * NS.h * eps_y   +   (1-sync_mask) * NS.h * eps_y2x   -   weight_mask * (sigma*(eps_x + noise_bongflow))
                                    if EO("sync_x2y"):
                                        data_[row] = x_0   +   sync_mask * NS.h * eps_x   +   (1-sync_mask) * NS.h * eps_x2y   -   weight_mask * (sigma*(eps_x2y + noise_bongflow)) 

                                else:

                                    data_[row] = x_0   +   sync_mask * NS.h * eps_x   +   (1-sync_mask) * NS.h * eps_x2y   -   weight_mask * (NS.h * eps_y + sigma*(noise_bongflow-y0_bongflow)) 
                                    data_barf_y = yt_0   +   sync_mask * NS.h * eps_y   +   (1-sync_mask) * NS.h * eps_y2x   -   weight_mask * (NS.h * eps_x + sigma*(noise_bongflow-y0_bongflow)) 
                                    if EO("sync_x2y"):
                                        data_[row] = x_0   +   sync_mask * NS.h * eps_x   +   (1-sync_mask) * NS.h * eps_x2y   -   weight_mask * (NS.h * eps_x2y + sigma*(noise_bongflow-y0_bongflow)) 

                                if EO("data_is_y0_with_lure_x_mask"):
                                    data_[row] = data_[row] + lure_x_mask * (y0_bongflow - data_[row])

                                if EO("eps_is_y0_with_lure_x_mask"):
                                    if RK.EXPONENTIAL:
                                        eps_[row] = eps_[row] + lure_x_mask * ((y0_bongflow - x_0) - eps_[row])
                                    else:
                                        eps_[row] = eps_[row] + lure_x_mask * (((x_0 - y0_bongflow) / sigma) - eps_[row])
                                data_barf   = data_[row]
                                data_cached = data_x
                                
                                eps_x_ [row] = eps_x
                                data_x_[row] = data_x
                                
                                eps_y_ [row] = eps_y
                                data_y_[row] = data_y

                                if EO("sync_use_fake_eps_y"):
                                    if RK.EXPONENTIAL:
                                        if VE_MODEL:
                                            eps_y_ [row] = sigma * ( - noise_bongflow)
                                        else:
                                            eps_y_ [row] = sigma * (y0_bongflow - noise_bongflow)
                                    else:
                                        if VE_MODEL:
                                            eps_y_ [row] = noise_bongflow
                                        else:
                                            eps_y_ [row] = noise_bongflow - y0_bongflow
                                if EO("sync_use_fake_data_y"):
                                    data_y_[row] = y0_bongflow
                                    

                                
                            
                            elif LG.guide_mode.startswith("flow") and (LG.lgw[step_sched] > 0 or LG.lgw_inv[step_sched] > 0) and not FLOW_STOPPED and not EO("flow_sync") :
                                lgw_mask_, lgw_mask_inv_ = LG.get_masks_for_step(step)
                                if not FLOW_STARTED and not FLOW_RESUMED:
                                    FLOW_STARTED = True
                                    data_x_prev_ = torch.zeros_like(data_prev_)

                                    y0 = LG.HAS_LATENT_GUIDE * LG.mask * LG.y0   +   LG.HAS_LATENT_GUIDE_INV * LG.mask_inv * LG.y0_inv 
                                    
                                    yx0 = y0.clone()
                                    
                                    if EO("flow_slerp"):
                                        y0_inv                 = LG.HAS_LATENT_GUIDE * LG.mask * LG.y0_inv   +   LG.HAS_LATENT_GUIDE_INV * LG.mask_inv * LG.y0 
                                        y0 = LG.y0.clone()
                                        y0_inv = LG.y0_inv.clone()
                                        flow_slerp_guide_ratio = EO("flow_slerp_guide_ratio", 0.5)
                                        y_slerp                = slerp_tensor(flow_slerp_guide_ratio, y0, y0_inv)
                                        yx0                    = y_slerp.clone()
                                    
                                    x_[row], x_0 =  yx0.clone(), yx0.clone()
                                    if EO("guide_step_cutoff") or EO("guide_step_min"):
                                        x_0_orig = yx0.clone()
                                    
                                    if EO("flow_yx0_init_y0_inv"):
                                        yx0 = LG.HAS_LATENT_GUIDE * LG.mask * LG.y0_inv   +   LG.HAS_LATENT_GUIDE_INV * LG.mask_inv * LG.y0
                                    
                                    if step > 0:
                                        if EO("flow_manual_masks"):
                                            y0  = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv)) * denoised   +   LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask * LG.y0   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv * LG.y0_inv
                                        else:
                                            y0  = (1 - (lgw_mask_ + lgw_mask_inv_)) * denoised   +   lgw_mask_ * LG.y0   +   lgw_mask_inv_ * LG.y0_inv
                                        yx0 = y0.clone()
                                        
                                        if EO("flow_slerp"):
                                            if EO("flow_manual_masks"):
                                                y0_inv                 = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv)) * denoised   +   LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask * LG.y0_inv   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv * LG.y0
                                            else:
                                                y0_inv  = (1 - (lgw_mask_ + lgw_mask_inv_)) * denoised   +   lgw_mask_ * LG.y0_inv   +   lgw_mask_inv_ * LG.y0
                                            flow_slerp_guide_ratio = EO("flow_slerp_guide_ratio", 0.5)
                                            y_slerp                = slerp_tensor(flow_slerp_guide_ratio, y0, y0_inv)
                                            yx0                    = y_slerp.clone()
                                
                                else:
                                    yx0_prev = data_cached
                                    if EO("flow_manual_masks"):
                                        yx0 = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv)) * yx0_prev   +   LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask * x_tmp   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv * x_tmp
                                    else:
                                        yx0 = (1 - (lgw_mask_ + lgw_mask_inv_)) * yx0_prev   +   (lgw_mask_ + lgw_mask_inv_) * x_tmp

                                    if not EO("flow_static_guides"):
                                        if EO("flow_manual_masks"):
                                            y0 = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv)) * yx0_prev   +   LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask * LG.y0   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv * LG.y0_inv
                                        else:
                                            y0 = (1 - (lgw_mask_ + lgw_mask_inv_)) * yx0_prev   +   lgw_mask_ * LG.y0   +   lgw_mask_inv_ * LG.y0_inv
                                        
                                        if EO("flow_slerp"):
                                            if EO("flow_manual_masks"):
                                                y0_inv = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv)) * yx0_prev   +   LG.HAS_LATENT_GUIDE * LG.lgw[step_sched] * LG.mask * LG.y0_inv   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step_sched] * LG.mask_inv * LG.y0
                                            else:
                                                y0_inv = (1 - (lgw_mask_ + lgw_mask_inv_)) * yx0_prev   +   lgw_mask_ * LG.y0_inv   +   lgw_mask_inv_ * LG.y0

                                y0_orig = y0.clone()
                                if EO("flow_proj_xy"):
                                    d_collinear_d_lerp = get_collinear(yx0, y0_orig)  
                                    d_lerp_ortho_d     = get_orthogonal(y0_orig, yx0)  
                                    y0                 = d_collinear_d_lerp + d_lerp_ortho_d
                                
                                if EO("flow_proj_yx"):
                                    d_collinear_d_lerp = get_collinear(y0_orig, yx0)  
                                    d_lerp_ortho_d     = get_orthogonal(yx0, y0_orig)  
                                    yx0                = d_collinear_d_lerp + d_lerp_ortho_d
                                
                                y0_inv_orig = None
                                if EO("flow_proj_xy_inv"):
                                    y0_inv_orig = y0_inv.clone()
                                    d_collinear_d_lerp = get_collinear(yx0, y0_inv)  
                                    d_lerp_ortho_d     = get_orthogonal(y0_inv, yx0)  
                                    y0_inv             = d_collinear_d_lerp + d_lerp_ortho_d
                                    
                                if EO("flow_proj_yx_inv"):
                                    y0_inv_orig = y0_inv if y0_inv_orig is None else y0_inv_orig
                                    d_collinear_d_lerp = get_collinear(y0_inv_orig, yx0)  
                                    d_lerp_ortho_d     = get_orthogonal(yx0, y0_inv_orig)  
                                    yx0                = d_collinear_d_lerp + d_lerp_ortho_d
                                del y0_orig

                                flow_cossim_iter = EO("flow_cossim_iter", 1)

                                if step == 0:
                                    noise_yt = noise_fn(y0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                if not EO("flow_disable_renoise_y0"):
                                    if noise_yt is None:
                                        noise_yt = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)
                                    else:
                                        noise_yt = (1-eta) * noise_yt + eta * noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)

                                if VE_MODEL:
                                    yt        = y0 + s_tmp * noise_yt
                                else:
                                    yt        = (NS.sigma_max-s_tmp) * y0 + (s_tmp/NS.sigma_max) * noise_yt
                                if not EO("flow_disable_doublenoise_y0"):
                                    if noise_yt is None:
                                        noise_yt = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)
                                    else:
                                        noise_yt = (1-eta) * noise_yt + eta * noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)

                                if VE_MODEL:
                                    y0_noised = y0 + sigma * noise_yt
                                else:
                                    y0_noised = (NS.sigma_max-sigma) * y0 + sigma * noise_yt
                                
                                if EO("flow_slerp"):
                                    noise = noise_fn(y0_inv, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) 
                                    yt_inv        = (NS.sigma_max-s_tmp) * y0_inv + (s_tmp/NS.sigma_max) * noise
                                    if not EO("flow_disable_doublenoise_y0_inv"):
                                        noise = noise_fn(y0_inv, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) 
                                    y0_noised_inv = (NS.sigma_max-sigma) * y0_inv + sigma * noise
                                
                                if step == 0:
                                    noise_xt = noise_fn(yx0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) 
                                if EO("flow_slerp"):
                                    xt         = yx0 + (s_tmp/NS.sigma_max) * (noise - y_slerp)
                                    if not EO("flow_disable_doublenoise_x_0"):
                                        noise = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) 
                                    x_0_noised = x_0 + sigma * (noise - y_slerp)
                                else:
                                    if not EO("flow_disable_renoise_x_0"):
                                        if noise_xt is None:
                                            noise_xt = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)
                                        else:
                                            noise_xt = (1-eta_substep) * noise_xt + eta_substep * noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)

                                    if VE_MODEL:
                                        xt         = yx0 + (s_tmp) * yx0 + (s_tmp) * (noise_xt - y0)
                                    else:
                                        xt         = yx0 + (s_tmp/NS.sigma_max) * (noise_xt - y0)
                                    if not EO("flow_disable_doublenoise_x_0"):
                                        if noise_xt is None:
                                            noise_xt = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)
                                        else:
                                            noise_xt = (1-eta_substep) * noise_xt + eta_substep * noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter)
                                    if VE_MODEL:
                                        x_0_noised = x_0 + (sigma) * x_0 + (sigma) * (noise_xt - y0)
                                    else:
                                        x_0_noised = x_0 + (sigma/NS.sigma_max) * (noise_xt - y0)    # just lerp noise add, (1-sigma)*y0 + sigma*noise assuming x_0 == y0, which is true initially...
                                
                                eps_y, data_y = RK(yt, s_tmp, y0_noised,  sigma, transformer_options={'latent_type': 'yt'})
                                eps_x, data_x = RK(xt, s_tmp, x_0_noised, sigma, transformer_options={'latent_type': 'xt'})
                    
                                if EO("flow_slerp"):
                                    eps_y_inv, data_y_inv = RK(yt_inv, s_tmp, y0_noised_inv, sigma, transformer_options={'latent_type': 'yt_inv'})
                                
                                if LG.lgw[step+1] == 0 and LG.lgw_inv[step+1] == 0:    # break out of differentiating x0 and return to differentiating eps/velocity field
                                    if EO("flow_shit_out_yx0"):
                                        eps_ [row]       = eps_x - eps_y
                                        data_[row]       = yx0
                                        if row == 0:
                                            x_[row] = x_0 = xt 
                                        else:
                                            x_[row] = xt
                                    if not EO("flow_shit_out_new"):
                                        eps_ [row]       = eps_x
                                        data_[row]       = data_x
                                        if row == 0:
                                            x_[row] = x_0 = xt 
                                        else:
                                            x_[row] = xt
                                    
                                    else:
                                        eps_ [row]        = (1 - (lgw_mask_ + lgw_mask_inv_)) * eps_x   +   (lgw_mask_ + lgw_mask_inv_) * eps_y
                                        data_[row]        = (1 - (lgw_mask_ + lgw_mask_inv_)) * data_x   +   (lgw_mask_ + lgw_mask_inv_) * data_y
                                        if row == 0:
                                            x_[row] = x_0 = (1 - (lgw_mask_ + lgw_mask_inv_)) * xt   +   (lgw_mask_ + lgw_mask_inv_) * yt 
                                        else:
                                            x_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * xt   +   (lgw_mask_ + lgw_mask_inv_) * yt
                                
                                    FLOW_STOPPED = True
                                else:
                                    if not EO("flow_slerp"):
                                        if RK.EXPONENTIAL:
                                            eps_y_alt = data_y - x_0
                                            eps_x_alt = data_x - x_0
                                        else:
                                            eps_y_alt = (x_0 - data_y) / sigma
                                            eps_x_alt = (x_0 - data_x) / sigma
                                            
                                        if EO("flow_y_zero"):
                                            eps_y_alt *= LG.mask
                                        
                                        eps_[row]  = eps_yx = (eps_y_alt - eps_x_alt)
                                        eps_y_lin           = (x_0 - data_y) / sigma
                                        if EO("flow_y_zero"):
                                            eps_y_lin *= LG.mask
                                        eps_x_lin           = (x_0 - data_x) / sigma
                                        eps_yx_lin          = (eps_y_lin - eps_x_lin)
                                        
                                        data_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * data_x   +   (lgw_mask_ + lgw_mask_inv_) * data_y
                                        
                                        if EO("flow_reverse_data_masks"):
                                            data_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * data_y   +   (lgw_mask_ + lgw_mask_inv_) * data_x

                                        if flow_sync_eps != 0.0:
                                            if RK.EXPONENTIAL:
                                                eps_[row] = (1-flow_sync_eps) * eps_[row] + flow_sync_eps * (data_[row] - x_0)
                                            else:
                                                eps_[row] = (1-flow_sync_eps) * eps_[row] + flow_sync_eps * (x_0 - data_[row]) / sigma
                                        
                                        if EO("flow_sync_eps_mask"): 
                                            flow_sync_eps = EO("flow_sync_eps_mask", 1.0)
                                            if RK.EXPONENTIAL:
                                                eps_[row] = (lgw_mask_ + lgw_mask_inv_) * (1-flow_sync_eps) * eps_[row] + (1 - (lgw_mask_ + lgw_mask_inv_)) * flow_sync_eps * (data_[row] - x_0) 
                                            else:
                                                eps_[row] = (lgw_mask_ + lgw_mask_inv_) * (1-flow_sync_eps) * eps_[row] + (1 - (lgw_mask_ + lgw_mask_inv_)) * flow_sync_eps * (x_0 - data_[row]) / sigma

                                        if EO("flow_sync_eps_revmask"): 
                                            flow_sync_eps = EO("flow_sync_eps_revmask", 1.0)
                                            if RK.EXPONENTIAL:
                                                eps_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * (1-flow_sync_eps) * eps_[row] + (lgw_mask_ + lgw_mask_inv_) * flow_sync_eps * (data_[row] - x_0) 
                                            else:
                                                eps_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * (1-flow_sync_eps) * eps_[row] + (lgw_mask_ + lgw_mask_inv_) * flow_sync_eps * (x_0 - data_[row]) / sigma

                                        if EO("flow_sync_eps_maskonly"):
                                            flow_sync_eps = EO("flow_sync_eps_maskonly", 1.0)
                                            if RK.EXPONENTIAL:
                                                eps_[row] = (lgw_mask_ + lgw_mask_inv_) * eps_[row] + (1 - (lgw_mask_ + lgw_mask_inv_)) * (data_[row] - x_0) 
                                            else:
                                                eps_[row] = (lgw_mask_ + lgw_mask_inv_) * eps_[row] + (1 - (lgw_mask_ + lgw_mask_inv_)) * (x_0 - data_[row]) / sigma

                                        if EO("flow_sync_eps_revmaskonly"): 
                                            flow_sync_eps = EO("flow_sync_eps_revmaskonly", 1.0)
                                            if RK.EXPONENTIAL:
                                                eps_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * eps_[row] + (lgw_mask_ + lgw_mask_inv_) * (data_[row] - x_0) 
                                            else:
                                                eps_[row] = (1 - (lgw_mask_ + lgw_mask_inv_)) * eps_[row] + (lgw_mask_ + lgw_mask_inv_) * (x_0 - data_[row]) / sigma

                                    if EO("flow_slerp"):
                                        if RK.EXPONENTIAL:
                                            eps_y_alt     = data_y     - x_0
                                            eps_y_alt_inv = data_y_inv - x_0
                                            eps_x_alt     = data_x     - x_0
                                        else:
                                            eps_y_alt     = (x_0 - data_y)     / sigma
                                            eps_y_alt_inv = (x_0 - data_y_inv) / sigma
                                            eps_x_alt     = (x_0 - data_x)     / sigma
                                        
                                        flow_slerp_ratio2 = EO("flow_slerp_ratio2", 0.5)

                                        eps_yx     = (eps_y_alt - eps_x_alt)
                                        eps_y_lin  = (x_0 - data_y) / sigma
                                        eps_x_lin  = (x_0 - data_x) / sigma
                                        eps_yx_lin = (eps_y_lin - eps_x_lin)
                                        
                                        eps_yx_inv     = (eps_y_alt_inv - eps_x_alt)
                                        eps_y_lin_inv  = (x_0 - data_y_inv) / sigma
                                        eps_x_lin      = (x_0 - data_x)     / sigma
                                        eps_yx_lin_inv = (eps_y_lin_inv - eps_x_lin)
                                        
                                        data_row     = x_0 - sigma * eps_yx_lin
                                        data_row_inv = x_0 - sigma * eps_yx_lin_inv
                                        
                                        if EO("flow_slerp_similarity_ratio"):
                                            flow_slerp_similarity_ratio = EO("flow_slerp_similarity_ratio", 1.0)
                                            flow_slerp_ratio2           = find_slerp_ratio_grid(data_row, data_row_inv, LG.y0.clone(), LG.y0_inv.clone(), flow_slerp_similarity_ratio)
                                        
                                        eps_ [row] = slerp_tensor(flow_slerp_ratio2, eps_yx,   eps_yx_inv)
                                        data_[row] = slerp_tensor(flow_slerp_ratio2, data_row, data_row_inv)
                                        
                                        if EO("flow_slerp_autoalter"):
                                            data_row_slerp = slerp_tensor(0.5, data_row, data_row_inv)
                                            y0_pearsim     = get_pearson_similarity(data_row_slerp, y0)
                                            y0_pearsim_inv = get_pearson_similarity(data_row_slerp, y0_inv)
                                            
                                            if y0_pearsim > y0_pearsim_inv:
                                                data_[row] = data_row_inv 
                                                eps_ [row] = (eps_y_alt_inv - eps_x_alt)
                                            else:
                                                data_[row] = data_row
                                                eps_ [row] = (eps_y_alt     - eps_x_alt)
                                            
                                        if EO("flow_slerp_recalc_eps_row"):
                                            if RK.EXPONENTIAL:
                                                eps_[row]  = data_[row] - x_0
                                            else:
                                                eps_[row]  = (x_0 - data_[row]) / sigma
                                        
                                        if EO("flow_slerp_recalc_data_row"):
                                            if RK.EXPONENTIAL:
                                                data_[row] = x_0 + eps_[row]
                                            else:
                                                data_[row] = x_0 - sigma * eps_[row]

                                    data_cached = data_x 

                            if step < EO("direct_pre_pseudo_guide", 0) and step > 0:
                                for i_pseudo in range(EO("direct_pre_pseudo_guide_iter", 1)):
                                    x_tmp += LG.lgw[step_sched] * LG.mask * (NS.sigma_max - s_tmp) * (LG.y0 - denoised)     +     LG.lgw_inv[step_sched] * LG.mask_inv * (NS.sigma_max - s_tmp) * (LG.y0_inv - denoised)
                                    eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                            
                            # MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL
                            
                            if    SYNC_GUIDE_ACTIVE:
                                pass
                            elif not ((not LG.guide_mode.startswith("flow"))   or   FLOW_STOPPED   or    (LG.guide_mode.startswith("flow") and LG.lgw[step_sched] == 0 and LG.lgw_inv[step_sched] == 0)): #(LG.guide_mode.startswith("flow") and (LG.lgw[step_sched] != 0 or LG.lgw_inv[step_sched] != 0)) or FLOW_STOPPED:
                                pass
                            elif LG.guide_mode.startswith("lure") and (LG.lgw[step_sched] > 0 or LG.lgw_inv[step_sched] > 0):
                                eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'yt'})
                                
                            else:
                                eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)

                            if RK.extra_args['model_options']['transformer_options'].get('y0_standard_guide') is not None:
                                if hasattr(model.inner_model.inner_model.diffusion_model, "y0_standard_guide"):
                                    LG.y0 = model.inner_model.inner_model.diffusion_model.y0_standard_guide.clone()
                                    del model.inner_model.inner_model.diffusion_model.y0_standard_guide
                                    RK.extra_args['model_options']['transformer_options']['y0_standard_guide'] = None
                                
                            if RK.extra_args['model_options']['transformer_options'].get('y0_inv_standard_guide') is not None:
                                if hasattr(model.inner_model.inner_model.diffusion_model, "y0_inv_standard_guide"):
                                    LG.y0_inv = model.inner_model.inner_model.diffusion_model.y0_inv_standard_guide.clone() # RK.extra_args['model_options']['transformer_options'].get('y0_standard_guide')
                                    del model.inner_model.inner_model.diffusion_model.y0_inv_standard_guide
                                    RK.extra_args['model_options']['transformer_options']['y0_inv_standard_guide'] = None

                            if LG.guide_mode.startswith("lure") and (LG.lgw[step_sched] > 0 or LG.lgw_inv[step_sched] > 0):
                                x_tmp = LG.process_guides_data_substep(x_tmp, data_[row], step_sched, s_tmp)
                                eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'xt'})

                            if momentum != 0.0:
                                data_[row] = data_[row] - momentum * (data_prev_[0] - data_[row])  #negative!
                                eps_[row]  = RK.get_epsilon(x_0, x_tmp, data_[row], sigma, s_tmp)    # ... why was this here??? for momentum maybe?

                            if row < RK.rows and noise_scaling_weight != 0 and noise_scaling_type in {"sampler", "sampler_substep"}:
                                if noise_scaling_type == "sampler_substep":
                                    sub_lying_su, sub_lying_sigma, sub_lying_sd, sub_lying_alpha_ratio = NS.get_sde_substep(NS.s_[row], NS.s_[row+RK.row_offset+RK.multistep_stages], noise_scaling_eta, noise_scaling_mode)
                                    for _ in range(noise_scaling_cycles-1):
                                        sub_lying_su, sub_lying_sigma, sub_lying_sd, sub_lying_alpha_ratio = NS.get_sde_substep(NS.s_[row], sub_lying_sd, noise_scaling_eta, noise_scaling_mode)
                                    lying_s_[row+1] = sub_lying_sd
                                substep_noise_scaling_ratio = NS.s_[row+1]/lying_s_[row+1]
                                if RK.multistep_stages > 0:
                                    substep_noise_scaling_ratio = sigma_next/lying_sd                   #fails with resample?
                                
                                lying_eps_row_factor = (1 - noise_scaling_weight*(substep_noise_scaling_ratio-1))



                        # GUIDE 
                        if not EO("disable_guides_eps_substep"):
                            eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step_sched, NS.sigma, NS.sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                        if not EO("disable_guides_eps_prev_substep"):
                            eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step_sched, NS.sigma, NS.sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                        
                        if LG.y0_mean is not None and LG.y0_mean.sum() != 0.0:
                            if EO("guide_mean_pw"):
                                data_row_new = adain_patchwise_row_batch(data_[row].clone(), LG.y0_mean.clone(), sigma=EO("guide_mean_pw_sigma", 1.0), kernel_size=EO("guide_mean_pw_kernel_size", 7))
                                if RK.EXPONENTIAL:
                                    eps_row_mean = data_row_new - x_0
                                else:
                                    eps_row_mean = (x_0 - data_row_new) / s_tmp
                            else:
                                eps_row_mean = eps_[row] - eps_[row].mean(dim=(-2,-1), keepdim=True) + (LG.y0_mean - x_0).mean(dim=(-2,-1), keepdim=True)
                            
                            if LG.mask_mean is not None:
                                eps_row_mean = LG.mask_mean * eps_row_mean + (1 - LG.mask_mean) * eps_[row]
                            
                            eps_[row] = eps_[row] + LG.lgw_mean[step_sched] * (eps_row_mean - eps_[row])
                            
                        if (full_iter == 0 and diag_iter == 0)   or   EO("newton_iter_post_use_on_implicit_steps"):
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "post", SYNC_GUIDE_ACTIVE)

                    # UPDATE   #for row in range(RK.rows - RK.multistep_stages - RK.row_offset + 1):
                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, RK.row_offset, NS.h_new, NS.h_new_orig, lying_eps_row_factor=lying_eps_row_factor)   #modifies eps_[row] if lying_eps_row_factor != 1.0
                    
                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                    
                    if SYNC_GUIDE_ACTIVE: #yt_ is not None:
                        #yt_ = RK.update_substep(yt_0, yt_, eps_y_, eps_prev_y_, row, RK.row_offset, NS.h_new, NS.h_new_orig, lying_eps_row_factor=lying_eps_row_factor)   #modifies eps_[row] if lying_eps_row_factor != 1.0
                        yt_ = RK.update_substep(yt_0, yt_, eps_yt_, eps_prev_y_, row, RK.row_offset, NS.h_new, NS.h_new_orig, lying_eps_row_factor=lying_eps_row_factor)   #modifies eps_[row] if lying_eps_row_factor != 1.0
                        yt_[row+RK.row_offset] = NS.rebound_overshoot_substep(yt_0, yt_[row+RK.row_offset])
                    
                    if not RK.IMPLICIT and NS.noise_mode_sde_substep != "hard_sq":

                        x_means_per_substep = x_[row+RK.row_offset].mean(dim=(-2,-1), keepdim=True)

                        if not LG.guide_mode.startswith("flow") or (LG.lgw[step_sched] == 0 and LG.lgw[step+1] == 0   and   LG.lgw_inv[step_sched] == 0 and LG.lgw_inv[step+1] == 0):
                            #if LG.guide_mode.startswith("sync") and (LG.lgw[step_sched] != 0.0 or LG.lgw_inv[step_sched] != 0.0):
                            #    x_row_tmp = x_[row+RK.row_offset].clone()
                                
                            #x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset], mask=sde_mask, guide=LG.y0)
                            x_row_tmp = NS.swap_noise_substep(x_0, x_[row+RK.row_offset], mask=sde_mask, guide=LG.y0)
                            
                            if SYNC_GUIDE_ACTIVE:
                                noise_bongflow_new = (x_row_tmp - x_[row+RK.row_offset]) / s_tmp + noise_bongflow
                                yt_[row+RK.row_offset] += s_tmp * (noise_bongflow_new - noise_bongflow)
                                
                                x_0 += sigma * (noise_bongflow_new - noise_bongflow)
                                
                                
                                noise_bongflow = noise_bongflow_new
                                
                            x_[row+RK.row_offset] = x_row_tmp
                            
                        elif LG.guide_mode.startswith("flow"):
                            pass

                    if not LG.guide_mode.startswith("lure"):
                        x_[row+RK.row_offset] = LG.process_guides_data_substep(x_[row+RK.row_offset], data_[row], step_sched, NS.s_[row])

                    if SYNC_GUIDE_ACTIVE: # # # # ## # # ## # YIIIIKES ---------------------------------------------------------------------------------------------------------
                        if VE_MODEL:
                            yt_[:NS.s_.shape[0], 0] = y0_bongflow + NS.s_.view(-1, 1, 1, 1) * (noise_bongflow)
                            yt_0   = y0_bongflow + sigma * (noise_bongflow)
                        else:
                            yt_[:NS.s_.shape[0], 0] = y0_bongflow + NS.s_.view(-1, 1, 1, 1) * (noise_bongflow - y0_bongflow)
                            yt_0   = y0_bongflow + sigma * (noise_bongflow - y0_bongflow)
                        if RK.EXPONENTIAL:
                            eps_y_ = data_y_ - yt_0 # yt_        # watch out for fuckery with size of tableau being smaller later in a chained sampler
                        else:
                            if BONGMATH:
                                eps_y_[:NS.s_.shape[0]] = (yt_[:NS.s_.shape[0]] - data_y_[:NS.s_.shape[0]]) / NS.s_.view(-1,1,1,1,1) 
                            else:
                                eps_y_[:NS.s_.shape[0]] = (yt_0.repeat(NS.s_.shape[0], 1,1,1,1) - data_y_[:NS.s_.shape[0]]) / sigma    # calc exact to c0 node
                        if not BONGMATH and (eta != 0 or eta_substep != 0):
                            if RK.EXPONENTIAL:
                                eps_x_ = data_x_ - x_0 
                            else:
                                eps_x_ = (x_0 - data_x_) / sigma

                            weight_mask = lgw_mask_+lgw_mask_inv_
                            if LG.SYNC_SEPARATE:
                                sync_mask = lgw_mask_sync_+lgw_mask_sync_inv_
                            else:
                                sync_mask = 1.
                            
                            for ms in range(len(eps_)):
                                if RK.EXPONENTIAL:
                                    if VE_MODEL:
                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + sigma*(-noise_bongflow))
                                        if EO("sync_x2y"):
                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + sigma*(-noise_bongflow))
                                    else:
                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + sigma*(y0_bongflow-noise_bongflow))
                                        if EO("sync_x2y"):
                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + sigma*(y0_bongflow-noise_bongflow))
                                else:
                                    if VE_MODEL:
                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + (noise_bongflow))
                                        if EO("sync_x2y"):
                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + (noise_bongflow))
                                    else:
                                        eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_y_[ms] + (noise_bongflow-y0_bongflow))
                                        if EO("sync_x2y"):
                                            eps_[ms] = sync_mask * eps_x_[ms]  +  (1-sync_mask) * eps_x2y_[ms]  +  weight_mask * (-eps_x2y_[ms] + (noise_bongflow-y0_bongflow))

                    if BONGMATH and NS.s_[row] > RK.sigma_min and NS.h < RK.sigma_max/2   and   (diag_iter == implicit_steps_diag or EO("enable_diag_explicit_bongmath_all"))   and not EO("disable_terminal_bongmath"):
                        if step == 0 and UNSAMPLE:
                            pass
                        elif full_iter == implicit_steps_full or not EO("disable_fully_explicit_bongmath_except_final"):
                            if sigma > 0.03:
                                BONGMATH_Y = SYNC_GUIDE_ACTIVE
                                x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step, step_sched,
                                                            BONGMATH_Y, y0_bongflow, noise_bongflow, eps_x_, eps_y_, data_x_, data_y_, LG)
                                #                            BONGMATH_Y, y0_bongflow, noise_bongflow, eps_x_, eps_y_, eps_x2y_, data_x_, LG)

                    diag_iter += 1

                    #progress_bar.update( round(1 / implicit_steps_total, 2) )
                    
                    #step_update = round(1 / implicit_steps_total, 2)
                    #progress_bar.update(float(f"{step_update:.2f}")) 

            x_next = x_[RK.rows - RK.multistep_stages - RK.row_offset + 1]
            x_next = NS.rebound_overshoot_step(x_0, x_next)
            
            if SYNC_GUIDE_ACTIVE:           # YT_NEXT UPDATE STEP --------------------------------------
                yt_next = yt_[RK.rows - RK.multistep_stages - RK.row_offset + 1]
                yt_next = NS.rebound_overshoot_step(yt_0, yt_next)
            
            eps = (x_0 - x_next) / (sigma - sigma_next)
            denoised = x_0 - sigma * eps
            
            x_0_prev = x_0.clone()

            x_means_per_step = x_next.mean(dim=(-2,-1), keepdim=True)

            if eta == 0.0:
                x = x_next
                if SYNC_GUIDE_ACTIVE:
                    yt_0 = yt_[0] = yt_next
                #elif LG.guide_mode.startswith("sync") and (LG.lgw[step_sched] != 0.0 or LG.lgw_inv[step_sched] != 0.0):
                #    noise_sync_new = NS.noise_sampler(sigma=sigma, sigma_next=sigma_next)
                #    x = x_next + sigma * eta * (noise_sync_new - noise_bongflow)
                #    noise_bongflow += eta * (noise_sync_new - noise_bongflow)
            elif not LG.guide_mode.startswith("flow") or (LG.lgw[step_sched] == 0 and LG.lgw[step+1] == 0   and   LG.lgw_inv[step_sched] == 0 and LG.lgw_inv[step+1] == 0):
                x = NS.swap_noise_step(x_0, x_next, mask=sde_mask)
                if SYNC_GUIDE_ACTIVE:
                    noise_bongflow_new = (x - x_next) / sigma_next + noise_bongflow
                    yt_next += sigma_next * (noise_bongflow_new - noise_bongflow)
                    x_0 += sigma * (noise_bongflow_new - noise_bongflow)
                    if not EO("disable_i_bong"):
                        for i_bong in range(len(NS.s_)):
                            x_[i_bong] += NS.s_[i_bong] * (noise_bongflow_new - noise_bongflow)
                    #x_[0] += sigma * (noise_bongflow_new - noise_bongflow)
                    yt_0 = yt_[0] = yt_next
                    noise_bongflow = noise_bongflow_new
            else:
                x = x_next
            
            if EO("keep_step_means"):
                x = x - x.mean(dim=(-2,-1), keepdim=True) + x_means_per_step

            
            callback_step = len(sigmas)-1 - step if sampler_mode == "unsample" else step
            preview_callback(x, eps, denoised, x_, eps_, data_, callback_step, sigma, sigma_next, callback, EO, preview_override=data_cached, FLOW_STOPPED=FLOW_STOPPED)
            
            h_prev = NS.h
            x_prev = x_0
            
            denoised_prev2 = denoised_prev
            denoised_prev  = denoised
            
            full_iter += 1
            
            if LG.lgw[step_sched] > 0 and step >= EO("guide_cutoff_start_step", 0) and cossim_counter < EO("guide_cutoff_max_iter", 10) and (EO("guide_cutoff") or EO("guide_min")):
                guide_cutoff = EO("guide_cutoff", 1.0)
                denoised_norm = data_[0] - data_[0].mean(dim=(-2,-1), keepdim=True)
                y0_norm       = LG.y0    - LG.y0   .mean(dim=(-2,-1), keepdim=True)
                y0_cossim     = get_cosine_similarity(denoised_norm, y0_norm)
                if y0_cossim > guide_cutoff and LG.lgw[step_sched] > EO("guide_cutoff_floor", 0.0):
                    if not EO("guide_cutoff_fast"):
                        LG.lgw[step_sched] *= EO("guide_cutoff_factor", 0.9)
                    else:
                        LG.lgw *= EO("guide_cutoff_factor", 0.9)
                    full_iter -= 1
                if y0_cossim < EO("guide_min", 0.0) and LG.lgw[step_sched] < EO("guide_min_ceiling", 1.0):
                    if not EO("guide_cutoff_fast"):
                        LG.lgw[step_sched] *= EO("guide_min_factor", 1.1)
                    else:
                        LG.lgw *= EO("guide_min_factor", 1.1)
                    full_iter -= 1
        
        if FLOW_STARTED and FLOW_STOPPED:
            data_prev_ = data_x_prev_
        if FLOW_STARTED and not FLOW_STOPPED:
            data_x_prev_[0] = data_cached       # data_cached is data_x from flow mode. this allows multistep to resume seamlessly.
            for ms in range(recycled_stages):
                data_x_prev_[recycled_stages - ms] = data_x_prev_[recycled_stages - ms - 1]

        #if LG.guide_mode.startswith("sync") and (LG.lgw[step_sched] != 0.0 or LG.lgw_inv[step_sched] != 0.0):
        #    data_prev_[0] = x_0 - sigma * eps_[0]
        #else:
        data_prev_[0] = data_[0]                # with flow mode, this will be the differentiated guide/"denoised"
        for ms in range(recycled_stages):
            data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]   # TODO: verify that this does not run on every substep...

        if SYNC_GUIDE_ACTIVE:
            data_prev_x_[0] = data_x      
            for ms in range(recycled_stages):
                data_prev_x_[recycled_stages - ms] = data_prev_x_[recycled_stages - ms - 1] 

            data_prev_y_[0] = data_y     
            for ms in range(recycled_stages):
                data_prev_y_[recycled_stages - ms] = data_prev_y_[recycled_stages - ms - 1] 
        
        rk_type = RK.swap_rk_type_at_step_or_threshold(x_0, data_prev_, NS, sigmas, step, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print)
        if step > rk_swap_step:
            implicit_steps_full = 0
            implicit_steps_diag = 0

        if EO("bong2m") or EO("bong3m"):
            denoised_data_prev2 = denoised_data_prev
            denoised_data_prev = data_[0]
        
        if SKIP_PSEUDO and not LG.guide_mode.startswith("flow"):
            if SKIP_PSEUDO_Y == "y0":
                LG.y0 = denoised
                LG.HAS_LATENT_GUIDE = True
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
            if sigmas.max() > NS.sigma_max:
                sigmas = sigmas / NS.sigma_max
        if d_noise_inv_start_step == step:
            sigmas = sigmas.clone() / d_noise_inv
            if sigmas.max() > NS.sigma_max:
                sigmas = sigmas / NS.sigma_max
        
        if LG.lgw[step_sched] > 0 and step >= EO("guide_step_cutoff_start_step", 0) and cossim_counter < EO("guide_step_cutoff_max_iter", 10) and (EO("guide_step_cutoff") or EO("guide_step_min")):
            guide_cutoff = EO("guide_step_cutoff", 1.0)
            eps_trash, data_trash = RK(x, sigma_next, x_0, sigma)
            denoised_norm = data_trash - data_trash.mean(dim=(-2,-1), keepdim=True)
            y0_norm       = LG.y0    - LG.y0   .mean(dim=(-2,-1), keepdim=True)
            y0_cossim     = get_cosine_similarity(denoised_norm, y0_norm)
            if y0_cossim > guide_cutoff and LG.lgw[step_sched] > EO("guide_step_cutoff_floor", 0.0):
                if not EO("guide_step_cutoff_fast"):
                    LG.lgw[step_sched] *= EO("guide_step_cutoff_factor", 0.9)
                else:
                    LG.lgw *= EO("guide_step_cutoff_factor", 0.9)
                step -= 1
                x_0 = x = x_[0] = x_0_orig.clone()
            if y0_cossim < EO("guide_step_min", 0.0) and LG.lgw[step_sched] < EO("guide_step_min_ceiling", 1.0):
                if not EO("guide_step_cutoff_fast"):
                    LG.lgw[step_sched] *= EO("guide_step_min_factor", 1.1)
                else:
                    LG.lgw *= EO("guide_step_min_factor", 1.1)
                step -= 1
                x_0 = x = x_[0] = x_0_orig.clone()
        # END SAMPLING LOOP ---------------------------------------------------------------------------------------------------

    #progress_bar.close()
    RK.update_transformer_options({'update_cross_attn':  None})
    if step == len(sigmas)-2 and sigmas[-1] == 0 and sigmas[-2] == NS.sigma_min and not INIT_SAMPLE_LOOP:
        eps, denoised = RK(x, NS.sigma_min, x, NS.sigma_min)
        x = denoised
        #progress_bar.update(1)

    eps      = eps     .to(model_device)
    denoised = denoised.to(model_device)
    x        = x       .to(model_device)
    
    progress_bar.close()

    if not (UNSAMPLE and sigmas[1] > sigmas[0]) and not EO("preview_last_step_always") and sigma is not None   and   not (FLOW_STARTED and not FLOW_STOPPED):
        callback_step = len(sigmas)-1 - step if sampler_mode == "unsample" else step
        preview_callback(x, eps, denoised, x_, eps_, data_, callback_step, sigma, sigma_next, callback, EO, preview_override=data_cached, FLOW_STOPPED=FLOW_STOPPED)

    if INIT_SAMPLE_LOOP:
        state_info_out = state_info
    else:
        if guides is not None and guides.get('guide_mode', "") == 'inversion':
            guide_inversion_y0     = state_info.get('guide_inversion_y0')
            guide_inversion_y0_inv = state_info.get('guide_inversion_y0_inv')
            
            if sampler_mode == "unsample" and guide_inversion_y0 is None:
                guide_inversion_y0     = LG.y0.clone()
            if sampler_mode == "unsample" and guide_inversion_y0_inv is None:
                guide_inversion_y0_inv = LG.y0_inv.clone()
                
            if sampler_mode in {"standard", "resample"} and guide_inversion_y0 is None:
                guide_inversion_y0 = NS.noise_sampler(sigma=NS.sigma_max, sigma_next=NS.sigma_min).to(x)
                guide_inversion_y0 = normalize_zscore(guide_inversion_y0, channelwise=True, inplace=True)
            if sampler_mode in {"standard", "resample"} and guide_inversion_y0_inv is None:
                guide_inversion_y0_inv = NS.noise_sampler(sigma=NS.sigma_max, sigma_next=NS.sigma_min).to(x)
                guide_inversion_y0_inv = normalize_zscore(guide_inversion_y0_inv, channelwise=True, inplace=True)
                
            state_info_out['guide_inversion_y0']     = guide_inversion_y0
            state_info_out['guide_inversion_y0_inv'] = guide_inversion_y0_inv

        state_info_out['raw_x']             = x.to('cpu')
        state_info_out['denoised']          = denoised.to('cpu')
        state_info_out['data_prev_']        = data_prev_.to('cpu')
        state_info_out['end_step']          = step
        state_info_out['sigma_next']        = sigma_next.clone()
        state_info_out['sigmas']            = sigmas_scheduled.clone()
        state_info_out['sampler_mode']      = sampler_mode
        state_info_out['last_rng']          = NS.noise_sampler .generator.get_state().clone()
        state_info_out['last_rng_substep']  = NS.noise_sampler2.generator.get_state().clone()
        state_info_out['completed']         = step == len(sigmas)-2 and sigmas[-1] == 0 and sigmas[-2] == NS.sigma_min
        state_info_out['FLOW_STARTED']      = FLOW_STARTED
        state_info_out['FLOW_STOPPED']      = FLOW_STOPPED
        state_info_out['noise_bongflow']    = noise_bongflow
        state_info_out['y0_bongflow']       = y0_bongflow
        state_info_out['y0_bongflow_orig']  = y0_bongflow_orig
        state_info_out['y0_standard_guide']       = y0_standard_guide
        state_info_out['y0_inv_standard_guide']  = y0_inv_standard_guide
        state_info_out['data_prev_y_']      = data_prev_y_
        state_info_out['data_prev_x_']      = data_prev_x_

        if FLOW_STARTED and not FLOW_STOPPED:
            state_info_out['y0']           = y0.to('cpu') 
            #state_info_out['y0_inv']       = y0_inv.to('cpu')       # TODO: implement this?
            state_info_out['data_cached']  = data_cached.to('cpu')
            state_info_out['data_x_prev_'] = data_x_prev_.to('cpu')

    return x

def noise_fn(x, sigma, sigma_next, noise_sampler, cossim_iter=1):
    
    noise  = normalize_zscore(noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
    cossim = get_pearson_similarity(x, noise)
    
    for i in range(cossim_iter):
        noise_new  = normalize_zscore(noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
        cossim_new = get_pearson_similarity(x, noise_new)
        
        if cossim_new > cossim:
            noise  = noise_new
            cossim = cossim_new
    
    return noise


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
                    preview_override : Optional[Tensor] = None,
                    FLOW_STOPPED : bool = False):

    if EO("eps_substep_preview"):
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
        
    elif preview_override is not None and FLOW_STOPPED == False:
        denoised_callback = preview_override
        
    else:
        denoised_callback = data_[0]
        
    callback({'x': x, 'i': step, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised_callback.to(torch.float32)}) if callback is not None else None
    
    return




def adain_patchwise_row_batch(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss = (gauss / gauss.sum()).to(dtype)
    kernel_2d = (gauss[:, None] * gauss[None, :])

    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    for i in range(H):
        c_row_patches = torch.stack([
            content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)  # [W, B, C, k, k]

        s_row_patches = torch.stack([
            style_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)

        w = weight.expand_as(c_row_patches[0])

        c_mean = (c_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        c_std  = ((c_row_patches - c_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
        s_mean = (s_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        s_std  = ((s_row_patches - s_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center:center+1, center:center+1]
        normed = (central - c_mean) / c_std
        stylized = normed * s_std + s_mean

        result[:, :, i, :] = stylized.squeeze(-1).squeeze(-1).permute(1, 2, 0)  # [B,C,W]

    return result

