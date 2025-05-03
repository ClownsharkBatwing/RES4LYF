import torch
from torch import Tensor
from tqdm.auto import trange
import gc
from typing import Optional, Callable, Tuple, Dict, Any, Union
import math
import copy

from comfy.model_sampling import EPS

from ..res4lyf              import RESplain
from ..helper               import ExtraOptions, FrameWeightsManager
from ..latents              import lagrange_interpolation, get_collinear, get_orthogonal, get_cosine_similarity, get_pearson_similarity, get_slerp_weight_for_cossim, get_slerp_ratio, slerp_tensor, normalize_zscore, compute_slerp_ratio_for_target, find_slerp_ratio_grid

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

        state_info                    : Optional[dict[str, Any]] = None,
        state_info_out                : Optional[dict[str, Any]] = None,
        
        rk_swap_type                  : str                = "",
        rk_swap_step                  : int                = MAX_STEPS,
        rk_swap_threshold             : float              = 0.0,
        rk_swap_print                 : bool               = False,
        
        steps_to_run                  : int                = -1,
        
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

    if 'raw_x' in state_info and sampler_mode in {"resample", "unsample"}:
        x = state_info['raw_x'].to(work_device) #clone()
        RESplain("Continuing from raw latent from previous sampler.", debug=False)
    

    
    start_step = 0
    if 'end_step' in state_info and (sampler_mode == "resample" or sampler_mode == "unsample"):

        if state_info['completed'] != True and state_info['end_step'] != 0 and state_info['end_step'] != -1 and state_info['end_step'] < len(state_info['sigmas'])-1 :   #incomplete run in previous sampler node
            
            if state_info['sampler_mode'] in {"standard","resample"} and sampler_mode == "unsample":
                sigmas = torch.flip(state_info['sigmas'], dims=[0])
                start_step = (len(sigmas)-1) - (state_info['end_step']) #-1) #removed -1 at the end here. correct?
                
            if state_info['sampler_mode'] == "unsample"              and sampler_mode == "resample":
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
        
    RK            = RK_Method_Beta.create(model, rk_type, noise_anchor, noise_boost_normalize, model_device=model_device, work_device=work_device, dtype=default_dtype, extra_options=extra_options)
    RK.extra_args = RK.init_cfg_channelwise(x, cfg_cw, **extra_args)
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = 0.0
    RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = 0.0

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
            
    state_info_sigma_next = state_info.get('sigma_next', -1)
    state_info_start_step = (sigmas == state_info_sigma_next).nonzero().flatten()
    if state_info_start_step.shape[0] > 0:
        start_step = state_info_start_step.item()
    
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
    

    # SETUP GUIDES
    LG = LatentGuide(model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options, device=work_device, dtype=default_dtype, frame_weights_mgr=frame_weights_mgr)
    x = LG.init_guides(x, RK.IMPLICIT, guides, NS.noise_sampler, batch_num)
    if (LG.mask != 1.0).any()   and  ((LG.y0 == 0).all() or (LG.y0_inv == 0).all()) : #  and   not LG.guide_mode.startswith("flow"):  # (LG.y0.sum() == 0 or LG.y0_inv.sum() == 0):
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
    data_cached = None
        
    if EO("pseudo_mix_strength"):
        orig_y0     = LG.y0.clone()
        orig_y0_inv = LG.y0_inv.clone()
        
    #gc.collect()

    VE_MODEL = isinstance(model.inner_model.inner_model.model_sampling, EPS)
    BASE_STARTED = False
    INV_STARTED  = False
    FLOW_STARTED = False
    FLOW_STOPPED = False
    FLOW_RESUMED = False
    if state_info.get('FLOW_STARTED', False) and not state_info.get('FLOW_STOPPED', False):
        FLOW_RESUMED = True
        y0 = state_info['y0'].to(work_device) 
        data_cached = state_info['data_cached'].to(work_device) 
        data_x_prev_ = state_info['data_x_prev_'].to(work_device) 
    if EO("flow_use_init_noise"):
        x_init = x.clone()

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
    sigmas_scheduled = sigmas.clone() # store for return in state_info_out
    
    if EO("sigma_restarts"):
        sigma_restarts = 1 + EO("sigma_restarts", 0)
        sigmas = sigmas[step:num_steps+1].repeat(sigma_restarts)
        step = 0
        num_steps = 2 * sigma_restarts - 1
    
    while step < num_steps:
        sigma, sigma_next = sigmas[step], sigmas[step+1]
        
        if LG.HAS_LATENT_GUIDE_ADAIN:
            if LG.lgw_adain[step] == 0.0:
                RK.update_transformer_options({'y0_adain': None})
                RK.update_transformer_options({'blocks_adain': {}})
            else:
                RK.update_transformer_options({'y0_adain': LG.y0_adain.clone()})
                if 'blocks_adain_mmdit' in guides:
                    blocks_adain = {
                        "double_weights": [val * LG.lgw_adain[step] for val in guides['blocks_adain_mmdit']['double_weights']],
                        "single_weights": [val * LG.lgw_adain[step] for val in guides['blocks_adain_mmdit']['single_weights']],
                        "double_blocks" : guides['blocks_adain_mmdit']['double_blocks'],
                        "single_blocks" : guides['blocks_adain_mmdit']['single_blocks'],
                    }
                RK.update_transformer_options({'blocks_adain': blocks_adain})
        
        if LG.HAS_LATENT_GUIDE_ATTNINJ:
            if LG.lgw_attninj[step] == 0.0:
                RK.update_transformer_options({'y0_attninj': None})
                RK.update_transformer_options({'blocks_attninj'    : {}})
                RK.update_transformer_options({'blocks_attninj_qkv': {}})
            else:
                RK.update_transformer_options({'y0_attninj': LG.y0_attninj.clone()})
                if 'blocks_attninj_mmdit' in guides:
                    blocks_attninj = {
                        "double_weights": [val * LG.lgw_attninj[step] for val in guides['blocks_attninj_mmdit']['double_weights']],
                        "single_weights": [val * LG.lgw_attninj[step] for val in guides['blocks_attninj_mmdit']['single_weights']],
                        "double_blocks" : guides['blocks_attninj_mmdit']['double_blocks'],
                        "single_blocks" : guides['blocks_attninj_mmdit']['single_blocks'],
                    }
                RK.update_transformer_options({'blocks_attninj'    : blocks_attninj})
                RK.update_transformer_options({'blocks_attninj_qkv': guides['blocks_attninj_qkv']})

        if LG.HAS_LATENT_GUIDE_STYLE_POS:
            if LG.lgw_style_pos[step] == 0.0:
                RK.update_transformer_options({'y0_style_pos':        None})
                RK.update_transformer_options({'y0_style_pos_weight': 0.0})
                RK.update_transformer_options({'y0_style_pos_synweight': 0.0})
                RK.update_transformer_options({'y0_style_pos_mask': None})
            else:
                RK.update_transformer_options({'y0_style_pos':        LG.y0_style_pos.clone()})
                RK.update_transformer_options({'y0_style_pos_weight': LG.lgw_style_pos[step]})
                RK.update_transformer_options({'y0_style_pos_synweight': guides['synweight_style_pos']})
                RK.update_transformer_options({'y0_style_pos_mask': LG.mask_style_pos})

        if LG.HAS_LATENT_GUIDE_STYLE_NEG:
            if LG.lgw_style_neg[step] == 0.0:
                RK.update_transformer_options({'y0_style_neg':        None})
                RK.update_transformer_options({'y0_style_neg_weight': 0.0})
                RK.update_transformer_options({'y0_style_neg_synweight': 0.0})
                RK.update_transformer_options({'y0_style_neg_mask': None})
            else:
                RK.update_transformer_options({'y0_style_neg':        LG.y0_style_neg.clone()})
                RK.update_transformer_options({'y0_style_neg_weight': LG.lgw_style_neg[step]})
                RK.update_transformer_options({'y0_style_neg_synweight': guides['synweight_style_neg']})
                RK.update_transformer_options({'y0_style_neg_mask': LG.mask_style_neg})

        if AttnMask_neg is not None:
            RK.update_transformer_options({'regional_conditioning_weight_neg': RegParam_neg.weights[step]})
            RK.update_transformer_options({'regional_conditioning_floor_neg':  RegParam_neg.floors[step]})

        if AttnMask is not None:
            RK.update_transformer_options({'regional_conditioning_weight': RegParam.weights[step]})
            RK.update_transformer_options({'regional_conditioning_floor':  RegParam.floors[step]})

        elif regional_conditioning_weights is not None:
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_weight'] = regional_conditioning_weights[step]
            RK.extra_args['model_options']['transformer_options']['regional_conditioning_floor']  = regional_conditioning_floors [step]
        
        epsilon_scale        = float(epsilon_scales [step]) if epsilon_scales        is not None else None
        eta                  = etas                 [step]  if etas                  is not None else eta
        eta_substep          = etas_substep         [step]  if etas_substep          is not None else eta_substep
        s_noise              = s_noises             [step]  if s_noises              is not None else s_noise
        s_noise_substep      = s_noises_substep     [step]  if s_noises_substep      is not None else s_noise_substep
        noise_scaling_eta    = noise_scaling_etas   [step]  if noise_scaling_etas    is not None else noise_scaling_eta
        noise_scaling_weight = noise_scaling_weights[step]  if noise_scaling_weights is not None else noise_scaling_weight
        
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
                    data_prev_ = state_info['data_prev_'].clone().to(dtype=default_dtype, device=work_device)
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
        x_0      = x_[0].clone()
        if EO("guide_step_cutoff") or EO("guide_step_min"):
            x_0_orig = x_0.clone()
        
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
        #for full_iter in range(implicit_steps_full + 1):
        cossim_counter = 0
        adaptive_lgw = LG.lgw.clone()
        full_iter = 0
        while full_iter < implicit_steps_full+1:

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
                #for diag_iter in range(implicit_steps_diag+1):
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
                                    eps_, x_      = LG.process_guides_substep(x_0, x_, eps_,      data_, row, step, sigma, sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                                    eps_prev_, x_ = LG.process_guides_substep(x_0, x_, eps_prev_, data_, row, step, sigma, sigma_next, NS.sigma_down, NS.s_, epsilon_scale, RK)
                                if row == 0 and (EO("implicit_lagrange_init")  or   EO("radaucycle")):
                                    pass
                                else:
                                    x_[row+RK.row_offset] = x_0 + NS.h_new * RK.zum(row+RK.row_offset, eps_, eps_prev_)
                                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                                    if row > 0:
                                        if not LG.guide_mode.startswith("flow") or (LG.lgw[step] == 0 and LG.lgw[step+1] == 0   and   LG.lgw_inv[step] == 0 and LG.lgw_inv[step+1] == 0):
                                            x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset])
                                        if BONGMATH and step < sigmas.shape[0]-1 and not EO("disable_implicit_prebong"):
                                            x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step)     # TRY WITH h_new ??
                                    x_tmp = x_[row+RK.row_offset]


                        lying_eps_row_factor = 1.0
                        # MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL
                        if RK.IMPLICIT   and   row == 0   and   (EO("implicit_lazy_recycle_first_model_call_at_start")   or   EO("radaucycle")  or RK.C[0] == 0.0):
                            pass
                        else: 
                            if s_tmp == 0:
                                break
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "pre") # will this do anything? not x_tmp

                            if noise_scaling_type == "model_alpha" and noise_scaling_weight != 0 and noise_scaling_eta > 0:
                                s_tmp = s_tmp + noise_scaling_weight * (s_tmp * lying_alpha_ratio   -   s_tmp)
                                
                            if noise_scaling_type == "model"       and noise_scaling_weight != 0 and noise_scaling_eta > 0:
                                s_tmp = lying_s_[row]
                                if RK.multistep_stages > 0:
                                    s_tmp = lying_sd

                            if LG.guide_mode.startswith("flow") and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0) and not FLOW_STOPPED and not EO("flow_sync") :
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
                                            y0  = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv)) * denoised   +   LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask * LG.y0   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv * LG.y0_inv
                                        else:
                                            y0  = (1 - (lgw_mask_ + lgw_mask_inv_)) * denoised   +   lgw_mask_ * LG.y0   +   lgw_mask_inv_ * LG.y0_inv
                                        yx0 = y0.clone()
                                        
                                        if EO("flow_slerp"):
                                            if EO("flow_manual_masks"):
                                                y0_inv                 = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv)) * denoised   +   LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask * LG.y0_inv   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv * LG.y0
                                            else:
                                                y0_inv  = (1 - (lgw_mask_ + lgw_mask_inv_)) * denoised   +   lgw_mask_ * LG.y0_inv   +   lgw_mask_inv_ * LG.y0
                                            flow_slerp_guide_ratio = EO("flow_slerp_guide_ratio", 0.5)
                                            y_slerp                = slerp_tensor(flow_slerp_guide_ratio, y0, y0_inv)
                                            yx0                    = y_slerp.clone()
                                
                                else:
                                    yx0_prev = data_cached
                                    if EO("flow_manual_masks"):
                                        yx0      = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv)) * yx0_prev   +   LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask * x_tmp   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv * x_tmp
                                    else:
                                        yx0  = (1 - (lgw_mask_ + lgw_mask_inv_)) * yx0_prev   +   (lgw_mask_ + lgw_mask_inv_) * x_tmp

                                    if not EO("flow_static_guides"):
                                        if EO("flow_manual_masks"):
                                            y0       = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv)) * yx0_prev   +   LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask * LG.y0   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv * LG.y0_inv
                                        else:
                                            y0  = (1 - (lgw_mask_ + lgw_mask_inv_)) * yx0_prev   +   lgw_mask_ * LG.y0   +   lgw_mask_inv_ * LG.y0_inv
                                            
                                        if EO("flow_slerp"):
                                            if EO("flow_manual_masks"):
                                                y0_inv  = (1 - (LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask + LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv)) * yx0_prev   +   LG.HAS_LATENT_GUIDE * LG.lgw[step] * LG.mask * LG.y0_inv   +   LG.HAS_LATENT_GUIDE_INV * LG.lgw_inv[step] * LG.mask_inv * LG.y0
                                            else:
                                                y0_inv  = (1 - (lgw_mask_ + lgw_mask_inv_)) * yx0_prev   +   lgw_mask_ * LG.y0_inv   +   lgw_mask_inv_ * LG.y0

                                y0_orig = y0.clone()
                                #yx0_orig = yx0.clone()
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
                                
                                if EO("flow_use_init_noise"):
                                    noise = x_init.clone()
                                else:
                                    noise = noise_fn(y0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                if VE_MODEL:
                                    yt        = y0 + s_tmp * noise
                                else:
                                    yt        = (NS.sigma_max-s_tmp) * y0 + (s_tmp/NS.sigma_max) * noise
                                if not EO("flow_disable_doublenoise_y0"):
                                    noise = noise_fn(y0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                    
                                if VE_MODEL:
                                    y0_noised = y0 + sigma * noise
                                else:
                                    y0_noised = (NS.sigma_max-sigma) * y0 + sigma * noise
                                
                                if EO("flow_slerp"):
                                    noise = noise_fn(y0_inv, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                    yt_inv        = (NS.sigma_max-s_tmp) * y0_inv + (s_tmp/NS.sigma_max) * noise
                                    if not EO("flow_disable_doublenoise_y0_inv"):
                                        noise = noise_fn(y0_inv, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                    y0_noised_inv = (NS.sigma_max-sigma) * y0_inv + sigma * noise
                                
                                if not EO("flow_disable_separatenoise_x_0"):
                                    noise = noise_fn(yx0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                if EO("flow_slerp"):
                                    xt         = yx0 + (s_tmp/NS.sigma_max) * (noise - y_slerp)
                                    if not EO("flow_disable_doublenoise_x_0"):
                                        noise = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                    x_0_noised = x_0 + sigma * (noise - y_slerp)
                                else:
                                    if VE_MODEL:
                                        xt         = yx0 + (s_tmp) * yx0 + (s_tmp) * (noise - y0)
                                    else:
                                        xt         = yx0 + (s_tmp/NS.sigma_max) * (noise - y0)
                                    if not EO("flow_disable_doublenoise_x_0"):
                                        noise = noise_fn(x_0, sigma, sigma_next, NS.noise_sampler, flow_cossim_iter) # normalize_zscore(NS.noise_sampler(sigma=sigma, sigma_next=sigma_next), channelwise=True, inplace=True)
                                    if VE_MODEL:
                                        x_0_noised = x_0 + (sigma) * x_0 + (sigma) * (noise - y0)
                                    else:
                                        x_0_noised = x_0 + (sigma/NS.sigma_max) * (noise - y0)
                                
                                eps_y, data_y = RK(yt, s_tmp, y0_noised,  sigma, transformer_options={'latent_type': 'yt'})
                                eps_x, data_x = RK(xt, s_tmp, x_0_noised, sigma, transformer_options={'latent_type': 'xt'})

                                if EO("flow_slerp"):
                                    eps_y_inv, data_y_inv = RK(yt_inv, s_tmp, y0_noised_inv, sigma, transformer_options={'latent_type': 'yt_inv'})
                                
                                if LG.lgw[step+1] == 0 and LG.lgw_inv[step+1] == 0:
                                    eps_ [row]       = eps_x
                                    data_[row]       = data_x
                                    x_   [row] = x_0 = xt 
                                
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
                                        

                                        
                                        data_[row] = x_0 - sigma * eps_yx_lin
                                    
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
                                        #eps_y_slerp       = slerp_tensor(flow_slerp_ratio2, eps_y_alt, eps_y_alt_inv)
                                        #data_y_slerp      = slerp_tensor(flow_slerp_ratio2, data_y,    data_y_inv)
                                        
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
                                    x_tmp += LG.lgw[step] * LG.mask * (NS.sigma_max - s_tmp) * (LG.y0 - denoised)     +     LG.lgw_inv[step] * LG.mask_inv * (NS.sigma_max - s_tmp) * (LG.y0_inv - denoised)
                                    eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                            
                            # MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL MODEL CALL
                            
                            if (not LG.guide_mode.startswith("flow"))   or   FLOW_STOPPED   or    (LG.guide_mode.startswith("flow") and LG.lgw[step] == 0 and LG.lgw_inv[step] == 0):
                                if LG.guide_mode.startswith("lure") and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0):
                                    eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma, transformer_options={'latent_type': 'yt'})
                                else:
                                    if EO("t_pad"):
                                        x_tmp_pad = torch.cat([x_tmp, x_tmp[...,0:1,:,:]], dim=-3)
                                        x_0_pad   = torch.cat([x_0,   x_0  [...,0:1,:,:]], dim=-3)
                                        if EO("t_pad_flip"):
                                            x_tmp_pad    = torch.flip(x_tmp_pad, dims=[-3])
                                        eps_row_pad, data_row_pad = RK(x_tmp_pad, s_tmp, x_0_pad, sigma)
                                        if EO("t_pad_flip"):
                                            eps_row_pad  = torch.flip(eps_row_pad,  dims=[-3])    
                                            data_row_pad = torch.flip(data_row_pad, dims=[-3])    
                                        eps_ [row] = eps_row_pad [...,:-1,:,:]
                                        data_[row] = data_row_pad[...,:-1,:,:]
                                    if EO("t_prepad"):
                                        x_tmp_pad = torch.cat([x_tmp[...,-1:,:,:], x_tmp], dim=-3)
                                        x_0_pad   = torch.cat([x_0  [...,-1:,:,:], x_0  ], dim=-3)
                                        if EO("t_pad_flip"):
                                            x_tmp_pad    = torch.flip(x_tmp_pad, dims=[-3])
                                        eps_row_pad, data_row_pad = RK(x_tmp_pad, s_tmp, x_0_pad, sigma)
                                        if EO("t_pad_flip"):
                                            eps_row_pad  = torch.flip(eps_row_pad,  dims=[-3])    
                                            data_row_pad = torch.flip(data_row_pad, dims=[-3])    
                                        eps_ [row] = eps_row_pad [...,1:,:,:]
                                        data_[row] = data_row_pad[...,1:,:,:]
                                        
                                    if EO("t_prepostpad"):
                                        
                                        x_tmp_pad = torch.cat([x_tmp[...,-1:,:,:], x_tmp], dim=-3)
                                        x_0_pad   = torch.cat([x_0  [...,-1:,:,:], x_0  ], dim=-3)
                                        
                                        x_tmp_pad = torch.cat([x_tmp_pad, x_tmp[...,0:1,:,:]], dim=-3)
                                        x_0_pad   = torch.cat([x_0_pad,   x_0  [...,0:1,:,:]], dim=-3)
                                        
                                        if EO("t_pad_flip"):
                                            x_tmp_pad    = torch.flip(x_tmp_pad, dims=[-3])
                                        eps_row_pad, data_row_pad = RK(x_tmp_pad, s_tmp, x_0_pad, sigma)
                                        if EO("t_pad_flip"):
                                            eps_row_pad  = torch.flip(eps_row_pad,  dims=[-3])    
                                            data_row_pad = torch.flip(data_row_pad, dims=[-3])    
                                        eps_ [row] = eps_row_pad [...,1:-1,:,:]
                                        data_[row] = data_row_pad[...,1:-1,:,:]
                                        
                                    elif EO("t_pad2"):
                                        x_tmp_pad = torch.cat([x_tmp, x_tmp[...,1:2,:,:]], dim=-3)
                                        x_0_pad   = torch.cat([x_0,   x_0  [...,1:2,:,:]], dim=-3)
                                        if EO("t_pad2_flip"):
                                            x_tmp_pad    = torch.flip(x_tmp_pad, dims=[-3])
                                        eps_row_pad, data_row_pad = RK(x_tmp_pad, s_tmp, x_0_pad, sigma)
                                        if EO("t_pad2_flip"):
                                            eps_row_pad  = torch.flip(eps_row_pad,  dims=[-3])    
                                            data_row_pad = torch.flip(data_row_pad, dims=[-3])    
                                        eps_ [row] = eps_row_pad [...,:-1,:,:]
                                        data_[row] = data_row_pad[...,:-1,:,:]
                                    elif EO("t_pad3"):
                                        x_tmp_pad = torch.cat([x_tmp, x_tmp], dim=-3)
                                        x_0_pad   = torch.cat([x_0,   x_0  ], dim=-3)
                                        if EO("t_pad3_flip"):
                                            x_tmp_pad    = torch.flip(x_tmp_pad, dims=[-3])
                                        eps_row_pad, data_row_pad = RK(x_tmp_pad, s_tmp, x_0_pad, sigma)
                                        if EO("t_pad3_flip"):
                                            eps_row_pad  = torch.flip(eps_row_pad,  dims=[-3])    
                                            data_row_pad = torch.flip(data_row_pad, dims=[-3])    
                                        eps_ [row] = eps_row_pad [...,:eps_[row].shape[-3],:,:]
                                        data_[row] = data_row_pad[...,:data_[row].shape[-3],:,:]
                                    elif EO("t_replace"):
                                        x_tmp[...,-1:,:,:] = x_tmp[...,0:1,:,:]
                                        x_0[...,-1:,:,:]   = x_0[...,0:1,:,:]
                                        eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                                    elif EO("t_replace_first_with_last"):
                                        x_tmp[...,0:1,:,:] = x_tmp[...,-1:,:,:]
                                        x_0[...,0:1,:,:] = x_0[...,-1:,:,:] 
                                        eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                            
                                    elif EO("t_rotato"):
                                        if diag_iter % 2 == 0:
                                            eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                                        else:
                                            x_tmp_roll = torch.roll(x_tmp, dims=-3, shifts=-1)
                                            x_0_roll = torch.roll(x_0, dims=-3, shifts=-1)
                                            eps_row_roll, data_row_roll = RK(x_tmp_roll, s_tmp, x_0_roll, sigma)
                                            eps_row_roll = torch.roll(eps_row_roll, dims=-3, shifts=1)
                                            data_row_roll = torch.roll(data_row_roll, dims=-3, shifts=1)
                                            eps_[row] = eps_row_roll
                                            data_[row] = data_row_roll
                                            
                                    elif EO("t_rotato_pos"):
                                        if diag_iter % 2 == 0:
                                            eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                                        else:
                                            x_tmp_roll = torch.roll(x_tmp, dims=-3, shifts=1)
                                            x_0_roll = torch.roll(x_0, dims=-3, shifts=1)
                                            eps_row_roll, data_row_roll = RK(x_tmp_roll, s_tmp, x_0_roll, sigma)
                                            eps_row_roll = torch.roll(eps_row_roll, dims=-3, shifts=-1)
                                            data_row_roll = torch.roll(data_row_roll, dims=-3, shifts=-1)
                                            eps_[row] = eps_row_roll
                                            data_[row] = data_row_roll
                                    
                                    elif EO("t_stitch"):
                                        t_blk0_end = EO("t_blk0_end", 11)
                                        t_blk1_start = EO("t_blk1_start", 10)
                                        
                                        eps_row_blk0, data_row_blk0 = RK(x_tmp[...,:t_blk0_end   ,:,:], s_tmp, x_0[...,:t_blk0_end   ,:,:], sigma)
                                        eps_row_blk1, data_row_blk1 = RK(x_tmp[..., t_blk1_start:,:,:], s_tmp, x_0[... ,t_blk1_start:,:,:], sigma)
                                        
                                        eps_ [row] = torch.cat([ eps_row_blk0[...,:t_blk0_end,:,:] ,  eps_row_blk1[...,t_blk0_end-t_blk1_start:,:,:]], dim=-3)
                                        data_[row] = torch.cat([data_row_blk0[...,:t_blk0_end,:,:] , data_row_blk1[...,t_blk0_end-t_blk1_start:,:,:]], dim=-3)
                                        
                                    elif EO("t_stitch_end"):
                                        t_blk0_end = EO("t_blk0_end", 11)
                                        t_blk1_start = EO("t_blk1_start", 10)
                                        
                                        eps_row_blk0, data_row_blk0 = RK(x_tmp[...,:t_blk0_end   ,:,:], s_tmp, x_0[...,:t_blk0_end   ,:,:], sigma)
                                        eps_row_blk1, data_row_blk1 = RK(x_tmp[..., t_blk1_start:,:,:], s_tmp, x_0[... ,t_blk1_start:,:,:], sigma)
                                        
                                        eps_ [row] = torch.cat([ eps_row_blk0[...,:t_blk1_start,:,:] ,  eps_row_blk1], dim=-3)
                                        data_[row] = torch.cat([data_row_blk0[...,:t_blk1_start,:,:] , data_row_blk1], dim=-3)
                                        
                                        
                                    elif EO("t_stitch_avg"):
                                        t_blk0_end = EO("t_blk0_end", 11)
                                        t_blk1_start = EO("t_blk1_start", 10)
                                        
                                        eps_row_blk0, data_row_blk0 = RK(x_tmp[...,:t_blk0_end   ,:,:], s_tmp, x_0[...,:t_blk0_end   ,:,:], sigma)
                                        eps_row_blk1, data_row_blk1 = RK(x_tmp[..., t_blk1_start:,:,:], s_tmp, x_0[... ,t_blk1_start:,:,:], sigma)
                                        
                                        eps_ [row] = torch.cat([
                                            eps_row_blk0[...,:t_blk1_start,:,:] ,  
                                            
                                            (eps_row_blk0[...,t_blk1_start:,:,:]  +  eps_row_blk1[...,:t_blk0_end-t_blk1_start,:,:]) / 2,
                                            
                                            eps_row_blk1[...,t_blk0_end-t_blk1_start:,:,:]
                                            ], dim=-3)
                                        
                                        data_[row] = torch.cat([
                                            data_row_blk0[...,:t_blk1_start,:,:] , 
                                            
                                            (data_row_blk0[...,t_blk1_start:,:,:] + data_row_blk1[...,:t_blk0_end-t_blk1_start,:,:]) / 2,

                                            data_row_blk1[...,t_blk0_end-t_blk1_start:,:,:]
                                            ], dim=-3)
                                        
                                    elif EO("t_stitch_slerp"):
                                        t_blk0_end = EO("t_blk0_end", 11)
                                        t_blk1_start = EO("t_blk1_start", 10)
                                        
                                        eps_row_blk0, data_row_blk0 = RK(x_tmp[...,:t_blk0_end   ,:,:], s_tmp, x_0[...,:t_blk0_end   ,:,:], sigma)
                                        eps_row_blk1, data_row_blk1 = RK(x_tmp[..., t_blk1_start:,:,:], s_tmp, x_0[... ,t_blk1_start:,:,:], sigma)
                                        
                                        eps_ [row] = torch.cat([
                                            eps_row_blk0[...,:t_blk1_start,:,:] ,  
                                            
                                            slerp_tensor(0.5, eps_row_blk0[...,t_blk1_start:,:,:], eps_row_blk1[...,:t_blk0_end-t_blk1_start,:,:]),
                                            
                                            eps_row_blk1[...,t_blk0_end-t_blk1_start:,:,:]
                                            ], dim=-3)
                                        
                                        data_[row] = torch.cat([
                                            data_row_blk0[...,:t_blk1_start,:,:] , 
                                            
                                            slerp_tensor(0.5, data_row_blk0[...,t_blk1_start:,:,:], data_row_blk1[...,:t_blk0_end-t_blk1_start,:,:]),

                                            data_row_blk1[...,t_blk0_end-t_blk1_start:,:,:]
                                            ], dim=-3)
                                    
                                    
                                    elif not EO("t_roll"):
                                        eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                                    elif step % 2 == 0:
                                        eps_[row], data_[row] = RK(x_tmp, s_tmp, x_0, sigma)
                                    else:
                                        #import einops
                                        #b, c, t, h, w = x.shape
                                        #patches = einops.rearrange(x, "b c (t h w) c -> b t (h w) c", h=30, w=30)
                                        x_tmp_roll = torch.roll(x_tmp, dims=-3, shifts=-1)
                                        x_0_roll = torch.roll(x_0, dims=-3, shifts=-1)
                                        eps_row_roll, data_row_roll = RK(x_tmp_roll, s_tmp, x_0_roll, sigma)
                                        eps_row_roll = torch.roll(eps_row_roll, dims=-3, shifts=1)
                                        data_row_roll = torch.roll(data_row_roll, dims=-3, shifts=1)
                                        eps_[row] = eps_row_roll
                                        data_[row] = data_row_roll
                                        #eps_[row] = (eps_[row] + eps_row_roll) / 2
                                        #data_[row] = (data_[row] + data_row_roll) / 2
                                        
                                        


                            if LG.guide_mode.startswith("lure") and (LG.lgw[step] > 0 or LG.lgw_inv[step] > 0):
                                x_tmp = LG.process_guides_data_substep(x_tmp, data_[row], step, s_tmp)
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
                        

                        if LG.y0_mean is not None and LG.y0_mean.sum() != 0.0:
                            eps_row_mean = eps_[row] - eps_[row].mean(dim=(-2,-1), keepdim=True) + (LG.y0_mean - x_0).mean(dim=(-2,-1), keepdim=True)
                            
                            if LG.mask_mean is not None:
                                eps_row_mean = LG.mask_mean * eps_row_mean + (1 - LG.mask_mean) * eps_[row]
                            
                            eps_[row] = eps_[row] + LG.lgw_mean[step] * (eps_row_mean - eps_[row])
                            

                        if (full_iter == 0 and diag_iter == 0)   or   EO("newton_iter_post_use_on_implicit_steps"):
                            x_, eps_ = RK.newton_iter(x_0, x_, eps_, eps_prev_, data_, NS.s_, row, NS.h, sigmas, step, "post")


                    # UPDATE
                    x_ = RK.update_substep(x_0, x_, eps_, eps_prev_, row, RK.row_offset, NS.h_new, NS.h_new_orig, lying_eps_row_factor=lying_eps_row_factor)   #modifies eps_[row] if lying_eps_row_factor != 1.0
                    
                    x_[row+RK.row_offset] = NS.rebound_overshoot_substep(x_0, x_[row+RK.row_offset])
                    
                    if not RK.IMPLICIT and NS.noise_mode_sde_substep != "hard_sq":
                        if EO("dynamic_sde_mask") and denoised_prev is not None and denoised_prev.sum() != 0:
                            sde_mask = torch.norm(data_[row] - denoised_prev, p=2, dim=0, keepdim=True)
                            sde_mask = sde_mask / sde_mask.max()
                        if EO("dynamic_eps_mask"):
                            sde_mask = eps_[row] ** 2
                            sde_mask = sde_mask / sde_mask.max()
                        if EO("dynamic_scaled_eps_mask"):
                            sde_mask = eps_[row] ** 2
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                        if EO("dynamic_inv_eps_mask"):
                            sde_mask = eps_[row] ** 2
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                        if EO("dynamic_scaled_inv_eps_mask"):
                            sde_mask = eps_[row] ** 2
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                        if EO("dynamic_scaled_mean_inv_eps_mask"):
                            sde_mask = eps_[row] ** 2
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                            dyn_scale = EO("dynamic_scaled_mean_inv_eps_mask", 2.0)
                            sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale

                            #sde_mask = (1 + sde_mask) / 2
                        
                        if EO("dynamic_scaled_mean_inv_eps_mask_abs"):
                            sde_mask = eps_[row].abs()
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                            dyn_scale = EO("dynamic_scaled_mean_inv_eps_mask", 2.0)
                            sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale
                            
                        if EO("dynamic_scaled_mean_batch_inv_eps_mask"):
                            sde_mask = eps_[row] ** 2
                            sde_mask = sde_mask - sde_mask.mean(dim=(-2,-1), keepdim=True)
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                            dyn_scale = EO("dynamic_scaled_mean_inv_eps_mask", 2.0)
                            sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale

                            #sde_mask = (1 + sde_mask) / 2
                            
                        if EO("dynamic_scaled_mean_batch2_inv_eps_mask"):
                            sde_mask = eps_[row].mean(dim=-3, keepdim=True) ** 2
                            sde_mask = sde_mask.expand_as(x)
                            sde_mask = sde_mask - sde_mask.mean(dim=(-2,-1), keepdim=True)
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                            dyn_scale = EO("dynamic_scaled_mean_inv_eps_mask", 2.0)
                            sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale

                        if EO("dynamic_tiled_mean_inv_eps_mask"):
                            mean_tile_size=EO("dynamic_tiled_mean_inv_eps_mask", 8)
                            from einops import rearrange
                            eps_row_tiled  = rearrange(eps_[row], "b c (h t1) (w t2) -> (t1 t2) b c h w", t1=mean_tile_size, t2=mean_tile_size)
                            
                            sde_mask = eps_row_tiled ** 2
                            sde_mask = sde_mask - sde_mask.min()
                            sde_mask = sde_mask / sde_mask.max()
                            sde_mask = 1-sde_mask
                            sde_mask = (1 + sde_mask) / 2
                            
                            sde_mask[:,...] = sde_mask.mean(dim=(-2,-1), keepdim=True)
                            
                            sde_mask = rearrange(sde_mask,     "(t1 t2) b c h w -> b c (h t1) (w t2)", t1=mean_tile_size, t2=mean_tile_size)
                            
                            
                        x_means_per_substep = x_[row+RK.row_offset].mean(dim=(-2,-1), keepdim=True)
                                                
                        if EO("sde_mask_floor"):
                            sde_mask_floor = EO("sde_mask_floor", 0.0)
                            sde_mask_ceiling = EO("sde_mask_ceiling", 1.0)
                            sde_mask = ((sde_mask - sde_mask.min()) * (sde_mask_floor - sde_mask_ceiling)) / (sde_mask.max() - sde_mask.min()) + sde_mask_ceiling     
                            
                        if not LG.guide_mode.startswith("flow") or (LG.lgw[step] == 0 and LG.lgw[step+1] == 0   and   LG.lgw_inv[step] == 0 and LG.lgw_inv[step+1] == 0):
                            x_[row+RK.row_offset] = NS.swap_noise_substep(x_0, x_[row+RK.row_offset], mask=sde_mask, guide=LG.y0)
                        elif LG.guide_mode.startswith("flow"):
                            pass
                            #x_[row+RK.row_offset] = NS.swap_noise_inv_substep(x_0, x_[row+RK.row_offset], eta_substep, row, row+RK.row_offset+RK.multistep_stages, mask=sde_mask, guide=LG.y0)
                        if EO("swap_noise_substep_update_eps"):
                            eps_[row+RK.row_offset] = RK.get_epsilon(x_0, x_[row+RK.row_offset], data_[row+RK.row_offset], sigma, NS.s_[row+RK.row_offset])
                        
                        if EO("keep_substep_means"):
                            x_[row+RK.row_offset] = x_[row+RK.row_offset] - x_[row+RK.row_offset].mean(dim=(-2,-1), keepdim=True) + x_means_per_substep



                    if not LG.guide_mode.startswith("lure"):
                        x_[row+RK.row_offset] = LG.process_guides_data_substep(x_[row+RK.row_offset], data_[row], step, NS.s_[row])

                    if BONGMATH and NS.s_[row] > RK.sigma_min and NS.h < RK.sigma_max/2   and   (diag_iter == implicit_steps_diag or EO("enable_diag_explicit_bongmath_all"))   and not EO("disable_terminal_bongmath"):
                        if step == 0 and UNSAMPLE:
                            pass
                        elif full_iter == implicit_steps_full or not EO("disable_fully_explicit_bongmath_except_final"):
                            x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step)

                    diag_iter += 1

                    #progress_bar.update( round(1 / implicit_steps_total, 2) )
                    
                    #step_update = round(1 / implicit_steps_total, 2)
                    #progress_bar.update(float(f"{step_update:.2f}")) 

            x_next = x_[RK.rows - RK.multistep_stages - RK.row_offset + 1]
            x_next = NS.rebound_overshoot_step(x_0, x_next)
            
            eps = (x_0 - x_next) / (sigma - sigma_next)
            denoised = x_0 - sigma * eps
            
            #x_next = LG.process_guides_poststep(x_next, denoised, eps, step)
            
            if EO("kill_step_sde_mask"):
                sde_mask = None
                
            if EO("dynamic_step_scaled_mean_inv_eps_mask"):
                sde_mask = eps ** 2
                sde_mask = sde_mask - sde_mask.min()
                sde_mask = sde_mask / sde_mask.max()
                sde_mask = 1-sde_mask
                dyn_scale = EO("dynamic_step_scaled_mean_inv_eps_mask", 2.0)
                sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale
                
            if EO("dynamic_expstep_scaled_mean_inv_eps_mask"):
                eps_exp = denoised - x_0
                sde_mask = eps_exp ** 2
                sde_mask = sde_mask - sde_mask.min()
                sde_mask = sde_mask / sde_mask.max()
                sde_mask = 1-sde_mask
                dyn_scale = EO("dynamic_step_scaled_mean_inv_eps_mask", 2.0)
                sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale
                
            if EO("dynamic_step_scaled_mean_inv_eps_mask_abs"):
                sde_mask = eps.abs()
                sde_mask = sde_mask - sde_mask.min()
                sde_mask = sde_mask / sde_mask.max()
                sde_mask = 1-sde_mask
                dyn_scale = EO("dynamic_step_scaled_mean_inv_eps_mask", 2.0)
                sde_mask = ((dyn_scale-1) + sde_mask) / dyn_scale
            
            
            x_means_per_step = x_next.mean(dim=(-2,-1), keepdim=True)
            if EO("sde_mask_floor"):
                sde_mask_floor = EO("sde_mask_floor", 0.0)
                sde_mask_ceiling = EO("sde_mask_ceiling", 1.0)
                sde_mask = ((sde_mask - sde_mask.min()) * (sde_mask_floor - sde_mask_ceiling)) / (sde_mask.max() - sde_mask.min()) + sde_mask_ceiling    
            
            if not LG.guide_mode.startswith("flow") or (LG.lgw[step] == 0 and LG.lgw[step+1] == 0   and   LG.lgw_inv[step] == 0 and LG.lgw_inv[step+1] == 0):
                x = NS.swap_noise_step(x_0, x_next, mask=sde_mask)
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
            
            if LG.lgw[step] > 0 and step >= EO("guide_cutoff_start_step", 0) and cossim_counter < EO("guide_cutoff_max_iter", 10) and (EO("guide_cutoff") or EO("guide_min")):
                guide_cutoff = EO("guide_cutoff", 1.0)
                #denoised_norm = denoised - denoised.mean(dim=(-2,-1), keepdim=True)
                denoised_norm = data_[0] - data_[0].mean(dim=(-2,-1), keepdim=True)
                y0_norm       = LG.y0    - LG.y0   .mean(dim=(-2,-1), keepdim=True)
                y0_cossim     = get_cosine_similarity(denoised_norm, y0_norm)
                if y0_cossim > guide_cutoff and LG.lgw[step] > EO("guide_cutoff_floor", 0.0):
                    if not EO("guide_cutoff_fast"):
                        LG.lgw[step] *= EO("guide_cutoff_factor", 0.9)
                    else:
                        LG.lgw *= EO("guide_cutoff_factor", 0.9)
                    full_iter -= 1
                if y0_cossim < EO("guide_min", 0.0) and LG.lgw[step] < EO("guide_min_ceiling", 1.0):
                    if not EO("guide_cutoff_fast"):
                        LG.lgw[step] *= EO("guide_min_factor", 1.1)
                    else:
                        LG.lgw *= EO("guide_min_factor", 1.1)
                    full_iter -= 1
                    
        if FLOW_STARTED and FLOW_STOPPED:
            data_prev_ = data_x_prev_
        if FLOW_STARTED and not FLOW_STOPPED:
            data_x_prev_[0] = data_cached       # data_cached is data_x from flow mode. this allows multistep to resume seamlessly.
            for ms in range(recycled_stages):
                data_x_prev_[recycled_stages - ms] = data_x_prev_[recycled_stages - ms - 1]

        data_prev_[0] = data_[0]                # with flow mode, this will be the differentiated guide/"denoised"
        for ms in range(recycled_stages):
            data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]   # TODO: verify that this does not run on every substep...

        
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
        
        if LG.lgw[step] > 0 and step >= EO("guide_step_cutoff_start_step", 0) and cossim_counter < EO("guide_step_cutoff_max_iter", 10) and (EO("guide_step_cutoff") or EO("guide_step_min")):
            guide_cutoff = EO("guide_step_cutoff", 1.0)
            #denoised_norm = denoised - denoised.mean(dim=(-2,-1), keepdim=True)
            eps_trash, data_trash = RK(x, sigma_next, x_0, sigma)
            #denoised_norm = data_[0] - data_[0].mean(dim=(-2,-1), keepdim=True)
            denoised_norm = data_trash - data_trash.mean(dim=(-2,-1), keepdim=True)
            y0_norm       = LG.y0    - LG.y0   .mean(dim=(-2,-1), keepdim=True)
            y0_cossim     = get_cosine_similarity(denoised_norm, y0_norm)
            if y0_cossim > guide_cutoff and LG.lgw[step] > EO("guide_step_cutoff_floor", 0.0):
                if not EO("guide_step_cutoff_fast"):
                    LG.lgw[step] *= EO("guide_step_cutoff_factor", 0.9)
                else:
                    LG.lgw *= EO("guide_step_cutoff_factor", 0.9)
                step -= 1
                x_0 = x = x_[0] = x_0_orig.clone()
            if y0_cossim < EO("guide_step_min", 0.0) and LG.lgw[step] < EO("guide_step_min_ceiling", 1.0):
                if not EO("guide_step_cutoff_fast"):
                    LG.lgw[step] *= EO("guide_step_min_factor", 1.1)
                else:
                    LG.lgw *= EO("guide_step_min_factor", 1.1)
                step -= 1
                x_0 = x = x_[0] = x_0_orig.clone()
        # end sampling loop

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

    #if not (UNSAMPLE and sigmas[1] > sigmas[0]) and not EO("preview_last_step_always"):
    if not (UNSAMPLE and sigmas[1] > sigmas[0]) and not EO("preview_last_step_always") and sigma is not None   and   not (FLOW_STARTED and not FLOW_STOPPED):
        callback_step = len(sigmas)-1 - step if sampler_mode == "unsample" else step
        preview_callback(x, eps, denoised, x_, eps_, data_, callback_step, sigma, sigma_next, callback, EO)

    if INIT_SAMPLE_LOOP:
        state_info_out = state_info
    else:
        state_info_out['raw_x']             = x.to('cpu')
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
        if FLOW_STARTED and not FLOW_STOPPED:
            state_info_out['y0']           = y0.to('cpu') 
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



