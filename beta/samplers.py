import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Callable, Tuple, Dict, Any, Union
import copy
import gc

import comfy.samplers
import comfy.sample
import comfy.supported_models
import comfy.utils
import comfy.nested_tensor
import comfy.patcher_extension
from comfy.samplers import CFGGuider, sampling_function

from comfy_api.latest import io

import re
import latent_preview

from ..helper               import initialize_or_scale, get_res4lyf_scheduler_list, OptionsManager, ExtraOptions, extract_cond_from_guider
from ..res4lyf              import RESplain
from ..latents              import normalize_zscore, get_orthogonal
from ..sigmas               import get_sigmas
#import ..models              # import ReFluxPatcher

from .constants             import MAX_STEPS, IMPLICIT_TYPE_NAMES
from .noise_classes         import NOISE_GENERATOR_CLASSES_SIMPLE, NOISE_GENERATOR_NAMES_SIMPLE, NOISE_GENERATOR_NAMES
from .rk_noise_sampler_beta import NOISE_MODE_NAMES
from .rk_coefficients_beta  import get_default_sampler_name, get_sampler_name_list, process_sampler_name


def copy_cond(conditioning):
    new_conditioning = []
    if type(conditioning[0][0]) == list:
        for i in range(len(conditioning)):
            new_conditioning_i = []
            for embedding, cond in conditioning[i]:
                cond_copy = {}
                for k, v in cond.items():
                    if isinstance(v, torch.Tensor):
                        cond_copy[k] = v.clone()
                    else:
                        cond_copy[k] = v  # ensure we're not copying huge shit like controlnets
                new_conditioning_i.append([embedding.clone(), cond_copy])
            new_conditioning.append(new_conditioning_i)
    else:
        for embedding, cond in conditioning:
            cond_copy = {}
            for k, v in cond.items():
                if isinstance(v, torch.Tensor):
                    cond_copy[k] = v.clone()
                else:
                    cond_copy[k] = v  # ensure we're not copying huge shit like controlnets
            new_conditioning.append([embedding.clone(), cond_copy])

    return new_conditioning


def has_custom_cfg_handling(guider):
    # MultimodalGuider and similar use a 'parameters' dict with per-modality CFG
    if hasattr(guider, 'parameters') and isinstance(getattr(guider, 'parameters', None), dict):
        return True
    # DualCFGGuider uses cfg1/cfg2 instead of cfg
    if hasattr(guider, 'cfg1') and hasattr(guider, 'cfg2'):
        return True
    return False


def generate_init_noise(x, seed, noise_type_init, noise_stdev, noise_mean, noise_normalize,
                        sigma_max, sigma_min, alpha_init=None, k_init=None, EO=None):
    if noise_type_init == "none" or noise_stdev == 0.0:
        return torch.zeros_like(x)

    if EO is not None and EO("bypass_noise_norm") and noise_type_init == "gaussian":
        noise = comfy.sample.prepare_noise(x, seed).to(device=x.device, dtype=x.dtype)
        RESplain("bypass_noise_norm: init noise from comfy.sample.prepare_noise (bypassing RES4LYF normalization)", debug=False)
        return noise

    noise_sampler_init = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type_init)(
        x=x, seed=seed, sigma_max=sigma_max, sigma_min=sigma_min
    )

    if noise_type_init == "fractal":
        noise_sampler_init.alpha = alpha_init
        noise_sampler_init.k = k_init
        noise_sampler_init.scale = 0.1

    noise = noise_sampler_init(sigma=sigma_max * noise_stdev, sigma_next=sigma_min)

    if noise_normalize and noise.std() > 0:
        channelwise = EO("init_noise_normalize_channelwise", "true") if EO else "true"
        channelwise = True if channelwise == "true" else False
        noise = normalize_zscore(noise, channelwise=channelwise, inplace=True)

    noise *= noise_stdev
    noise = (noise - noise.mean()) + noise_mean
    return noise


def apply_nested_normalization(x, idx_0_factor, idx_1_factor):
    """Apply normalization to NestedTensor latent before sampling."""
    if idx_0_factor == 1.0 and idx_1_factor == 1.0:
        return x

    if not (hasattr(x, 'is_nested') and x.is_nested):
        return x

    RESplain(f"Latent normalize (pre-sampling): idx_0={idx_0_factor}, idx_1={idx_1_factor}", debug=True)

    tensors = x.unbind()
    normalized = []
    factors = [idx_0_factor, idx_1_factor]
    for idx, t in enumerate(tensors):
        factor = factors[idx] if idx < len(factors) else 1.0
        normalized.append(t * factor)
    return comfy.nested_tensor.NestedTensor(normalized)


class SharkGuider(CFGGuider):
    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self.cfgs = {}

    def set_conds(self, **kwargs):
        self.inner_set_conds(kwargs)

    def set_cfgs(self, **kwargs):
        self.cfgs = {**kwargs}
        self.cfg  = self.cfgs.get('xt', self.cfg)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        latent_type = model_options['transformer_options'].get('latent_type', 'xt')
        positive = self.conds.get(f'{latent_type}_positive', self.conds.get('positive'))
        negative = self.conds.get(f'{latent_type}_negative', self.conds.get('negative'))
        positive = self.conds.get('positive') if positive is None else positive
        negative = self.conds.get('negative') if negative is None else negative
        cfg      = self.cfgs.get(latent_type, self.cfg)
        
        model_options['transformer_options']['yt_positive'] = self.conds.get('yt_positive')
        model_options['transformer_options']['yt_negative'] = self.conds.get('yt_negative')
        
        return sampling_function(self.inner_model, x, timestep, negative, positive, cfg, model_options=model_options, seed=seed)



class SharkSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "noise_stdev":     ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                "noise_seed":      ("INT",                        {"default": 0,   "min": -1,       "max": 0xffffffffffffffff}),
                "sampler_mode":    (['unsample', 'standard', 'resample'], {"default": "standard"}),
                "scheduler":       (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "steps":           ("INT",                        {"default": 30,  "min": 1,        "max": 10000.0}),
                "denoise":         ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01}),
                "denoise_alt":     ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01}),
                "cfg":             ("FLOAT",                      {"default": 5.5, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
                },
            "optional": {
                "model":           ("MODEL",),
                "positive":        ("CONDITIONING", ),
                "negative":        ("CONDITIONING", ),
                "sampler":         ("SAMPLER", ),
                "sigmas":          ("SIGMAS", ),
                "latent_image":    ("LATENT", ),     
                "extra_options":   ("STRING",                     {"default": "", "multiline": True}),   
                "options":         ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("LATENT", 
                    "LATENT",  
                    "LATENT",)
    
    RETURN_NAMES = ("output", 
                    "denoised",
                    "sde_noise",) 
    
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/samplers"
    EXPERIMENTAL = True
    
    def main(self, 
            model                                       = None,
            cfg                : float                  =  5.5, 
            scheduler          : str                    = "beta57", 
            steps              : int                    = 30, 
            steps_to_run       : int                    = -1,
            sampler_mode       : str                    = "standard",
            denoise            : float                  =  1.0, 
            denoise_alt        : float                  =  1.0,
            noise_type_init    : str                    = "gaussian",
            latent_image       : Optional[dict[Tensor]] = None,
            
            positive                                    = None,
            negative                                    = None,
            sampler                                     = None,
            sigmas             : Optional[Tensor]       = None,
            noise_stdev        : float                  =  1.0,
            noise_mean         : float                  =  0.0,
            noise_normalize    : bool                   = True,
            
            d_noise            : float                  =  1.0,
            alpha_init         : float                  = -1.0,
            k_init             : float                  =  1.0,
            cfgpp              : float                  =  0.0,
            noise_seed         : int                    = -1,
            options                                     = None,
            sde_noise                                   = None,
            sde_noise_steps    : int                    =  1,
            
            rebounds           : int                    =  0,
            unsample_cfg       : float                  = 1.0,
            unsample_eta       : float                  = 0.5,
            unsampler_name     : str                    = "none",
            unsample_steps_to_run : int                 = -1,
            eta_decay_scale   : float                  = 1.0,
            
            #ultracascade_stage : str = "stage_UP",
            ultracascade_latent_image : Optional[dict[str,Any]] = None,
            ultracascade_guide_weights: Optional[Tuple] = None,
            
            ultracascade_latent_width : int = 0,
            ultracascade_latent_height: int = 0,

            extra_options      : str = "", 
            **kwargs,
            ): 
        
            
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED



            # INIT EXTENDABLE OPTIONS INPUTS
            
            options_mgr     = OptionsManager(options, **kwargs)
                        
            extra_options  += "\n" + options_mgr.get('extra_options', "")
            EO              = ExtraOptions(extra_options)
            default_dtype   = EO("default_dtype", torch.float64)
            default_device  = EO("work_device", "cuda" if torch.cuda.is_available() else "cpu")
            
            noise_stdev     = options_mgr.get('noise_init_stdev', noise_stdev)
            noise_mean      = options_mgr.get('noise_init_mean',  noise_mean)
            noise_type_init = options_mgr.get('noise_type_init',  noise_type_init)
            d_noise         = options_mgr.get('d_noise',          d_noise)
            alpha_init      = options_mgr.get('alpha_init',       alpha_init)
            k_init          = options_mgr.get('k_init',           k_init)
            sde_noise       = options_mgr.get('sde_noise',        sde_noise)
            sde_noise_steps = options_mgr.get('sde_noise_steps',  sde_noise_steps)
            rebounds        = options_mgr.get('rebounds',         rebounds)
            unsample_cfg    = options_mgr.get('unsample_cfg',     unsample_cfg)
            unsample_eta    = options_mgr.get('unsample_eta',     unsample_eta)
            unsampler_name  = options_mgr.get('unsampler_name',   unsampler_name)
            unsample_steps_to_run = options_mgr.get('unsample_steps_to_run',   unsample_steps_to_run)
            
            eta_decay_scale = options_mgr.get('eta_decay_scale',  eta_decay_scale)
            start_at_step   = options_mgr.get('start_at_step',    -1)
            tile_sizes      = options_mgr.get('tile_sizes',       None)
            flow_sync_eps   = options_mgr.get('flow_sync_eps',    0.0)
            
            unsampler_name, _ = process_sampler_name(unsampler_name)

            
            #ultracascade_stage        = options_mgr.get('ultracascade_stage',         ultracascade_stage)
            ultracascade_latent_image  = options_mgr.get('ultracascade_latent_image',  ultracascade_latent_image)
            ultracascade_latent_width  = options_mgr.get('ultracascade_latent_width',  ultracascade_latent_width)
            ultracascade_latent_height = options_mgr.get('ultracascade_latent_height', ultracascade_latent_height)
            
            
            if 'BONGMATH' in sampler.extra_options:
                sampler.extra_options['start_at_step'] = start_at_step
                sampler.extra_options['tile_sizes']    = tile_sizes
                
                sampler.extra_options['unsample_bongmath'] = options_mgr.get('unsample_bongmath', sampler.extra_options['BONGMATH'])   # allow turning off bongmath for unsampling with cycles
                sampler.extra_options['flow_sync_eps'] = flow_sync_eps
                
            is_chained = False
            if latent_image is not None:
                if 'sampler' in latent_image and sampler is None:
                    sampler = copy_cond(latent_image['sampler'])
                    is_chained = True

            if 'steps_to_run' in sampler.extra_options:
                sampler.extra_options['steps_to_run'] = steps_to_run

            guider_input = options_mgr.get('guider', None)
            guider_from_latent = latent_image.get('guider') if latent_image is not None else None

            if guider_input is not None:
                # Explicit guider input takes full precedence - all settings come from guider
                guider = copy.copy(guider_input)
                guider.original_conds = dict(guider.original_conds)
                work_model = guider.model_patcher
                RESplain("Using guider from SharkOptions_GuiderInput: ", work_model.model.diffusion_model.__class__.__name__)
                RESplain("SharkWarning: \"flow\" guide mode does not work with SharkOptions_GuiderInput")
                if has_custom_cfg_handling(guider):
                    RESplain(f"SharkWarning: Guider has custom CFG handling ({guider.__class__.__name__}) - using guider's internal CFG settings")
                elif hasattr(guider, 'cfg') and guider.cfg is not None:
                    cfg = guider.cfg
                    RESplain("Using cfg from SharkOptions_GuiderInput: ", cfg, debug=True)
                extracted_positive = extract_cond_from_guider(guider, 'positive')
                if extracted_positive is not None:
                    positive = extracted_positive
                    RESplain("Using positive cond from SharkOptions_GuiderInput", debug=True)
                extracted_negative = extract_cond_from_guider(guider, 'negative')
                if extracted_negative is not None:
                    negative = extracted_negative
                    RESplain("Using negative cond from SharkOptions_GuiderInput", debug=True)
            elif guider_from_latent is not None:
                guider = copy.copy(guider_from_latent)
                guider.original_conds = dict(guider.original_conds)
                if model is not None:
                    work_model = model
                    guider.model_patcher = model
                    guider.model_options = model.model_options
                    RESplain("Overriding chained guider model with provided model input", debug=True)
                else:
                    work_model = guider.model_patcher
                RESplain("Continuing guider from chained latent: ", work_model.model.diffusion_model.__class__.__name__)
                # CFG is set per-node, not inherited from chained guider (will be applied via set_cfg/set_cfgs later)
                if has_custom_cfg_handling(guider):
                    RESplain(f"SharkWarning: Guider has custom CFG handling ({guider.__class__.__name__}) - node CFG input will be ignored")
                else:
                    guider_cfg = guider.cfg if hasattr(guider, 'cfg') else None
                    if guider_cfg is not None and guider_cfg != cfg:
                        RESplain(f"SharkWarning: Chained guider CFG ({guider_cfg}) will be overridden with node CFG ({cfg})")
                if positive is None:
                    positive = extract_cond_from_guider(guider, 'positive')
                    if positive is not None:
                        RESplain("Using positive cond from chained guider", debug=True)
                if negative is None:
                    negative = extract_cond_from_guider(guider, 'negative')
                    if negative is not None:
                        RESplain("Using negative cond from chained guider", debug=True)
                is_chained = True
            else:
                guider = None
                work_model = model
            
            if latent_image is not None:
                latent_image = latent_image.copy()
                samples_fixed = comfy.sample.fix_empty_latent_channels(work_model, latent_image['samples'])
                if isinstance(samples_fixed, comfy.nested_tensor.NestedTensor):
                    latent_image['samples'] = comfy.nested_tensor.NestedTensor([t.clone() for t in samples_fixed.unbind()])
                else:
                    latent_image['samples'] = samples_fixed.clone()
                
            if positive is None or negative is None:
                from ..conditioning import EmptyConditioningGenerator
                EmptyCondGen       = EmptyConditioningGenerator(work_model)
                positive, negative = EmptyCondGen.zero_none_conditionings_([positive, negative])

            if cfg < 0:
                sampler.extra_options['cfg_cw'] = -cfg
                RESplain(f"Shark: Using channelwise CFG ({-cfg}), guider CFG set to 1.0")
                cfg = 1.0
            else:
                sampler.extra_options.pop("cfg_cw", None) 

            
            is_nested_input = latent_image is not None and 'samples' in latent_image and isinstance(latent_image['samples'], comfy.nested_tensor.NestedTensor)
            if EO("enable_dummy_sampler_init"):
                if is_nested_input:
                    raise NotImplementedError("Extra option 'enable_dummy_sampler_init' with nested tensors is not supported.")
                sampler_null = comfy.samplers.ksampler("rk_beta",
                    {
                        "sampler_mode": "NULL",
                    })
                if latent_image is not None and 'samples' in latent_image:
                    latent_vram_factor = EO("latent_vram_factor", 3)
                    x_null = torch.zeros_like(latent_image['samples']).repeat_interleave(latent_vram_factor, dim=-1)
                elif ultracascade_latent_height * ultracascade_latent_width > 0:
                    x_null = comfy.sample.fix_empty_latent_channels(model, torch.zeros((1,16,ultracascade_latent_height,ultracascade_latent_width)))
                else:
                    print("Fallback: spawning dummy 1,16,256,256 latent.")
                    x_null = comfy.sample.fix_empty_latent_channels(model, torch.zeros((1,16,256,256)))
                _ = comfy.sample.sample_custom(work_model, x_null, cfg, sampler_null, torch.linspace(1, 0, 10).to(x_null.dtype).to(x_null.device), negative, negative, x_null, noise_mask=None, callback=None, disable_pbar=disable_pbar, seed=noise_seed)

            sigma_min = work_model.get_model_object('model_sampling').sigma_min
            sigma_max = work_model.get_model_object('model_sampling').sigma_max
            
            if sampler is None:
                raise ValueError("sampler is required")
            else:
                sampler = copy.deepcopy(sampler)
        
        
        
            # INIT SIGMAS
            if sigmas is not None:
                sigmas = sigmas.clone().to(dtype=default_dtype, device=default_device) # does this type carry into clown after passing through comfy?
                sigmas *= denoise   # ... otherwise we have to interpolate and that might not be ideal for tiny custom schedules...
            else: 
                sigmas = get_sigmas(work_model, scheduler, steps, abs(denoise)).to(dtype=default_dtype, device=default_device)
            sigmas *= denoise_alt

            # USE NULL FLOATS AS "FLAGS" TO PREVENT COMFY NOISE ADDITION
            if sampler_mode.startswith("unsample"): 
                null   = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.flip(sigmas, dims=[0])
                sigmas = torch.cat([sigmas, null])
                
            elif sampler_mode.startswith("resample") and not EO("disable_resample_sigmas_padding"):
                null   = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.cat([null, sigmas])
                sigmas = torch.cat([sigmas, null])



            latent_x = {}
            # INIT STATE INFO FOR CONTINUING GENERATION ACROSS MULTIPLE SAMPLER NODES
            if latent_image is not None:
                samples = latent_image['samples']
                latent_x['samples'] = samples
                if 'noise_mask' in latent_image:
                    noise_mask = latent_image['noise_mask']
                    if isinstance(noise_mask, comfy.nested_tensor.NestedTensor):
                        latent_x['noise_mask'] = comfy.nested_tensor.NestedTensor([t.clone() for t in noise_mask.unbind()])
                    else:
                        latent_x['noise_mask'] = noise_mask.clone()
                state_info = copy.deepcopy(latent_image['state_info']) if 'state_info' in latent_image else {}
            else:
                state_info = {}
            state_info_out = {}
            
            
            
            # SETUP CONDITIONING EMBEDS
            
            pos_cond = copy_cond(positive)
            neg_cond = copy_cond(negative)
            
            
            
            # SETUP FOR ULTRACASCADE IF DETECTED
            if work_model.model.model_config.unet_config.get('stable_cascade_stage') == 'up':
                
                ultracascade_guide_weight = EO("ultracascade_guide_weight", 0.0)
                ultracascade_guide_type   = EO("ultracascade_guide_type", "residual")
                
                x_lr = None
                if ultracascade_latent_height * ultracascade_latent_width > 0:
                    x_lr        = latent_image['samples'].clone() if latent_image is not None else None
                    x_lr_bs     = 1                               if x_lr         is     None else x_lr.shape[-4]
                    x_lr_dtype  = default_dtype                   if x_lr         is     None else x_lr.dtype
                    x_lr_device = 'cuda'                          if x_lr         is     None else x_lr.device
                    
                    ultracascade_stage_up_upscale_align_corners = EO("ultracascade_stage_up_upscale_align_corners", False)
                    ultracascade_stage_up_upscale_mode          = EO("ultracascade_stage_up_upscale_mode",         "bicubic")
                    latent_x['samples'] = torch.zeros([x_lr_bs, 16, ultracascade_latent_height, ultracascade_latent_width], dtype=x_lr_dtype, device=x_lr_device)
                
                    data_prev_ = state_info.get('data_prev_')
                    if EO("ultracascade_stage_up_preserve_data_prev") and data_prev_ is not None:
                        if data_prev_.dim() == 5:  # [B, C, T, H, W]
                            data_prev_ = F.interpolate(
                                data_prev_.squeeze(2),  # Remove T dim for 2D interpolate
                                size=latent_x['samples'].shape[-2:],
                                mode=ultracascade_stage_up_upscale_mode,
                                align_corners=ultracascade_stage_up_upscale_align_corners
                                ).unsqueeze(2)  # Restore T dim
                        elif data_prev_.dim() == 4:
                            data_prev_ = F.interpolate(
                                data_prev_,
                                size=latent_x['samples'].shape[-2:],
                                mode=ultracascade_stage_up_upscale_mode,
                                align_corners=ultracascade_stage_up_upscale_align_corners
                                )
                        else:
                            print("data_prev_ upscale failed.")
                        state_info['data_prev_'] = data_prev_
                    
                    else:
                        state_info['data_prev_'] = data_prev_ #None   # = None was leading to errors even with sampler_mode=standard due to below with = state_info['data_prev_'][batch_num]
                
                if x_lr is not None:
                    if x_lr.shape[-2:] != latent_image['samples'].shape[-2:]:
                        x_height, x_width = latent_image['samples'].shape[-2:]
                        ultracascade_stage_up_upscale_align_corners = EO("ultracascade_stage_up_upscale_align_corners", False)
                        ultracascade_stage_up_upscale_mode          = EO("ultracascade_stage_up_upscale_mode",         "bicubic")

                        x_lr = F.interpolate(x_lr, size=(x_height, x_width), mode=ultracascade_stage_up_upscale_mode, align_corners=ultracascade_stage_up_upscale_align_corners)
                        
                ultracascade_guide_weights = initialize_or_scale(ultracascade_guide_weights, ultracascade_guide_weight, MAX_STEPS)

                patch = work_model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("ultracascade", {}).get("main")
                if patch is not None:
                    patch.update(x_lr=x_lr, guide_weights=ultracascade_guide_weights, guide_type=ultracascade_guide_type)
                else:
                    work_model.model.diffusion_model.set_sigmas_schedule(sigmas_schedule = sigmas)
                    work_model.model.diffusion_model.set_sigmas_prev    (sigmas_prev     = sigmas[:1])
                    work_model.model.diffusion_model.set_guide_weights  (guide_weights   = ultracascade_guide_weights)
                    work_model.model.diffusion_model.set_guide_type     (guide_type      = ultracascade_guide_type)
                    work_model.model.diffusion_model.set_x_lr           (x_lr            = x_lr)
                
            elif work_model.model.model_config.unet_config.get('stable_cascade_stage') == 'b':
                #if sampler_mode != "resample":
                #    state_info['data_prev_'] = None    #commented out as it was throwing an error below with = state_info['data_prev_'][batch_num]
                
                c_pos, c_neg = [], []
                for t in pos_cond:
                    d_pos = t[1].copy()
                    d_neg = t[1].copy()
                    
                    x_lr = None
                    if ultracascade_latent_height * ultracascade_latent_width > 0:
                        x_lr = latent_image['samples'].clone()
                        latent_x['samples'] = torch.zeros([x_lr.shape[-4], 4, ultracascade_latent_height // 4, ultracascade_latent_width // 4], dtype=x_lr.dtype, device=x_lr.device)
                    
                    d_pos['stable_cascade_prior'] = x_lr

                    pooled_output = d_neg.get("pooled_output", None)
                    if pooled_output is not None:
                        d_neg["pooled_output"] = torch.zeros_like(pooled_output)
                    
                    c_pos.append(                 [t[0],  d_pos])            
                    c_neg.append([torch.zeros_like(t[0]), d_neg])
                pos_cond = c_pos
                neg_cond = c_neg
                
            elif ultracascade_latent_height * ultracascade_latent_width > 0:
                latent_x['samples'] = torch.zeros([1, 16, ultracascade_latent_height, ultracascade_latent_width], dtype=default_dtype, device=sigmas.device)
            
            
            
            # NOISE, ORTHOGONALIZE, OR ZERO EMBEDS
            
            if pos_cond is None or neg_cond is None:
                from ..conditioning import EmptyConditioningGenerator
                EmptyCondGen       = EmptyConditioningGenerator(work_model)
                pos_cond, neg_cond = EmptyCondGen.zero_none_conditionings_([pos_cond, neg_cond])



            if EO(("cond_noise", "uncond_noise")):
                if noise_seed == -1:
                    cond_seed = torch.initial_seed() + 1
                else:
                    cond_seed = noise_seed
                
                t5_seed              = EO("t5_seed"             , cond_seed)
                clip_seed            = EO("clip_seed"           , cond_seed+1)
                t5_noise_type        = EO("t5_noise_type"       , "gaussian")
                clip_noise_type      = EO("clip_noise_type"     , "gaussian")
                t5_noise_sigma_max   = EO("t5_noise_sigma_max"  , "gaussian")
                t5_noise_sigma_min   = EO("t5_noise_sigma_min"  , "gaussian")
                clip_noise_sigma_max = EO("clip_noise_sigma_max", "gaussian")
                clip_noise_sigma_min = EO("clip_noise_sigma_min", "gaussian")
                
                noise_sampler_t5     = NOISE_GENERATOR_CLASSES_SIMPLE.get(  t5_noise_type)(x=pos_cond[0][0],                  seed=  t5_seed, sigma_max=  t5_noise_sigma_max, sigma_min=  t5_noise_sigma_min, )
                noise_sampler_clip   = NOISE_GENERATOR_CLASSES_SIMPLE.get(clip_noise_type)(x=pos_cond[0][1]['pooled_output'], seed=clip_seed, sigma_max=clip_noise_sigma_max, sigma_min=clip_noise_sigma_min, )
                
                t5_noise_scale   = EO("t5_noise_scale",   1.0)
                clip_noise_scale = EO("clip_noise_scale", 1.0)
                
                if EO("cond_noise"):
                    t5_noise   = noise_sampler_t5  (sigma=  t5_noise_sigma_max, sigma_next=  t5_noise_sigma_min)
                    clip_noise = noise_sampler_clip(sigma=clip_noise_sigma_max, sigma_next=clip_noise_sigma_min)
                    
                    pos_cond[0][0]                  = pos_cond[0][0]                  + t5_noise_scale   * (t5_noise   - pos_cond[0][0])
                    pos_cond[0][1]['pooled_output'] = pos_cond[0][1]['pooled_output'] + clip_noise_scale * (clip_noise - pos_cond[0][1]['pooled_output'])
                    
                if EO("uncond_noise"):
                    t5_noise   = noise_sampler_t5  (sigma=  t5_noise_sigma_max, sigma_next=  t5_noise_sigma_min)
                    clip_noise = noise_sampler_clip(sigma=clip_noise_sigma_max, sigma_next=clip_noise_sigma_min)
                    
                    neg_cond[0][0]                  = neg_cond[0][0]                  + t5_noise_scale   * (t5_noise   - neg_cond[0][0])
                    neg_cond[0][1]['pooled_output'] = neg_cond[0][1]['pooled_output'] + clip_noise_scale * (clip_noise - neg_cond[0][1]['pooled_output'])

            if EO("uncond_ortho"):
                neg_cond[0][0]                  = get_orthogonal(neg_cond[0][0],                  pos_cond[0][0])
                neg_cond[0][1]['pooled_output'] = get_orthogonal(neg_cond[0][1]['pooled_output'], pos_cond[0][1]['pooled_output'])
            

            if "noise_seed" in sampler.extra_options:
                if sampler.extra_options['noise_seed'] == -1 and noise_seed != -1:
                    sampler.extra_options['noise_seed'] = noise_seed + 1
                    RESplain("Shark: setting clown noise seed to: ", sampler.extra_options['noise_seed'], debug=True)

            if "sampler_mode" in sampler.extra_options:
                sampler.extra_options['sampler_mode'] = sampler_mode

            if "extra_options" in sampler.extra_options:
                extra_options += "\n"
                extra_options += sampler.extra_options['extra_options']
                sampler.extra_options['extra_options'] = extra_options

            samples = latent_x['samples']
            latent_image_batch = {"samples": samples}
            if 'noise_mask' in latent_x and latent_x['noise_mask'] is not None:
                noise_mask = latent_x['noise_mask']
                latent_image_batch['noise_mask'] = noise_mask

            if not EO("use_batch_loop"):
                x = latent_image_batch['samples'].to(default_dtype)

                if isinstance(x, comfy.nested_tensor.NestedTensor):
                    noise = comfy.nested_tensor.NestedTensor([
                        generate_init_noise(
                            x=t.clone(), seed=noise_seed + idx,
                            noise_type_init=noise_type_init, noise_stdev=noise_stdev,
                            noise_mean=noise_mean, noise_normalize=noise_normalize,
                            sigma_max=sigma_max, sigma_min=sigma_min,
                            alpha_init=alpha_init, k_init=k_init, EO=EO
                        )
                        for idx, t in enumerate(x.unbind())
                    ])
                else:
                    noise = generate_init_noise(
                        x=x.clone(), seed=noise_seed,
                        noise_type_init=noise_type_init, noise_stdev=noise_stdev,
                        noise_mean=noise_mean, noise_normalize=noise_normalize,
                        sigma_max=sigma_max, sigma_min=sigma_min,
                        alpha_init=alpha_init, k_init=k_init, EO=EO
                    )

                # SETUP REGIONAL COND
                if pos_cond[0][1] is not None:
                    if 'callback_regional' in pos_cond[0][1]:
                        pos_cond = pos_cond[0][1]['callback_regional'](work_model)

                    if 'AttnMask' in pos_cond[0][1]:
                        sampler.extra_options['AttnMask']   = pos_cond[0][1]['AttnMask']
                        sampler.extra_options['RegContext'] = pos_cond[0][1]['RegContext']
                        sampler.extra_options['RegParam']   = pos_cond[0][1]['RegParam']

                        if isinstance(model.model.model_config, (comfy.supported_models.SDXL, comfy.supported_models.SD15)):
                            latent_up_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] * 2, latent_image_batch['samples'].shape[-1] * 2), mode="nearest")
                            sampler.extra_options['AttnMask'].set_latent(latent_up_dummy)
                            sampler.extra_options['AttnMask'].generate()
                            sampler.extra_options['AttnMask'].mask_up = sampler.extra_options['AttnMask'].attn_mask.mask

                            latent_down_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] // 2, latent_image_batch['samples'].shape[-1] // 2), mode="nearest")
                            sampler.extra_options['AttnMask'].set_latent(latent_down_dummy)
                            sampler.extra_options['AttnMask'].generate()
                            sampler.extra_options['AttnMask'].mask_down = sampler.extra_options['AttnMask'].attn_mask.mask

                            if isinstance(model.model.model_config, comfy.supported_models.SD15):
                                latent_down_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] // 4, latent_image_batch['samples'].shape[-1] // 4), mode="nearest")
                                sampler.extra_options['AttnMask'].set_latent(latent_down_dummy)
                                sampler.extra_options['AttnMask'].generate()
                                sampler.extra_options['AttnMask'].mask_down2 = sampler.extra_options['AttnMask'].attn_mask.mask

                        if isinstance(model.model.model_config, comfy.supported_models.Stable_Cascade_C):
                            latent_up_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] * 2, latent_image_batch['samples'].shape[-1] * 2), mode="nearest")
                            sampler.extra_options['AttnMask'].set_latent(latent_up_dummy)
                            sampler.extra_options['AttnMask'].context_lens = [context_len + 8 for context_len in sampler.extra_options['AttnMask'].context_lens]
                            sampler.extra_options['AttnMask'].text_len = sum(sampler.extra_options['AttnMask'].context_lens)
                        else:
                            sampler.extra_options['AttnMask'].set_latent(latent_image_batch['samples'])
                        sampler.extra_options['AttnMask'].generate()

                if neg_cond[0][1] is not None:
                    if 'callback_regional' in neg_cond[0][1]:
                        neg_cond = neg_cond[0][1]['callback_regional'](work_model)

                    if 'AttnMask' in neg_cond[0][1]:
                        sampler.extra_options['AttnMask_neg']   = neg_cond[0][1]['AttnMask']
                        sampler.extra_options['RegContext_neg'] = neg_cond[0][1]['RegContext']
                        sampler.extra_options['RegParam_neg']   = neg_cond[0][1]['RegParam']

                        if isinstance(model.model.model_config, (comfy.supported_models.SDXL, comfy.supported_models.SD15)):
                            latent_up_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] * 2, latent_image_batch['samples'].shape[-1] * 2), mode="nearest")
                            sampler.extra_options['AttnMask_neg'].set_latent(latent_up_dummy)
                            sampler.extra_options['AttnMask_neg'].generate()
                            sampler.extra_options['AttnMask_neg'].mask_up = sampler.extra_options['AttnMask_neg'].attn_mask.mask

                            latent_down_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] // 2, latent_image_batch['samples'].shape[-1] // 2), mode="nearest")
                            sampler.extra_options['AttnMask_neg'].set_latent(latent_down_dummy)
                            sampler.extra_options['AttnMask_neg'].generate()
                            sampler.extra_options['AttnMask_neg'].mask_down = sampler.extra_options['AttnMask_neg'].attn_mask.mask

                            if isinstance(model.model.model_config, comfy.supported_models.SD15):
                                latent_down_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] // 4, latent_image_batch['samples'].shape[-1] // 4), mode="nearest")
                                sampler.extra_options['AttnMask_neg'].set_latent(latent_down_dummy)
                                sampler.extra_options['AttnMask_neg'].generate()
                                sampler.extra_options['AttnMask_neg'].mask_down2 = sampler.extra_options['AttnMask_neg'].attn_mask.mask

                        if isinstance(model.model.model_config, comfy.supported_models.Stable_Cascade_C):
                            latent_up_dummy = F.interpolate(latent_image_batch['samples'].to(torch.float16), size=(latent_image_batch['samples'].shape[-2] * 2, latent_image_batch['samples'].shape[-1] * 2), mode="nearest")
                            sampler.extra_options['AttnMask_neg'].set_latent(latent_up_dummy)
                            sampler.extra_options['AttnMask_neg'].context_lens = [context_len + 8 for context_len in sampler.extra_options['AttnMask_neg'].context_lens]
                            sampler.extra_options['AttnMask_neg'].text_len = sum(sampler.extra_options['AttnMask_neg'].context_lens)
                        else:
                            sampler.extra_options['AttnMask_neg'].set_latent(latent_image_batch['samples'])
                        sampler.extra_options['AttnMask_neg'].generate()

                if guider is None:
                    guider = SharkGuider(work_model)
                    flow_cond = options_mgr.get('flow_cond', {})
                    if flow_cond and 'yt_positive' in flow_cond:
                        if 'yt_inv_positive' not in flow_cond:
                            guider.set_conds(yt_positive=flow_cond.get('yt_positive'),
                                             yt_negative=flow_cond.get('yt_negative'))
                            guider.set_cfgs(yt=flow_cond.get('yt_cfg'), xt=cfg)
                        else:
                            guider.set_conds(yt_positive=flow_cond.get('yt_positive'),
                                             yt_negative=flow_cond.get('yt_negative'),
                                             yt_inv_positive=flow_cond.get('yt_inv_positive'),
                                             yt_inv_negative=flow_cond.get('yt_inv_negative'))
                            guider.set_cfgs(yt=flow_cond.get('yt_cfg'),
                                           yt_inv=flow_cond.get('yt_inv_cfg'), xt=cfg)
                    else:
                        guider.set_cfgs(xt=cfg)
                    guider.set_conds(positive=pos_cond, negative=neg_cond)
                elif type(guider) == SharkGuider:
                    guider.cfgs['xt'] = cfg
                    guider.cfg = cfg
                    guider.set_conds(positive=pos_cond, negative=neg_cond)
                    RESplain(f"Shark: Applied CFG ({cfg}) to SharkGuider", debug=True)
                else:
                    if has_custom_cfg_handling(guider):
                        # Guider has its own CFG handling - set_cfg would succeed but be ignored
                        RESplain(f"Shark: Guider ({guider.__class__.__name__}) has custom CFG - using guider's internal settings", debug=True)
                        try:
                            guider.set_conds(pos_cond, neg_cond)
                        except:
                            pass
                    else:
                        try:
                            guider.set_cfg(cfg)
                            guider.set_conds(pos_cond, neg_cond)
                            RESplain(f"Shark: Applied CFG ({cfg}) to guider", debug=True)
                        except:
                            RESplain(f"SharkWarning: guider.set_cfg failed - guider will use its original CFG settings (node CFG {cfg} ignored)")
                            pass

                if latent_image is not None and 'state_info' in latent_image and 'sigmas' in latent_image['state_info']:
                    steps_len = max(sigmas.shape[-1] - 1, latent_image['state_info']['sigmas'].shape[-1] - 1)
                else:
                    steps_len = sigmas.shape[-1] - 1

                x0_output = {}
                try:
                    callback = latent_preview.prepare_callback(work_model, steps_len, x0_output,
                        shape=x.shape if hasattr(x, 'is_nested') and x.is_nested else None)
                except TypeError:
                    callback = latent_preview.prepare_callback(work_model, steps_len, x0_output)

                noise_mask = latent_image_batch.get("noise_mask", None)

                if noise_mask is not None and sampler_mode in {"resample", "unsample"}:
                    stored_image = state_info.get('image_initial')
                    if stored_image is not None and stored_image.shape == x.shape:
                        x_initial = stored_image
                    else:
                        x_initial = x
                    stored_noise = state_info.get('noise_initial')
                    if stored_noise is not None and stored_noise.shape == noise.shape:
                        noise_initial = stored_noise
                    else:
                        noise_initial = noise
                else:
                    x_initial = x
                    noise_initial = noise

                state_info_out = {}
                if 'BONGMATH' in sampler.extra_options:
                    sampler.extra_options['state_info'] = state_info
                    sampler.extra_options['state_info_out'] = state_info_out
                    sampler.extra_options['image_initial'] = x_initial
                    sampler.extra_options['noise_initial'] = noise_initial

                if rebounds > 0:
                    if has_custom_cfg_handling(guider):
                        RESplain(f"SharkWarning: Rebounds with guider ({guider.__class__.__name__}) that has custom CFG - unsample_cfg will be ignored")
                    has_cfgs = hasattr(guider, 'cfgs')
                    cfgs_cached = guider.cfgs if has_cfgs else None
                    cfg_cached = guider.cfg
                    steps_to_run_cached = sampler.extra_options['steps_to_run']
                    eta_cached         = sampler.extra_options['eta']
                    eta_substep_cached = sampler.extra_options['eta_substep']

                    etas_cached         = sampler.extra_options['etas'].clone()
                    etas_substep_cached = sampler.extra_options['etas_substep'].clone()

                    unsample_etas = torch.full_like(etas_cached, unsample_eta)
                    rk_type_cached = sampler.extra_options['rk_type']

                    if sampler.extra_options['sampler_mode'] == "unsample":
                        if has_cfgs:
                            guider.cfgs = {'xt': unsample_cfg, 'yt': unsample_cfg}
                            RESplain(f"Shark: Rebounds init - setting unsample CFG (cfgs): xt={unsample_cfg}, yt={unsample_cfg}", debug=True)
                        else:
                            guider.cfg = unsample_cfg
                            RESplain(f"Shark: Rebounds init - setting unsample CFG: {unsample_cfg}", debug=True)
                        if unsample_eta != -1.0:
                            sampler.extra_options['eta_substep']  = unsample_eta
                            sampler.extra_options['eta']          = unsample_eta
                            sampler.extra_options['etas_substep'] = unsample_etas
                            sampler.extra_options['etas']         = unsample_etas
                        if unsampler_name != "none":
                            sampler.extra_options['rk_type']      = unsampler_name
                        if unsample_steps_to_run > -1:
                            sampler.extra_options['steps_to_run'] = unsample_steps_to_run
                    else:
                        if has_cfgs:
                            guider.cfgs = cfgs_cached
                        else:
                            guider.cfg = cfg_cached

                    if has_cfgs:
                        guider.cfgs = cfgs_cached
                    else:
                        guider.cfg = cfg_cached
                    sampler.extra_options['steps_to_run'] = steps_to_run_cached

                    eta_decay           = eta_cached
                    eta_substep_decay   = eta_substep_cached
                    unsample_eta_decay  = unsample_eta

                    etas_decay          = etas_cached
                    etas_substep_decay  = etas_substep_cached
                    unsample_etas_decay = unsample_etas

                # Apply pre-sampling latent normalization for NestedTensors
                idx_0_factor = options_mgr.get('latent_normalize_idx_0', 1.0)
                idx_1_factor = options_mgr.get('latent_normalize_idx_1', 1.0)
                factors_0_steps = options_mgr.get('latent_normalize_idx_0_steps', [1.0])
                factors_1_steps = options_mgr.get('latent_normalize_idx_1_steps', [1.0])

                # Per-step normalization: pass shapes and factors to inner sampler
                has_per_step_factors = len(factors_0_steps) > 1 or len(factors_1_steps) > 1
                if isinstance(x, comfy.nested_tensor.NestedTensor) and 'rk_type' in sampler.extra_options:
                    sampler.extra_options['latent_shapes'] = [t.shape for t in x.unbind()]
                    sampler.extra_options['latent_normalize_idx_0_steps'] = factors_0_steps
                    sampler.extra_options['latent_normalize_idx_1_steps'] = factors_1_steps

                # Node-level normalization only for single-value factors (backward compat)
                # Per-step factors are applied in the inner loop instead
                if not has_per_step_factors and (idx_0_factor != 1.0 or idx_1_factor != 1.0):
                    # Normalize x for standard sampling
                    x = apply_nested_normalization(x, idx_0_factor, idx_1_factor)
                    # Also normalize raw_x for chainsampling (rk_sampler_beta replaces x with raw_x)
                    # raw_x is a packed tensor, so we need to unpack, normalize, repack
                    if 'raw_x' in state_info and hasattr(x, 'is_nested') and x.is_nested:
                        raw_x = state_info['raw_x']
                        latent_shapes = [t.shape for t in x.unbind()]
                        tensors = comfy.utils.unpack_latents(raw_x, latent_shapes)
                        factors = [idx_0_factor, idx_1_factor]
                        for idx, t in enumerate(tensors):
                            factor = factors[idx] if idx < len(factors) else 1.0
                            if factor != 1.0:
                                tensors[idx] = t * factor
                        state_info['raw_x'], _ = comfy.utils.pack_latents(tensors)
                        RESplain(f"Latent normalize: applied to raw_x (packed), idx_0={idx_0_factor}, idx_1={idx_1_factor}", debug=True)

                sampler.extra_options['outer_sigmas_len'] = sigmas.shape[-1]
                samples = guider.sample(noise, x_initial, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

                if rebounds > 0:
                    noise_seed_cached   = sampler.extra_options['noise_seed']
                    cfgs_cached         = guider.cfgs if has_cfgs else None
                    cfg_cached          = guider.cfg
                    sampler_mode_cached = sampler.extra_options['sampler_mode']

                    for restarts_iter in range(rebounds):
                        sampler.extra_options['state_info'] = sampler.extra_options['state_info_out']

                        sigmas = sampler.extra_options['state_info_out']['sigmas'] if sigmas is None else sigmas

                        if   sampler.extra_options['sampler_mode'] == "standard":
                            sampler.extra_options['sampler_mode'] = "unsample"
                        elif sampler.extra_options['sampler_mode'] == "unsample":
                            sampler.extra_options['sampler_mode'] = "resample"
                        elif sampler.extra_options['sampler_mode'] == "resample":
                            sampler.extra_options['sampler_mode'] = "unsample"

                        sampler.extra_options['noise_seed'] = -1

                        if sampler.extra_options['sampler_mode'] == "unsample":
                            if has_cfgs:
                                guider.cfgs = {'xt': unsample_cfg, 'yt': unsample_cfg}
                                RESplain(f"Shark: Rebounds unsample - CFG (cfgs): xt={unsample_cfg}, yt={unsample_cfg}", debug=True)
                            else:
                                guider.cfg = unsample_cfg
                                RESplain(f"Shark: Rebounds unsample - CFG: {unsample_cfg}", debug=True)
                            if unsample_eta != -1.0:
                                sampler.extra_options['eta_substep']  = unsample_eta_decay
                                sampler.extra_options['eta']          = unsample_eta_decay
                                sampler.extra_options['etas_substep'] = unsample_etas
                                sampler.extra_options['etas']         = unsample_etas
                            else:
                                sampler.extra_options['eta_substep']  = eta_substep_decay
                                sampler.extra_options['eta']          = eta_decay
                                sampler.extra_options['etas_substep'] = etas_substep_decay
                                sampler.extra_options['etas']         = etas_decay
                            if unsampler_name != "none":
                                sampler.extra_options['rk_type']  = unsampler_name
                            if unsample_steps_to_run > -1:
                                sampler.extra_options['steps_to_run'] = unsample_steps_to_run
                        else:
                            if has_cfgs:
                                guider.cfgs = cfgs_cached
                                RESplain(f"Shark: Rebounds resample - restored CFG (cfgs)", debug=True)
                            else:
                                guider.cfg = cfg_cached
                                RESplain(f"Shark: Rebounds resample - restored CFG: {cfg_cached}", debug=True)
                            sampler.extra_options['eta_substep']  = eta_substep_decay
                            sampler.extra_options['eta']          = eta_decay
                            sampler.extra_options['etas_substep'] = etas_substep_decay
                            sampler.extra_options['etas']         = etas_decay
                            sampler.extra_options['rk_type']      = rk_type_cached
                            if unsample_steps_to_run > -1:
                                sampler.extra_options['steps_to_run'] = unsample_steps_to_run
                            else:
                                sampler.extra_options['steps_to_run'] = steps_to_run_cached

                        sampler.extra_options['outer_sigmas_len'] = sigmas.shape[-1]
                        samples = guider.sample(noise, x_initial, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=-1)

                        eta_substep_decay   *= eta_decay_scale
                        eta_decay           *= eta_decay_scale
                        unsample_eta_decay  *= eta_decay_scale

                        etas_substep_decay  *= eta_decay_scale
                        etas_decay          *= eta_decay_scale
                        unsample_etas_decay *= eta_decay_scale

                    sampler.extra_options['noise_seed'] = noise_seed_cached
                    if has_cfgs:
                        guider.cfgs = cfgs_cached
                        RESplain(f"Shark: Rebounds complete - restored original CFG (cfgs)", debug=True)
                    else:
                        guider.cfg = cfg_cached
                        RESplain(f"Shark: Rebounds complete - restored original CFG: {cfg_cached}", debug=True)
                    sampler.extra_options['sampler_mode'] = sampler_mode_cached
                    sampler.extra_options['eta_substep']  = eta_substep_cached
                    sampler.extra_options['eta']          = eta_cached
                    sampler.extra_options['etas_substep'] = etas_substep_cached
                    sampler.extra_options['etas']         = etas_cached

                if noise_mask is not None:
                    if hasattr(samples, 'is_nested') and samples.is_nested:
                        blended = []
                        x_initial_list = x_initial.unbind() if hasattr(x_initial, 'is_nested') and x_initial.is_nested else [x_initial]
                        if hasattr(noise_mask, 'is_nested') and noise_mask.is_nested:
                            mask_list = noise_mask.unbind()
                        else:
                            mask_list = [noise_mask]
                        for idx, s in enumerate(samples.unbind()):
                            xi = x_initial_list[idx] if idx < len(x_initial_list) else x_initial_list[0]
                            m = mask_list[idx] if idx < len(mask_list) else mask_list[0]
                            reshaped_mask = comfy.utils.reshape_mask(m, s.shape).to(s.device)
                            blended.append(s * reshaped_mask + xi.to(s.device) * (1.0 - reshaped_mask))
                            # probably don't need to check ndim here, comfy utils handles it
                            # if s.ndim == m.ndim:
                            #     reshaped_mask = comfy.utils.reshape_mask(m, s.shape).to(s.device)
                            #     blended.append(s * reshaped_mask + xi.to(s.device) * (1.0 - reshaped_mask))
                            # else:
                            #     blended.append(s)
                        samples = comfy.nested_tensor.NestedTensor(blended)
                    else:
                        if hasattr(noise_mask, 'is_nested') and noise_mask.is_nested:
                            noise_mask = noise_mask.unbind()[0]
                        reshaped_mask = comfy.utils.reshape_mask(noise_mask, samples.shape).to(samples.device)
                        samples = samples * reshaped_mask + x_initial.to(samples.device) * (1.0 - reshaped_mask)

                samples = samples.to(comfy.model_management.intermediate_device())

                out = latent_x.copy()
                out["samples"] = samples

                if "x0" in x0_output:
                    x0_out = work_model.model.process_latent_out(x0_output["x0"].cpu())
                    if hasattr(samples, 'is_nested') and samples.is_nested:
                        latent_shapes = [t.shape for t in samples.unbind()]
                        x0_out = comfy.nested_tensor.NestedTensor(
                            comfy.utils.unpack_latents(x0_out, latent_shapes)
                        )
                    if hasattr(x0_out, 'is_nested') and x0_out.is_nested:
                        x0_out = comfy.nested_tensor.NestedTensor([t.to(torch.float32) for t in x0_out.unbind()])
                    else:
                        x0_out = x0_out.to(torch.float32)
                    out_denoised = latent_x.copy()
                    out_denoised["samples"] = x0_out
                else:
                    out_denoised = latent_x.copy()
                    if hasattr(samples, 'is_nested') and samples.is_nested:
                        out_denoised["samples"] = comfy.nested_tensor.NestedTensor([t.to(torch.float32) for t in samples.unbind()])
                    else:
                        out_denoised["samples"] = samples.to(torch.float32)
                        
                out['sampler'] = sampler
                out['guider'] = guider

                if noise_mask is not None:
                    state_info_out['image_initial'] = x_initial
                    state_info_out['noise_initial'] = noise_initial

                out['state_info'] = state_info_out

                return (out, out_denoised, None)

            # Old batch loop path, kept for reference but not supported
            else:
                raise NotImplementedError("The batch_loop is dead, long live no_batch_loop! Set use_batch_loop=False (or remove it) to use the supported code path.")
            
                out_samples          = []
                out_denoised_samples = []
                out_state_info       = []
                
                for batch_num in range(latent_image_batch['samples'].shape[0]):
                    latent_unbatch            = copy.deepcopy(latent_x)
                    if isinstance(latent_image_batch['samples'][batch_num], comfy.nested_tensor.NestedTensor):
                        latent_unbatch['samples'] = latent_image_batch['samples'][batch_num]._copy()
                    else:
                        latent_unbatch['samples'] = latent_image_batch['samples'][batch_num].clone().unsqueeze(0)
                    
                    if 'BONGMATH' in sampler.extra_options:
                        sampler.extra_options['batch_num'] = batch_num


                    if noise_seed == -1 and sampler_mode in {"unsample", "resample"}:
                        if latent_image.get('state_info', {}).get('last_rng', None) is not None:
                            seed = torch.initial_seed() + batch_num
                        else:
                            seed = torch.initial_seed() + 1 + batch_num
                    else:
                        if EO("lock_batch_seed"):
                            seed = noise_seed
                        else:
                            seed = noise_seed + batch_num
                        torch     .manual_seed(seed)
                        torch.cuda.manual_seed(seed)

                    x = latent_unbatch["samples"].to(default_dtype)



                    if sde_noise is None and sampler_mode.startswith("unsample"):
                        sde_noise = []
                    else:
                        sde_noise_steps = 1

                    for total_steps_iter in range (sde_noise_steps):

                        if noise_type_init != "none" and noise_stdev != 0.0:
                            RESplain("Initial latent noise seed: ", seed, debug=True)

                        noise = generate_init_noise(
                            x=x, seed=seed,
                            noise_type_init=noise_type_init, noise_stdev=noise_stdev,
                            noise_mean=noise_mean, noise_normalize=noise_normalize,
                            sigma_max=sigma_max, sigma_min=sigma_min,
                            alpha_init=alpha_init, k_init=k_init, EO=EO
                        )

                        noise_mask = latent_unbatch["noise_mask"] if "noise_mask" in latent_unbatch else None

                        x_input = x
                        if noise_mask is not None and 'noise_initial' in state_info:
                            stored_noise = state_info.get('noise_initial')
                            if stored_noise is not None:
                                if stored_noise.dim() > 3 and stored_noise.shape[0] > batch_num:
                                    stored_noise = stored_noise[batch_num:batch_num+1]
                                if stored_noise.shape == noise.shape:
                                    noise = stored_noise.to(noise.device, dtype=noise.dtype)
                                    RESplain("Using stored noise_initial from previous sampler", debug=True)

                            stored_image = state_info.get('image_initial')
                            if stored_image is not None:
                                if stored_image.dim() > 3 and stored_image.shape[0] > batch_num:
                                    stored_image = stored_image[batch_num:batch_num+1]
                                if stored_image.shape == x.shape:
                                    x_input = stored_image.to(x.device, dtype=x.dtype)
                                    RESplain("Using stored image_initial from previous sampler", debug=True)

                        if 'BONGMATH' in sampler.extra_options:
                            sampler.extra_options['noise_initial'] = noise
                            sampler.extra_options['image_initial'] = x_input

                        x0_output = {}

                        if latent_image is not None and 'state_info' in latent_image and 'sigmas' in latent_image['state_info']:
                            steps_len = max(sigmas.shape[-1] - 1,    latent_image['state_info']['sigmas'].shape[-1]-1)
                        else:
                            steps_len = sigmas.shape[-1]-1
                        callback     = latent_preview.prepare_callback(work_model, steps_len, x0_output)

                        if 'BONGMATH' in sampler.extra_options: # verify the sampler is rk_sampler_beta()
                            sampler.extra_options['state_info']     = copy.deepcopy(state_info)         ##############################
                            if state_info != {} and state_info != {'data_prev_': None}:  #second condition is for ultracascade
                                sampler.extra_options['state_info']['raw_x']            = state_info['raw_x']           [batch_num:batch_num+1]
                                sampler.extra_options['state_info']['data_prev_']       = state_info['data_prev_']      [batch_num]  # Use index, not slice - first dim is recycled_stages
                                sampler.extra_options['state_info']['last_rng']         = state_info['last_rng']        [batch_num]
                                sampler.extra_options['state_info']['last_rng_substep'] = state_info['last_rng_substep'][batch_num]
                                if 'image_initial' in state_info and state_info['image_initial'].dim() > 3:
                                    sampler.extra_options['state_info']['image_initial'] = state_info['image_initial'][batch_num:batch_num+1]
                                if 'noise_initial' in state_info and state_info['noise_initial'].dim() > 3:
                                    sampler.extra_options['state_info']['noise_initial'] = state_info['noise_initial'][batch_num:batch_num+1]
                            #state_info     = copy.deepcopy(latent_image['state_info']) if 'state_info' in latent_image else {}
                            state_info_out = {}
                            sampler.extra_options['state_info_out'] = state_info_out
                            
                        if type(pos_cond[0][0]) == list:
                            pos_cond_tmp = pos_cond[batch_num]
                            positive_tmp = positive[batch_num]
                        else:
                            pos_cond_tmp = pos_cond
                            positive_tmp = positive
                        
                        for i in range(len(neg_cond)): # crude fix for copy.deepcopy converting superclass into real object
                            if 'control' in neg_cond[i][1]:
                                neg_cond[i][1]['control']          = negative[i][1]['control']
                                if hasattr(negative[i][1]['control'], 'base'):
                                    neg_cond[i][1]['control'].base     = negative[i][1]['control'].base
                        for i in range(len(pos_cond_tmp)): # crude fix for copy.deepcopy converting superclass into real object
                            if 'control' in pos_cond_tmp[i][1]:
                                pos_cond_tmp[i][1]['control']      = positive_tmp[i][1]['control']
                                if hasattr(positive[i][1]['control'], 'base'):
                                    pos_cond_tmp[i][1]['control'].base = positive_tmp[i][1]['control'].base
                        
                        # SETUP REGIONAL COND
                        
                        if pos_cond_tmp[0][1] is not None: 
                            if 'callback_regional' in pos_cond_tmp[0][1]:
                                pos_cond_tmp = pos_cond_tmp[0][1]['callback_regional'](work_model)
                            
                            if 'AttnMask' in pos_cond_tmp[0][1]:
                                sampler.extra_options['AttnMask']   = pos_cond_tmp[0][1]['AttnMask']
                                sampler.extra_options['RegContext'] = pos_cond_tmp[0][1]['RegContext']
                                sampler.extra_options['RegParam']   = pos_cond_tmp[0][1]['RegParam']
                                
                                if isinstance(model.model.model_config, (comfy.supported_models.SDXL, comfy.supported_models.SD15)):
                                    latent_up_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] * 2, latent_image['samples'].shape[-1] * 2), mode="nearest")
                                    sampler.extra_options['AttnMask'].set_latent(latent_up_dummy)
                                    sampler.extra_options['AttnMask'].generate()
                                    sampler.extra_options['AttnMask'].mask_up   = sampler.extra_options['AttnMask'].attn_mask.mask
                                    
                                    latent_down_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] // 2, latent_image['samples'].shape[-1] // 2), mode="nearest")
                                    sampler.extra_options['AttnMask'].set_latent(latent_down_dummy)
                                    sampler.extra_options['AttnMask'].generate()
                                    sampler.extra_options['AttnMask'].mask_down = sampler.extra_options['AttnMask'].attn_mask.mask
                                    
                                    if isinstance(model.model.model_config, comfy.supported_models.SD15):
                                        latent_down_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] // 4, latent_image['samples'].shape[-1] // 4), mode="nearest")
                                        sampler.extra_options['AttnMask'].set_latent(latent_down_dummy)
                                        sampler.extra_options['AttnMask'].generate()
                                        sampler.extra_options['AttnMask'].mask_down2 = sampler.extra_options['AttnMask'].attn_mask.mask
                                        
                                if isinstance(model.model.model_config, (comfy.supported_models.Stable_Cascade_C)):
                                    latent_up_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] * 2, latent_image['samples'].shape[-1] * 2), mode="nearest")
                                    sampler.extra_options['AttnMask'].set_latent(latent_up_dummy)
                                    # cascade concats 4 + 4 tokens (clip_text_pooled, clip_img)
                                    sampler.extra_options['AttnMask'].context_lens = [context_len + 8 for context_len in sampler.extra_options['AttnMask'].context_lens] 
                                    sampler.extra_options['AttnMask'].text_len = sum(sampler.extra_options['AttnMask'].context_lens)
                                else:
                                    sampler.extra_options['AttnMask'].set_latent(latent_image['samples'])
                                sampler.extra_options['AttnMask'].generate()
                                
                        if neg_cond[0][1] is not None: 
                            if 'callback_regional' in neg_cond[0][1]:
                                neg_cond = neg_cond[0][1]['callback_regional'](work_model)
                            
                            if 'AttnMask' in neg_cond[0][1]:
                                sampler.extra_options['AttnMask_neg']   = neg_cond[0][1]['AttnMask']
                                sampler.extra_options['RegContext_neg'] = neg_cond[0][1]['RegContext']
                                sampler.extra_options['RegParam_neg']   = neg_cond[0][1]['RegParam']
                                
                                if isinstance(model.model.model_config, (comfy.supported_models.SDXL, comfy.supported_models.SD15)):
                                    latent_up_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] * 2, latent_image['samples'].shape[-1] * 2), mode="nearest")
                                    sampler.extra_options['AttnMask_neg'].set_latent(latent_up_dummy)
                                    sampler.extra_options['AttnMask_neg'].generate()
                                    sampler.extra_options['AttnMask_neg'].mask_up   = sampler.extra_options['AttnMask_neg'].attn_mask.mask
                                    
                                    latent_down_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] // 2, latent_image['samples'].shape[-1] // 2), mode="nearest")
                                    sampler.extra_options['AttnMask_neg'].set_latent(latent_down_dummy)
                                    sampler.extra_options['AttnMask_neg'].generate()
                                    sampler.extra_options['AttnMask_neg'].mask_down = sampler.extra_options['AttnMask_neg'].attn_mask.mask
                                    
                                    if isinstance(model.model.model_config, comfy.supported_models.SD15):
                                        latent_down_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] // 4, latent_image['samples'].shape[-1] // 4), mode="nearest")
                                        sampler.extra_options['AttnMask_neg'].set_latent(latent_down_dummy)
                                        sampler.extra_options['AttnMask_neg'].generate()
                                        sampler.extra_options['AttnMask_neg'].mask_down2 = sampler.extra_options['AttnMask_neg'].attn_mask.mask
                                
                                if isinstance(model.model.model_config, (comfy.supported_models.Stable_Cascade_C)):
                                    latent_up_dummy = F.interpolate(latent_image['samples'].to(torch.float16), size=(latent_image['samples'].shape[-2] * 2, latent_image['samples'].shape[-1] * 2), mode="nearest")
                                    sampler.extra_options['AttnMask'].set_latent(latent_up_dummy)
                                    # cascade concats 4 + 4 tokens (clip_text_pooled, clip_img)
                                    sampler.extra_options['AttnMask'].context_lens = [context_len + 8 for context_len in sampler.extra_options['AttnMask'].context_lens] 
                                    sampler.extra_options['AttnMask'].text_len = sum(sampler.extra_options['AttnMask'].context_lens)
                                else:
                                    sampler.extra_options['AttnMask_neg'].set_latent(latent_image['samples'])
                                sampler.extra_options['AttnMask_neg'].generate()
                        
                        
                        
                        
                        
                        if guider is None:
                            guider = SharkGuider(work_model)
                            flow_cond = options_mgr.get('flow_cond', {})
                            if flow_cond != {} and 'yt_positive' in flow_cond and not 'yt_inv_positive' in flow_cond:   #and not 'yt_inv;_positive' in flow_cond:   # typo???
                                guider.set_conds(yt_positive=flow_cond.get('yt_positive'), yt_negative=flow_cond.get('yt_negative'),)
                                guider.set_cfgs(yt=flow_cond.get('yt_cfg'), xt=cfg)
                            elif flow_cond != {} and 'yt_positive' in flow_cond and 'yt_inv_positive' in flow_cond:
                                guider.set_conds(yt_positive=flow_cond.get('yt_positive'), yt_negative=flow_cond.get('yt_negative'), yt_inv_positive=flow_cond.get('yt_inv_positive'), yt_inv_negative=flow_cond.get('yt_inv_negative'),)
                                guider.set_cfgs(yt=flow_cond.get('yt_cfg'), yt_inv=flow_cond.get('yt_inv_cfg'), xt=cfg)
                            else:
                                guider.set_cfgs(xt=cfg)
                            
                            guider.set_conds(positive=pos_cond_tmp, negative=neg_cond)

                        elif type(guider) == SharkGuider:
                            guider.set_cfgs(xt=cfg)
                            guider.set_conds(positive=pos_cond_tmp, negative=neg_cond)
                        else:
                            try:
                                guider.set_cfg(cfg)
                            except:
                                RESplain("SharkWarning: guider.set_cfg failed but assuming cfg already set correctly.")
                            try:
                                guider.set_conds(pos_cond_tmp, neg_cond)
                            except:
                                RESplain("SharkWarning: guider.set_conds failed but assuming conds already set correctly.")
                        
                        if rebounds > 0:
                            cfgs_cached = guider.cfgs
                            steps_to_run_cached = sampler.extra_options['steps_to_run']
                            eta_cached         = sampler.extra_options['eta']
                            eta_substep_cached = sampler.extra_options['eta_substep']
                            
                            etas_cached         = sampler.extra_options['etas'].clone()
                            etas_substep_cached = sampler.extra_options['etas_substep'].clone()
                            
                            unsample_etas = torch.full_like(etas_cached, unsample_eta)
                            rk_type_cached = sampler.extra_options['rk_type']
                            
                            if sampler.extra_options['sampler_mode'] == "unsample":
                                guider.cfgs = {
                                    'xt': unsample_cfg,
                                    'yt': unsample_cfg,
                                }
                                if unsample_eta != -1.0:
                                    sampler.extra_options['eta_substep']  = unsample_eta
                                    sampler.extra_options['eta']          = unsample_eta
                                    sampler.extra_options['etas_substep'] = unsample_etas
                                    sampler.extra_options['etas']         = unsample_etas
                                if unsampler_name != "none":
                                    sampler.extra_options['rk_type']      = unsampler_name
                                if unsample_steps_to_run > -1:
                                    sampler.extra_options['steps_to_run'] = unsample_steps_to_run
                                    
                            else:
                                guider.cfgs = cfgs_cached
                            
                            guider.cfgs = cfgs_cached
                            sampler.extra_options['steps_to_run'] = steps_to_run_cached
                            
                            eta_decay           = eta_cached
                            eta_substep_decay   = eta_substep_cached
                            unsample_eta_decay  = unsample_eta
                            
                            etas_decay          = etas_cached
                            etas_substep_decay  = etas_substep_cached
                            unsample_etas_decay = unsample_etas

                        # Apply pre-sampling latent normalization for NestedTensors
                        idx_0_factor = options_mgr.get('latent_normalize_idx_0', 1.0)
                        idx_1_factor = options_mgr.get('latent_normalize_idx_1', 1.0)
                        factors_0_steps = options_mgr.get('latent_normalize_idx_0_steps', [1.0])
                        factors_1_steps = options_mgr.get('latent_normalize_idx_1_steps', [1.0])
                        RESplain(f"Latent normalize check: idx_0={idx_0_factor}, idx_1={idx_1_factor}, x_input.is_nested={hasattr(x_input, 'is_nested') and x_input.is_nested}", debug=True)

                        # Per-step normalization: pass shapes and factors to inner sampler
                        has_per_step_factors = len(factors_0_steps) > 1 or len(factors_1_steps) > 1
                        if isinstance(x_input, comfy.nested_tensor.NestedTensor) and 'rk_type' in sampler.extra_options:
                            sampler.extra_options['latent_shapes'] = [t.shape for t in x_input.unbind()]
                            sampler.extra_options['latent_normalize_idx_0_steps'] = factors_0_steps
                            sampler.extra_options['latent_normalize_idx_1_steps'] = factors_1_steps

                        # Node-level normalization only for single-value factors (backward compat)
                        # Per-step factors are applied in the inner loop instead
                        if not has_per_step_factors and (idx_0_factor != 1.0 or idx_1_factor != 1.0):
                            x_input = apply_nested_normalization(x_input, idx_0_factor, idx_1_factor)

                        sampler.extra_options['outer_sigmas_len'] = sigmas.shape[-1]
                        if isinstance(x_input, comfy.nested_tensor.NestedTensor):
                            samples = guider.sample(noise, x_input._copy(), sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
                        else:
                            samples = guider.sample(noise, x_input.clone(), sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
                        
                        if rebounds > 0: 
                            noise_seed_cached   = sampler.extra_options['noise_seed']
                            cfgs_cached         = guider.cfgs
                            sampler_mode_cached = sampler.extra_options['sampler_mode']
                            
                            for restarts_iter in range(rebounds):
                                sampler.extra_options['state_info'] = sampler.extra_options['state_info_out']
                                
                                #steps = sampler.extra_options['state_info_out']['sigmas'].shape[-1] - 3
                                sigmas = sampler.extra_options['state_info_out']['sigmas'] if sigmas is None else sigmas
                                #if len(sigmas) > 2 and sigmas[1] < sigmas[2] and sampler.extra_options['state_info_out']['sampler_mode'] == "unsample": # and sampler_mode == "resample":
                                #    sigmas = torch.flip(sigmas, dims=[0])
                                    
                                if   sampler.extra_options['sampler_mode'] == "standard":
                                    sampler.extra_options['sampler_mode'] = "unsample"
                                elif sampler.extra_options['sampler_mode'] == "unsample":
                                    sampler.extra_options['sampler_mode'] = "resample"
                                elif sampler.extra_options['sampler_mode'] == "resample":
                                    sampler.extra_options['sampler_mode'] = "unsample"
                                
                                sampler.extra_options['noise_seed'] = -1
                                
                                if sampler.extra_options['sampler_mode'] == "unsample":
                                    guider.cfgs = {
                                        'xt': unsample_cfg,
                                        'yt': unsample_cfg,
                                    }
                                    if unsample_eta != -1.0:
                                        sampler.extra_options['eta_substep']  = unsample_eta_decay
                                        sampler.extra_options['eta']          = unsample_eta_decay
                                        sampler.extra_options['etas_substep'] = unsample_etas
                                        sampler.extra_options['etas']         = unsample_etas
                                    else:
                                        sampler.extra_options['eta_substep']  = eta_substep_decay
                                        sampler.extra_options['eta']          = eta_decay
                                        sampler.extra_options['etas_substep'] = etas_substep_decay
                                        sampler.extra_options['etas']         = etas_decay
                                    if unsampler_name != "none":
                                        sampler.extra_options['rk_type']  = unsampler_name
                                    if unsample_steps_to_run > -1:
                                        sampler.extra_options['steps_to_run'] = unsample_steps_to_run
                                else:
                                    guider.cfgs = cfgs_cached
                                    sampler.extra_options['eta_substep']  = eta_substep_decay
                                    sampler.extra_options['eta']          = eta_decay
                                    sampler.extra_options['etas_substep'] = etas_substep_decay
                                    sampler.extra_options['etas']         = etas_decay
                                    sampler.extra_options['rk_type']      = rk_type_cached
                                    if unsample_steps_to_run > -1:
                                        sampler.extra_options['steps_to_run'] = unsample_steps_to_run
                                    else:
                                        sampler.extra_options['steps_to_run'] = steps_to_run_cached


                                sampler.extra_options['outer_sigmas_len'] = sigmas.shape[-1]
                                samples = guider.sample(noise, samples.clone(), sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=-1)

                                eta_substep_decay   *= eta_decay_scale
                                eta_decay           *= eta_decay_scale
                                unsample_eta_decay  *= eta_decay_scale
                                
                                etas_substep_decay  *= eta_decay_scale
                                etas_decay          *= eta_decay_scale
                                unsample_etas_decay *= eta_decay_scale                        
                            
                            sampler.extra_options['noise_seed'] = noise_seed_cached
                            guider.cfgs = cfgs_cached
                            sampler.extra_options['sampler_mode'] = sampler_mode_cached
                            sampler.extra_options['eta_substep']  = eta_substep_cached
                            sampler.extra_options['eta']          = eta_cached
                            sampler.extra_options['etas_substep'] = etas_substep_cached
                            sampler.extra_options['etas']         = etas_cached
                            sampler.extra_options['rk_type']      = rk_type_cached
                            sampler.extra_options['steps_to_run'] = steps_to_run_cached   # TODO: verify this is carried on

                        if noise_mask is not None:
                            if 'BONGMATH' in sampler.extra_options:
                                batch_state_info = sampler.extra_options.get('state_info', {})
                                latent_for_mask = batch_state_info.get('image_initial', x)
                            else:
                                stored_image = state_info.get('image_initial')
                                if stored_image is not None and stored_image.dim() > 3:
                                    latent_for_mask = stored_image[batch_num]
                                elif stored_image is not None:
                                    latent_for_mask = stored_image
                                else:
                                    latent_for_mask = x
                            reshaped_mask = comfy.utils.reshape_mask(noise_mask, samples.shape).to(samples.device)
                            samples = samples * reshaped_mask + latent_for_mask.to(samples.device) * (1.0 - reshaped_mask)

                        out = latent_unbatch.copy()
                        out["samples"] = samples
                        
                        if "x0" in x0_output:
                            out_denoised            = latent_unbatch.copy()
                            out_denoised["samples"] = work_model.model.process_latent_out(x0_output["x0"].cpu())
                        else:
                            out_denoised            = out

                        out_samples         .append(out         ["samples"])
                        out_denoised_samples.append(out_denoised["samples"])
                        
                        
                        
                        # ACCUMULATE UNSAMPLED SDE NOISE
                        if total_steps_iter > 1: 
                            if 'raw_x' in state_info_out:
                                sde_noise_out = state_info_out['raw_x']
                            else:
                                sde_noise_out = out["samples"]  
                            sde_noise.append(normalize_zscore(sde_noise_out, channelwise=True, inplace=True))    
                        
                        out_state_info.append(state_info_out)
                        
                        # INCREMENT BATCH LOOP
                        if not EO("lock_batch_seed"):
                            seed += 1
                        if latent_image is not None: #needed for ultracascade, where latent_image input is not really used for stage C/first stage
                            if latent_image.get('state_info', {}).get('last_rng', None) is None:
                                torch.manual_seed(seed)


                gc.collect()

                # CAT SDE NOISES, SAVE STATE INFO
                state_info_out = out_state_info[0]
                if 'raw_x' in out_state_info[0]:
                    state_info_out['raw_x']            = torch.cat([out_state_info[_]['raw_x']            for _ in range(len(out_state_info))], dim=0)
                    state_info_out['data_prev_']       = torch.stack([out_state_info[_]['data_prev_']       for _ in range(len(out_state_info))])  # Keep stack - first dim is recycled_stages, not batch
                    state_info_out['last_rng']         = torch.stack([out_state_info[_]['last_rng']         for _ in range(len(out_state_info))])  # Keep stack - 1D RNG state
                    state_info_out['last_rng_substep'] = torch.stack([out_state_info[_]['last_rng_substep'] for _ in range(len(out_state_info))])  # Keep stack - 1D RNG state
                    if 'image_initial' in out_state_info[0]:
                        state_info_out['image_initial'] = torch.cat([out_state_info[_]['image_initial'] for _ in range(len(out_state_info))], dim=0)
                    if 'noise_initial' in out_state_info[0]:
                        state_info_out['noise_initial'] = torch.cat([out_state_info[_]['noise_initial'] for _ in range(len(out_state_info))], dim=0)
                elif 'raw_x' in state_info:
                    state_info_out = state_info

                out_samples             = [tensor.squeeze(0) for tensor in out_samples]
                out_denoised_samples    = [tensor.squeeze(0) for tensor in out_denoised_samples]

                out         ['samples'] = torch.stack(out_samples,          dim=0)
                out_denoised['samples'] = torch.stack(out_denoised_samples, dim=0)

                out['state_info']       = copy.deepcopy(state_info_out)
                state_info              = {}

                out['sampler']  = sampler
                out['guider']   = guider

                return (out, out_denoised, sde_noise,)
                



class SharkSampler_Beta(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharkSampler_Beta",
            display_name="SharkSampler",
            category="RES4LYF/samplers",
            inputs=[
                io.Combo.Input("scheduler", options=get_res4lyf_scheduler_list(), default="beta57"),
                io.Int.Input("steps", default=30, min=1, max=10000),
                io.Int.Input("steps_to_run", default=-1, min=-1, max=MAX_STEPS),
                io.Float.Input("denoise", default=1.0, min=-10000.0, max=10000.0, step=0.01),
                io.Float.Input("cfg", default=5.5, min=-10000.0, max=10000.0, step=0.01, round=False,
                               tooltip="Negative values use channelwise CFG."),
                io.Int.Input("seed", default=0, min=-1, max=0xffffffffffffffff),
                io.Combo.Input("sampler_mode", options=["unsample", "standard", "resample"], default="standard"),
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Sampler.Input("sampler", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                io.Latent.Input("latent_image", optional=True),
                io.Autogrow.Input(
                    "options_group",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Custom("OPTIONS").Input("options", optional=True),
                        prefix="options",
                        min=0,
                        max=20,
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised"),
                io.Custom("OPTIONS").Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls,
                scheduler="beta57",
                steps=30,
                steps_to_run=-1,
                denoise=1.0,
                cfg=5.5,
                seed=0,
                sampler_mode="standard",
                model=None,
                positive=None,
                negative=None,
                sampler=None,
                sigmas=None,
                latent_image=None,
                options_group=None,
                **kwargs):

        options_mgr = OptionsManager(options_group=options_group, **kwargs)
        first_options = options_mgr.options_list[0] if options_mgr.options_list else None

        denoise_alt = 1.0
        if denoise < 0:
            denoise_alt = -denoise
            denoise = 1.0

        if latent_image is not None and 'sampler' in latent_image and sampler is None:
            sampler = latent_image['sampler']

        output, denoised, sde_noise = SharkSampler().main(
            model           = model,
            cfg             = cfg,
            scheduler       = scheduler,
            steps           = steps,
            steps_to_run    = steps_to_run,
            denoise         = denoise,
            latent_image    = latent_image,
            positive        = positive,
            negative        = negative,
            sampler         = sampler,
            cfgpp           = 0.0,
            noise_seed      = seed,
            options         = first_options,
            sde_noise       = None,
            sde_noise_steps = 1,
            noise_type_init = "gaussian",
            noise_stdev     = 1.0,
            sampler_mode    = sampler_mode,
            denoise_alt     = denoise_alt,
            sigmas          = sigmas,
            extra_options   = "",
        )

        return io.NodeOutput(output, denoised, options_mgr.as_dict())





class SharkChainsampler_Beta(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SharkChainsampler_Beta",
            display_name="SharkChainsampler",
            category="RES4LYF/samplers",
            inputs=[
                io.Int.Input("steps_to_run", default=-1, min=-1, max=MAX_STEPS),
                io.Float.Input("cfg", default=5.5, min=-10000.0, max=10000.0, step=0.01, round=False,
                               tooltip="Negative values use channelwise CFG."),
                io.Combo.Input("sampler_mode", options=["unsample", "resample"], default="resample"),
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Sampler.Input("sampler", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                io.Latent.Input("latent_image", optional=True),
                io.Autogrow.Input(
                    "options_group",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Custom("OPTIONS").Input("options", optional=True),
                        prefix="options",
                        min=0,
                        max=20,
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised"),
                io.Custom("OPTIONS").Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls,
                steps_to_run=-1,
                cfg=5.5,
                sampler_mode="resample",
                model=None,
                positive=None,
                negative=None,
                sampler=None,
                sigmas=None,
                latent_image=None,
                options_group=None,
                **kwargs):

        steps = latent_image['state_info']['sigmas'].shape[-1] - 3
        sigmas = latent_image['state_info']['sigmas'] if sigmas is None else sigmas
        if len(sigmas) > 2 and sigmas[1] < sigmas[2] and latent_image['state_info']['sampler_mode'] == "unsample" and sampler_mode == "resample":
            sigmas = torch.flip(sigmas, dims=[0])

        return SharkSampler_Beta.execute(
            model=model,
            sampler_mode=sampler_mode,
            steps_to_run=steps_to_run,
            sigmas=sigmas,
            steps=steps,
            cfg=cfg,
            seed=-1,
            latent_image=latent_image,
            positive=positive,
            negative=negative,
            sampler=sampler,
            options_group=options_group,
            **{k: v for k, v in kwargs.items() if isinstance(k, str) and k.startswith('options')},
        )





class ClownSamplerAdvanced_Beta(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ClownSamplerAdvanced_Beta",
            display_name="ClownSamplerAdvanced",
            category="RES4LYF/samplers",
            is_experimental=True,
            inputs=[
                io.Combo.Input("noise_type_sde", options=NOISE_GENERATOR_NAMES_SIMPLE, default="gaussian"),
                io.Combo.Input("noise_type_sde_substep", options=NOISE_GENERATOR_NAMES_SIMPLE, default="gaussian"),
                io.Combo.Input("noise_mode_sde", options=NOISE_MODE_NAMES, default="hard",
                               tooltip="How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."),
                io.Combo.Input("noise_mode_sde_substep", options=NOISE_MODE_NAMES, default="hard",
                               tooltip="How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."),
                io.Combo.Input("overshoot_mode", options=NOISE_MODE_NAMES, default="hard",
                               tooltip="How step size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."),
                io.Combo.Input("overshoot_mode_substep", options=NOISE_MODE_NAMES, default="hard",
                               tooltip="How substep size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."),
                io.Float.Input("eta", default=0.5, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Calculated noise amount to be added, then removed, after each step."),
                io.Float.Input("eta_substep", default=0.5, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Calculated noise amount to be added, then removed, after each step."),
                io.Float.Input("overshoot", default=0.0, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Boost the size of each denoising step, then rescale to match the original. Has a softening effect."),
                io.Float.Input("overshoot_substep", default=0.0, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Boost the size of each denoising substep, then rescale to match the original. Has a softening effect."),
                io.Float.Input("noise_scaling_weight", default=0.0, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."),
                io.Float.Input("noise_boost_step", default=0.0, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."),
                io.Float.Input("noise_boost_substep", default=0.0, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."),
                io.Float.Input("noise_anchor", default=1.0, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."),
                io.Float.Input("s_noise", default=1.0, min=-10000.0, max=10000.0, step=0.01,
                               tooltip="Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."),
                io.Float.Input("s_noise_substep", default=1.0, min=-10000.0, max=10000.0, step=0.01,
                               tooltip="Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."),
                io.Float.Input("d_noise", default=1.0, min=-10000.0, max=10000.0, step=0.01,
                               tooltip="Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."),
                io.Float.Input("momentum", default=1.0, min=-10000.0, max=10000.0, step=0.01,
                               tooltip="Accelerate convergence with positive values when sampling, negative values when unsampling."),
                io.Int.Input("noise_seed_sde", default=-1, min=-1, max=0xffffffffffffffff),
                io.Combo.Input("sampler_name", options=get_sampler_name_list(), default=get_default_sampler_name()),
                io.Combo.Input("implicit_type", options=IMPLICIT_TYPE_NAMES, default="predictor-corrector"),
                io.Combo.Input("implicit_type_substeps", options=IMPLICIT_TYPE_NAMES, default="predictor-corrector"),
                io.Int.Input("implicit_steps", default=0, min=0, max=10000),
                io.Int.Input("implicit_substeps", default=0, min=0, max=10000),
                io.Boolean.Input("bongmath", default=True),
                io.Custom("GUIDES").Input("guides", optional=True),
                io.Custom("AUTOMATION").Input("automation", optional=True),
                io.String.Input("extra_options", optional=True, multiline=True, default=""),
                io.Autogrow.Input(
                    "options_group",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Custom("OPTIONS").Input("options", optional=True),
                        prefix="options",
                        min=0,
                        max=20,
                    ),
                ),
            ],
            outputs=[
                io.Sampler.Output(display_name="sampler"),
            ],
        )

    @classmethod
    def execute(cls,
            noise_type_sde                : str = "gaussian",
            noise_type_sde_substep        : str = "gaussian",
            noise_mode_sde                : str = "hard",
            overshoot_mode                : str = "hard",
            overshoot_mode_substep        : str = "hard",
            
            eta                           : float = 0.5,
            eta_substep                   : float = 0.5,
            momentum                      : float = 0.0,



            noise_scaling_weight          : float = 0.0,
            noise_scaling_type            : str   = "sampler",
            noise_scaling_mode            : str   = "linear",
            noise_scaling_eta             : float = 0.0,
            noise_scaling_cycles          : int   = 1,
            
            noise_scaling_weights         : Optional[Tensor]       = None,
            noise_scaling_etas            : Optional[Tensor]       = None,
            
            noise_boost_step              : float = 0.0,
            noise_boost_substep           : float = 0.0,
            noise_boost_normalize         : bool  = True,
            noise_anchor                  : float = 1.0,
            
            s_noise                       : float = 1.0,
            s_noise_substep               : float = 1.0,
            d_noise                       : float = 1.0,
            d_noise_start_step            : int   = 0,
            d_noise_inv                   : float = 1.0,
            d_noise_inv_start_step        : int   = 0,
            
            
            
            alpha_sde                     : float = -1.0,
            k_sde                         : float = 1.0,
            cfgpp                         : float = 0.0,
            c1                            : float = 0.0,
            c2                            : float = 0.5,
            c3                            : float = 1.0,
            noise_seed_sde                : int = -1,
            sampler_name                  : str = "res_2m",
            implicit_sampler_name         : str = "gauss-legendre_2s",
            
            implicit_substeps             : int = 0,
            implicit_steps                : int = 0,
            
            rescale_floor                 : bool = True,
            sigmas_override               : Optional[Tensor] = None,

            guides                        = None,
            options_group                 = None,
            sde_noise                     = None,
            sde_noise_steps               : int = 1,
            
            extra_options                 : str = "",
            automation                    = None,
            etas                          : Optional[Tensor] = None,
            etas_substep                  : Optional[Tensor] = None,
            s_noises                      : Optional[Tensor] = None,
            s_noises_substep              : Optional[Tensor] = None,
            epsilon_scales                : Optional[Tensor] = None,
            regional_conditioning_weights : Optional[Tensor] = None,
            frame_weights_mgr             = None,
            noise_mode_sde_substep        : str = "hard",
            
            overshoot                     : float = 0.0,
            overshoot_substep             : float = 0.0,

            bongmath                      : bool = True,
            
            implicit_type                 : str = "predictor-corrector",
            implicit_type_substeps        : str = "predictor-corrector",
            
            rk_swaps                      : list = [],
            
            steps_to_run                  : int = -1,
            
            sde_mask                      : Optional[Tensor] = None,
            
            **kwargs,
            ):

            options_mgr = OptionsManager(options_group=options_group, **kwargs)
            extra_options    += "\n" + options_mgr.get('extra_options', "")
            EO = ExtraOptions(extra_options)
            default_dtype = EO("default_dtype", torch.float64)

    
    
            sampler_name, implicit_sampler_name = process_sampler_name(sampler_name)

            implicit_steps_diag = implicit_substeps
            implicit_steps_full = implicit_steps

            if noise_mode_sde == "none":
                eta = 0.0
                noise_mode_sde = "hard"

            noise_type_sde    = options_mgr.get('noise_type_sde'   , noise_type_sde)
            noise_mode_sde    = options_mgr.get('noise_mode_sde'   , noise_mode_sde)
            eta               = options_mgr.get('eta'              , eta)
            eta_substep       = options_mgr.get('eta_substep'      , eta_substep)
            
            
            
            noise_scaling_weight   = options_mgr.get('noise_scaling_weight'  , noise_scaling_weight)
            noise_scaling_type     = options_mgr.get('noise_scaling_type'    , noise_scaling_type)
            noise_scaling_mode     = options_mgr.get('noise_scaling_mode'    , noise_scaling_mode)
            noise_scaling_eta      = options_mgr.get('noise_scaling_eta'     , noise_scaling_eta)
            noise_scaling_cycles   = options_mgr.get('noise_scaling_cycles'  , noise_scaling_cycles)
            
            noise_scaling_weights  = options_mgr.get('noise_scaling_weights' , noise_scaling_weights)
            noise_scaling_etas     = options_mgr.get('noise_scaling_etas'    , noise_scaling_etas)
            
            noise_boost_step       = options_mgr.get('noise_boost_step'      , noise_boost_step)
            noise_boost_substep    = options_mgr.get('noise_boost_substep'   , noise_boost_substep)
            noise_boost_normalize  = options_mgr.get('noise_boost_normalize' , noise_boost_normalize)
            noise_anchor           = options_mgr.get('noise_anchor'          , noise_anchor)
            
            s_noise                = options_mgr.get('s_noise'               , s_noise)
            s_noise_substep        = options_mgr.get('s_noise_substep'       , s_noise_substep)
            d_noise                = options_mgr.get('d_noise'               , d_noise)
            d_noise_start_step     = options_mgr.get('d_noise_start_step'    , d_noise_start_step)
            d_noise_inv            = options_mgr.get('d_noise_inv'           , d_noise_inv)
            d_noise_inv_start_step = options_mgr.get('d_noise_inv_start_step', d_noise_inv_start_step)
            
            
            
            alpha_sde         = options_mgr.get('alpha_sde'        , alpha_sde)
            k_sde             = options_mgr.get('k_sde'            , k_sde)
            c1                = options_mgr.get('c1'               , c1)
            c2                = options_mgr.get('c2'               , c2)
            c3                = options_mgr.get('c3'               , c3)

            frame_weights_mgr = options_mgr.get('frame_weights_mgr', frame_weights_mgr)
            sde_noise         = options_mgr.get('sde_noise'        , sde_noise)
            sde_noise_steps   = options_mgr.get('sde_noise_steps'  , sde_noise_steps)
            
            rk_swaps = options_mgr.get('rk_swaps', rk_swaps)
            if not rk_swaps:
                _swap_type = options_mgr.get('rk_swap_type', '')
                if _swap_type:
                    rk_swaps = [{'type': _swap_type, 'step': options_mgr.get('rk_swap_step', MAX_STEPS),
                                 'threshold': options_mgr.get('rk_swap_threshold', 0.0), 'print': options_mgr.get('rk_swap_print', False)}]

            steps_to_run      = options_mgr.get('steps_to_run'     , steps_to_run)
            
            noise_seed_sde    = options_mgr.get('noise_seed_sde'   , noise_seed_sde)
            momentum          = options_mgr.get('momentum'         , momentum)

            sde_mask          = options_mgr.get('sde_mask'         , sde_mask)


            rescale_floor = EO("rescale_floor")

            if automation is not None:
                etas              = automation['etas']              if 'etas'              in automation else None
                etas_substep      = automation['etas_substep']      if 'etas_substep'      in automation else None
                s_noises          = automation['s_noises']          if 's_noises'          in automation else None
                s_noises_substep  = automation['s_noises_substep']  if 's_noises_substep'  in automation else None
                epsilon_scales    = automation['epsilon_scales']    if 'epsilon_scales'    in automation else None
                frame_weights_mgr = automation['frame_weights_mgr'] if 'frame_weights_mgr' in automation else None
                
            etas             = options_mgr.get('etas',         etas)
            etas_substep     = options_mgr.get('etas_substep', etas_substep)
            
            s_noises         = options_mgr.get('s_noises',         s_noises)
            s_noises_substep = options_mgr.get('s_noises_substep', s_noises_substep)

            etas             = initialize_or_scale(etas,             eta,             MAX_STEPS).to(default_dtype)
            etas_substep     = initialize_or_scale(etas_substep,     eta_substep,     MAX_STEPS).to(default_dtype)
            s_noises         = initialize_or_scale(s_noises,         s_noise,         MAX_STEPS).to(default_dtype)
            s_noises_substep = initialize_or_scale(s_noises_substep, s_noise_substep, MAX_STEPS).to(default_dtype)

            etas             = F.pad(etas,             (0, MAX_STEPS), value=0.0)
            etas_substep     = F.pad(etas_substep,     (0, MAX_STEPS), value=0.0)
            s_noises         = F.pad(s_noises,         (0, MAX_STEPS), value=1.0)
            s_noises_substep = F.pad(s_noises_substep, (0, MAX_STEPS), value=1.0)

            if sde_noise is None:
                sde_noise = []
            else:
                sde_noise = copy.deepcopy(sde_noise)
                sde_noise = normalize_zscore(sde_noise, channelwise=True, inplace=True)


            sampler = comfy.samplers.ksampler("rk_beta", 
                {
                    "eta"                           : eta,
                    "eta_substep"                   : eta_substep,

                    "alpha"                         : alpha_sde,
                    "k"                             : k_sde,
                    "c1"                            : c1,
                    "c2"                            : c2,
                    "c3"                            : c3,
                    "cfgpp"                         : cfgpp,

                    "noise_sampler_type"            : noise_type_sde,
                    "noise_sampler_type_substep"    : noise_type_sde_substep,
                    "noise_mode_sde"                : noise_mode_sde,
                    "noise_seed"                    : noise_seed_sde,
                    "rk_type"                       : sampler_name,
                    "implicit_sampler_name"         : implicit_sampler_name,

                    "implicit_steps_diag"           : implicit_steps_diag,
                    "implicit_steps_full"           : implicit_steps_full,

                    "LGW_MASK_RESCALE_MIN"          : rescale_floor,
                    "sigmas_override"               : sigmas_override,
                    "sde_noise"                     : sde_noise,

                    "extra_options"                 : extra_options,
                    "sampler_mode"                  : "standard",

                    "etas"                          : etas,
                    "etas_substep"                  : etas_substep,
                    
                    
                    
                    "s_noises"                      : s_noises,
                    "s_noises_substep"              : s_noises_substep,
                    "epsilon_scales"                : epsilon_scales,
                    "regional_conditioning_weights" : regional_conditioning_weights,

                    "guides"                        : guides,
                    "frame_weights_mgr"             : frame_weights_mgr,
                    "eta_substep"                   : eta_substep,
                    "noise_mode_sde_substep"        : noise_mode_sde_substep,
                    
                    
                    
                    "noise_scaling_weight"          : noise_scaling_weight,
                    "noise_scaling_type"            : noise_scaling_type,
                    "noise_scaling_mode"            : noise_scaling_mode,
                    "noise_scaling_eta"             : noise_scaling_eta,
                    "noise_scaling_cycles"          : noise_scaling_cycles,
                    
                    "noise_scaling_weights"         : noise_scaling_weights,
                    "noise_scaling_etas"            : noise_scaling_etas,
                    
                    "noise_boost_step"              : noise_boost_step,
                    "noise_boost_substep"           : noise_boost_substep,
                    "noise_boost_normalize"         : noise_boost_normalize,
                    "noise_anchor"                  : noise_anchor,
                    
                    "s_noise"                       : s_noise,
                    "s_noise_substep"               : s_noise_substep,
                    "d_noise"                       : d_noise,
                    "d_noise_start_step"            : d_noise_start_step,
                    "d_noise_inv"                   : d_noise_inv,
                    "d_noise_inv_start_step"        : d_noise_inv_start_step,



                    "overshoot_mode"                : overshoot_mode,
                    "overshoot_mode_substep"        : overshoot_mode_substep,
                    "overshoot"                     : overshoot,
                    "overshoot_substep"             : overshoot_substep,
                    "BONGMATH"                      : bongmath,

                    "implicit_type"                 : implicit_type,
                    "implicit_type_substeps"        : implicit_type_substeps,

                    "rk_swaps"                      : rk_swaps,
                    
                    "steps_to_run"                  : steps_to_run,
                    
                    "sde_mask"                      : sde_mask,
                    
                    "momentum"                      : momentum,
                })


            return io.NodeOutput(sampler)







class ClownsharKSampler_Beta(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ClownsharKSampler_Beta",
            display_name="ClownsharKSampler",
            category="RES4LYF/samplers",
            inputs=[
                io.Float.Input("eta", default=0.5, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Calculated noise amount to be added, then removed, after each step."),
                io.Combo.Input("sampler_name", options=get_sampler_name_list(), default=get_default_sampler_name()),
                io.Combo.Input("scheduler", options=get_res4lyf_scheduler_list(), default="beta57"),
                io.Int.Input("steps", default=30, min=1, max=MAX_STEPS),
                io.Int.Input("steps_to_run", default=-1, min=-1, max=MAX_STEPS),
                io.Float.Input("denoise", default=1.0, min=-10000.0, max=float(MAX_STEPS), step=0.01),
                io.Float.Input("cfg", default=5.5, min=-100.0, max=100.0, step=0.01, round=False),
                io.Int.Input("seed", default=0, min=-1, max=0xffffffffffffffff),
                io.Combo.Input("sampler_mode", options=["unsample", "standard", "resample"], default="standard"),
                io.Boolean.Input("bongmath", default=True),
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent_image", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                io.Custom("GUIDES").Input("guides", optional=True),
                io.Autogrow.Input(
                    "options_group",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Custom("OPTIONS").Input("options", optional=True),
                        prefix="options",
                        min=0,
                        max=20,
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised"),
                io.Custom("OPTIONS").Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls,
            model                                                  = None,
            denoise                       : float                  = 1.0, 
            scheduler                     : str                    = "beta57", 
            cfg                           : float                  = 1.0, 
            seed                          : int                    = -1, 
            positive                                               = None, 
            negative                                               = None, 
            latent_image                  : Optional[dict[Tensor]] = None, 
            steps                         : int                    = 30,
            steps_to_run                  : int                    = -1,
            bongmath                      : bool                   = True,
            sampler_mode                  : str                    = "standard",
            
            noise_type_sde                : str                    = "gaussian", 
            noise_type_sde_substep        : str                    = "gaussian", 
            noise_mode_sde                : str                    = "hard",
            noise_mode_sde_substep        : str                    = "hard",

            
            overshoot_mode                : str                    = "hard", 
            overshoot_mode_substep        : str                    = "hard",
            overshoot                     : float                  = 0.0, 
            overshoot_substep             : float                  = 0.0,
            
            eta                           : float                  = 0.5, 
            eta_substep                   : float                  = 0.5,
            momentum                      : float                  = 0.0,
            
            
            
            noise_scaling_weight         : float                  = 0.0,
            noise_scaling_type            : str                    = "sampler",
            noise_scaling_mode            : str                    = "linear",
            noise_scaling_eta             : float                  = 0.0,
            noise_scaling_cycles          : int                    = 1,
            
            noise_scaling_weights         : Optional[Tensor]       = None,
            noise_scaling_etas            : Optional[Tensor]       = None,
            
            noise_boost_step              : float                  = 0.0,
            noise_boost_substep           : float                  = 0.0,
            noise_boost_normalize         : bool                   = True,
            noise_anchor                  : float                  = 1.0,
            
            s_noise                       : float                  = 1.0,
            s_noise_substep               : float                  = 1.0,
            d_noise                       : float                  = 1.0,
            d_noise_start_step            : int                    = 0,
            d_noise_inv                   : float                  = 1.0,
            d_noise_inv_start_step        : int                    = 0,
            
            
            
            alpha_sde                     : float                  = -1.0, 
            k_sde                         : float                  = 1.0,
            cfgpp                         : float                  = 0.0,
            c1                            : float                  = 0.0, 
            c2                            : float                  = 0.5, 
            c3                            : float                  = 1.0,
            noise_seed_sde                : int                    = -1,
            sampler_name                  : str                    = "res_2m", 
            implicit_sampler_name         : str                    = "use_explicit",
            
            implicit_type                 : str                    = "bongmath",
            implicit_type_substeps        : str                    = "bongmath",
            implicit_steps                : int                    = 0,
            implicit_substeps             : int                    = 0, 

            sigmas                        : Optional[Tensor]       = None,
            sigmas_override               : Optional[Tensor]       = None,
            guides                                                 = None,
            options_group                                          = None,
            sde_noise                                              = None,
            sde_noise_steps               : int                    = 1, 
            extra_options                 : str                    = "", 
            automation                                             = None, 

            epsilon_scales                : Optional[Tensor]       = None, 
            regional_conditioning_weights : Optional[Tensor]       = None,
            frame_weights_mgr                                      = None, 


            rescale_floor                 : bool                   = True, 
            
            rk_swaps                      : list                   = [],
            
            sde_mask                      : Optional[Tensor]       = None,

            #start_at_step                 : int                    = 0,
            #stop_at_step                  : int                    = MAX_STEPS,
                        
            **kwargs
            ): 
        
        options_mgr = OptionsManager(options_group=options_group, **kwargs)
        extra_options    += "\n" + options_mgr.get('extra_options', "")

        #if model is None:
        #    model = latent_image['model']
        
        
        # defaults for ClownSampler
        eta_substep = eta
        
        # defaults for SharkSampler
        noise_type_init = "gaussian"
        noise_stdev     = 1.0
        denoise_alt     = 1.0
        channelwise_cfg = False
        
        if denoise < 0:
            denoise_alt = -denoise
            denoise = 1.0
        
        is_chained = False

        guider_input = options_mgr.get('guider', None)
        guider_from_latent = latent_image.get('guider') if latent_image is not None else None

        if guider_input is not None:
            # Explicit guider input takes full precedence - all settings come from guider
            guider = copy.copy(guider_input)
            guider.original_conds = dict(guider.original_conds)
            model = guider.model_patcher
            if has_custom_cfg_handling(guider):
                RESplain(f"Clown: Guider has custom CFG handling ({guider.__class__.__name__}) - using guider's internal CFG settings")
            elif hasattr(guider, 'cfg') and guider.cfg is not None:
                cfg = guider.cfg
                RESplain(f"Clown: Using CFG from explicit guider input: {cfg}")
            extracted_positive = extract_cond_from_guider(guider, 'positive')
            if extracted_positive is not None:
                positive = extracted_positive
            extracted_negative = extract_cond_from_guider(guider, 'negative')
            if extracted_negative is not None:
                negative = extracted_negative
        elif guider_from_latent is not None:
            # Chained guider - node inputs can override
            guider = copy.copy(guider_from_latent)
            guider.original_conds = dict(guider.original_conds)
            if model is not None:
                guider.model_patcher = model
                guider.model_options = model.model_options
            else:
                model = guider.model_patcher
            if has_custom_cfg_handling(guider):
                RESplain(f"ClownWarning: Guider has custom CFG handling ({guider.__class__.__name__}) - node CFG input will be ignored")
            if positive is None:
                positive = extract_cond_from_guider(guider, 'positive')
            if negative is None:
                negative = extract_cond_from_guider(guider, 'negative')
            is_chained = True
        else:
            guider = None

        if model.model.model_config.unet_config.get('stable_cascade_stage') == 'b':
            noise_type_sde         = "pyramid-cascade_B"
            noise_type_sde_substep = "pyramid-cascade_B"
        
        
        #if options is not None:
        #options_mgr = OptionsManager(options_inputs)
        noise_seed_sde         = options_mgr.get('noise_seed_sde'        , noise_seed_sde)
        
        
        noise_type_sde         = options_mgr.get('noise_type_sde'        , noise_type_sde)
        noise_type_sde_substep = options_mgr.get('noise_type_sde_substep', noise_type_sde_substep)
        
        options_mgr.update('noise_type_sde',         noise_type_sde)
        options_mgr.update('noise_type_sde_substep', noise_type_sde_substep)
        
        noise_mode_sde         = options_mgr.get('noise_mode_sde'        , noise_mode_sde)
        noise_mode_sde_substep = options_mgr.get('noise_mode_sde_substep', noise_mode_sde_substep)
        
        overshoot_mode         = options_mgr.get('overshoot_mode'        , overshoot_mode)
        overshoot_mode_substep = options_mgr.get('overshoot_mode_substep', overshoot_mode_substep)

        eta                    = options_mgr.get('eta'                   , eta)
        eta_substep            = options_mgr.get('eta_substep'           , eta_substep)
        
        options_mgr.update('eta',         eta)
        options_mgr.update('eta_substep', eta_substep)

        overshoot              = options_mgr.get('overshoot'             , overshoot)
        overshoot_substep      = options_mgr.get('overshoot_substep'     , overshoot_substep)
        
        
    
        noise_scaling_weight   = options_mgr.get('noise_scaling_weight' , noise_scaling_weight)
        noise_scaling_type     = options_mgr.get('noise_scaling_type'    , noise_scaling_type)
        noise_scaling_mode     = options_mgr.get('noise_scaling_mode'    , noise_scaling_mode)
        noise_scaling_eta      = options_mgr.get('noise_scaling_eta'     , noise_scaling_eta)
        noise_scaling_cycles   = options_mgr.get('noise_scaling_cycles'  , noise_scaling_cycles)
        
        noise_scaling_weights  = options_mgr.get('noise_scaling_weights' , noise_scaling_weights)
        noise_scaling_etas     = options_mgr.get('noise_scaling_etas'    , noise_scaling_etas)
        
        noise_boost_step       = options_mgr.get('noise_boost_step'      , noise_boost_step)
        noise_boost_substep    = options_mgr.get('noise_boost_substep'   , noise_boost_substep)
        noise_boost_normalize  = options_mgr.get('noise_boost_normalize' , noise_boost_normalize)
        noise_anchor           = options_mgr.get('noise_anchor'          , noise_anchor)
        
        s_noise                = options_mgr.get('s_noise'               , s_noise)
        s_noise_substep        = options_mgr.get('s_noise_substep'       , s_noise_substep)
        d_noise                = options_mgr.get('d_noise'               , d_noise)
        d_noise_start_step     = options_mgr.get('d_noise_start_step'          , d_noise_start_step)
        d_noise_inv            = options_mgr.get('d_noise_inv'           , d_noise_inv)
        d_noise_inv_start_step = options_mgr.get('d_noise_inv_start_step', d_noise_inv_start_step)
        
        
        momentum               = options_mgr.get('momentum'              , momentum)

        
        implicit_type          = options_mgr.get('implicit_type'         , implicit_type)
        implicit_type_substeps = options_mgr.get('implicit_type_substeps', implicit_type_substeps)
        implicit_steps         = options_mgr.get('implicit_steps'        , implicit_steps)
        implicit_substeps      = options_mgr.get('implicit_substeps'     , implicit_substeps)
        
        alpha_sde              = options_mgr.get('alpha_sde'             , alpha_sde)
        k_sde                  = options_mgr.get('k_sde'                 , k_sde)
        c1                     = options_mgr.get('c1'                    , c1)
        c2                     = options_mgr.get('c2'                    , c2)
        c3                     = options_mgr.get('c3'                    , c3)

        frame_weights_mgr      = options_mgr.get('frame_weights_mgr'     , frame_weights_mgr)
        
        sde_noise              = options_mgr.get('sde_noise'             , sde_noise)
        sde_noise_steps        = options_mgr.get('sde_noise_steps'       , sde_noise_steps)
        
        extra_options          = options_mgr.get('extra_options'         , extra_options)
        
        automation             = options_mgr.get('automation'            , automation)
        
        # SharkSampler Options
        noise_type_init        = options_mgr.get('noise_type_init'       , noise_type_init)
        noise_stdev            = options_mgr.get('noise_stdev'           , noise_stdev)
        sampler_mode           = options_mgr.get('sampler_mode'          , sampler_mode)
        denoise_alt            = options_mgr.get('denoise_alt'           , denoise_alt)
        
        channelwise_cfg        = options_mgr.get('channelwise_cfg'       , channelwise_cfg)
        
        options_mgr.update('noise_type_init', noise_type_init)
        options_mgr.update('noise_stdev',     noise_stdev)
        options_mgr.update('denoise_alt',     denoise_alt)
        #options_mgr.update('channelwise_cfg', channelwise_cfg)
        
        sigmas                 = options_mgr.get('sigmas'                , sigmas)
        
        rk_swaps = options_mgr.get('rk_swaps', rk_swaps)
        if not rk_swaps:
            _swap_type = options_mgr.get('rk_swap_type', '')
            if _swap_type:
                rk_swaps = [{'type': _swap_type, 'step': options_mgr.get('rk_swap_step', MAX_STEPS),
                             'threshold': options_mgr.get('rk_swap_threshold', 0.0), 'print': options_mgr.get('rk_swap_print', False)}]
        
        sde_mask               = options_mgr.get('sde_mask'              , sde_mask)

        
        #start_at_step          = options_mgr.get('start_at_step'         , start_at_step)
        #stop_at_ste            = options_mgr.get('stop_at_step'          , stop_at_step)
                
        if channelwise_cfg: # != 1.0:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw




        _advanced_out = ClownSamplerAdvanced_Beta.execute(
            noise_type_sde                = noise_type_sde,
            noise_type_sde_substep        = noise_type_sde_substep,
            noise_mode_sde                = noise_mode_sde,
            noise_mode_sde_substep        = noise_mode_sde_substep,
            
            eta                           = eta,
            eta_substep                   = eta_substep,
            

            
            overshoot                     = overshoot,
            overshoot_substep             = overshoot_substep,
            
            overshoot_mode                = overshoot_mode,
            overshoot_mode_substep        = overshoot_mode_substep,
            
            
            momentum                      = momentum,

            alpha_sde                     = alpha_sde,
            k_sde                         = k_sde,
            cfgpp                         = cfgpp,
            c1                            = c1,
            c2                            = c2,
            c3                            = c3,
            sampler_name                  = sampler_name,
            implicit_sampler_name         = implicit_sampler_name,

            implicit_type                 = implicit_type,
            implicit_type_substeps        = implicit_type_substeps,
            implicit_steps                = implicit_steps,
            implicit_substeps             = implicit_substeps,

            rescale_floor                 = rescale_floor,
            sigmas_override               = sigmas_override,
            
            noise_seed_sde                = noise_seed_sde,
            
            guides                        = guides,
            options_group                 = {"options0": options_mgr.as_dict()},

            extra_options                 = extra_options,
            automation                    = automation,



            noise_scaling_weight          = noise_scaling_weight,
            noise_scaling_type            = noise_scaling_type,
            noise_scaling_mode            = noise_scaling_mode,
            noise_scaling_eta             = noise_scaling_eta,
            noise_scaling_cycles          = noise_scaling_cycles,
            
            noise_scaling_weights         = noise_scaling_weights,
            noise_scaling_etas            = noise_scaling_etas,

            noise_boost_step              = noise_boost_step,
            noise_boost_substep           = noise_boost_substep,
            noise_boost_normalize         = noise_boost_normalize,
            noise_anchor                  = noise_anchor,
            
            s_noise                       = s_noise,
            s_noise_substep               = s_noise_substep,
            d_noise                       = d_noise,
            d_noise_start_step            = d_noise_start_step,
            d_noise_inv                   = d_noise_inv,
            d_noise_inv_start_step        = d_noise_inv_start_step,
            
            
            epsilon_scales                = epsilon_scales,
            regional_conditioning_weights = regional_conditioning_weights,
            frame_weights_mgr             = frame_weights_mgr,
            
            sde_noise                     = sde_noise,
            sde_noise_steps               = sde_noise_steps,
            
            rk_swaps                      = rk_swaps,
            
            steps_to_run                  = steps_to_run,
            
            sde_mask                      = sde_mask,

            bongmath                      = bongmath,
            )
        sampler = _advanced_out[0]


        output, denoised, sde_noise = SharkSampler().main(
            model           = model,
            cfg             = cfg,
            scheduler       = scheduler,
            steps           = steps, 
            steps_to_run    = steps_to_run,
            denoise         = denoise,
            latent_image    = latent_image, 
            positive        = positive,
            negative        = negative, 
            sampler         = sampler, 
            cfgpp           = cfgpp, 
            noise_seed      = seed, 
            options         = options_mgr.as_dict(), 
            sde_noise       = sde_noise, 
            sde_noise_steps = sde_noise_steps, 
            noise_type_init = noise_type_init,
            noise_stdev     = noise_stdev,
            sampler_mode    = sampler_mode,
            denoise_alt     = denoise_alt,
            sigmas          = sigmas,

            extra_options   = extra_options)
        
        return io.NodeOutput(output, denoised, options_mgr.as_dict())











class ClownsharkChainsampler_Beta(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ClownsharkChainsampler_Beta",
            display_name="ClownsharkChainsampler",
            category="RES4LYF/samplers",
            inputs=[
                io.Float.Input("eta", default=0.5, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Calculated noise amount to be added, then removed, after each step."),
                io.Combo.Input("sampler_name", options=get_sampler_name_list(), default=get_default_sampler_name()),
                io.Int.Input("steps_to_run", default=-1, min=-1, max=MAX_STEPS),
                io.Float.Input("cfg", default=5.5, min=-10000.0, max=10000.0, step=0.01, round=False,
                               tooltip="Negative values use channelwise CFG."),
                io.Combo.Input("sampler_mode", options=["unsample", "resample"], default="resample"),
                io.Boolean.Input("bongmath", default=True),
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Sigmas.Input("sigmas", optional=True),
                io.Latent.Input("latent_image", optional=True),
                io.Custom("GUIDES").Input("guides", optional=True),
                io.Autogrow.Input(
                    "options_group",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Custom("OPTIONS").Input("options", optional=True),
                        prefix="options",
                        min=0,
                        max=20,
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised"),
                io.Custom("OPTIONS").Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls,
                eta=0.5,
                sampler_name="res_2m",
                steps_to_run=-1,
                cfg=5.5,
                sampler_mode="resample",
                bongmath=True,
                model=None,
                positive=None,
                negative=None,
                sigmas=None,
                latent_image=None,
                guides=None,
                options_group=None,
                **kwargs):

        steps = latent_image['state_info']['sigmas'].shape[-1] - 3
        sigmas = latent_image['state_info']['sigmas'] if sigmas is None else sigmas
        if len(sigmas) > 2 and sigmas[1] < sigmas[2] and latent_image['state_info']['sampler_mode'] == "unsample" and sampler_mode == "resample":
            sigmas = torch.flip(sigmas, dims=[0])

        return ClownsharKSampler_Beta.execute(
            eta=eta,
            sampler_name=sampler_name,
            sampler_mode=sampler_mode,
            sigmas=sigmas,
            steps_to_run=steps_to_run,
            steps=steps,
            cfg=cfg,
            bongmath=bongmath,
            seed=-1,
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            guides=guides,
            options_group=options_group,
            **{k: v for k, v in kwargs.items() if isinstance(k, str) and k.startswith('options')},
        )






class ClownSampler_Beta(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ClownSampler_Beta",
            display_name="ClownSampler",
            category="RES4LYF/samplers",
            inputs=[
                io.Float.Input("eta", default=0.5, min=-100.0, max=100.0, step=0.01, round=False,
                               tooltip="Calculated noise amount to be added, then removed, after each step."),
                io.Combo.Input("sampler_name", options=get_sampler_name_list(), default=get_default_sampler_name()),
                io.Int.Input("seed", default=-1, min=-1, max=0xffffffffffffffff),
                io.Boolean.Input("bongmath", default=True),
                io.Custom("GUIDES").Input("guides", optional=True),
                io.Autogrow.Input(
                    "options_group",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        io.Custom("OPTIONS").Input("options", optional=True),
                        prefix="options",
                        min=0,
                        max=20,
                    ),
                ),
            ],
            outputs=[
                io.Sampler.Output(display_name="sampler"),
            ],
        )

    @classmethod
    def execute(cls,
            model                                                  = None,
            denoise                       : float                  = 1.0, 
            scheduler                     : str                    = "beta57", 
            cfg                           : float                  = 1.0, 
            seed                          : int                    = -1, 
            positive                                               = None, 
            negative                                               = None, 
            latent_image                  : Optional[dict[Tensor]] = None, 
            steps                         : int                    = 30,
            steps_to_run                  : int                    = -1,
            bongmath                      : bool                   = True,
            sampler_mode                  : str                    = "standard",
            
            noise_type_sde                : str                    = "gaussian", 
            noise_type_sde_substep        : str                    = "gaussian", 
            noise_mode_sde                : str                    = "hard",
            noise_mode_sde_substep        : str                    = "hard",

            
            overshoot_mode                : str                    = "hard", 
            overshoot_mode_substep        : str                    = "hard",
            overshoot                     : float                  = 0.0, 
            overshoot_substep             : float                  = 0.0,
            
            eta                           : float                  = 0.5, 
            eta_substep                   : float                  = 0.5,
            
            noise_scaling_weight          : float                  = 0.0,
            noise_boost_step              : float                  = 0.0, 
            noise_boost_substep           : float                  = 0.0, 
            noise_anchor                  : float                  = 1.0,
            
            s_noise                       : float                  = 1.0, 
            s_noise_substep               : float                  = 1.0, 
            d_noise                       : float                  = 1.0, 
            d_noise_start_step            : int                    = 0,
            d_noise_inv                   : float                  = 1.0,
            d_noise_inv_start_step        : int                    = 0,

            
            alpha_sde                     : float                  = -1.0, 
            k_sde                         : float                  = 1.0,
            cfgpp                         : float                  = 0.0,
            c1                            : float                  = 0.0, 
            c2                            : float                  = 0.5, 
            c3                            : float                  = 1.0,
            noise_seed_sde                : int                    = -1,
            sampler_name                  : str                    = "res_2m", 
            implicit_sampler_name         : str                    = "use_explicit",
            
            implicit_type                 : str                    = "bongmath",
            implicit_type_substeps        : str                    = "bongmath",
            implicit_steps                : int                    = 0,
            implicit_substeps             : int                    = 0, 

            sigmas                        : Optional[Tensor]       = None,
            sigmas_override               : Optional[Tensor]       = None,
            guides                                                 = None,
            options_group                                          = None,
            sde_noise                                              = None,
            sde_noise_steps               : int                    = 1, 
            extra_options                 : str                    = "", 
            automation                                             = None, 

            epsilon_scales                : Optional[Tensor]       = None, 
            regional_conditioning_weights : Optional[Tensor]       = None,
            frame_weights_mgr                                      = None, 


            rescale_floor                 : bool                   = True, 
            
            rk_swaps                      : list                   = [],
            
            sde_mask                      : Optional[Tensor]       = None,

            
            #start_at_step                 : int                    = 0,
            #stop_at_step                  : int                    = MAX_STEPS,

            **kwargs
            ):

        options_mgr = OptionsManager(options_group=options_group, **kwargs)
        extra_options    += "\n" + options_mgr.get('extra_options', "")


        # defaults for ClownSampler
        eta_substep = eta
        
        # defaults for SharkSampler
        noise_type_init = "gaussian"
        noise_stdev     = 1.0
        denoise_alt     = 1.0
        channelwise_cfg = False #1.0
        
        
        #if options is not None:
        #options_mgr = OptionsManager(options_inputs)
        noise_type_sde         = options_mgr.get('noise_type_sde'        , noise_type_sde)
        noise_type_sde_substep = options_mgr.get('noise_type_sde_substep', noise_type_sde_substep)
        
        noise_mode_sde         = options_mgr.get('noise_mode_sde'        , noise_mode_sde)
        noise_mode_sde_substep = options_mgr.get('noise_mode_sde_substep', noise_mode_sde_substep)
        
        overshoot_mode         = options_mgr.get('overshoot_mode'        , overshoot_mode)
        overshoot_mode_substep = options_mgr.get('overshoot_mode_substep', overshoot_mode_substep)

        eta                    = options_mgr.get('eta'                   , eta)
        eta_substep            = options_mgr.get('eta_substep'           , eta_substep)

        overshoot              = options_mgr.get('overshoot'             , overshoot)
        overshoot_substep      = options_mgr.get('overshoot_substep'     , overshoot_substep)
        
        noise_scaling_weight   = options_mgr.get('noise_scaling_weight' , noise_scaling_weight)
        noise_boost_step       = options_mgr.get('noise_boost_step'      , noise_boost_step)
        noise_boost_substep    = options_mgr.get('noise_boost_substep'   , noise_boost_substep)
        
        noise_anchor           = options_mgr.get('noise_anchor'          , noise_anchor)

        s_noise                = options_mgr.get('s_noise'               , s_noise)
        s_noise_substep        = options_mgr.get('s_noise_substep'       , s_noise_substep)

        d_noise                = options_mgr.get('d_noise'               , d_noise)
        d_noise_start_step     = options_mgr.get('d_noise_start_step'    , d_noise_start_step)
        d_noise_inv            = options_mgr.get('d_noise_inv'           , d_noise_inv)
        d_noise_inv_start_step = options_mgr.get('d_noise_inv_start_step', d_noise_inv_start_step)
        
        implicit_type          = options_mgr.get('implicit_type'         , implicit_type)
        implicit_type_substeps = options_mgr.get('implicit_type_substeps', implicit_type_substeps)
        implicit_steps         = options_mgr.get('implicit_steps'        , implicit_steps)
        implicit_substeps      = options_mgr.get('implicit_substeps'     , implicit_substeps)
        
        alpha_sde              = options_mgr.get('alpha_sde'             , alpha_sde)
        k_sde                  = options_mgr.get('k_sde'                 , k_sde)
        c1                     = options_mgr.get('c1'                    , c1)
        c2                     = options_mgr.get('c2'                    , c2)
        c3                     = options_mgr.get('c3'                    , c3)

        frame_weights_mgr      = options_mgr.get('frame_weights_mgr'     , frame_weights_mgr)
        
        sde_noise              = options_mgr.get('sde_noise'             , sde_noise)
        sde_noise_steps        = options_mgr.get('sde_noise_steps'       , sde_noise_steps)
        
        extra_options          = options_mgr.get('extra_options'         , extra_options)
        
        automation             = options_mgr.get('automation'            , automation)
        
        # SharkSampler Options
        noise_type_init        = options_mgr.get('noise_type_init'       , noise_type_init)
        noise_stdev            = options_mgr.get('noise_stdev'           , noise_stdev)
        sampler_mode           = options_mgr.get('sampler_mode'          , sampler_mode)
        denoise_alt            = options_mgr.get('denoise_alt'           , denoise_alt)
        
        channelwise_cfg        = options_mgr.get('channelwise_cfg'       , channelwise_cfg)
        
        sigmas                 = options_mgr.get('sigmas'                , sigmas)
        
        rk_swaps = options_mgr.get('rk_swaps', rk_swaps)
        if not rk_swaps:
            _swap_type = options_mgr.get('rk_swap_type', '')
            if _swap_type:
                rk_swaps = [{'type': _swap_type, 'step': options_mgr.get('rk_swap_step', MAX_STEPS),
                             'threshold': options_mgr.get('rk_swap_threshold', 0.0), 'print': options_mgr.get('rk_swap_print', False)}]
        
        sde_mask               = options_mgr.get('sde_mask'              , sde_mask)

        
        #start_at_step          = options_mgr.get('start_at_step'         , start_at_step)
        #stop_at_ste            = options_mgr.get('stop_at_step'          , stop_at_step)
                
        if channelwise_cfg: # != 1.0:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw

        noise_seed_sde = seed


        _advanced_out = ClownSamplerAdvanced_Beta.execute(
            noise_type_sde                = noise_type_sde,
            noise_type_sde_substep        = noise_type_sde_substep,
            noise_mode_sde                = noise_mode_sde,
            noise_mode_sde_substep        = noise_mode_sde_substep,

            eta                           = eta,
            eta_substep                   = eta_substep,

            s_noise                       = s_noise,
            s_noise_substep               = s_noise_substep,

            overshoot                     = overshoot,
            overshoot_substep             = overshoot_substep,

            overshoot_mode                = overshoot_mode,
            overshoot_mode_substep        = overshoot_mode_substep,

            d_noise                       = d_noise,
            d_noise_start_step            = d_noise_start_step,
            d_noise_inv                   = d_noise_inv,
            d_noise_inv_start_step        = d_noise_inv_start_step,

            alpha_sde                     = alpha_sde,
            k_sde                         = k_sde,
            cfgpp                         = cfgpp,
            c1                            = c1,
            c2                            = c2,
            c3                            = c3,
            sampler_name                  = sampler_name,
            implicit_sampler_name         = implicit_sampler_name,

            implicit_type                 = implicit_type,
            implicit_type_substeps        = implicit_type_substeps,
            implicit_steps                = implicit_steps,
            implicit_substeps             = implicit_substeps,

            rescale_floor                 = rescale_floor,
            sigmas_override               = sigmas_override,

            noise_seed_sde                = noise_seed_sde,

            guides                        = guides,
            options_group                 = {"options0": options_mgr.as_dict()},

            extra_options                 = extra_options,
            automation                    = automation,

            noise_scaling_weight          = noise_scaling_weight,
            noise_boost_step              = noise_boost_step,
            noise_boost_substep           = noise_boost_substep,

            epsilon_scales                = epsilon_scales,
            regional_conditioning_weights = regional_conditioning_weights,
            frame_weights_mgr             = frame_weights_mgr,

            sde_noise                     = sde_noise,
            sde_noise_steps               = sde_noise_steps,

            rk_swaps                      = rk_swaps,

            steps_to_run                  = steps_to_run,

            sde_mask                      = sde_mask,

            bongmath                      = bongmath,
        )

        return io.NodeOutput(_advanced_out[0])
    








class BongSampler:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required":
                    {
                    "model":        ("MODEL",),
                    "seed":         ("INT",                        {"default": 0,   "min": -1,     "max": 0xffffffffffffffff}),
                    "steps":        ("INT",                        {"default": 30,  "min":  1,     "max": MAX_STEPS}),
                    "cfg":          ("FLOAT",                      {"default": 5.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, }),
                    "sampler_name": (["res_2m", "res_3m", "res_2s", "res_3s","res_2m_sde", "res_3m_sde", "res_2s_sde", "res_3s_sde"], {"default": "res_2s_sde"}), 
                    "scheduler":    (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "denoise":      ("FLOAT",                      {"default": 1.0, "min": -10000, "max": MAX_STEPS, "step":0.01}),
                    },
                "optional": 
                    {
                    "positive":     ("CONDITIONING",),
                    "negative":     ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    }
                }
        
        return inputs

    RETURN_TYPES = ("LATENT", )
    
    RETURN_NAMES = ("output", )
    
    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"
    
    def main(self, 
            model                                                  = None,
            denoise                       : float                  = 1.0, 
            scheduler                     : str                    = "beta57", 
            cfg                           : float                  = 1.0, 
            seed                          : int                    = 42, 
            positive                                               = None, 
            negative                                               = None, 
            latent_image                  : Optional[dict[Tensor]] = None, 
            steps                         : int                    = 30,
            steps_to_run                  : int                    = -1,
            bongmath                      : bool                   = True,
            sampler_mode                  : str                    = "standard",
            
            noise_type_sde                : str                    = "brownian", 
            noise_type_sde_substep        : str                    = "brownian", 
            noise_mode_sde                : str                    = "hard",
            noise_mode_sde_substep        : str                    = "hard",

            
            overshoot_mode                : str                    = "hard", 
            overshoot_mode_substep        : str                    = "hard",
            overshoot                     : float                  = 0.0, 
            overshoot_substep             : float                  = 0.0,
            
            eta                           : float                  = 0.5, 
            eta_substep                   : float                  = 0.5,
            d_noise                       : float                  = 1.0, 
            s_noise                       : float                  = 1.0, 
            s_noise_substep               : float                  = 1.0, 
            
            alpha_sde                     : float                  = -1.0, 
            k_sde                         : float                  = 1.0,
            cfgpp                         : float                  = 0.0,
            c1                            : float                  = 0.0, 
            c2                            : float                  = 0.5, 
            c3                            : float                  = 1.0,
            noise_seed_sde                : int                    = -1,
            sampler_name                  : str                    = "res_2m", 
            implicit_sampler_name         : str                    = "use_explicit",
            
            implicit_type                 : str                    = "bongmath",
            implicit_type_substeps        : str                    = "bongmath",
            implicit_steps                : int                    = 0,
            implicit_substeps             : int                    = 0, 

            sigmas                        : Optional[Tensor]       = None,
            sigmas_override               : Optional[Tensor]       = None,
            guides                                                 = None,
            options_group                                          = None,
            sde_noise                                              = None,
            sde_noise_steps               : int                    = 1, 
            extra_options                 : str                    = "", 
            automation                                             = None, 

            epsilon_scales                : Optional[Tensor]       = None, 
            regional_conditioning_weights : Optional[Tensor]       = None,
            frame_weights_mgr                                      = None, 
            noise_scaling_weight         : float                  = 0.0, 
            noise_boost_step              : float                  = 0.0, 
            noise_boost_substep           : float                  = 0.0, 
            noise_anchor                  : float                  = 1.0,

            rescale_floor                 : bool                   = True, 
            
            rk_swaps                      : list                   = [],
            
            #start_at_step                 : int                    = 0,
            #stop_at_step                  : int                    = MAX_STEPS,
                        
            **kwargs
            ): 
        
        options_mgr = OptionsManager(options, **kwargs)
        extra_options    += "\n" + options_mgr.get('extra_options', "")
        
        if model.model.model_config.unet_config.get('stable_cascade_stage') == 'b':
            noise_type_sde         = "pyramid-cascade_B"
            noise_type_sde_substep = "pyramid-cascade_B"
        
        if sampler_name.endswith("_sde"):
            sampler_name = sampler_name[:-4]
            eta = 0.5
        else:
            eta = 0.0
        
        # defaults for ClownSampler
        eta_substep = eta
        
        # defaults for SharkSampler
        noise_type_init = "gaussian"
        noise_stdev     = 1.0
        denoise_alt     = 1.0
        channelwise_cfg = False #1.0
        
        
        #if options is not None:
        #options_mgr = OptionsManager(options_inputs)
        noise_type_sde         = options_mgr.get('noise_type_sde'        , noise_type_sde)
        noise_type_sde_substep = options_mgr.get('noise_type_sde_substep', noise_type_sde_substep)
        
        noise_mode_sde         = options_mgr.get('noise_mode_sde'        , noise_mode_sde)
        noise_mode_sde_substep = options_mgr.get('noise_mode_sde_substep', noise_mode_sde_substep)
        
        overshoot_mode         = options_mgr.get('overshoot_mode'        , overshoot_mode)
        overshoot_mode_substep = options_mgr.get('overshoot_mode_substep', overshoot_mode_substep)

        eta                    = options_mgr.get('eta'                   , eta)
        eta_substep            = options_mgr.get('eta_substep'           , eta_substep)

        overshoot              = options_mgr.get('overshoot'             , overshoot)
        overshoot_substep      = options_mgr.get('overshoot_substep'     , overshoot_substep)
        
        noise_scaling_weight  = options_mgr.get('noise_scaling_weight' , noise_scaling_weight)

        noise_boost_step       = options_mgr.get('noise_boost_step'      , noise_boost_step)
        noise_boost_substep    = options_mgr.get('noise_boost_substep'   , noise_boost_substep)
        
        noise_anchor           = options_mgr.get('noise_anchor'          , noise_anchor)

        s_noise                = options_mgr.get('s_noise'               , s_noise)
        s_noise_substep        = options_mgr.get('s_noise_substep'       , s_noise_substep)

        d_noise                = options_mgr.get('d_noise'               , d_noise)
        
        implicit_type          = options_mgr.get('implicit_type'         , implicit_type)
        implicit_type_substeps = options_mgr.get('implicit_type_substeps', implicit_type_substeps)
        implicit_steps         = options_mgr.get('implicit_steps'        , implicit_steps)
        implicit_substeps      = options_mgr.get('implicit_substeps'     , implicit_substeps)
        
        alpha_sde              = options_mgr.get('alpha_sde'             , alpha_sde)
        k_sde                  = options_mgr.get('k_sde'                 , k_sde)
        c1                     = options_mgr.get('c1'                    , c1)
        c2                     = options_mgr.get('c2'                    , c2)
        c3                     = options_mgr.get('c3'                    , c3)

        frame_weights_mgr      = options_mgr.get('frame_weights_mgr'     , frame_weights_mgr)
        
        sde_noise              = options_mgr.get('sde_noise'             , sde_noise)
        sde_noise_steps        = options_mgr.get('sde_noise_steps'       , sde_noise_steps)
        
        extra_options          = options_mgr.get('extra_options'         , extra_options)
        
        automation             = options_mgr.get('automation'            , automation)
        
        # SharkSampler Options
        noise_type_init        = options_mgr.get('noise_type_init'       , noise_type_init)
        noise_stdev            = options_mgr.get('noise_stdev'           , noise_stdev)
        sampler_mode           = options_mgr.get('sampler_mode'          , sampler_mode)
        denoise_alt            = options_mgr.get('denoise_alt'           , denoise_alt)
        
        channelwise_cfg        = options_mgr.get('channelwise_cfg'       , channelwise_cfg)
        
        sigmas                 = options_mgr.get('sigmas'                , sigmas)
        
        rk_swaps = options_mgr.get('rk_swaps', rk_swaps)
        if not rk_swaps:
            _swap_type = options_mgr.get('rk_swap_type', '')
            if _swap_type:
                rk_swaps = [{'type': _swap_type, 'step': options_mgr.get('rk_swap_step', MAX_STEPS),
                             'threshold': options_mgr.get('rk_swap_threshold', 0.0), 'print': options_mgr.get('rk_swap_print', False)}]
        
        #start_at_step          = options_mgr.get('start_at_step'         , start_at_step)
        #stop_at_ste            = options_mgr.get('stop_at_step'          , stop_at_step)
                
        if channelwise_cfg: # != 1.0:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw




        _advanced_out = ClownSamplerAdvanced_Beta.execute(
            noise_type_sde                = noise_type_sde,
            noise_type_sde_substep        = noise_type_sde_substep,
            noise_mode_sde                = noise_mode_sde,
            noise_mode_sde_substep        = noise_mode_sde_substep,
            
            eta                           = eta,
            eta_substep                   = eta_substep,
            
            s_noise                       = s_noise,
            s_noise_substep               = s_noise_substep,
            
            overshoot                     = overshoot,
            overshoot_substep             = overshoot_substep,
            
            overshoot_mode                = overshoot_mode,
            overshoot_mode_substep        = overshoot_mode_substep,
            
            d_noise                       = d_noise,
            #d_noise_start_step            = d_noise_start_step,
            #d_noise_inv                   = d_noise_inv,
            #d_noise_inv_start_step        = d_noise_inv_start_step,

            alpha_sde                     = alpha_sde,
            k_sde                         = k_sde,
            cfgpp                         = cfgpp,
            c1                            = c1,
            c2                            = c2,
            c3                            = c3,
            sampler_name                  = sampler_name,
            implicit_sampler_name         = implicit_sampler_name,

            implicit_type                 = implicit_type,
            implicit_type_substeps        = implicit_type_substeps,
            implicit_steps                = implicit_steps,
            implicit_substeps             = implicit_substeps,

            rescale_floor                 = rescale_floor,
            sigmas_override               = sigmas_override,
            
            noise_seed_sde                = noise_seed_sde,
            
            guides                        = guides,
            options_group                 = {"options0": options_mgr.as_dict()},

            extra_options                 = extra_options,
            automation                    = automation,

            noise_scaling_weight         = noise_scaling_weight,
            noise_boost_step              = noise_boost_step,
            noise_boost_substep           = noise_boost_substep,
            
            epsilon_scales                = epsilon_scales,
            regional_conditioning_weights = regional_conditioning_weights,
            frame_weights_mgr             = frame_weights_mgr,
            
            sde_noise                     = sde_noise,
            sde_noise_steps               = sde_noise_steps,
            
            rk_swaps                      = rk_swaps,
            
            steps_to_run                  = steps_to_run,

            bongmath                      = bongmath,
            )
        sampler = _advanced_out[0]


        output, denoised, sde_noise = SharkSampler().main(
            model           = model,
            cfg             = cfg,
            scheduler       = scheduler,
            steps           = steps, 
            steps_to_run    = steps_to_run,
            denoise         = denoise,
            latent_image    = latent_image, 
            positive        = positive,
            negative        = negative, 
            sampler         = sampler, 
            cfgpp           = cfgpp, 
            noise_seed      = seed, 
            options         = options_mgr.as_dict(), 
            sde_noise       = sde_noise, 
            sde_noise_steps = sde_noise_steps, 
            noise_type_init = noise_type_init,
            noise_stdev     = noise_stdev,
            sampler_mode    = sampler_mode,
            denoise_alt     = denoise_alt,
            sigmas          = sigmas,

            extra_options   = extra_options)
        
        return (output, )