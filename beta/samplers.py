import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Callable, Tuple, Dict, Any, Union
import copy
import gc

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd
import comfy.supported_models
from comfy.samplers import CFGGuider, sampling_function

import latent_preview

from ..helper               import initialize_or_scale, get_res4lyf_scheduler_list, OptionsManager, ExtraOptions
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
        positive = self.conds.get(f'{latent_type}_positive', self.conds.get('xt_positive'))
        negative = self.conds.get(f'{latent_type}_negative', self.conds.get('xt_negative'))
        positive = self.conds.get('xt_positive') if positive is None else positive
        negative = self.conds.get('xt_negative') if negative is None else negative
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
                if 'positive' in latent_image and positive is None:
                    positive = copy_cond(latent_image['positive'])
                    if positive is not None and 'control' in positive[0][1]:
                        for i in range(len(positive)):
                            positive[i][1]['control']      = latent_image['positive'][i][1]['control']
                            if hasattr(latent_image['positive'][i][1]['control'], 'base'):
                                positive[i][1]['control'].base = latent_image['positive'][i][1]['control'].base
                    is_chained = True
                if 'negative' in latent_image and negative is None:
                    negative = copy_cond(latent_image['negative'])
                    if negative is not None and 'control' in negative[0][1]:
                        for i in range(len(negative)):
                            negative[i][1]['control']      = latent_image['negative'][i][1]['control']
                            if hasattr(latent_image['negative'][i][1]['control'], 'base'):
                                negative[i][1]['control'].base = latent_image['negative'][i][1]['control'].base
                    is_chained = True
                if 'sampler' in latent_image and sampler is None:
                    sampler = copy_cond(latent_image['sampler'])  #.clone()
                    is_chained = True
            
            if 'steps_to_run' in sampler.extra_options:
                sampler.extra_options['steps_to_run'] = steps_to_run

            guider_input = options_mgr.get('guider', None)
            if guider_input is not None and is_chained is False:
                guider = guider_input
                work_model = guider.model_patcher
                RESplain("Shark: Using model from ClownOptions_GuiderInput: ", guider.model_patcher.model.diffusion_model.__class__.__name__)
                RESplain("SharkWarning: \"flow\" guide mode does not work with ClownOptions_GuiderInput")
                if hasattr(guider, 'cfg') and guider.cfg is not None:
                    cfg = guider.cfg
                    RESplain("Shark: Using cfg from ClownOptions_GuiderInput: ", cfg)
                if hasattr(guider, 'original_conds') and guider.original_conds is not None:
                    if 'positive' in guider.original_conds:
                        first_ = guider.original_conds['positive'][0]['cross_attn']
                        second_ = {k: v for k, v in guider.original_conds['positive'][0].items() if k != 'cross_attn'}
                        positive = [[first_, second_],]
                        RESplain("Shark: Using positive cond from ClownOptions_GuiderInput")
                    if 'negative' in guider.original_conds:
                        first_ = guider.original_conds['negative'][0]['cross_attn']
                        second_ = {k: v for k, v in guider.original_conds['negative'][0].items() if k != 'cross_attn'}
                        negative = [[first_, second_],]
                        RESplain("Shark: Using negative cond from ClownOptions_GuiderInput")
            else:
                guider = None
                work_model   = model#.clone()
            
            if latent_image is not None:
                latent_image['samples'] = comfy.sample.fix_empty_latent_channels(work_model, latent_image['samples'])
                
            if positive is None or negative is None:
                from ..conditioning import EmptyConditioningGenerator
                EmptyCondGen       = EmptyConditioningGenerator(work_model)
                positive, negative = EmptyCondGen.zero_none_conditionings_([positive, negative])

            if cfg < 0:
                sampler.extra_options['cfg_cw'] = -cfg
                cfg = 1.0
            else:
                sampler.extra_options.pop("cfg_cw", None) 

            
            if not EO("disable_dummy_sampler_init"):
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
                
            elif sampler_mode.startswith("resample"):
                null   = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.cat([null, sigmas])
                sigmas = torch.cat([sigmas, null])



            latent_x = {}
            # INIT STATE INFO FOR CONTINUING GENERATION ACROSS MULTIPLE SAMPLER NODES
            if latent_image is not None:
                latent_x['samples'] = latent_image['samples'].clone()
                if 'noise_mask' in latent_image:
                    latent_x['noise_mask'] = latent_image['noise_mask'].clone()
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
                        data_prev_ = data_prev_.squeeze(1) 

                        if data_prev_.dim() == 4: 
                            data_prev_ = F.interpolate(
                                data_prev_,
                                size=latent_x['samples'].shape[-2:],
                                mode=ultracascade_stage_up_upscale_mode,
                                align_corners=ultracascade_stage_up_upscale_align_corners
                                )
                        else:
                            print("data_prev_ upscale failed.")
                        state_info['data_prev_'] = data_prev_.unsqueeze(1)
                    
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

            latent_image_batch = {"samples": latent_x['samples'].clone()}
            if 'noise_mask' in latent_x and latent_x['noise_mask'] is not None:
                latent_image_batch['noise_mask'] = latent_x['noise_mask'].clone()
            
            # UNROLL BATCHES
            
            out_samples          = []
            out_denoised_samples = []
            out_state_info       = []
            
            for batch_num in range(latent_image_batch['samples'].shape[0]):
                latent_unbatch            = copy.deepcopy(latent_x)
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


                x = latent_unbatch["samples"].clone().to(default_dtype) # does this type carry into clown after passing through comfy?



                if sde_noise is None and sampler_mode.startswith("unsample"):
                    sde_noise = []
                else:
                    sde_noise_steps = 1

                for total_steps_iter in range (sde_noise_steps):
                        
                    if noise_type_init == "none" or noise_stdev == 0.0:
                        noise = torch.zeros_like(x)
                    else:
                        RESplain("Initial latent noise seed: ", seed, debug=True)
                        
                        noise_sampler_init = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type_init)(x=x, seed=seed, sigma_max=sigma_max, sigma_min=sigma_min)
                    
                        if noise_type_init == "fractal":
                            noise_sampler_init.alpha = alpha_init
                            noise_sampler_init.k     = k_init
                            noise_sampler_init.scale = 0.1
                            
                        """if EO("rare_noise"):
                            noise, _, _ = sample_most_divergent_noise(noise_sampler_init, sigma_max, sigma_min, EO("rare_noise", 100))
                        else:
                            noise = noise_sampler_init(sigma=sigma_max * noise_stdev, sigma_next=sigma_min)"""
                        noise = noise_sampler_init(sigma=sigma_max * noise_stdev, sigma_next=sigma_min)          # is sigma_max * noise_stdev really a good idea here?
                        
                        

                    if noise_normalize and noise.std() > 0:
                        channelwise = EO("init_noise_normalize_channelwise", "true")
                        channelwise = True if channelwise == "true" else False
                        noise = normalize_zscore(noise, channelwise=channelwise, inplace=True)
                        
                    noise *= noise_stdev
                    noise = (noise - noise.mean()) + noise_mean
                    
                    if 'BONGMATH' in sampler.extra_options:
                        sampler.extra_options['noise_initial'] = noise
                        sampler.extra_options['image_initial'] = x

                    noise_mask = latent_unbatch["noise_mask"] if "noise_mask" in latent_unbatch else None

                    x0_output = {}

                    if latent_image is not None and 'state_info' in latent_image and 'sigmas' in latent_image['state_info']:
                        steps_len = max(sigmas.shape[-1] - 1,    latent_image['state_info']['sigmas'].shape[-1]-1)
                    else:
                        steps_len = sigmas.shape[-1]-1
                    callback     = latent_preview.prepare_callback(work_model, steps_len, x0_output)

                    if 'BONGMATH' in sampler.extra_options: # verify the sampler is rk_sampler_beta()
                        sampler.extra_options['state_info']     = copy.deepcopy(state_info)         ##############################
                        if state_info != {} and state_info != {'data_prev_': None}:  #second condition is for ultracascade
                            sampler.extra_options['state_info']['raw_x']            = state_info['raw_x']           [batch_num]
                            sampler.extra_options['state_info']['data_prev_']       = state_info['data_prev_']      [batch_num]
                            sampler.extra_options['state_info']['last_rng']         = state_info['last_rng']        [batch_num]
                            sampler.extra_options['state_info']['last_rng_substep'] = state_info['last_rng_substep'][batch_num]
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
                        
                        guider.set_conds(xt_positive=pos_cond_tmp, xt_negative=neg_cond)
                        
                    elif type(guider) == SharkGuider:
                        guider.set_cfgs(xt=cfg)
                        guider.set_conds(xt_positive=pos_cond_tmp, xt_negative=neg_cond)
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

                    samples = guider.sample(noise, x.clone(), sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

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
                                sampler.extra_options['steps_to_run'] = steps_to_run_cached

                                
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

            # STACK SDE NOISES, SAVE STATE INFO
            state_info_out = out_state_info[0]
            if 'raw_x' in out_state_info[0]:
                state_info_out['raw_x']            = torch.stack([out_state_info[_]['raw_x']            for _ in range(len(out_state_info))])
                state_info_out['data_prev_']       = torch.stack([out_state_info[_]['data_prev_']       for _ in range(len(out_state_info))])
                state_info_out['last_rng']         = torch.stack([out_state_info[_]['last_rng']         for _ in range(len(out_state_info))])
                state_info_out['last_rng_substep'] = torch.stack([out_state_info[_]['last_rng_substep'] for _ in range(len(out_state_info))])
            elif 'raw_x' in state_info:
                state_info_out = state_info

            out_samples             = [tensor.squeeze(0) for tensor in out_samples]
            out_denoised_samples    = [tensor.squeeze(0) for tensor in out_denoised_samples]

            out         ['samples'] = torch.stack(out_samples,          dim=0)
            out_denoised['samples'] = torch.stack(out_denoised_samples, dim=0)

            out['state_info']       = copy.deepcopy(state_info_out)
            state_info              = {}
            
            out['positive'] = positive
            out['negative'] = negative
            out['model']    = work_model#.clone()
            out['sampler']  = sampler
            

            return (out, out_denoised, sde_noise,)




class SharkSampler_Beta:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler":       (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "steps":           ("INT",                        {"default": 30,  "min": 1,        "max": 10000.0}),
                "steps_to_run":    ("INT",                        {"default": -1,  "min": -1,       "max": MAX_STEPS}),
                "denoise":         ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01}),
                "cfg":             ("FLOAT",                      {"default": 5.5, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
                "seed":            ("INT",                        {"default": 0,   "min": -1,       "max": 0xffffffffffffffff}),
                "sampler_mode": (['unsample', 'standard', 'resample'], {"default": "standard"}),
                },
            "optional": {
                "model":           ("MODEL",),
                "positive":        ("CONDITIONING", ),
                "negative":        ("CONDITIONING", ),
                "sampler":         ("SAMPLER", ),
                "sigmas":          ("SIGMAS", ),
                "latent_image":    ("LATENT", ),     
                "options":         ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("LATENT", 
                    "LATENT", 
                    "OPTIONS",)
    
    RETURN_NAMES = ("output", 
                    "denoised",
                    "options",) 
    
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/samplers"
    
    def main(self, 
            model                                    = None,
            cfg             : float                  =  5.5, 
            scheduler       : str                    = "beta57", 
            steps           : int                    = 30, 
            steps_to_run    : int                    = -1,
            sampler_mode    : str                    = "standard",
            denoise         : float                  =  1.0, 
            denoise_alt     : float                  =  1.0,
            noise_type_init : str                    = "gaussian",
            latent_image    : Optional[dict[Tensor]] = None,
            
            positive                                 = None,
            negative                                 = None,
            sampler                                  = None,
            sigmas          : Optional[Tensor]       = None,
            noise_stdev     : float                  =  1.0,
            noise_mean      : float                  =  0.0,
            noise_normalize : bool                   = True,
            
            d_noise         : float                  =  1.0,
            alpha_init      : float                  = -1.0,
            k_init          : float                  =  1.0,
            cfgpp           : float                  =  0.0,
            seed            : int                    = -1,
            options                                  = None,
            sde_noise                                = None,
            sde_noise_steps : int                    =  1,
        
            extra_options   : str                    = "", 
            **kwargs,
            ): 
        

        options_mgr = OptionsManager(options, **kwargs)
        
        if denoise < 0:
            denoise_alt = -denoise
            denoise = 1.0
        
        #if 'steps_to_run' in sampler.extra_options:
        #    sampler.extra_options['steps_to_run'] = steps_to_run
        if 'positive' in latent_image and positive is None:
            positive = latent_image['positive']
        if 'negative' in latent_image and negative is None:
            negative = latent_image['negative']
        if 'sampler'  in latent_image and sampler  is None:
            sampler  = latent_image['sampler']
        if 'model'    in latent_image and model    is None:
            model    = latent_image['model']
            
        #if model.model.model_config.unet_config.get('stable_cascade_stage') == 'b':
        #    if 'noise_type_sde' in sampler.extra_options:
        #        noise_type_sde         = "pyramid-cascade_B"
        #        noise_type_sde_substep = "pyramid-cascade_B"
        
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
            options         = options, 
            sde_noise       = sde_noise, 
            sde_noise_steps = sde_noise_steps, 
            noise_type_init = noise_type_init,
            noise_stdev     = noise_stdev,
            sampler_mode    = sampler_mode,
            denoise_alt     = denoise_alt,
            sigmas          = sigmas,

            extra_options   = extra_options)
        
        return (output, denoised,options_mgr.as_dict())





class SharkChainsampler_Beta(SharkSampler_Beta):  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps_to_run":    ("INT",                        {"default": -1,  "min": -1,       "max": MAX_STEPS}),
                "cfg":             ("FLOAT",                      {"default": 5.5, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
                "sampler_mode": (['unsample', 'resample'], {"default": "resample"}),
                },
            "optional": {
                "model":           ("MODEL",),
                "positive":        ("CONDITIONING", ),
                "negative":        ("CONDITIONING", ),
                "sampler":         ("SAMPLER", ),
                "sigmas":          ("SIGMAS", ),
                "latent_image":    ("LATENT", ),     
                "options":         ("OPTIONS", ),   
                }
            }

    def main(self, 
            model                 = None,
            steps_to_run          = -1, 
            cfg                   = 5.5, 
            latent_image          = None,
            sigmas                = None,
            sampler_mode          = "",
            seed            : int = -1, 
             **kwargs):  
        
        steps = latent_image['state_info']['sigmas'].shape[-1] - 3
        sigmas = latent_image['state_info']['sigmas'] if sigmas is None else sigmas
        if len(sigmas) > 2 and sigmas[1] < sigmas[2] and latent_image['state_info']['sampler_mode'] == "unsample" and sampler_mode == "resample":
            sigmas = torch.flip(sigmas, dims=[0])
        
        return super().main(model=model, sampler_mode=sampler_mode, steps_to_run=steps_to_run, sigmas=sigmas, steps=steps, cfg=cfg, seed=seed, latent_image=latent_image, **kwargs)





class ClownSamplerAdvanced_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "noise_type_sde":         (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde_substep": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde":         (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "noise_mode_sde_substep": (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot_mode":         (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How step size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot_mode_substep": (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How substep size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta":                    ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "eta_substep":            ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "overshoot":              ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising step, then rescale to match the original. Has a softening effect."}),
                    "overshoot_substep":      ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising substep, then rescale to match the original. Has a softening effect."}),
                    "noise_scaling_weight":   ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_boost_step":       ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_boost_substep":    ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_anchor":           ("FLOAT",                      {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."}),
                    "s_noise":                ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "s_noise_substep":        ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "d_noise":                ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
                    "momentum":               ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Accelerate convergence with positive values when sampling, negative values when unsampling."}),
                    "noise_seed_sde":         ("INT",                        {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_name":           (get_sampler_name_list(),      {"default": get_default_sampler_name()}), 

                    "implicit_type":          (IMPLICIT_TYPE_NAMES,          {"default": "predictor-corrector"}), 
                    "implicit_type_substeps": (IMPLICIT_TYPE_NAMES,          {"default": "predictor-corrector"}), 
                    "implicit_steps":         ("INT",                        {"default": 0, "min": 0, "max": 10000}),
                    "implicit_substeps":      ("INT",                        {"default": 0, "min": 0, "max": 10000}),
                    "bongmath":               ("BOOLEAN",                    {"default": True}),
                    },
                "optional": 
                    {
                    "guides":                 ("GUIDES", ),     
                    "automation":             ("AUTOMATION", ),
                    "extra_options":          ("STRING",                     {"default": "", "multiline": True}),   
                    "options":                ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler", ) 
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/samplers"
    EXPERIMENTAL = True
    
    def main(self, 
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
            options                       = None,
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
            
            rk_swap_step                  : int = MAX_STEPS,
            rk_swap_print                 : bool = False,
            rk_swap_threshold             : float = 0.0,
            rk_swap_type                  : str = "",
            
            steps_to_run                  : int = -1,
            
            sde_mask                      : Optional[Tensor] = None,
            
            **kwargs,
            ): 
        
        
        
            options_mgr = OptionsManager(options, **kwargs)
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
            
            rk_swap_step      = options_mgr.get('rk_swap_step'     , rk_swap_step)
            rk_swap_print     = options_mgr.get('rk_swap_print'    , rk_swap_print)
            rk_swap_threshold = options_mgr.get('rk_swap_threshold', rk_swap_threshold)
            rk_swap_type      = options_mgr.get('rk_swap_type'     , rk_swap_type)

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
                    
                    "rk_swap_step"                  : rk_swap_step,
                    "rk_swap_print"                 : rk_swap_print,
                    "rk_swap_threshold"             : rk_swap_threshold,
                    "rk_swap_type"                  : rk_swap_type,
                    
                    "steps_to_run"                  : steps_to_run,
                    
                    "sde_mask"                      : sde_mask,
                    
                    "momentum"                      : momentum,
                })


            return (sampler, )







class ClownsharKSampler_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required":
                    {
                    "eta":          ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "sampler_name": (get_sampler_name_list     (), {"default": get_default_sampler_name()}), 
                    "scheduler":    (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "steps":        ("INT",                        {"default": 30,  "min":  1,     "max": MAX_STEPS}),
                    "steps_to_run": ("INT",                        {"default": -1,  "min": -1,     "max": MAX_STEPS}),
                    "denoise":      ("FLOAT",                      {"default": 1.0, "min": -10000, "max": MAX_STEPS, "step":0.01}),
                    "cfg":          ("FLOAT",                      {"default": 5.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, }),
                    "seed":         ("INT",                        {"default": 0,   "min": -1,     "max": 0xffffffffffffffff}),
                    "sampler_mode": (['unsample', 'standard', 'resample'], {"default": "standard"}),
                    "bongmath":     ("BOOLEAN",                    {"default": True}),
                    },
                "optional": 
                    {
                    "model":        ("MODEL",),
                    "positive":     ("CONDITIONING",),
                    "negative":     ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    "sigmas":       ("SIGMAS",), 
                    "guides":       ("GUIDES",), 
                    "options":      ("OPTIONS", {}),   
                    }
                }
        
        return inputs

    RETURN_TYPES = ("LATENT", 
                    "LATENT",
                    "OPTIONS",
                    )
    
    RETURN_NAMES = ("output", 
                    "denoised",
                    "options",
                    ) 
    
    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"
    
    def main(self, 
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
            options                                                = None, 
            sde_noise                                              = None,
            sde_noise_steps               : int                    = 1, 
            extra_options                 : str                    = "", 
            automation                                             = None, 

            epsilon_scales                : Optional[Tensor]       = None, 
            regional_conditioning_weights : Optional[Tensor]       = None,
            frame_weights_mgr                                      = None, 


            rescale_floor                 : bool                   = True, 
            
            rk_swap_step                  : int                    = MAX_STEPS,
            rk_swap_print                 : bool                   = False,
            rk_swap_threshold             : float                  = 0.0,
            rk_swap_type                  : str                    = "",
            
            sde_mask                      : Optional[Tensor]       = None,

            #start_at_step                 : int                    = 0,
            #stop_at_step                  : int                    = MAX_STEPS,
                        
            **kwargs
            ): 
        
        options_mgr = OptionsManager(options, **kwargs)
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

        if latent_image is not None and 'positive' in latent_image and positive is None:
            positive = latent_image['positive']
            is_chained = True
        if latent_image is not None and 'negative' in latent_image and negative is None:
            negative = latent_image['negative']
            is_chained = True
        if latent_image is not None and 'model' in latent_image and model is None:
            model = latent_image['model']
            is_chained = True
        
        guider = options_mgr.get('guider', None)
        if is_chained is False and guider is not None:
            model = guider.model_patcher

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
        
        rk_swap_type           = options_mgr.get('rk_swap_type'          , rk_swap_type)
        rk_swap_step           = options_mgr.get('rk_swap_step'          , rk_swap_step)
        rk_swap_threshold      = options_mgr.get('rk_swap_threshold'     , rk_swap_threshold)
        rk_swap_print          = options_mgr.get('rk_swap_print'         , rk_swap_print)
        
        sde_mask               = options_mgr.get('sde_mask'              , sde_mask)

        
        #start_at_step          = options_mgr.get('start_at_step'         , start_at_step)
        #stop_at_ste            = options_mgr.get('stop_at_step'          , stop_at_step)
                
        if channelwise_cfg: # != 1.0:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw




        sampler, = ClownSamplerAdvanced_Beta().main(
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
            options                       = options_mgr.as_dict(),

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
            
            rk_swap_step                  = rk_swap_step,
            rk_swap_print                 = rk_swap_print,
            rk_swap_threshold             = rk_swap_threshold,
            rk_swap_type                  = rk_swap_type,
            
            steps_to_run                  = steps_to_run,
            
            sde_mask                      = sde_mask,
            
            bongmath                      = bongmath,
            )
            
        
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
        
        return (output, denoised, options_mgr.as_dict(),) # {'model':model,},)











class ClownsharkChainsampler_Beta(ClownsharKSampler_Beta):  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta":          ("FLOAT",                 {"default": 0.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                "sampler_name": (get_sampler_name_list(), {"default": get_default_sampler_name()}), 
                "steps_to_run": ("INT",                   {"default": -1,  "min": -1,       "max": MAX_STEPS}),
                "cfg":          ("FLOAT",                 {"default": 5.5, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
                "sampler_mode": (['unsample', 'resample'],{"default": "resample"}),
                "bongmath":     ("BOOLEAN",               {"default": True}),
                },
            "optional": {
                "model":        ("MODEL",),
                "positive":     ("CONDITIONING", ),
                "negative":     ("CONDITIONING", ),
                #"sampler":      ("SAMPLER", ),
                "sigmas":       ("SIGMAS", ),
                "latent_image": ("LATENT", ),     
                "guides":       ("GUIDES", ),   
                "options":      ("OPTIONS", ),   
                }
            }

    def main(self, 
            eta                   =  0.5,
            sampler_name          = "res_2m",
            steps_to_run          = -1, 
            cfg                   =  5.5, 
            bongmath              = True,
            seed            : int = -1, 
            latent_image          = None,
            sigmas                = None,
            sampler_mode          = "",
            
             **kwargs):  
        
        steps = latent_image['state_info']['sigmas'].shape[-1] - 3
        sigmas = latent_image['state_info']['sigmas'] if sigmas is None else sigmas
        if len(sigmas) > 2 and sigmas[1] < sigmas[2] and latent_image['state_info']['sampler_mode'] == "unsample" and sampler_mode == "resample":
            sigmas = torch.flip(sigmas, dims=[0])
        
        return super().main(eta=eta, sampler_name=sampler_name, sampler_mode=sampler_mode, sigmas=sigmas, steps_to_run=steps_to_run, steps=steps, cfg=cfg, bongmath=bongmath, seed=seed, latent_image=latent_image, **kwargs)






class ClownSampler_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required":
                    {
                    "eta":          ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "sampler_name": (get_sampler_name_list     (), {"default": get_default_sampler_name()}), 
                    "seed":         ("INT",                        {"default": -1,   "min": -1,     "max": 0xffffffffffffffff}),
                    "bongmath":     ("BOOLEAN",                    {"default": True}),
                    },
                "optional": 
                    {
                    "guides":       ("GUIDES",), 
                    "options":      ("OPTIONS", {}),   
                    }
                }
        
        return inputs

    RETURN_TYPES = ("SAMPLER",)
    
    RETURN_NAMES = ("sampler",) 
    
    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"
    
    def main(self, 
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
            options                                                = None, 
            sde_noise                                              = None,
            sde_noise_steps               : int                    = 1, 
            extra_options                 : str                    = "", 
            automation                                             = None, 

            epsilon_scales                : Optional[Tensor]       = None, 
            regional_conditioning_weights : Optional[Tensor]       = None,
            frame_weights_mgr                                      = None, 


            rescale_floor                 : bool                   = True, 
            
            rk_swap_step                  : int                    = MAX_STEPS,
            rk_swap_print                 : bool                   = False,
            rk_swap_threshold             : float                  = 0.0,
            rk_swap_type                  : str                    = "",
            
            sde_mask                      : Optional[Tensor]       = None,

            
            #start_at_step                 : int                    = 0,
            #stop_at_step                  : int                    = MAX_STEPS,
                        
            **kwargs
            ): 
        
        options_mgr = OptionsManager(options, **kwargs)
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
        
        rk_swap_type           = options_mgr.get('rk_swap_type'          , rk_swap_type)
        rk_swap_step           = options_mgr.get('rk_swap_step'          , rk_swap_step)
        rk_swap_threshold      = options_mgr.get('rk_swap_threshold'     , rk_swap_threshold)
        rk_swap_print          = options_mgr.get('rk_swap_print'         , rk_swap_print)
        
        sde_mask               = options_mgr.get('sde_mask'              , sde_mask)

        
        #start_at_step          = options_mgr.get('start_at_step'         , start_at_step)
        #stop_at_ste            = options_mgr.get('stop_at_step'          , stop_at_step)
                
        if channelwise_cfg: # != 1.0:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw

        noise_seed_sde = seed


        sampler, = ClownSamplerAdvanced_Beta().main(
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
            options                       = options_mgr.as_dict(),

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
            
            rk_swap_step                  = rk_swap_step,
            rk_swap_print                 = rk_swap_print,
            rk_swap_threshold             = rk_swap_threshold,
            rk_swap_type                  = rk_swap_type,
            
            steps_to_run                  = steps_to_run,
            
            sde_mask                      = sde_mask,
            
            bongmath                      = bongmath,
            )
            
        return (sampler,)
    








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
            options                                                = None, 
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
            
            rk_swap_step                  : int                    = MAX_STEPS,
            rk_swap_print                 : bool                   = False,
            rk_swap_threshold             : float                  = 0.0,
            rk_swap_type                  : str                    = "",
            
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
        
        rk_swap_type           = options_mgr.get('rk_swap_type'          , rk_swap_type)
        rk_swap_step           = options_mgr.get('rk_swap_step'          , rk_swap_step)
        rk_swap_threshold      = options_mgr.get('rk_swap_threshold'     , rk_swap_threshold)
        rk_swap_print          = options_mgr.get('rk_swap_print'         , rk_swap_print)
        
        #start_at_step          = options_mgr.get('start_at_step'         , start_at_step)
        #stop_at_ste            = options_mgr.get('stop_at_step'          , stop_at_step)
                
        if channelwise_cfg: # != 1.0:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw




        sampler, = ClownSamplerAdvanced_Beta().main(
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
            options                       = options_mgr.as_dict(),

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
            
            rk_swap_step                  = rk_swap_step,
            rk_swap_print                 = rk_swap_print,
            rk_swap_threshold             = rk_swap_threshold,
            rk_swap_type                  = rk_swap_type,
            
            steps_to_run                  = steps_to_run,
            
            bongmath                      = bongmath,
            )
            
        
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