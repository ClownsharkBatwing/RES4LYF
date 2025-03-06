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

import latent_preview

from ..helper               import initialize_or_scale, get_res4lyf_scheduler_list, OptionsManager, ExtraOptions
from ..res4lyf              import RESplain
from ..latents              import normalize_zscore, get_orthogonal
from ..sigmas               import get_sigmas
import RES4LYF.models              # import ReFluxPatcher

from .constants             import MAX_STEPS, IMPLICIT_TYPE_NAMES
from .noise_classes         import NOISE_GENERATOR_CLASSES_SIMPLE, NOISE_GENERATOR_NAMES_SIMPLE, NOISE_GENERATOR_NAMES
from .rk_noise_sampler_beta import NOISE_MODE_NAMES
from .rk_coefficients_beta  import get_default_sampler_name, get_sampler_name_list, process_sampler_name



class SharkSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("MODEL",),
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
    
    def main(self, 
            model,
            cfg : float, 
            scheduler, 
            steps : int, 
            sampler_mode       : str = "standard",
            denoise            : float = 1.0, 
            denoise_alt        : float = 1.0,
            noise_type_init    : str = "gaussian",
            latent_image       : Optional[dict[Tensor]] = None,
            
            positive           = None,
            negative           = None,
            sampler            = None,
            sigmas             : Optional[Tensor] = None,
            noise_stdev        : float = 1.0,
            noise_mean         : float = 0.0,
            noise_normalize    : bool = True,
            
            d_noise            : float =  1.0,
            alpha_init         : float = -1.0,
            k_init             : float = 1.0,
            cfgpp              : float = 0.0,
            noise_seed         : int = -1,
            options            = None,
            sde_noise          = None,
            sde_noise_steps    : int = 1,
            
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
            
            options_mgr    = OptionsManager(options, **kwargs)
            extra_options += "\n" + options_mgr.get('extra_options', "")
            EO = ExtraOptions(extra_options)
            default_dtype = EO("default_dtype", torch.float64)

            
            noise_stdev     = options_mgr.get('noise_init_stdev', noise_stdev)
            noise_mean      = options_mgr.get('noise_init_mean',  noise_mean)
            noise_type_init = options_mgr.get('noise_type_init',  noise_type_init)
            d_noise         = options_mgr.get('d_noise',          d_noise)
            alpha_init      = options_mgr.get('alpha_init',       alpha_init)
            k_init          = options_mgr.get('k_init',           k_init)
            sde_noise       = options_mgr.get('sde_noise',        sde_noise)
            sde_noise_steps = options_mgr.get('sde_noise_steps',  sde_noise_steps)
            
            #ultracascade_stage        = options_mgr.get('ultracascade_stage',         ultracascade_stage)
            ultracascade_latent_image  = options_mgr.get('ultracascade_latent_image',  ultracascade_latent_image)
            ultracascade_latent_width  = options_mgr.get('ultracascade_latent_width',  ultracascade_latent_width)
            ultracascade_latent_height = options_mgr.get('ultracascade_latent_height', ultracascade_latent_height)

        
            if cfg < 0:
                sampler.extra_options['cfg_cw'] = -cfg
                cfg = 1.0
            else:
                sampler.extra_options.pop("cfg_cw", None) 

            work_model   = model.clone()
            sigma_min    = work_model.model.model_sampling.sigma_min
            sigma_max    = work_model.model.model_sampling.sigma_max
        
            if sampler is None:
                raise ValueError("sampler is required")
            else:
                sampler = copy.deepcopy(sampler)
        
        
        
            # INIT SIGMAS
            if sigmas is not None:
                sigmas = sigmas.clone().to(default_dtype) # does this type carry into clown after passing through comfy?
            else: 
                sigmas = get_sigmas(model, scheduler, steps, denoise).to(default_dtype)
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
                state_info     = copy.deepcopy(latent_image['state_info']) if 'state_info' in latent_image else {}
            else:
                state_info = {}
            state_info_out = {}
            
            
            
            # SETUP CONDITIONING EMBEDS
            
            pos_cond    = copy.deepcopy(positive)
            neg_cond    = copy.deepcopy(negative)
            
            
            
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
                        state_info['data_prev_'] = None
                
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
                if sampler_mode != "resample":
                    state_info['data_prev_'] = None
                
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
            
            if   isinstance(work_model.model.model_config, comfy.supported_models.Flux) or isinstance(work_model.model.model_config, comfy.supported_models.FluxSchnell):
                pos_cond = [[torch.zeros((1, 256, 4096)), {'pooled_output': torch.zeros((1,  768))}]] if pos_cond is None else pos_cond
                neg_cond = [[torch.zeros((1, 256, 4096)), {'pooled_output': torch.zeros((1,  768))}]] if neg_cond is None else neg_cond
                
            elif isinstance(work_model.model.model_config, comfy.supported_models.SD3):
                pos_cond = [[torch.zeros((1, 154, 4096)), {'pooled_output': torch.zeros((1, 2048))}]] if pos_cond is None else pos_cond
                neg_cond = [[torch.zeros((1, 154, 4096)), {'pooled_output': torch.zeros((1, 2048))}]] if neg_cond is None else neg_cond
            


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
            


            # SETUP FLUX REGIONAL COND

            if pos_cond[0][1] is not None: 
                if 'callback_regional' in pos_cond[0][1]:
                    if   isinstance(work_model.model.model_config, comfy.supported_models.SD3):
                        text_len_base = 154
                        pooled_len    = 2048
                    elif isinstance(work_model.model.model_config, comfy.supported_models.Flux) or isinstance(work_model.model.model_config, comfy.supported_models.FluxSchnell):
                        text_len_base = 256
                        pooled_len    = 768
                    
                    pos_cond = pos_cond[0][1]['callback_regional'](model)
                
                if "regional_conditioning_weights" in pos_cond[0][1]:
                    sampler.extra_options['regional_conditioning_weights'] = pos_cond[0][1]['regional_conditioning_weights']
                    sampler.extra_options['regional_conditioning_floors']  = pos_cond[0][1]['regional_conditioning_floors']
                    regional_generate_conditionings_and_masks_fn           = pos_cond[0][1]['regional_generate_conditionings_and_masks_fn']
                    regional_conditioning, regional_mask                   = regional_generate_conditionings_and_masks_fn(latent_x['samples'])
                    regional_conditioning                                  = copy.deepcopy(regional_conditioning)
                    regional_mask                                          = copy.deepcopy(regional_mask)
                    
                    if EO("enable_auto_enable_reflux"):
                        work_model, = RES4LYF.models.ReFluxPatcher().main(work_model, enable=True) #, force=True)
                        if EO("compile_reflux_model"):
                            work_model, = RES4LYF.models.TorchCompileModelFluxAdvanced().main(work_model)

                    work_model.set_model_patch(regional_conditioning, 'regional_conditioning_positive')
                    work_model.set_model_patch(regional_mask,         'regional_conditioning_mask')
                else:
                    if EO("enable_auto_disable_reflux"):
                        work_model, = RES4LYF.models.ReFluxPatcher().main(work_model, enable=False)
                        if EO("compile_reflux_model"):
                            work_model, = RES4LYF.models.TorchCompileModelFluxAdvanced().main(work_model)
            
            
            if "noise_seed" in sampler.extra_options:
                if sampler.extra_options['noise_seed'] == -1 and noise_seed != -1:
                    sampler.extra_options['noise_seed'] = noise_seed + 1
                    RESplain("Shark: setting clown noise seed to: ", sampler.extra_options['noise_seed'], debug=False)

            if "sampler_mode" in sampler.extra_options:
                sampler.extra_options['sampler_mode'] = sampler_mode

            if "extra_options" in sampler.extra_options:
                extra_options += "\n"
                extra_options += sampler.extra_options['extra_options']
                sampler.extra_options['extra_options'] = extra_options



            batch_size = EO("batch_size", 1)
            if batch_size > 1:
                latent_x['samples'] = latent_x['samples'].repeat(batch_size, 1, 1, 1) 
            
            latent_image_batch = {"samples": latent_x['samples']}
            
            
            
            # UNROLL BATCHES
            
            out_samples          = []
            out_denoised_samples = []
            
            for batch_num in range(latent_image_batch['samples'].shape[0]):
                latent_unbatch            = copy.deepcopy(latent_x)
                latent_unbatch['samples'] = latent_image_batch['samples'][batch_num].clone().unsqueeze(0)
                


                if noise_seed == -1:
                    seed = torch.initial_seed() + 1 + batch_num
                else:
                    seed = noise_seed + batch_num
                    torch.manual_seed(seed)
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
                        RESplain("Initial latent noise seed: ", seed, debug=False)
                        
                        noise_sampler_init = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type_init)(x=x, seed=seed, sigma_max=sigma_max, sigma_min=sigma_min)
                    
                        if noise_type_init == "fractal":
                            noise_sampler_init.alpha = alpha_init
                            noise_sampler_init.k     = k_init
                            noise_sampler_init.scale = 0.1
                        
                        noise = noise_sampler_init(sigma=sigma_max * noise_stdev, sigma_next=sigma_min)

                    if noise_normalize and noise.std() > 0:
                        noise = normalize_zscore(noise, channelwise=True, inplace=True)
                        
                    noise *= noise_stdev
                    noise = (noise - noise.mean()) + noise_mean

                    noise_mask = latent_unbatch["noise_mask"] if "noise_mask" in latent_unbatch else None

                    x0_output = {}


                    callback     = latent_preview.prepare_callback(work_model, sigmas.shape[-1] - 1, x0_output)
                    
                    if 'BONGMATH' in sampler.extra_options:
                        sampler.extra_options['state_info']     = state_info
                        sampler.extra_options['state_info_out'] = state_info_out
                    
                    samples = comfy.sample.sample_custom(work_model, noise, cfg, sampler, sigmas, pos_cond, neg_cond, x.clone(), noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)



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
                    
                    
                    
                    # INCREMENT BATCH LOOP
                    seed += 1
                    torch.manual_seed(seed)



            # STACK SDE NOISES, SAVE STATE INFO

            out_samples             = [tensor.squeeze(0) for tensor in out_samples]
            out_denoised_samples    = [tensor.squeeze(0) for tensor in out_denoised_samples]

            out         ['samples'] = torch.stack(out_samples,          dim=0)
            out_denoised['samples'] = torch.stack(out_denoised_samples, dim=0)

            out['state_info']       = copy.deepcopy(state_info_out)
            state_info              = {}
            
            gc.collect()

            return (out, out_denoised, sde_noise,)








class SharkSampler_Beta:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("MODEL",),
                "scheduler":       (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "steps":           ("INT",                        {"default": 30,  "min": 1,        "max": 10000.0}),
                "steps_to_run":    ("INT",                        {"default": -1,  "min": -1,       "max": MAX_STEPS}),
                "denoise":         ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01}),
                "cfg":             ("FLOAT",                      {"default": 5.5, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
                "seed":            ("INT",                        {"default": 0,   "min": -1,       "max": 0xffffffffffffffff}),
                "sampler_mode": (['unsample', 'standard', 'resample'], {"default": "standard"}),
                },
            "optional": {
                "positive":        ("CONDITIONING", ),
                "negative":        ("CONDITIONING", ),
                "sampler":         ("SAMPLER", ),
                "sigmas":          ("SIGMAS", ),
                "latent_image":    ("LATENT", ),     
                "options":         ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("LATENT", 
                    "LATENT",)  
                    #"LATENT",)
    
    RETURN_NAMES = ("output", 
                    "denoised",)
                    #"sde_noise",) 
    
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/samplers"
    
    def main(self, 
            model,
            cfg : float, 
            scheduler, 
            steps              : int = 30, 
            steps_to_run       : int = -1,
            sampler_mode       : str = "standard",
            denoise            : float = 1.0, 
            denoise_alt        : float = 1.0,
            noise_type_init    : str = "gaussian",
            latent_image       : Optional[dict[Tensor]] = None,
            
            positive           = None,
            negative           = None,
            sampler            = None,
            sigmas             : Optional[Tensor] = None,
            noise_stdev        : float = 1.0,
            noise_mean         : float = 0.0,
            noise_normalize    : bool = True,
            
            d_noise            : float =  1.0,
            alpha_init         : float = -1.0,
            k_init             : float = 1.0,
            cfgpp              : float = 0.0,
            seed               : int = -1,
            options            = None,
            sde_noise          = None,
            sde_noise_steps    : int = 1,
        
            extra_options      : str = "", 
            **kwargs,
            ): 
        
        if 'steps_to_run' in sampler.extra_options:
            sampler.extra_options['steps_to_run'] = steps_to_run
        
        output, denoised, sde_noise = SharkSampler().main(
            model           = model, 
            cfg             = cfg, 
            scheduler       = scheduler,
            steps           = steps, 
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
        
        return (output, denoised,)













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
                    "noise_boost_step":       ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_boost_substep":    ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_anchor":           ("FLOAT",                      {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."}),
                    "s_noise":                ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "s_noise_substep":        ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "d_noise":                ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
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
    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"
    
    def main(self, 
            noise_type_sde                : str = "gaussian",
            noise_type_sde_substep        : str = "gaussian",
            noise_mode_sde                : str = "hard",
            overshoot_mode                : str = "hard",
            overshoot_mode_substep        : str = "hard",
            
            eta                           : float = 0.5,
            eta_substep                   : float = 0.5,
            d_noise                       : float = 1.0,
            s_noise                       : float = 1.0,
            s_noise_substep               : float = 1.0,
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
            frame_weights_grp             = None,
            noise_mode_sde_substep        : str = "hard",
            
            overshoot                     : float = 0.0,
            overshoot_substep             : float = 0.0,
            noise_boost_step              : float = 0.0,
            noise_boost_substep           : float = 0.0,
            bongmath                      : bool = True,
            noise_anchor                  : float = 1.0,
            
            implicit_type                 : str = "predictor-corrector",
            implicit_type_substeps        : str = "predictor-corrector",
            
            rk_swap_step                  : int = MAX_STEPS,
            rk_swap_print                 : bool = False,
            rk_swap_threshold             : float = 0.0,
            rk_swap_type                  : str = "",
            
            steps_to_run                  : int = -1,
            
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
            s_noise           = options_mgr.get('s_noise'          , s_noise)
            d_noise           = options_mgr.get('d_noise'          , d_noise)
            alpha_sde         = options_mgr.get('alpha_sde'        , alpha_sde)
            k_sde             = options_mgr.get('k_sde'            , k_sde)
            c1                = options_mgr.get('c1'               , c1)
            c2                = options_mgr.get('c2'               , c2)
            c3                = options_mgr.get('c3'               , c3)

            frame_weights_grp = options_mgr.get('frame_weights_grp', frame_weights_grp)
            sde_noise         = options_mgr.get('sde_noise'        , sde_noise)
            sde_noise_steps   = options_mgr.get('sde_noise_steps'  , sde_noise_steps)
            
            rk_swap_step      = options_mgr.get('rk_swap_step'     , rk_swap_step)
            rk_swap_print     = options_mgr.get('rk_swap_print'    , rk_swap_print)
            rk_swap_threshold = options_mgr.get('rk_swap_threshold', rk_swap_threshold)
            rk_swap_type      = options_mgr.get('rk_swap_type'     , rk_swap_type)

            steps_to_run      = options_mgr.get('steps_to_run'     , steps_to_run)


            rescale_floor = EO("rescale_floor")

            if automation is not None:
                etas              = automation['etas']              if 'etas'              in automation else None
                etas_substep      = automation['etas_substep']      if 'etas_substep'      in automation else None
                s_noises          = automation['s_noises']          if 's_noises'          in automation else None
                s_noises_substep  = automation['s_noise_substep']   if 's_noise_substep'   in automation else None
                epsilon_scales    = automation['epsilon_scales']    if 'epsilon_scales'    in automation else None
                frame_weights_grp = automation['frame_weights_grp'] if 'frame_weights_grp' in automation else None

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
                    "s_noise"                       : s_noise,
                    "s_noise_substep"               : s_noise_substep,
                    "d_noise"                       : d_noise,
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
                    "frame_weights_grp"             : frame_weights_grp,
                    "eta_substep"                   : eta_substep,
                    "noise_mode_sde_substep"        : noise_mode_sde_substep,
                    "noise_boost_step"              : noise_boost_step,
                    "noise_boost_substep"           : noise_boost_substep,

                    "overshoot_mode"                : overshoot_mode,
                    "overshoot_mode_substep"        : overshoot_mode_substep,
                    "overshoot"                     : overshoot,
                    "overshoot_substep"             : overshoot_substep,
                    "BONGMATH"                      : bongmath,
                    "noise_anchor"                  : noise_anchor,

                    "implicit_type"                 : implicit_type,
                    "implicit_type_substeps"        : implicit_type_substeps,
                    
                    "rk_swap_step"                  : rk_swap_step,
                    "rk_swap_print"                 : rk_swap_print,
                    "rk_swap_threshold"             : rk_swap_threshold,
                    "rk_swap_type"                  : rk_swap_type,
                    
                    "steps_to_run"                  : steps_to_run,
                })


            return (sampler, )







class ClownsharKSampler_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required":
                    {
                    "model":        ("MODEL",),
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
                    "LATENT",)
    
    RETURN_NAMES = ("output", 
                    "denoised",) 
    
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
            frame_weights_grp                                      = None, 
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
        
        # defaults for ClownSampler
        eta_substep = eta
        
        # defaults for SharkSampler
        noise_type_init = "gaussian"
        noise_stdev     = 1.0
        denoise_alt     = 1.0
        channelwise_cfg = 1.0
        
        
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

        frame_weights_grp      = options_mgr.get('frame_weights_grp'     , frame_weights_grp)
        
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
                
        if channelwise_cfg != 1.0:
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

            noise_boost_step              = noise_boost_step,
            noise_boost_substep           = noise_boost_substep,
            
            epsilon_scales                = epsilon_scales,
            regional_conditioning_weights = regional_conditioning_weights,
            frame_weights_grp             = frame_weights_grp,
            
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
        
        return (output, denoised,)



















class ClownSampler_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required":
                    {
                    #"model":        ("MODEL",),
                    "eta":          ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "sampler_name": (get_sampler_name_list     (), {"default": get_default_sampler_name()}), 
                    #"scheduler":    (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    #"steps":        ("INT",                        {"default": 30,  "min":  1,     "max": MAX_STEPS}),
                    #"steps_to_run": ("INT",                        {"default": -1,  "min": -1,     "max": MAX_STEPS}),
                    #"denoise":      ("FLOAT",                      {"default": 1.0, "min": -10000, "max": MAX_STEPS, "step":0.01}),
                    #"cfg":          ("FLOAT",                      {"default": 5.5, "min": -100.0, "max": 100.0,     "step":0.01, "round": False, }),
                    "seed":         ("INT",                        {"default": -1,   "min": -1,     "max": 0xffffffffffffffff}),
                    #"sampler_mode": (['standard', 'unsample', 'resample'],),
                    "bongmath":     ("BOOLEAN",                    {"default": True}),
                    },
                "optional": 
                    {
                    #"positive":     ("CONDITIONING",),
                    #"negative":     ("CONDITIONING",),
                    #"latent_image": ("LATENT",),
                    #"sigmas":       ("SIGMAS",), 
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
            seed                          : int                    = 42, 
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
            frame_weights_grp                                      = None, 
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
        
        
        # defaults for ClownSampler
        eta_substep = eta
        
        # defaults for SharkSampler
        noise_type_init = "gaussian"
        noise_stdev     = 1.0
        denoise_alt     = 1.0
        channelwise_cfg = 1.0
        
        
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

        frame_weights_grp      = options_mgr.get('frame_weights_grp'     , frame_weights_grp)
        
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
                
        if channelwise_cfg != 1.0:
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

            noise_boost_step              = noise_boost_step,
            noise_boost_substep           = noise_boost_substep,
            
            epsilon_scales                = epsilon_scales,
            regional_conditioning_weights = regional_conditioning_weights,
            frame_weights_grp             = frame_weights_grp,
            
            sde_noise                     = sde_noise,
            sde_noise_steps               = sde_noise_steps,
            
            rk_swap_step                  = rk_swap_step,
            rk_swap_print                 = rk_swap_print,
            rk_swap_threshold             = rk_swap_threshold,
            rk_swap_type                  = rk_swap_type,
            
            steps_to_run                  = steps_to_run,
            
            bongmath                      = bongmath,
            )
            
        return (sampler,)
    
