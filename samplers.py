from .noise_classes import *
from .sigmas import get_sigmas
from .rk_sampler import sample_rk
from .rk_coefficients import RK_SAMPLER_NAMES, IRK_SAMPLER_NAMES

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd
from comfy_extras.nodes_model_advanced import ModelSamplingSD3, ModelSamplingFlux, ModelSamplingAuraFlow, ModelSamplingStableCascade
import comfy.supported_models

import latent_preview
import torch
import torch.nn.functional as F

import math
import copy

from .helper import get_extra_options_kv, extra_options_flag
from .latents import initialize_or_scale


    
def move_to_same_device(*tensors):
    if not tensors:
        return tensors

    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)


#SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["beta57"]

NOISE_MODE_NAMES = ["none",
                    "hard_sq",
                    "hard",
                    "lorentzian", 
                    "soft", 
                    "soft-linear",
                    "softer",
                    "sinusoidal",
                    "exp", 
                    "hard_var", 
                    ]



class SharkSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_stdev": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                    "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_mode": (['standard', 'unsample', 'resample'],),
                    "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "denoise_alt": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.1, "round": False, }),
                    #"shift": ("FLOAT", {"default": 1.35, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    #"base_shift": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    #"shift_scaling": (["exponential", "linear"], {"default": "exponential"}),
                     },
                "optional": 
                    {
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),     
                    "options": ("OPTIONS", ),   
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT", "LATENT",)
    RETURN_NAMES = ("output", "denoised","sde_noise",) 

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, model, cfg, sampler_mode, scheduler, steps, denoise=1.0, denoise_alt=1.0,
             noise_type_init="gaussian", latent_image=None, 
             positive=None, negative=None, sampler=None, sigmas=None, latent_noise=None, latent_noise_match=None,
             noise_stdev=1.0, noise_mean=0.0, noise_normalize=True, noise_is_latent=False, 
             d_noise=1.0, alpha_init=-1.0, k_init=1.0, cfgpp=0.0, noise_seed=-1,
                    shift=3.0, base_shift=0.85, options=None, sde_noise=None,sde_noise_steps=1, shift_scaling="exponential", unsampler_type="linear",
                    extra_options="", 
                    ): 

            latent_image_batch = {"samples": latent_image['samples']}
            out_samples, out_samples_fp64, out_denoised_samples, out_denoised_samples_fp64 = [], [], [], []
            for batch_num in range(latent_image_batch['samples'].shape[0]):
                latent_image['samples'] = latent_image_batch['samples'][batch_num].clone().unsqueeze(0)
                default_dtype = torch.float64
                max_steps = 10000

                if noise_seed == -1:
                    seed = torch.initial_seed() + 1 + batch_num
                else:
                    seed = noise_seed + batch_num
                    torch.manual_seed(noise_seed + batch_num)

                if options is not None:
                    noise_stdev = options.get('noise_init_stdev', noise_stdev)
                    noise_mean = options.get('noise_init_mean', noise_mean)
                    noise_type_init = options.get('noise_type_init', noise_type_init)
                    d_noise = options.get('d_noise', d_noise)
                    alpha_init = options.get('alpha_init', alpha_init)
                    k_init = options.get('k_init', k_init)
                    unsampler_type = options.get('unsampler_type', unsampler_type)
                    sde_noise = options.get('sde_noise', sde_noise)
                    sde_noise_steps = options.get('sde_noise_steps', sde_noise_steps)

                latent = latent_image
                latent_image_dtype = latent_image['samples'].dtype

                if positive is None:
                    positive = [[
                        torch.zeros((1, 154, 4096)),
                        {'pooled_output': torch.zeros((1, 2048))}
                        ]]
                
                if negative is None:
                    negative = [[
                        torch.zeros((1, 154, 4096)),
                        {'pooled_output': torch.zeros((1, 2048))}
                        ]]
                    
                if denoise_alt < 0:
                    d_noise = denoise_alt = -denoise_alt
                if options is not None:
                    d_noise = options.get('d_noise', d_noise)
                
                if sigmas is not None:
                    sigmas = sigmas.clone().to(default_dtype)
                else: 
                    sigmas = get_sigmas(model, scheduler, steps, denoise).to(default_dtype)
                sigmas *= denoise_alt

                if sampler_mode.startswith("unsample"): 
                    null = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                    sigmas = torch.flip(sigmas, dims=[0])
                    sigmas = torch.cat([sigmas, null])
                elif sampler_mode.startswith("resample"):
                    null = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                    sigmas = torch.cat([null, sigmas])
                    sigmas = torch.cat([sigmas, null])
                
                if sampler_mode.startswith("unsample_"):
                    unsampler_type = sampler_mode.split("_", 1)[1]
                elif sampler_mode.startswith("resample_"):
                    unsampler_type = sampler_mode.split("_", 1)[1]
                else:
                    unsampler_type = ""
                    
                x = latent_image["samples"].clone().to(default_dtype) 
                if latent_image is not None:
                    if "samples_fp64" in latent_image:
                        if latent_image['samples'].shape == latent_image['samples_fp64'].shape:
                            if torch.norm(latent_image['samples'] - latent_image['samples_fp64']) < 0.01:
                                x = latent_image["samples_fp64"].clone()
                    
                if latent_noise is not None:
                    latent_noise["samples"] = latent_noise["samples"].clone().to(default_dtype)  
                if latent_noise_match is not None:
                    latent_noise_match["samples"] = latent_noise_match["samples"].clone().to(default_dtype)

                truncate_conditioning = extra_options_flag("truncate_conditioning", extra_options)
                if truncate_conditioning == "true" or truncate_conditioning == "true_and_zero_neg":
                    if positive is not None:
                        positive[0][0] = positive[0][0].clone().to(default_dtype)
                        positive[0][1]["pooled_output"] = positive[0][1]["pooled_output"].clone().to(default_dtype)
                    if negative is not None:
                        negative[0][0] = negative[0][0].clone().to(default_dtype)
                        negative[0][1]["pooled_output"] = negative[0][1]["pooled_output"].clone().to(default_dtype)
                    c = []
                    for t in positive:
                        d = t[1].copy()
                        pooled_output = d.get("pooled_output", None)
                        if pooled_output is not None:default_dtype = torch.float64
                    for t in negative:
                        d = t[1].copy()
                        pooled_output = d.get("pooled_output", None)
                        if pooled_output is not None:
                            if truncate_conditioning == "true_and_zero_neg":
                                d["pooled_output"] = torch.zeros((1,2048), dtype=t[0].dtype, device=t[0].device)
                                n = [torch.zeros((1,154,4096), dtype=t[0].dtype, device=t[0].device), d]
                            else:
                                d["pooled_output"] = d["pooled_output"][:, :2048]
                                n = [t[0][:, :154, :4096], d]
                        c.append(n)
                    negative = c
                
                sigmin = model.model.model_sampling.sigma_min
                sigmax = model.model.model_sampling.sigma_max

                if sde_noise is None and sampler_mode.startswith("unsample"):
                    total_steps = len(sigmas)+1
                    sde_noise = []
                else:
                    total_steps = 1

                for total_steps_iter in range (sde_noise_steps):
                        
                    if noise_type_init == "none":
                        noise = torch.zeros_like(x)
                    elif latent_noise is None:
                        noise_sampler_init = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type_init)(x=x, seed=seed, sigma_min=sigmin, sigma_max=sigmax)
                    
                        if noise_type_init == "fractal":
                            noise_sampler_init.alpha = alpha_init
                            noise_sampler_init.k = k_init
                            noise_sampler_init.scale = 0.1
                        noise = noise_sampler_init(sigma=sigmax, sigma_next=sigmin)
                    else:
                        noise = latent_noise["samples"]

                    if noise_is_latent: #add noise and latent together and normalize --> noise
                        noise += x.cpu()
                        noise.sub_(noise.mean()).div_(noise.std())

                    if noise_normalize and noise.std() > 0:
                        noise.sub_(noise.mean()).div_(noise.std())
                    noise *= noise_stdev
                    noise = (noise - noise.mean()) + noise_mean
                    
                    if latent_noise_match:
                        for i in range(latent_noise_match["samples"].shape[1]):
                            noise[0][i] = (noise[0][i] - noise[0][i].mean())
                            noise[0][i] = (noise[0][i]) + latent_noise_match["samples"][0][i].mean()

                    noise_mask = latent["noise_mask"] if "noise_mask" in latent else None

                    x0_output = {}


                    
                    if cfg < 0:
                        cfgpp = -cfg
                        cfg = 1.0
                        
                    if sde_noise is None:
                        sde_noise = []
                    else:
                        sde_noise = copy.deepcopy(sde_noise)
                        for i in range(len(sde_noise)):
                            sde_noise[i] = sde_noise[i].to('cuda')
                            for j in range(sde_noise[i].shape[1]):
                                sde_noise[i][0][j] = ((sde_noise[i][0][j] - sde_noise[i][0][j].mean()) / sde_noise[i][0][j].std()) #.to('cuda')
                                
                    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
                    #disable_pbar = False
                    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, x.clone(), noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

                    out = latent.copy()
                    out["samples"] = samples
                    if "x0" in x0_output:
                        out_denoised = latent.copy()
                        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
                    else:
                        out_denoised = out
                    
                    out["samples_fp64"] = out["samples"].clone()
                    out["samples"] = out["samples"].to(latent_image_dtype)
                    
                    out_denoised["samples_fp64"] = out_denoised["samples"].clone()
                    out_denoised["samples"] = out_denoised["samples"].to(latent_image_dtype)
                    
                    out_samples.append(out["samples"])
                    out_samples_fp64.append(out["samples_fp64"])
                    
                    out_denoised_samples.append(out_denoised["samples"])
                    out_denoised_samples_fp64.append(out_denoised["samples_fp64"])
                    
                    seed += 1
                    torch.manual_seed(seed)
                    if total_steps_iter > 1: 
                        sde_noise.append(out["samples_fp64"])
                        
            out_samples = [tensor.squeeze(0) for tensor in out_samples]
            out_samples_fp64 = [tensor.squeeze(0) for tensor in out_samples_fp64]
            out_denoised_samples = [tensor.squeeze(0) for tensor in out_denoised_samples]
            out_denoised_samples_fp64 = [tensor.squeeze(0) for tensor in out_denoised_samples_fp64]

            out['samples'] = torch.stack(out_samples, dim=0)
            out['samples_fp64'] = torch.stack(out_samples_fp64, dim=0)
            
            out_denoised['samples'] = torch.stack(out_denoised_samples, dim=0)
            out_denoised['samples_fp64'] = torch.stack(out_denoised_samples_fp64, dim=0)

            return ( out, out_denoised, sde_noise,)






class ClownSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "noise_type_sde": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "d_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "noise_seed_sde": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_name": (RK_SAMPLER_NAMES, {"default": "res_2m"}), 
                    "implicit_sampler_name": (IRK_SAMPLER_NAMES, {"default": "gauss-legendre_2s"}), 
                    "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     },
                "optional": 
                    {
                    "guides": ("GUIDES", ),     
                    "options": ("OPTIONS", ),   
                    "automation": ("AUTOMATION", ),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    }
                }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler", ) 

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, 
             noise_type_sde="brownian", noise_mode_sde="hard",
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, noise_seed_sde=-1, sampler_name="res_2m", implicit_sampler_name="gauss-legendre_2s",
                    t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, guide_mode="blend", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None, unsampler_type="linear",
                    guides=None, options=None, sde_noise=None,sde_noise_steps=1, 
                    extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, 
                    ): 
            if implicit_sampler_name == "none":
                implicit_steps = 0 
                implicit_sampler_name = "gauss-legendre_2s"

            if noise_mode_sde == "none":
                eta, eta_var = 0.0, 0.0
                noise_mode_sde = "hard"
        
            default_dtype = torch.float64
            max_steps = 10000

            unsample_resample_scales_override = unsample_resample_scales

            if options is not None:
                noise_type_sde = options.get('noise_type_sde', noise_type_sde)
                noise_mode_sde = options.get('noise_mode_sde', noise_mode_sde)
                eta = options.get('eta', eta)
                s_noise = options.get('s_noise', s_noise)
                d_noise = options.get('d_noise', d_noise)
                alpha_sde = options.get('alpha_sde', alpha_sde)
                k_sde = options.get('k_sde', k_sde)
                noise_seed_sde = options.get('noise_seed_sde', noise_seed_sde)
                c1 = options.get('c1', c1)
                c2 = options.get('c2', c2)
                c3 = options.get('c3', c3)
                t_fn_formula = options.get('t_fn_formula', t_fn_formula)
                sigma_fn_formula = options.get('sigma_fn_formula', sigma_fn_formula)
                unsampler_type = options.get('unsampler_type', unsampler_type)
                
                sde_noise = options.get('sde_noise', sde_noise)
                sde_noise_steps = options.get('sde_noise_steps', sde_noise_steps)

            noise_seed_sde = torch.initial_seed()+1 if noise_seed_sde < 0 else noise_seed_sde 

            rescale_floor = extra_options_flag("rescale_floor", extra_options)

            if automation is not None:
                etas, s_noises, unsample_resample_scales = automation
            etas = initialize_or_scale(etas, eta, max_steps).to(default_dtype)
            etas = F.pad(etas, (0, max_steps), value=0.0)
            s_noises = initialize_or_scale(s_noises, s_noise, max_steps).to(default_dtype)
            s_noises = F.pad(s_noises, (0, max_steps), value=0.0)
        
            truncate_conditioning = extra_options_flag("truncate_conditioning", extra_options)
            if truncate_conditioning == "true" or truncate_conditioning == "true_and_zero_neg":
                if positive is not None:
                    positive[0][0] = positive[0][0].clone().to(default_dtype)
                    positive[0][1]["pooled_output"] = positive[0][1]["pooled_output"].clone().to(default_dtype)
                if negative is not None:
                    negative[0][0] = negative[0][0].clone().to(default_dtype)
                    negative[0][1]["pooled_output"] = negative[0][1]["pooled_output"].clone().to(default_dtype)
                c = []
                for t in positive:
                    d = t[1].copy()
                    pooled_output = d.get("pooled_output", None)
                    if pooled_output is not None:
                        d["pooled_output"] = d["pooled_output"][:, :2048]
                        n = [t[0][:, :154, :4096], d]
                    c.append(n)
                positive = c
                
                c = []
                for t in negative:
                    d = t[1].copy()
                    pooled_output = d.get("pooled_output", None)
                    if pooled_output is not None:
                        if truncate_conditioning == "true_and_zero_neg":
                            d["pooled_output"] = torch.zeros((1,2048), dtype=t[0].dtype, device=t[0].device)
                            n = [torch.zeros((1,154,4096), dtype=t[0].dtype, device=t[0].device), d]
                        else:
                            d["pooled_output"] = d["pooled_output"][:, :2048]
                            n = [t[0][:, :154, :4096], d]
                    c.append(n)
                negative = c

            if sde_noise is None:
                sde_noise = []
            else:
                sde_noise = copy.deepcopy(sde_noise)
                for i in range(len(sde_noise)):
                    sde_noise[i] = sde_noise[i].to('cuda')
                    for j in range(sde_noise[i].shape[1]):
                        sde_noise[i][0][j] = ((sde_noise[i][0][j] - sde_noise[i][0][j].mean()) / sde_noise[i][0][j].std()) #.to('cuda')
                        
            if unsample_resample_scales_override is not None:
                unsample_resample_scales = unsample_resample_scales_override

            sampler = comfy.samplers.ksampler("rk", {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha_sde, "k": k_sde, "c1": c1, "c2": c2, "c3": c3, "cfgpp": cfgpp, 
                                                    "noise_sampler_type": noise_type_sde, "noise_mode": noise_mode_sde, "noise_seed": noise_seed_sde, "rk_type": sampler_name, "implicit_sampler_name": implicit_sampler_name,
                                                            "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                            "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, "mask_inv": latent_guide_mask_inv,
                                                            "latent_guide_weights": latent_guide_weights, "latent_guide_weights_inv": latent_guide_weights_inv, "guide_mode": guide_mode, "unsampler_type": unsampler_type,
                                                            "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override, "sde_noise": sde_noise,
                                                            "extra_options": extra_options,
                                                            "etas": etas, "s_noises": s_noises, "unsample_resample_scales": unsample_resample_scales,
                                                            "guides": guides,
                                                            })

            return (sampler, )

                    




class ClownsharKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_mode": (['standard', 'unsample', 'resample'],),
                    "sampler_name": (RK_SAMPLER_NAMES, {"default": "res_2m"}), 
                    "implicit_sampler_name": (IRK_SAMPLER_NAMES, {"default": "gauss-legendre_2s"}), 
                    "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "denoise_alt": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.1, "round": False, }),
                    #"shift": ("FLOAT", {"default": 1.35, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    #"base_shift": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    #"shift_scaling": (["exponential", "linear"], {"default": "exponential"}),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                     },
                "optional": 
                    {
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),     
                    "guides": ("GUIDES", ),     
                    "options": ("OPTIONS", ),   
                    "automation": ("AUTOMATION", ),
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT", "LATENT",)
    RETURN_NAMES = ("output", "denoised","sde_noise",) 

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, model, cfg, sampler_mode, scheduler, steps, denoise=1.0, denoise_alt=1.0,
             noise_type_init="gaussian", noise_type_sde="brownian", noise_mode_sde="hard", latent_image=None, 
             positive=None, negative=None, sigmas=None, latent_noise=None, latent_noise_match=None,
             noise_stdev=1.0, noise_mean=0.0, noise_normalize=True, noise_is_latent=False, 
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_init=-1.0, k_init=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, noise_seed=-1, sampler_name="res_2m", implicit_sampler_name="default",
                    t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, guide_mode="blend", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None, unsampler_type="linear",
                    shift=3.0, base_shift=0.85, guides=None, options=None, sde_noise=None,sde_noise_steps=1, shift_scaling="exponential",
                    extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, 
                    ): 

        noise_seed_sde = -1

        sampler = ClownSampler().main(
                noise_type_sde, noise_mode_sde,
                eta, eta_var, d_noise, s_noise, alpha_sde, k_sde, cfgpp, c1, c2, c3, noise_seed_sde, sampler_name, implicit_sampler_name,
                t_fn_formula, sigma_fn_formula, implicit_steps,
                latent_guide, latent_guide_inv, guide_mode, latent_guide_weights, latent_guide_weights_inv, latent_guide_mask, latent_guide_mask_inv, rescale_floor, sigmas_override, unsampler_type,
                guides, options, sde_noise, sde_noise_steps, 
                extra_options, automation, etas, s_noises, unsample_resample_scales)
            
        return SharkSampler().main(
            model, cfg, sampler_mode, scheduler, steps, denoise, denoise_alt,
            noise_type_init, latent_image, 
            positive, negative, sampler[0], sigmas, latent_noise, latent_noise_match,
            noise_stdev, noise_mean, noise_normalize, noise_is_latent, 
            d_noise, alpha_init, k_init, cfgpp, noise_seed,
            shift, base_shift, options, sde_noise, sde_noise_steps, shift_scaling,
            extra_options)





class UltraSharkSampler:  
    # for use with https://github.com/ClownsharkBatwing/UltraCascade
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_is_latent": ("BOOLEAN", {"default": False}),
                "noise_type": (NOISE_GENERATOR_NAMES, ),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": 0.01}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": 0.01}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "latent_image": ("LATENT", ),               
                "guide_type": (['residual', 'weighted'], ),
                "guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
            },
            "optional": {
                "latent_noise": ("LATENT", ),
                "guide": ("LATENT",),
                "guide_weights": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("LATENT","LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output", "latent_batch")

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    DESCRIPTION = "For use with Stable Cascade and UltraCascade."
    
    def main(self, model, add_noise, noise_is_latent, noise_type, noise_seed, cfg, alpha, k, positive, negative, sampler, 
               sigmas, guide_type, guide_weight, latent_image, latent_noise=None, guide=None, guide_weights=None): 

            if model.model.model_config.unet_config.get('stable_cascade_stage') == 'up':
                model = model.clone()
                x_lr = guide['samples'] if guide is not None else None
                guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                #model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
                #model.model.diffusion_model.set_guide_type(guide_type=guide_type)
                #model.model.diffusion_model.set_x_lr(x_lr=x_lr)
                patch = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("ultracascade", {}).get("main")
                if patch is not None:
                    patch.update(x_lr=x_lr, guide_weights=guide_weights, guide_type=guide_type)
                else:
                    model.model.diffusion_model.set_sigmas_schedule(sigmas_schedule=sigmas)
                    model.model.diffusion_model.set_sigmas_prev(sigmas_prev=sigmas[:1])
                    model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
                    model.model.diffusion_model.set_guide_type(guide_type=guide_type)
                    model.model.diffusion_model.set_x_lr(x_lr=x_lr)
                
            elif model.model.model_config.unet_config['stable_cascade_stage'] == 'b':
                c_pos, c_neg = [], []
                for t in positive:
                    d_pos = t[1].copy()
                    d_neg = t[1].copy()
                    
                    d_pos['stable_cascade_prior'] = guide['samples']

                    pooled_output = d_neg.get("pooled_output", None)
                    if pooled_output is not None:
                        d_neg["pooled_output"] = torch.zeros_like(pooled_output)
                    
                    c_pos.append([t[0], d_pos])            
                    c_neg.append([torch.zeros_like(t[0]), d_neg])
                positive = c_pos
                negative = c_neg
        
            latent = latent_image
            latent_image = latent["samples"]
            torch.manual_seed(noise_seed)

            if not add_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
            elif latent_noise is None:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds, alpha, k)
            else:
                noise = latent_noise["samples"]#.to(torch.float64)

            if noise_is_latent:
                noise += latent_image.cpu()
                noise.sub_(noise.mean()).div_(noise.std())

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
            disable_pbar = False

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, 
                                                 noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, 
                                                 seed=noise_seed)

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
                
            return (out, out_denoised)


    