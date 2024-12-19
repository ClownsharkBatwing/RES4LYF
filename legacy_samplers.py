from .noise_classes import *
from .sigmas import get_sigmas

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

from .legacy_sampler_rk import legacy_sample_rk

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor
    
def move_to_same_device(*tensors):
    if not tensors:
        return tensors

    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)



RK_SAMPLER_NAMES = ["res_2m",
                    "res_3m",
                    "res_2s", 
                    "res_3s",
                    "rk_exp_5s",

                    "deis_2m",
                    "deis_3m", 
                    "deis_4m",
                    
                    "ralston_2s",
                    "ralston_3s",
                    "ralston_4s", 
                    
                    "dpmpp_2m",
                    "dpmpp_3m",
                    "dpmpp_2s",
                    "dpmpp_sde_2s",
                    "dpmpp_3s",
                    
                    "midpoint_2s",
                    "heun_2s", 
                    "heun_3s", 
                    
                    "houwen-wray_3s",
                    "kutta_3s", 
                    "ssprk3_3s",
                    
                    "rk38_4s",
                    "rk4_4s", 

                    "dormand-prince_6s", 
                    "dormand-prince_13s", 
                    "bogacki-shampine_7s",

                    "ddim",
                    "euler",
                    ]


IRK_SAMPLER_NAMES = [
                    "gauss-legendre_2s",
                    "gauss-legendre_3s", 
                    "gauss-legendre_4s",
                    "gauss-legendre_5s",
                    
                    "radau_iia_2s",
                    "radau_iia_3s",
                    
                    "lobatto_iiic_2s",
                    "lobatto_iiic_3s",
                    
                    "crouzeix_2s",
                    "crouzeix_3s",
                    
                    "irk_exp_diag_2s",

                    "use_explicit", 
                    ]


class Legacy_ClownsharKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    #"add_noise": ("BOOLEAN", {"default": True}),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "brownian"}),
                    "noise_mode_sde": (["hard", "hard_var", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                    #"sampler_mode": (['standard', 'unsample', 'resample'],),
                    "sampler_mode": (['standard', 'unsample', 'resample',],),
                    "sampler_name": (RK_SAMPLER_NAMES, {"default": "res_2m"}), 
                    "implicit_sampler_name": (["default", 
                                               "gauss-legendre_5s",
                                               "gauss-legendre_4s",
                                               "gauss-legendre_3s", 
                                               "gauss-legendre_2s",
                                               "crouzeix_2s",
                                               "radau_iia_3s",
                                               "radau_iia_2s",
                                               "lobatto_iiic_3s",
                                               "lobatto_iiic_2s",
                                               ], {"default": "default"}), 
                    "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "denoise_alt": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step":0.1, "round": False, }),
                    "shift": ("FLOAT", {"default": 3.0, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    "base_shift": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    "truncate_conditioning": (['false', 'true'], {"default": "true"}),
                     },
                "optional": 
                    {
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),     
                    "guides": ("GUIDES", ),     
                    "options": ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT", ) #"LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised",) # "output_fp64", "denoised_fp64")

    FUNCTION = "main"

    CATEGORY = "RES4LYF/legacy/samplers"
    
    def main(self, model, cfg, truncate_conditioning, sampler_mode, scheduler, steps, denoise=1.0, denoise_alt=1.0,
             noise_type_init="gaussian", noise_type_sde="brownian", noise_mode_sde="hard", latent_image=None, 
             positive=None, negative=None, sigmas=None, latent_noise=None, latent_noise_match=None,
             noise_stdev=1.0, noise_mean=0.0, noise_normalize=True, noise_is_latent=False, 
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_init=-1.0, k_init=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, multistep=False, noise_seed=-1, sampler_name="res_2m", implicit_sampler_name="default",
                    exp_mode=False, t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, guide_mode="blend", latent_guide_weights=None, latent_guide_mask=None, rescale_floor=True, sigmas_override=None, unsampler_type="linear",
                    shift=3.0, base_shift=0.85, guides=None, options=None,
                    ): 
            default_dtype = torch.float64
            max_steps = 10000


            if noise_seed == -1:
                seed = torch.initial_seed() + 1
            else:
                seed = noise_seed
                torch.manual_seed(noise_seed)
            noise_seed_sde = seed + 1

            if options is not None:
                noise_stdev = options.get('noise_init_stdev', noise_stdev)
                noise_mean = options.get('noise_init_mean', noise_mean)
                noise_type_init = options.get('noise_type_init', noise_type_init)
                noise_type_sde = options.get('noise_type_sde', noise_type_sde)
                noise_mode_sde = options.get('noise_mode_sde', noise_mode_sde)
                eta = options.get('eta', eta)
                s_noise = options.get('s_noise', s_noise)
                d_noise = options.get('d_noise', d_noise)
                alpha_init = options.get('alpha_init', alpha_init)
                k_init = options.get('k_init', k_init)
                alpha_sde = options.get('alpha_sde', alpha_sde)
                k_sde = options.get('k_sde', k_sde)
                noise_seed_sde = options.get('noise_seed_sde', noise_seed+1)
                c1 = options.get('c1', c1)
                c2 = options.get('c2', c2)
                c3 = options.get('c3', c3)
                t_fn_formula = options.get('t_fn_formula', t_fn_formula)
                sigma_fn_formula = options.get('sigma_fn_formula', sigma_fn_formula)
                #unsampler_type = options.get('unsampler_type', unsampler_type)

            if guides is not None:
                guide_mode, rescale_floor, latent_guide_weight, latent_guide_weights, t_is, latent_guide, latent_guide_inv, latent_guide_mask, scheduler_, steps_, denoise_ = guides
                """if scheduler == "constant": 
                    latent_guide_weights = initialize_or_scale(latent_guide_weights, latent_guide_weight, max_steps).to(default_dtype)
                    latent_guide_weights = F.pad(latent_guide_weights, (0, max_steps), value=0.0)"""
                if scheduler_ != "constant":
                    latent_guide_weights = get_sigmas(model, scheduler_, steps_, denoise_).to(default_dtype)
            latent_guide_weights = initialize_or_scale(latent_guide_weights, latent_guide_weight, max_steps).to(default_dtype)
            latent_guide_weights = F.pad(latent_guide_weights, (0, max_steps), value=0.0)
            
            if shift >= 0:
                if isinstance(model.model.model_config, comfy.supported_models.SD3):
                    model = ModelSamplingSD3().patch(model, shift)[0] 
                elif isinstance(model.model.model_config, comfy.supported_models.AuraFlow):
                    model = ModelSamplingAuraFlow().patch_aura(model, shift)[0] 
                elif isinstance(model.model.model_config, comfy.supported_models.Stable_Cascade_C):
                    model = ModelSamplingStableCascade().patch(model, shift)[0] 
            if shift >= 0 and base_shift >= 0:
                if isinstance(model.model.model_config, comfy.supported_models.Flux) or isinstance(model.model.model_config, comfy.supported_models.FluxSchnell):
                    model = ModelSamplingFlux().patch(model, shift, base_shift, latent_image['samples'].shape[3], latent_image['samples'].shape[2])[0] 

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
            
            sigmin = model.model.model_sampling.sigma_min
            sigmax = model.model.model_sampling.sigma_max

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

            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

            disable_pbar = False
            
            if noise_type_sde == "none":
                eta_var = eta = 0.0
                noise_type_sde = "gaussian"
            if noise_mode_sde == "hard_var":
                eta_var = eta
                eta = 0.0
            
            if cfg < 0:
                cfgpp = -cfg
                cfg = 1.0
                
            sampler = comfy.samplers.ksampler("legacy_rk", {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha_sde, "k": k_sde, "c1": c1, "c2": c2, "c3": c3, "cfgpp": cfgpp, "MULTISTEP": multistep, 
                                                     "noise_sampler_type": noise_type_sde, "noise_mode": noise_mode_sde, "noise_seed": noise_seed_sde, "rk_type": sampler_name, "implicit_sampler_name": implicit_sampler_name,
                                                            "exp_mode": exp_mode, "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                            "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, 
                                                            "latent_guide_weights": latent_guide_weights, "guide_mode": guide_mode, #"unsampler_type": unsampler_type,
                                                            "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override})

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, x.clone(), 
                                                 noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

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

            return ( out, out_denoised, )





class Legacy_SamplerRK:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                     "eta_var": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculate variance-corrected noise amount (overrides eta/noise_mode settings). Cannot be used at very low sigma values; reverts to eta/noise_mode for final steps."}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),
                     "d_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Ratio of calculated noise amount actually added after each step. >1.0 will leave extra noise behind, <1.0 will remove more noise than it adds."}),

                     "noise_mode": (["hard", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "brownian"}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False, "tooltip": "Fractal noise mode: <0 = extra high frequency noise, >0 = extra low frequency noise, 0 = white noise."}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False, "tooltip": "Fractal noise mode: all that matters is positive vs. negative. Effect unclear."}),
                     "noise_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for the SDE noise that is added after each step if eta or eta_var are non-zero. If set to -1, it will use the increment the seed most recently used by the workflow."}),
                     "rk_type": (RK_SAMPLER_NAMES, {"default": "res_2m"}), 
                     "exp_mode": ("BOOLEAN", {"default": False, "tooltip": "Convert linear RK methods to exponential form."}), 
                     "multistep": ("BOOLEAN", {"default": False, "tooltip": "For samplers ending in S only. Reduces cost by one model call per step by reusing the previous step as the current predictor step."}),
                     "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit Runge-Kutta refinement steps to run after each explicit step."}),
                     "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "CFG++ scale. Use in place of, or with, CFG. Currently only working with RES, DPMPP, and DDIM samplers."}),
                     "latent_guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     #"guide_mode": (["hard_light", "mean_std", "mean", "std", "noise_mean", "blend", "inversion"], {"default": 'mean', "tooltip": "The mode used. noise_mean and inversion are currently for test purposes only."}),
                     "guide_mode": (["hard_light", "mean_std", "mean", "std", "blend",], {"default": 'mean', "tooltip": "The mode used. noise_mean and inversion are currently for test purposes only."}),
                     #"guide_mode": (["hard_light", "blend", "mean_std", "mean", "std"], {"default": 'mean', "tooltip": "The mode used."}),
                     "rescale_floor": ("BOOLEAN", {"default": True, "tooltip": "Latent_guide_weight(s) control the minimum value for the latent_guide_mask. If false, they control the maximum value."}),
                    },
                    "optional":
                    {
                        "latent_guide": ("LATENT", ),
                        "latent_guide_inv": ("LATENT", ),

                        "latent_guide_mask": ("MASK", ),
                        "latent_guide_weights": ("SIGMAS", ),
                        "sigmas_override": ("SIGMAS", ),
                    }  
                    
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "RES4LYF/legacy/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha=-1.0, k=1.0, cfgpp=0.0, multistep=False, noise_sampler_type="brownian", noise_mode="hard", noise_seed=-1, rk_type="dormand-prince", 
                    exp_mode=False, t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, guide_mode="hard_light", latent_guide_weights=None, latent_guide_mask=None, rescale_floor=True, sigmas_override=None,
                    ):
        sampler_name = "legacy_rk"

        if latent_guide is None and latent_guide_inv is None:
            latent_guide_weight = 0.0

        steps = 10000
        latent_guide_weights = initialize_or_scale(latent_guide_weights, latent_guide_weight, steps)
            
        latent_guide_weights = F.pad(latent_guide_weights, (0, 10000), value=0.0)

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha, "k": k, "cfgpp": cfgpp, "MULTISTEP": multistep, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "noise_seed": noise_seed, "rk_type": rk_type, 
                                                         "exp_mode": exp_mode, "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                         "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, "latent_guide_weight": latent_guide_weight, "latent_guide_weights": latent_guide_weights, "guide_mode": guide_mode,
                                                         "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override})
        return (sampler, )




class Legacy_ClownsharKSamplerGuides:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (["hard_light", "mean_std", "mean", "std", "blend"], {"default": 'blend', "tooltip": "The mode used."}),
                     "latent_guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "scheduler": (["constant"] + comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "rescale_floor": ("BOOLEAN", {"default": False, "tooltip": "If true, latent_guide_weight(s) primarily affect the masked areas. If false, they control the unmasked areas."}),
                    },
                    "optional": 
                    {
                        "latent_guide": ("LATENT", ),
                        "latent_guide_inv": ("LATENT", ),
                        "latent_guide_mask": ("MASK", ),
                        "latent_guide_weights": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    CATEGORY = "RES4LYF/legacy/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, model=None, scheduler="constant", steps=30, denoise=1.0, latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, guide_mode="blend", latent_guide_weights=None, latent_guide_mask=None, rescale_floor=True, t_is=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        #if scheduler != "constant": 
        #    latent_guide_weights = get_sigmas(model, scheduler, steps, latent_guide_weight).to(default_dtype)
            
        if scheduler == "constant": 
            latent_guide_weights = initialize_or_scale(None, latent_guide_weight, steps).to(default_dtype)
            latent_guide_weights = F.pad(latent_guide_weights, (0, max_steps), value=0.0)
        
        if latent_guide is not None:
            x = latent_guide["samples"].clone().to(default_dtype) 
        if latent_guide_inv is not None:
            x = latent_guide_inv["samples"].clone().to(default_dtype) 

        guides = (guide_mode, rescale_floor, latent_guide_weight, latent_guide_weights, t_is, latent_guide, latent_guide_inv, latent_guide_mask, scheduler, steps, denoise)
        return (guides, )





class Legacy_SharkSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_normalize": ("BOOLEAN", {"default": True}),
                    "noise_stdev": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                    "noise_mean": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                    "noise_is_latent": ("BOOLEAN", {"default": False}),
                    "noise_type": (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                    "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False, }),
                    "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False, }),
                    "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_mode": (['standard', 'unsample', 'resample'],),
                    #"scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                    "scheduler": (comfy.samplers.SCHEDULER_NAMES + ["beta57"],),

                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000, "step":0.01}),
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.5, "round": False, }),
                    "truncate_conditioning": (['false', 'true', 'true_and_zero_neg'], ),
                    
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "latent_image": ("LATENT", ),               
                     },
                "optional": 
                    {
                    "sigmas": ("SIGMAS", ),
                    "latent_noise": ("LATENT", ),
                    "latent_noise_match": ("LATENT",),
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT","LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised", "output_fp64", "denoised_fp64")

    FUNCTION = "main"

    CATEGORY = "RES4LYF/legacy/samplers"
        
    def main(self, model, add_noise, noise_stdev, noise_mean, noise_normalize, noise_is_latent, noise_type, noise_seed, cfg, truncate_conditioning, alpha, k, positive, negative, sampler,
             latent_image, sampler_mode, scheduler, steps, denoise, sigmas=None, latent_noise=None, latent_noise_match=None,): 

            latent = latent_image
            latent_image_dtype = latent_image['samples'].dtype
            
            default_dtype = torch.float64
            
            if positive is None:
                positive = [[
                    torch.zeros((1, 154, 4096)),  # blah[0][0], a tensor of shape (1, 154, 4096)
                    {'pooled_output': torch.zeros((1, 2048))}
                    ]]
            if negative is None:
                negative = [[
                    torch.zeros((1, 154, 4096)),  # blah[0][0], a tensor of shape (1, 154, 4096)
                    {'pooled_output': torch.zeros((1, 2048))}
                    ]]
                
            if denoise < 0:
                sampler.extra_options['d_noise'] = -denoise
                denoise = 1.0
            if sigmas is not None:
                sigmas = sigmas.clone().to(default_dtype)
            else: 
                sigmas = get_sigmas(model, scheduler, steps, denoise).to(default_dtype)
                
            #sigmas = sigmas.clone().to(torch.float64)
        
            if sampler_mode == "unsample": 
                null = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.flip(sigmas, dims=[0])
                sigmas = torch.cat([sigmas, null])
            elif sampler_mode == "resample":
                null = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.cat([null, sigmas])
                sigmas = torch.cat([sigmas, null])
                
            if latent_image is not None:
                x = latent_image["samples"].clone().to(default_dtype) 
                #x = {"samples": x}
                
            if latent_noise is not None:
                latent_noise["samples"] = latent_noise["samples"].clone().to(default_dtype)  
            if latent_noise_match is not None:
                latent_noise_match["samples"] = latent_noise_match["samples"].clone().to(default_dtype)

            if truncate_conditioning == "true" or truncate_conditioning == "true_and_zero_neg":
                if positive is not None:
                    positive[0][0] = positive[0][0].clone().to(default_dtype)
                    positive[0][1]["pooled_output"] = positive[0][1]["pooled_output"].clone().to(default_dtype)
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
                    if negative is not None:
                        negative[0][0] = negative[0][0].clone().to(default_dtype)
                        negative[0][1]["pooled_output"] = negative[0][1]["pooled_output"].clone().to(default_dtype)
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

            if noise_seed == -1:
                seed = torch.initial_seed() + 1
            else:
                seed = noise_seed
                torch.manual_seed(noise_seed)
            
            noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_type)(x=x, seed=seed, sigma_min=sigmin, sigma_max=sigmax)
            
            if noise_type == "fractal":
                noise_sampler.alpha = alpha
                noise_sampler.k = k
                noise_sampler.scale = 0.1
        
            if not add_noise:
                noise = torch.zeros_like(x)
            elif latent_noise is None:
                noise = noise_sampler(sigma=sigmax, sigma_next=sigmin)
            else:
                noise = latent_noise["samples"]

            if noise_is_latent: #add noise and latent together and normalize --> noise
                noise += x.cpu()
                noise.sub_(noise.mean()).div_(noise.std())

            if noise_normalize:
                noise.sub_(noise.mean()).div_(noise.std())
            noise *= noise_stdev
            noise = (noise - noise.mean()) + noise_mean
            
            if latent_noise_match:
                for i in range(latent_noise_match["samples"].shape[1]):
                    noise[0][i] = (noise[0][i] - noise[0][i].mean())
                    noise[0][i] = (noise[0][i]) + latent_noise_match["samples"][0][i].mean()

                
            noise_mask = latent["noise_mask"] if "noise_mask" in latent else None

            x0_output = {}

            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

            disable_pbar = False
 
            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, x, 
                                                 noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
                
            out_orig_dtype = out['samples'].clone().to(latent_image_dtype)
            out_denoised_orig_dtype = out_denoised['samples'].clone().to(latent_image_dtype)
                
            return ( {'samples': out_orig_dtype}, {'samples': out_denoised_orig_dtype}, out, out_denoised,)
            