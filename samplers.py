from .noise_classes import *

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd

import latent_preview
import torch
import torch.nn.functional as F

import math
import copy

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

class AdvancedNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": 0.01}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": 0.01}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "noise_type": (NOISE_GENERATOR_NAMES, ),
            },
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self, noise_seed, noise_type, alpha, k):
        return (Noise_RandomNoise(noise_seed, noise_type, alpha, k),)

class Noise_RandomNoise:
    def __init__(self, seed, noise_type, alpha, k):
        self.seed = seed
        self.noise_type = noise_type
        self.alpha = alpha
        self.k = k

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return prepare_noise(latent_image, self.seed, self.noise_type, batch_inds, self.alpha, self.k)


class LatentNoised:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_is_latent": ("BOOLEAN", {"default": False}),
                    "noise_type": (NOISE_GENERATOR_NAMES, ),
                    "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": 0.01}),
                    "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": 0.01}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "latent_image": ("LATENT", ),
                    "noise_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                    "normalize": (["false", "true"], {"default": "false"}),
                     },
                "optional": 
                    {
                    "latent_noise": ("LATENT", ),
                    "mask": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_noised",)

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, add_noise, noise_is_latent, noise_type, noise_seed, alpha, k, latent_image, noise_strength, normalize, latent_noise=None, mask=None):
            latent = latent_image
            latent_image = latent["samples"]

            torch.manual_seed(noise_seed)

            if not add_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
            elif latent_noise is None:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds, alpha, k)
            else:
                noise = latent_noise["samples"]

            if normalize == "true":
                latent_mean = latent_image.mean()
                latent_std = latent_image.std()
                noise = noise * latent_std + latent_mean

            if noise_is_latent:
                noise += latent_image.cpu()
                noise.sub_(noise.mean()).div_(noise.std())
            
            noise = noise * noise_strength

            if mask is not None:
                mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                                     size=(latent_image.shape[2], latent_image.shape[3]), 
                                     mode="bilinear")
                mask = mask.expand((-1, latent_image.shape[1], -1, -1)).to(latent_image.device)
                if mask.shape[0] < latent_image.shape[0]:
                    mask = mask.repeat((latent_image.shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:latent_image.shape[0]]
                elif mask.shape[0] > latent_image.shape[0]:
                    mask = mask[:latent_image.shape[0]]
                
                noise = mask * noise + (1 - mask) * torch.zeros_like(noise)

            noised_latent = latent_image.cpu() + noise

            return ({'samples': noised_latent},)

def get_sigmas(model, scheduler, steps, denoise): #adapted from comfyui
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = int(steps/denoise)

    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    sigmas = sigmas[-(steps + 1):]
    return sigmas

class SharkSampler:
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
                    "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000}),
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

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, model, add_noise, noise_stdev, noise_mean, noise_normalize, noise_is_latent, noise_type, noise_seed, cfg, truncate_conditioning, alpha, k, positive, negative, sampler, latent_image, sampler_mode, scheduler, steps, denoise, sigmas=None, latent_noise=None, latent_noise_match=None,): 
            latent = latent_image
            latent_image_dtype = latent_image['samples'].dtype
            
            default_dtype = torch.float64
            
            if positive is not None:
                positive[0][0] = positive[0][0].clone().to(default_dtype)
                positive[0][1]["pooled_output"] = positive[0][1]["pooled_output"].clone().to(default_dtype)
            
            if negative is not None:
                negative[0][0] = negative[0][0].clone().to(default_dtype)
                negative[0][1]["pooled_output"] = negative[0][1]["pooled_output"].clone().to(default_dtype)
                
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

            #sigmin = model.object_patches_backup['model_sampling'].sigma_min
            #sigmax = model.object_patches_backup['model_sampling'].sigma_max
            
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
                
            return ( {'samples', out_orig_dtype}, {'samples', out_denoised_orig_dtype}, out, out_denoised,)
            #return (out.to(latent_image_dtype), out_denoised.to(latent_image_dtype))


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
                guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)
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


class SamplerRK:
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
                     "rk_type": (["dormand-prince_6s", 
                                  "dormand-prince_7s", 
                                  "dormand-prince_13s", 
                                  "rk4_4s", 
                                  "rk38_4s",
                                  "ralston_4s",
                                  "dpmpp_3s",
                                  "heun_3s", 
                                  "houwen-wray_3s",
                                  "kutta_3s", 
                                  "ralston_3s",
                                  "res_3s",
                                  "ssprk3_3s",
                                  "dpmpp_2s",
                                  "dpmpp_sde_2s",
                                  "heun_2s", 
                                  "midpoint_2s",
                                  "ralston_2s",
                                  "res_2s", 
                                  "dpmpp_3m",
                                  "res_3m",
                                  "dpmpp_2m",
                                  "res_2m",
                                  "deis_2m",
                                  "deis_3m", 
                                  "deis_4m",
                                  "ddim",
                                  "euler"], {"default": "res_2m"}), 
                     "exp_mode": ("BOOLEAN", {"default": False, "tooltip": "Convert linear RK methods to exponential form."}), 
                     "multistep": ("BOOLEAN", {"default": False, "tooltip": "For samplers ending in S only. Reduces cost by one model call per step by reusing the previous step as the current predictor step."}),
                     "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 100, "step":1, "tooltip": "Number of implicit Runge-Kutta refinement steps to run after each explicit step."}),
                     "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "CFG++ scale. Use in place of, or with, CFG. Currently only working with RES, DPMPP, and DDIM samplers."}),
                     "latent_guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "guide_mode": (["hard_light", "mean_std", "mean", "std", "noise_mean", "blend", "inversion"], {"default": 'mean', "tooltip": "The mode used. noise_mean and inversion are currently for test purposes only."}),
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
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha=-1.0, k=1.0, cfgpp=0.0, multistep=False, noise_sampler_type="brownian", noise_mode="hard", noise_seed=-1, rk_type="dormand-prince", 
                    exp_mode=False, t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, guide_mode="hard_light", latent_guide_weights=None, latent_guide_mask=None, rescale_floor=True, sigmas_override=None,
                    ):
        sampler_name = "rk"

        steps = 10000
        latent_guide_weights = initialize_or_scale(latent_guide_weights, latent_guide_weight, steps)
            
        latent_guide_weights = F.pad(latent_guide_weights, (0, 10000), value=0.0)

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha, "k": k, "cfgpp": cfgpp, "MULTISTEP": multistep, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "noise_seed": noise_seed, "rk_type": rk_type, 
                                                         "exp_mode": exp_mode, "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                         "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, "latent_guide_weight": latent_guide_weight, "latent_guide_weights": latent_guide_weights, "guide_mode": guide_mode,
                                                         "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override})
        return (sampler, )

    sigma_fn = lambda t: (t.exp() + 1) ** -1 + 1e-5
    t_fn = lambda sigma: ((1-sigma)/sigma + 1e-5).log()
    
    
class SamplerOptions_TimestepScaling:
    # for patching the t_fn and sigma_fn (sigma <-> timestep) formulas to allow picking Runge-Kutta Ci values ("midpoints") with different scaling.
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "sampler": ("SAMPLER", ),
                     "t_fn_formula": ("STRING", {"default": "1/((sigma).exp()+1)", "multiline": True}),
                     "sigma_fn_formula": ("STRING", {"default": "((1-t)/t).log()", "multiline": True}),
                    },
                     "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "set_sampler_extra_options"
    
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = "Patches ClownSampler's t_fn and sigma_fn (sigma <-> timestep) formulas to allow picking Runge-Kutta Ci values (midpoints) with different scaling."

    def set_sampler_extra_options(self, sampler, t_fn_formula=None, sigma_fn_formula=None, ):

        sampler = copy.deepcopy(sampler)

        sampler.extra_options['t_fn_formula']     = t_fn_formula
        sampler.extra_options['sigma_fn_formula'] = sigma_fn_formula

        return (sampler, )


def time_snr_shift_exponential(alpha, t):
    return math.exp(alpha) / (math.exp(alpha) + (1 / t - 1) ** 1.0)

def time_snr_shift_linear(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

    
class SamplerOptions_GarbageCollection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "sampler": ("SAMPLER", ),
                     "garbage_collection": ("BOOLEAN", {"default": True}),
                    },
                     "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "set_sampler_extra_options"
    
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = "Patches ClownSampler's to use garbage collection after every step. This can help with OOM issues during inference for large models like Flux. The tradeoff is slower sampling."

    def set_sampler_extra_options(self, sampler, garbage_collection):
        sampler = copy.deepcopy(sampler)
        sampler.extra_options['GARBAGE_COLLECT'] = garbage_collection
        return (sampler, )


def time_snr_shift_exponential(alpha, t):
    return math.exp(alpha) / (math.exp(alpha) + (1 / t - 1) ** 1.0)

def time_snr_shift_linear(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

class SD35L_TimestepPatcher:
    # this is used to set the "shift" using either exponential scaling (default for SD3.5M and Flux) or linear scaling (default for SD3.5L and SD3 2B beta)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "scaling": (["exponential", "linear"], {"default": 'exponential'}), 
                    "shift": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    
                }
               }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "SD35L"

    def sigma_exponential(self, timestep):
        return time_snr_shift_exponential(self.shift, timestep / self.multiplier)

    def sigma_linear(self, timestep):
        return time_snr_shift_linear(self.shift, timestep / self.multiplier)

    def main(self, model, scaling, shift):
        self.shift = shift
        self.multiplier = 1000
        timesteps = 10000

        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)
        if scaling == "exponential": 
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear": 
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        model.model.model_sampling.shift = self.shift
        model.model.model_sampling.multiplier = self.multiplier
        model.model.model_sampling.register_buffer('sigmas', ts)
        
        return (model,)
