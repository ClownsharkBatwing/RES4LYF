from .extra_samplers import prepare_noise
from .noise_classes import *

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers

import latent_preview
import torch
import torch.nn.functional as F

#from sys import settrace 
import sys
import os

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

class ClownGuides:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "offset": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "guide_1": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "guide_2": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "guide_mode_1": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "guide_mode_2": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "latent_self_guide_1": ("BOOLEAN", {"default": False}),
                "latent_shift_guide_1": ("BOOLEAN", {"default": False}),
                "guide_1_Luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "guide_1_CyanRed": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "guide_1_LimePurple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "guide_1_PatternStruct": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),             
            },
            "optional": {
                "offsets": ("SIGMAS", ),
                "guides_1": ("SIGMAS", ),
                "guides_2": ("SIGMAS", ),
                "latent_guide_1": ("LATENT", ),
                "latent_guide_2": ("LATENT", ),              
            }
        }
    
    RETURN_TYPES = ("GUIDES",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, offset, guide_1, guide_2, guide_mode_1, guide_mode_2, 
                    guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct, 
                    offsets=None, guides_1=None, guides_2=None, latent_guide_1=None, latent_guide_2=None, latent_self_guide_1=False, latent_shift_guide_1=False, ):

        """steps = 10000
        guides_1 = initialize_or_scale(guides_1, guide_1, steps)
        guides_2 = initialize_or_scale(guides_2, guide_2, steps)

        if latent_guide_1 is not None:
            latent_guide_1 = latent_guide_1["samples"]

        if latent_guide_2 is not None:
            latent_guide_2 = latent_guide_2["samples"]

        guide_1_channels = torch.tensor([guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct])

        latent_noise_samples = latent_noise["samples"] if latent_noise and "samples" in latent_noise else None"""
        
        return ( (offset, guide_1, guide_2, guide_mode_1, guide_mode_2, 
                    guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct, 
                    offsets, guides_1, guides_2, latent_guide_1, latent_guide_2, latent_self_guide_1, latent_shift_guide_1,) , )


class ClownSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eulers_mom": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "momentum": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "eta1": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "eta2": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "eta_var1": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "eta_var2": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "s_noise1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "s_noise2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "c2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "auto_c2": ("BOOLEAN", {"default": True}),
                "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "branch_mode": (['latent_match', 'latent_match_d', 'latent_match_sdxl_color_d', 'latent_match_sdxl_luminosity_d','latent_match_sdxl_pattern_d','cos_reversal', 'mean', 'mean_d', 'median', 'median_d', 'zmean_d','zmedian_d','gradient_max_full', 'gradient_max_full_d', 'gradient_min_full', 'gradient_min_full_d', 'gradient_max', 'gradient_max_d', 'gradient_min', 'gradient_min_d', 'cos_similarity', 'cos_similarity_d','cos_linearity', 'cos_linearity_d', 'cos_perpendicular', 'cos_perpendicular_d'], {"default": 'mean'}),
                "branch_depth": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "branch_width": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),

                "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "brownian"}),
                "noise_mode": (["hard", "soft", "softer"], {"default": 'hard'}), 
                "noise_scale": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                "ancestral_noise": ("BOOLEAN", {"default": True}),   
                #"noisy_cfg": ("BOOLEAN", {"default": False}),
                "clownseed": ("INT", {"default": -1.0, "min": -10000.0, "max": 0xffffffffffffffff}),
                "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 2}),      
                "denoise_to_zero": ("BOOLEAN", {"default": False}),
                "simple_phi_calc": ("BOOLEAN", {"default": False}),    
                #"skip_corrector": ("BOOLEAN", {"default": False}), 
                "order": ("INT", {"default": 2, "min": 1, "max": 2}),
                "t_fn_formula": ("STRING", {"default": "sigma.log().neg()", "multiline": True}),
                "sigma_fn_formula": ("STRING", {"default": "t.neg().exp()", "multiline": True}),   
            },
            "optional": {
                "eulers_moms": ("SIGMAS", ),
                "momentums": ("SIGMAS", ),
                "etas1": ("SIGMAS", ),
                "etas2": ("SIGMAS", ),
                "eta_vars1": ("SIGMAS", ),
                "eta_vars2": ("SIGMAS", ),
                "s_noises1": ("SIGMAS", ),
                "s_noises2": ("SIGMAS", ),
                "c2s": ("SIGMAS", ),
                "cfgpps": ("SIGMAS", ),
                "alphas": ("SIGMAS", ),
                "latent_noise": ("LATENT", ),  
                "guides": ("GUIDES", ),
                "alpha_ratios": ("SIGMAS", ),
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, clownseed, noise_sampler_type, noise_mode, noise_scale, ancestral_noise, denoise_to_zero, simple_phi_calc, cfgpp, eulers_mom, momentum, c2, eta1, eta2, eta_var1, eta_var2, s_noise1, s_noise2, branch_mode, branch_depth, branch_width,
                    alpha, k, noisy_cfg=False, 
                    alphas=None, latent_noise=None,
                    eulers_moms=None, momentums=None, etas1=None, etas2=None, eta_vars1=None, eta_vars2=None, s_noises1=None, s_noises2=None, c2s=None, cfgpps=None, offsets=None, guides=None, alpha_ratios=None, t_fn_formula=None, sigma_fn_formula=None,skip_corrector=False,
                    corrector_is_predictor=False, order=1, auto_c2=False,
                    ):
        
        if guides is not None:
            (offset, guide_1, guide_2, guide_mode_1, guide_mode_2, 
            guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct, 
            offsets, guides_1, guides_2, latent_guide_1, latent_guide_2, latent_self_guide_1, latent_shift_guide_1) = guides
        else:
            offset, guide_1, guide_2, guide_mode_1, guide_mode_2, guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct = 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0
            offsets, guides_1, guides_2, latent_guide_1, latent_guide_2, latent_self_guide_1, latent_shift_guide_1 = None, None, None, None, None, None, None

        steps = 10000
        eulers_moms = initialize_or_scale(eulers_moms, eulers_mom, steps)
        momentums = initialize_or_scale(momentums, momentum, steps)
        etas1 = initialize_or_scale(etas1, eta1, steps)
        etas2 = initialize_or_scale(etas2, eta2, steps)
        eta_vars1 = initialize_or_scale(eta_vars1, eta_var1, steps)
        eta_vars2 = initialize_or_scale(eta_vars2, eta_var2, steps)

        s_noises1 = initialize_or_scale(s_noises1, s_noise1, steps)
        s_noises2 = initialize_or_scale(s_noises2, s_noise2, steps)
        c2s = initialize_or_scale(c2s, c2, steps)
        cfgpps = initialize_or_scale(cfgpps, cfgpp, steps)
        offsets = initialize_or_scale(offsets, offset, steps)
        guides_1 = initialize_or_scale(guides_1, guide_1, steps)
        guides_2 = initialize_or_scale(guides_2, guide_2, steps)
        alphas = initialize_or_scale(alphas, alpha, steps)

        #import pdb; pdb.set_trace()

        if latent_guide_1 is not None:
            latent_guide_1 = latent_guide_1["samples"]

        if latent_guide_2 is not None:
            latent_guide_2 = latent_guide_2["samples"]

        guide_1_channels = torch.tensor([guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct])

        latent_noise_samples = latent_noise["samples"] if latent_noise and "samples" in latent_noise else None
        
        sampler = comfy.samplers.ksampler(
            "res_momentumized_advanced",
            {
                "noise_sampler_type": noise_sampler_type,
                "noise_mode": noise_mode,
                "noise_scale": noise_scale, 
                "ancestral_noise": ancestral_noise,
                "noisy_cfg": noisy_cfg,
                "denoise_to_zero": denoise_to_zero,
                "simple_phi_calc": simple_phi_calc,
                "branch_mode": branch_mode,
                "branch_depth": branch_depth,
                "branch_width": branch_width,
                "eulers_moms": eulers_moms,
                "momentums": momentums,
                "etas1": etas1,
                "etas2": etas2,
                "eta_vars1": eta_vars1,
                "eta_vars2": eta_vars2,
                "s_noises1": s_noises1,
                "s_noises2": s_noises2,
                "c2s": c2s,
                "cfgpps": cfgpps,
                "offsets": offsets,
                "guides_1": guides_1,
                "guides_2": guides_2,
                "latent_guide_1": latent_guide_1,
                "latent_guide_2": latent_guide_2,
                "guide_mode_1": guide_mode_1,
                "guide_mode_2": guide_mode_2,
                "guide_1_channels": guide_1_channels,
                "alphas": alphas,
                "k": k,
                "clownseed": clownseed+1,
                "latent_noise": latent_noise_samples,
                "latent_self_guide_1": latent_self_guide_1,
                "latent_shift_guide_1": latent_shift_guide_1,
                "alpha_ratios": alpha_ratios,
                "t_fn_formula": t_fn_formula,
                "sigma_fn_formula": sigma_fn_formula,
                "skip_corrector": skip_corrector,
                "corrector_is_predictor": corrector_is_predictor,
                "order": order,
                "auto_c2": auto_c2,
            }
        )
        return (sampler, )

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
            
            # Apply noise_strength
            noise = noise * noise_strength

            # Apply mask if provided
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

            # Combine noise with latent image
            noised_latent = latent_image.cpu() + noise

            return ({'samples': noised_latent},)



class SharkSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
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
                     },
                "optional": 
                    {"latent_noise": ("LATENT", ),
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output", "latent_batch")

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, model, add_noise, noise_is_latent, noise_type, noise_seed, cfg, alpha, k, positive, negative, sampler, sigmas, latent_image, latent_noise=None): 
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

            if noise_is_latent: #add noise and latent together and normalize --> noise
                noise += latent_image.cpu()
                noise.sub_(noise.mean()).div_(noise.std())

            noise_mask = latent["noise_mask"] if "noise_mask" in latent else None

            x0_output = {}

            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

            disable_pbar = False

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, 
                                                 noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
            return (out, out_denoised)


class UltraSharkSampler:
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


class SamplerDPMPP_SDE_ADVANCED:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta1": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta2": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "denoise_boost": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "brownian"}),
                     "noise_mode": (["hard", "hard_var", "soft", "softer"], {"default": 'hard'}), 
                     "noisy_cfg": ("BOOLEAN", {"default": False}),
                     "noise_scale": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "t_fn_formula": ("STRING", {"default": "1/((sigma).exp()+1)", "multiline": True}),
                     "sigma_fn_formula": ("STRING", {"default": "((1-t)/t).log()", "multiline": True}),
                      },
                    "optional": 
                    {
                        "momentums": ("SIGMAS", ),
                        "etas1": ("SIGMAS", ),
                        "etas2": ("SIGMAS", ),
                        "s_noises1": ("SIGMAS", ),
                        "s_noises2": ("SIGMAS", ),
                        "denoise_boosts": ("SIGMAS", ),
                        "rs": ("SIGMAS", ),
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, momentum=0.0, eta1=1.0, eta2=1.0, s_noise1=1.0, s_noise2=1.0, denoise_boost=0.0, r=0.5, alpha=-1.0, k=1.0, noise_sampler_type="brownian", noise_mode="soft", noise_scale=1.0, noisy_cfg=False,
                    momentums=None, etas1=None, etas2=None, s_noises1=None, s_noises2=None, denoise_boosts=None, rs=None, alphas=None, t_fn_formula=None, sigma_fn_formula=None,
                    ):
        sampler_name = "dpmpp_sde_advanced"

        steps = 10000
        momentums = initialize_or_scale(momentums, momentum, steps)
        etas1 = initialize_or_scale(etas1, eta1, steps)
        etas2 = initialize_or_scale(etas2, eta2, steps)
        s_noises1 = initialize_or_scale(s_noises1, s_noise1, steps)
        s_noises2 = initialize_or_scale(s_noises2, s_noise2, steps)
        denoise_boosts = initialize_or_scale(denoise_boosts, denoise_boost, steps)
        rs = initialize_or_scale(rs, r, steps)
        alphas = initialize_or_scale(alphas, alpha, steps)

        sampler = comfy.samplers.ksampler(sampler_name, {"momentums": momentums, "etas1": etas1, "etas2": etas2, "s_noises1": s_noises1, "s_noises2": s_noises2, "noisy_cfg": noisy_cfg,
                                                         "denoise_boosts": denoise_boosts, "alphas": alphas, "rs": rs, "k": k, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "noise_scale": noise_scale,
                                                         "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula,})
        return (sampler, )
    
class SamplerDEIS_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"momentum": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, {"default": "brownian"}),
                     "noise_mode": (["hard", "hard_var", "soft", "softer"], {"default": "hard"}), 
                     "noise_scale": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "deis_mode": (["rhoab", "tab"], {"default": "rhoab"}), 
                     "step_type": (["simple", "res_a", "dpmpp_sde"], {"default": "simple"}), 
                     "denoised_type": (["1_2", "2"], {"default": "1_2"}), 
                     "max_order": ("INT", {"default": 3, "min": 1, "max": 4, "step":1}),
                      },
                    "optional": 
                    {
                        "momentums": ("SIGMAS", ),
                        "etas": ("SIGMAS", ),
                        "s_noises": ("SIGMAS", ),
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, momentum, eta, s_noise, alpha, k, noise_sampler_type, noise_mode, noise_scale, deis_mode, step_type, denoised_type, max_order, momentums=None, etas=None, s_noises=None, alphas=None, ):
        sampler_name = "deis_sde"

        steps = 10000
        momentums = initialize_or_scale(momentums, momentum, steps)
        etas = initialize_or_scale(etas, eta, steps)
        s_noises = initialize_or_scale(s_noises, s_noise, steps)
        alphas = initialize_or_scale(alphas, alpha, steps)

        sampler = comfy.samplers.ksampler(sampler_name, {"momentums": momentums, "etas": etas, "s_noises": s_noises, "alpha": alphas, "k": k, 
                                                         "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode, "noise_scale": noise_scale, "deis_mode": deis_mode, "step_type": step_type, 
                                                         "denoised_type": denoised_type, "max_order": max_order,})
        return (sampler, )

 
class SamplerDPMPP_SDE_CFGPP_ADVANCED:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "eulers_mom": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False}),
                     "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      },
                    "optional": 
                    {
                        "eulers_moms": ("SIGMAS", ),
                        "cfgpps": ("SIGMAS", ),
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, r, eulers_mom, cfgpp, alpha, k, noise_sampler_type, eulers_moms=None, cfgpps=None, alphas=None):
        sampler_name = "dpmpp_sde_cfgpp_advanced"

        steps = 10000
        eulers_moms = initialize_or_scale(eulers_moms, eulers_mom, steps)
        alphas = initialize_or_scale(alphas, alpha, steps)
        cfgpps = initialize_or_scale(cfgpps, cfgpp, steps)

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "r": r, "eulers_mom": eulers_moms, "cfgpp": cfgpps, "alpha": alphas, "k": k, "noise_sampler_type": noise_sampler_type})
        return (sampler, )

class SamplerEulerAncestral_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                     "noise_mode": (["hard", "soft"], {"default": 'soft'}), 
                      },
                    "optional": 
                    {
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, alpha, k, noise_sampler_type, noise_mode, alphas=None):
        
        steps = 10000
        alphas = initialize_or_scale(alphas, alpha, steps)

        sampler = comfy.samplers.ksampler("euler_ancestral_advanced", {"eta": eta, "s_noise": s_noise, "alpha": alphas, "k": k, "noise_sampler_type": noise_sampler_type, "noise_mode": noise_mode,})
        return (sampler, )
    
    
class SamplerDPMPP_2S_Ancestral_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, noise_sampler_type):
        sampler = comfy.samplers.ksampler("dpmpp_2s_ancestral_advanced", {"eta": eta, "s_noise": s_noise, "noise_sampler_type": noise_sampler_type})
        return (sampler, )
    
class SamplerDPMPP_2M_SDE_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, noise_sampler_type):
        sampler = comfy.samplers.ksampler("dpmpp_2m_sde_advanced", {"eta": eta, "s_noise": s_noise, "noise_sampler_type": noise_sampler_type})
        return (sampler, )
    

class SamplerDPMPP_3M_SDE_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"momentum": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      },
                    "optional": 
                    {
                        "momentums": ("SIGMAS", ),
                        "etas": ("SIGMAS", ),
                        "s_noises": ("SIGMAS", ),
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, momentum, eta, s_noise, alpha, k, noise_sampler_type, momentums=None, etas=None, s_noises=None, alphas=None, ):
        sampler_name = "dpmpp_3m_sde_advanced"

        steps = 10000
        momentums = initialize_or_scale(momentums, momentum, steps)
        etas = initialize_or_scale(etas, eta, steps)
        s_noises = initialize_or_scale(s_noises, s_noise, steps)
        alphas = initialize_or_scale(alphas, alpha, steps)

        sampler = comfy.samplers.ksampler(sampler_name, {"momentums": momentums, "etas": etas, "s_noises": s_noises, "s_noise": s_noise, "alpha": alphas, "k": k, "noise_sampler_type": noise_sampler_type, })
        return (sampler, )


class SamplerDPMPP_DUALSDE_MOMENTUMIZED_ADVANCED:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                "momentum": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step":0.01}),
                "eta": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                "s_noise": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step":0.01}),
                "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01}),
            },
            "optional": 
            {
                "momentums": ("SIGMAS", ),
                "etas": ("SIGMAS", ),
                "s_noises": ("SIGMAS", ),
                "rs": ("SIGMAS", ),
            }
        }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, noise_sampler_type, momentum, eta, s_noise, r, momentums=None, etas=None, s_noises=None, rs=None):
            scheduled_r=False

            steps = 10000
            momentums = initialize_or_scale(momentums, momentum, steps)
            etas = initialize_or_scale(etas, eta, steps)
            s_noises = initialize_or_scale(s_noises, s_noise, steps)

            if rs is None:
                rs = torch.full((steps,), r)
            else:
                rs = r * rs
                scheduled_r = True

            sampler = comfy.samplers.ksampler(
                "dpmpp_dualsde_momentumized_advanced",
                {
                    "noise_sampler_type": noise_sampler_type, 
                    "momentums": momentums,
                    "etas": etas,
                    "s_noises": s_noises,
                    "rs": rs,
                    "scheduled_r": scheduled_r
                }
            )
            return (sampler, )



