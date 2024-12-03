from .noise_classes import *
from .sigmas import get_sigmas
from .rk_sampler import sample_rk

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
            latent_out = latent_image.copy()
            samples = latent_out["samples"].clone()

            torch.manual_seed(noise_seed)

            if not add_noise:
                noise = torch.zeros(samples.size(), dtype=samples.dtype, layout=samples.layout, device="cpu")
            elif latent_noise is None:
                batch_inds = latent_out["batch_index"] if "batch_index" in latent_out else None
                noise = prepare_noise(samples, noise_seed, noise_type, batch_inds, alpha, k)
            else:
                noise = latent_noise["samples"]

            if normalize == "true":
                latent_mean = samples.mean()
                latent_std = samples.std()
                noise = noise * latent_std + latent_mean

            if noise_is_latent:
                noise += samples.cpu()
                noise.sub_(noise.mean()).div_(noise.std())
            
            noise = noise * noise_strength

            if mask is not None:
                mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                                    size=(samples.shape[2], samples.shape[3]), 
                                    mode="bilinear")
                mask = mask.expand((-1, samples.shape[1], -1, -1)).to(samples.device)
                if mask.shape[0] < samples.shape[0]:
                    mask = mask.repeat((samples.shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:samples.shape[0]]
                elif mask.shape[0] > samples.shape[0]:
                    mask = mask[:samples.shape[0]]
                
                noise = mask * noise + (1 - mask) * torch.zeros_like(noise)

            latent_out["samples"] = samples.cpu() + noise

            return (latent_out,)


#SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["beta57"]

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

    CATEGORY = "sampling/custom_sampling"
    
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
            

RK_SAMPLER_NAMES = ["res_2m",
                    "res_3m",
                    "res_2s", 
                    "res_3s",
                    "res_5s",


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


IRK_SAMPLER_NAMES = ["none",
                    "gauss-legendre_2s",
                    "gauss-legendre_3s", 
                    "gauss-legendre_4s",
                    "gauss-legendre_5s",
                    
                    "radau_ia_2s",
                    "radau_ia_3s",
                    "radau_iia_2s",
                    "radau_iia_3s",
                    
                    "lobatto_iiia_2s",
                    "lobatto_iiia_3s",
                    "lobatto_iiib_2s",
                    "lobatto_iiib_3s",
                    "lobatto_iiic_2s",
                    "lobatto_iiic_3s",
                    "lobatto_iiic_star_2s",
                    "lobatto_iiic_star_3s",
                    "lobatto_iiid_2s",
                    "lobatto_iiid_3s",
                    
                    "kraaijevanger_spijker_2s",
                    "qin_zhang_2s",
                    
                    "pareschi_russo_2s",
                    "pareschi_russo_alt_2s",
                    
                    "crouzeix_2s",
                    "crouzeix_3s",
                    
                    "irk_exp_diag_2s",
                    "use_explicit", 
                    ]


class StyleModelApplyAdvanced: 
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "main"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Use with Flux Redux."

    def main(self, clip_vision_output, style_model, conditioning, strength=1.0):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        cond = strength * cond
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )


NOISE_MODE_NAMES = ["none",
                    "hard_sq",
                    "hard",
                    "lorentzian", 
                    "soft", 
                    "softer",
                    "exp", 
                    "hard_var", 
                    ]


class ClownsharKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    #"noise_mode_sde": (["lorentzian", "hard", "hard_var", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "noise_mode_sde": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    #"noise_mode_sde": (["linear", "linear_mild", "linear_strong", "decay", "rapid_decay", "dynamic", "dynamic_mild"], {"default": 'linear', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
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
                    "shift": ("FLOAT", {"default": 1.35, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    "base_shift": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 100.0, "step":0.1, "round": False, }),
                    "shift_scaling": (["exponential", "linear"], {"default": "exponential"}),
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
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_init=-1.0, k_init=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, multistep=False, noise_seed=-1, sampler_name="res_2m", implicit_sampler_name="default",
                    exp_mode=False, t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, latent_guide_weight_inv=0.0, guide_mode="blend", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None, unsampler_type="linear",
                    shift=3.0, base_shift=0.85, guides=None, options=None, sde_noise=None,sde_noise_steps=1, shift_scaling="exponential",
                    extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, 
                    EXPORT_SAMPLER=False,
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
                unsampler_type = options.get('unsampler_type', unsampler_type)
                
                sde_noise = options.get('sde_noise', sde_noise)
                sde_noise_steps = options.get('sde_noise_steps', sde_noise_steps)
                
            rescale_floor = extra_options_flag("rescale_floor", extra_options)
            if guides is not None:
                guide_mode, latent_guide_weight, latent_guide_weight_inv, latent_guide_weights, latent_guide_weights_inv, latent_guide, latent_guide_inv, latent_guide_mask, latent_guide_mask_inv, scheduler_, scheduler_inv_, steps_, steps_inv_, denoise_, denoise_inv_ = guides
                if scheduler_ != "constant" and latent_guide_weights is None:
                    latent_guide_weights = get_sigmas(model, scheduler_, steps_, denoise_).to(default_dtype)
                if scheduler_inv_ != "constant" and latent_guide_weights_inv is None:
                    latent_guide_weights_inv = get_sigmas(model, scheduler_inv_, steps_inv_, denoise_inv_).to(default_dtype)
                    
            latent_guide_weights = initialize_or_scale(latent_guide_weights, latent_guide_weight, max_steps).to(default_dtype)
            latent_guide_weights = F.pad(latent_guide_weights, (0, max_steps), value=0.0)
            latent_guide_weights_inv = initialize_or_scale(latent_guide_weights_inv, latent_guide_weight_inv, max_steps).to(default_dtype)
            latent_guide_weights_inv = F.pad(latent_guide_weights_inv, (0, max_steps), value=0.0)
            
            if automation is not None:
                etas, s_noises, unsample_resample_scales = automation
            etas = initialize_or_scale(etas, eta, max_steps).to(default_dtype)
            etas = F.pad(etas, (0, max_steps), value=0.0)
            s_noises = initialize_or_scale(s_noises, s_noise, max_steps).to(default_dtype)
            s_noises = F.pad(s_noises, (0, max_steps), value=0.0)
            
            if shift >= 0:
                if isinstance(model.model.model_config, comfy.supported_models.SD3):
                    model = ModelSamplingSD3().patch(model, shift)[0] 
                    model = SD35L_TimestepPatcher().main(model, shift_scaling, shift)[0]
                    model.object_patches['model_sampling'].sigmas = model.model.model_sampling.sigmas
                elif isinstance(model.model.model_config, comfy.supported_models.AuraFlow):
                    model = ModelSamplingAuraFlow().patch_aura(model, shift)[0] 
                    model = SD35L_TimestepPatcher().main(model, shift_scaling, shift)[0]
                    model.object_patches['model_sampling'].sigmas = model.model.model_sampling.sigmas
                elif isinstance(model.model.model_config, comfy.supported_models.Stable_Cascade_C):
                    model = ModelSamplingStableCascade().patch(model, shift)[0] 
            if shift >= 0 and base_shift >= 0:
                if isinstance(model.model.model_config, comfy.supported_models.Flux) or isinstance(model.model.model_config, comfy.supported_models.FluxSchnell):
                    model = ModelSamplingFlux().patch(model, shift, base_shift, latent_image['samples'].shape[3], latent_image['samples'].shape[2])[0] 
                    model = SD35L_TimestepPatcher().main(model, shift_scaling, shift)[0]
                    model.object_patches['model_sampling'].sigmas = model.model.model_sampling.sigmas

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

            truncate_conditioning = extra_options_flag("truncate_conditionting", extra_options)
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

                sampler = comfy.samplers.ksampler("rk", {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha_sde, "k": k_sde, "c1": c1, "c2": c2, "c3": c3, "cfgpp": cfgpp, "MULTISTEP": multistep, 
                                                        "noise_sampler_type": noise_type_sde, "noise_mode": noise_mode_sde, "noise_seed": noise_seed_sde, "rk_type": sampler_name, "implicit_sampler_name": implicit_sampler_name,
                                                                "exp_mode": exp_mode, "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                                "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, "mask_inv": latent_guide_mask_inv,
                                                                "latent_guide_weights": latent_guide_weights, "latent_guide_weights_inv": latent_guide_weights_inv, "guide_mode": guide_mode, "unsampler_type": unsampler_type,
                                                                "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override, "sde_noise": sde_noise,
                                                                "extra_options": extra_options,
                                                                "etas": etas, "s_noises": s_noises, "unsample_resample_scales": unsample_resample_scales,
                                                                })

                if EXPORT_SAMPLER:
                    return (sampler, )

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
                
                seed += 1
                torch.manual_seed(seed)
                noise_seed_sde += 1
                if total_steps_iter > 1: 
                    sde_noise.append(out["samples_fp64"])

            return ( out, out_denoised, sde_noise,)



class ClownSampler(ClownsharKSampler):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde": (["lorentzian", "hard", "hard_var", "hard_sq", "soft", "softer", "exp",], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_name": (RK_SAMPLER_NAMES, {"default": "res_2m"}), 
                    "implicit_sampler_name": (IRK_SAMPLER_NAMES, {"default": "gauss-legendre_2s"}), 
                    "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                     },
                "optional": 
                    {
                    "guides": ("GUIDES", ),     
                    "options": ("OPTIONS", ),   
                    "automation": ("AUTOMATION", ),
                    }
                }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",) 

    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling"
    
    def main(self, model, cfg=1, denoise=1.0, denoise_alt=1.0, sampler_mode="standard",scheduler=None, steps=0, 
             noise_type_init="gaussian", noise_type_sde="brownian", noise_mode_sde="hard", latent_image=None, 
             positive=None, negative=None, sigmas=None, latent_noise=None, latent_noise_match=None,
             noise_stdev=1.0, noise_mean=0.0, noise_normalize=True, noise_is_latent=False, 
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_init=-1.0, k_init=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, multistep=False, noise_seed=-1, sampler_name="res_2m", implicit_sampler_name="default",
             exp_mode=False, t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
             latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, latent_guide_weight_inv=0.0, guide_mode="blend", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None, unsampler_type="linear",
             shift=-1, base_shift=-1, guides=None, options=None, sde_noise=None,sde_noise_steps=1, shift_scaling="exponential",
             extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, 
             EXPORT_SAMPLER=True, 
             ): 
        return super().main(model, cfg, sampler_mode, scheduler, steps, denoise, denoise_alt,
                            noise_type_init, noise_type_sde, noise_mode_sde, latent_image,
                            positive, negative, sigmas, latent_noise, latent_noise_match,
                            noise_stdev, noise_mean, noise_normalize, noise_is_latent,
                            eta, eta_var, d_noise, s_noise, alpha_init, k_init, alpha_sde, k_sde, cfgpp, c1, c2, c3, multistep, noise_seed, sampler_name, implicit_sampler_name,
                            exp_mode, t_fn_formula, sigma_fn_formula, implicit_steps,
                            latent_guide, latent_guide_inv, latent_guide_weight, latent_guide_weight_inv, guide_mode, latent_guide_weights, latent_guide_weights_inv, latent_guide_mask, latent_guide_mask_inv, rescale_floor, sigmas_override, unsampler_type,
                            shift, base_shift, guides, options, sde_noise, sde_noise_steps, shift_scaling,
                            extra_options, automation, etas, s_noises, unsample_resample_scales, EXPORT_SAMPLER)




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


def create_sigma_function(multiplier):
    def new_sigma(self, timestep):
        return timestep * multiplier + self.shift
    return new_sigma


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
        timesteps = 1000 #this was 10000

        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)
        if scaling == "exponential": 
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear": 
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        model.model.model_sampling.shift = self.shift
        model.model.model_sampling.multiplier = self.multiplier
        model.model.model_sampling.register_buffer('sigmas', ts)
        
        return (model,)




GUIDE_MODE_NAMES = ["unsample", 
                    "resample", 
                    "epsilon",
                    "epsilon_dynamic_mean",
                    "epsilon_dynamic_mean_std", 
                    "epsilon_dynamic_mean_from_bkg", 
                    "epsilon_guide_mean_std_from_bkg",
                    "hard_light", 
                    "blend", 
                    "mean_std", 
                    "mean", 
                    "std", 
                    "data",
]


class ClownsharKSamplerGuides:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "guide_weight_bkg": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "guide_weight_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Another way to control guide_weight strength. It works like the denoise control for sigmas schedules. Can be used together with guide_weight."}),
                     "guide_weight_bkg_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Another way to control guide_weight strength. It works like the denoise control for sigmas schedules. Can be used together with guide_weight_bkg."}),
                    "guide_weight_scheduler": (["constant"] + comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "guide_weight_scheduler_bkg": (["constant"] + comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "guide_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    "guide_bkg_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "guide": ("LATENT", ),
                        "guide_bkg": ("LATENT", ),
                        "guide_mask": ("MASK", ),
                        "guide_mask_bkg": ("MASK", ),
                        "guide_weights": ("SIGMAS", ),
                        "guide_weights_bkg": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
        
        if guide_weight_scheduler == "constant": 
            guide_weights = initialize_or_scale(None, guide_weight, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant": 
            guide_weights_bkg = initialize_or_scale(None, guide_weight_bkg, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
            
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, guide_mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        return (guides, )




class ClownsharKSamplerAutomation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    },
                    "optional": 
                    {
                        "etas": ("SIGMAS", ),
                        "s_noises": ("SIGMAS", ),
                        "unsample_resample_scales": ("SIGMAS", ),

                    }  
               }
    RETURN_TYPES = ("AUTOMATION",)
    RETURN_NAMES = ("automation",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, etas=None, s_noises=None, unsample_resample_scales=None,
                    ):

        automation = (etas, s_noises, unsample_resample_scales)
        return (automation, )



class ClownsharKSamplerOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_init_stdev": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                "noise_init_mean": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                "noise_type_init": (NOISE_GENERATOR_NAMES, {"default": "gaussian"}),
                "noise_type_sde": (NOISE_GENERATOR_NAMES, {"default": "brownian"}),
                "noise_mode_sde": (["hard", "hard_var", "hard_sq", "soft", "softer", "exp"], {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                "eta": ("FLOAT", {"default": 0.25, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "round": False}),
                "d_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                "alpha_init": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "k_init": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 2}),      
                "alpha_sde": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "k_sde": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 2}),      
                "noise_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for the SDE noise that is added after each step if eta or eta_var are non-zero. If set to -1, it will use the increment the seed most recently used by the workflow."}),
                "c1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "c2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "c3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "t_fn_formula": ("STRING", {"default": "", "multiline": True}),
                "sigma_fn_formula": ("STRING", {"default": "", "multiline": True}),   
                #"unsampler_type": (['linear', 'exponential', 'constant'],),
            },
            "optional": {
                "options": ("OPTIONS",),
            }
        }
    
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, noise_init_stdev, noise_init_mean, c1, c2, c3, eta, s_noise, d_noise, noise_type_init, noise_type_sde, noise_mode_sde, noise_seed,
                    alpha_init, k_init, alpha_sde, k_sde, t_fn_formula=None, sigma_fn_formula=None, unsampler_type="linear",
                    alphas=None, etas=None, s_noises=None, d_noises=None, c2s=None, c3s=None,
                    options=None,
                    ):
    
        if options is None:
            options = {}

        options['noise_init_stdev'] = noise_init_stdev
        options['noise_init_mean'] = noise_init_mean
        options['noise_type_init'] = noise_type_init
        options['noise_type_sde'] = noise_type_sde
        options['noise_mode_sde'] = noise_mode_sde
        options['eta'] = eta
        options['s_noise'] = s_noise
        options['d_noise'] = d_noise
        options['alpha_init'] = alpha_init
        options['k_init'] = k_init
        options['alpha_sde'] = alpha_sde
        options['k_sde'] = k_sde
        options['noise_seed_sde'] = noise_seed
        options['c1'] = c1
        options['c2'] = c2
        options['c3'] = c3
        options['t_fn_formula'] = t_fn_formula
        options['sigma_fn_formula'] = sigma_fn_formula
        options['unsampler_type'] = unsampler_type
        
        return (options,)
    


class ClownsharKSamplerOptions_SDE_Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sde_noise_steps": ("INT", {"default": 1, "min": 1, "max": 10000}),
            },
            "optional": {
                "sde_noise": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, sde_noise_steps, sde_noise, options=None,):
    
        if options is None:
            options = {}

        options['sde_noise_steps'] = sde_noise_steps
        options['sde_noise'] = sde_noise
        
        return (options,)



    
class TextBox1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "text1": ("STRING", {"default": "", "multiline": True}),
                    },
                     "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text1",)
    FUNCTION = "main"
    
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = "Multiline textbox."

    def main(self, text1):

        return (text1,)

class TextBox3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "text1": ("STRING", {"default": "", "multiline": True}),
                     "text2": ("STRING", {"default": "", "multiline": True}),
                     "text3": ("STRING", {"default": "", "multiline": True}),
                    },
                     "optional": 
                    {
                    }  
               }
    RETURN_TYPES = ("STRING", "STRING","STRING",)
    RETURN_NAMES = ("text1", "text2", "text3",)
    FUNCTION = "main"
    
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = "Multiline textbox."

    def main(self, text1, text2, text3 ):

        return (text1, text2, text3, )




class CLIPTextEncodeFluxUnguided:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l, t5xxl, guidance):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]], )
