from .extra_samplers import prepare_noise
from .noise_classes import *

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers

import latent_preview
import torch

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

class ClownSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "momentum": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "ita": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "c2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "clownseed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "guide_1": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "guide_2": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "guide_mode_1": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "guide_mode_2": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                "denoise_to_zero": ("BOOLEAN", {"default": True}),
                "simple_phi_calc": ("BOOLEAN", {"default": False}),
                #"cfgpp": ("BOOLEAN", {"default": False}),
                "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),

                "latent_self_guide_1": ("BOOLEAN", {"default": False}),
                "latent_shift_guide_1": ("BOOLEAN", {"default": False}),
                "guide_1_Luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "guide_1_CyanRed": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "guide_1_LimePurple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "guide_1_PatternStruct": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),

                "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 2}),                
            },
            "optional": {
                "momentums": ("SIGMAS", ),
                "itas": ("SIGMAS", ),
                "c2s": ("SIGMAS", ),
                "cfgpps": ("SIGMAS", ),
                "offsets": ("SIGMAS", ),
                "guides_1": ("SIGMAS", ),
                "guides_2": ("SIGMAS", ),
                "alphas": ("SIGMAS", ),
                "latent_guide_1": ("LATENT", ),
                "latent_guide_2": ("LATENT", ),
                "latent_noise": ("LATENT", ),                
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, clownseed, noise_sampler_type, denoise_to_zero, simple_phi_calc, cfgpp, momentum, c2, ita, offset, 
                    guide_1, guide_2, guide_mode_1, guide_mode_2, 
                    guide_1_Luminosity, guide_1_CyanRed, guide_1_LimePurple, guide_1_PatternStruct, 
                    alpha, k,
                    alphas=None, latent_noise=None,
                    guides_1=None, guides_2=None, latent_guide_1=None, latent_guide_2=None, latent_self_guide_1=False, latent_shift_guide_1=False, 
                    momentums=None, itas=None, c2s=None, cfgpps=None, offsets=None):

        steps = 10000
        momentums = initialize_or_scale(momentums, momentum, steps)
        itas = initialize_or_scale(itas, ita, steps)
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
                "denoise_to_zero": denoise_to_zero,
                "simple_phi_calc": simple_phi_calc,
                #"cfgpp": cfgpp,
                #"cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "momentums": momentums,
                "itas": itas,
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
                #"alpha": alpha,
                "k": k,
                "clownseed": clownseed,
                "latent_noise": latent_noise_samples,
                "latent_self_guide_1": latent_self_guide_1,
                "latent_shift_guide_1": latent_shift_guide_1,
            }
        )
        return (sampler, )

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

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"
    
    @cast_fp64
    def sample(self, model, add_noise, noise_is_latent, noise_type, noise_seed, cfg, alpha, k, positive, negative, sampler, 
               sigmas, latent_image, latent_noise=None):
            latent = latent_image
            latent_image = latent["samples"].to(torch.float64)
            #import pdb; pdb.set_trace()

            torch.manual_seed(noise_seed)

            if not add_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
            elif latent_noise is None:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = prepare_noise(latent_image, noise_seed, noise_type, batch_inds, alpha, k)
            else:
                noise = latent_noise["samples"].to(torch.float64)

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
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      },
                    "optional": 
                    {
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, r, alpha, k, noise_device, noise_sampler_type, alphas=None):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_sde_advanced"
        else:
            sampler_name = "dpmpp_sde_gpu_advanced"

        steps = 10000
        alphas = initialize_or_scale(alphas, alpha, steps)

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "r": r, "alpha": alphas, "k": k, "noise_sampler_type": noise_sampler_type})
        return (sampler, )
    
class SamplerDPMPP_SDE_CFGPP_ADVANCED:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "cfgpp": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                     "alpha": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": False}),
                     "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": False}),
                     #"noise_device": (['gpu', 'cpu'], ),
                     "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      },
                    "optional": 
                    {
                        "cfgpps": ("SIGMAS", ),
                        "alphas": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, r, cfgpp, alpha, k, noise_sampler_type, cfgpps=None, alphas=None):
        sampler_name = "dpmpp_sde_cfgpp_advanced"
        #if noise_device == 'cpu':
        #    sampler_name = "dpmpp_sde_cfgpp_advanced"
        #else:
        #    sampler_name = "dpmpp_sde_gpu_advanced"

        steps = 10000
        alphas = initialize_or_scale(alphas, alpha, steps)
        cfgpps = initialize_or_scale(cfgpps, cfgpp, steps)

        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "r": r, "cfgpp": cfgpps, "alpha": alphas, "k": k, "noise_sampler_type": noise_sampler_type})
        return (sampler, )

class SamplerEulerAncestral_Advanced:
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
        sampler = comfy.samplers.ksampler("euler_ancestral_advanced", {"eta": eta, "s_noise": s_noise, "noise_sampler_type": noise_sampler_type})
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
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      "noise_sampler_type": (NOISE_GENERATOR_NAMES, ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, noise_sampler_type):
        sampler = comfy.samplers.ksampler("dpmpp_3m_sde_advanced", {"eta": eta, "s_noise": s_noise, "noise_sampler_type": noise_sampler_type})
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
