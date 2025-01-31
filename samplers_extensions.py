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

import torch
import torch.nn.functional as F

import math
import copy

from .helper import initialize_or_scale, get_res4lyf_scheduler_list, get_extra_options_kv, extra_options_flag


def move_to_same_device(*tensors):
    if not tensors:
        return tensors

    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)


    
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
    
    CATEGORY = "RES4LYF/sampler_extensions"
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
    
    CATEGORY = "RES4LYF/sampler_extensions"
    DESCRIPTION = "Patches ClownSampler's to use garbage collection after every step. This can help with OOM issues during inference for large models like Flux. The tradeoff is slower sampling."

    def set_sampler_extra_options(self, sampler, garbage_collection):
        sampler = copy.deepcopy(sampler)
        sampler.extra_options['GARBAGE_COLLECT'] = garbage_collection
        return (sampler, )


GUIDE_MODE_NAMES_BETA = ["unsample", 
                    "resample", 
                    "unsample_projection", 
                    "resample_projection", 
                    "unsample_projection_cw", 
                    "resample_projection_cw", 
                    "pseudoimplicit",
                    "pseudoimplicit_projection",
                    "fully_pseudoimplicit",
                    "fully_pseudoimplicit_projection",
                    "epsilon",
                    "epsilon_projection",
                    "epsilon_projection_cw",
                    "epsilon_dynamic_mean",
                    "epsilon_dynamic_mean_std", 
                    "epsilon_dynamic_mean_from_bkg", 
                    "epsilon_guide_mean_std_from_bkg",
                    "hard_light", 
                    "blend", 
                    "blend_projection",
                    "mean_std", 
                    "mean", 
                    "mean_tiled",
                    "std", 
                    "data",
                    #"data_projection",
                    "none",
]


GUIDE_MODE_NAMES = ["unsample", 
                    "resample", 
                    "epsilon",
                    "epsilon_projection",
                    "epsilon_dynamic_mean",
                    "epsilon_dynamic_mean_std", 
                    "epsilon_dynamic_mean_from_bkg", 
                    "epsilon_guide_mean_std_from_bkg",
                    "hard_light", 
                    "blend", 
                    "blend_projection",
                    "mean_std", 
                    "mean", 
                    "mean_tiled",
                    "std", 
                    "data",
                    #"data_projection",
                    "none",
]

from .conditioning import FluxRegionalPrompt, FluxRegionalConditioning
from .models import ReFluxPatcher

class ClownInpaint: ##################################################################################################################################
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"guide_mode": (GUIDE_MODE_NAMES, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight":               ("FLOAT", {"default": 0.10, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "guide_weight_bkg":           ("FLOAT", {"default": 1.00, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "guide_weight_scheduler_bkg": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "guide_end_step":              ("INT", {"default": 15, "min": 1, "max": 10000}),
                    "guide_bkg_end_step":          ("INT", {"default": 10000, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "model":             ("MODEL", ),
                        "positive_inpaint":  ("CONDITIONING", ),
                        "positive_bkg":      ("CONDITIONING", ),
                        "negative":          ("CONDITIONING", ),
                        "latent_image":      ("LATENT", ),
                        "mask":              ("MASK", ),
                        "guide_weights":     ("SIGMAS", ),
                        "guide_weights_bkg": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT","GUIDES",)
    RETURN_NAMES = ("model","positive"    ,"negative"    ,"latent","guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=10000, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=1.0, guide_weight_bkg=1.0, 
                    guide_mode="epsilon", guide_weights=None, guide_weights_bkg=None, guide_mask_bkg=None,
                    model=None, positive_inpaint=None, positive_bkg=None, negative=None, latent_image=None, mask=None, 
                    ):
        default_dtype = torch.float64
        guide = latent_image
        guide_bkg = {'samples': latent_image['samples'].clone()}
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
        
        if guide_weight_scheduler == "constant": 
            guide_weights = initialize_or_scale(None, guide_weight, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant": 
            guide_weights_bkg = initialize_or_scale(None, guide_weight_bkg, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
            
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        
        latent = {'samples': torch.zeros_like(latent_image['samples'])}
        if (positive_inpaint is None) and (positive_bkg is None):
            positive = None
        else:
            if positive_bkg is None:
                if positive_bkg is None:
                    positive_bkg = [[
                        torch.zeros((1, 256, 4096)),
                        {'pooled_output': torch.zeros((1, 768))}
                        ]]
            cond_regional, mask_inv     = FluxRegionalPrompt().main(cond=positive_inpaint,                              mask=mask)
            cond_regional, mask_inv_inv = FluxRegionalPrompt().main(cond=positive_bkg    , cond_regional=cond_regional, mask=mask_inv)
            
            positive, = FluxRegionalConditioning().main(conditioning_regional=cond_regional, self_attn_floor=0.0)
            
        model, = ReFluxPatcher().main(model, enable=True)
        
        return (model, positive, negative, latent, guides, )
    
    
class ClownInpaintSimple: ##################################################################################################################################
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {#"guide_mode": (GUIDE_MODE_NAMES, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight":               ("FLOAT", {"default": 0.10, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "guide_end_step":              ("INT", {"default": 15, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "model":             ("MODEL", ),
                        "positive_inpaint":  ("CONDITIONING", ),
                        "negative":          ("CONDITIONING", ),
                        "latent_image":      ("LATENT", ),
                        "mask":              ("MASK", ),
                    }
               }
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT","GUIDES",)
    RETURN_NAMES = ("model","positive"    ,"negative"    ,"latent","guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=10000, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=1.0, guide_weight_bkg=1.0, 
                    guide_mode="epsilon", guide_weights=None, guide_weights_bkg=None, guide_mask_bkg=None,
                    model=None, positive_inpaint=None, positive_bkg=None, negative=None, latent_image=None, mask=None, 
                    ):
        default_dtype = torch.float64
        guide = latent_image
        guide_bkg = {'samples': latent_image['samples'].clone()}
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
        
        if guide_weight_scheduler == "constant": 
            guide_weights = initialize_or_scale(None, guide_weight, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant": 
            guide_weights_bkg = initialize_or_scale(None, guide_weight_bkg, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
            
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        
        latent = {'samples': torch.zeros_like(latent_image['samples'])}
        if (positive_inpaint is None) and (positive_bkg is None):
            positive = None
        else:
            if positive_bkg is None:
                if positive_bkg is None:
                    positive_bkg = [[
                        torch.zeros((1, 256, 4096)),
                        {'pooled_output': torch.zeros((1, 768))}
                        ]]
            cond_regional, mask_inv     = FluxRegionalPrompt().main(cond=positive_inpaint,                              mask=mask)
            cond_regional, mask_inv_inv = FluxRegionalPrompt().main(cond=positive_bkg    , cond_regional=cond_regional, mask=mask_inv)
            
            positive, = FluxRegionalConditioning().main(conditioning_regional=cond_regional, self_attn_floor=1.0)
            
        model, = ReFluxPatcher().main(model, enable=True)
        
        return (model, positive, negative, latent, guides, )
    

##################################################################################################################################


class ClownsharKSamplerGuide:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES, {"default": 'epsilon_projection', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     #"guide_weight_bkg": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "guide_weight_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     #"guide_weight_bkg_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    #"guide_weight_scheduler_bkg": (["constant"] + comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "guide_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    #"guide_bkg_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "guide": ("LATENT", ),
                        #"guide_bkg": ("LATENT", ),
                        "guide_mask": ("MASK", ),
                        #"guide_mask_bkg": ("MASK", ),
                        "guide_weights": ("SIGMAS", ),
                        #"guide_weights_bkg": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
      
        if guide_weight_scheduler == "constant" and guide_weights == None: 
            guide_weights = initialize_or_scale(None, 1.0, guide_end_step).to(default_dtype)
            #guide_weights = initialize_or_scale(None, guide_weight, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant": 
            guide_weights_bkg = initialize_or_scale(None, 0.0, guide_bkg_end_step).to(default_dtype)
            #guide_weights_bkg = initialize_or_scale(None, guide_weight_bkg, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
            
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, guide_mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        return (guides, )




class ClownsharKSamplerGuides:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES, {"default": 'epsilon_projection', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "guide_weight_bkg": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "guide_weight_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     "guide_weight_bkg_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "guide_weight_scheduler_bkg": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
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
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
        
        if guide_weight_scheduler == "constant" and guide_weights == None: 
            guide_weights = initialize_or_scale(None, 1.0, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant" and guide_weights_bkg == None: 
            guide_weights_bkg = initialize_or_scale(None, 1.0, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
    
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, guide_mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        return (guides, )










class ClownsharKSamplerGuide_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA, {"default": 'epsilon_projection', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     #"guide_weight_bkg": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "guide_weight_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     #"guide_weight_bkg_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    #"guide_weight_scheduler_bkg": (["constant"] + comfy.samplers.SCHEDULER_NAMES + ["beta57"], {"default": "beta57"},),
                    "guide_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    #"guide_bkg_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "guide": ("LATENT", ),
                        #"guide_bkg": ("LATENT", ),
                        "guide_mask": ("MASK", ),
                        #"guide_mask_bkg": ("MASK", ),
                        "guide_weights": ("SIGMAS", ),
                        #"guide_weights_bkg": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
      
        if guide_weight_scheduler == "constant" and guide_weights == None: 
            guide_weights = initialize_or_scale(None, 1.0, guide_end_step).to(default_dtype)
            #guide_weights = initialize_or_scale(None, guide_weight, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant": 
            guide_weights_bkg = initialize_or_scale(None, 0.0, guide_bkg_end_step).to(default_dtype)
            #guide_weights_bkg = initialize_or_scale(None, guide_weight_bkg, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
            
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, guide_mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        return (guides, )




class ClownsharKSamplerGuides_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA, {"default": 'epsilon_projection', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "guide_weight_bkg": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "guide_weight_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     "guide_weight_bkg_scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "guide_weight_scheduler_bkg": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
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
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_scale=1.0, guide_weight_bkg_scale=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_scale, guide_weight_bkg_scale
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
        
        if guide_weight_scheduler == "constant" and guide_weights == None: 
            guide_weights = initialize_or_scale(None, 1.0, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant" and guide_weights_bkg == None: 
            guide_weights_bkg = initialize_or_scale(None, 1.0, guide_bkg_end_step).to(default_dtype)
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
    CATEGORY = "RES4LYF/sampler_extensions"
    
    FUNCTION = "main"

    def main(self, etas=None, s_noises=None, unsample_resample_scales=None,):
        automation = (etas, s_noises, unsample_resample_scales)
        return (automation, )



class ClownsharKSamplerAutomation_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    },
                    "optional": 
                    {
                        "etas": ("SIGMAS", ),
                        "etas_substep": ("SIGMAS", ),
                        "s_noises": ("SIGMAS", ),
                        "s_noises_substep": ("SIGMAS", ),
                        "unsample_resample_scales": ("SIGMAS", ),

                    }  
               }
    RETURN_TYPES = ("AUTOMATION",)
    RETURN_NAMES = ("automation",)
    CATEGORY = "RES4LYF/sampler_extensions"
    
    FUNCTION = "main"

    def main(self, etas=None, etas_substep=None, s_noises=None, s_noises_substep=None, unsample_resample_scales=None,):
        automation = (etas, etas_substep, s_noises, s_noises_substep, unsample_resample_scales)
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
                "c1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "c2": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "c3": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
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
    CATEGORY = "RES4LYF/sampler_extensions"

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
                "options": ("OPTIONS",),
            }
        }
    
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, sde_noise_steps, sde_noise, options=None,):
    
        if options is None:
            options = {}

        options['sde_noise_steps'] = sde_noise_steps
        options['sde_noise'] = sde_noise
        
        return (options,)



class ClownsharKSamplerOptions_FrameWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_weights": ("SIGMAS", ),
            },
            "optional": {
                "options": ("OPTIONS",),
            }
        }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, frame_weights, options=None,):

        if options is None:
            options = {}

        options['frame_weights'] = frame_weights

        return (options,)


