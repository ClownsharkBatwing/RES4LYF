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


from .rk_coefficients_beta import RK_SAMPLER_NAMES_BETA_FOLDERS
from .noise_sigmas_timesteps_scaling import NOISE_MODE_NAMES


class ClownSamplerSelector_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (RK_SAMPLER_NAMES_BETA_FOLDERS, {"default": "multistep/res_2m"}), 
                     },
                "optional": 
                    {
                    }
                }

    RETURN_TYPES = (RK_SAMPLER_NAMES_BETA_FOLDERS,)
    RETURN_NAMES = ("sampler_name",) 

    FUNCTION = "main"

    CATEGORY = "RES4LYF/sampler_options"
    
    def main(self,sampler_name="res_2m",):
        
        return (sampler_name,)




class ClownOptions_SDE_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "noise_type_sde": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde_substep": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "noise_mode_sde_substep": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "eta_substep": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                     },
                "optional": 
                    {
                    "options": ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)

    FUNCTION = "main"

    CATEGORY = "RES4LYF/sampler_options"
    
    def main(self, noise_type_sde="gaussian", noise_type_sde_substep="gaussian", noise_mode_sde="hard", noise_mode_sde_substep="hard", eta=0.5, eta_substep=0.5, options=None): 
        
        if options is None:
            options = {}
            
        options['noise_type_sde'] = noise_type_sde
        options['noise_type_sde_substep'] = noise_type_sde_substep
        options['noise_mode_sde'] = noise_mode_sde
        options['noise_mode_sde_substep'] = noise_mode_sde_substep
        options['eta'] = eta
        options['eta_substep'] = eta_substep

        return (options,)



class ClownOptions_StepSize_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "overshoot_mode":         (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How step size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot_mode_substep": (NOISE_MODE_NAMES, {"default": 'hard', "tooltip": "How substep size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot":         ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising step, then rescale to match the original. Has a softening effect."}),
                    "overshoot_substep": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising substep, then rescale to match the original. Has a softening effect."}),
                     },
                "optional": 
                    {
                    "options": ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)

    FUNCTION = "main"

    CATEGORY = "RES4LYF/sampler_options"
    
    def main(self, overshoot_mode="hard", overshoot_mode_substep="hard", overshoot=0.0, overshoot_substep=0.0, options=None): 
        
        if options is None:
            options = {}
            
        options['overshoot_mode'] = overshoot_mode
        options['overshoot_mode_substep'] = overshoot_mode_substep
        options['overshoot'] = overshoot
        options['overshoot_substep'] = overshoot_substep

        return (options,)





class ClownOptions_DetailBoost_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "noise_boost_step": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_boost_substep": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_anchor": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."}),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "s_noise_substep": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "d_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
                     },
                "optional": 
                    {
                    "options": ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)

    FUNCTION = "main"

    CATEGORY = "RES4LYF/sampler_options"
    
    def main(self, noise_boost_step=0.0, noise_boost_substep=0.0, noise_anchor=1.0, s_noise=1.0, s_noise_substep=1.0, d_noise=1.0, options=None): 
        
        if options is None:
            options = {}
            
        options['noise_boost_step'] = noise_boost_step
        options['noise_boost_substep'] = noise_boost_substep
        options['noise_anchor'] = noise_anchor
        options['s_noise'] = s_noise
        options['s_noise_substep'] = s_noise_substep
        options['d_noise'] = d_noise

        return (options,)





class ClownOptions_ImplicitSteps_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "implicit_substeps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     },
                "optional": 
                    {
                    "options": ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)

    FUNCTION = "main"

    CATEGORY = "RES4LYF/sampler_options"
    
    def main(self, implicit_steps=0, implicit_substeps=0, options=None): 
        
        if options is None:
            options = {}
            
        options['implicit_steps'] = implicit_steps
        options['implicit_substeps'] = implicit_substeps

        return (options,)



class ClownOptions_ExtraOptions_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
                    },
                    "optional": 
                    {
                    "options": ("OPTIONS", ),   
                    }  
               }
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "RES4LYF/sampler_options"
    
    FUNCTION = "main"

    def main(self, extra_options="",options=None):
                
        if options is None:
            options = {}
            
        options['extra_options'] = extra_options

        return (options, )




class ClownOptions_Automation_Beta:
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
                        "epsilon_scales": ("SIGMAS", ),
                        "options": ("OPTIONS", ),  
                    }  
               }
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "RES4LYF/sampler_options"
    
    FUNCTION = "main"

    def main(self, etas=None, etas_substep=None, s_noises=None, s_noises_substep=None, epsilon_scales=None,options=None):
                
        if options is None:
            options = {}
            
        automation = {
            "etas": etas,
            "etas_substep": etas_substep,
            "s_noises": s_noises,
            "s_noises_substep": s_noises_substep,
            "epsilon_scales": epsilon_scales,
        }
        
        options["automation"] = automation
        #options['automation'] = (etas, etas_substep, s_noises, s_noises_substep, epsilon_scales) # epsilon_scales was called unsample_resample_scales

        return (options, )





class SharkOptions_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_stdev": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                    "sampler_mode": (['standard', 'unsample', 'resample'],),
                    "denoise_alt": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "channelwise_cfg": ("BOOLEAN", {"default": False}),
                     },
                "optional": 
                    {
                    "sigmas": ("SIGMAS", ),
                    "options": ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)

    FUNCTION = "main"

    CATEGORY = "RES4LYF/sampler_options"
    
    def main(self, noise_type_init="gaussian", noise_stdev=1.0, sampler_mode="standard", denoise_alt=1.0, channelwise_cfg=False, options=None): 
        
        if options is None:
            options = {}
            
        options['noise_type_init'] = noise_type_init
        options['noise_stdev'] = noise_stdev
        options['sampler_mode'] = sampler_mode
        options['denoise_alt'] = denoise_alt
        options['channelwise_cfg'] = channelwise_cfg

        return (options,)









    
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


class ClownGuidesFluxAdvanced_Beta: ##################################################################################################################################
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"regional_model":             (["auto", "deactivate"], {"default": "auto"}),
                     "guide_mode":                  (GUIDE_MODE_NAMES_BETA_SIMPLE, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "channelwise_mode":           ("BOOLEAN", {"default": False}),
                     "projection_mode":            ("BOOLEAN", {"default": False}),
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
                        "guide":      ("LATENT", ),
                        "guide_bkg":      ("LATENT", ),
                        "mask":              ("MASK", ),
                        "mask_bkg":              ("MASK", ),
                        "guide_weights":     ("SIGMAS", ),
                        "guide_weights_bkg": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("MODEL","CONDITIONING","CONDITIONING","LATENT","GUIDES",)
    RETURN_NAMES = ("model","positive"    ,"negative"    ,"latent","guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, regional_model, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=10000, guide_bkg_end_step=30, guide_weight_cutoff=1.0, guide_weight_bkg_cutoff=1.0, guide=None, guide_bkg=None, guide_weight=1.0, guide_weight_bkg=1.0, 
                    guide_mode="epsilon", channelwise_mode=False, projection_mode=False, guide_weights=None, guide_weights_bkg=None, guide_mask_bkg=None,
                    model=None, positive_inpaint=None, positive_bkg=None, negative=None, latent_image=None, mask=None, mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_cutoff, guide_weight_bkg_cutoff
        
        if projection_mode:
            guide_mode = guide_mode + "_projection"
        
        if channelwise_mode:
            guide_mode = guide_mode + "_cw"
            
        if guide_mode == "unsample_cw":
            guide_mode = "unsample"
        if guide_mode == "resample_cw":
            guide_mode = "resample"
        
        if guide_weight_scheduler == "constant" and guide_weights == None: 
            guide_weights = initialize_or_scale(None, 1.0, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant" and guide_weights_bkg == None: 
            guide_weights_bkg = initialize_or_scale(None, 1.0, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
        
        h, w = guide["samples"].shape[-2], guide["samples"].shape[-1]
        
        if mask is None:
            mask = torch.ones((1, 8*h, 8*w), dtype=guide["samples"].dtype, device=guide["samples"].device)
        guide_mask =     1-mask       if mask           is not None else None
        guide_mask_bkg = 1-mask_bkg   if mask_bkg       is not None else None
        guide_mask_bkg = 1-guide_mask if guide_mask_bkg is None     else guide_mask_bkg
    
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, guide_mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        
        latent = {'samples': torch.zeros_like(guide['samples'])}
        if (positive_inpaint is None) and (positive_bkg is None):
            positive = None
            reflux_enable = False
        elif mask is not None:
            if regional_model == "auto":
                reflux_enable = True
            else:
                reflux_enable = False
            
            if positive_bkg is None:
                if positive_bkg is None:
                    positive_bkg = [[
                        torch.zeros((1, 256, 4096)),
                        {'pooled_output': torch.zeros((1, 768))}
                        ]]
            cond_regional, mask_inv     = FluxRegionalPrompt().main(cond=positive_inpaint,                              mask=mask)
            cond_regional, mask_inv_inv = FluxRegionalPrompt().main(cond=positive_bkg    , cond_regional=cond_regional, mask=mask_inv)
            
            positive, = FluxRegionalConditioning().main(conditioning_regional=cond_regional, self_attn_floor=0.0)
        else:
            positive = positive_inpaint
            reflux_enable = False

        model, = ReFluxPatcher().main(model, enable=reflux_enable)
        
        return (model, positive, negative, latent, guides, )
    

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





GUIDE_MODE_NAMES_BETA_MASTER_LIST = ["unsample", 
                    "resample", 
                    "unsample_projection", 
                    "resample_projection", 
                    "unsample_projection_cw", 
                    "resample_projection_cw", 
                    "pseudoimplicit",
                    "pseudoimplicit_cw",
                    "pseudoimplicit_projection",
                    "pseudoimplicit_projection_cw",
                    "fully_pseudoimplicit",
                    "fully_pseudoimplicit_cw",
                    "fully_pseudoimplicit_projection",
                    "fully_pseudoimplicit_projection_cw",
                    "epsilon",
                    "epsilon_cw",
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



GUIDE_MODE_NAMES_BETA_MISC = [
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




GUIDE_MODE_NAMES_BETA = ["unsample", 
                    "resample", 
                    "unsample_projection", 
                    "resample_projection", 
                    "epsilon",
                    "epsilon_projection",
                    "pseudoimplicit",
                    "pseudoimplicit_projection",
                    "fully_pseudoimplicit",
                    "fully_pseudoimplicit_projection",
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



GUIDE_MODE_NAMES_BETA_SIMPLE = [
                    "epsilon",
                    "pseudoimplicit",
                    "fully_pseudoimplicit",
                    "none",
]



GUIDE_MODE_NAMES_BETA_CHANNELWISE_SUPPORTED = [                    
                    "unsample", 
                    "resample_projection", 
                    "pseudoimplicit",
                    "pseudoimplicit_projection",
                    "fully_pseudoimplicit",
                    "fully_pseudoimplicit_projection",
                    "epsilon",
                    "epsilon_projection",
                    ]




class ClownsharKSamplerGuideMisc_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA_MISC, {"default": 'blend', "tooltip": "Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.05, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "guide_weight_cutoff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "guide_weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "guide_end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "guide": ("LATENT", ),
                        "guide_mask": ("MASK", ),
                        "guide_weights": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_cutoff=1.0, guide_weight_bkg_cutoff=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_cutoff, guide_weight_bkg_cutoff
        
        if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
            print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
            guide_mode = "epsilon"
        
        if guide_weight_scheduler == "constant" and guide_weights == None: 
            guide_weights = initialize_or_scale(None, 1.0, guide_end_step).to(default_dtype)
            guide_weights = F.pad(guide_weights, (0, max_steps), value=0.0)
        
        if guide_weight_scheduler_bkg == "constant": 
            guide_weights_bkg = initialize_or_scale(None, 0.0, guide_bkg_end_step).to(default_dtype)
            guide_weights_bkg = F.pad(guide_weights_bkg, (0, max_steps), value=0.0)
        
        guides = (guide_mode, guide_weight, guide_weight_bkg, guide_weights, guide_weights_bkg, guide, guide_bkg, guide_mask, guide_mask_bkg,
                  guide_weight_scheduler, guide_weight_scheduler_bkg, guide_end_step, guide_bkg_end_step, denoise, denoise_bkg)
        return (guides, )




class ClownsharKSamplerGuidesMisc_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA_MISC, {"default": 'blend', "tooltip": "Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "guide_weight": ("FLOAT", {"default": 0.05, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "guide_weight_bkg": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "guide_weight_cutoff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     "guide_weight_bkg_cutoff": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
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

    def main(self, guide_weight_scheduler="constant", guide_weight_scheduler_bkg="constant", guide_end_step=30, guide_bkg_end_step=30, guide_weight_cutoff=1.0, guide_weight_bkg_cutoff=1.0, guide=None, guide_bkg=None, guide_weight=0.0, guide_weight_bkg=0.0, 
                    guide_mode="blend", guide_weights=None, guide_weights_bkg=None, guide_mask=None, guide_mask_bkg=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        denoise, denoise_bkg = guide_weight_cutoff, guide_weight_bkg_cutoff
        
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





class ClownGuide_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA_SIMPLE, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "channelwise_mode": ("BOOLEAN", {"default": True}),
                     "projection_mode": ("BOOLEAN", {"default": True}),
                     "weight": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "cutoff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_step": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    "invert_mask": ("BOOLEAN", {"default": False}),
                    },
                    "optional": 
                    {
                        "guide": ("LATENT", ),
                        "mask": ("MASK", ),
                        "weights": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, weight_scheduler="constant", weight_scheduler_unmasked="constant", start_step=0, start_step_unmasked=0, end_step=30, end_step_unmasked=30, cutoff=1.0, cutoff_unmasked=1.0, guide=None, guide_unmasked=None, weight=0.0, weight_unmasked=0.0, 
                    guide_mode="epsilon", channelwise_mode=False, projection_mode=False, weights=None, weights_unmasked=None, mask=None, unmask=None, invert_mask=False,
                    ):
        CG = ClownGuides_Beta()
        mask = 1-mask if mask is not None else None
        guides = CG.main(weight_scheduler, weight_scheduler_unmasked, start_step, start_step_unmasked, end_step, end_step_unmasked, cutoff, cutoff_unmasked, guide, guide_unmasked, weight, weight_unmasked, 
                    guide_mode, channelwise_mode, projection_mode, weights, weights_unmasked, mask, unmask, invert_mask,
                    )
        return (guides[0], )






class ClownGuides_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA_SIMPLE, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "channelwise_mode": ("BOOLEAN", {"default": True}),
                     "projection_mode": ("BOOLEAN", {"default": True}),
                     "weight_masked": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "weight_unmasked": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "cutoff_masked": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     "cutoff_unmasked": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "weight_scheduler_unmasked": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "start_step_masked": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "start_step_unmasked": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_step_masked": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    "end_step_unmasked": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    "invert_mask": ("BOOLEAN", {"default": False}),
                    },
                    "optional": 
                    {
                        "guide_masked": ("LATENT", ),
                        "guide_unmasked": ("LATENT", ),
                        "mask": ("MASK", ),
                        "weights_masked": ("SIGMAS", ),
                        "weights_unmasked": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, weight_scheduler_masked="constant", weight_scheduler_unmasked="constant", start_step_masked=0, start_step_unmasked=0, end_step_masked=30, end_step_unmasked=30, cutoff_masked=1.0, cutoff_unmasked=1.0, guide_masked=None, guide_unmasked=None, weight_masked=0.0, weight_unmasked=0.0, 
                    guide_mode="epsilon", channelwise_mode=False, projection_mode=False, weights_masked=None, weights_unmasked=None, mask=None, unmask=None, invert_mask=False,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
        
        if invert_mask and mask is not None:
            mask = 1-mask
                
        if projection_mode:
            guide_mode = guide_mode + "_projection"
        
        if channelwise_mode:
            guide_mode = guide_mode + "_cw"
            
        if guide_mode == "unsample_cw":
            guide_mode = "unsample"
        if guide_mode == "resample_cw":
            guide_mode = "resample"
        
        if weight_scheduler_masked == "constant" and weights_masked == None: 
            weights_masked = initialize_or_scale(None, weight_masked, end_step_masked).to(default_dtype)
            weights_masked = F.pad(weights_masked, (0, max_steps), value=0.0)
        
        if weight_scheduler_unmasked == "constant" and weights_unmasked == None: 
            weights_unmasked = initialize_or_scale(None, weight_unmasked, end_step_unmasked).to(default_dtype)
            weights_unmasked = F.pad(weights_unmasked, (0, max_steps), value=0.0)
    
        guides = (guide_mode, weight_masked, weight_unmasked, weights_masked, weights_unmasked, guide_masked, guide_unmasked, mask, unmask,
                  weight_scheduler_masked, weight_scheduler_unmasked, start_step_masked, start_step_unmasked, end_step_masked, end_step_unmasked, cutoff_masked, cutoff_unmasked)
        return (guides, )










class ClownGuidesAB_Beta:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guide_mode": (GUIDE_MODE_NAMES_BETA_SIMPLE, {"default": 'epsilon', "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                     "channelwise_mode": ("BOOLEAN", {"default": False}),
                     "projection_mode": ("BOOLEAN", {"default": False}),
                     "weight_A": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                     "weight_B": ("FLOAT", {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                     "cutoff_A": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                     "cutoff_B": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler_A":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "weight_scheduler_B": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "start_step_A": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "start_step_B": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_step_A": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    "end_step_B": ("INT", {"default": 15, "min": 1, "max": 10000}),
                    },
                    "optional": 
                    {
                        "guide_A": ("LATENT", ),
                        "guide_B": ("LATENT", ),
                        "mask_A": ("MASK", ),
                        "mask_B": ("MASK", ),
                        "weights_A": ("SIGMAS", ),
                        "weights_B": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    CATEGORY = "RES4LYF/sampler_extensions"

    FUNCTION = "main"

    def main(self, weight_scheduler_A="constant", weight_scheduler_B="constant", start_step_A=0, start_step_B=0, end_step_A=30, end_step_B=30, cutoff_A=1.0, cutoff_B=1.0, guide_A=None, guide_B=None, weight_A=0.0, weight_B=0.0, 
                    guide_mode="epsilon", channelwise_mode=False, projection_mode=False, weights_A=None, weights_B=None, mask_A=None, mask_B=None,
                    ):
        default_dtype = torch.float64
        
        max_steps = 10000
                
        if projection_mode:
            guide_mode = guide_mode + "_projection"
        
        if channelwise_mode:
            guide_mode = guide_mode + "_cw"
            
        if guide_mode == "unsample_cw":
            guide_mode = "unsample"
        if guide_mode == "resample_cw":
            guide_mode = "resample"
        
        if weight_scheduler_A == "constant" and weights_A == None: 
            weights_A = initialize_or_scale(None, weight_A, end_step_A).to(default_dtype)
            weights_A = F.pad(weights_A, (0, max_steps), value=0.0)
        
        if weight_scheduler_B == "constant" and weights_B == None: 
            weights_B = initialize_or_scale(None, weight_B, end_step_B).to(default_dtype)
            weights_B = F.pad(weights_B, (0, max_steps), value=0.0)
    
        guides = (guide_mode, weight_A, weight_B, weights_A, weights_B, guide_A, guide_B, mask_A, mask_B,
                  weight_scheduler_A, weight_scheduler_B, start_step_A, start_step_B, end_step_A, end_step_B, cutoff_A, cutoff_B)
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


class ClownsharKSamplerAutomation_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    },
                    "optional": 
                    {
                        "automation": ("AUTOMATION", ),
                        "etas": ("SIGMAS", ),
                        "etas_substep": ("SIGMAS", ),
                        "s_noises": ("SIGMAS", ),
                        "unsample_resample_scales": ("SIGMAS", ),
                        "frame_weights": ("SIGMAS", ),
                        "frame_weights_bkg": ("SIGMAS", ),
                    }  
               }
    RETURN_TYPES = ("AUTOMATION",)
    RETURN_NAMES = ("automation",)
    CATEGORY = "RES4LYF/sampler_extensions"
    
    FUNCTION = "main"

    def main(self, automation=None, etas=None, etas_substep=None, s_noises=None, unsample_resample_scales=None, frame_weights=None, frame_weights_bkg=None):
        
        if automation is None:
            automation = {}
        
        frame_weights_grp = (frame_weights, frame_weights_bkg)

        automation['etas'] = etas
        automation['etas_substep'] = etas_substep
        automation['s_noises'] = s_noises
        automation['unsample_resample_scales'] = unsample_resample_scales
        automation['frame_weights_grp'] = frame_weights_grp

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
    
    DEPRECATED = True
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "main"

    def main(self, frame_weights, options=None,):

        if options is None:
            options = {}

        frame_weights_grp = (frame_weights, frame_weights)
        options['frame_weights_grp'] = frame_weights_grp

        return (options,)


