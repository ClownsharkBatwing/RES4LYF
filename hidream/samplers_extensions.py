import torch
from torch import Tensor
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from typing import Optional, Callable, Tuple, Dict, Any, Union
import copy

from nodes import MAX_RESOLUTION

from ..latents               import get_edge_mask
from ..helper                import OptionsManager, FrameWeightsManager, initialize_or_scale, get_res4lyf_scheduler_list, parse_range_string, parse_tile_sizes

from .rk_coefficients_beta   import RK_SAMPLER_NAMES_BETA_FOLDERS, get_default_sampler_name, get_sampler_name_list, process_sampler_name

from .noise_classes          import NOISE_GENERATOR_NAMES_SIMPLE
from .rk_noise_sampler_beta  import NOISE_MODE_NAMES
from .constants              import IMPLICIT_TYPE_NAMES, GUIDE_MODE_NAMES_BETA_SIMPLE, MAX_STEPS, FRAME_WEIGHTS_CONFIG_NAMES, FRAME_WEIGHTS_DYNAMICS_NAMES, FRAME_WEIGHTS_SCHEDULE_NAMES


class ClownSamplerSelector_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "sampler_name": (get_sampler_name_list(),  {"default": get_default_sampler_name()}), 
                    },
                "optional": 
                    {
                    }
                }

    RETURN_TYPES = (RK_SAMPLER_NAMES_BETA_FOLDERS,)
    RETURN_NAMES = ("sampler_name",) 
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            sampler_name = "res_2m",
            ):
        
        sampler_name, implicit_sampler_name = process_sampler_name(sampler_name)
        
        sampler_name = sampler_name if implicit_sampler_name == "use_explicit" else implicit_sampler_name
        
        return (sampler_name,)



class ClownOptions_SDE_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "noise_type_sde":         (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde_substep": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde":         (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "noise_mode_sde_substep": (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta":                    ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "eta_substep":            ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "seed":                   ("INT",                        {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                    },
                "optional": 
                    {
                    "etas":                   ("SIGMAS", ),
                    "etas_substep":           ("SIGMAS", ),
                    "options":                ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            noise_type_sde         = "gaussian",
            noise_type_sde_substep = "gaussian",
            noise_mode_sde         = "hard",
            noise_mode_sde_substep = "hard",
            eta                    = 0.5,
            eta_substep            = 0.5,
            seed             : int = -1,
            etas             : Optional[Tensor] = None,
            etas_substep     : Optional[Tensor] = None,
            options                = None,
            ): 
        
        options = options if options is not None else {}
        
        if noise_mode_sde == "none":
            noise_mode_sde = "hard"
            eta = 0.0
            
        if noise_mode_sde_substep == "none":
            noise_mode_sde_substep = "hard"
            eta_substep = 0.0
            
        if noise_type_sde == "none":
            noise_type_sde = "gaussian"
            eta = 0.0
            
        if noise_type_sde_substep == "none":
            noise_type_sde_substep = "gaussian"
            eta_substep = 0.0
            
        options['noise_type_sde']         = noise_type_sde
        options['noise_type_sde_substep'] = noise_type_sde_substep
        options['noise_mode_sde']         = noise_mode_sde
        options['noise_mode_sde_substep'] = noise_mode_sde_substep
        options['eta']                    = eta
        options['eta_substep']            = eta_substep
        options['noise_seed_sde']         = seed
        
        options['etas']                   = etas
        options['etas_substep']           = etas_substep

        return (options,)



class ClownOptions_StepSize_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "overshoot_mode":         (NOISE_MODE_NAMES, {"default": 'hard',                                                        "tooltip": "How step size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot_mode_substep": (NOISE_MODE_NAMES, {"default": 'hard',                                                        "tooltip": "How substep size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot":              ("FLOAT",          {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising step, then rescale to match the original. Has a softening effect."}),
                    "overshoot_substep":      ("FLOAT",          {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising substep, then rescale to match the original. Has a softening effect."}),
                    },
                "optional": 
                    {
                    "options":                ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            overshoot_mode         = "hard",
            overshoot_mode_substep = "hard",
            overshoot              = 0.0,
            overshoot_substep      = 0.0,
            options                = None,
            ): 
        
        options = options if options is not None else {}
            
        options['overshoot_mode']         = overshoot_mode
        options['overshoot_mode_substep'] = overshoot_mode_substep
        options['overshoot']              = overshoot
        options['overshoot_substep']      = overshoot_substep

        return (options,
            )


@dataclass
class DetailBoostOptions:
    noise_scaling_weight : float = 0.0
    noise_boost_step     : float = 0.0
    noise_boost_substep  : float = 0.0
    noise_anchor         : float = 1.0
    s_noise              : float = 1.0
    s_noise_substep      : float = 1.0
    d_noise              : float = 1.0

DETAIL_BOOST_METHODS = [
    'sampler',
    'sampler_normal',
    'sampler_substep',
    'sampler_substep_normal',
    'model',
    'model_alpha',
    ]

class ClownOptions_DetailBoost_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight":     ("FLOAT",              {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "method":     (DETAIL_BOOST_METHODS, {"default": "model",                                                       "tooltip": "Determines whether the sampler or the model underestimates the noise level."}),
                    #"noise_scaling_mode":    (['linear'] + NOISE_MODE_NAMES,  {"default": 'hard',                                          "tooltip": "Changes the steps where the effect is greatest. Most affect early steps, sinusoidal affects middle steps."}),
                    "mode":       (NOISE_MODE_NAMES,     {"default": 'hard',                                                        "tooltip": "Changes the steps where the effect is greatest. Most affect early steps, sinusoidal affects middle steps."}),
                    "eta":        ("FLOAT",              {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "The strength of the effect of the noise_scaling_mode. Linear ignores this parameter."}),
                    "start_step": ("INT",                {"default": 3,   "min": 0,      "max": MAX_STEPS}),
                    "end_step":   ("INT",                {"default": 10,  "min": -1,     "max": MAX_STEPS}),

                    #"noise_scaling_cycles":  ("INT",              {"default": 1, "min": 1, "max": MAX_STEPS}),

                    #"noise_boost_step":      ("FLOAT",            {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    #"noise_boost_substep":   ("FLOAT",            {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    #"sampler_scaling_normalize":("BOOLEAN",          {"default": False,                                                          "tooltip": "Limit saturation and luminosity drift."}),
                    },
                "optional": 
                    {
                    "weights": ("SIGMAS", ),
                    "etas":    ("SIGMAS", ),
                    "options":               ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            weight      : float = 0.0,
            method      : str   = "sampler",
            mode        : str   = "linear",
            eta         : float = 0.5,
            start_step  : int   = 0,
            end_step    : int   = -1,


            noise_scaling_cycles      : int   = 1,

            noise_boost_step          : float = 0.0,
            noise_boost_substep       : float = 0.0,
            sampler_scaling_normalize : bool  = False,

            weights     : Optional[Tensor] = None,
            etas        : Optional[Tensor] = None,
            
            options                        = None
            ):
        
        noise_scaling_weight     = weight
        noise_scaling_type       = method
        noise_scaling_mode       = mode
        noise_scaling_eta        = eta
        noise_scaling_start_step = start_step
        noise_scaling_end_step   = end_step
        
        noise_scaling_weights = weights
        noise_scaling_etas    = etas
        
        
        options = options if options is not None else {}
        
        default_dtype = torch.float64
        default_device = torch.device('cuda')
        
        if noise_scaling_type.endswith("_normal"):
            sampler_scaling_normalize = True
            noise_scaling_type = noise_scaling_type[:-7]
        
        if noise_scaling_end_step == -1:
            noise_scaling_end_step = MAX_STEPS
        
        if noise_scaling_weights == None: 
            noise_scaling_weights = initialize_or_scale(None, noise_scaling_weight, MAX_STEPS).to(default_dtype).to(default_device)
        
        if noise_scaling_etas == None: 
            noise_scaling_etas = initialize_or_scale(None, noise_scaling_eta, MAX_STEPS).to(default_dtype).to(default_device)
        
        noise_scaling_prepend = torch.zeros((noise_scaling_start_step,), dtype=default_dtype, device=default_device)
        
        noise_scaling_weights = torch.cat((noise_scaling_prepend, noise_scaling_weights), dim=0)
        noise_scaling_etas    = torch.cat((noise_scaling_prepend, noise_scaling_etas),    dim=0)

        if noise_scaling_weights.shape[-1] > noise_scaling_end_step:
            noise_scaling_weights = noise_scaling_weights[:noise_scaling_end_step]
            
        if noise_scaling_etas.shape[-1] > noise_scaling_end_step:
            noise_scaling_etas = noise_scaling_etas[:noise_scaling_end_step]
        
        noise_scaling_weights = F.pad(noise_scaling_weights, (0, MAX_STEPS), value=0.0)
        noise_scaling_etas = F.pad(noise_scaling_etas, (0, MAX_STEPS), value=0.0)
        
        options['noise_scaling_weight']  = noise_scaling_weight
        options['noise_scaling_type']    = noise_scaling_type
        options['noise_scaling_mode']    = noise_scaling_mode
        options['noise_scaling_eta']     = noise_scaling_eta
        options['noise_scaling_cycles']  = noise_scaling_cycles
        
        options['noise_scaling_weights'] = noise_scaling_weights
        options['noise_scaling_etas']    = noise_scaling_etas
        
        options['noise_boost_step']      = noise_boost_step
        options['noise_boost_substep']   = noise_boost_substep
        options['noise_boost_normalize'] = sampler_scaling_normalize

        """options['DetailBoostOptions'] = DetailBoostOptions(
            noise_scaling_weight = noise_scaling_weight,
            noise_scaling_type    = noise_scaling_type,
            noise_scaling_mode    = noise_scaling_mode,
            noise_scaling_eta     = noise_scaling_eta,
            
            noise_boost_step      = noise_boost_step,
            noise_boost_substep   = noise_boost_substep,
            noise_boost_normalize = noise_boost_normalize,
            
            noise_anchor          = noise_anchor,
            s_noise               = s_noise,
            s_noise_substep       = s_noise_substep,
            d_noise               = d_noise
            d_noise_start_step    = d_noise_start_step
        )"""

        return (options,)




class ClownOptions_SigmaScaling_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "s_noise":              ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "s_noise_substep":      ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "noise_anchor_sde":     ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."}),
                    
                    "lying":                ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
                    "lying_inv":            ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Upscales the sigma schedule. Will soften the image and deepen colors. Use after d_noise to counteract desaturation."}),
                    "lying_start_step":     ("INT",   {"default": 0, "min": 0, "max": MAX_STEPS}),
                    "lying_inv_start_step": ("INT",   {"default": 1, "min": 0, "max": MAX_STEPS}),

                    },
                "optional": 
                    {
                    "s_noises":             ("SIGMAS", ),
                    "s_noises_substep":     ("SIGMAS", ),
                    "options":              ("OPTIONS", ),
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            noise_anchor_sde        : float = 1.0,
            
            s_noise                 : float = 1.0,
            s_noise_substep         : float = 1.0,
            lying                   : float = 1.0,
            lying_start_step        : int   = 0,
            
            lying_inv               : float = 1.0,
            lying_inv_start_step    : int   = 1,
            
            s_noises                : Optional[Tensor] = None,
            s_noises_substep        : Optional[Tensor] = None,
            options                         = None
            ):
        
        options = options if options is not None else {}
        
        default_dtype = torch.float64
        default_device = torch.device('cuda')
        
        
        
        options['noise_anchor']           = noise_anchor_sde
        options['s_noise']                = s_noise
        options['s_noise_substep']        = s_noise_substep
        options['d_noise']                = lying
        options['d_noise_start_step']     = lying_start_step
        options['d_noise_inv']            = lying_inv
        options['d_noise_inv_start_step'] = lying_inv_start_step

        options['s_noises']                = s_noises
        options['s_noises_substep']        = s_noises_substep

        return (options,)



class ClownOptions_FlowGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "sync_eps": ("FLOAT", {"default": 0.75, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Accelerate convergence with positive values when sampling, negative values when unsampling."}),
                    },
                "optional": 
                    {
                    "options":               ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            sync_eps = 0.75,
            options  = None
            ):
        
        options = options if options is not None else {}
            
        options['flow_sync_eps'] = sync_eps

        return (options,)



class ClownOptions_Momentum_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "momentum": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Accelerate convergence with positive values when sampling, negative values when unsampling."}),
                    },
                "optional": 
                    {
                    "options":               ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            momentum = 0.0,
            options  = None
            ):
        
        options = options if options is not None else {}
            
        options['momentum'] = momentum

        return (options,)



class ClownOptions_ImplicitSteps_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "implicit_type":          (IMPLICIT_TYPE_NAMES, {"default": "bongmath"}), 
                    "implicit_type_substeps": (IMPLICIT_TYPE_NAMES, {"default": "bongmath"}), 
                    "implicit_steps":         ("INT",               {"default": 0, "min": 0, "max": 10000}),
                    "implicit_substeps":      ("INT",               {"default": 0, "min": 0, "max": 10000}),
                    },
                "optional": 
                    {
                    "options":                ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            implicit_type          = "bongmath",
            implicit_type_substeps = "bongmath",
            implicit_steps         = 0,
            implicit_substeps      = 0,
            options                = None
            ): 
        
        options = options if options is not None else {}
            
        options['implicit_type']          = implicit_type
        options['implicit_type_substeps'] = implicit_type_substeps
        options['implicit_steps']         = implicit_steps
        options['implicit_substeps']      = implicit_substeps

        return (options,)



class ClownOptions_Cycles_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "cycles"          : ("FLOAT", {"default": 0.0, "min":  0.0,   "max": 10000, "step":0.5,  "round": 0.5}),
                    "eta_decay_scale" : ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Multiplies etas by this number after every cycle. May help drive convergence." }),
                    "unsample_eta"    : ("FLOAT", {"default": 0.5, "min": -10000, "max": 10000, "step":0.01}),
                    "unsampler_override"  : (get_sampler_name_list(), {"default": "none"}),
                    "unsample_steps_to_run"  : ("INT", {"default": -1, "min":  -1,   "max": 10000, "step":1,  "round": 1}),
                    "unsample_cfg"    : ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "unsample_bongmath" : ("BOOLEAN", {"default": False}),
                    },
                "optional": 
                    {
                    "options":    ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            cycles          = 0,
            unsample_eta    = 0.5,
            eta_decay_scale = 1.0,
            unsample_cfg    = 1.0,
            unsampler_override  = "none",
            unsample_steps_to_run = -1,
            unsample_bongmath = False,
            options         = None
            ): 
        
        options = options if options is not None else {}
            
        options['rebounds']        = int(cycles * 2)
        options['unsample_eta']    = unsample_eta
        options['unsampler_name']  = unsampler_override
        options['eta_decay_scale'] = eta_decay_scale
        options['unsample_steps_to_run'] = unsample_steps_to_run
        options['unsample_cfg']    = unsample_cfg
        options['unsample_bongmath'] = unsample_bongmath

        return (options,)




class SharkOptions_StartStep_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "start_at_step": ("INT", {"default": 0, "min": -1, "max": 10000, "step":1,}),

                    },
                "optional": 
                    {
                    "options":    ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            start_at_step = 0,
            options       = None
            ): 
        
        options = options if options is not None else {}
            
        options['start_at_step'] = start_at_step

        return (options,)




class ClownOptions_Tile_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "tile_width" : ("INT", {"default": 1024, "min": -1, "max": 10000, "step":1,}),
                    "tile_height": ("INT", {"default": 1024, "min": -1, "max": 10000, "step":1,}),
                    },
                "optional": 
                    {
                    "options":    ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            tile_height = 1024,
            tile_width  = 1024,
            options     = None
            ): 
        
        options = options if options is not None else {}
        
        tile_sizes = options.get('tile_sizes', [])
        tile_sizes.append((tile_height, tile_width))
        options['tile_sizes'] = tile_sizes

        return (options,)



class ClownOptions_Tile_Advanced_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "tile_sizes": ("STRING", {"default": "1024,1024", "multiline": True}),   
                    },
                "optional": 
                    {
                    "options":    ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            tile_sizes = "1024,1024",
            options    = None
            ): 
        
        options = options if options is not None else {}
            
        tiles_height_width = parse_tile_sizes(tile_sizes)
        options['tile_sizes'] = [(tile[-1], tile[-2]) for tile in tiles_height_width]  # swap height and width to be consistent... width, height

        return (options,)




class ClownOptions_ExtraOptions_Beta:
    @classmethod
    def INPUT_TYPES(cls):
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
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"

    def main(self,
            extra_options = "",
            options       = None
            ):

        options = options if options is not None else {}
        
        if 'extra_options' in options:
            options['extra_options'] += '\n' + extra_options
        else:
            options['extra_options']  = extra_options

        return (options, )




class ClownOptions_DenoisedSampling_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "cycles"          : ("FLOAT", {"default": 0.0, "min":  0.0,   "max": 10000, "step":0.5,  "round": 0.5}),
                    "eta_decay_scale" : ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Multiplies etas by this number after every cycle. May help drive convergence." }),
                    "unsample_eta"    : ("FLOAT", {"default": 0.5, "min": -10000, "max": 10000, "step":0.01}),
                    "unsampler_override"  : (get_sampler_name_list(), {"default": "none"}),
                    "unsample_steps_to_run"  : ("INT", {"default": -1, "min":  -1,   "max": 10000, "step":1,  "round": 1}),
                    "unsample_cfg"    : ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "unsample_bongmath" : ("BOOLEAN", {"default": False}),                    
                    },
                "optional": 
                    {
                    "options": ("OPTIONS", ),   
                    }  
            }
        
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"

    def main(self,
            extra_options = "",
            options       = None
            ):

        options = options if options is not None else {}
        
        if 'extra_options' in options:
            options['extra_options'] += '\n' + extra_options
        else:
            options['extra_options']  = extra_options

        return (options, )




class ClownOptions_Automation_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {
                    "etas":             ("SIGMAS", ),
                    "etas_substep":     ("SIGMAS", ),
                    "s_noises":         ("SIGMAS", ),
                    "s_noises_substep": ("SIGMAS", ),
                    "epsilon_scales":   ("SIGMAS", ),
                    "frame_weights":    ("SIGMAS", ),
                    "options":          ("OPTIONS",),  
                    }  
                }
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"

    def main(self,
            etas             = None,
            etas_substep     = None,
            s_noises         = None,
            s_noises_substep = None,
            epsilon_scales   = None,
            frame_weights    = None,
            options          = None
            ):
                
        options = options if options is not None else {}
            
        options_mgr = OptionsManager(options)

        frame_weights_mgr = options_mgr.get("frame_weights_mgr")
        if frame_weights_mgr is None and frame_weights is not None:
            frame_weights_mgr = FrameWeightsManager()
            frame_weights_mgr.set_custom_weights("frame_weights", frame_weights)
            
        automation = {
            "etas"              : etas,
            "etas_substep"      : etas_substep,
            "s_noises"          : s_noises,
            "s_noises_substep"  : s_noises_substep,
            "epsilon_scales"    : epsilon_scales,
            "frame_weights_mgr" : frame_weights_mgr,
        }
        
        options["automation"] = automation

        return (options, )





class SharkOptions_GuideCond_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {
                    "positive" : ("CONDITIONING", ),
                    "negative" : ("CONDITIONING", ),
                    "cfg"      : ("FLOAT",   {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "options"  : ("OPTIONS",),  
                    }  
                }
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"

    def main(self,
            positive = None,
            negative = None,
            cfg      = 1.0,
            options  = None,
            ):
                
        options = options if options is not None else {}

        flow_cond = {
            "yt_positive" : positive,
            "yt_negative" : negative,
            "yt_cfg"      : cfg,
        }
        
        options["flow_cond"] = flow_cond

        return (options, )




class SharkOptions_GuideConds_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {
                    "positive_masked"   : ("CONDITIONING", ),
                    "positive_unmasked" : ("CONDITIONING", ),
                    "negative_masked"   : ("CONDITIONING", ),
                    "negative_unmasked" : ("CONDITIONING", ),
                    "cfg_masked"        : ("FLOAT",   {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg_unmasked"      : ("FLOAT",   {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "options"           : ("OPTIONS",),  
                    }  
                }
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"

    def main(self,
            positive_masked   = None,
            negative_masked   = None,
            cfg_masked        = 1.0,
            positive_unmasked = None,
            negative_unmasked = None,
            cfg_unmasked      = 1.0,
            options  = None,
            ):
                
        options = options if options is not None else {}

        flow_cond = {
            "yt_positive"     : positive_masked,
            "yt_negative"     : negative_masked,
            "yt_cfg"          : cfg_masked,
            "yt_inv_positive" : positive_unmasked,
            "yt_inv_negative" : negative_unmasked,
            "yt_inv_cfg"      : cfg_unmasked,
        }
        
        options["flow_cond"] = flow_cond

        return (options, )





class SharkOptions_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "s_noise_init":    ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                "denoise_alt":     ("FLOAT",                      {"default": 1.0, "min": -10000,   "max": 10000,   "step":0.01}),
                "channelwise_cfg": ("BOOLEAN",                    {"default": False}),
                },
            "optional": {
                "options":         ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            noise_type_init = "gaussian",
            s_noise_init    = 1.0,
            denoise_alt     = 1.0,
            channelwise_cfg = False,
            options         = None
            ): 
        
        options = options if options is not None else {}
            
        options['noise_type_init']  = noise_type_init
        options['noise_init_stdev'] = s_noise_init
        options['denoise_alt']      = denoise_alt
        options['channelwise_cfg']  = channelwise_cfg

        return (options,)
    



class SharkOptions_UltraCascade_Latent_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width":   ("INT", {"default": 60, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height":  ("INT", {"default": 36, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                },
            "optional": {
                "options": ("OPTIONS",),   
                }
            }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            width  : int = 60,
            height : int = 36,
            options       = None,
            ): 
        
        options = options if options is not None else {}
            
        options['ultracascade_latent_width']  = width
        options['ultracascade_latent_height'] = height

        return (options,)




class ClownOptions_SwapSampler_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name":       (get_sampler_name_list(), {"default": get_default_sampler_name()}), 
                "swap_below_err":     ("FLOAT",                 {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Swap samplers if the error per step falls below this threshold."}),
                "swap_at_step":       ("INT",                   {"default": 30,  "min": 1,      "max": 10000}),
                "log_err_to_console": ("BOOLEAN",               {"default": False}),
                },
            "optional": {
                "options":            ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            sampler_name       = "res_3m",
            swap_below_err     = 0.0,
            swap_at_step       = 30,
            log_err_to_console = False,
            options            = None,
            ): 
        
        sampler_name, implicit_sampler_name = process_sampler_name(sampler_name)
        
        sampler_name = sampler_name if implicit_sampler_name == "use_explicit" else implicit_sampler_name
                
        options = options if options is not None else {}
            
        options['rk_swap_type']      = sampler_name
        options['rk_swap_threshold'] = swap_below_err
        options['rk_swap_step']      = swap_at_step
        options['rk_swap_print']     = log_err_to_console

        return (options,)
    
    

    
    
class ClownOptions_SDE_Mask_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max":               ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Clamp the max value for the mask."}),
                "min":               ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Clamp the min value for the mask."}),
                "invert_mask":       ("BOOLEAN",                                   {"default": False}),
                },
            "optional": {
                "mask":              ("MASK", ),
                "options":           ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            max = 1.0,
            min = 0.0,
            invert_mask = False,
            mask     = None,
            options      = None,
            ): 
        
        options = copy.deepcopy(options) if options is not None else {}
        
        if invert_mask:
            mask = 1-mask
        
        mask = ((mask - mask.min()) * (max - min)) / (mask.max() - mask.min()) + min    
        
        options['sde_mask'] = mask

        return (options,)




class ClownGuide_Mean_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight":               ("FLOAT",                                     {"default": 0.75, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "cutoff":               ("FLOAT",                                     {"default": 1.0,  "min":  0.0,    "max": 1.0,   "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "start_step":           ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step":             ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":          ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide":                ("LATENT", ),
                    "mask":                 ("MASK", ),
                    "weights":              ("SIGMAS", ),
                    "guides":               ("GUIDES", ),
                    }  
                }
        
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight_scheduler          = "constant",
            start_step                = 0,
            end_step                  = 30,
            cutoff                    = 1.0,
            guide                     = None,
            weight                    = 0.0,

            channelwise_mode          = False,
            projection_mode           = False,
            weights                   = None,
            mask                      = None,
            invert_mask               = False,
            
            guides                    = None,
            ):
        
        default_dtype = torch.float64
        
        mask = 1-mask if mask is not None else None
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if guide is not None:
            raw_x = guide.get('state_info', {}).get('raw_x', None)
            if raw_x is not None:
                guide          = {'samples': guide['state_info']['raw_x'].clone()}
            else:
                guide          = {'samples': guide['samples'].clone()}
        
        if weight_scheduler == "constant": # and weights == None: 
            weights = initialize_or_scale(None, weight, end_step).to(default_dtype)
            weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
            
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['weight_mean']           = weight
        guides['weights_mean']          = weights
        guides['guide_mean']            = guide
        guides['mask_mean']             = mask
        
        guides['weight_scheduler_mean'] = weight_scheduler
        guides['start_step_mean']       = start_step
        guides['end_step_mean']         = end_step
        
        guides['cutoff_mean']           = cutoff
        
        return (guides, )



class ClownGuide_FrequencySeparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "apply_to"       : (["AdaIN"], {"default": "AdaIN"}),
                    "method"         : (["gaussian", "gaussian_pw", "median", "median_pw",], {"default": "median"}),
                    "sigma":             ("FLOAT", {"default": 3.0, "min":  -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Low values produce results closer to the guide image. No effect with median."}),
                    "kernel_size":       ("INT",   {"default": 8,    "min":  1,      "max": 11111, "step": 1, "tooltip": "Primary control with median. Set the Re___Patcher node to float32 or lower precision if you have OOMs. You may have them regardless at higher kernel sizes with median."}),
                    "inner_kernel_size": ("INT",   {"default": 2,    "min":  1,      "max": 11111, "step": 1, "tooltip": "Should be equal to, or less than, kernel_size."}),
                    "stride":            ("INT",   {"default": 2,    "min":  1,      "max": 11111, "step": 1, "tooltip": "Should be equal to, or less than, inner_kernel_size."}),


                    "lowpass_weight":    ("FLOAT", {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Typically should be set to 1.0. Lower values may sharpen the image, higher values may blur the image."}),
                    "highpass_weight":   ("FLOAT", {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Typically should be set to 1.0. Higher values may sharpen the image, lower values may blur the image."}),

                    "guides":            ("GUIDES", ),
                    },
                "optional": 
                    {
                    "mask"           : ("MASK",),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"
    EXPERIMENTAL = True

    def main(self,
            apply_to       = "AdaIN",
            method         = "median",
            sigma          = 3.0,
            kernel_size       = 9,
            inner_kernel_size = 2,
            stride            = 2,
            lowpass_weight    = 1.0,
            highpass_weight   = 1.0,
            guides            = None,
            mask              = None,
            ):
        
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['freqsep_apply_to']       = apply_to
        guides['freqsep_lowpass_method'] = method
        guides['freqsep_sigma']          = sigma
        guides['freqsep_kernel_size']    = kernel_size
        guides['freqsep_inner_kernel_size']    = inner_kernel_size
        guides['freqsep_stride']         = stride

        guides['freqsep_lowpass_weight'] = lowpass_weight
        guides['freqsep_highpass_weight']= highpass_weight
        guides['freqsep_mask']           = mask

        return (guides, )




class ClownGuide_Style_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "apply_to":         (["positive", "negative"],                    {"default": "positive", "tooltip": "When using CFG, decides whether to apply the guide to the positive or negative conditioning."}),
                    "method":           (["AdaIN", "WCT", "scattersort","none"],      {"default": "WCT"}),
                    "weight":           ("FLOAT",                                     {"default": 1.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide by multiplying all other weights by this value."}),
                    "synweight":        ("FLOAT",                                     {"default": 1.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the relative strength of the guide on the opposite conditioning to what was selected: i.e., negative if positive in apply_to. Recommended to avoid CFG burn."}),
                    "weight_scheduler": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant", "tooltip": "Selecting any scheduler except constant will cause the strength to gradually decay to zero. Try beta57 vs. linear quadratic."},),
                    "start_step":       ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step":         ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    "invert_mask":      ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide":            ("LATENT", ),
                    "mask":             ("MASK", ),
                    "weights":          ("SIGMAS", ),
                    "guides":           ("GUIDES", ),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"
    DESCRIPTION  = "Transfer some visual aspects of style from a guide (reference) image. If nothing about style is specified in the prompt, it may just transfer the lighting and color scheme." + \
                "If using CFG results in burn, or a very dark/bright image in the preview followed by a bad output, try duplicating and chaining this node, so that the guide may be applied to both positive and negative conditioning." + \
                "Currently supported models: SD1.5, SDXL, Stable Cascade, SD3.5, AuraFlow, Flux, HiDream, WAN, and LTXV."

    def main(self,
            apply_to         = "all",
            method           = "WCT",
            weight           = 1.0,
            synweight        = 1.0,
            weight_scheduler = "constant",
            start_step       = 0,
            end_step         = 15,
            invert_mask      = False,
            
            guide            = None,
            mask             = None,
            weights          = None,
            guides           = None,
            ):
        
        default_dtype = torch.float64
        
        mask = 1-mask if mask is not None else None
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if guide is not None:
            raw_x = guide.get('state_info', {}).get('raw_x', None)
            if raw_x is not None:
                guide          = {'samples': guide['state_info']['raw_x'].clone()}
            else:
                guide          = {'samples': guide['samples'].clone()}
        
        if weight_scheduler == "constant": # and weights == None: 
            weights = initialize_or_scale(None, weight, end_step).to(default_dtype)
            prepend = torch.zeros(start_step).to(weights)
            weights = torch.cat([prepend, weights])
            weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
            
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['style_method'] = method
        
        if apply_to in {"positive", "all"}:
        
            guides['weight_style_pos']           = weight
            guides['weights_style_pos']          = weights

            guides['synweight_style_pos']        = synweight

            guides['guide_style_pos']            = guide
            guides['mask_style_pos']             = mask

            guides['weight_scheduler_style_pos'] = weight_scheduler
            guides['start_step_style_pos']       = start_step
            guides['end_step_style_pos']         = end_step
            
        if apply_to in {"negative", "all"}:
            guides['weight_style_neg']           = weight
            guides['weights_style_neg']          = weights

            guides['synweight_style_neg']        = synweight

            guides['guide_style_neg']            = guide
            guides['mask_style_neg']             = mask

            guides['weight_scheduler_style_neg'] = weight_scheduler
            guides['start_step_style_neg']       = start_step
            guides['end_step_style_neg']         = end_step
        
        return (guides, )






class ClownGuide_Style_EdgeWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "edge_width":       ("INT",     {"default": 20,  "min":  1,   "max": 10000}),
                    },
                "optional": 
                    {
                    "guides":           ("GUIDES", ),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"
    DESCRIPTION  = "Set an edge mask for some style guide types such as scattersort. Can help mitigate seams."

    def main(self,
            edge_width = 20,
            guides     = None,
            ):
        
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        if guides.get('mask_style_pos') is not None:
            guides['mask_edge_style_pos'] = get_edge_mask(guides.get('mask_style_pos'), edge_width)
            
        if guides.get('mask_style_neg') is not None:
            guides['mask_edge_style_neg'] = get_edge_mask(guides.get('mask_style_neg'), edge_width)
        
        return (guides, )





class ClownGuide_Style_TileSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "height": ("INT",     {"default": 128,  "min":  16,   "max": 10000, "step": 16}),
                    "width" : ("INT",     {"default": 128,  "min":  16,   "max": 10000, "step": 16}),
                    "padding" : ("INT",     {"default": 64,  "min":  0,   "max": 10000, "step": 16}),
                    },
                "optional": 
                    {
                    "guides":           ("GUIDES", ),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"
    DESCRIPTION  = "Set a tile size for some style guide types such as scattersort. Can improve adherence to the input image."

    def main(self,
            height = 128,
            width  = 128,
            padding = 64,
            guides = None,
            ):
        
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['style_tile_height']  = height  // 16
        guides['style_tile_width']   = width   // 16
        guides['style_tile_padding'] = padding // 16

        return (guides, )





class ClownGuide_AdaIN_MMDiT_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight":           ("FLOAT",                                     {"default": 1.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide by multiplying all other weights by this value."}),
                    "weight_scheduler": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "double_blocks"   : ("STRING",                                    {"default": "", "multiline": True}),
                    "double_weights"  : ("STRING",                                    {"default": "", "multiline": True}),
                    "single_blocks"   : ("STRING",                                    {"default": "20", "multiline": True}),
                    "single_weights"  : ("STRING",                                    {"default": "0.5", "multiline": True}),
                    "start_step":       ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step":         ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":      ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide":            ("LATENT", ),
                    "mask":             ("MASK", ),
                    "weights":          ("SIGMAS", ),
                    "guides":           ("GUIDES", ),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight           = 1.0,
            weight_scheduler = "constant",
            double_weights   = "0.1",
            single_weights   = "0.0", 
            double_blocks    = "all",
            single_blocks    = "all", 
            start_step       = 0,
            end_step         = 15,
            invert_mask      = False,
            
            guide            = None,
            mask             = None,
            weights          = None,
            guides           = None,
            ):
        
        default_dtype = torch.float64
        
        mask = 1-mask if mask is not None else None
        
        double_weights = parse_range_string(double_weights)
        single_weights = parse_range_string(single_weights)
        
        if len(double_weights) == 0:
            double_weights.append(0.0)
        if len(single_weights) == 0:
            single_weights.append(0.0)
            
        if len(double_weights) == 1:
            double_weights = double_weights * 100
        if len(single_weights) == 1:
            single_weights = single_weights * 100
            
        if type(double_weights[0]) == int:
            double_weights = [float(val) for val in double_weights]
        if type(single_weights[0]) == int:
            single_weights = [float(val) for val in single_weights]
        
        if double_blocks == "all":
            double_blocks  = [val for val in range(100)]
            if len(double_weights) == 1:
                double_weights = [double_weights[0]] * 100
        else:
            double_blocks  = parse_range_string(double_blocks)
            
            weights_expanded = [0.0] * 100
            for b, w in zip(double_blocks, double_weights):
                weights_expanded[b] = w
            double_weights = weights_expanded
            
        
        if single_blocks == "all":
            single_blocks = [val for val in range(100)]
            if len(single_weights) == 1:
                single_weights = [single_weights[0]] * 100
        else:
            single_blocks  = parse_range_string(single_blocks)
            
            weights_expanded = [0.0] * 100
            for b, w in zip(single_blocks, single_weights):
                weights_expanded[b] = w
            single_weights = weights_expanded
        
        
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if guide is not None:
            raw_x = guide.get('state_info', {}).get('raw_x', None)
            if raw_x is not None:
                guide          = {'samples': guide['state_info']['raw_x'].clone()}
            else:
                guide          = {'samples': guide['samples'].clone()}
        
        if weight_scheduler == "constant": # and weights == None: 
            weights = initialize_or_scale(None, weight, end_step).to(default_dtype)
            prepend = torch.zeros(start_step).to(weights)
            weights = torch.cat([prepend, weights])
            weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['weight_adain']           = weight
        guides['weights_adain']          = weights
        
        guides['blocks_adain_mmdit'] = {
            "double_weights": double_weights,
            "single_weights": single_weights,
            "double_blocks" : double_blocks,
            "single_blocks" : single_blocks,
        }
        
        guides['guide_adain']            = guide
        guides['mask_adain']             = mask

        guides['weight_scheduler_adain'] = weight_scheduler
        guides['start_step_adain']       = start_step
        guides['end_step_adain']         = end_step
        
        return (guides, )







class ClownGuide_AttnInj_MMDiT_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight":           ("FLOAT",                                     {"default": 1.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide by multiplying all other weights by this value."}),
                    "weight_scheduler": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "double_blocks"   : ("STRING",                                    {"default": "0,1,3", "multiline": True}),
                    "double_weights"  : ("STRING",                                    {"default": "1.0", "multiline": True}),
                    "single_blocks"   : ("STRING",                                    {"default": "20", "multiline": True}),
                    "single_weights"  : ("STRING",                                    {"default": "0.5", "multiline": True}),
                    
                    "img_q":            ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "img_k":            ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "img_v":            ("FLOAT",                                     {"default": 1.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),

                    "txt_q":            ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "txt_k":            ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "txt_v":            ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),

                    "img_q_norm":       ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "img_k_norm":       ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "img_v_norm":       ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),

                    "txt_q_norm":       ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "txt_k_norm":       ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),
                    "txt_v_norm":       ("FLOAT",                                     {"default": 0.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set relative injection strength."}),

                    "start_step":       ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step":         ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":      ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide":            ("LATENT", ),
                    "mask":             ("MASK", ),
                    "weights":          ("SIGMAS", ),
                    "guides":           ("GUIDES", ),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight           = 1.0,
            weight_scheduler = "constant",
            double_weights   = "0.1",
            single_weights   = "0.0", 
            double_blocks    = "all",
            single_blocks    = "all", 
            
            img_q            = 0.0,
            img_k            = 0.0,
            img_v            = 0.0,
            
            txt_q            = 0.0,
            txt_k            = 0.0,
            txt_v            = 0.0,
            
            img_q_norm       = 0.0,
            img_k_norm       = 0.0,
            img_v_norm       = 0.0,
            
            txt_q_norm       = 0.0,
            txt_k_norm       = 0.0,
            txt_v_norm       = 0.0,
            
            start_step       = 0,
            end_step         = 15,
            invert_mask      = False,
            
            guide            = None,
            mask             = None,
            weights          = None,
            guides           = None,
            ):
        
        default_dtype = torch.float64
        
        mask = 1-mask if mask is not None else None
        
        double_weights = parse_range_string(double_weights)
        single_weights = parse_range_string(single_weights)
        
        if len(double_weights) == 0:
            double_weights.append(0.0)
        if len(single_weights) == 0:
            single_weights.append(0.0)
        
        if len(double_weights) == 1:
            double_weights = double_weights * 100
        if len(single_weights) == 1:
            single_weights = single_weights * 100
        
        if type(double_weights[0]) == int:
            double_weights = [float(val) for val in double_weights]
        if type(single_weights[0]) == int:
            single_weights = [float(val) for val in single_weights]
        
        if double_blocks == "all":
            double_blocks  = [val for val in range(100)]
            if len(double_weights) == 1:
                double_weights = [double_weights[0]] * 100
        else:
            double_blocks  = parse_range_string(double_blocks)
            
            weights_expanded = [0.0] * 100
            for b, w in zip(double_blocks, double_weights):
                weights_expanded[b] = w
            double_weights = weights_expanded
            
        
        if single_blocks == "all":
            single_blocks = [val for val in range(100)]
            if len(single_weights) == 1:
                single_weights = [single_weights[0]] * 100
        else:
            single_blocks  = parse_range_string(single_blocks)
            
            weights_expanded = [0.0] * 100
            for b, w in zip(single_blocks, single_weights):
                weights_expanded[b] = w
            single_weights = weights_expanded
        
        
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if guide is not None:
            raw_x = guide.get('state_info', {}).get('raw_x', None)
            if raw_x is not None:
                guide          = {'samples': guide['state_info']['raw_x'].clone()}
            else:
                guide          = {'samples': guide['samples'].clone()}
        
        if weight_scheduler == "constant": # and weights == None: 
            weights = initialize_or_scale(None, weight, end_step).to(default_dtype)
            prepend = torch.zeros(start_step).to(weights)
            weights = torch.cat([prepend, weights])
            weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['weight_attninj']           = weight
        guides['weights_attninj']          = weights
        
        guides['blocks_attninj_mmdit'] = {
            "double_weights": double_weights,
            "single_weights": single_weights,
            "double_blocks" : double_blocks,
            "single_blocks" : single_blocks,
        }
        
        guides['blocks_attninj_qkv'] = {
            "img_q": img_q,
            "img_k": img_k,
            "img_v": img_v,
            "txt_q": txt_q,
            "txt_k": txt_k,
            "txt_v": txt_v,
            
            "img_q_norm": img_q_norm,
            "img_k_norm": img_k_norm,
            "img_v_norm": img_v_norm,
            "txt_q_norm": txt_q_norm,
            "txt_k_norm": txt_k_norm,
            "txt_v_norm": txt_v_norm,
        }
        
        guides['guide_attninj']            = guide
        guides['mask_attninj']             = mask

        guides['weight_scheduler_attninj'] = weight_scheduler
        guides['start_step_attninj']       = start_step
        guides['end_step_attninj']         = end_step
        
        return (guides, )






class ClownGuide_StyleNorm_Advanced_HiDream:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight":           ("FLOAT",                                     {"default": 1.0, "min":  -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide by multiplying all other weights by this value."}),
                    "weight_scheduler": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    
                    "double_blocks"   : ("STRING",                                    {"default": "", "multiline": True}),
                    "double_weights"  : ("STRING",                                    {"default": "", "multiline": True}),
                    "single_blocks"   : ("STRING",                                    {"default": "20", "multiline": True}),
                    "single_weights"  : ("STRING",                                    {"default": "0.5", "multiline": True}),

                    "mode": (["scattersort", "AdaIN"], {"default": "scattersort"},),
                    "noise_mode": (["direct", "update", "smart", "bonanza"], {"default": "smart"},),

                    "moe_gate"               : ("BOOLEAN", {"default": False}),
                    "moe_ff"                 : ("BOOLEAN", {"default": False}),
                    "ff":                      ("BOOLEAN", {"default": False}),
                    "shared_experts":          ("BOOLEAN", {"default": False}),

                    "double_img_io":           ("BOOLEAN", {"default": False}),
                    "double_img_norm0":        ("BOOLEAN", {"default": False}),
                    "double_img_attn":         ("BOOLEAN", {"default": False}),
                    "double_img_attn_gated":   ("BOOLEAN", {"default": False}),
                    "double_img":              ("BOOLEAN", {"default": False}),
                    "double_img_norm1":        ("BOOLEAN", {"default": False}),
                    "double_img_ff_i":         ("BOOLEAN", {"default": False}),

                    "double_txt_io":           ("BOOLEAN", {"default": False}),
                    "double_txt_norm0":        ("BOOLEAN", {"default": False}),
                    "double_txt_attn":         ("BOOLEAN", {"default": False}),
                    "double_txt_attn_gated":   ("BOOLEAN", {"default": False}),
                    "double_txt":              ("BOOLEAN", {"default": False}),
                    "double_txt_norm1":        ("BOOLEAN", {"default": False}),
                    "double_txt_ff_t":         ("BOOLEAN", {"default": False}),

                    "single_img_io":           ("BOOLEAN", {"default": False}),
                    "single_img_norm0":        ("BOOLEAN", {"default": False}),
                    "single_img_attn":         ("BOOLEAN", {"default": False}),
                    "single_img_attn_gated":   ("BOOLEAN", {"default": False}),
                    "single_img":              ("BOOLEAN", {"default": False}),
                    "single_img_norm1":        ("BOOLEAN", {"default": False}),
                    "single_img_ff_i":         ("BOOLEAN", {"default": False}),
                    
                    "attn_img_q_norm"       : ("BOOLEAN", {"default": False}),
                    "attn_img_k_norm"       : ("BOOLEAN", {"default": False}),
                    "attn_img_v_norm"       : ("BOOLEAN", {"default": False}),
                    "attn_txt_q_norm"       : ("BOOLEAN", {"default": False}),
                    "attn_txt_k_norm"       : ("BOOLEAN", {"default": False}),
                    "attn_txt_v_norm"       : ("BOOLEAN", {"default": False}),
                    "attn_img_double"       : ("BOOLEAN", {"default": False}),
                    "attn_txt_double"       : ("BOOLEAN", {"default": False}),
                    "attn_img_single"       : ("BOOLEAN", {"default": False}),
                    
                    "start_step":       ("INT",      {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step":         ("INT",      {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":      ("BOOLEAN",  {"default": False}),
                    },
                "optional": 
                    {
                    "guide":            ("LATENT", ),
                    "mask":             ("MASK", ),
                    "weights":          ("SIGMAS", ),
                    "guides":           ("GUIDES", ),
                    }  
                }
    
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight           = 1.0,
            weight_scheduler = "constant",
            mode             = "scattersort",
            noise_mode       = "smart",
            double_weights   = "0.1",
            single_weights   = "0.0", 
            double_blocks    = "all",
            single_blocks    = "all", 
            start_step       = 0,
            end_step         = 15,
            invert_mask      = False,

            moe_gate                = False,
            moe_ff                  = False,
            ff                      = False,
            shared_experts          = False,

            double_img_io           = False,
            double_img_norm0        = False,
            double_img_attn         = False,
            double_img_norm1        = False,
            double_img_attn_gated   = False,
            double_img              = False,
            double_img_ff_i         = False,

            double_txt_io           = False,
            double_txt_norm0        = False,
            double_txt_attn         = False,
            double_txt_attn_gated   = False,
            double_txt              = False,
            double_txt_norm1        = False,
            double_txt_ff_t         = False,

            single_img_io           = False,
            single_img_norm0        = False,
            single_img_attn         = False,
            single_img_attn_gated   = False,
            single_img              = False,
            single_img_norm1        = False,
            single_img_ff_i         = False,
            
            attn_img_q_norm         = False,
            attn_img_k_norm         = False,
            attn_img_v_norm         = False,
            attn_txt_q_norm         = False,
            attn_txt_k_norm         = False,
            attn_txt_v_norm         = False,
            attn_img_single         = False,
            attn_img_double         = False,
            attn_txt_double         = False,

            guide            = None,
            mask             = None,
            weights          = None,
            guides           = None,
            ):
        
        default_dtype = torch.float64
        
        mask = 1-mask if mask is not None else None
        
        double_weights = parse_range_string(double_weights)
        single_weights = parse_range_string(single_weights)
        
        if len(double_weights) == 0:
            double_weights.append(0.0)
        if len(single_weights) == 0:
            single_weights.append(0.0)
            
        if len(double_weights) == 1:
            double_weights = double_weights * 100
        if len(single_weights) == 1:
            single_weights = single_weights * 100
            
        if type(double_weights[0]) == int:
            double_weights = [float(val) for val in double_weights]
        if type(single_weights[0]) == int:
            single_weights = [float(val) for val in single_weights]
        
        if double_blocks == "all":
            double_blocks  = [val for val in range(100)]
            if len(double_weights) == 1:
                double_weights = [double_weights[0]] * 100
        else:
            double_blocks  = parse_range_string(double_blocks)
            
            weights_expanded = [0.0] * 100
            for b, w in zip(double_blocks, double_weights):
                weights_expanded[b] = w
            double_weights = weights_expanded
            
        
        if single_blocks == "all":
            single_blocks = [val for val in range(100)]
            if len(single_weights) == 1:
                single_weights = [single_weights[0]] * 100
        else:
            single_blocks  = parse_range_string(single_blocks)
            
            weights_expanded = [0.0] * 100
            for b, w in zip(single_blocks, single_weights):
                weights_expanded[b] = w
            single_weights = weights_expanded
        
        
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if guide is not None:
            raw_x = guide.get('state_info', {}).get('raw_x', None)
            if raw_x is not None:
                guide          = {'samples': guide['state_info']['raw_x'].clone()}
            else:
                guide          = {'samples': guide['samples'].clone()}
        
        if weight_scheduler == "constant": # and weights == None: 
            weights = initialize_or_scale(None, weight, end_step).to(default_dtype)
            prepend = torch.zeros(start_step).to(weights)
            weights = torch.cat([prepend, weights])
            weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        guides = copy.deepcopy(guides) if guides is not None else {}
        
        guides['weight_adain']           = weight
        guides['weights_adain']          = weights
        
        guides['blocks_adain_mmdit'] = {
            "double_weights": double_weights,
            "single_weights": single_weights,
            "double_blocks" : double_blocks,
            "single_blocks" : single_blocks,
        }
        guides['sort_and_scatter'] = {
            "mode"                  : mode,

            "moe_gate"              : moe_gate,
            "moe_ff"                : moe_ff,

            "ff"                    : ff,
            "shared_experts"        : shared_experts,

            "double_img_io"         : double_img_io,
            "double_img_norm0"      : double_img_norm0,
            "double_img_attn"       : double_img_attn,
            "double_img_norm1"      : double_img_norm1,
            "double_img_attn_gated" : double_img_attn_gated,
            "double_img"            : double_img,
            "double_img_ff_i"       : double_img_ff_i,

            "double_txt_io"         : double_txt_io,
            "double_txt_norm0"      : double_txt_norm0,
            "double_txt_attn"       : double_txt_attn,
            "double_txt_attn_gated" : double_txt_attn_gated,
            "double_txt"            : double_txt,
            "double_txt_norm1"      : double_txt_norm1,
            "double_txt_ff_t"       : double_txt_ff_t,

            "single_img_io"         : single_img_io,
            "single_img_norm0"      : single_img_norm0,
            "single_img_attn"       : single_img_attn,
            "single_img_attn_gated" : single_img_attn_gated,
            "single_img"            : single_img,
            "single_img_norm1"      : single_img_norm1,
            "single_img_ff_i"       : single_img_ff_i,
            
            "attn_img_q_norm"       : attn_img_q_norm,
            "attn_img_k_norm"       : attn_img_k_norm,
            "attn_img_v_norm"       : attn_img_v_norm,
            "attn_txt_q_norm"       : attn_txt_q_norm,
            "attn_txt_k_norm"       : attn_txt_k_norm,
            "attn_txt_v_norm"       : attn_txt_v_norm,
            "attn_img_single"       : attn_img_single,
            "attn_img_double"       : attn_img_double,
        }
        
        guides['guide_adain']            = guide
        guides['mask_adain']             = mask

        guides['weight_scheduler_adain'] = weight_scheduler
        guides['start_step_adain']       = start_step
        guides['end_step_adain']         = end_step
        
        return (guides, )






class ClownGuides_Sync:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight_masked":               ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "weight_unmasked":             ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "weight_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "weight_scheduler_unmasked":   (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "weight_start_step_masked":           ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "weight_start_step_unmasked":         ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "weight_end_step_masked":             ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "weight_end_step_unmasked":           ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    
                    "sync_masked":               ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "sync_unmasked":             ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "sync_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "sync_scheduler_unmasked":   (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "sync_start_step_masked":           ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "sync_start_step_unmasked":         ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "sync_end_step_masked":             ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "sync_end_step_unmasked":           ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":                 ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide_masked":                ("LATENT", ),
                    "guide_unmasked":              ("LATENT", ),
                    "mask":                        ("MASK", ),
                    "weights_masked":              ("SIGMAS", ),
                    "weights_unmasked":            ("SIGMAS", ),
                    "syncs_masked":              ("SIGMAS", ),
                    "syncs_unmasked":            ("SIGMAS", ),
                    }  
                }
        
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"
    EXPERIMENTAL = True

    def main(self,
            weight_masked              = 0.0,
            weight_unmasked            = 0.0,
            weight_scheduler_masked    = "constant",
            weight_scheduler_unmasked  = "constant",
            weight_start_step_masked   = 0,
            weight_start_step_unmasked = 0,
            weight_end_step_masked     = 30,
            weight_end_step_unmasked   = 30,

            sync_masked                = 0.0,
            sync_unmasked              = 0.0,
            sync_scheduler_masked      = "constant",
            sync_scheduler_unmasked    = "constant",
            sync_start_step_masked     = 0,
            sync_start_step_unmasked   = 0,
            sync_end_step_masked       = 30,
            sync_end_step_unmasked     = 30,

            guide_masked               = None,
            guide_unmasked             = None,
            
            weights_masked             = None,
            weights_unmasked           = None,
            syncs_masked               = None,
            syncs_unmasked             = None,
            mask                       = None,
            unmask                     = None,
            invert_mask                = False,
            
            guide_mode                 = "sync",
            channelwise_mode           = False,
            projection_mode            = False,
            
            cutoff_masked              = 1.0,
            cutoff_unmasked            = 1.0,
            ):

        default_dtype = torch.float64
        
        if weight_end_step_masked   == -1:
            weight_end_step_masked   = MAX_STEPS
        if weight_end_step_unmasked == -1:
            weight_end_step_unmasked = MAX_STEPS
            
        if sync_end_step_masked   == -1:
            sync_end_step_masked   = MAX_STEPS
        if sync_end_step_unmasked == -1:
            sync_end_step_unmasked = MAX_STEPS
        
        if guide_masked is None:
            weight_scheduler_masked = "constant"
            weight_start_step_masked       = 0
            weight_end_step_masked         = 30
            weight_masked           = 0.0
            weights_masked          = None
            
            sync_scheduler_masked = "constant"
            sync_start_step_masked       = 0
            sync_end_step_masked         = 30
            sync_masked           = 0.0
            syncs_masked          = None
        
        if guide_unmasked is None:
            weight_scheduler_unmasked = "constant"
            weight_start_step_unmasked       = 0
            weight_end_step_unmasked         = 30
            weight_unmasked           = 0.0
            weights_unmasked          = None
        
            sync_scheduler_unmasked = "constant"
            sync_start_step_unmasked       = 0
            sync_end_step_unmasked         = 30
            sync_unmasked           = 0.0
            syncs_unmasked          = None
        
        if guide_masked is not None:
            raw_x = guide_masked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_masked   = {'samples': guide_masked['state_info']['raw_x'].clone()}
            else:
                guide_masked   = {'samples': guide_masked['samples'].clone()}
        
        if guide_unmasked is not None:
            raw_x = guide_unmasked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_unmasked = {'samples': guide_unmasked['state_info']['raw_x'].clone()}
            else:
                guide_unmasked = {'samples': guide_unmasked['samples'].clone()}
        
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
            weights_masked = initialize_or_scale(None, weight_masked, weight_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(weight_start_step_masked, dtype=default_dtype, device=weights_masked.device)
            weights_masked = torch.cat((prepend, weights_masked), dim=0)
            weights_masked = F.pad(weights_masked, (0, MAX_STEPS), value=0.0)
        
        if weight_scheduler_unmasked == "constant" and weights_unmasked == None: 
            weights_unmasked = initialize_or_scale(None, weight_unmasked, weight_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(weight_start_step_unmasked, dtype=default_dtype, device=weights_unmasked.device)
            weights_unmasked = torch.cat((prepend, weights_unmasked), dim=0)
            weights_unmasked = F.pad(weights_unmasked, (0, MAX_STEPS), value=0.0)
        
        # Values for the sync scheduler will be inverted in rk_guide_func_beta.py as it's easier to understand:
        # makes it so that a sync weight of 1.0 = full guide strength (which previously was 0.0)
        if sync_scheduler_masked == "constant" and syncs_masked == None: 
            syncs_masked = initialize_or_scale(None, sync_masked, sync_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(sync_start_step_masked, dtype=default_dtype, device=syncs_masked.device)
            syncs_masked = torch.cat((prepend, syncs_masked), dim=0)
            syncs_masked = F.pad(syncs_masked, (0, MAX_STEPS), value=0.0)
        
        if sync_scheduler_unmasked == "constant" and syncs_unmasked == None: 
            syncs_unmasked = initialize_or_scale(None, sync_unmasked, sync_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(sync_start_step_unmasked, dtype=default_dtype, device=syncs_unmasked.device)
            syncs_unmasked = torch.cat((prepend, syncs_unmasked), dim=0)
            syncs_unmasked = F.pad(syncs_unmasked, (0, MAX_STEPS), value=0.0)
        
        guides = {
            "guide_mode"                : guide_mode,

            "guide_masked"              : guide_masked,
            "guide_unmasked"            : guide_unmasked,
            "mask"                      : mask,
            "unmask"                    : unmask,

            "weight_masked"             : weight_masked,
            "weight_unmasked"           : weight_unmasked,
            "weight_scheduler_masked"   : weight_scheduler_masked,
            "weight_scheduler_unmasked" : weight_scheduler_unmasked,
            "start_step_masked"         : weight_start_step_masked,
            "start_step_unmasked"       : weight_start_step_unmasked,
            "end_step_masked"           : weight_end_step_masked,
            "end_step_unmasked"         : weight_end_step_unmasked,
            
            "weights_masked"            : weights_masked,
            "weights_unmasked"          : weights_unmasked,
            
            "weight_masked_sync"             : sync_masked,
            "weight_unmasked_sync"           : sync_unmasked,
            "weight_scheduler_masked_sync"   : sync_scheduler_masked,
            "weight_scheduler_unmasked_sync" : sync_scheduler_unmasked,
            "start_step_masked_sync"         : sync_start_step_masked,
            "start_step_unmasked_sync"       : sync_start_step_unmasked,
            "end_step_masked_sync"           : sync_end_step_masked,
            "end_step_unmasked_sync"         : sync_end_step_unmasked,
            
            "weights_masked_sync"            : syncs_masked,
            "weights_unmasked_sync"          : syncs_unmasked,
            
            "cutoff_masked"             : cutoff_masked,
            "cutoff_unmasked"           : cutoff_unmasked
        }
        
        
        return (guides, )






class ClownGuides_Sync_Advanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "weight_masked":               ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "weight_unmasked":             ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "weight_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "weight_scheduler_unmasked":   (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "weight_start_step_masked":    ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "weight_start_step_unmasked":  ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "weight_end_step_masked":      ("INT",                                       {"default": 30,   "min": -1,      "max": 10000}),
                    "weight_end_step_unmasked":    ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    
                    "sync_masked":                 ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "sync_unmasked":               ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "sync_scheduler_masked":       (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "sync_scheduler_unmasked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "sync_start_step_masked":      ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "sync_start_step_unmasked":    ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "sync_end_step_masked":        ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    "sync_end_step_unmasked":      ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    
                    "drift_x_data":                ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_x_sync":                ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_x_masked":              ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_x_unmasked":            ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "drift_x_scheduler_masked":    (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "drift_x_scheduler_unmasked":  (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "drift_x_start_step_masked":   ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "drift_x_start_step_unmasked": ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "drift_x_end_step_masked":     ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    "drift_x_end_step_unmasked":   ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    
                    "drift_y_data":                ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_y_sync":                ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_y_guide":               ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_y_masked":              ("FLOAT",                                     {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "drift_y_unmasked":            ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "drift_y_scheduler_masked":    (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "drift_y_scheduler_unmasked":  (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "drift_y_start_step_masked":   ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "drift_y_start_step_unmasked": ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "drift_y_end_step_masked":     ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    "drift_y_end_step_unmasked":   ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    
                    "lure_x_masked":               ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "lure_x_unmasked":             ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "lure_x_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "lure_x_scheduler_unmasked":   (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "lure_x_start_step_masked":    ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "lure_x_start_step_unmasked":  ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "lure_x_end_step_masked":      ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    "lure_x_end_step_unmasked":    ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    
                    "lure_y_masked":               ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "lure_y_unmasked":             ("FLOAT",                                     {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "lure_y_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "lure_y_scheduler_unmasked":   (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "lure_y_start_step_masked":    ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "lure_y_start_step_unmasked":  ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "lure_y_end_step_masked":      ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    "lure_y_end_step_unmasked":    ("INT",                                       {"default": -1,   "min": -1,      "max": 10000}),
                    
                    "lure_iter":          ("INT",                                       {"default": 0,   "min": 0,      "max": 10000}),
                    "lure_sequence":      (["x -> y", "y -> x", "xy -> xy"],                                   {"default": "y -> x"}),
                    
                    "invert_mask":        ("BOOLEAN",                                   {"default": False}),
                    "invert_mask_sync":   ("BOOLEAN",                                   {"default": False}),
                    "invert_mask_drift_x": ("BOOLEAN",                                  {"default": False}),
                    "invert_mask_drift_y": ("BOOLEAN",                                  {"default": False}),
                    "invert_mask_lure_x": ("BOOLEAN",                                   {"default": False}),
                    "invert_mask_lure_y": ("BOOLEAN",                                   {"default": False}),

                    },
                "optional": 
                    {
                    "guide_masked":       ("LATENT", ),
                    "guide_unmasked":     ("LATENT", ),
                    "mask":               ("MASK", ),
                    "mask_sync":          ("MASK", ),
                    "mask_drift_x":        ("MASK", ),
                    "mask_drift_y":        ("MASK", ),
                    "mask_lure_x":        ("MASK", ),
                    "mask_lure_y":        ("MASK", ),
                    "weights_masked":     ("SIGMAS", ),
                    "weights_unmasked":   ("SIGMAS", ),
                    "syncs_masked":       ("SIGMAS", ),
                    "syncs_unmasked":     ("SIGMAS", ),
                    "drift_xs_masked":     ("SIGMAS", ),
                    "drift_xs_unmasked":   ("SIGMAS", ),
                    "drift_ys_masked":     ("SIGMAS", ),
                    "drift_ys_unmasked":   ("SIGMAS", ),
                    "lure_xs_masked":     ("SIGMAS", ),
                    "lure_xs_unmasked":   ("SIGMAS", ),
                    "lure_ys_masked":     ("SIGMAS", ),
                    "lure_ys_unmasked":   ("SIGMAS", ),
                    }  
                }
        
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"
    EXPERIMENTAL = True

    def main(self,
            weight_masked              = 0.0,
            weight_unmasked            = 0.0,
            weight_scheduler_masked    = "constant",
            weight_scheduler_unmasked  = "constant",
            weight_start_step_masked   = 0,
            weight_start_step_unmasked = 0,
            weight_end_step_masked     = 30,
            weight_end_step_unmasked   = 30,

            sync_masked                = 0.0,
            sync_unmasked              = 0.0,
            sync_scheduler_masked      = "constant",
            sync_scheduler_unmasked    = "constant",
            sync_start_step_masked     = 0,
            sync_start_step_unmasked   = 0,
            sync_end_step_masked       = 30,
            sync_end_step_unmasked     = 30,
            
            drift_x_data = 0.0,
            drift_x_sync = 0.0,
            drift_y_data = 0.0,
            drift_y_sync = 0.0,
            drift_y_guide = 0.0,

            drift_x_masked                = 0.0,
            drift_x_unmasked              = 0.0,
            drift_x_scheduler_masked      = "constant",
            drift_x_scheduler_unmasked    = "constant",
            drift_x_start_step_masked     = 0,
            drift_x_start_step_unmasked   = 0,
            drift_x_end_step_masked       = 30,
            drift_x_end_step_unmasked     = 30,
            
            drift_y_masked                = 0.0,
            drift_y_unmasked              = 0.0,
            drift_y_scheduler_masked      = "constant",
            drift_y_scheduler_unmasked    = "constant",
            drift_y_start_step_masked     = 0,
            drift_y_start_step_unmasked   = 0,
            drift_y_end_step_masked       = 30,
            drift_y_end_step_unmasked     = 30,

            lure_x_masked                = 0.0,
            lure_x_unmasked              = 0.0,
            lure_x_scheduler_masked      = "constant",
            lure_x_scheduler_unmasked    = "constant",
            lure_x_start_step_masked     = 0,
            lure_x_start_step_unmasked   = 0,
            lure_x_end_step_masked       = 30,
            lure_x_end_step_unmasked     = 30,
            
            lure_y_masked                = 0.0,
            lure_y_unmasked              = 0.0,
            lure_y_scheduler_masked      = "constant",
            lure_y_scheduler_unmasked    = "constant",
            lure_y_start_step_masked     = 0,
            lure_y_start_step_unmasked   = 0,
            lure_y_end_step_masked       = 30,
            lure_y_end_step_unmasked     = 30,

            guide_masked               = None,
            guide_unmasked             = None,
            
            weights_masked             = None,
            weights_unmasked           = None,
            syncs_masked               = None,
            syncs_unmasked             = None,
            drift_xs_masked             = None,
            drift_xs_unmasked           = None,
            drift_ys_masked             = None,
            drift_ys_unmasked           = None,
            lure_xs_masked             = None,
            lure_xs_unmasked           = None,
            lure_ys_masked             = None,
            lure_ys_unmasked           = None,
            
            lure_iter                  = 0,
            lure_sequence              = "x -> y",
            
            mask                       = None,
            unmask                     = None,
            mask_sync                  = None,
            mask_drift_x                = None,
            mask_drift_y                = None,
            mask_lure_x                = None,
            mask_lure_y                = None,

            invert_mask                = False,
            invert_mask_sync           = False,
            invert_mask_drift_x         = False,
            invert_mask_drift_y         = False,
            invert_mask_lure_x         = False,
            invert_mask_lure_y         = False,
            
            guide_mode                 = "sync",
            channelwise_mode           = False,
            projection_mode            = False,
            
            cutoff_masked              = 1.0,
            cutoff_unmasked            = 1.0,
            ):

        default_dtype = torch.float64
        
        if weight_end_step_masked   == -1:
            weight_end_step_masked   = MAX_STEPS
        if weight_end_step_unmasked == -1:
            weight_end_step_unmasked = MAX_STEPS
        
        if sync_end_step_masked   == -1:
            sync_end_step_masked   = MAX_STEPS
        if sync_end_step_unmasked == -1:
            sync_end_step_unmasked = MAX_STEPS
        
        if drift_x_end_step_masked   == -1:
            drift_x_end_step_masked   = MAX_STEPS
        if drift_x_end_step_unmasked == -1:
            drift_x_end_step_unmasked = MAX_STEPS
        if drift_y_end_step_masked   == -1:
            drift_y_end_step_masked   = MAX_STEPS
        if drift_y_end_step_unmasked == -1:
            drift_y_end_step_unmasked = MAX_STEPS
        
        if lure_x_end_step_masked   == -1:
            lure_x_end_step_masked   = MAX_STEPS
        if lure_x_end_step_unmasked == -1:
            lure_x_end_step_unmasked = MAX_STEPS
        if lure_y_end_step_masked   == -1:
            lure_y_end_step_masked   = MAX_STEPS
        if lure_y_end_step_unmasked == -1:
            lure_y_end_step_unmasked = MAX_STEPS
        
        
        
        if guide_masked is None:
            weight_scheduler_masked = "constant"
            weight_start_step_masked       = 0
            weight_end_step_masked         = 30
            weight_masked           = 0.0
            weights_masked          = None
            
            sync_scheduler_masked = "constant"
            sync_start_step_masked       = 0
            sync_end_step_masked         = 30
            sync_masked           = 0.0
            syncs_masked          = None
        
            drift_x_scheduler_masked = "constant"
            drift_x_start_step_masked       = 0
            drift_x_end_step_masked         = 30
            drift_x_masked           = 0.0
            drift_xs_masked          = None
        
            drift_y_scheduler_masked = "constant"
            drift_y_start_step_masked       = 0
            drift_y_end_step_masked         = 30
            drift_y_masked           = 0.0
            drift_ys_masked          = None
        
            lure_x_scheduler_masked = "constant"
            lure_x_start_step_masked       = 0
            lure_x_end_step_masked         = 30
            lure_x_masked           = 0.0
            lure_xs_masked          = None
        
            lure_y_scheduler_masked = "constant"
            lure_y_start_step_masked       = 0
            lure_y_end_step_masked         = 30
            lure_y_masked           = 0.0
            lure_ys_masked          = None
        
        if guide_unmasked is None:
            weight_scheduler_unmasked = "constant"
            weight_start_step_unmasked       = 0
            weight_end_step_unmasked         = 30
            weight_unmasked           = 0.0
            weights_unmasked          = None
        
            sync_scheduler_unmasked = "constant"
            sync_start_step_unmasked       = 0
            sync_end_step_unmasked         = 30
            sync_unmasked           = 0.0
            syncs_unmasked          = None
        
            drift_x_scheduler_unmasked = "constant"
            drift_x_start_step_unmasked       = 0
            drift_x_end_step_unmasked         = 30
            drift_x_unmasked           = 0.0
            drift_xs_unmasked          = None
        
            drift_y_scheduler_unmasked = "constant"
            drift_y_start_step_unmasked       = 0
            drift_y_end_step_unmasked         = 30
            drift_y_unmasked           = 0.0
            drift_ys_unmasked          = None
        
            lure_x_scheduler_unmasked = "constant"
            lure_x_start_step_unmasked       = 0
            lure_x_end_step_unmasked         = 30
            lure_x_unmasked           = 0.0
            lure_xs_unmasked          = None
        
            lure_y_scheduler_unmasked = "constant"
            lure_y_start_step_unmasked       = 0
            lure_y_end_step_unmasked         = 30
            lure_y_unmasked           = 0.0
            lure_ys_unmasked          = None
        
        
        if guide_masked is not None:
            raw_x = guide_masked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_masked   = {'samples': guide_masked['state_info']['raw_x'].clone()}
            else:
                guide_masked   = {'samples': guide_masked['samples'].clone()}
        
        if guide_unmasked is not None:
            raw_x = guide_unmasked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_unmasked = {'samples': guide_unmasked['state_info']['raw_x'].clone()}
            else:
                guide_unmasked = {'samples': guide_unmasked['samples'].clone()}
        
        if invert_mask and mask is not None:
            mask = 1-mask
        if invert_mask_sync and mask_sync is not None:
            mask_sync = 1-mask_sync
        if invert_mask_drift_x and mask_drift_x is not None:
            mask_drift_x = 1-mask_drift_x
        if invert_mask_drift_y and mask_drift_y is not None:
            mask_drift_y = 1-mask_drift_y
        if invert_mask_lure_x and mask_lure_x is not None:
            mask_lure_x = 1-mask_lure_x
        if invert_mask_lure_y and mask_lure_y is not None:
            mask_lure_y = 1-mask_lure_y
        
        if projection_mode:
            guide_mode = guide_mode + "_projection"
        
        if channelwise_mode:
            guide_mode = guide_mode + "_cw"
            
        if guide_mode == "unsample_cw":
            guide_mode = "unsample"
        if guide_mode == "resample_cw":
            guide_mode = "resample"
        
        if weight_scheduler_masked == "constant" and weights_masked == None: 
            weights_masked = initialize_or_scale(None, weight_masked, weight_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(weight_start_step_masked, dtype=default_dtype, device=weights_masked.device)
            weights_masked = torch.cat((prepend, weights_masked), dim=0)
            weights_masked = F.pad(weights_masked, (0, MAX_STEPS), value=0.0)
        
        if weight_scheduler_unmasked == "constant" and weights_unmasked == None: 
            weights_unmasked = initialize_or_scale(None, weight_unmasked, weight_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(weight_start_step_unmasked, dtype=default_dtype, device=weights_unmasked.device)
            weights_unmasked = torch.cat((prepend, weights_unmasked), dim=0)
            weights_unmasked = F.pad(weights_unmasked, (0, MAX_STEPS), value=0.0)
        
        # Values for the sync scheduler will be inverted in rk_guide_func_beta.py as it's easier to understand:
        # makes it so that a sync weight of 1.0 = full guide strength (which previously was 0.0)
        if sync_scheduler_masked == "constant" and syncs_masked == None: 
            syncs_masked = initialize_or_scale(None, sync_masked, sync_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(sync_start_step_masked, dtype=default_dtype, device=syncs_masked.device)
            syncs_masked = torch.cat((prepend, syncs_masked), dim=0)
            syncs_masked = F.pad(syncs_masked, (0, MAX_STEPS), value=0.0)
        
        if sync_scheduler_unmasked == "constant" and syncs_unmasked == None: 
            syncs_unmasked = initialize_or_scale(None, sync_unmasked, sync_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(sync_start_step_unmasked, dtype=default_dtype, device=syncs_unmasked.device)
            syncs_unmasked = torch.cat((prepend, syncs_unmasked), dim=0)
            syncs_unmasked = F.pad(syncs_unmasked, (0, MAX_STEPS), value=0.0)
        
        if drift_x_scheduler_masked == "constant" and drift_xs_masked == None: 
            drift_xs_masked = initialize_or_scale(None, drift_x_masked, drift_x_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(drift_x_start_step_masked, dtype=default_dtype, device=drift_xs_masked.device)
            drift_xs_masked = torch.cat((prepend, drift_xs_masked), dim=0)
            drift_xs_masked = F.pad(drift_xs_masked, (0, MAX_STEPS), value=0.0)
        
        if drift_x_scheduler_unmasked == "constant" and drift_xs_unmasked == None: 
            drift_xs_unmasked = initialize_or_scale(None, drift_x_unmasked, drift_x_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(drift_x_start_step_unmasked, dtype=default_dtype, device=drift_xs_unmasked.device)
            drift_xs_unmasked = torch.cat((prepend, drift_xs_unmasked), dim=0)
            drift_xs_unmasked = F.pad(drift_xs_unmasked, (0, MAX_STEPS), value=0.0)
        
        if drift_y_scheduler_masked == "constant" and drift_ys_masked == None: 
            drift_ys_masked = initialize_or_scale(None, drift_y_masked, drift_y_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(drift_y_start_step_masked, dtype=default_dtype, device=drift_ys_masked.device)
            drift_ys_masked = torch.cat((prepend, drift_ys_masked), dim=0)
            drift_ys_masked = F.pad(drift_ys_masked, (0, MAX_STEPS), value=0.0)
        
        if drift_y_scheduler_unmasked == "constant" and drift_ys_unmasked == None: 
            drift_ys_unmasked = initialize_or_scale(None, drift_y_unmasked, drift_y_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(drift_y_start_step_unmasked, dtype=default_dtype, device=drift_ys_unmasked.device)
            drift_ys_unmasked = torch.cat((prepend, drift_ys_unmasked), dim=0)
            drift_ys_unmasked = F.pad(drift_ys_unmasked, (0, MAX_STEPS), value=0.0)
        
        if lure_x_scheduler_masked == "constant" and lure_xs_masked == None: 
            lure_xs_masked = initialize_or_scale(None, lure_x_masked, lure_x_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(lure_x_start_step_masked, dtype=default_dtype, device=lure_xs_masked.device)
            lure_xs_masked = torch.cat((prepend, lure_xs_masked), dim=0)
            lure_xs_masked = F.pad(lure_xs_masked, (0, MAX_STEPS), value=0.0)
        
        if lure_x_scheduler_unmasked == "constant" and lure_xs_unmasked == None: 
            lure_xs_unmasked = initialize_or_scale(None, lure_x_unmasked, lure_x_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(lure_x_start_step_unmasked, dtype=default_dtype, device=lure_xs_unmasked.device)
            lure_xs_unmasked = torch.cat((prepend, lure_xs_unmasked), dim=0)
            lure_xs_unmasked = F.pad(lure_xs_unmasked, (0, MAX_STEPS), value=0.0)
        
        if lure_y_scheduler_masked == "constant" and lure_ys_masked == None: 
            lure_ys_masked = initialize_or_scale(None, lure_y_masked, lure_y_end_step_masked).to(default_dtype)
            prepend      = torch.zeros(lure_y_start_step_masked, dtype=default_dtype, device=lure_ys_masked.device)
            lure_ys_masked = torch.cat((prepend, lure_ys_masked), dim=0)
            lure_ys_masked = F.pad(lure_ys_masked, (0, MAX_STEPS), value=0.0)
        
        if lure_y_scheduler_unmasked == "constant" and lure_ys_unmasked == None: 
            lure_ys_unmasked = initialize_or_scale(None, lure_y_unmasked, lure_y_end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(lure_y_start_step_unmasked, dtype=default_dtype, device=lure_ys_unmasked.device)
            lure_ys_unmasked = torch.cat((prepend, lure_ys_unmasked), dim=0)
            lure_ys_unmasked = F.pad(lure_ys_unmasked, (0, MAX_STEPS), value=0.0)
        
        
        guides = {
            "guide_mode"                        : guide_mode,
        
            "guide_masked"                      : guide_masked,
            "guide_unmasked"                    : guide_unmasked,
            "mask"                              : mask,
            "unmask"                            : unmask,
            "mask_sync"                         : mask_sync,
            "mask_lure_x"                       : mask_lure_x,
            "mask_lure_y"                       : mask_lure_y,
        
            "weight_masked"                     : weight_masked,
            "weight_unmasked"                   : weight_unmasked,
            "weight_scheduler_masked"           : weight_scheduler_masked,
            "weight_scheduler_unmasked"         : weight_scheduler_unmasked,
            "start_step_masked"                 : weight_start_step_masked,
            "start_step_unmasked"               : weight_start_step_unmasked,
            "end_step_masked"                   : weight_end_step_masked,
            "end_step_unmasked"                 : weight_end_step_unmasked,
            
            "weights_masked"                    : weights_masked,
            "weights_unmasked"                  : weights_unmasked,
            
            "weight_masked_sync"                : sync_masked,
            "weight_unmasked_sync"              : sync_unmasked,
            "weight_scheduler_masked_sync"      : sync_scheduler_masked,
            "weight_scheduler_unmasked_sync"    : sync_scheduler_unmasked,
            "start_step_masked_sync"            : sync_start_step_masked,
            "start_step_unmasked_sync"          : sync_start_step_unmasked,
            "end_step_masked_sync"              : sync_end_step_masked,
            "end_step_unmasked_sync"            : sync_end_step_unmasked,
            
            "weights_masked_sync"               : syncs_masked,
            "weights_unmasked_sync"             : syncs_unmasked,
            
            "drift_x_data"                      : drift_x_data,
            "drift_x_sync"                      : drift_x_sync,
            "drift_y_data"                      : drift_y_data,
            "drift_y_sync"                      : drift_y_sync,
            "drift_y_guide"                     : drift_y_guide,
            
            "weight_masked_drift_x"             : drift_x_masked,
            "weight_unmasked_drift_x"           : drift_x_unmasked,
            "weight_scheduler_masked_drift_x"   : drift_x_scheduler_masked,
            "weight_scheduler_unmasked_drift_x" : drift_x_scheduler_unmasked,
            "start_step_masked_drift_x"         : drift_x_start_step_masked,
            "start_step_unmasked_drift_x"       : drift_x_start_step_unmasked,
            "end_step_masked_drift_x"           : drift_x_end_step_masked,
            "end_step_unmasked_drift_x"         : drift_x_end_step_unmasked,
            
            "weights_masked_drift_x"            : drift_xs_masked,
            "weights_unmasked_drift_x"          : drift_xs_unmasked,
            
            
            "weight_masked_drift_y"             : drift_y_masked,
            "weight_unmasked_drift_y"           : drift_y_unmasked,
            "weight_scheduler_masked_drift_y"   : drift_y_scheduler_masked,
            "weight_scheduler_unmasked_drift_y" : drift_y_scheduler_unmasked,
            "start_step_masked_drift_y"         : drift_y_start_step_masked,
            "start_step_unmasked_drift_y"       : drift_y_start_step_unmasked,
            "end_step_masked_drift_y"           : drift_y_end_step_masked,
            "end_step_unmasked_drift_y"         : drift_y_end_step_unmasked,
            
            "weights_masked_drift_y"            : drift_ys_masked,
            "weights_unmasked_drift_y"          : drift_ys_unmasked,
            
            "weight_masked_lure_x"              : lure_x_masked,
            "weight_unmasked_lure_x"            : lure_x_unmasked,
            "weight_scheduler_masked_lure_x"    : lure_x_scheduler_masked,
            "weight_scheduler_unmasked_lure_x"  : lure_x_scheduler_unmasked,
            "start_step_masked_lure_x"          : lure_x_start_step_masked,
            "start_step_unmasked_lure_x"        : lure_x_start_step_unmasked,
            "end_step_masked_lure_x"            : lure_x_end_step_masked,
            "end_step_unmasked_lure_x"          : lure_x_end_step_unmasked,
            
            "weights_masked_lure_x"             : lure_xs_masked,
            "weights_unmasked_lure_x"           : lure_xs_unmasked,
            
            
            "weight_masked_lure_y"              : lure_y_masked,
            "weight_unmasked_lure_y"            : lure_y_unmasked,
            "weight_scheduler_masked_lure_y"    : lure_y_scheduler_masked,
            "weight_scheduler_unmasked_lure_y"  : lure_y_scheduler_unmasked,
            "start_step_masked_lure_y"          : lure_y_start_step_masked,
            "start_step_unmasked_lure_y"        : lure_y_start_step_unmasked,
            "end_step_masked_lure_y"            : lure_y_end_step_masked,
            "end_step_unmasked_lure_y"          : lure_y_end_step_unmasked,
            
            "weights_masked_lure_y"             : lure_ys_masked,
            "weights_unmasked_lure_y"           : lure_ys_unmasked,
            
            "sync_lure_iter"                    : lure_iter,
            "sync_lure_sequence"                : lure_sequence,

            "cutoff_masked"                     : cutoff_masked,
            "cutoff_unmasked"                   : cutoff_unmasked
        }
        
        
        return (guides, )









class ClownGuide_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "guide_mode":           (GUIDE_MODE_NAMES_BETA_SIMPLE,       {"default": 'epsilon',                                                      "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                    "channelwise_mode":     ("BOOLEAN",                                   {"default": True}),
                    "projection_mode":      ("BOOLEAN",                                   {"default": True}),
                    "weight":               ("FLOAT",                                     {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "cutoff":               ("FLOAT",                                     {"default": 1.0,  "min": 0.0,    "max": 1.0,   "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "start_step":           ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step":             ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":          ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide":                ("LATENT", ),
                    "mask":                 ("MASK", ),
                    "weights":              ("SIGMAS", ),
                    }  
                }
        
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight_scheduler          = "constant",
            weight_scheduler_unmasked = "constant",
            start_step                = 0,
            start_step_unmasked       = 0,
            end_step                  = 30,
            end_step_unmasked         = 30,
            cutoff                    = 1.0,
            cutoff_unmasked           = 1.0,
            guide                     = None,
            guide_unmasked            = None,
            weight                    = 0.0,
            weight_unmasked           = 0.0,

            guide_mode                = "epsilon",
            channelwise_mode          = False,
            projection_mode           = False,
            weights                   = None,
            weights_unmasked          = None,
            mask                      = None,
            unmask                    = None,
            invert_mask               = False,
            ):
        
        CG = ClownGuides_Beta()
        
        mask = 1-mask if mask is not None else None
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if guide is not None:
            raw_x = guide.get('state_info', {}).get('raw_x', None)
            
            if False: # raw_x is not None:
                guide          = {'samples': guide['state_info']['raw_x'].clone()}
            else:
                guide          = {'samples': guide['samples'].clone()}
                
        if guide_unmasked is not None:
            raw_x = guide_unmasked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_unmasked = {'samples': guide_unmasked['state_info']['raw_x'].clone()}
            else:
                guide_unmasked = {'samples': guide_unmasked['samples'].clone()}
        
        guides, = CG.main(
            weight_scheduler_masked   = weight_scheduler,
            weight_scheduler_unmasked = weight_scheduler_unmasked,
            start_step_masked         = start_step,
            start_step_unmasked       = start_step_unmasked,
            end_step_masked           = end_step,
            end_step_unmasked         = end_step_unmasked,
            cutoff_masked             = cutoff,
            cutoff_unmasked           = cutoff_unmasked,
            guide_masked              = guide,
            guide_unmasked            = guide_unmasked,
            weight_masked             = weight,
            weight_unmasked           = weight_unmasked,

            guide_mode                = guide_mode,
            channelwise_mode          = channelwise_mode,
            projection_mode           = projection_mode,
            weights_masked            = weights,
            weights_unmasked          = weights_unmasked,
            mask                      = mask,
            unmask                    = unmask,
            invert_mask               = invert_mask
        )

        return (guides, )


        #return (guides[0], )




class ClownGuides_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "guide_mode":                  (GUIDE_MODE_NAMES_BETA_SIMPLE,                {"default": 'epsilon',                                                      "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                    "channelwise_mode":            ("BOOLEAN",                                   {"default": True}),
                    "projection_mode":             ("BOOLEAN",                                   {"default": True}),
                    "weight_masked":               ("FLOAT",                                     {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "weight_unmasked":             ("FLOAT",                                     {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "cutoff_masked":               ("FLOAT",                                     {"default": 1.0,  "min": 0.0,    "max": 1.0,   "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "cutoff_unmasked":             ("FLOAT",                                     {"default": 1.0,  "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler_masked":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "weight_scheduler_unmasked":   (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "start_step_masked":           ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "start_step_unmasked":         ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step_masked":             ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "end_step_unmasked":           ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_mask":                 ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide_masked":                ("LATENT", ),
                    "guide_unmasked":              ("LATENT", ),
                    "mask":                        ("MASK", ),
                    "weights_masked":              ("SIGMAS", ),
                    "weights_unmasked":            ("SIGMAS", ),
                    }  
                }
        
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight_scheduler_masked   = "constant",
            weight_scheduler_unmasked = "constant",
            start_step_masked         = 0,
            start_step_unmasked       = 0,
            end_step_masked           = 30,
            end_step_unmasked         = 30,
            cutoff_masked             = 1.0,
            cutoff_unmasked           = 1.0,
            guide_masked              = None,
            guide_unmasked            = None,
            weight_masked             = 0.0,
            weight_unmasked           = 0.0,

            guide_mode                = "epsilon",
            channelwise_mode          = False,
            projection_mode           = False,
            weights_masked            = None,
            weights_unmasked          = None,
            mask                      = None,
            unmask                    = None,
            invert_mask               = False,
            ):

        default_dtype = torch.float64
        
        if end_step_masked   == -1:
            end_step_masked   = MAX_STEPS
        if end_step_unmasked == -1:
            end_step_unmasked = MAX_STEPS
        
        if guide_masked is None:
            weight_scheduler_masked = "constant"
            start_step_masked       = 0
            end_step_masked         = 30
            cutoff_masked           = 1.0
            guide_masked            = None
            weight_masked           = 0.0
            weights_masked          = None
            #mask                    = None
        
        if guide_unmasked is None:
            weight_scheduler_unmasked = "constant"
            start_step_unmasked       = 0
            end_step_unmasked         = 30
            cutoff_unmasked           = 1.0
            guide_unmasked            = None
            weight_unmasked           = 0.0
            weights_unmasked          = None
            #unmask                    = None
        
        if guide_masked is not None:
            raw_x = guide_masked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_masked   = {'samples': guide_masked['state_info']['raw_x'].clone()}
            else:
                guide_masked   = {'samples': guide_masked['samples'].clone()}
        
        if guide_unmasked is not None:
            raw_x = guide_unmasked.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_unmasked = {'samples': guide_unmasked['state_info']['raw_x'].clone()}
            else:
                guide_unmasked = {'samples': guide_unmasked['samples'].clone()}
        
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
            prepend      = torch.zeros(start_step_masked, dtype=default_dtype, device=weights_masked.device)
            weights_masked = torch.cat((prepend, weights_masked), dim=0)
            weights_masked = F.pad(weights_masked, (0, MAX_STEPS), value=0.0)
        
        if weight_scheduler_unmasked == "constant" and weights_unmasked == None: 
            weights_unmasked = initialize_or_scale(None, weight_unmasked, end_step_unmasked).to(default_dtype)
            prepend      = torch.zeros(start_step_unmasked, dtype=default_dtype, device=weights_unmasked.device)
            weights_unmasked = torch.cat((prepend, weights_unmasked), dim=0)
            weights_unmasked = F.pad(weights_unmasked, (0, MAX_STEPS), value=0.0)
        
        guides = {
            "guide_mode"                : guide_mode,
            "weight_masked"             : weight_masked,
            "weight_unmasked"           : weight_unmasked,
            "weights_masked"            : weights_masked,
            "weights_unmasked"          : weights_unmasked,
            "guide_masked"              : guide_masked,
            "guide_unmasked"            : guide_unmasked,
            "mask"                      : mask,
            "unmask"                    : unmask,

            "weight_scheduler_masked"   : weight_scheduler_masked,
            "weight_scheduler_unmasked" : weight_scheduler_unmasked,
            "start_step_masked"         : start_step_masked,
            "start_step_unmasked"       : start_step_unmasked,
            "end_step_masked"           : end_step_masked,
            "end_step_unmasked"         : end_step_unmasked,
            "cutoff_masked"             : cutoff_masked,
            "cutoff_unmasked"           : cutoff_unmasked
        }
        
        
        return (guides, )




class ClownGuidesAB_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "guide_mode":         (GUIDE_MODE_NAMES_BETA_SIMPLE,                {"default": 'epsilon',                                                      "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                    "channelwise_mode":   ("BOOLEAN",                                   {"default": False}),
                    "projection_mode":    ("BOOLEAN",                                   {"default": False}),
                    "weight_A":           ("FLOAT",                                     {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "weight_B":           ("FLOAT",                                     {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide_bkg."}),
                    "cutoff_A":           ("FLOAT",                                     {"default": 1.0,  "min": 0.0,    "max": 1.0,   "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "cutoff_B":           ("FLOAT",                                     {"default": 1.0,  "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler_A": (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "weight_scheduler_B": (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                    "start_step_A":       ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "start_step_B":       ("INT",                                       {"default": 0,    "min":  0,      "max": 10000}),
                    "end_step_A":         ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "end_step_B":         ("INT",                                       {"default": 15,   "min": -1,      "max": 10000}),
                    "invert_masks":       ("BOOLEAN",                                   {"default": False}),
                    },
                "optional": 
                    {
                    "guide_A":            ("LATENT", ),
                    "guide_B":            ("LATENT", ),
                    "mask_A":             ("MASK", ),
                    "mask_B":             ("MASK", ),
                    "weights_A":          ("SIGMAS", ),
                    "weights_B":          ("SIGMAS", ),
                    }  
                }
        
    RETURN_TYPES = ("GUIDES",)
    RETURN_NAMES = ("guides",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_extensions"

    def main(self,
            weight_scheduler_A = "constant",
            weight_scheduler_B = "constant",
            start_step_A       = 0,
            start_step_B       = 0,
            end_step_A         = 30,
            end_step_B         = 30,
            cutoff_A           = 1.0,
            cutoff_B           = 1.0,
            guide_A            = None,
            guide_B            = None,
            weight_A           = 0.0,
            weight_B           = 0.0,

            guide_mode         = "epsilon",
            channelwise_mode   = False,
            projection_mode    = False,
            weights_A          = None,
            weights_B          = None,
            mask_A             = None,
            mask_B             = None,
            invert_masks       : bool = False,
            ):
        
        default_dtype = torch.float64
        
        if end_step_A == -1:
            end_step_A = MAX_STEPS
        if end_step_B == -1:
            end_step_B = MAX_STEPS
        
        if guide_A is not None:
            raw_x = guide_A.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_A          = {'samples': guide_A['state_info']['raw_x'].clone()}
            else:
                guide_A          = {'samples': guide_A['samples'].clone()}
                
        if guide_B is not None:
            raw_x = guide_B.get('state_info', {}).get('raw_x', None)
            if False: #raw_x is not None:
                guide_B = {'samples': guide_B['state_info']['raw_x'].clone()}
            else:
                guide_B = {'samples': guide_B['samples'].clone()}
        
        if guide_A is None:
            guide_A  = guide_B
            guide_B  = None
            mask_A   = mask_B
            mask_B   = None
            weight_B = 0.0
            
        if guide_B is None:
            weight_B = 0.0
            
        if mask_A is None and mask_B is not None:
            mask_A = 1-mask_B
                        
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
            prepend      = torch.zeros(start_step_A, dtype=default_dtype, device=weights_A.device)
            weights_A = torch.cat((prepend, weights_A), dim=0)
            weights_A = F.pad(weights_A, (0, MAX_STEPS), value=0.0)
        
        if weight_scheduler_B == "constant" and weights_B == None: 
            weights_B = initialize_or_scale(None, weight_B, end_step_B).to(default_dtype)
            prepend      = torch.zeros(start_step_B, dtype=default_dtype, device=weights_B.device)
            weights_B = torch.cat((prepend, weights_B), dim=0)
            weights_B = F.pad(weights_B, (0, MAX_STEPS), value=0.0)
            
        if invert_masks:
            mask_A = 1-mask_A if mask_A is not None else None
            mask_B = 1-mask_B if mask_B is not None else None
    
        guides = {
            "guide_mode"                : guide_mode,
            "weight_masked"             : weight_A,
            "weight_unmasked"           : weight_B,
            "weights_masked"            : weights_A,
            "weights_unmasked"          : weights_B,
            "guide_masked"              : guide_A,
            "guide_unmasked"            : guide_B,
            "mask"                      : mask_A,
            "unmask"                    : mask_B,

            "weight_scheduler_masked"   : weight_scheduler_A,
            "weight_scheduler_unmasked" : weight_scheduler_B,
            "start_step_masked"         : start_step_A,
            "start_step_unmasked"       : start_step_B,
            "end_step_masked"           : end_step_A,
            "end_step_unmasked"         : end_step_B,
            "cutoff_masked"             : cutoff_A,
            "cutoff_unmasked"           : cutoff_B
        }
        
        return (guides, )
    


class ClownOptions_Combine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "options": ("OPTIONS",),
            },
        }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/sampler_options"

    def main(self, options, **kwargs):
        options_mgr = OptionsManager(options, **kwargs)
        return (options_mgr.as_dict(),)



class ClownOptions_Frameweights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config_name": (FRAME_WEIGHTS_CONFIG_NAMES, {"default": "frame_weights", "tooltip": "Apply to specific type of per-frame weights."}),
                "dynamics": (FRAME_WEIGHTS_DYNAMICS_NAMES, {"default": "ease_out", "tooltip": "The function type used for the dynamic period. constant: no change, linear: steady change, ease_out: starts fast, ease_in: starts slow"}),
                "schedule": (FRAME_WEIGHTS_SCHEDULE_NAMES, {"default": "moderate_early", "tooltip": "fast_early: fast change starts immediately, slow_late: slow change starts later"}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of change over the course of the frame weights. 1.0 means that the guides have no influence by the end."}),
                "reverse": ("BOOLEAN", {"default": False, "tooltip": "Reverse the frame weights"}),
            },
            "optional": {
                "frame_weights": ("SIGMAS", {"tooltip": "Overrides all other settings EXCEPT reverse."}),
                "custom_string": ("STRING", {"tooltip": "Overrides all other settings EXCEPT reverse.", "multiline": True}),
                "options": ("OPTIONS",),
            },
        }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/sampler_options"

    def main(self,
            config_name,
            dynamics,
            schedule,
            scale,
            reverse,
            frame_weights = None,
            custom_string = None,
            options       = None,
            ):
        
        options_mgr = OptionsManager(options if options is not None else {})

        frame_weights_mgr = options_mgr.get("frame_weights_mgr")
        if frame_weights_mgr is None:
            frame_weights_mgr = FrameWeightsManager()

        if custom_string is not None and custom_string.strip() == "":
            custom_string = None
        
        frame_weights_mgr.add_weight_config(
            config_name,
            dynamics=dynamics,
            schedule=schedule,
            scale=scale,
            is_reversed=reverse,
            frame_weights=frame_weights,
            custom_string=custom_string
        )
        
        options_mgr.update("frame_weights_mgr", frame_weights_mgr)
        
        return (options_mgr.as_dict(),)


class SharkOptions_GuiderInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"guider": ("GUIDER", ),
                    },
                "optional":
                    {"options": ("OPTIONS", ),
                    }
                }
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"

    def main(self, guider, options=None):
        options_mgr = OptionsManager(options if options is not None else {})
        
        if isinstance(guider, dict):
            guider = guider.get('samples', None)
            
        if isinstance(guider, torch.Tensor):
            guider = guider.detach().cpu()
        
        if options_mgr is None:
            options_mgr = OptionsManager()
            
        options_mgr.update("guider", guider)
        
        return (options_mgr.as_dict(), )
