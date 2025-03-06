import torch
import torch.nn.functional as F

from nodes import MAX_RESOLUTION

from ..helper                import OptionsManager, initialize_or_scale, get_res4lyf_scheduler_list

from .rk_coefficients_beta   import RK_SAMPLER_NAMES_BETA_FOLDERS, get_default_sampler_name, get_sampler_name_list, process_sampler_name

from .noise_classes          import NOISE_GENERATOR_NAMES_SIMPLE
from .rk_noise_sampler_beta  import NOISE_MODE_NAMES
from .constants              import IMPLICIT_TYPE_NAMES, GUIDE_MODE_NAMES_BETA_MISC, GUIDE_MODE_NAMES_BETA_SIMPLE



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
            noise_type_sde         = "gaussian",
            noise_type_sde_substep = "gaussian",
            noise_mode_sde         = "hard",
            noise_mode_sde_substep = "hard",
            eta                    = 0.5,
            eta_substep            = 0.5,
            options                = None,
            ): 
        
        options = options if options is not None else {}
            
        options['noise_type_sde']         = noise_type_sde
        options['noise_type_sde_substep'] = noise_type_sde_substep
        options['noise_mode_sde']         = noise_mode_sde
        options['noise_mode_sde_substep'] = noise_mode_sde_substep
        options['eta']                    = eta
        options['eta_substep']            = eta_substep

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



class ClownOptions_DetailBoost_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "noise_boost_step":    ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_boost_substep": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_anchor":        ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."}),
                    "s_noise":             ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "s_noise_substep":     ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "d_noise":             ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
                     },
                "optional": 
                    {
                    "options":             ("OPTIONS", ),   
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            noise_boost_step    = 0.0,
            noise_boost_substep = 0.0,
            noise_anchor        = 1.0,
            s_noise             = 1.0,
            s_noise_substep     = 1.0,
            d_noise             = 1.0,
            options             = None
            ):
        
        options = options if options is not None else {}
            
        options['noise_boost_step']    = noise_boost_step
        options['noise_boost_substep'] = noise_boost_substep
        options['noise_anchor']        = noise_anchor
        options['s_noise']             = s_noise
        options['s_noise_substep']     = s_noise_substep
        options['d_noise']             = d_noise

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
            
        options['extra_options'] = extra_options

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
            
        frame_weights_grp = (frame_weights, frame_weights)

        automation = {
            "etas"              : etas,
            "etas_substep"      : etas_substep,
            "s_noises"          : s_noises,
            "s_noises_substep"  : s_noises_substep,
            "epsilon_scales"    : epsilon_scales,
            "frame_weights_grp" : frame_weights_grp,
        }
        
        options["automation"] = automation

        return (options, )



class SharkOptions_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "noise_stdev":     ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                #"sampler_mode":    (['standard', 'unsample', 'resample'],),
                "denoise_alt":     ("FLOAT",                      {"default": 1.0, "min": -10000,   "max": 10000,   "step":0.01}),
                "channelwise_cfg": ("BOOLEAN",                    {"default": False}),
                },
            "optional": {
                #"sigmas":          ("SIGMAS", ),
                "options":         ("OPTIONS", ),   
                }
            }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            noise_type_init = "gaussian",
            noise_stdev     = 1.0,
            #sampler_mode    = "standard",
            denoise_alt     = 1.0,
            channelwise_cfg = False,
            #sigmas          = None,
            options         = None
            ): 
        
        options = options if options is not None else {}
            
        options['noise_type_init'] = noise_type_init
        options['noise_stdev']     = noise_stdev
        #options['sampler_mode']    = sampler_mode
        options['denoise_alt']     = denoise_alt
        options['channelwise_cfg'] = channelwise_cfg
        #options['sigmas']          = sigmas

        return (options,)
    
    
    
class SharkOptions_UltraCascade_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #"ultracascade_stage": (["stage_UP", "stage_B"], {"default": "stage_UP"}),
                "latent_image":       ("LATENT",),
                },
            "optional": {
                "options":            ("OPTIONS",),   
                }
            }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/sampler_options"
    
    def main(self,
            #ultracascade_stage = "stage_UP",
            latent_image       = None,
            options            = None
            ): 
        
        options = options if options is not None else {}
            
        #options['ultracascade_stage']        = ultracascade_stage
        options['ultracascade_latent_image'] = latent_image

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
    
    
    
class ClownOptions_StepsToRun_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps_to_run":  ("INT", {"default": -1,  "min": -1, "max": 10000}),
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
            steps_to_run = -1,
            options      = None,
            ): 
        
        options = options if options is not None else {}
            
        options['steps_to_run'] = steps_to_run

        return (options,)



class ClownGuide_Misc_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_mode":           (GUIDE_MODE_NAMES_BETA_MISC,                  {"default": 'blend',                                                        "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                "channelwise_mode":     ("BOOLEAN",                                   {"default": False}),
                "projection_mode":      ("BOOLEAN",                                   {"default": False}),
                "weight":               ("FLOAT",                                     {"default": 0.05, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                "cutoff":               ("FLOAT",                                     {"default": 1.0,  "min": 0.0, "   max": 1.0,   "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                "weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "start_step":           ("INT",                                       {"default": 0,    "min": 0, "     max": 10000}),
                "end_step":             ("INT",                                       {"default": 15,   "min": 1, "     max": 10000}),
                "invert_mask":          ("BOOLEAN",                                   {"default": False}),
                },
            "optional": {
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

            guide_mode                = "blend",
            channelwise_mode          = False,
            projection_mode           = False,
            weights                   = None,
            weights_unmasked          = None,
            mask                      = None,
            unmask                    = None,
            invert_mask               = False,
            ):
        
        #if guide_mode.startswith("epsilon_") and not guide_mode.startswith("epsilon_projection") and guide_bkg == None:
        #    print("Warning: need two latent inputs for guide_mode=",guide_mode," to work. Falling back to epsilon.")
        #    guide_mode = "epsilon"
        
        CG = ClownGuides_Beta()
        
        mask = 1-mask if mask is not None else None
        
        guides = CG.main(
                        weight_scheduler,
                        weight_scheduler_unmasked,
                        start_step,
                        start_step_unmasked,
                        end_step,
                        end_step_unmasked,
                        cutoff,
                        cutoff_unmasked,
                        guide,
                        guide_unmasked,
                        weight,
                        weight_unmasked,

                        guide_mode,
                        channelwise_mode,
                        projection_mode,
                        weights,
                        weights_unmasked,
                        mask,
                        unmask,
                        invert_mask,
                        )
        
        return (guides[0], )



class ClownGuide_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "guide_mode":           (GUIDE_MODE_NAMES_BETA_SIMPLE,                {"default": 'epsilon',                                                      "tooltip": "Recommended: epsilon or mean/mean_std with sampler_mode = standard, and unsample/resample with sampler_mode = unsample/resample. Epsilon_dynamic_mean, etc. are only used with two latent inputs and a mask. Blend/hard_light/mean/mean_std etc. require low strengths, start with 0.01-0.02."}),
                    "channelwise_mode":     ("BOOLEAN",                                   {"default": True}),
                    "projection_mode":      ("BOOLEAN",                                   {"default": True}),
                    "weight":               ("FLOAT",                                     {"default": 0.75, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set the strength of the guide."}),
                    "cutoff":               ("FLOAT",                                     {"default": 1.0,  "min": 0.0,    "max": 1.0,   "step":0.01, "round": False, "tooltip": "Disables the guide for the next step when the denoised image is similar to the guide. Higher values will strengthen the effect."}),
                    "weight_scheduler":     (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "start_step":           ("INT",                                       {"default": 0,    "min": 0,      "max": 10000}),
                    "end_step":             ("INT",                                       {"default": 15,   "min": 1,      "max": 10000}),
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
        
        guides = CG.main(
                        weight_scheduler,
                        weight_scheduler_unmasked,
                        start_step,
                        start_step_unmasked,
                        end_step,
                        end_step_unmasked,
                        cutoff,
                        cutoff_unmasked,
                        guide,
                        guide_unmasked,
                        weight,
                        weight_unmasked,

                        guide_mode,
                        channelwise_mode,
                        projection_mode,
                        weights,
                        weights_unmasked,
                        mask,
                        unmask,
                        invert_mask,
                        )
        
        return (guides[0], )




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
                    "start_step_masked":           ("INT",                                       {"default": 0,    "min": 0,      "max": 10000}),
                    "start_step_unmasked":         ("INT",                                       {"default": 0,    "min": 0,      "max": 10000}),
                    "end_step_masked":             ("INT",                                       {"default": 15,   "min": 1,      "max": 10000}),
                    "end_step_unmasked":           ("INT",                                       {"default": 15,   "min": 1,      "max": 10000}),
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
    
        guides = (
                guide_mode,
                weight_masked,
                weight_unmasked,
                weights_masked,
                weights_unmasked,
                guide_masked,
                guide_unmasked,
                mask,
                unmask,

                weight_scheduler_masked,
                weight_scheduler_unmasked,
                start_step_masked,
                start_step_unmasked,
                end_step_masked,
                end_step_unmasked,
                cutoff_masked,
                cutoff_unmasked
                )
        
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
                    "start_step_A":       ("INT",                                       {"default": 0,    "min": 0,      "max": 10000}),
                    "start_step_B":       ("INT",                                       {"default": 0,    "min": 0,      "max": 10000}),
                    "end_step_A":         ("INT",                                       {"default": 15,   "min": 1,      "max": 10000}),
                    "end_step_B":         ("INT",                                       {"default": 15,   "min": 1,      "max": 10000}),
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
    
        guides = (
                guide_mode,
                weight_A,
                weight_B,
                weights_A,
                weights_B,
                guide_A,
                guide_B,
                mask_A,
                mask_B,

                weight_scheduler_A,
                weight_scheduler_B,
                start_step_A,
                start_step_B,
                end_step_A,
                end_step_B,
                cutoff_A,
                cutoff_B
                )
        
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

    def main(self, **kwargs):

        options_inputs = []

        if "options" in kwargs and kwargs["options"] is not None:
            options_inputs.append(kwargs["options"])

        i = 2
        while True:
            option_name = f"options {i}"
            if option_name in kwargs and kwargs[option_name] is not None:
                options_inputs.append(kwargs[option_name])
                i += 1
            else:
                break

        options = OptionsManager(options_inputs)

        return (options.as_dict(),)
    
class ClownOptions_Frameweights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "frame_weights": ("SIGMAS",),
                "frame_weights_inv": ("SIGMAS",),
                "options": ("OPTIONS",),
            },
        }

    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/sampler_options"

    def main(self,          
            frame_weights    = None,
            frame_weights_inv = None,
            options          = None
            ):
        
        options_mgr = OptionsManager(options)
        
        frame_weights_grp = (frame_weights, frame_weights_inv)
            
        if frame_weights_grp[0] is not None or frame_weights_grp[1] is not None:
            if "automation" in options_mgr and "frame_weights_grp" in options_mgr["automation"]:
                current_frame_weights_grp = options_mgr["automation"]["frame_weights_grp"]
                frame_weights_grp[0] = frame_weights_grp[0] if frame_weights_grp[0] is not None else current_frame_weights_grp[0]
                frame_weights_grp[1] = frame_weights_grp[1] if frame_weights_grp[1] is not None else current_frame_weights_grp[1]
                
        options_mgr.update("automation.frame_weights_grp", frame_weights_grp)
        
        return (options_mgr.as_dict(),)