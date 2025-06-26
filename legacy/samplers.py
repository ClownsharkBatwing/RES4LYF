from .noise_classes import prepare_noise, NOISE_GENERATOR_CLASSES_SIMPLE, NOISE_GENERATOR_NAMES_SIMPLE, NOISE_GENERATOR_NAMES
from .sigmas import get_sigmas


from .constants import MAX_STEPS

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd
import comfy.supported_models

import latent_preview
import torch
import torch.nn.functional as F

import math
import copy

from .helper import get_extra_options_kv, extra_options_flag, get_res4lyf_scheduler_list
from .latents import initialize_or_scale

from .noise_classes import prepare_noise, NOISE_GENERATOR_CLASSES_SIMPLE, NOISE_GENERATOR_NAMES_SIMPLE, NOISE_GENERATOR_NAMES
from .sigmas import get_sigmas


from .rk_sampler import sample_rk
from .rk_coefficients import RK_SAMPLER_NAMES, IRK_SAMPLER_NAMES
from .rk_guide_func import get_orthogonal
from .noise_sigmas_timesteps_scaling import NOISE_MODE_NAMES


def move_to_same_device(*tensors):
    if not tensors:
        return tensors

    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)


#SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["beta57"]



class ClownSamplerAdvanced:
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
                    "s_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "d_noise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
                    "noise_seed_sde": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_name": (RK_SAMPLER_NAMES, {"default": "res_2m"}), 
                    "implicit_sampler_name": (IRK_SAMPLER_NAMES, {"default": "explicit_diagonal"}), 
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

    CATEGORY = "RES4LYF/legacy/samplers"
    DEPRECATED = True
    
    def main(self, 
             noise_type_sde="gaussian", noise_type_sde_substep="gaussian", noise_mode_sde="hard",
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, noise_seed_sde=-1, sampler_name="res_2m", implicit_sampler_name="gauss-legendre_2s",
                    t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, guide_mode="", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None, 
                    guides=None, options=None, sde_noise=None,sde_noise_steps=1, 
                    extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, regional_conditioning_weights=None,frame_weights_grp=None, eta_substep=0.5, noise_mode_sde_substep="hard",
                    ): 
            if implicit_sampler_name == "none":
                implicit_steps = 0 
                implicit_sampler_name = "gauss-legendre_2s"

            if noise_mode_sde == "none":
                eta, eta_var = 0.0, 0.0
                noise_mode_sde = "hard"
        
            default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "float64", extra_options), torch.float64)

            unsample_resample_scales_override = unsample_resample_scales

            if options is not None:
                noise_type_sde = options.get('noise_type_sde', noise_type_sde)
                noise_mode_sde = options.get('noise_mode_sde', noise_mode_sde)
                eta = options.get('eta', eta)
                s_noise = options.get('s_noise', s_noise)
                d_noise = options.get('d_noise', d_noise)
                alpha_sde = options.get('alpha_sde', alpha_sde)
                k_sde = options.get('k_sde', k_sde)
                c1 = options.get('c1', c1)
                c2 = options.get('c2', c2)
                c3 = options.get('c3', c3)
                t_fn_formula = options.get('t_fn_formula', t_fn_formula)
                sigma_fn_formula = options.get('sigma_fn_formula', sigma_fn_formula)
                frame_weights_grp = options.get('frame_weights_grp', frame_weights_grp)
                sde_noise = options.get('sde_noise', sde_noise)
                sde_noise_steps = options.get('sde_noise_steps', sde_noise_steps)

            #noise_seed_sde = torch.initial_seed()+1 if noise_seed_sde < 0 else noise_seed_sde 

            rescale_floor = extra_options_flag("rescale_floor", extra_options)

            if automation is not None:
                etas = automation['etas'] if 'etas' in automation else None
                s_noises = automation['s_noises'] if 's_noises' in automation else None
                unsample_resample_scales = automation['unsample_resample_scales'] if 'unsample_resample_scales' in automation else None
                frame_weights_grp = automation['frame_weights_grp'] if 'frame_weights_grp' in automation else None

            etas = initialize_or_scale(etas, eta, MAX_STEPS).to(default_dtype)
            etas = F.pad(etas, (0, MAX_STEPS), value=0.0)
            s_noises = initialize_or_scale(s_noises, s_noise, MAX_STEPS).to(default_dtype)
            s_noises = F.pad(s_noises, (0, MAX_STEPS), value=0.0)
        
            if sde_noise is None:
                sde_noise = []
            else:
                sde_noise = copy.deepcopy(sde_noise)
                for i in range(len(sde_noise)):
                    sde_noise[i] = sde_noise[i]
                    for j in range(sde_noise[i].shape[1]):
                        sde_noise[i][0][j] = ((sde_noise[i][0][j] - sde_noise[i][0][j].mean()) / sde_noise[i][0][j].std())
                        
            if unsample_resample_scales_override is not None:
                unsample_resample_scales = unsample_resample_scales_override

            sampler = comfy.samplers.ksampler("rk", {"eta": eta, "eta_var": eta_var, "s_noise": s_noise, "d_noise": d_noise, "alpha": alpha_sde, "k": k_sde, "c1": c1, "c2": c2, "c3": c3, "cfgpp": cfgpp, 
                                                    "noise_sampler_type": noise_type_sde, "noise_mode": noise_mode_sde, "noise_seed": noise_seed_sde, "rk_type": sampler_name, "implicit_sampler_name": implicit_sampler_name,
                                                            "t_fn_formula": t_fn_formula, "sigma_fn_formula": sigma_fn_formula, "implicit_steps": implicit_steps,
                                                            "latent_guide": latent_guide, "latent_guide_inv": latent_guide_inv, "mask": latent_guide_mask, "mask_inv": latent_guide_mask_inv,
                                                            "latent_guide_weights": latent_guide_weights, "latent_guide_weights_inv": latent_guide_weights_inv, "guide_mode": guide_mode,
                                                            "LGW_MASK_RESCALE_MIN": rescale_floor, "sigmas_override": sigmas_override, "sde_noise": sde_noise,
                                                            "extra_options": extra_options,
                                                            "etas": etas, "s_noises": s_noises, "unsample_resample_scales": unsample_resample_scales, "regional_conditioning_weights": regional_conditioning_weights,
                                                            "guides": guides, "frame_weights_grp": frame_weights_grp, "eta_substep": eta_substep, "noise_mode_sde_substep": noise_mode_sde_substep,
                                                            })

            return (sampler, )




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
                    "implicit_sampler_name": (IRK_SAMPLER_NAMES, {"default": "explicit_diagonal"}), 
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

    CATEGORY = "RES4LYF/legacy/samplers"
    DEPRECATED = True
    
    def main(self, 
             noise_type_sde="gaussian", noise_type_sde_substep="gaussian", noise_mode_sde="hard",
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, noise_seed_sde=-1, sampler_name="res_2m", implicit_sampler_name="gauss-legendre_2s",
                    t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, guide_mode="", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None,
                    guides=None, options=None, sde_noise=None,sde_noise_steps=1, 
                    extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, regional_conditioning_weights=None,frame_weights_grp=None,eta_substep=0.0, noise_mode_sde_substep="hard",
                    ): 

        eta_substep = eta
        noise_mode_sde_substep = noise_mode_sde
        noise_type_sde_substep = noise_type_sde

        sampler = ClownSamplerAdvanced().main(
                noise_type_sde=noise_type_sde, noise_type_sde_substep=noise_type_sde_substep, noise_mode_sde=noise_mode_sde,
             eta=eta, eta_var=eta_var, d_noise=d_noise, s_noise=s_noise, alpha_sde=alpha_sde, k_sde=k_sde, cfgpp=cfgpp, c1=c1, c2=c2, c3=c3, noise_seed_sde=noise_seed_sde, sampler_name=sampler_name, implicit_sampler_name=implicit_sampler_name,
                    t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, implicit_steps=implicit_steps,
                    latent_guide=latent_guide, latent_guide_inv=latent_guide_inv, guide_mode=guide_mode, latent_guide_weights=latent_guide_weights, latent_guide_weights_inv=latent_guide_weights_inv, latent_guide_mask=latent_guide_mask, latent_guide_mask_inv=latent_guide_mask_inv, rescale_floor=rescale_floor, sigmas_override=sigmas_override,
                    guides=guides, options=options, sde_noise=sde_noise,sde_noise_steps=sde_noise_steps, 
                    extra_options=extra_options, automation=automation, etas=etas, s_noises=s_noises,unsample_resample_scales=unsample_resample_scales, regional_conditioning_weights=regional_conditioning_weights,frame_weights_grp=frame_weights_grp, eta_substep=eta_substep, noise_mode_sde_substep=noise_mode_sde_substep,
                    )
        
        return sampler



def process_sampler_name(selected_value):
    processed_name = selected_value.split("/")[-1]
    
    if selected_value.startswith("fully_implicit") or selected_value.startswith("diag_implicit"):
        implicit_sampler_name = processed_name
        sampler_name = "buehler"
    else:
        sampler_name = processed_name
        implicit_sampler_name = "use_explicit"
    
    return sampler_name, implicit_sampler_name


def copy_cond(positive):
    new_positive = []
    for embedding, cond in positive:
        cond_copy = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor):
                cond_copy[k] = v.clone()
            else:
                cond_copy[k] = v  # ensure we're not copying huge shit like controlnets
        new_positive.append([embedding.clone(), cond_copy])
    return new_positive



class SharkSamplerAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_stdev": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                    "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_mode": (['standard', 'unsample', 'resample'],),
                    "scheduler": (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "denoise_alt": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
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

    CATEGORY = "RES4LYF/legacy/samplers"
    DEPRECATED = True
    
    def main(self, model, cfg, scheduler, steps, sampler_mode="standard",denoise=1.0, denoise_alt=1.0,
             noise_type_init="gaussian", latent_image=None, 
             positive=None, negative=None, sampler=None, sigmas=None, latent_noise=None, latent_noise_match=None,
             noise_stdev=1.0, noise_mean=0.0, noise_normalize=True, 
             d_noise=1.0, alpha_init=-1.0, k_init=1.0, cfgpp=0.0, noise_seed=-1,
                    options=None, sde_noise=None,sde_noise_steps=1, 
                    extra_options="", 
                    ): 
            # blame comfy here
            raw_x = latent_image['raw_x'] if 'raw_x' in latent_image else None
            last_seed = latent_image['last_seed'] if 'last_seed' in latent_image else None
            
            pos_cond = copy_cond(positive)
            neg_cond = copy_cond(negative)

            if sampler is None:
                raise ValueError("sampler is required")
            else:
                sampler = copy.deepcopy(sampler)

            default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "float64", extra_options), torch.float64)
                     
            model = model.clone()
            if pos_cond[0][1] is not None: 
                if "regional_conditioning_weights" in pos_cond[0][1]:
                    sampler.extra_options['regional_conditioning_weights'] = pos_cond[0][1]['regional_conditioning_weights']
                    sampler.extra_options['regional_conditioning_floors']  = pos_cond[0][1]['regional_conditioning_floors']
                    regional_generate_conditionings_and_masks_fn = pos_cond[0][1]['regional_generate_conditionings_and_masks_fn']
                    regional_conditioning, regional_mask = regional_generate_conditionings_and_masks_fn(latent_image['samples'])
                    regional_conditioning = copy.deepcopy(regional_conditioning)
                    regional_mask = copy.deepcopy(regional_mask)
                    model.set_model_patch(regional_conditioning, 'regional_conditioning_positive')
                    model.set_model_patch(regional_mask,         'regional_conditioning_mask')
                    
            if "noise_seed" in sampler.extra_options:
                if sampler.extra_options['noise_seed'] == -1 and noise_seed != -1:
                    sampler.extra_options['noise_seed'] = noise_seed + 1
                    #print("Shark: setting clown noise seed to: ", sampler.extra_options['noise_seed'])

            if "sampler_mode" in sampler.extra_options:
                sampler.extra_options['sampler_mode'] = sampler_mode

            if "extra_options" in sampler.extra_options:
                extra_options += " "
                extra_options += sampler.extra_options['extra_options']
                sampler.extra_options['extra_options'] = extra_options

            batch_size = int(get_extra_options_kv("batch_size", "1", extra_options))
            if batch_size > 1:
                latent_image['samples'] = latent_image['samples'].repeat(batch_size, 1, 1, 1) 
            
            latent_image_batch = {"samples": latent_image['samples']}
            out_samples, out_samples_fp64, out_denoised_samples, out_denoised_samples_fp64 = [], [], [], []
            for batch_num in range(latent_image_batch['samples'].shape[0]):
                latent_unbatch = copy.deepcopy(latent_image)
                latent_unbatch['samples'] = latent_image_batch['samples'][batch_num].clone().unsqueeze(0)
                


                if noise_seed == -1:
                    seed = torch.initial_seed() + 1 + batch_num
                else:
                    seed = noise_seed + batch_num
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    #torch.cuda.manual_seed_all(seed)


                if options is not None:
                    noise_stdev     = options.get('noise_init_stdev', noise_stdev)
                    noise_mean      = options.get('noise_init_mean',  noise_mean)
                    noise_type_init = options.get('noise_type_init',  noise_type_init)
                    d_noise         = options.get('d_noise',          d_noise)
                    alpha_init      = options.get('alpha_init',       alpha_init)
                    k_init          = options.get('k_init',           k_init)
                    sde_noise       = options.get('sde_noise',        sde_noise)
                    sde_noise_steps = options.get('sde_noise_steps',  sde_noise_steps)

                latent_image_dtype = latent_unbatch['samples'].dtype

                if isinstance(model.model.model_config, comfy.supported_models.Flux) or isinstance(model.model.model_config, comfy.supported_models.FluxSchnell):
                    if pos_cond is None:
                        pos_cond = [[
                            torch.zeros((1, 256, 4096)),
                            {'pooled_output': torch.zeros((1, 768))}
                            ]]

                    if extra_options_flag("uncond_ortho_flux", extra_options):
                        if neg_cond is None:
                            print("uncond_ortho_flux: using random negative conditioning...")
                            neg_cond = [[
                                torch.randn((1, 256, 4096)),
                                {'pooled_output': torch.randn((1, 768))}
                                ]]
                        #neg_cond[0][0] = get_orthogonal(neg_cond[0][0].to(torch.bfloat16), pos_cond[0][0].to(torch.bfloat16))
                        #neg_cond[0][1]['pooled_output'] = get_orthogonal(neg_cond[0][1]['pooled_output'].to(torch.bfloat16), pos_cond[0][1]['pooled_output'].to(torch.bfloat16))
                        neg_cond[0][0] = get_orthogonal(neg_cond[0][0], pos_cond[0][0])
                        neg_cond[0][1]['pooled_output'] = get_orthogonal(neg_cond[0][1]['pooled_output'], pos_cond[0][1]['pooled_output'])
                        
                    if neg_cond is None:
                        neg_cond = [[
                            torch.zeros((1, 256, 4096)),
                            {'pooled_output': torch.zeros((1, 768))}
                            ]]
                else:
                    if pos_cond is None:
                        pos_cond = [[
                            torch.zeros((1, 154, 4096)),
                            {'pooled_output': torch.zeros((1, 2048))}
                            ]]

                    if extra_options_flag("uncond_ortho_sd35", extra_options):
                        if neg_cond is None:
                            neg_cond = [[
                                torch.randn((1, 154, 4096)),
                                {'pooled_output': torch.randn((1, 2048))}
                                ]]
                        
                        neg_cond[0][0] = get_orthogonal(neg_cond[0][0], pos_cond[0][0])
                        neg_cond[0][1]['pooled_output'] = get_orthogonal(neg_cond[0][1]['pooled_output'], pos_cond[0][1]['pooled_output'])
                        

                    if neg_cond is None:
                        neg_cond = [[
                            torch.zeros((1, 154, 4096)),
                            {'pooled_output': torch.zeros((1, 2048))}
                            ]]
                        
                        
                if extra_options_flag("zero_uncond_t5", extra_options):
                    neg_cond[0][0] = torch.zeros_like(neg_cond[0][0])
                    
                if extra_options_flag("zero_uncond_pooled_output", extra_options):
                    neg_cond[0][1]['pooled_output'] = torch.zeros_like(neg_cond[0][1]['pooled_output'])
                        
                if extra_options_flag("zero_pooled_output", extra_options):
                    pos_cond[0][1]['pooled_output'] = torch.zeros_like(pos_cond[0][1]['pooled_output'])
                    neg_cond[0][1]['pooled_output'] = torch.zeros_like(neg_cond[0][1]['pooled_output'])

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

                x = latent_unbatch["samples"].clone().to(default_dtype) 
                if latent_unbatch is not None:
                    if "samples_fp64" in latent_unbatch:
                        if latent_unbatch['samples'].shape == latent_unbatch['samples_fp64'].shape:
                            if torch.norm(latent_unbatch['samples'] - latent_unbatch['samples_fp64']) < 0.01:
                                x = latent_unbatch["samples_fp64"].clone()

                if latent_noise is not None:
                    latent_noise_samples = latent_noise["samples"].clone().to(default_dtype)  
                if latent_noise_match is not None:
                    latent_noise_match_samples = latent_noise_match["samples"].clone().to(default_dtype)

                truncate_conditioning = extra_options_flag("truncate_conditioning", extra_options)
                if truncate_conditioning == "true" or truncate_conditioning == "true_and_zero_neg":
                    if pos_cond is not None:
                        pos_cond[0][0] = pos_cond[0][0].clone().to(default_dtype)
                        pos_cond[0][1]["pooled_output"] = pos_cond[0][1]["pooled_output"].clone().to(default_dtype)
                    if neg_cond is not None:
                        neg_cond[0][0] = neg_cond[0][0].clone().to(default_dtype)
                        neg_cond[0][1]["pooled_output"] = neg_cond[0][1]["pooled_output"].clone().to(default_dtype)
                    c = []
                    for t in pos_cond:
                        d = t[1].copy()
                        pooled_output = d.get("pooled_output", None)

                    for t in neg_cond:
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
                    neg_cond = c
                
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
                        print("Initial latent noise seed: ", seed)
                        noise_sampler_init = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_type_init)(x=x, seed=seed, sigma_min=sigmin, sigma_max=sigmax)
                    
                        if noise_type_init == "fractal":
                            noise_sampler_init.alpha = alpha_init
                            noise_sampler_init.k = k_init
                            noise_sampler_init.scale = 0.1
                        noise = noise_sampler_init(sigma=sigmax, sigma_next=sigmin)
                    else:
                        noise = latent_noise_samples

                    if noise_normalize and noise.std() > 0:
                        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)
                        #noise.sub_(noise.mean()).div_(noise.std())
                    noise *= noise_stdev
                    noise = (noise - noise.mean()) + noise_mean
                    
                    if latent_noise_match is not None:
                        for i in range(latent_noise_match_samples.shape[1]):
                            noise[0][i] = (noise[0][i] - noise[0][i].mean())
                            noise[0][i] = (noise[0][i]) + latent_noise_match_samples[0][i].mean()

                    noise_mask = latent_unbatch["noise_mask"] if "noise_mask" in latent_unbatch else None

                    x0_output = {}


                    if cfg < 0:
                        sampler.extra_options['cfg_cw'] = -cfg
                        cfg = 1.0
                    else:
                        sampler.extra_options.pop("cfg_cw", None) 
                        
                    
                    if sde_noise is None:
                        sde_noise = []
                    else:
                        sde_noise = copy.deepcopy(sde_noise)
                        for i in range(len(sde_noise)):
                            sde_noise[i] = sde_noise[i]
                            for j in range(sde_noise[i].shape[1]):
                                sde_noise[i][0][j] = ((sde_noise[i][0][j] - sde_noise[i][0][j].mean()) / sde_noise[i][0][j].std())
                                
                    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

                    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                    
                    model.model.diffusion_model.raw_x = raw_x
                    model.model.diffusion_model.last_seed = last_seed
                    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, pos_cond, neg_cond, x.clone(), noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

                    out = latent_unbatch.copy()
                    out["samples"] = samples
                    if "x0" in x0_output:
                        out_denoised = latent_unbatch.copy()
                        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
                    else:
                        out_denoised = out
                    
                    out["samples_fp64"] = out["samples"].clone()
                    out["samples"]      = out["samples"].to(latent_image_dtype)
                    
                    out_denoised["samples_fp64"] = out_denoised["samples"].clone()
                    out_denoised["samples"]      = out_denoised["samples"].to(latent_image_dtype)
                    
                    out_samples.     append(out["samples"])
                    out_samples_fp64.append(out["samples_fp64"])
                    
                    out_denoised_samples.     append(out_denoised["samples"])
                    out_denoised_samples_fp64.append(out_denoised["samples_fp64"])
                    
                    seed += 1
                    torch.manual_seed(seed)
                    if total_steps_iter > 1: 
                        sde_noise.append(out["samples_fp64"])
                        
            out_samples               = [tensor.squeeze(0) for tensor in out_samples]
            out_samples_fp64          = [tensor.squeeze(0) for tensor in out_samples_fp64]
            out_denoised_samples      = [tensor.squeeze(0) for tensor in out_denoised_samples]
            out_denoised_samples_fp64 = [tensor.squeeze(0) for tensor in out_denoised_samples_fp64]

            out['samples']      = torch.stack(out_samples,     dim=0)
            out['samples_fp64'] = torch.stack(out_samples_fp64, dim=0)
            
            out_denoised['samples']      = torch.stack(out_denoised_samples,     dim=0)
            out_denoised['samples_fp64'] = torch.stack(out_denoised_samples_fp64, dim=0)

            out['raw_x'] = None
            if hasattr(model.model.diffusion_model, "raw_x"):
                if model.model.diffusion_model.raw_x is not None:
                    out['raw_x'] = model.model.diffusion_model.raw_x.clone()
                    del model.model.diffusion_model.raw_x

            out['last_seed'] = None
            if hasattr(model.model.diffusion_model, "last_seed"):
                if model.model.diffusion_model.last_seed is not None:
                    out['last_seed'] = model.model.diffusion_model.last_seed
                    del model.model.diffusion_model.last_seed

            return ( out, out_denoised, sde_noise,)



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
                    "implicit_sampler_name": (IRK_SAMPLER_NAMES, {"default": "explicit_diagonal"}), 
                    "scheduler": (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "implicit_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "denoise_alt": ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, }),
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

    CATEGORY = "RES4LYF/legacy/samplers"
    DEPRECATED = True
    
    def main(self, model, cfg, sampler_mode, scheduler, steps, denoise=1.0, denoise_alt=1.0,
             noise_type_init="gaussian", noise_type_sde="brownian", noise_mode_sde="hard", latent_image=None, 
             positive=None, negative=None, sigmas=None, latent_noise=None, latent_noise_match=None,
             noise_stdev=1.0, noise_mean=0.0, noise_normalize=True, noise_is_latent=False, 
             eta=0.25, eta_var=0.0, d_noise=1.0, s_noise=1.0, alpha_init=-1.0, k_init=1.0, alpha_sde=-1.0, k_sde=1.0, cfgpp=0.0, c1=0.0, c2=0.5, c3=1.0, noise_seed=-1, sampler_name="res_2m", implicit_sampler_name="default",
                    t_fn_formula=None, sigma_fn_formula=None, implicit_steps=0,
                    latent_guide=None, latent_guide_inv=None, guide_mode="blend", latent_guide_weights=None, latent_guide_weights_inv=None, latent_guide_mask=None, latent_guide_mask_inv=None, rescale_floor=True, sigmas_override=None, 
                    shift=3.0, base_shift=0.85, guides=None, options=None, sde_noise=None,sde_noise_steps=1, shift_scaling="exponential",
                    extra_options="", automation=None, etas=None, s_noises=None,unsample_resample_scales=None, regional_conditioning_weights=None,frame_weights_grp=None,
                    ): 

        if noise_seed >= 0:
            noise_seed_sde = noise_seed + 1
        else:
            noise_seed_sde = -1
        
        eta_substep = eta
        noise_mode_sde_substep = noise_mode_sde
        noise_type_sde_substep = noise_type_sde 

        sampler = ClownSamplerAdvanced().main(
                noise_type_sde=noise_type_sde, noise_type_sde_substep=noise_type_sde_substep, noise_mode_sde=noise_mode_sde,
             eta=eta, eta_var=eta_var, d_noise=d_noise, s_noise=s_noise, alpha_sde=alpha_sde, k_sde=k_sde, cfgpp=cfgpp, c1=c1, c2=c2, c3=c3, noise_seed_sde=noise_seed_sde, sampler_name=sampler_name, implicit_sampler_name=implicit_sampler_name,
                    t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, implicit_steps=implicit_steps,
                    latent_guide=latent_guide, latent_guide_inv=latent_guide_inv, guide_mode=guide_mode, latent_guide_weights=latent_guide_weights, latent_guide_weights_inv=latent_guide_weights_inv, latent_guide_mask=latent_guide_mask, latent_guide_mask_inv=latent_guide_mask_inv, rescale_floor=rescale_floor, sigmas_override=sigmas_override, 
                    guides=guides, options=options, sde_noise=sde_noise,sde_noise_steps=sde_noise_steps, 
                    extra_options=extra_options, automation=automation, etas=etas, s_noises=s_noises,unsample_resample_scales=unsample_resample_scales, regional_conditioning_weights=regional_conditioning_weights,frame_weights_grp=frame_weights_grp, eta_substep=eta_substep, noise_mode_sde_substep=noise_mode_sde_substep,
                    )

        return SharkSamplerAlpha().main(
            model=model, cfg=cfg, sampler_mode=sampler_mode, scheduler=scheduler, steps=steps, 
            denoise=denoise, denoise_alt=denoise_alt, noise_type_init=noise_type_init, 
            latent_image=latent_image, positive=positive, negative=negative, sampler=sampler[0], 
            sigmas=sigmas, latent_noise=latent_noise, latent_noise_match=latent_noise_match, 
            noise_stdev=noise_stdev, noise_mean=noise_mean, noise_normalize=noise_normalize, 
            d_noise=d_noise, alpha_init=alpha_init, k_init=k_init, cfgpp=cfgpp, noise_seed=noise_seed, 
            options=options, sde_noise=sde_noise, sde_noise_steps=sde_noise_steps, 
            extra_options=extra_options
        )





class UltraSharkSampler:  
    # for use with https://github.com/ClownsharkBatwing/UltraCascade
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": ("BOOLEAN", {"default": True}),
                "normalize_noise": ("BOOLEAN", {"default": False}),
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
                #"latent_noise": ("LATENT", ),
                "guide": ("LATENT",),
                "guide_weights": ("SIGMAS",),
                #"style": ("CONDITIONING", ),
                #"img_style": ("CONDITIONING", ),
            }
        }

    RETURN_TYPES = ("LATENT","LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output", "latent_batch")

    FUNCTION = "main"

    CATEGORY = "RES4LYF/legacy/samplers/UltraCascade"
    DESCRIPTION = "For use with Stable Cascade and UltraCascade."
    DEPRECATED = True
    
    def main(self, model, add_noise, normalize_noise, noise_type, noise_seed, cfg, alpha, k, positive, negative, sampler, 
               sigmas, guide_type, guide_weight, latent_image, latent_noise=None, guide=None, guide_weights=None, style=None, img_style=None): 

            if model.model.model_config.unet_config.get('stable_cascade_stage') == 'up':
                model = model.clone()
                x_lr = guide['samples'] if guide is not None else None
                guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)#("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
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
                
            if style is not None:
                model.set_model_patch(style, 'style_cond')
            if img_style is not None:
                model.set_model_patch(img_style,'img_style_cond')
        
            # 1, 768      clip_style[0][0][1]['unclip_conditioning'][0]['clip_vision_output'].image_embeds.shape
            # 1, 1280     clip_style[0][0][1]['pooled_output'].shape 
            # 1, 77, 1280 clip_style[0][0][0].shape
        
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

            if normalize_noise and noise.std() > 0:
                noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

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


