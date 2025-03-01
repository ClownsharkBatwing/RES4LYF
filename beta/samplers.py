import torch
import torch.nn.functional as F

import copy

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd
import comfy.supported_models

import latent_preview

from ..helper               import initialize_or_scale, get_extra_options_kv, extra_options_flag, get_res4lyf_scheduler_list, OptionsManager
from ..res4lyf              import RESplain
from ..latents              import get_orthogonal, get_collinear
from ..sigmas               import get_sigmas
import RES4LYF.models              # import ReFluxPatcher

from .constants             import MAX_STEPS, IMPLICIT_TYPE_NAMES
from .noise_classes         import NOISE_GENERATOR_CLASSES_SIMPLE, NOISE_GENERATOR_NAMES_SIMPLE, NOISE_GENERATOR_NAMES
from .rk_noise_sampler_beta import NOISE_MODE_NAMES
from .rk_coefficients_beta  import get_default_sampler_name, get_sampler_name_list, process_sampler_name



class SharkSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model":          ("MODEL",),
                    "noise_type_init": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_stdev":     ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, }),
                    "noise_seed":      ("INT",                        {"default": 0,   "min": -1,       "max": 0xffffffffffffffff}),
                    "sampler_mode":    (['standard', 'unsample', 'resample'],),
                    "scheduler":       (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "steps":           ("INT",                        {"default": 30,  "min": 1,        "max": 10000.0}),
                    "denoise":         ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01}),
                    "denoise_alt":     ("FLOAT",                      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.01}),
                    "cfg":             ("FLOAT",                      {"default": 3.0, "min": -10000.0, "max": 10000.0, "step":0.01, "round": False, "tooltip": "Negative values use channelwise CFG." }),
                    },
                "optional": 
                    {
                    "positive":        ("CONDITIONING", ),
                    "negative":        ("CONDITIONING", ),
                    "sampler":         ("SAMPLER", ),
                    "sigmas":          ("SIGMAS", ),
                    "latent_image":    ("LATENT", ),     
                    "options":         ("OPTIONS", ),   
                    "extra_options":   ("STRING",                     {"default": "", "multiline": True}),   
                    }
                }

    RETURN_TYPES = ("LATENT", 
                    "LATENT",  
                    "LATENT",)
    
    RETURN_NAMES = ("output", 
                    "denoised",
                    "sde_noise",) 
    
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/samplers"
    
    def main(self, 
            model,
            cfg, 
            scheduler, 
            steps, 
            sampler_mode       = "standard",
            denoise            = 1.0, 
            denoise_alt        = 1.0,
            noise_type_init    = "gaussian",
            latent_image       = None,
            
            positive           = None,
            negative           = None,
            sampler            = None,
            sigmas             = None,
            latent_noise       = None,
            latent_noise_match = None,
            noise_stdev        = 1.0,
            noise_mean         = 0.0,
            noise_normalize    = True,
            
            d_noise            = 1.0,
            alpha_init         = -1.0,
            k_init             = 1.0,
            cfgpp              = 0.0,
            noise_seed         = -1,
            options            = None,
            sde_noise          = None,
            sde_noise_steps    = 1,
        
            extra_options      = "", 
            **kwargs,
            ): 
        
            options_inputs = []

            if options is not None:
                options_inputs.append(options)

            i = 2
            while True:
                option_name = f"options {i}"
                if option_name in kwargs and kwargs[option_name] is not None:
                    options_inputs.append(kwargs[option_name])
                    i += 1
                else:
                    break

            options_mgr = OptionsManager(options_inputs)
        
            # blame comfy here
            state_info  = copy.deepcopy(latent_image['state_info']) if 'state_info' in latent_image else {}

            #state_info  = latent_image['state_info'] if 'state_info' in latent_image else {}
            #latent_image['state_info'] = {}
            pos_cond    = copy.deepcopy(positive)
            neg_cond    = copy.deepcopy(negative)

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
                    regional_generate_conditionings_and_masks_fn           = pos_cond[0][1]['regional_generate_conditionings_and_masks_fn']
                    regional_conditioning, regional_mask                   = regional_generate_conditionings_and_masks_fn(latent_image['samples'])
                    regional_conditioning                                  = copy.deepcopy(regional_conditioning)
                    regional_mask                                          = copy.deepcopy(regional_mask)
                    
                    model, = RES4LYF.models.ReFluxPatcher().main(model, enable=True)
                    model.set_model_patch(regional_conditioning, 'regional_conditioning_positive')
                    model.set_model_patch(regional_mask,         'regional_conditioning_mask')
                else:
                    model, = RES4LYF.models.ReFluxPatcher().main(model, enable=False)

                    
            if "noise_seed" in sampler.extra_options:
                if sampler.extra_options['noise_seed'] == -1 and noise_seed != -1:
                    sampler.extra_options['noise_seed'] = noise_seed + 1
                    print("Shark: setting clown noise seed to: ", sampler.extra_options['noise_seed'])

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
                latent_unbatch            = copy.deepcopy(latent_image)
                latent_unbatch['samples'] = latent_image_batch['samples'][batch_num].clone().unsqueeze(0)
                

                if noise_seed == -1:
                    seed = torch.initial_seed() + 1 + batch_num
                else:
                    seed = noise_seed + batch_num
                    #if state_info == {}:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)


                noise_stdev     = options_mgr.get('noise_init_stdev', noise_stdev)
                noise_mean      = options_mgr.get('noise_init_mean',  noise_mean)
                noise_type_init = options_mgr.get('noise_type_init',  noise_type_init)
                d_noise         = options_mgr.get('d_noise',          d_noise)
                alpha_init      = options_mgr.get('alpha_init',       alpha_init)
                k_init          = options_mgr.get('k_init',           k_init)
                sde_noise       = options_mgr.get('sde_noise',        sde_noise)
                sde_noise_steps = options_mgr.get('sde_noise_steps',  sde_noise_steps)

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
                        neg_cond[0][0]                  = get_orthogonal(neg_cond[0][0],                  pos_cond[0][0])
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
                d_noise = options_mgr.get('d_noise', d_noise)

                if sigmas is not None:
                    sigmas = sigmas.clone().to(default_dtype)
                else: 
                    sigmas = get_sigmas(model, scheduler, steps, denoise).to(default_dtype)
                sigmas *= denoise_alt

                if sampler_mode.startswith("unsample"): 
                    null   = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                    sigmas = torch.flip(sigmas, dims=[0])
                    sigmas = torch.cat([sigmas, null])
                    
                elif sampler_mode.startswith("resample"):
                    null   = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
                    sigmas = torch.cat([null, sigmas])
                    sigmas = torch.cat([sigmas, null])

                x = latent_unbatch["samples"].clone().to(default_dtype) 
                if latent_unbatch is not None:
                    if "samples_fp64" in latent_unbatch:
                        if latent_unbatch['samples'].shape == latent_unbatch['samples_fp64'].shape:
                            if torch.norm(latent_unbatch['samples'] - latent_unbatch['samples_fp64']) < 0.01:
                                x = latent_unbatch["samples_fp64"].clone()

                if latent_noise is not None:
                    latent_noise_samples       = latent_noise["samples"]      .clone().to(default_dtype)  
                if latent_noise_match is not None:
                    latent_noise_match_samples = latent_noise_match["samples"].clone().to(default_dtype)

                truncate_conditioning = extra_options_flag("truncate_conditioning", extra_options)
                if truncate_conditioning == "true" or truncate_conditioning == "true_and_zero_neg":
                    if pos_cond is not None:
                        pos_cond[0][0]                  = pos_cond[0][0]                 .clone().to(default_dtype)
                        pos_cond[0][1]["pooled_output"] = pos_cond[0][1]["pooled_output"].clone().to(default_dtype)
                    if neg_cond is not None:
                        neg_cond[0][0]                  = neg_cond[0][0]                 .clone().to(default_dtype)
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
                                d["pooled_output"] =  torch.zeros((1,2048),     dtype=t[0].dtype, device=t[0].device)
                                n                  = [torch.zeros((1,154,4096), dtype=t[0].dtype, device=t[0].device), d]
                            else:
                                d["pooled_output"] = d["pooled_output"][:, :2048]
                                n                  = [t[0][:, :154, :4096], d]
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
                            noise_sampler_init.k     = k_init
                            noise_sampler_init.scale = 0.1
                        noise = noise_sampler_init(sigma=sigmax, sigma_next=sigmin)
                    else:
                        noise = latent_noise_samples

                    if noise_normalize and noise.std() > 0:
                        #noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)
                        noise.sub_(noise.mean(dim=(-2, -1), keepdim=True)).div_(noise.std(dim=(-2, -1), keepdim=True))
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
                            sde_noise[i] = sde_noise[i]                  #WTF                            #WTF                                    #WTF
                            for j in range(sde_noise[i].shape[1]):
                                sde_noise[i][0][j] = ((sde_noise[i][0][j] - sde_noise[i][0][j].mean()) / sde_noise[i][0][j].std())
                                
                    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

                    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

                    sampler.extra_options['state_info'] = state_info
                    
                    samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, pos_cond, neg_cond, x.clone(), noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

                    out = latent_unbatch.copy()
                    out["samples"] = samples
                    if "x0" in x0_output:
                        out_denoised             = latent_unbatch.copy()
                        out_denoised["samples"]  = model.model.process_latent_out(x0_output["x0"].cpu())
                    else:
                        out_denoised = out
                    
                    out["samples_fp64"]          = out["samples"]         .clone()                    
                    out_denoised["samples_fp64"] = out_denoised["samples"].clone()
                    
                    out["samples"]               = out["samples"]         .to(latent_image_dtype)
                    out_denoised["samples"]      = out_denoised["samples"].to(latent_image_dtype)
                    
                    out_samples              .append(out["samples"])
                    out_samples_fp64         .append(out["samples_fp64"])
                    out_denoised_samples     .append(out_denoised["samples"])
                    out_denoised_samples_fp64.append(out_denoised["samples_fp64"])
                    
                    seed += 1
                    torch.manual_seed(seed)
                    if total_steps_iter > 1: 
                        sde_noise.append(out["samples_fp64"])
                        
            out_samples               = [tensor.squeeze(0) for tensor in out_samples]
            out_samples_fp64          = [tensor.squeeze(0) for tensor in out_samples_fp64]
            out_denoised_samples      = [tensor.squeeze(0) for tensor in out_denoised_samples]
            out_denoised_samples_fp64 = [tensor.squeeze(0) for tensor in out_denoised_samples_fp64]

            out['samples']      = torch.stack(out_samples,      dim=0)
            out['samples_fp64'] = torch.stack(out_samples_fp64, dim=0)
            
            out_denoised['samples']      = torch.stack(out_denoised_samples,      dim=0)
            out_denoised['samples_fp64'] = torch.stack(out_denoised_samples_fp64, dim=0)

            out['state_info'] = copy.deepcopy(state_info)
            state_info = {}
            #out['state_info'] = state_info

            return ( out, out_denoised, sde_noise,)



class ClownSamplerAdvanced_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "noise_type_sde":         (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_type_sde_substep": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                    "noise_mode_sde":         (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "noise_mode_sde_substep": (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot_mode":         (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How step size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "overshoot_mode_substep": (NOISE_MODE_NAMES,             {"default": 'hard',                                                        "tooltip": "How substep size overshoot scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                    "eta":                    ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "eta_substep":            ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "overshoot":              ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising step, then rescale to match the original. Has a softening effect."}),
                    "overshoot_substep":      ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Boost the size of each denoising substep, then rescale to match the original. Has a softening effect."}),
                    "noise_boost_step":       ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_boost_substep":    ("FLOAT",                      {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Set to positive values to create a sharper, grittier, more detailed image. Set to negative values to soften and deepen the colors."}),
                    "noise_anchor":           ("FLOAT",                      {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Typically set to between 1.0 and 0.0. Lower values cerate a grittier, more detailed image."}),
                    "s_noise":                ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "s_noise_substep":        ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Adds extra SDE noise. Values around 1.03-1.07 can lead to a moderate boost in detail and paint textures."}),
                    "d_noise":                ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01,                 "tooltip": "Downscales the sigma schedule. Values around 0.98-0.95 can lead to a large boost in detail and paint textures."}),
                    "noise_seed_sde":         ("INT",                        {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                    "sampler_name":           (get_sampler_name_list(),      {"default": get_default_sampler_name()}), 

                    "implicit_type":          (IMPLICIT_TYPE_NAMES,          {"default": "predictor-corrector"}), 
                    "implicit_type_substeps": (IMPLICIT_TYPE_NAMES,          {"default": "predictor-corrector"}), 
                    "implicit_steps":         ("INT",                        {"default": 0, "min": 0, "max": 10000}),
                    "implicit_substeps":      ("INT",                        {"default": 0, "min": 0, "max": 10000}),
                    "bongmath":               ("BOOLEAN",                    {"default": True}),
                    },
                "optional": 
                    {
                    "guides":                 ("GUIDES", ),     
                    "automation":             ("AUTOMATION", ),
                    "options":                ("OPTIONS", ),   
                    "extra_options":          ("STRING",                     {"default": "", "multiline": True}),   
                    }
                }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler", ) 
    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"
    
    def main(self, 
            noise_type_sde                = "gaussian",
            noise_type_sde_substep        = "gaussian",
            noise_mode_sde                = "hard",
            overshoot_mode                = "hard",
            overshoot_mode_substep        = "hard",
            
            eta                           = 0.5,
            eta_substep                   = 0.5,
            d_noise                       = 1.0,
            s_noise                       = 1.0,
            s_noise_substep               = 1.0,
            alpha_sde                     = -1.0,
            k_sde                         = 1.0,
            cfgpp                         = 0.0,
            c1                            = 0.0,
            c2                            = 0.5,
            c3                            = 1.0,
            noise_seed_sde                = -1,
            sampler_name                  = "res_2m",
            implicit_sampler_name         = "gauss-legendre_2s",
            
            implicit_substeps             = 0,
            implicit_steps                = 0,
            
            rescale_floor                 = True,
            sigmas_override               = None,
            
            guides                        = None,
            options                       = None,
            sde_noise                     = None,
            sde_noise_steps               = 1,
            
            extra_options                 = "",
            automation                    = None,
            etas                          = None,
            etas_substep                  = None,
            s_noises                      = None,
            s_noises_substep              = None,
            epsilon_scales                = None,
            regional_conditioning_weights = None,
            frame_weights_grp             = None,
            noise_mode_sde_substep        = "hard",
            
            overshoot                     = 0.0,
            overshoot_substep             = 0.0,
            noise_boost_step              = 0.0,
            noise_boost_substep           = 0.0,
            bongmath                      = True,
            noise_anchor                  = 1.0,
            
            implicit_type                 = "predictor-corrector",
            implicit_type_substeps        = "predictor-corrector",
            
            rk_swap_step                  = MAX_STEPS,
            rk_swap_print                 = False,
            rk_swap_threshold             = 0.0,
            rk_swap_type                  = "",
            
            **kwargs,
            ): 
        
            options_inputs = []

            if options is not None:
                options_inputs.append(options)

            i = 2
            while True:
                option_name = f"options {i}"
                if option_name in kwargs and kwargs[option_name] is not None:
                    options_inputs.append(kwargs[option_name])
                    i += 1
                else:
                    break

            options_mgr = OptionsManager(options_inputs)
    
            sampler_name, implicit_sampler_name = process_sampler_name(sampler_name)

            implicit_steps_diag = implicit_substeps
            implicit_steps_full = implicit_steps

            if noise_mode_sde == "none":
                eta = 0.0
                noise_mode_sde = "hard"

            default_dtype = getattr(torch, get_extra_options_kv("default_dtype", "float64", extra_options), torch.float64)

            noise_type_sde    = options_mgr.get('noise_type_sde'   , noise_type_sde)
            noise_mode_sde    = options_mgr.get('noise_mode_sde'   , noise_mode_sde)
            eta               = options_mgr.get('eta'              , eta)
            s_noise           = options_mgr.get('s_noise'          , s_noise)
            d_noise           = options_mgr.get('d_noise'          , d_noise)
            alpha_sde         = options_mgr.get('alpha_sde'        , alpha_sde)
            k_sde             = options_mgr.get('k_sde'            , k_sde)
            c1                = options_mgr.get('c1'               , c1)
            c2                = options_mgr.get('c2'               , c2)
            c3                = options_mgr.get('c3'               , c3)

            frame_weights_grp = options_mgr.get('frame_weights_grp', frame_weights_grp)
            sde_noise         = options_mgr.get('sde_noise'        , sde_noise)
            sde_noise_steps   = options_mgr.get('sde_noise_steps'  , sde_noise_steps)
            
            rk_swap_step      = options_mgr.get('rk_swap_step'     , rk_swap_step)
            rk_swap_print     = options_mgr.get('rk_swap_print'    , rk_swap_print)
            rk_swap_threshold = options_mgr.get('rk_swap_threshold', rk_swap_threshold)
            rk_swap_type      = options_mgr.get('rk_swap_type'     , rk_swap_type)


            rescale_floor = extra_options_flag("rescale_floor", extra_options)

            if automation is not None:
                etas              = automation['etas']              if 'etas'              in automation else None
                etas_substep      = automation['etas_substep']      if 'etas_substep'      in automation else None
                s_noises          = automation['s_noises']          if 's_noises'          in automation else None
                s_noises_substep  = automation['s_noise_substep']   if 's_noise_substep'   in automation else None
                epsilon_scales    = automation['epsilon_scales']    if 'epsilon_scales'    in automation else None
                frame_weights_grp = automation['frame_weights_grp'] if 'frame_weights_grp' in automation else None

            etas             = initialize_or_scale(etas,             eta,             MAX_STEPS).to(default_dtype)
            etas_substep     = initialize_or_scale(etas_substep,     eta_substep,     MAX_STEPS).to(default_dtype)
            s_noises         = initialize_or_scale(s_noises,         s_noise,         MAX_STEPS).to(default_dtype)
            s_noises_substep = initialize_or_scale(s_noises_substep, s_noise_substep, MAX_STEPS).to(default_dtype)

            etas             = F.pad(etas,             (0, MAX_STEPS), value=0.0)
            etas_substep     = F.pad(etas_substep,     (0, MAX_STEPS), value=0.0)
            s_noises         = F.pad(s_noises,         (0, MAX_STEPS), value=1.0)
            s_noises_substep = F.pad(s_noises_substep, (0, MAX_STEPS), value=1.0)

            if sde_noise is None:
                sde_noise = []
            else:
                sde_noise = copy.deepcopy(sde_noise)
                for i in range(len(sde_noise)):
                    sde_noise[i] = sde_noise[i]
                    for j in range(sde_noise[i].shape[1]):
                        sde_noise[i][0][j] = ((sde_noise[i][0][j] - sde_noise[i][0][j].mean()) / sde_noise[i][0][j].std())
                        

            sampler = comfy.samplers.ksampler("rk_beta", 
                {
                    "eta"                           : eta,
                    "s_noise"                       : s_noise,
                    "s_noise_substep"               : s_noise_substep,
                    "d_noise"                       : d_noise,
                    "alpha"                         : alpha_sde,
                    "k"                             : k_sde,
                    "c1"                            : c1,
                    "c2"                            : c2,
                    "c3"                            : c3,
                    "cfgpp"                         : cfgpp,

                    "noise_sampler_type"            : noise_type_sde,
                    "noise_sampler_type_substep"    : noise_type_sde_substep,
                    "noise_mode_sde"                : noise_mode_sde,
                    "noise_seed"                    : noise_seed_sde,
                    "rk_type"                       : sampler_name,
                    "implicit_sampler_name"         : implicit_sampler_name,

                    "implicit_steps_diag"           : implicit_steps_diag,
                    "implicit_steps_full"           : implicit_steps_full,

                    "LGW_MASK_RESCALE_MIN"          : rescale_floor,
                    "sigmas_override"               : sigmas_override,
                    "sde_noise"                     : sde_noise,

                    "extra_options"                 : extra_options,
                    "sampler_mode"                  : "standard",

                    "etas"                          : etas,
                    "etas_substep"                  : etas_substep,
                    "s_noises"                      : s_noises,
                    "s_noises_substep"              : s_noises_substep,
                    "epsilon_scales"                : epsilon_scales,
                    "regional_conditioning_weights" : regional_conditioning_weights,

                    "guides"                        : guides,
                    "frame_weights_grp"             : frame_weights_grp,
                    "eta_substep"                   : eta_substep,
                    "noise_mode_sde_substep"        : noise_mode_sde_substep,
                    "noise_boost_step"              : noise_boost_step,
                    "noise_boost_substep"           : noise_boost_substep,

                    "overshoot_mode"                : overshoot_mode,
                    "overshoot_mode_substep"        : overshoot_mode_substep,
                    "overshoot"                     : overshoot,
                    "overshoot_substep"             : overshoot_substep,
                    "BONGMATH"                      : bongmath,
                    "noise_anchor"                  : noise_anchor,

                    "implicit_type"                 : implicit_type,
                    "implicit_type_substeps"        : implicit_type_substeps,
                    
                    "rk_swap_step"                  : rk_swap_step,
                    "rk_swap_print"                 : rk_swap_print,
                    "rk_swap_threshold"             : rk_swap_threshold,
                    "rk_swap_type"                  : rk_swap_type,
                })


            return (sampler, )







class ClownsharKSampler_Beta:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required":
                    {
                    "model":        ("MODEL",),
                    "eta":          ("FLOAT",                      {"default": 0.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, "tooltip": "Calculated noise amount to be added, then removed, after each step."}),
                    "sampler_name": (get_sampler_name_list     (), {"default": get_default_sampler_name()}), 
                    "scheduler":    (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                    "steps":        ("INT",                        {"default": 30,  "min": 1,      "max": 10000}),
                    "denoise":      ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                    "cfg":          ("FLOAT",                      {"default": 5.5, "min": -100.0, "max": 100.0, "step":0.01, "round": False, }),
                    "seed":         ("INT",                        {"default": 0,   "min": -1,     "max": 0xffffffffffffffff}),
                    "sampler_mode": (['standard', 'unsample', 'resample'],),
                    "bongmath":     ("BOOLEAN",                    {"default": True}),
                    },
                "optional": 
                    {
                    "positive":     ("CONDITIONING",),
                    "negative":     ("CONDITIONING",),
                    "latent_image": ("LATENT",),
                    "sigmas":       ("SIGMAS",), 
                    "guides":       ("GUIDES",), 
                    "options":      ("OPTIONS", {}),   
                    }
                }
        
        return inputs

    RETURN_TYPES = ("LATENT", 
                    "LATENT",)
    
    RETURN_NAMES = ("output", 
                    "denoised",) 
    
    FUNCTION = "main"
    CATEGORY = "RES4LYF/samplers"
    
    def main(self, 
            model                         = None,
            denoise                       = 1.0, 
            scheduler                     = "beta57", 
            cfg                           = 1.0, 
            seed                          = 42, 
            positive                      = None, 
            negative                      = None, 
            latent_image                  = None, 
            steps                         = 30,
            bongmath                      = True,
            sampler_mode                  = "standard",
            
            noise_type_sde                = "gaussian", 
            noise_type_sde_substep        = "gaussian", 
            noise_mode_sde                = "hard",
            noise_mode_sde_substep        = "hard",

            
            overshoot_mode                = "hard", 
            overshoot_mode_substep        = "hard",
            overshoot                     = 0.0, 
            overshoot_substep             = 0.0,
            
            eta                           = 0.5, 
            eta_substep                   = 0.5,
            d_noise                       = 1.0, 
            s_noise                       = 1.0, 
            s_noise_substep               = 1.0, 
            
            alpha_sde                     = -1.0, 
            k_sde                         = 1.0,
            cfgpp                         = 0.0,
            c1                            = 0.0, 
            c2                            = 0.5, 
            c3                            = 1.0,
            noise_seed_sde                = -1,
            sampler_name                  = "res_2m", 
            implicit_sampler_name         = "use_explicit",

            implicit_type                 = "bongmath",
            implicit_type_substeps        = "bongmath",
            implicit_steps                = 0,
            implicit_substeps             = 0, 

            sigmas                        = None,
            sigmas_override               = None, 
            guides                        = None, 
            options                       = None, 
            sde_noise                     = None,
            sde_noise_steps               = 1, 
            extra_options                 = "", 
            automation                    = None, 

            epsilon_scales                = None, 
            regional_conditioning_weights = None,
            frame_weights_grp             = None, 
            noise_boost_step              = 0.0, 
            noise_boost_substep           = 0.0, 
            noise_anchor                  = 1.0,

            rescale_floor                 = True, 
            
            rk_swap_step                  = MAX_STEPS,
            rk_swap_print                 = False,
            rk_swap_threshold             = 0.0,
            rk_swap_type                  = "",
            
            **kwargs
            ): 
        
        options_inputs = []

        if options is not None:
            options_inputs.append(options)

        i = 2
        while True:
            option_name = f"options {i}"
            if option_name in kwargs and kwargs[option_name] is not None:
                options_inputs.append(kwargs[option_name])
                i += 1
            else:
                break

        options_mgr = OptionsManager(options_inputs)

        #noise_seed_sde = seed+1
        
        
        # defaults for ClownSampler
        eta_substep = eta
        
        # defaults for SharkSampler
        noise_type_init = "gaussian"
        noise_stdev     = 1.0
        denoise_alt     = 1.0
        channelwise_cfg = 1.0
        
        
        #if options is not None:
        options_mgr = OptionsManager(options_inputs)
        noise_type_sde         = options_mgr.get('noise_type_sde'        , noise_type_sde)
        noise_type_sde_substep = options_mgr.get('noise_type_sde_substep', noise_type_sde_substep)
        
        noise_mode_sde         = options_mgr.get('noise_mode_sde'        , noise_mode_sde)
        noise_mode_sde_substep = options_mgr.get('noise_mode_sde_substep', noise_mode_sde_substep)
        
        overshoot_mode         = options_mgr.get('overshoot_mode'        , overshoot_mode)
        overshoot_mode_substep = options_mgr.get('overshoot_mode_substep', overshoot_mode_substep)

        eta                    = options_mgr.get('eta'                   , eta)
        eta_substep            = options_mgr.get('eta_substep'           , eta_substep)

        overshoot              = options_mgr.get('overshoot'             , overshoot)
        overshoot_substep      = options_mgr.get('overshoot_substep'     , overshoot_substep)
        
        noise_boost_step       = options_mgr.get('noise_boost_step'      , noise_boost_step)
        noise_boost_substep    = options_mgr.get('noise_boost_substep'   , noise_boost_substep)
        
        noise_anchor           = options_mgr.get('noise_anchor'          , noise_anchor)

        s_noise                = options_mgr.get('s_noise'               , s_noise)
        s_noise_substep        = options_mgr.get('s_noise_substep'       , s_noise_substep)

        d_noise                = options_mgr.get('d_noise'               , d_noise)
        
        implicit_type          = options_mgr.get('implicit_type'         , implicit_type)
        implicit_type_substeps = options_mgr.get('implicit_type_substeps', implicit_type_substeps)
        implicit_steps         = options_mgr.get('implicit_steps'        , implicit_steps)
        implicit_substeps      = options_mgr.get('implicit_substeps'     , implicit_substeps)
        
        alpha_sde              = options_mgr.get('alpha_sde'             , alpha_sde)
        k_sde                  = options_mgr.get('k_sde'                 , k_sde)
        c1                     = options_mgr.get('c1'                    , c1)
        c2                     = options_mgr.get('c2'                    , c2)
        c3                     = options_mgr.get('c3'                    , c3)

        frame_weights_grp      = options_mgr.get('frame_weights_grp'     , frame_weights_grp)
        
        sde_noise              = options_mgr.get('sde_noise'             , sde_noise)
        sde_noise_steps        = options_mgr.get('sde_noise_steps'       , sde_noise_steps)
        
        extra_options          = options_mgr.get('extra_options'         , extra_options)
        
        automation             = options_mgr.get('automation'            , automation)
        
        # SharkSampler Options
        noise_type_init        = options_mgr.get('noise_type_init'       , noise_type_init)
        noise_stdev            = options_mgr.get('noise_stdev'           , noise_stdev)
        sampler_mode           = options_mgr.get('sampler_mode'          , sampler_mode)
        denoise_alt            = options_mgr.get('denoise_alt'           , denoise_alt)
        
        channelwise_cfg        = options_mgr.get('channelwise_cfg'       , channelwise_cfg)
        
        sigmas                 = options_mgr.get('sigmas'                , sigmas)
        
        rk_swap_type           = options_mgr.get('rk_swap_type'          , rk_swap_type)
        rk_swap_step           = options_mgr.get('rk_swap_step'          , rk_swap_step)
        rk_swap_threshold      = options_mgr.get('rk_swap_threshold'     , rk_swap_threshold)
        rk_swap_print          = options_mgr.get('rk_swap_print'         , rk_swap_print)
        
        if channelwise_cfg:
            cfg = -abs(cfg)  # set cfg negative for shark, to flag as cfg_cw




        sampler = ClownSamplerAdvanced_Beta().main(
            noise_type_sde                = noise_type_sde,
            noise_type_sde_substep        = noise_type_sde_substep,
            noise_mode_sde                = noise_mode_sde,
            noise_mode_sde_substep        = noise_mode_sde_substep,
            
            eta                           = eta,
            eta_substep                   = eta_substep,
            
            s_noise                       = s_noise,
            s_noise_substep               = s_noise_substep,
            
            overshoot                     = overshoot,
            overshoot_substep             = overshoot_substep,
            
            overshoot_mode                = overshoot_mode,
            overshoot_mode_substep        = overshoot_mode_substep,
            
            d_noise                       = d_noise,

            alpha_sde                     = alpha_sde,
            k_sde                         = k_sde,
            cfgpp                         = cfgpp,
            c1                            = c1,
            c2                            = c2,
            c3                            = c3,
            sampler_name                  = sampler_name,
            implicit_sampler_name         = implicit_sampler_name,

            implicit_type                 = implicit_type,
            implicit_type_substeps        = implicit_type_substeps,
            implicit_steps                = implicit_steps,
            implicit_substeps             = implicit_substeps,

            rescale_floor                 = rescale_floor,
            sigmas_override               = sigmas_override,
            
            noise_seed_sde                = noise_seed_sde,
            
            guides                        = guides,
            options                       = options_mgr.as_dict(),

            extra_options                 = extra_options,
            automation                    = automation,

            noise_boost_step              = noise_boost_step,
            noise_boost_substep           = noise_boost_substep,
            
            epsilon_scales                = epsilon_scales,
            regional_conditioning_weights = regional_conditioning_weights,
            frame_weights_grp             = frame_weights_grp,
            
            sde_noise                     = sde_noise,
            sde_noise_steps               = sde_noise_steps,
            
            rk_swap_step                  = rk_swap_step,
            rk_swap_print                 = rk_swap_print,
            rk_swap_threshold             = rk_swap_threshold,
            rk_swap_type                  = rk_swap_type,
                        
            bongmath                      = bongmath,
            )
            
        
        output, denoised, sde_noise = SharkSampler().main(
            model           = model, 
            cfg             = cfg, 
            scheduler       = scheduler,
            steps           = steps, 
            denoise         = denoise,
            latent_image    = latent_image, 
            positive        = positive,
            negative        = negative, 
            sampler         = sampler[0], 
            cfgpp           = cfgpp, 
            noise_seed      = seed, 
            options         = options_mgr.as_dict(), 
            sde_noise       = sde_noise, 
            sde_noise_steps = sde_noise_steps, 
            noise_type_init = noise_type_init,
            noise_stdev     = noise_stdev,
            sampler_mode    = sampler_mode,
            denoise_alt     = denoise_alt,
            sigmas          = sigmas,

            extra_options   = extra_options)
        
        return (output, denoised,)


