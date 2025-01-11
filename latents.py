import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
    
import itertools

import torch
import math

from .noise_classes import *

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor
    
def latent_normalize_channels(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std  = x.std (dim=(2, 3), keepdim=True)
    return  (x - mean) / std

def latent_stdize_channels(x):
    std  = x.std (dim=(2, 3), keepdim=True)
    return  x / std

def latent_meancenter_channels(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    return  x - mean


def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor


def normalize_latent(target, source=None, mean=True, std=True, set_mean=None, set_std=None, channelwise=True):
    target = target.clone()
    source = source.clone() if source is not None else None
    def normalize_single_latent(single_target, single_source=None):
        y = torch.zeros_like(single_target)
        for b in range(y.shape[0]):
            if channelwise:
                for c in range(y.shape[1]):
                    single_source_mean = single_source[b][c].mean() if set_mean is None else set_mean
                    single_source_std  = single_source[b][c].std()  if set_std  is None else set_std
                    
                    if mean and std:
                        y[b][c] = (single_target[b][c] - single_target[b][c].mean()) / single_target[b][c].std()
                        if single_source is not None:
                            y[b][c] = y[b][c] * single_source_std + single_source_mean
                    elif mean:
                        y[b][c] = single_target[b][c] - single_target[b][c].mean()
                        if single_source is not None:
                            y[b][c] = y[b][c] + single_source_mean
                    elif std:
                        y[b][c] = single_target[b][c] / single_target[b][c].std()
                        if single_source is not None:
                            y[b][c] = y[b][c] * single_source_std
            else:
                single_source_mean = single_source[b].mean() if set_mean is None else set_mean
                single_source_std  = single_source[b].std()  if set_std  is None else set_std
                
                if mean and std:
                    y[b] = (single_target[b] - single_target[b].mean()) / single_target[b].std()
                    if single_source is not None:
                        y[b] = y[b] * single_source_std + single_source_mean
                elif mean:
                    y[b] = single_target[b] - single_target[b].mean()
                    if single_source is not None:
                        y[b] = y[b] + single_source_mean
                elif std:
                    y[b] = single_target[b] / single_target[b].std()
                    if single_source is not None:
                        y[b] = y[b] * single_source_std
        return y

    if isinstance(target, (list, tuple)):
        if source is not None:
            assert isinstance(source, (list, tuple)) and len(source) == len(target), \
                "If target is a list/tuple, source must be a list/tuple of the same length."
            return [normalize_single_latent(t, s) for t, s in zip(target, source)]
        else:
            return [normalize_single_latent(t) for t in target]
    else:
        return normalize_single_latent(target, source)




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
    CATEGORY = "RES4LYF/noise"

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

    CATEGORY = "RES4LYF/noise"
    
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





class set_precision:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent_image": ("LATENT", ),      
                    "precision": (["16", "32", "64"], ),
                    "set_default": ("BOOLEAN", {"default": False})
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "RES4LYF/precision"

    FUNCTION = "main"

    def main(self, precision="32", latent_image=None, set_default=False):
        match precision:
            case "16":
                if set_default is True:
                    torch.set_default_dtype(torch.float16)
                x = latent_image["samples"].to(torch.float16)
            case "32":
                if set_default is True:
                    torch.set_default_dtype(torch.float32)
                x = latent_image["samples"].to(torch.float32)
            case "64":
                if set_default is True:
                    torch.set_default_dtype(torch.float64)
                x = latent_image["samples"].to(torch.float64)
        return ({"samples": x}, )
    

class set_precision_universal:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "precision": (["bf16", "fp16", "fp32", "fp64", "passthrough"], {"default": "fp32"}),
                    "set_default": ("BOOLEAN", {"default": False})
                    },
            "optional": {
                    "cond_pos": ("CONDITIONING",),
                    "cond_neg": ("CONDITIONING",),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "SIGMAS", "LATENT",)
    RETURN_NAMES = ("cond_pos","cond_neg","sigmas","latent_image",)
    CATEGORY = "RES4LYF/precision"

    FUNCTION = "main"

    def main(self, precision="fp32", cond_pos=None, cond_neg=None, sigmas=None, latent_image=None, set_default=False):
        dtype = None
        match precision:
            case "bf16":
                dtype = torch.bfloat16
            case "fp16":
                dtype = torch.float16
            case "fp32":
                dtype = torch.float32
            case "fp64":
                dtype = torch.float64
            case "passthrough":
                return (cond_pos, cond_neg, sigmas, latent_image, )
        
        if cond_pos is not None:
            cond_pos[0][0] = cond_pos[0][0].clone().to(dtype)
            cond_pos[0][1]["pooled_output"] = cond_pos[0][1]["pooled_output"].clone().to(dtype)
        
        if cond_neg is not None:
            cond_neg[0][0] = cond_neg[0][0].clone().to(dtype)
            cond_neg[0][1]["pooled_output"] = cond_neg[0][1]["pooled_output"].clone().to(dtype)
            
        if sigmas is not None:
            sigmas = sigmas.clone().to(dtype)
        
        if latent_image is not None:
            x = latent_image["samples"].clone().to(dtype)    
            latent_image = {"samples": x}

        if set_default is True:
            torch.set_default_dtype(dtype)
        
        return (cond_pos, cond_neg, sigmas, latent_image, )
    
    
class set_precision_advanced:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent_image": ("LATENT", ),      
                    "global_precision": (["64", "32", "16"], ),
                    "shark_precision": (["64", "32", "16"], ),
                     },
                }

    RETURN_TYPES = ("LATENT","LATENT","LATENT","LATENT","LATENT",)
    RETURN_NAMES = ("PASSTHROUGH","LATENT_CAST_TO_GLOBAL","LATENT_16","LATENT_32","LATENT_64",)
    CATEGORY = "RES4LYF/precision"

    FUNCTION = "main"

    def main(self, global_precision="32", shark_precision="64", latent_image=None):
        dtype_map = {
            "16": torch.float16,
            "32": torch.float32,
            "64": torch.float64
        }
        precision_map = {
            "16": 'fp16',
            "32": 'fp32',
            "64": 'fp64'
        }

        torch.set_default_dtype(dtype_map[global_precision])
        precision_tool.set_cast_type(precision_map[shark_precision])

        latent_passthrough = latent_image["samples"]

        latent_out16 = latent_image["samples"].to(torch.float16)
        latent_out32 = latent_image["samples"].to(torch.float32)
        latent_out64 = latent_image["samples"].to(torch.float64)

        target_dtype = dtype_map[global_precision]
        if latent_image["samples"].dtype != target_dtype:
            latent_image["samples"] = latent_image["samples"].to(target_dtype)

        latent_cast_to_global = latent_image["samples"]

        return ({"samples": latent_passthrough}, {"samples": latent_cast_to_global}, {"samples": latent_out16}, {"samples": latent_out32}, {"samples": latent_out64})
    
class latent_to_cuda:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", ),      
                    "to_cuda": ("BOOLEAN", {"default": True}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "RES4LYF/latents"

    FUNCTION = "main"

    def main(self, latent, to_cuda):
        match to_cuda:
            case "True":
                latent = latent.to('cuda')
            case "False":
                latent = latent.to('cpu')
        return (latent,)

class latent_batch:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", ),      
                    "batch_size": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    CATEGORY = "RES4LYF/latents"

    FUNCTION = "main"

    def main(self, latent, batch_size):
        latent = latent["samples"]
        b, c, h, w = latent.shape
        batch_latents = torch.zeros([batch_size, 4, h, w], device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return ({"samples": batch_latents}, )

class LatentPhaseMagnitude:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_0_batch": ("LATENT",),
                "latent_1_batch": ("LATENT",),

                "phase_mix_power": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_mix_power": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "phase_luminosity": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal": ("BOOLEAN", {"default": True}),
                "latent_1_normal": ("BOOLEAN", {"default": True}),
                "latent_out_normal": ("BOOLEAN", {"default": True}),
                "latent_0_stdize": ("BOOLEAN", {"default": True}),
                "latent_1_stdize": ("BOOLEAN", {"default": True}),
                "latent_out_stdize": ("BOOLEAN", {"default": True}),
                "latent_0_meancenter": ("BOOLEAN", {"default": True}),
                "latent_1_meancenter": ("BOOLEAN", {"default": True}),
                "latent_out_meancenter": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "phase_mix_powers": ("SIGMAS", ),
                "magnitude_mix_powers": ("SIGMAS", ),

                "phase_luminositys": ("SIGMAS", ),
                "phase_cyan_reds": ("SIGMAS", ),
                "phase_lime_purples": ("SIGMAS", ),
                "phase_pattern_structures": ("SIGMAS", ),

                "magnitude_luminositys": ("SIGMAS", ),
                "magnitude_cyan_reds": ("SIGMAS", ),
                "magnitude_lime_purples": ("SIGMAS", ),
                "magnitude_pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/latents"
    
    @staticmethod
    def latent_repeat(latent, batch_size):
        b, c, h, w = latent.shape
        batch_latents = torch.zeros((batch_size, c, h, w), dtype=latent.dtype, layout=latent.layout, device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return batch_latents

    @staticmethod
    def mix_latent_phase_magnitude(latent_0, latent_1, power_phase, power_magnitude,
                                    phase_luminosity, phase_cyan_red, phase_lime_purple, phase_pattern_structure,
                                    magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure
                                    ):
        dtype = torch.promote_types(latent_0.dtype, latent_1.dtype)
        # big accuracy problems with fp32 FFT! let's avoid that
        latent_0 = latent_0.double()
        latent_1 = latent_1.double()

        latent_0_fft = torch.fft.fft2(latent_0)
        latent_1_fft = torch.fft.fft2(latent_1)

        latent_0_phase = torch.angle(latent_0_fft)
        latent_1_phase = torch.angle(latent_1_fft)
        latent_0_magnitude = torch.abs(latent_0_fft)
        latent_1_magnitude = torch.abs(latent_1_fft)

        # DC corruption...? handle separately??
        #dc_index = (0, 0)
        #dc_0 = latent_0_fft[:, :, dc_index[0], dc_index[1]]
        #dc_1 = latent_1_fft[:, :, dc_index[0], dc_index[1]]
        #mixed_dc = dc_0 * 0.5 + dc_1 * 0.5
        #mixed_dc = dc_0 * (1 - phase_weight) + dc_1 * phase_weight

        # create complex FFT using a weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
        mixed_phase     = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)
        mixed_magnitude = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)

        for i in range(4):
            mixed_phase[:, i]     = ( (latent_0_phase[:,i] * (1-chan_weights_phase[i])) ** power_phase + (latent_1_phase[:,i] * chan_weights_phase[i]) ** power_phase) ** (1/power_phase)
            mixed_magnitude[:, i]     = ( (latent_0_magnitude[:,i] * (1-chan_weights_magnitude[i])) ** power_magnitude + (latent_1_magnitude[:,i] * chan_weights_magnitude[i]) ** power_magnitude) ** (1/power_magnitude)

        new_fft = mixed_magnitude * torch.exp(1j * mixed_phase)

        #new_fft[:, :, dc_index[0], dc_index[1]] = mixed_dc

        # inverse FFT to convert back to spatial domain
        mixed_phase_magnitude = torch.fft.ifft2(new_fft).real

        return mixed_phase_magnitude.to(dtype)
    
    def main(self, #batch_size, latent_1_repeat,
             latent_0_batch,  latent_1_batch, latent_0_normal, latent_1_normal, latent_out_normal,
             latent_0_stdize, latent_1_stdize, latent_out_stdize, 
             latent_0_meancenter, latent_1_meancenter, latent_out_meancenter, 
             phase_mix_power, magnitude_mix_power, 
             phase_luminosity,           phase_cyan_red,           phase_lime_purple,           phase_pattern_structure, 
             magnitude_luminosity,       magnitude_cyan_red,       magnitude_lime_purple,       magnitude_pattern_structure, 
             phase_mix_powers=None,      magnitude_mix_powers=None,
             phase_luminositys=None,     phase_cyan_reds=None,     phase_lime_purples=None,     phase_pattern_structures=None,
             magnitude_luminositys=None, magnitude_cyan_reds=None, magnitude_lime_purples=None, magnitude_pattern_structures=None
             ):
        latent_0_batch = latent_0_batch["samples"].double()
        latent_1_batch = latent_1_batch["samples"].double().to(latent_0_batch.device)

        #if batch_size == 0:
        batch_size = latent_0_batch.shape[0]
        if latent_1_batch.shape[0] == 1:
            latent_1_batch = self.latent_repeat(latent_1_batch, batch_size)


        magnitude_mix_powers         = initialize_or_scale(magnitude_mix_powers,         magnitude_mix_power,         batch_size)
        phase_mix_powers             = initialize_or_scale(phase_mix_powers,             phase_mix_power,            batch_size)

        phase_luminositys            = initialize_or_scale(phase_luminositys,            phase_luminosity,            batch_size)
        phase_cyan_reds              = initialize_or_scale(phase_cyan_reds,              phase_cyan_red,              batch_size)
        phase_lime_purples           = initialize_or_scale(phase_lime_purples,           phase_lime_purple,           batch_size)
        phase_pattern_structures     = initialize_or_scale(phase_pattern_structures,     phase_pattern_structure,     batch_size)

        magnitude_luminositys        = initialize_or_scale(magnitude_luminositys,        magnitude_luminosity,        batch_size)
        magnitude_cyan_reds          = initialize_or_scale(magnitude_cyan_reds,          magnitude_cyan_red,          batch_size)
        magnitude_lime_purples       = initialize_or_scale(magnitude_lime_purples,       magnitude_lime_purple,       batch_size)
        magnitude_pattern_structures = initialize_or_scale(magnitude_pattern_structures, magnitude_pattern_structure, batch_size)    

        mixed_phase_magnitude_batch = torch.zeros(latent_0_batch.shape, device=latent_0_batch.device)

        if latent_0_normal == True:
            latent_0_batch = latent_normalize_channels(latent_0_batch)
        if latent_1_normal == True:
            latent_1_batch = latent_normalize_channels(latent_1_batch)
        if latent_0_meancenter == True:
            latent_0_batch = latent_meancenter_channels(latent_0_batch)
        if latent_1_meancenter == True:
            latent_1_batch = latent_meancenter_channels(latent_1_batch)
        if latent_0_stdize == True:
            latent_0_batch = latent_stdize_channels(latent_0_batch)
        if latent_1_stdize == True:
            latent_1_batch = latent_stdize_channels(latent_1_batch)
 
        for i in range(batch_size):
            mixed_phase_magnitude = self.mix_latent_phase_magnitude(latent_0_batch[i:i+1], latent_1_batch[i:i+1], phase_mix_powers[i].item(), magnitude_mix_powers[i].item(),
                                                    phase_luminositys[i].item(), phase_cyan_reds[i].item(),phase_lime_purples[i].item(),phase_pattern_structures[i].item(),
                                                    magnitude_luminositys[i].item(), magnitude_cyan_reds[i].item(),magnitude_lime_purples[i].item(),magnitude_pattern_structures[i].item()
                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)
            if latent_out_stdize == True:
                mixed_phase_magnitude = latent_stdize_channels(mixed_phase_magnitude)
            if latent_out_meancenter == True:
                mixed_phase_magnitude = latent_meancenter_channels(mixed_phase_magnitude)                                

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )

class LatentPhaseMagnitudeMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_0_batch": ("LATENT",),

                "phase_luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal": ("BOOLEAN", {"default": False}),
                "latent_out_normal": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_luminositys": ("SIGMAS", ),
                "phase_cyan_reds": ("SIGMAS", ),
                "phase_lime_purples": ("SIGMAS", ),
                "phase_pattern_structures": ("SIGMAS", ),

                "magnitude_luminositys": ("SIGMAS", ),
                "magnitude_cyan_reds": ("SIGMAS", ),
                "magnitude_lime_purples": ("SIGMAS", ),
                "magnitude_pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/latents"

    @staticmethod
    def latent_repeat(latent, batch_size):
        b, c, h, w = latent.shape
        batch_latents = torch.zeros((batch_size, c, h, w), dtype=latent.dtype, layout=latent.layout, device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return batch_latents

    @staticmethod
    def mix_latent_phase_magnitude(latent_0,  
                                    phase_luminosity, phase_cyan_red, phase_lime_purple, phase_pattern_structure,
                                    magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure
                                    ):
        dtype = latent_0.dtype
        # avoid big accuracy problems with fp32 FFT!
        latent_0 = latent_0.double()

        latent_0_fft = torch.fft.fft2(latent_0)

        latent_0_phase = torch.angle(latent_0_fft)
        latent_0_magnitude = torch.abs(latent_0_fft)

        # create new complex FFT using weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [ w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
        mixed_phase     = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)
        mixed_magnitude = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)

        for i in range(4):
            mixed_phase[:, i]     = latent_0_phase[:,i]     * chan_weights_phase[i]
            mixed_magnitude[:, i] = latent_0_magnitude[:,i] * chan_weights_magnitude[i]

        new_fft = mixed_magnitude * torch.exp(1j * mixed_phase)
        
        # inverse FFT to convert back to spatial domain
        mixed_phase_magnitude = torch.fft.ifft2(new_fft).real

        return mixed_phase_magnitude.to(dtype)
    
    def main(self,
             latent_0_batch, latent_0_normal, latent_out_normal,
             phase_luminosity,           phase_cyan_red,           phase_lime_purple,           phase_pattern_structure, 
             magnitude_luminosity,       magnitude_cyan_red,       magnitude_lime_purple,       magnitude_pattern_structure, 
             phase_luminositys=None,     phase_cyan_reds=None,     phase_lime_purples=None,     phase_pattern_structures=None,
             magnitude_luminositys=None, magnitude_cyan_reds=None, magnitude_lime_purples=None, magnitude_pattern_structures=None
             ):
        latent_0_batch = latent_0_batch["samples"].double()

        batch_size = latent_0_batch.shape[0]

        phase_luminositys            = initialize_or_scale(phase_luminositys,            phase_luminosity,            batch_size)
        phase_cyan_reds              = initialize_or_scale(phase_cyan_reds,              phase_cyan_red,              batch_size)
        phase_lime_purples           = initialize_or_scale(phase_lime_purples,           phase_lime_purple,           batch_size)
        phase_pattern_structures     = initialize_or_scale(phase_pattern_structures,     phase_pattern_structure,     batch_size)

        magnitude_luminositys        = initialize_or_scale(magnitude_luminositys,        magnitude_luminosity,        batch_size)
        magnitude_cyan_reds          = initialize_or_scale(magnitude_cyan_reds,          magnitude_cyan_red,          batch_size)
        magnitude_lime_purples       = initialize_or_scale(magnitude_lime_purples,       magnitude_lime_purple,       batch_size)
        magnitude_pattern_structures = initialize_or_scale(magnitude_pattern_structures, magnitude_pattern_structure, batch_size)    

        mixed_phase_magnitude_batch = torch.zeros(latent_0_batch.shape, device=latent_0_batch.device)

        if latent_0_normal == True:
            latent_0_batch = latent_normalize_channels(latent_0_batch)
 
        for i in range(batch_size):
            mixed_phase_magnitude = self.mix_latent_phase_magnitude(latent_0_batch[i:i+1],
                                                    phase_luminositys[i].item(), phase_cyan_reds[i].item(),phase_lime_purples[i].item(),phase_pattern_structures[i].item(),
                                                    magnitude_luminositys[i].item(), magnitude_cyan_reds[i].item(),magnitude_lime_purples[i].item(),magnitude_pattern_structures[i].item()
                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )



class LatentPhaseMagnitudeOffset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_0_batch": ("LATENT",),

                "phase_luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal": ("BOOLEAN", {"default": False}),
                "latent_out_normal": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_luminositys": ("SIGMAS", ),
                "phase_cyan_reds": ("SIGMAS", ),
                "phase_lime_purples": ("SIGMAS", ),
                "phase_pattern_structures": ("SIGMAS", ),

                "magnitude_luminositys": ("SIGMAS", ),
                "magnitude_cyan_reds": ("SIGMAS", ),
                "magnitude_lime_purples": ("SIGMAS", ),
                "magnitude_pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/latents"
    
    @staticmethod
    def latent_repeat(latent, batch_size):
        b, c, h, w = latent.shape
        batch_latents = torch.zeros((batch_size, c, h, w), dtype=latent.dtype, layout=latent.layout, device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return batch_latents

    @staticmethod
    def mix_latent_phase_magnitude(latent_0,  
                                    phase_luminosity, phase_cyan_red, phase_lime_purple, phase_pattern_structure,
                                    magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure
                                    ):
        dtype = latent_0.dtype
        # avoid big accuracy problems with fp32 FFT!
        latent_0 = latent_0.double()

        latent_0_fft = torch.fft.fft2(latent_0)

        latent_0_phase = torch.angle(latent_0_fft)
        latent_0_magnitude = torch.abs(latent_0_fft)

        # create new complex FFT using a weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [ w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
        mixed_phase     = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)
        mixed_magnitude = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)

        for i in range(4):
            mixed_phase[:, i]     = latent_0_phase[:,i]     + chan_weights_phase[i]
            mixed_magnitude[:, i] = latent_0_magnitude[:,i] + chan_weights_magnitude[i]

        new_fft = mixed_magnitude * torch.exp(1j * mixed_phase)
        
        # inverse FFT to convert back to spatial domain
        mixed_phase_magnitude = torch.fft.ifft2(new_fft).real

        return mixed_phase_magnitude.to(dtype)
    
    def main(self,
             latent_0_batch, latent_0_normal, latent_out_normal,
             phase_luminosity,           phase_cyan_red,           phase_lime_purple,           phase_pattern_structure, 
             magnitude_luminosity,       magnitude_cyan_red,       magnitude_lime_purple,       magnitude_pattern_structure, 
             phase_luminositys=None,     phase_cyan_reds=None,     phase_lime_purples=None,     phase_pattern_structures=None,
             magnitude_luminositys=None, magnitude_cyan_reds=None, magnitude_lime_purples=None, magnitude_pattern_structures=None
             ):
        latent_0_batch = latent_0_batch["samples"].double()

        batch_size = latent_0_batch.shape[0]

        phase_luminositys            = initialize_or_scale(phase_luminositys,            phase_luminosity,            batch_size)
        phase_cyan_reds              = initialize_or_scale(phase_cyan_reds,              phase_cyan_red,              batch_size)
        phase_lime_purples           = initialize_or_scale(phase_lime_purples,           phase_lime_purple,           batch_size)
        phase_pattern_structures     = initialize_or_scale(phase_pattern_structures,     phase_pattern_structure,     batch_size)

        magnitude_luminositys        = initialize_or_scale(magnitude_luminositys,        magnitude_luminosity,        batch_size)
        magnitude_cyan_reds          = initialize_or_scale(magnitude_cyan_reds,          magnitude_cyan_red,          batch_size)
        magnitude_lime_purples       = initialize_or_scale(magnitude_lime_purples,       magnitude_lime_purple,       batch_size)
        magnitude_pattern_structures = initialize_or_scale(magnitude_pattern_structures, magnitude_pattern_structure, batch_size)    

        mixed_phase_magnitude_batch = torch.zeros(latent_0_batch.shape, device=latent_0_batch.device)

        if latent_0_normal == True:
            latent_0_batch = latent_normalize_channels(latent_0_batch)
 
        for i in range(batch_size):
            mixed_phase_magnitude = self.mix_latent_phase_magnitude(latent_0_batch[i:i+1],
                                                    phase_luminositys[i].item(), phase_cyan_reds[i].item(),phase_lime_purples[i].item(),phase_pattern_structures[i].item(),
                                                    magnitude_luminositys[i].item(), magnitude_cyan_reds[i].item(),magnitude_lime_purples[i].item(),magnitude_pattern_structures[i].item()
                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )



class LatentPhaseMagnitudePower:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_0_batch": ("LATENT",),

                "phase_luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal": ("BOOLEAN", {"default": False}),
                "latent_out_normal": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_luminositys": ("SIGMAS", ),
                "phase_cyan_reds": ("SIGMAS", ),
                "phase_lime_purples": ("SIGMAS", ),
                "phase_pattern_structures": ("SIGMAS", ),

                "magnitude_luminositys": ("SIGMAS", ),
                "magnitude_cyan_reds": ("SIGMAS", ),
                "magnitude_lime_purples": ("SIGMAS", ),
                "magnitude_pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/latents"
    
    @staticmethod
    def latent_repeat(latent, batch_size):
        b, c, h, w = latent.shape
        batch_latents = torch.zeros((batch_size, c, h, w), dtype=latent.dtype, layout=latent.layout, device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return batch_latents

    @staticmethod
    def mix_latent_phase_magnitude(latent_0,  
                                    phase_luminosity, phase_cyan_red, phase_lime_purple, phase_pattern_structure,
                                    magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure
                                    ):
        dtype = latent_0.dtype
        # avoid big accuracy problems with fp32 FFT!
        latent_0 = latent_0.double()

        latent_0_fft = torch.fft.fft2(latent_0)

        latent_0_phase = torch.angle(latent_0_fft)
        latent_0_magnitude = torch.abs(latent_0_fft)

        # create new complex FFT using a weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [ w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
        mixed_phase     = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)
        mixed_magnitude = torch.zeros_like(latent_0, dtype=latent_0.dtype, layout=latent_0.layout, device=latent_0.device)

        for i in range(4):
            mixed_phase[:, i]     = latent_0_phase[:,i]     ** chan_weights_phase[i]
            mixed_magnitude[:, i] = latent_0_magnitude[:,i] ** chan_weights_magnitude[i]

        new_fft = mixed_magnitude * torch.exp(1j * mixed_phase)
        
        # inverse FFT to convert back to spatial domain
        mixed_phase_magnitude = torch.fft.ifft2(new_fft).real

        return mixed_phase_magnitude.to(dtype)
    
    def main(self,
             latent_0_batch, latent_0_normal, latent_out_normal,
             phase_luminosity,           phase_cyan_red,           phase_lime_purple,           phase_pattern_structure, 
             magnitude_luminosity,       magnitude_cyan_red,       magnitude_lime_purple,       magnitude_pattern_structure, 
             phase_luminositys=None,     phase_cyan_reds=None,     phase_lime_purples=None,     phase_pattern_structures=None,
             magnitude_luminositys=None, magnitude_cyan_reds=None, magnitude_lime_purples=None, magnitude_pattern_structures=None
             ):
        latent_0_batch = latent_0_batch["samples"].double()

        batch_size = latent_0_batch.shape[0]

        phase_luminositys            = initialize_or_scale(phase_luminositys,            phase_luminosity,            batch_size)
        phase_cyan_reds              = initialize_or_scale(phase_cyan_reds,              phase_cyan_red,              batch_size)
        phase_lime_purples           = initialize_or_scale(phase_lime_purples,           phase_lime_purple,           batch_size)
        phase_pattern_structures     = initialize_or_scale(phase_pattern_structures,     phase_pattern_structure,     batch_size)

        magnitude_luminositys        = initialize_or_scale(magnitude_luminositys,        magnitude_luminosity,        batch_size)
        magnitude_cyan_reds          = initialize_or_scale(magnitude_cyan_reds,          magnitude_cyan_red,          batch_size)
        magnitude_lime_purples       = initialize_or_scale(magnitude_lime_purples,       magnitude_lime_purple,       batch_size)
        magnitude_pattern_structures = initialize_or_scale(magnitude_pattern_structures, magnitude_pattern_structure, batch_size)    

        mixed_phase_magnitude_batch = torch.zeros(latent_0_batch.shape, device=latent_0_batch.device)

        if latent_0_normal == True:
            latent_0_batch = latent_normalize_channels(latent_0_batch)
 
        for i in range(batch_size):
            mixed_phase_magnitude = self.mix_latent_phase_magnitude(latent_0_batch[i:i+1],
                                                    phase_luminositys[i].item(), phase_cyan_reds[i].item(),phase_lime_purples[i].item(),phase_pattern_structures[i].item(),
                                                    magnitude_luminositys[i].item(), magnitude_cyan_reds[i].item(),magnitude_lime_purples[i].item(),magnitude_pattern_structures[i].item()
                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )



class StableCascade_StageC_VAEEncode_Exact:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
            "width": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
            "height": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
        }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_c",)
    FUNCTION = "generate"

    CATEGORY = "RES4LYF/vae"
    
    def generate(self, image, vae, width, height):
        out_width = (width) * vae.downscale_ratio #downscale_ratio = 32
        out_height = (height) * vae.downscale_ratio
        #movedim(-1,1) goes from 1,1024,1024,3 to 1,3,1024,1024
        s = comfy.utils.common_upscale(image.movedim(-1,1), out_width, out_height, "lanczos", "center").movedim(1,-1)

        c_latent = vae.encode(s[:,:,:,:3]) #to slice off alpha channel?
        return ({
            "samples": c_latent,
        },)
        


class StableCascade_StageC_VAEEncode_Exact_Tiled:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE", ),
            "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
            "overlap": ("INT", {"default": 16, "min": 8, "max": 128, "step": 8}),
        }}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_c",)
    FUNCTION = "generate"

    CATEGORY = "RES4LYF/vae"

    def generate(self, image, vae, tile_size, overlap):
        img_width = image.shape[-2]
        img_height = image.shape[-3]
        upscale_amount = vae.downscale_ratio  # downscale_ratio = 32

        image = image.movedim(-1, 1)  # bhwc -> bchw 

        encode_fn = lambda img: vae.encode(img.to(vae.device)).to("cpu")

        c_latent = tiled_scale_multidim(
            image, encode_fn,
            tile=(tile_size // 8, tile_size // 8),
            overlap=overlap,
            upscale_amount=upscale_amount,
            out_channels=16, 
            output_device=self.device
        )

        return ({
            "samples": c_latent,
        },)

@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None):
    dims = len(tile)
    output_shape = [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:]))
    output = torch.zeros(output_shape, device=output_device)

    for b in range(samples.shape[0]):
        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(samples.shape[2:], tile))):
            s_in = samples[b:b+1]
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s_in.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s_in.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)

            for t in range(feather):
                for d in range(2, dims + 2):
                    mask.narrow(d, t, 1).mul_((1.0 / feather) * (t + 1))
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_((1.0 / feather) * (t + 1))

            o = output[b:b+1]
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)

            if pbar is not None:
                pbar.update(1)

    return output

    


class EmptyLatentImageCustom:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "width": ("INT", {"default": 24, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "height": ("INT", {"default": 24, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),

            "channels": (['4', '16'], {"default": '4'}),
            "mode": (['sdxl', 'cascade_b', 'cascade_c', 'exact'], {"default": 'default'}),
            "compression": ("INT", {"default": 42, "min": 4, "max": 128, "step": 1}),
            "precision": (['fp16', 'fp32', 'fp64'], {"default": 'fp32'}),
            
        }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "RES4LYF/latents"

    def generate(self, width, height, batch_size, channels, mode, compression, precision):
        c = int(channels)

        ratio = 1
        match mode:
            case "sdxl":
                ratio = 8
            case "cascade_b":
                ratio = 4
            case "cascade_c":
                ratio = compression
            case "exact":
                ratio = 1

        dtype=torch.float32
        match precision:
            case "fp16":
                dtype=torch.float16
            case "fp32":
                dtype=torch.float32
            case "fp64":
                dtype=torch.float64

        latent = torch.zeros([batch_size, c, height // ratio, width // ratio], dtype=dtype, device=self.device)
        return ({"samples":latent}, )

class EmptyLatentImage64:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "RES4LYF/latents"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], dtype=torch.float64, device=self.device)
        return ({"samples":latent}, )

"""class CheckpointLoader32:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "config_name": (folder_paths.get_filename_list("configs"), ),
                              "ckpt_name": (folder_paths.get_filename_list("checkpoints"), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "advanced/loaders"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        #torch.set_default_dtype(torch.float64)
        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))"""

MAX_RESOLUTION=8192

class LatentNoiseBatch_perlin:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): 
        return {"required": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "width": ("INT", {"default": 1024, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 256}),
            "detail_level": ("FLOAT", {"default": 0, "min": -1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "details": ("SIGMAS", ),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "create_noisy_latents_perlin"
    CATEGORY = "RES4LYF/noise"

    # found at https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
    # which was ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
    def rand_perlin_2d(self, shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def rand_perlin_2d_octaves(self, shape, res, octaves=1, persistence=0.5):
        noise = torch.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
            frequency *= 2
            amplitude *= persistence
        noise = torch.remainder(torch.abs(noise)*1000000,11)/11
        # noise = (torch.sin(torch.remainder(noise*1000000,83))+1)/2
        return noise
    
    def scale_tensor(self, x):
        min_value = x.min()
        max_value = x.max()
        x = (x - min_value) / (max_value - min_value)
        return x

    def create_noisy_latents_perlin(self, seed, width, height, batch_size, detail_level, details=None):
        if details is None:
             details = torch.full((10000,), detail_level)
        else:
            details = detail_level * details
        torch.manual_seed(seed)
        noise = torch.zeros((batch_size, 4, height // 8, width // 8), dtype=torch.float32, device="cpu").cpu()
        for i in range(batch_size):
            for j in range(4):
                noise_values = self.rand_perlin_2d_octaves((height // 8, width // 8), (1,1), 1, 1)
                result = (1+details[i]/10)*torch.erfinv(2 * noise_values - 1) * (2 ** 0.5)
                result = torch.clamp(result,-5,5)
                noise[i, j, :, :] = result
        return ({"samples": noise},)


class LatentNoiseBatch_gaussian_channels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "mean": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_luminosity": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_cyan_red": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_lime_purple": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_pattern_structure": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "means": ("SIGMAS", ),
                "mean_luminositys": ("SIGMAS", ),
                "mean_cyan_reds": ("SIGMAS", ),
                "mean_lime_purples": ("SIGMAS", ),
                "mean_pattern_structures": ("SIGMAS", ),
                "stds": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/noise"

    """    @staticmethod
    def gaussian_noise_channels_like(x, mean=0.0, mean_luminosity = -0.1, mean_cyan_red = 0.0, mean_lime_purple=0.0, mean_pattern_structure=0.0, std_dev=1.0, seed=42):
        x = x.squeeze(0)

        noise = torch.randn_like(x) * std_dev + mean

        luminosity = noise[0:1] + mean_luminosity
        cyan_red = noise[1:2] + mean_cyan_red
        lime_purple = noise[2:3] + mean_lime_purple
        pattern_structure = noise[3:4] + mean_pattern_structure

        noise = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)

        return noise.to(x.device)"""
    
    @staticmethod
    def gaussian_noise_channels(x, mean_luminosity = -0.1, mean_cyan_red = 0.0, mean_lime_purple=0.0, mean_pattern_structure=0.0):
        x = x.squeeze(0)

        luminosity = x[0:1] + mean_luminosity
        cyan_red = x[1:2] + mean_cyan_red
        lime_purple = x[2:3] + mean_lime_purple
        pattern_structure = x[3:4] + mean_pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)

        return x

    def main(self, latent, steps, seed, 
              mean, mean_luminosity, mean_cyan_red, mean_lime_purple, mean_pattern_structure, std,
              means=None, mean_luminositys=None, mean_cyan_reds=None, mean_lime_purples=None, mean_pattern_structures=None, stds=None):
        if steps == 0:
            steps = len(means)

        x = latent["samples"]
        b, c, h, w = x.shape  

        noise_latents = torch.zeros([steps, 4, h, w], dtype=x.dtype, layout=x.layout, device=x.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed = seed)

        means = initialize_or_scale(means, mean, steps)
        mean_luminositys = initialize_or_scale(mean_luminositys, mean_luminosity, steps)
        mean_cyan_reds = initialize_or_scale(mean_cyan_reds, mean_cyan_red, steps)
        mean_lime_purples = initialize_or_scale(mean_lime_purples, mean_lime_purple, steps)
        mean_pattern_structures = initialize_or_scale(mean_pattern_structures, mean_pattern_structure, steps)

        stds = initialize_or_scale(stds, std, steps)

        for i in range(steps):
            noise = noise_sampler(mean=means[i].item(), std=stds[i].item())
            noise = self.gaussian_noise_channels(noise, mean_luminositys[i].item(), mean_cyan_reds[i].item(), mean_lime_purples[i].item(), mean_pattern_structures[i].item())
            noise_latents[i] = x + noise

        return ({"samples": noise_latents}, )

class LatentNoiseBatch_gaussian:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "mean": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "means": ("SIGMAS", ),
                "stds": ("SIGMAS", ),
                "steps_": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/noise"

    def main(self, latent, mean, std, steps, seed, means=None, stds=None, steps_=None):
        if steps_ is not None:
            steps = len(steps_)

        means = initialize_or_scale(means, mean, steps)
        stds = initialize_or_scale(stds, std, steps)    

        latent_samples = latent["samples"]
        b, c, h, w = latent_samples.shape  

        noise_latents = torch.zeros([steps, c, h, w], dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('gaussian')(x=latent_samples, seed = seed)

        for i in range(steps):
            noise_latents[i] = noise_sampler(mean=means[i].item(), std=stds[i].item())
        return ({"samples": noise_latents}, )

class LatentNoiseBatch_fractal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "k_flip": ("BOOLEAN", {"default": False}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "alphas": ("SIGMAS", ),
                "ks": ("SIGMAS", ),
                "steps_": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/noise"

    def main(self, latent, alpha, k_flip, steps, seed=42, alphas=None, ks=None, sigmas_=None, steps_=None):
        if steps_ is not None:
            steps = len(steps_)

        alphas = initialize_or_scale(alphas, alpha, steps)
        k_flip = -1 if k_flip else 1
        ks = initialize_or_scale(ks, k_flip, steps)

        latent_samples = latent["samples"]
        b, c, h, w = latent_samples.shape  
        noise_latents = torch.zeros([steps, c, h, w], dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('fractal')(x=latent_samples, seed = seed)

        for i in range(steps):
            noise_latents[i] = noise_sampler(alpha=alphas[i].item(), k=ks[i].item(), scale=0.1)

        return ({"samples": noise_latents}, )

class LatentNoiseList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "k_flip": ("BOOLEAN", {"default": False}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "alphas": ("SIGMAS", ),
                "ks": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "main"
    
    CATEGORY = "RES4LYF/noise"

    def main(self, seed, latent, alpha, k_flip, steps, alphas=None, ks=None):
        alphas = initialize_or_scale(alphas, alpha, steps)
        k_flip = -1 if k_flip else 1
        ks = initialize_or_scale(ks, k_flip, steps)    

        latent_samples = latent["samples"]
        latents = []
        size = latent_samples.shape

        steps = len(alphas) if steps == 0 else steps

        noise_sampler = NOISE_GENERATOR_CLASSES.get('fractal')(x=latent_samples, seed=seed)

        for i in range(steps):
            noise = noise_sampler(alpha=alphas[i].item(), k=ks[i].item(), scale=0.1)
            noisy_latent = latent_samples + noise
            new_latent = {"samples": noisy_latent}
            latents.append(new_latent)

        return (latents, )
    
class LatentBatch_channels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "mode": (["offset", "multiply", "power"],),
                "luminosity": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "cyan_red": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "lime_purple": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "pattern_structure": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            },
            "optional": {
                "luminositys": ("SIGMAS", ),
                "cyan_reds": ("SIGMAS", ),
                "lime_purples": ("SIGMAS", ),
                "pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/latents"
    
    @staticmethod
    def latent_channels_multiply(x, luminosity = -0.1, cyan_red = 0.0, lime_purple=0.0, pattern_structure=0.0):
        luminosity = x[0:1] * luminosity
        cyan_red = x[1:2] * cyan_red
        lime_purple = x[2:3] * lime_purple
        pattern_structure = x[3:4] * pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        return x

    @staticmethod
    def latent_channels_offset(x, luminosity = -0.1, cyan_red = 0.0, lime_purple=0.0, pattern_structure=0.0):
        luminosity = x[0:1] + luminosity
        cyan_red = x[1:2] + cyan_red
        lime_purple = x[2:3] + lime_purple
        pattern_structure = x[3:4] + pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        return x
    
    @staticmethod
    def latent_channels_power(x, luminosity = -0.1, cyan_red = 0.0, lime_purple=0.0, pattern_structure=0.0):
        luminosity = x[0:1] ** luminosity
        cyan_red = x[1:2] ** cyan_red
        lime_purple = x[2:3] ** lime_purple
        pattern_structure = x[3:4] ** pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        return x

    def main(self, latent, mode,
              luminosity, cyan_red, lime_purple, pattern_structure, 
              luminositys=None, cyan_reds=None, lime_purples=None, pattern_structures=None):
        
        x = latent["samples"]
        b, c, h, w = x.shape  

        noise_latents = torch.zeros([b, c, h, w], dtype=x.dtype, layout=x.layout, device=x.device)

        luminositys = initialize_or_scale(luminositys, luminosity, b)
        cyan_reds = initialize_or_scale(cyan_reds, cyan_red, b)
        lime_purples = initialize_or_scale(lime_purples, lime_purple, b)
        pattern_structures = initialize_or_scale(pattern_structures, pattern_structure, b)

        for i in range(b):
            if mode == "offset":
                noise = self.latent_channels_offset(x[i], luminositys[i].item(), cyan_reds[i].item(), lime_purples[i].item(), pattern_structures[i].item())
            elif mode == "multiply":  
                noise = self.latent_channels_multiply(x[i], luminositys[i].item(), cyan_reds[i].item(), lime_purples[i].item(), pattern_structures[i].item())
            elif mode == "power":  
                noise = self.latent_channels_power(x[i], luminositys[i].item(), cyan_reds[i].item(), lime_purples[i].item(), pattern_structures[i].item())
            noise_latents[i] = noise

        return ({"samples": noise_latents}, )
    

class LatentBatch_channels_16:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "mode": (["offset", "multiply", "power"],),
                "chan_1": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_2": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_3": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_4": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_5": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_6": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_7": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_8": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_9": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_10": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_11": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_12": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_13": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_14": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_15": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_16": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            },
            "optional": {
                "chan_1s": ("SIGMAS", ),
                "chan_2s": ("SIGMAS", ),
                "chan_3s": ("SIGMAS", ),
                "chan_4s": ("SIGMAS", ),
                "chan_5s": ("SIGMAS", ),
                "chan_6s": ("SIGMAS", ),
                "chan_7s": ("SIGMAS", ),
                "chan_8s": ("SIGMAS", ),
                "chan_9s": ("SIGMAS", ),
                "chan_10s": ("SIGMAS", ),
                "chan_11s": ("SIGMAS", ),
                "chan_12s": ("SIGMAS", ),
                "chan_13s": ("SIGMAS", ),
                "chan_14s": ("SIGMAS", ),
                "chan_15s": ("SIGMAS", ),
                "chan_16s": ("SIGMAS", ),

            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/latents"
    
    @staticmethod
    def latent_channels_multiply(x, chan_1 = 0.0, chan_2 = 0.0, chan_3 = 0.0, chan_4 = 0.0, chan_5 = 0.0, chan_6 = 0.0, chan_7 = 0.0, chan_8 = 0.0, chan_9 = 0.0, chan_10 = 0.0, chan_11 = 0.0, chan_12 = 0.0, chan_13 = 0.0, chan_14 = 0.0, chan_15 = 0.0, chan_16 = 0.0):
        chan_1 = x[0:1] * chan_1
        chan_2 = x[1:2] * chan_2
        chan_3 = x[2:3] * chan_3
        chan_4 = x[3:4] * chan_4
        chan_5 = x[4:5] * chan_5
        chan_6 = x[5:6] * chan_6
        chan_7 = x[6:7] * chan_7
        chan_8 = x[7:8] * chan_8
        chan_9 = x[8:9] * chan_9
        chan_10 = x[9:10] * chan_10
        chan_11 = x[10:11] * chan_11
        chan_12 = x[11:12] * chan_12
        chan_13 = x[12:13] * chan_13
        chan_14 = x[13:14] * chan_14
        chan_15 = x[14:15] * chan_15
        chan_16 = x[15:16] * chan_16

        x = torch.unsqueeze(torch.cat([chan_1, chan_2, chan_3, chan_4, chan_5, chan_6, chan_7, chan_8, chan_9, chan_10, chan_11, chan_12, chan_13, chan_14, chan_15, chan_16]), 0)
        return x

    @staticmethod
    def latent_channels_offset(x, chan_1 = 0.0, chan_2 = 0.0, chan_3 = 0.0, chan_4 = 0.0, chan_5 = 0.0, chan_6 = 0.0, chan_7 = 0.0, chan_8 = 0.0, chan_9 = 0.0, chan_10 = 0.0, chan_11 = 0.0, chan_12 = 0.0, chan_13 = 0.0, chan_14 = 0.0, chan_15 = 0.0, chan_16 = 0.0):
        chan_1 = x[0:1] + chan_1
        chan_2 = x[1:2] + chan_2
        chan_3 = x[2:3] + chan_3
        chan_4 = x[3:4] + chan_4
        chan_5 = x[4:5] + chan_5
        chan_6 = x[5:6] + chan_6
        chan_7 = x[6:7] + chan_7
        chan_8 = x[7:8] + chan_8
        chan_9 = x[8:9] + chan_9
        chan_10 = x[9:10] + chan_10
        chan_11 = x[10:11] + chan_11
        chan_12 = x[11:12] + chan_12
        chan_13 = x[12:13] + chan_13
        chan_14 = x[13:14] + chan_14
        chan_15 = x[14:15] + chan_15
        chan_16 = x[15:16] + chan_16

        x = torch.unsqueeze(torch.cat([chan_1, chan_2, chan_3, chan_4, chan_5, chan_6, chan_7, chan_8, chan_9, chan_10, chan_11, chan_12, chan_13, chan_14, chan_15, chan_16]), 0)
        return x

    @staticmethod
    def latent_channels_power(x, chan_1 = 0.0, chan_2 = 0.0, chan_3 = 0.0, chan_4 = 0.0, chan_5 = 0.0, chan_6 = 0.0, chan_7 = 0.0, chan_8 = 0.0, chan_9 = 0.0, chan_10 = 0.0, chan_11 = 0.0, chan_12 = 0.0, chan_13 = 0.0, chan_14 = 0.0, chan_15 = 0.0, chan_16 = 0.0):
        chan_1 = x[0:1] ** chan_1
        chan_2 = x[1:2] ** chan_2
        chan_3 = x[2:3] ** chan_3
        chan_4 = x[3:4] ** chan_4
        chan_5 = x[4:5] ** chan_5
        chan_6 = x[5:6] ** chan_6
        chan_7 = x[6:7] ** chan_7
        chan_8 = x[7:8] ** chan_8
        chan_9 = x[8:9] ** chan_9
        chan_10 = x[9:10] ** chan_10
        chan_11 = x[10:11] ** chan_11
        chan_12 = x[11:12] ** chan_12
        chan_13 = x[12:13] ** chan_13
        chan_14 = x[13:14] ** chan_14
        chan_15 = x[14:15] ** chan_15
        chan_16 = x[15:16] ** chan_16

        x = torch.unsqueeze(torch.cat([chan_1, chan_2, chan_3, chan_4, chan_5, chan_6, chan_7, chan_8, chan_9, chan_10, chan_11, chan_12, chan_13, chan_14, chan_15, chan_16]), 0)
        return x

    def main(self, latent, mode,
              chan_1, chan_2, chan_3, chan_4, chan_5, chan_6, chan_7, chan_8, chan_9, chan_10, chan_11, chan_12, chan_13, chan_14, chan_15, chan_16,
              chan_1s=None, chan_2s=None, chan_3s=None, chan_4s=None, chan_5s=None, chan_6s=None, chan_7s=None, chan_8s=None, chan_9s=None, chan_10s=None, chan_11s=None, chan_12s=None, chan_13s=None, chan_14s=None, chan_15s=None, chan_16s=None):
        
        x = latent["samples"]
        b, c, h, w = x.shape  

        noise_latents = torch.zeros([b, c, h, w], dtype=x.dtype, layout=x.layout, device=x.device)
        chan_1s = initialize_or_scale(chan_1s, chan_1, b)
        chan_2s = initialize_or_scale(chan_2s, chan_2, b)
        chan_3s = initialize_or_scale(chan_3s, chan_3, b)
        chan_4s = initialize_or_scale(chan_4s, chan_4, b)
        chan_5s = initialize_or_scale(chan_5s, chan_5, b)
        chan_6s = initialize_or_scale(chan_6s, chan_6, b)
        chan_7s = initialize_or_scale(chan_7s, chan_7, b)
        chan_8s = initialize_or_scale(chan_8s, chan_8, b)
        chan_9s = initialize_or_scale(chan_9s, chan_9, b)
        chan_10s = initialize_or_scale(chan_10s, chan_10, b)
        chan_11s = initialize_or_scale(chan_11s, chan_11, b)
        chan_12s = initialize_or_scale(chan_12s, chan_12, b)
        chan_13s = initialize_or_scale(chan_13s, chan_13, b)
        chan_14s = initialize_or_scale(chan_14s, chan_14, b)
        chan_15s = initialize_or_scale(chan_15s, chan_15, b)
        chan_16s = initialize_or_scale(chan_16s, chan_16, b)

        for i in range(b):
            if mode == "offset":
                noise = self.latent_channels_offset(x[i], chan_1s[i].item(), chan_2s[i].item(), chan_3s[i].item(), chan_4s[i].item(), chan_5s[i].item(), chan_6s[i].item(), chan_7s[i].item(), chan_8s[i].item(), chan_9s[i].item(), chan_10s[i].item(), chan_11s[i].item(), chan_12s[i].item(), chan_13s[i].item(), chan_14s[i].item(), chan_15s[i].item(), chan_16s[i].item())
            elif mode == "multiply":  
                noise = self.latent_channels_multiply(x[i], chan_1s[i].item(), chan_2s[i].item(), chan_3s[i].item(), chan_4s[i].item(), chan_5s[i].item(), chan_6s[i].item(), chan_7s[i].item(), chan_8s[i].item(), chan_9s[i].item(), chan_10s[i].item(), chan_11s[i].item(), chan_12s[i].item(), chan_13s[i].item(), chan_14s[i].item(), chan_15s[i].item(), chan_16s[i].item())
            elif mode == "power":  
                noise = self.latent_channels_power(x[i], chan_1s[i].item(), chan_2s[i].item(), chan_3s[i].item(), chan_4s[i].item(), chan_5s[i].item(), chan_6s[i].item(), chan_7s[i].item(), chan_8s[i].item(), chan_9s[i].item(), chan_10s[i].item(), chan_11s[i].item(), chan_12s[i].item(), chan_13s[i].item(), chan_14s[i].item(), chan_15s[i].item(), chan_16s[i].item())
            noise_latents[i] = noise

        return ({"samples": noise_latents}, )
    
class latent_normalize_channels:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent": ("LATENT", ),     
                    "mode": (["full", "channels"],), 
                    "operation": (["normalize", "center", "standardize"],), 
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "RES4LYF/latents"

    FUNCTION = "main"

    def main(self, latent, mode, operation):
        x = latent["samples"]
        b, c, h, w = x.shape

        if mode == "full":
            if operation == "normalize":
                x = (x - x.mean()) / x.std()
            elif operation == "center":
                x = x - x.mean()
            elif operation == "standardize":
                x = x / x.std()

        elif mode == "channels":
            if operation == "normalize":
                for i in range(b):
                    for j in range(c):
                        x[i, j] = (x[i, j] - x[i, j].mean()) / x[i, j].std()
            elif operation == "center":
                for i in range(b):
                    for j in range(c):
                        x[i, j] = x[i, j] - x[i, j].mean()
            elif operation == "standardize":
                for i in range(b):
                    for j in range(c):
                        x[i, j] = x[i, j] / x[i, j].std()

        return ({"samples": x},)




def hard_light_blend(base_latent, blend_latent):
    if base_latent.sum() == 0 and base_latent.std() == 0:
        return base_latent
    
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0
    
    positive_latent = base_latent * positive_mask.float()
    negative_latent = base_latent * negative_mask.float()

    positive_result = torch.where(blend_latent < 0.5,
                                  2 * positive_latent * blend_latent,
                                  1 - 2 * (1 - positive_latent) * (1 - blend_latent))

    negative_result = torch.where(blend_latent < 0.5,
                                  2 * negative_latent.abs() * blend_latent,
                                  1 - 2 * (1 - negative_latent.abs()) * (1 - blend_latent))
    negative_result = -negative_result

    combined_result = positive_result * positive_mask.float() + negative_result * negative_mask.float()

    #combined_result *= base_latent.max()
    
    ks = combined_result
    ks2 = torch.zeros_like(base_latent)
    for n in range(base_latent.shape[1]):
        ks2[0][n] = (ks[0][n]) / ks[0][n].std()
        ks2[0][n] = (ks2[0][n] * base_latent[0][n].std())
    combined_result = ks2
    
    return combined_result



