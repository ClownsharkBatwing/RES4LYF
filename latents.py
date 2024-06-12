import comfy.samplers
import comfy.sample
import comfy.sampler_helpers

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

class set_precision:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "latent_image": ("LATENT", ),      
                    "precision": (["16", "32", "64"], ),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "sampling/custom_sampling/"

    FUNCTION = "main"

    def main(self, precision="32", latent_image=None):
        match precision:
            case "16":
                torch.set_default_dtype(torch.float16)
                x = latent_image["samples"].to(torch.float16)
            case "32":
                torch.set_default_dtype(torch.float32)
                x = latent_image["samples"].to(torch.float32)
            case "64":
                torch.set_default_dtype(torch.float64)
                x = latent_image["samples"].to(torch.float64)
        return ({"samples": x}, )
    
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
    CATEGORY = "sampling/custom_sampling/"

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
    CATEGORY = "sampling/custom_sampling/"

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
    CATEGORY = "sampling/custom_sampling/samplers"

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
    CATEGORY = "sampling/custom_sampling/samplers"

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
    CATEGORY = "sampling/custom_sampling/samplers"

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
    CATEGORY = "sampling/custom_sampling/samplers"

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

    CATEGORY = "latent"

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
    CATEGORY = "latent/noise"

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

    CATEGORY = "sampling/custom_sampling/samplers"

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
        # x_noised = torch.zeros([steps, 4, h, w], device=x.device)

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
                "mean": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "means": ("SIGMAS", ),
                "stds": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling/samplers"

    def main(self, latent, mean, std, steps, seed, means=None, stds=None):
        if steps == 0:
            steps = len(means)

        means = initialize_or_scale(means, mean, steps)
        stds = initialize_or_scale(stds, std, steps)    

        latent_samples = latent["samples"]
        b, c, h, w = latent_samples.shape  
        #noise_latents = torch.zeros([steps, 4, h, w], device=latent_samples.device)

        #for i in range(steps):
        #    noise = self.adjustable_gaussian_noise_like(latent_samples, means[i].item(), stds[i].item(), seed+i)
        #    noise_latents[i] = latent_samples + noise

        noise_latents = torch.zeros([steps, 4, h, w], dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('gaussian')(x=latent_samples, seed = seed)

        for i in range(steps):
            noise = noise_sampler(mean=means[i].item(), std=stds[i].item())
            noise_latents[i] = latent_samples + noise

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
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling/samplers"

    def main(self, latent, alpha, k_flip, steps, seed=42, alphas=None, ks=None):
        if steps == 0:
            steps = len(alphas)

        alphas = initialize_or_scale(alphas, alpha, steps)
        k_flip = -1 if k_flip else 1
        ks = initialize_or_scale(ks, k_flip, steps)

        latent_samples = latent["samples"]
        b, c, h, w = latent_samples.shape  
        noise_latents = torch.zeros([steps, 4, h, w], dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('fractal')(x=latent_samples, seed = seed)

        for i in range(steps):
            noise = noise_sampler(alpha=alphas[i].item(), k=ks[i].item(), scale=0.1)
            noise_latents[i] = latent_samples + noise

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

    CATEGORY = "sampling/custom_sampling/samplers"

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

class LatentBatch_channels_offset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "offset": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_luminosity": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_cyan_red": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_lime_purple": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_pattern_structure": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "offsets": ("SIGMAS", ),
                "offset_luminositys": ("SIGMAS", ),
                "offset_cyan_reds": ("SIGMAS", ),
                "offset_lime_purples": ("SIGMAS", ),
                "offset_pattern_structures": ("SIGMAS", ),
                "stds": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling/samplers"
    
    @staticmethod
    def latent_channels_offset(x, offset_luminosity = -0.1, offset_cyan_red = 0.0, offset_lime_purple=0.0, offset_pattern_structure=0.0):
        #x = torch.squeeze(x, )

        luminosity = x[0:1] + offset_luminosity
        cyan_red = x[1:2] + offset_cyan_red
        lime_purple = x[2:3] + offset_lime_purple
        pattern_structure = x[3:4] + offset_pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        #x = torch.cat([luminosity, cyan_red, lime_purple, pattern_structure])

        return x

    def main(self, latent, steps, seed, 
              offset, offset_luminosity, offset_cyan_red, offset_lime_purple, offset_pattern_structure, std,
              offsets=None, offset_luminositys=None, offset_cyan_reds=None, offset_lime_purples=None, offset_pattern_structures=None, stds=None):
        if steps == 0:
            steps = len(offsets)

        #pdb.set_trace()
        x = latent["samples"]
        b, c, h, w = x.shape  

        # x_noised = torch.zeros([steps, 4, h, w], device=x.device)

        noise_latents = torch.zeros([steps, 4, h, w], dtype=x.dtype, layout=x.layout, device=x.device)

        offsets = initialize_or_scale(offsets, offset, steps)
        offset_luminositys = initialize_or_scale(offset_luminositys, offset_luminosity, steps)
        offset_cyan_reds = initialize_or_scale(offset_cyan_reds, offset_cyan_red, steps)
        offset_lime_purples = initialize_or_scale(offset_lime_purples, offset_lime_purple, steps)
        offset_pattern_structures = initialize_or_scale(offset_pattern_structures, offset_pattern_structure, steps)

        stds = initialize_or_scale(stds, std, steps)

        for i in range(steps):
            noise = self.latent_channels_offset(x[i], offset_luminositys[i].item(), offset_cyan_reds[i].item(), offset_lime_purples[i].item(), offset_pattern_structures[i].item())
            #noise = self.latent_channels_offset(x[i:i+1], offset_luminositys[i].item(), offset_cyan_reds[i].item(), offset_lime_purples[i].item(), offset_pattern_structures[i].item())
            noise_latents[i] = noise

        return ({"samples": noise_latents}, )
    

class LatentBatch_channels_multiply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "offset": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_luminosity": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_cyan_red": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_lime_purple": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "offset_pattern_structure": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "offsets": ("SIGMAS", ),
                "offset_luminositys": ("SIGMAS", ),
                "offset_cyan_reds": ("SIGMAS", ),
                "offset_lime_purples": ("SIGMAS", ),
                "offset_pattern_structures": ("SIGMAS", ),
                "stds": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "sampling/custom_sampling/samplers"
    
    @staticmethod
    def latent_channels_offset(x, offset_luminosity = -0.1, offset_cyan_red = 0.0, offset_lime_purple=0.0, offset_pattern_structure=0.0):
        #x = torch.squeeze(x, )

        luminosity = x[0:1] * offset_luminosity
        cyan_red = x[1:2] * offset_cyan_red
        lime_purple = x[2:3] * offset_lime_purple
        pattern_structure = x[3:4] * offset_pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        #x = torch.cat([luminosity, cyan_red, lime_purple, pattern_structure])

        return x

    def main(self, latent, steps, seed, 
              offset, offset_luminosity, offset_cyan_red, offset_lime_purple, offset_pattern_structure, std,
              offsets=None, offset_luminositys=None, offset_cyan_reds=None, offset_lime_purples=None, offset_pattern_structures=None, stds=None):
        if steps == 0:
            steps = len(offsets)

        #pdb.set_trace()
        x = latent["samples"]
        b, c, h, w = x.shape  

        # x_noised = torch.zeros([steps, 4, h, w], device=x.device)

        noise_latents = torch.zeros([steps, 4, h, w], dtype=x.dtype, layout=x.layout, device=x.device)

        offsets = initialize_or_scale(offsets, offset, steps)
        offset_luminositys = initialize_or_scale(offset_luminositys, offset_luminosity, steps)
        offset_cyan_reds = initialize_or_scale(offset_cyan_reds, offset_cyan_red, steps)
        offset_lime_purples = initialize_or_scale(offset_lime_purples, offset_lime_purple, steps)
        offset_pattern_structures = initialize_or_scale(offset_pattern_structures, offset_pattern_structure, steps)

        stds = initialize_or_scale(stds, std, steps)

        for i in range(steps):
            noise = self.latent_channels_offset(x[i], offset_luminositys[i].item(), offset_cyan_reds[i].item(), offset_lime_purples[i].item(), offset_pattern_structures[i].item())
            #noise = self.latent_channels_offset(x[i:i+1], offset_luminositys[i].item(), offset_cyan_reds[i].item(), offset_lime_purples[i].item(), offset_pattern_structures[i].item())
            noise_latents[i] = noise

        return ({"samples": noise_latents}, )
    

