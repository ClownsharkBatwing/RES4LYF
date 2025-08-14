import torch.nn.functional as F

import copy

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
    
import itertools

import torch
import math

from nodes import MAX_RESOLUTION
#MAX_RESOLUTION=8192

from .helper             import ExtraOptions, initialize_or_scale, extra_options_flag, get_extra_options_list
from .latents            import latent_meancenter_channels, latent_stdize_channels, get_edge_mask, apply_to_state_info_tensors
from .beta.noise_classes import NOISE_GENERATOR_NAMES, NOISE_GENERATOR_CLASSES, prepare_noise

def fp_or(tensor1, tensor2):
    return torch.maximum(tensor1, tensor2)

def fp_and(tensor1, tensor2):
    return torch.minimum(tensor1, tensor2)


class AdvancedNoise:
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/noise"
    
    def main(self,
            add_noise,
            noise_is_latent,
            noise_type,
            noise_seed,
            alpha,
            k,
            latent_image,
            noise_strength,
            normalize,
            latent_noise = None,
            mask         = None
            ):
        
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
            if len(samples.shape) == 5:
                b, c, t, h, w = samples.shape
                mask_resized = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                                size=(h, w), 
                                mode="bilinear")
                if mask_resized.shape[0] < b:
                    mask_resized = mask_resized.repeat((b - 1) // mask_resized.shape[0] + 1, 1, 1, 1)[:b]
                elif mask_resized.shape[0] > b:
                    mask_resized = mask_resized[:b]
                mask_expanded = mask_resized.expand((-1, c, -1, -1))
                mask_temporal = mask_expanded.unsqueeze(2).expand(-1, -1, t, -1, -1).to(samples.device)
                noise = mask_temporal * noise + (1 - mask_temporal) * torch.zeros_like(noise)
            else:
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




class LatentNoiseList:
    @classmethod
    def INPUT_TYPES(cls):
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

    RETURN_TYPES   = ("LATENT",)
    RETURN_NAMES   = ("latent_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION       = "main"
    CATEGORY       = "RES4LYF/noise"

    def main(self,
            seed,
            latent,
            alpha,
            k_flip,
            steps,
            alphas = None,
            ks     = None
            ):
        
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



class MaskToggle:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "enable": ("BOOLEAN", {"default": True}),    
                    "mask":   ("MASK", ),
                     },
                }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"


    def main(self, enable=True, mask=None):
        if enable == False:
            mask = None
        return (mask, )



class latent_to_raw_x:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent": ("LATENT", ),      
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_raw_x",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent,):
        if 'state_info' not in latent:
            latent['state_info'] = {}
        
        latent['state_info']['raw_x'] = latent['samples'].to(torch.float64)
        return (latent,)


# Adapted from https://github.com/comfyanonymous/ComfyUI/blob/5ee381c058d606209dcafb568af20196e7884fc8/comfy_extras/nodes_wan.py
class TrimVideoLatent_state_info:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",),
                             "trim_amount": ("INT", {"default": 0, "min": 0, "max": 99999}),
                            }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"
    CATEGORY = "RES4LYF/latents"
    EXPERIMENTAL = True

    @staticmethod
    def _trim_tensor(tensor, trim_amount):
        """Trim frames from beginning of tensor along temporal dimension (-3)"""
        if tensor.shape[-3] > trim_amount:
            return tensor.narrow(-3, trim_amount, tensor.shape[-3] - trim_amount)
        return tensor
    
    def op(self, samples, trim_amount):
        ref_shape = samples["samples"].shape
        samples_out = apply_to_state_info_tensors(samples, ref_shape, self._trim_tensor, trim_amount)
        return (samples_out,)

# Adapted from https://github.com/comfyanonymous/ComfyUI/blob/05df2df489f6b237f63c5f7d42a943ae2be417e9/nodes.py
class LatentUpscaleBy_state_info:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent"

    def _upscale_tensor(tensor, upscale_method, scale_by):
        width = round(tensor.shape[-1] * scale_by)
        height = round(tensor.shape[-2] * scale_by)
        tensor = comfy.utils.common_upscale(tensor, width, height, upscale_method, "disabled")
        return tensor
    
    def op(self, samples, upscale_method, scale_by):
        ref_shape = samples["samples"].shape
        samples_out = apply_to_state_info_tensors(samples, ref_shape, self._upscale_tensor, upscale_method, scale_by)
        return (samples_out,)

class latent_clear_state_info:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent": ("LATENT", ),      
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent,):
        latent_out = {}
        if 'samples' in latent:
            latent_out['samples'] = latent['samples']
        return (latent_out,)


class latent_replace_state_info:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent": ("LATENT", ),
                    "clear_raw_x": ("BOOLEAN", {"default": False}),
                    "replace_end_step": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent, clear_raw_x, replace_end_step):
        latent_out = copy.deepcopy(latent)
        if 'state_info' not in latent_out:
            latent_out['state_info'] = {}
        if clear_raw_x:
            latent_out['state_info']['raw_x'] = None
        if replace_end_step != 0:
            latent_out['state_info']['end_step'] = replace_end_step
        return (latent_out,)


class latent_display_state_info:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent": ("LATENT", ),      
                     },
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION     = "execute"
    CATEGORY     = "RES4LYF/latents"
    OUTPUT_NODE  = True

    def execute(self, latent):
        text = ""
        if 'state_info' in latent:
            for key, value in latent['state_info'].items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        value_text = "empty tensor"
                    elif value.numel() == 1:
                        if value.dtype == torch.bool:
                            value_text = f"bool({value.item()})"
                        else:
                            value_text = f"str({value.item():.3f}), dtype: {value.dtype}"
                    else:
                        shape_str = str(list(value.shape)).replace(" ", "")
                        dtype = value.dtype

                        if torch.is_floating_point(value) is False:
                            if value.dtype == torch.bool:
                                value_text = f"shape: {shape_str}, dtype: {dtype}, true: {value.sum().item()}, false: {(~value).sum().item()}"
                            else:
                                max_val = value.float().max().item()
                                min_val = value.float().min().item()
                                value_text = f"shape: {shape_str}, dtype: {dtype}, max: {max_val}, min: {min_val}"
                        else:
                            mean = value.float().mean().item()
                            std = value.float().std().item()
                            value_text = f"shape: {shape_str}, dtype: {dtype}, mean: {mean:.3f}, std: {std:.3f}"
                else:
                    value_text = str(value)

                text += f"{key}: {value_text}\n"
        else:
            text = "No state info in latent"

        return {"ui": {"text": text}, "result": (text,)}





class latent_transfer_state_info:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent_to":   ("LATENT", ),      
                    "latent_from": ("LATENT", ),      
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent_to, latent_from):
        #if 'state_info' not in latent:
        #    latent['state_info'] = {}
        
        latent_to['state_info'] = copy.deepcopy(latent_from['state_info'])
        return (latent_to,)




class latent_mean_channels_from_to:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent_to":   ("LATENT", ),      
                    "latent_from": ("LATENT", ),      
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent_to, latent_from):
        latent_to['samples'] = latent_to['samples'] - latent_to['samples'].mean(dim=(-2,-1), keepdim=True) + latent_from['samples'].mean(dim=(-2,-1), keepdim=True)
        return (latent_to,)



class latent_get_channel_means:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent":   ("LATENT", ),      
                    },
                }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("channel_means",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent):
        channel_means = latent['samples'].mean(dim=(-2,-1)).squeeze(0)
        return (channel_means,)






class latent_to_cuda:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent": ("LATENT", ),      
                    "to_cuda": ("BOOLEAN", {"default": True}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent":     ("LATENT", ),      
                    "batch_size": ("INT", {"default": 0, "min": -10000, "max": 10000}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, latent, batch_size):
        latent = latent["samples"]
        b, c, h, w = latent.shape
        batch_latents = torch.zeros([batch_size, 4, h, w], device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return ({"samples": batch_latents}, )



class MaskFloatToBoolean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "mask": ("MASK",),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("binary_mask",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"

    def main(self, mask=None,):
        return (mask.bool().to(mask.dtype),)
    




class MaskEdge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "dilation": ("INT", {"default": 20, "min": -10000, "max": 10000}),
                "mode": [["percent", "absolute"], {"default": "percent"}],
                "internal": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                "external": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 10000.0, "step": 0.01}),
                #"blur": ("BOOLEAN", {"default": False}),
                "mask": ("MASK",),
            },
            "optional": {
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("edge_mask",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"

    def main(self, dilation=20, mode="percent", internal=1.0, external=1.0, blur=False, mask=None,):
        
        mask_dtype = mask.dtype
        mask = mask.float()
        
        if mode == "percent":
            dilation = (dilation/100) * int(mask.sum() ** 0.5)
        
        #if not blur:
        if int(internal * dilation) > 0:
            edge_mask_internal = get_edge_mask(mask, int(internal * dilation))
            edge_mask_internal = fp_and(edge_mask_internal,   mask)
        else:
            edge_mask_internal = mask
        
        if int(external * dilation) > 0:
            edge_mask_external = get_edge_mask(mask, int(external * dilation))
            edge_mask_external = fp_and(edge_mask_external, 1-mask)
        else:
            edge_mask_external = 1-mask
        
        edge_mask = fp_or(edge_mask_internal, edge_mask_external)

        return (edge_mask.to(mask_dtype),)
    
    
    


class Frame_Select_Latent_Raw:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("IMAGE",),
                "select": ("INT",  {"default": 0, "min": 0, "max": 10000}),
                
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, frames=None, select=0):
        frame = frames['state_info']['raw_x'][:,:,select,:,:].clone().unsqueeze(dim=2)
        return (frame,)
    

class Frames_Slice_Latent_Raw:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("LATENT",),
                "start":  ("INT",  {"default": 0, "min": 0, "max": 10000}),
                "stop":   ("INT",  {"default": 1, "min": 1, "max": 10000}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, frames=None, start=0, stop=1):
        frames_slice = frames['state_info']['raw_x'][:,:,start:stop,:,:].clone()
        return (frames_slice,)


class Frames_Concat_Latent_Raw:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames_0": ("LATENT",),
                "frames_1": ("LATENT",),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, frames_0, frames_1):
        frames_concat = torch.cat((frames_0, frames_1), dim=2).clone()
        return (frames_concat,)
    


class Frame_Select_Latent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("IMAGE",),
                "select": ("INT",  {"default": 0, "min": 0, "max": 10000}),
                
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, frames=None, select=0):
        frame = frames['samples'][:,:,select,:,:].clone().unsqueeze(dim=2)
        return ({"samples": frame},)
    

class Frames_Slice_Latent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("LATENT",),
                "start":  ("INT",  {"default": 0, "min": 0, "max": 10000}),
                "stop":   ("INT",  {"default": 1, "min": 1, "max": 10000}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, frames=None, start=0, stop=1):
        frames_slice = frames['samples'][:,:,start:stop,:,:].clone()
        return ({"samples": frames_slice},)


class Frames_Concat_Latent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames_0": ("LATENT",),
                "frames_1": ("LATENT",),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self, frames_0, frames_1):
        frames_concat = torch.cat((frames_0['samples'], frames_1['samples']), dim=2).clone()
        return ({"samples": frames_concat},)
    




class Frames_Concat_Masks:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames_0": ("MASK",),
                "frames_1": ("MASK",),

            },
            "optional": {
                "frames_2": ("MASK",),
                "frames_3": ("MASK",),
                "frames_4": ("MASK",),
                "frames_5": ("MASK",),
                "frames_6": ("MASK",),
                "frames_7": ("MASK",),
                "frames_8": ("MASK",),
                "frames_9": ("MASK",),
            },
        }
        
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"

    def main(self, frames_0, frames_1, frames_2=None, frames_3=None, frames_4=None, frames_5=None, frames_6=None, frames_7=None, frames_8=None, frames_9=None):
        frames_concat = torch.cat((frames_0,      frames_1), dim=-3).clone()
        
        frames_concat = torch.cat((frames_concat, frames_2), dim=-3).clone() if frames_2 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_3), dim=-3).clone() if frames_3 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_4), dim=-3).clone() if frames_4 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_5), dim=-3).clone() if frames_5 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_6), dim=-3).clone() if frames_6 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_7), dim=-3).clone() if frames_7 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_8), dim=-3).clone() if frames_8 is not None else frames_concat
        frames_concat = torch.cat((frames_concat, frames_9), dim=-3).clone() if frames_9 is not None else frames_concat
        
        if frames_concat.ndim == 3:
            frames_concat.unsqueeze_(0)

        return (frames_concat,)
    




class Frames_Masks_Uninterpolate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "raw_temporal_mask": ("MASK",),
                "frame_chunk_size" : ("INT", {"default": 4, "min": 1, "max": 10000, "step": 1}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"

    def main(self, raw_temporal_mask, frame_chunk_size):
        #assert raw_temporal_mask.ndim == 3, "Not a raw temporal mask!"
        
        raw_frames = raw_temporal_mask.shape[-3]
        raw_frames_offset = raw_frames - 1
        frames = raw_frames_offset // frame_chunk_size + 1
        indices = torch.linspace(0, raw_frames_offset, steps=frames).long()
        
        temporal_mask = raw_temporal_mask[...,indices,:,:].unsqueeze(0)
        return (temporal_mask,)



class Frames_Masks_ZeroOut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "temporal_mask": ("MASK",),
                "zero_out_frame" : ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"

    def main(self, temporal_mask, zero_out_frame):
        temporal_mask[...,zero_out_frame:zero_out_frame+1,:,:] = 1.0
        return (temporal_mask,)


class Frames_Latent_ReverseOrder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("LATENT",),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("frames_reversed",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"

    def main(self, frames,):
        samples = frames['samples']
        flipped_frames = torch.zeros_like(samples)
        
        t_len = samples.shape[-3]
        
        for i in range(t_len):
            flipped_frames[:,:,t_len-i-1,:,:] = samples[:,:,i,:,:]
        return (  {"samples": flipped_frames },)
        
        #return (  {"samples": torch.flip(frames['samples'], dims=[-3]) },)



class LatentPhaseMagnitude:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_0_batch":               ("LATENT",),
                "latent_1_batch":               ("LATENT",),

                "phase_mix_power":              ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_mix_power":          ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "phase_luminosity":             ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red":               ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple":            ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure":      ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity":         ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red":           ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple":        ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure":  ("FLOAT",   {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal":              ("BOOLEAN", {"default": True}),
                "latent_1_normal":              ("BOOLEAN", {"default": True}),
                "latent_out_normal":            ("BOOLEAN", {"default": True}),
                "latent_0_stdize":              ("BOOLEAN", {"default": True}),
                "latent_1_stdize":              ("BOOLEAN", {"default": True}),
                "latent_out_stdize":            ("BOOLEAN", {"default": True}),
                "latent_0_meancenter":          ("BOOLEAN", {"default": True}),
                "latent_1_meancenter":          ("BOOLEAN", {"default": True}),
                "latent_out_meancenter":        ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "phase_mix_powers":             ("SIGMAS", ),
                "magnitude_mix_powers":         ("SIGMAS", ),

                "phase_luminositys":            ("SIGMAS", ),
                "phase_cyan_reds":              ("SIGMAS", ),
                "phase_lime_purples":           ("SIGMAS", ),
                "phase_pattern_structures":     ("SIGMAS", ),

                "magnitude_luminositys":        ("SIGMAS", ),
                "magnitude_cyan_reds":          ("SIGMAS", ),
                "magnitude_lime_purples":       ("SIGMAS", ),
                "magnitude_pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"
    
    @staticmethod
    def latent_repeat(latent, batch_size):
        b, c, h, w = latent.shape
        batch_latents = torch.zeros((batch_size, c, h, w), dtype=latent.dtype, layout=latent.layout, device=latent.device)
        for i in range(batch_size):
            batch_latents[i] = latent
        return batch_latents

    @staticmethod
    def mix_latent_phase_magnitude(latent_0,
                                    latent_1,
                                    power_phase,
                                    power_magnitude,
                                    phase_luminosity,
                                    phase_cyan_red,
                                    phase_lime_purple,
                                    phase_pattern_structure,
                                    magnitude_luminosity,
                                    magnitude_cyan_red,
                                    magnitude_lime_purple,
                                    magnitude_pattern_structure,
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
            mixed_phase[:, i]     = ( (latent_0_phase[:,i]     * (1-chan_weights_phase[i]))     ** power_phase     + (latent_1_phase[:,i]     * chan_weights_phase[i])     ** power_phase)     ** (1/power_phase)
            mixed_magnitude[:, i] = ( (latent_0_magnitude[:,i] * (1-chan_weights_magnitude[i])) ** power_magnitude + (latent_1_magnitude[:,i] * chan_weights_magnitude[i]) ** power_magnitude) ** (1/power_magnitude)

        new_fft = mixed_magnitude * torch.exp(1j * mixed_phase)

        #new_fft[:, :, dc_index[0], dc_index[1]] = mixed_dc

        # inverse FFT to convert back to spatial domain
        mixed_phase_magnitude = torch.fft.ifft2(new_fft).real

        return mixed_phase_magnitude.to(dtype)
    
    def main(self,
            #batch_size,
            latent_1_repeat,
            latent_0_batch,
            latent_1_batch,
            latent_0_normal,
            latent_1_normal,
            latent_out_normal,
            latent_0_stdize,
            latent_1_stdize,
            latent_out_stdize,

            latent_0_meancenter,
            latent_1_meancenter,
            latent_out_meancenter,

            phase_mix_power,
            magnitude_mix_power,

            phase_luminosity,
            phase_cyan_red,
            phase_lime_purple,
            phase_pattern_structure,

            magnitude_luminosity,
            magnitude_cyan_red,
            magnitude_lime_purple,
            magnitude_pattern_structure,

            phase_mix_powers             = None, 
            magnitude_mix_powers         = None,
            phase_luminositys            = None,
            phase_cyan_reds              = None,
            phase_lime_purples           = None,
            phase_pattern_structures     = None,
            magnitude_luminositys        = None,
            magnitude_cyan_reds          = None,
            magnitude_lime_purples       = None,
            magnitude_pattern_structures = None
            ):
        
        latent_0_batch = latent_0_batch["samples"].double()
        latent_1_batch = latent_1_batch["samples"].double().to(latent_0_batch.device)

        #if batch_size == 0:
        batch_size = latent_0_batch.shape[0]
        if latent_1_batch.shape[0] == 1:
            latent_1_batch = self.latent_repeat(latent_1_batch, batch_size)

        magnitude_mix_powers         = initialize_or_scale(magnitude_mix_powers,         magnitude_mix_power,         batch_size)
        phase_mix_powers             = initialize_or_scale(phase_mix_powers,             phase_mix_power,             batch_size)

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
            mixed_phase_magnitude = self.mix_latent_phase_magnitude(latent_0_batch[i:i+1],
                                                                    latent_1_batch[i:i+1],
                                                                    
                                                                    phase_mix_powers[i]            .item(),
                                                                    magnitude_mix_powers[i]        .item(),

                                                                    phase_luminositys[i]           .item(),
                                                                    phase_cyan_reds[i]             .item(),
                                                                    phase_lime_purples[i]          .item(),
                                                                    phase_pattern_structures[i]    .item(),

                                                                    magnitude_luminositys[i]       .item(),
                                                                    magnitude_cyan_reds[i]         .item(),
                                                                    magnitude_lime_purples[i]      .item(),
                                                                    magnitude_pattern_structures[i].item()
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_0_batch":               ("LATENT",),

                "phase_luminosity":             ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red":               ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple":            ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure":      ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity":         ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red":           ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple":        ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure":  ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal":              ("BOOLEAN", {"default": False}),
                "latent_out_normal":            ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_luminositys":            ("SIGMAS", ),
                "phase_cyan_reds":              ("SIGMAS", ),
                "phase_lime_purples":           ("SIGMAS", ),
                "phase_pattern_structures":     ("SIGMAS", ),

                "magnitude_luminositys":        ("SIGMAS", ),
                "magnitude_cyan_reds":          ("SIGMAS", ),
                "magnitude_lime_purples":       ("SIGMAS", ),
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
                                    
                                    phase_luminosity,
                                    phase_cyan_red,
                                    phase_lime_purple,
                                    phase_pattern_structure,
                                    
                                    magnitude_luminosity,
                                    magnitude_cyan_red,
                                    magnitude_lime_purple,
                                    magnitude_pattern_structure
                                    ):
        dtype = latent_0.dtype
        # avoid big accuracy problems with fp32 FFT!
        latent_0 = latent_0.double()

        latent_0_fft = torch.fft.fft2(latent_0)

        latent_0_phase     = torch.angle(latent_0_fft)
        latent_0_magnitude = torch.abs  (latent_0_fft)

        # create new complex FFT using weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
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

                                                                    phase_luminositys[i].item(),
                                                                    phase_cyan_reds[i].item(),
                                                                    phase_lime_purples[i].item(),
                                                                    phase_pattern_structures[i].item(),

                                                                    magnitude_luminositys[i].item(),
                                                                    magnitude_cyan_reds[i].item(),
                                                                    magnitude_lime_purples[i].item(),
                                                                    magnitude_pattern_structures[i].item()
                                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )



class LatentPhaseMagnitudeOffset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_0_batch":               ("LATENT",),

                "phase_luminosity":             ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red":               ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple":            ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure":      ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity":         ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red":           ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple":        ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure":  ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal":              ("BOOLEAN", {"default": False}),
                "latent_out_normal":            ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_luminositys":            ("SIGMAS", ),
                "phase_cyan_reds":              ("SIGMAS", ),
                "phase_lime_purples":           ("SIGMAS", ),
                "phase_pattern_structures":     ("SIGMAS", ),

                "magnitude_luminositys":        ("SIGMAS", ),
                "magnitude_cyan_reds":          ("SIGMAS", ),
                "magnitude_lime_purples":       ("SIGMAS", ),
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
     
                                    phase_luminosity,
                                    phase_cyan_red,
                                    phase_lime_purple,
                                    phase_pattern_structure,
                                                                        
                                    magnitude_luminosity,
                                    magnitude_cyan_red,
                                    magnitude_lime_purple,
                                    magnitude_pattern_structure
                                    ):
        dtype = latent_0.dtype
        # avoid big accuracy problems with fp32 FFT!
        latent_0 = latent_0.double()

        latent_0_fft = torch.fft.fft2(latent_0)

        latent_0_phase = torch.angle(latent_0_fft)
        latent_0_magnitude = torch.abs(latent_0_fft)

        # create new complex FFT using a weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
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

                                                                    phase_luminositys[i]           .item(),
                                                                    phase_cyan_reds[i]             .item(),
                                                                    phase_lime_purples[i]          .item(),
                                                                    phase_pattern_structures[i]    .item(),

                                                                    magnitude_luminositys[i]       .item(),
                                                                    magnitude_cyan_reds[i]         .item(),
                                                                    magnitude_lime_purples[i]      .item(),
                                                                    magnitude_pattern_structures[i].item()
                                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )



class LatentPhaseMagnitudePower:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_0_batch":               ("LATENT",),

                "phase_luminosity":             ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_cyan_red":               ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_lime_purple":            ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "phase_pattern_structure":      ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "magnitude_luminosity":         ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_cyan_red":           ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_lime_purple":        ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "magnitude_pattern_structure":  ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),

                "latent_0_normal":              ("BOOLEAN", {"default": False}),
                "latent_out_normal":            ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_luminositys":            ("SIGMAS", ),
                "phase_cyan_reds":              ("SIGMAS", ),
                "phase_lime_purples":           ("SIGMAS", ),
                "phase_pattern_structures":     ("SIGMAS", ),

                "magnitude_luminositys":        ("SIGMAS", ),
                "magnitude_cyan_reds":          ("SIGMAS", ),
                "magnitude_lime_purples":       ("SIGMAS", ),
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
                                    phase_luminosity,
                                    phase_cyan_red,
                                    phase_lime_purple,
                                    phase_pattern_structure,
                                                                        
                                    magnitude_luminosity,
                                    magnitude_cyan_red,
                                    magnitude_lime_purple,
                                    magnitude_pattern_structure
                                    ):
        dtype = latent_0.dtype
        # avoid big accuracy problems with fp32 FFT!
        latent_0 = latent_0.double()

        latent_0_fft = torch.fft.fft2(latent_0)

        latent_0_phase = torch.angle(latent_0_fft)
        latent_0_magnitude = torch.abs(latent_0_fft)

        # create new complex FFT using a weighted mix of phases
        chan_weights_phase     = [w for w in [phase_luminosity,     phase_cyan_red,     phase_lime_purple,     phase_pattern_structure    ]]
        chan_weights_magnitude = [w for w in [magnitude_luminosity, magnitude_cyan_red, magnitude_lime_purple, magnitude_pattern_structure]]
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

                                                                    phase_luminositys[i]           .item(),
                                                                    phase_cyan_reds[i]             .item(),
                                                                    phase_lime_purples[i]          .item(),
                                                                    phase_pattern_structures[i]    .item(),

                                                                    magnitude_luminositys[i]       .item(),
                                                                    magnitude_cyan_reds[i]         .item(),
                                                                    magnitude_lime_purples[i]      .item(),
                                                                    magnitude_pattern_structures[i].item()
                                                                    )
            if latent_out_normal == True:
                mixed_phase_magnitude = latent_normalize_channels(mixed_phase_magnitude)

            mixed_phase_magnitude_batch[i, :, :, :] = mixed_phase_magnitude

        return ({"samples": mixed_phase_magnitude_batch}, )



class StableCascade_StageC_VAEEncode_Exact:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
                "height": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
            }
        }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_c",)
    FUNCTION     = "generate"
    CATEGORY     = "RES4LYF/vae"
    
    def generate(self, image, vae, width, height):
        out_width  = (width)  * vae.downscale_ratio #downscale_ratio = 32
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE", ),
                "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 16, "min": 8, "max": 128, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_c",)
    FUNCTION     = "generate"
    CATEGORY     = "RES4LYF/vae"

    def generate(self, image, vae, tile_size, overlap):

        upscale_amount = vae.downscale_ratio  # downscale_ratio = 32

        image = image.movedim(-1, 1)  # bhwc -> bchw 

        encode_fn = lambda img: vae.encode(img.to(vae.device)).to("cpu")

        c_latent = tiled_scale_multidim(image,
                                        encode_fn,
                                        tile           = (tile_size // 8, tile_size // 8),
                                        overlap        = overlap,
                                        upscale_amount = upscale_amount,
                                        out_channels   = 16, 
                                        output_device  = self.device
                                        )

        return ({"samples": c_latent,},)

@torch.inference_mode()
def tiled_scale_multidim(samples,
                        function,
                        tile           = (64, 64),
                        overlap        = 8,
                        upscale_amount = 4,
                        out_channels   = 3,
                        output_device  = "cpu",
                        pbar           = None
                        ):
    
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
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "width":       ("INT",                                       {"default": 24, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height":      ("INT",                                       {"default": 24, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "batch_size":  ("INT",                                       {"default": 1,  "min": 1, "max": 4096}),
                "channels":    (['4', '16'],                                 {"default": '4'}),
                "mode":        (['sdxl', 'cascade_b', 'cascade_c', 'exact'], {"default": 'default'}),
                "compression": ("INT",                                       {"default": 42, "min": 4, "max": 128, "step": 1}),
                "precision":   (['fp16', 'fp32', 'fp64'],                    {"default": 'fp32'}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "RES4LYF/latents"

    def generate(self,
                width,
                height,
                batch_size,
                channels,
                mode,
                compression,
                precision
                ):
        
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

        latent = torch.zeros([batch_size,
                            c,
                            height // ratio, 
                            width // ratio], 
                            dtype=dtype, 
                            device=self.device)
        
        return ({"samples":latent}, )

class EmptyLatentImage64:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "width":      ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height":     ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
                }
            }
        
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "generate"
    CATEGORY     = "RES4LYF/latents"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], dtype=torch.float64, device=self.device)
        return ({"samples":latent}, )




class LatentNoiseBatch_perlin:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls): 
        return {"required": {
            "seed":         ("INT",   {"default": 0,    "min": 0, "max": 0xffffffffffffffff}),
            "width":        ("INT",   {"default": 1024, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
            "height":       ("INT",   {"default": 1024, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
            "batch_size":   ("INT",   {"default": 1,    "min": 1, "max": 256}),
            "detail_level": ("FLOAT", {"default": 0,    "min":-1, "max": 1.0,            "step": 0.1}),
            },
            "optional": {
                "details":  ("SIGMAS", ),
            }
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":                  ("LATENT",),
                "mean":                    ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_luminosity":         ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_cyan_red":           ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_lime_purple":        ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "mean_pattern_structure":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std":                     ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps":                   ("INT",   {"default": 0,   "min": -10000,   "max": 10000}),
                "seed":                    ("INT",   {"default": 0,   "min": 0,        "max": 0xffffffffffffffff}),
            },
            "optional": {
                "means":                   ("SIGMAS", ),
                "mean_luminositys":        ("SIGMAS", ),
                "mean_cyan_reds":          ("SIGMAS", ),
                "mean_lime_purples":       ("SIGMAS", ),
                "mean_pattern_structures": ("SIGMAS", ),
                "stds":                    ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/noise"

    @staticmethod
    def gaussian_noise_channels(x, mean_luminosity = -0.1, mean_cyan_red = 0.0, mean_lime_purple=0.0, mean_pattern_structure=0.0):
        x = x.squeeze(0)

        luminosity        = x[0:1] + mean_luminosity
        cyan_red          = x[1:2] + mean_cyan_red
        lime_purple       = x[2:3] + mean_lime_purple
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

        means                   = initialize_or_scale(means                  , mean                  , steps)
        mean_luminositys        = initialize_or_scale(mean_luminositys       , mean_luminosity       , steps)
        mean_cyan_reds          = initialize_or_scale(mean_cyan_reds         , mean_cyan_red         , steps)
        mean_lime_purples       = initialize_or_scale(mean_lime_purples      , mean_lime_purple      , steps)
        mean_pattern_structures = initialize_or_scale(mean_pattern_structures, mean_pattern_structure, steps)

        stds = initialize_or_scale(stds, std, steps)

        for i in range(steps):
            noise = noise_sampler(mean=means[i].item(), std=stds[i].item())
            noise = self.gaussian_noise_channels(noise, mean_luminositys[i].item(), mean_cyan_reds[i].item(), mean_lime_purples[i].item(), mean_pattern_structures[i].item())
            noise_latents[i] = x + noise

        return ({"samples": noise_latents}, )

class LatentNoiseBatch_gaussian:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mean":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "std":    ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "steps":  ("INT",   {"default": 0,   "min": -10000,   "max": 10000}),
                "seed":   ("INT",   {"default": 0,   "min": 0,        "max": 0xffffffffffffffff}),
            },
            "optional": {
                "means":  ("SIGMAS", ),
                "stds":   ("SIGMAS", ),
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
        stds  = initialize_or_scale(stds,  std,  steps)    

        latent_samples = latent["samples"]
        b, c, h, w = latent_samples.shape  

        noise_latents = torch.zeros([steps, c, h, w], dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('gaussian')(x=latent_samples, seed = seed)

        for i in range(steps):
            noise_latents[i] = noise_sampler(mean=means[i].item(), std=stds[i].item())
        return ({"samples": noise_latents}, )

class LatentNoiseBatch_fractal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "alpha":  ("FLOAT",   {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.001}),
                "k_flip": ("BOOLEAN", {"default": False}),
                "steps":  ("INT",     {"default": 0,   "min": -10000,   "max": 10000}),
                "seed":   ("INT",     {"default": 0,   "min": 0,        "max": 0xffffffffffffffff}),
            },
            "optional": {
                "alphas": ("SIGMAS", ),
                "ks":     ("SIGMAS", ),
                "steps_": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/noise"

    def main(self,
            latent,
            alpha,
            k_flip,
            steps,
            seed    = 42,
            alphas  = None,
            ks      = None,
            sigmas_ = None,
            steps_  = None
            ):
        
        if steps_ is not None:
            steps = len(steps_)

        alphas = initialize_or_scale(alphas, alpha, steps)
        k_flip = -1 if k_flip else 1
        ks     = initialize_or_scale(ks  , k_flip, steps)

        latent_samples = latent["samples"]
        b, c, h, w = latent_samples.shape  
        noise_latents = torch.zeros([steps, c, h, w], dtype=latent_samples.dtype, layout=latent_samples.layout, device=latent_samples.device)

        noise_sampler = NOISE_GENERATOR_CLASSES.get('fractal')(x=latent_samples, seed = seed)

        for i in range(steps):
            noise_latents[i] = noise_sampler(alpha=alphas[i].item(), k=ks[i].item(), scale=0.1)

        return ({"samples": noise_latents}, )


class LatentBatch_channels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":             ("LATENT",),
                "mode":               (["offset", "multiply", "power"],),
                "luminosity":         ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "cyan_red":           ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "lime_purple":        ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "pattern_structure":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            },
            "optional": {
                "luminositys":        ("SIGMAS", ),
                "cyan_reds":          ("SIGMAS", ),
                "lime_purples":       ("SIGMAS", ),
                "pattern_structures": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/latents"
    
    @staticmethod
    def latent_channels_multiply(x, luminosity = -0.1, cyan_red = 0.0, lime_purple=0.0, pattern_structure=0.0):
        luminosity        = x[0:1] * luminosity
        cyan_red          = x[1:2] * cyan_red
        lime_purple       = x[2:3] * lime_purple
        pattern_structure = x[3:4] * pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        return x

    @staticmethod
    def latent_channels_offset(x, luminosity = -0.1, cyan_red = 0.0, lime_purple=0.0, pattern_structure=0.0):
        luminosity        = x[0:1] + luminosity
        cyan_red          = x[1:2] + cyan_red
        lime_purple       = x[2:3] + lime_purple
        pattern_structure = x[3:4] + pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        return x
    
    @staticmethod
    def latent_channels_power(x, luminosity = -0.1, cyan_red = 0.0, lime_purple=0.0, pattern_structure=0.0):
        luminosity        = x[0:1] ** luminosity
        cyan_red          = x[1:2] ** cyan_red
        lime_purple       = x[2:3] ** lime_purple
        pattern_structure = x[3:4] ** pattern_structure

        x = torch.unsqueeze(torch.cat([luminosity, cyan_red, lime_purple, pattern_structure]), 0)
        return x

    def main(self,
            latent,
            mode,
            luminosity,
            cyan_red,
            lime_purple,
            pattern_structure,
            
            luminositys        = None,
            cyan_reds          = None,
            lime_purples       = None,
            pattern_structures = None):
        
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":   ("LATENT",),
                "mode":     (["offset", "multiply", "power"],),
                "chan_1":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_2":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_3":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_4":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_5":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_6":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_7":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_8":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_9":   ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_10":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_11":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_12":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_13":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_14":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_15":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "chan_16":  ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }, 
            "optional": { 
                "chan_1s":  ("SIGMAS", ),
                "chan_2s":  ("SIGMAS", ),
                "chan_3s":  ("SIGMAS", ),
                "chan_4s":  ("SIGMAS", ),
                "chan_5s":  ("SIGMAS", ),
                "chan_6s":  ("SIGMAS", ),
                "chan_7s":  ("SIGMAS", ),
                "chan_8s":  ("SIGMAS", ),
                "chan_9s":  ("SIGMAS", ),
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
        chan_1  = x[0:1]   * chan_1
        chan_2  = x[1:2]   * chan_2
        chan_3  = x[2:3]   * chan_3
        chan_4  = x[3:4]   * chan_4
        chan_5  = x[4:5]   * chan_5
        chan_6  = x[5:6]   * chan_6
        chan_7  = x[6:7]   * chan_7
        chan_8  = x[7:8]   * chan_8
        chan_9  = x[8:9]   * chan_9
        chan_10 = x[9:10]  * chan_10
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
        chan_1  = x[0:1]   + chan_1
        chan_2  = x[1:2]   + chan_2
        chan_3  = x[2:3]   + chan_3
        chan_4  = x[3:4]   + chan_4
        chan_5  = x[4:5]   + chan_5
        chan_6  = x[5:6]   + chan_6
        chan_7  = x[6:7]   + chan_7
        chan_8  = x[7:8]   + chan_8
        chan_9  = x[8:9]   + chan_9
        chan_10 = x[9:10]  + chan_10
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent":     ("LATENT", ),     
                    "mode":      (["full", "channels"],), 
                    "operation": (["normalize", "center", "standardize"],), 
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

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





class latent_channelwise_match:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "model":         ("MODEL",),
                    "latent_target": ("LATENT", ),      
                    "latent_source": ("LATENT", ),      
                     },
            "optional": {
                    "mask_target":   ("MASK", ),      
                    "mask_source":   ("MASK", ),   
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
            }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_matched",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self,
            model,
            latent_target,
            mask_target,
            latent_source,
            mask_source,
            extra_options
            ):
        
        #EO = ExtraOptions(extra_options)
        dtype = latent_target['samples'].dtype

        exclude_channels = get_extra_options_list(exclude_channels, -1, extra_options)
        
        if extra_options_flag("disable_process_latent", extra_options):
            x_target = latent_target['samples'].clone()
            x_source = latent_source['samples'].clone()
        else:
            x_target = model.model.process_latent_in(latent_target['samples']).clone().to(torch.float64)
            x_source = model.model.process_latent_in(latent_source['samples']).clone().to(torch.float64)
        
        if mask_target is None:
            mask_target = torch.ones_like(x_target)
        else:
            mask_target = mask_target.unsqueeze(1)
            mask_target = mask_target.repeat(1, x_target.shape[1], 1, 1) 
            mask_target = F.interpolate(mask_target, size=(x_target.shape[2], x_target.shape[3]), mode='bilinear', align_corners=False)
            mask_target = mask_target.to(x_target.dtype).to(x_target.device)
        
        if mask_source is None:
            mask_source = torch.ones_like(x_target)
        else:
            mask_source = mask_source.unsqueeze(1)
            mask_source = mask_source.repeat(1, x_target.shape[1], 1, 1) 
            mask_source = F.interpolate(mask_source, size=(x_target.shape[2], x_target.shape[3]), mode='bilinear', align_corners=False)
            mask_source = mask_source.to(x_target.dtype).to(x_target.device)
        
        x_target_masked     = x_target * ((mask_target==1)*mask_target)
        x_target_masked_inv = x_target - x_target_masked
        #x_source_masked     = x_source * ((mask_source==1)*mask_source)
        
        x_matched = torch.zeros_like(x_target)
        for n in range(x_matched.shape[1]):
            if n in exclude_channels: 
                x_matched[0][n] = x_target[0][n] 
                continue
            
            x_target_masked_values = x_target[0][n][mask_target[0][n] == 1]
            x_source_masked_values = x_source[0][n][mask_source[0][n] == 1]
            
            x_target_masked_values_mean = x_target_masked_values.mean()
            x_target_masked_values_std  = x_target_masked_values.std()
            x_target_masked_source_mean = x_source_masked_values.mean()
            x_target_masked_source_std  = x_source_masked_values.std()
            
            x_target_mean = x_target.mean()
            x_target_std  = x_target.std()
            x_source_mean = x_source.mean()
            x_source_std  = x_source.std()
            
            #if re.search(r"\benable_std\b", extra_options) == None:
            if not extra_options_flag("enable_std", extra_options):
                x_target_std = x_target_masked_values_std = x_target_masked_source_std = 1
                
            #if re.search(r"\bdisable_mean\b", extra_options):
            if extra_options_flag("disable_mean", extra_options):
                x_target_mean = x_target_masked_values_mean = x_target_masked_source_mean = 1
            
            #if re.search(r"\bdisable_masks\b", extra_options):
            if extra_options_flag("disable_masks", extra_options):
                x_matched[0][n] = (x_target[0][n] - x_target_mean) / x_target_std
                x_matched[0][n] = (x_matched[0][n] * x_source_std) + x_source_mean
            else:
                x_matched[0][n] = (x_target_masked[0][n] - x_target_masked_values_mean) / x_target_masked_values_std
                x_matched[0][n] = (x_matched[0][n] * x_target_masked_source_std) + x_target_masked_source_mean
                x_matched[0][n] = x_target_masked_inv[0][n] + x_matched[0][n] * ((mask_target[0][n]==1)*mask_target[0][n])
        
        #if re.search(r"\bdisable_process_latent\b", extra_options) == None: 
        if not extra_options_flag("disable_process_latent", extra_options):
            x_matched = model.model.process_latent_out(x_matched).clone()
            
        
        return ({"samples": x_matched.to(dtype)}, )
                
                
    