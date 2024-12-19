import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
    
import itertools

import torch
import math
import re

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


class latent_channelwise_match:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "model": ("MODEL",),
                    "latent_target": ("LATENT", ),      
                    "latent_source": ("LATENT", ),      
                     },
            "optional": {
                    "mask_target": ("MASK", ),      
                    "mask_source": ("MASK", ),   
                    "extra_options": ("STRING", {"default": "", "multiline": True}),   
            }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_matched",)
    CATEGORY = "RES4LYF/latents"

    FUNCTION = "main"

    def main(self, model, latent_target, mask_target, latent_source, mask_source, extra_options):
        
        dtype = latent_target['samples'].dtype
        
        exclude_channels_match = re.search(r"exclude_channels=([\d,]+)", extra_options)
        exclude_channels = []
        if exclude_channels_match:
            exclude_channels = [int(ch.strip()) for ch in exclude_channels_match.group(1).split(",")]
        
        if re.search(r"\bdisable_process_latent\b", extra_options): 
            x_target = latent_target['samples'].clone()
            x_source = latent_source['samples'].clone()
        else:
            #x_target = model.inner_model.inner_model.process_latent_in(latent_target['samples']).clone() 
            #x_source = model.inner_model.inner_model.process_latent_in(latent_source['samples']).clone()
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
            x_target_masked_values_std = x_target_masked_values.std()
            x_target_masked_source_mean = x_source_masked_values.mean()
            x_target_masked_source_std = x_source_masked_values.std()
            
            x_target_mean = x_target.mean()
            x_target_std = x_target.std()
            x_source_mean = x_source.mean()
            x_source_std = x_source.std()
            
            if re.search(r"\benable_std\b", extra_options) == None:
                x_target_std = x_target_masked_values_std = x_target_masked_source_std = 1
                
            if re.search(r"\bdisable_mean\b", extra_options):
                x_target_mean = x_target_masked_values_mean = x_target_masked_source_mean = 1
            
            if re.search(r"\bdisable_masks\b", extra_options):
                x_matched[0][n] = (x_target[0][n] - x_target_mean) / x_target_std
                x_matched[0][n] = (x_matched[0][n] * x_source_std) + x_source_mean
            else:
                x_matched[0][n] = (x_target_masked[0][n] - x_target_masked_values_mean) / x_target_masked_values_std
                x_matched[0][n] = (x_matched[0][n] * x_target_masked_source_std) + x_target_masked_source_mean
                x_matched[0][n] = x_target_masked_inv[0][n] + x_matched[0][n] * ((mask_target[0][n]==1)*mask_target[0][n])
        
        if re.search(r"\bdisable_process_latent\b", extra_options) == None: 
            x_matched = model.model.process_latent_out(x_matched).clone()
            
        
        return ({"samples": x_matched.to(dtype)}, )
                
                
    