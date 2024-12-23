# Code adapted from https://github.com/comfyanonymous/ComfyUI/

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
from comfy.cli_args import args
from comfy_extras.nodes_model_advanced import ModelSamplingSD3, ModelSamplingFlux, ModelSamplingAuraFlow, ModelSamplingStableCascade


import torch

import folder_paths
import os
import json
import math

import comfy.model_management
    
from .flux.model  import ReFlux
from .flux.layers import SingleStreamBlock as ReSingleStreamBlock, DoubleStreamBlock as ReDoubleStreamBlock

from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock


class ReFluxPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "enable": ("BOOLEAN", {"default": True}),
           }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    CATEGORY = "model_patches"
    FUNCTION = "main"

    def main(self, model, enable=True):
        m = model #.clone()
        
        if enable:
            m.model.diffusion_model.__class__ = ReFlux
            m.model.diffusion_model.threshold_inv = False
            
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                block.__class__ = ReDoubleStreamBlock
                block.idx = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                block.__class__ = ReSingleStreamBlock
                block.idx = i
        else:
            m.model.diffusion_model.__class__ = Flux
            
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                block.__class__ = DoubleStreamBlock
                block.idx = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                block.__class__ = SingleStreamBlock
                block.idx = i
        
        return (m,)
    
    
    
class FluxGuidanceDisable:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "disable": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    CATEGORY = "model_patches"
    FUNCTION = "main"

    def main(self, model, disable=True):
        m = model.clone()
        if disable:
            m.model.diffusion_model.params.guidance_embed = False
        else:
            m.model.diffusion_model.params.guidance_embed = True
        return (m,)



def time_snr_shift_exponential(alpha, t):
    return math.exp(alpha) / (math.exp(alpha) + (1 / t - 1) ** 1.0)

def time_snr_shift_linear(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

class ModelSamplingAdvanced:
    # this is used to set the "shift" using either exponential scaling (default for SD3.5M and Flux) or linear scaling (default for SD3.5L and SD3 2B beta)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "scaling": (["exponential", "linear"], {"default": 'exponential'}), 
                    "shift": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    #"base_shift": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "model_patches"

    def sigma_exponential(self, timestep):
        return time_snr_shift_exponential(self.timestep_shift, timestep / self.multiplier)

    def sigma_linear(self, timestep):
        return time_snr_shift_linear(self.timestep_shift, timestep / self.multiplier)

    def main(self, model, scaling, shift):
        m = model.clone()
        
        self.timestep_shift = shift
        self.multiplier = 1000
        timesteps = 1000
        
        if isinstance(m.model.model_config, comfy.supported_models.Flux) or isinstance(m.model.model_config, comfy.supported_models.FluxSchnell):
            self.multiplier = 1
            timesteps = 10000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.AuraFlow):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.SD3):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        m.object_patches['model_sampling'] = m.model.model_sampling = ModelSamplingAdvanced(m.model.model_config)

        m.model.model_sampling.__dict__['shift'] = self.timestep_shift
        m.model.model_sampling.__dict__['multiplier'] = self.multiplier

        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)
        if scaling == "exponential": 
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear": 
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        m.model.model_sampling.register_buffer('sigmas', ts)
        m.object_patches['model_sampling'].sigmas = m.model.model_sampling.sigmas
        
        return (m,)

class ModelSamplingAdvancedResolution:
    # this is used to set the "shift" using either exponential scaling (default for SD3.5M and Flux) or linear scaling (default for SD3.5L and SD3 2B beta)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "scaling": (["exponential", "linear"], {"default": 'exponential'}), 
                    "max_shift": ("FLOAT", {"default": 1.35, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "base_shift": ("FLOAT", {"default": 0.85, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "latent_image": ("LATENT",),
                }
               }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "model_shift"

    def sigma_exponential(self, timestep):
        return time_snr_shift_exponential(self.timestep_shift, timestep / self.multiplier)

    def sigma_linear(self, timestep):
        return time_snr_shift_linear(self.timestep_shift, timestep / self.multiplier)

    def main(self, model, scaling, max_shift, base_shift, latent_image):
        m = model.clone()
        height, width = latent_image['samples'].shape[2:]
        
        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b
        
        self.timestep_shift = shift
        self.multiplier = 1000
        timesteps = 1000
        
        if isinstance(m.model.model_config, comfy.supported_models.Flux) or isinstance(m.model.model_config, comfy.supported_models.FluxSchnell):
            self.multiplier = 1
            timesteps = 10000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST

            
        elif isinstance(m.model.model_config, comfy.supported_models.AuraFlow):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.SD3):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        m.object_patches['model_sampling'] = m.model.model_sampling = ModelSamplingAdvanced(m.model.model_config)

        m.model.model_sampling.__dict__['shift'] = self.timestep_shift
        m.model.model_sampling.__dict__['multiplier'] = self.multiplier

        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)
        if scaling == "exponential": 
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear": 
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        m.model.model_sampling.register_buffer('sigmas', ts)
        m.object_patches['model_sampling'].sigmas = m.model.model_sampling.sigmas
        
        return (m,)
    
    
class UNetSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "models/ComfyUI"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "advanced/model_merging"
    DESCRIPTION = "Save a .safetensors containing only the model data."

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        save_checkpoint(model, clip=None, vae=None, filename_prefix=filename_prefix, output_dir=self.output_dir, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {}


def save_checkpoint(model, clip=None, vae=None, clip_vision=None, filename_prefix=None, output_dir=None, prompt=None, extra_pnginfo=None):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {}

    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        if isinstance(model.model, comfy.model_base.SDXL_instructpix2pix):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-edit"
        else:
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    elif isinstance(model.model, comfy.model_base.SVD_img2vid):
        metadata["modelspec.architecture"] = "stable-video-diffusion-img2vid-v1"
    elif isinstance(model.model, comfy.model_base.SD3):
        metadata["modelspec.architecture"] = "stable-diffusion-v3-medium" #TODO: other SD3 variants
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    #TODO:
    # "stable-diffusion-v1", "stable-diffusion-v1-inpainting", "stable-diffusion-v2-512",
    # "stable-diffusion-v2-768-v", "stable-diffusion-v2-unclip-l", "stable-diffusion-v2-unclip-h",
    # "v2-inpainting"

    extra_keys = {}
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"

    if not args.disable_metadata:
        metadata["prompt"] = prompt_info
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    output_checkpoint = f"{filename}_{counter:05}_.safetensors"
    output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

    sd_save_checkpoint(output_checkpoint, model, clip, vae, clip_vision, metadata=metadata, extra_keys=extra_keys)


def sd_save_checkpoint(output_path, model, clip=None, vae=None, clip_vision=None, metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()

    comfy.model_management.load_models_gpu(load_models, force_patch_weights=True)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    vae_sd = vae.get_sd() if vae is not None else None                             #THIS ALLOWS SAVING UNET ONLY
    sd = model.model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)






class TorchCompileModelFluxAdvanced: #adapted from https://github.com/kijai/ComfyUI-KJNodes
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model": ("MODEL",),
                    "backend": (["inductor", "cudagraphs"],),
                    "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("STRING", {"default": "0-18", "multiline": True}),
                    "single_blocks": ("STRING", {"default": "0-37", "multiline": True}),
                    "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"
    EXPERIMENTAL = True

    def parse_blocks(self, blocks_str):
        blocks = []
        for part in blocks_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                blocks.extend(range(start, end + 1))
            else:
                blocks.append(int(part))
        return blocks

    def patch(self, model, backend, mode, fullgraph, single_blocks, double_blocks, dynamic):
        single_block_list = self.parse_blocks(single_blocks)
        double_block_list = self.parse_blocks(double_blocks)
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.double_blocks):
                    if i in double_block_list:
                        #print("Compiling double_block", i)
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                for i, block in enumerate(diffusion_model.single_blocks):
                    if i in single_block_list:
                        #print("Compiling single block", i)
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model")
        
        return (m, )
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)
