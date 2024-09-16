from . import extra_samplers
from . import samplers
from . import samplers_tiled
from . import sigmas
from . import latents
from . import conditioning
from . import images
from . import models
from .res4lyf import init, get_ext_dir

import torch

import os
import shutil

"""def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_web_ext_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../web/extensions/res4lyf"))

def install_js():
    src_dir = get_ext_dir("web/js")
    dst_dir = get_web_ext_dir()
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.islink(dst_dir):
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

install_js()"""

extra_samplers.add_samplers()

#torch.set_default_dtype(torch.float64)

NODE_CLASS_MAPPINGS = {
    "ConditioningAverageScheduler": conditioning.ConditioningAverageScheduler,
    "ConditioningMultiply": conditioning.ConditioningMultiply,
    "ConditioningToBase64": conditioning.ConditioningToBase64,
    "Conditioning Recast FP64": conditioning.Conditioning_Recast64,
    "Base64ToConditioning": conditioning.Base64ToConditioning,

    "LatentNoised": samplers.LatentNoised,
    "LatentNoiseList": latents.LatentNoiseList,
    #"LatentBatch_channels_offset": latents.LatentBatch_channels_offset,
    "LatentBatch_channels": latents.LatentBatch_channels,
    "LatentBatch_channels_16": latents.LatentBatch_channels_16,
    "LatentNoiseBatch_fractal": latents.LatentNoiseBatch_fractal,
    "LatentNoiseBatch_gaussian": latents.LatentNoiseBatch_gaussian,
    "LatentNoiseBatch_gaussian_channels": latents.LatentNoiseBatch_gaussian_channels,
    
    "Latent to Cuda": latents.latent_to_cuda,
    "Latent Batcher": latents.latent_batch,
    "Latent Normalize Channels": latents.latent_normalize_channels,
    "Set Precision": latents.set_precision,
    "Set Precision Universal": latents.set_precision_universal,
    "Set Precision Advanced": latents.set_precision_advanced,
    "LatentNoiseBatch_perlin": latents.LatentNoiseBatch_perlin,
    "EmptyLatentImage64": latents.EmptyLatentImage64,
    "EmptyLatentImageCustom": latents.EmptyLatentImageCustom,
    "StableCascade_StageC_VAEEncode_Exact": latents.StableCascade_StageC_VAEEncode_Exact,

    "LatentPhaseMagnitude": latents.LatentPhaseMagnitude,
    "LatentPhaseMagnitudeMultiply": latents.LatentPhaseMagnitudeMultiply,
    "LatentPhaseMagnitudeOffset": latents.LatentPhaseMagnitudeOffset,
    "LatentPhaseMagnitudePower": latents.LatentPhaseMagnitudePower,

    "AdvancedNoise": samplers.AdvancedNoise,
    "ClownGuides": samplers.ClownGuides,
    "ClownSampler": samplers.ClownSampler,
    "SharkSampler": samplers.SharkSampler,
    "UltraSharkSampler": samplers.UltraSharkSampler,
    "UltraSharkSampler Tiled": samplers_tiled.UltraSharkSampler_Tiled,
    "SamplerDEIS_SDE": samplers.SamplerDEIS_SDE,
    "SamplerDPMPP_DualSDE_Advanced": samplers.SamplerDPMPP_DUALSDE_MOMENTUMIZED_ADVANCED,
    "SamplerDPMPP_SDE_Advanced": samplers.SamplerDPMPP_SDE_ADVANCED,
    "SamplerDPMPP_SDE_CFG++_Advanced": samplers.SamplerDPMPP_SDE_CFGPP_ADVANCED,
    "SamplerEulerAncestral_Advanced": samplers.SamplerEulerAncestral_Advanced,
    "SamplerDPMPP_2S_Ancestral_Advanced": samplers.SamplerDPMPP_2S_Ancestral_Advanced,
    "SamplerDPMPP_2M_SDE_Advanced": samplers.SamplerDPMPP_2M_SDE_Advanced,
    "SamplerDPMPP_3M_SDE_Advanced": samplers.SamplerDPMPP_3M_SDE_Advanced,

    "Sigmas Recast": sigmas.set_precision_sigmas,

    #Sigmas Interpolate": sigmas.sigmas_interpolate,
    "Sigmas Truncate": sigmas.sigmas_truncate,
    "Sigmas Start": sigmas.sigmas_start,
    "Sigmas Split": sigmas.sigmas_split,
    "Sigmas Concat": sigmas.sigmas_concatenate,
    "Sigmas Pad": sigmas.sigmas_pad,
    "Sigmas Unpad": sigmas.sigmas_unpad,
    
    "Sigmas SetFloor": sigmas.sigmas_set_floor,
    "Sigmas DeleteBelowFloor": sigmas.sigmas_delete_below_floor,
    "Sigmas DeleteDuplicates": sigmas.sigmas_delete_consecutive_duplicates,
    "Sigmas Cleanup": sigmas.sigmas_cleanup,
    
    "Sigmas Mult": sigmas.sigmas_mult,
    "Sigmas Modulus": sigmas.sigmas_modulus,
    "Sigmas Quotient": sigmas.sigmas_quotient,
    "Sigmas Add": sigmas.sigmas_add,
    "Sigmas Power": sigmas.sigmas_power,
    "Sigmas Abs": sigmas.sigmas_abs,
    
    "Sigmas2 Mult": sigmas.sigmas2_mult,
    "Sigmas2 Add": sigmas.sigmas2_add,

    "Sigmas Math1": sigmas.sigmas_math1,
    "Sigmas Math3": sigmas.sigmas_math3,

    "Sigmas Iteration Karras": sigmas.sigmas_iteration_karras,
    "Sigmas Iteration Polyexp": sigmas.sigmas_iteration_polyexp,

    "Tan Scheduler": sigmas.tan_scheduler,
    "Tan Scheduler 2": sigmas.tan_scheduler_2stage,
    "Tan Scheduler 2 Simple": sigmas.tan_scheduler_2stage_simple,
    
    "StableCascade_StageB_Conditioning64": conditioning.StableCascade_StageB_Conditioning64,
    
    "Film Grain": images.Film_Grain,
    #"Frequency Separation Vivid Light": images.Frequency_Separation_Vivid_Light,
    "Frequency Separation Hard Light": images.Frequency_Separation_Hard_Light,
    #"Frequency Separation FFT": images.Frequency_Separation_FFT,
    #"Frequency Separation Wavelet": images.Frequency_Separation_Wavelet,
    #"Frequency Separation TV": images.Frequency_Separation_TV,
    "Frequency Separation Hard Light LAB": images.Frequency_Separation_Hard_Light_LAB,
    
    "Image Channels LAB": images.Image_Channels_LAB,
    "Image Median Blur": images.ImageMedianBlur,
    "Image Pair Split": images.Image_Pair_Split,
    #"Image Smudge Blur": images.FastSmudgeBlur,
    "Image Crop Location Exact": images.Image_Crop_Location_Exact,

    "UNetSave": models.UNetSave,
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS",  "WEB_DIRECTORY"]

