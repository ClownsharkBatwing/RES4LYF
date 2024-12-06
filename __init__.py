from . import legacy_sampler_rk
from . import legacy_samplers
from . import rk_sampler
from . import samplers
from . import samplers_extensions
from . import samplers_tiled
from . import loaders
from . import sigmas
from . import latents
from . import latent_images
from . import conditioning
from . import images
from . import models
from . import helper_sigma_preview_image_preproc
from . import nodes_misc
from .res4lyf import init, get_ext_dir

import torch

import os
import shutil
import sys

flags = {
    "test_samplers": False,
}
try:
    from . import test_samplers
    flags["test_samplers"] = True
    print("Importing test_samplers.py")
except ImportError:
    pass

res4lyf.init()

discard_penultimate_sigma_samplers = set((
))

def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
    added = 0
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # *should* be last item in samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # add custom samplers (presumably) to end of list
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

extra_samplers = {
    "res_2m": rk_sampler.sample_res_2m,
    "res_2s": rk_sampler.sample_res_2s,
    "res_3s": rk_sampler.sample_res_3s,
    "res_5s": rk_sampler.sample_res_5s,
    "res_2m_sde": rk_sampler.sample_res_2m_sde,
    "res_2s_sde": rk_sampler.sample_res_2s_sde,
    "res_3s_sde": rk_sampler.sample_res_3s_sde,
    "res_5s_sde": rk_sampler.sample_res_5s_sde,
    "deis_2m": rk_sampler.sample_deis_2m,
    "deis_3m": rk_sampler.sample_deis_3m,
    "deis_4m": rk_sampler.sample_deis_4m,
    "deis_2m_sde": rk_sampler.sample_deis_2m_sde,
    "deis_3m_sde": rk_sampler.sample_deis_3m_sde,
    "deis_4m_sde": rk_sampler.sample_deis_4m_sde,
    "rk": rk_sampler.sample_rk,
    "legacy_rk": legacy_sampler_rk.legacy_sample_rk,
}

extra_samplers = dict(reversed(extra_samplers.items()))

NODE_CLASS_MAPPINGS = {
    "Legacy_ClownSampler": legacy_samplers.Legacy_SamplerRK,
    "Legacy_SharkSampler": legacy_samplers.Legacy_SharkSampler,
    "Legacy_ClownsharKSampler": legacy_samplers.Legacy_ClownsharKSampler,
    "Legacy_ClownsharKSamplerGuides": legacy_samplers.Legacy_ClownsharKSamplerGuides,
    
    "ClownSampler": samplers.ClownSampler,
    "SharkSampler": samplers.SharkSampler,
    "ClownsharKSampler": samplers.ClownsharKSampler,
    "ClownsharKSamplerGuides": samplers_extensions.ClownsharKSamplerGuides,
    
    "ClownsharKSamplerOptions": samplers_extensions.ClownsharKSamplerOptions,
    "ClownsharKSamplerOptions_SDE_Noise": samplers_extensions.ClownsharKSamplerOptions_SDE_Noise,
    "ClownsharKSamplerAutomation": samplers_extensions.ClownsharKSamplerAutomation,

    "UltraSharkSampler": samplers.UltraSharkSampler,
    "UltraSharkSampler Tiled": samplers_tiled.UltraSharkSampler_Tiled,

    "SamplerOptions_TimestepScaling": samplers_extensions.SamplerOptions_TimestepScaling,
    "SamplerOptions_GarbageCollection": samplers_extensions.SamplerOptions_GarbageCollection,
    "ModelTimestepPatcher": models.ModelTimestepPatcher,
    
    "AdvancedNoise": latents.AdvancedNoise,
    
    "FluxLoader": loaders.FluxLoader,
    "SD35Loader": loaders.SD35Loader,
    
    "TextBox1": nodes_misc.TextBox1,
    "TextBox3": nodes_misc.TextBox3,
    
    "CLIPTextEncodeFluxUnguided": conditioning.CLIPTextEncodeFluxUnguided,

    "ConditioningAverageScheduler": conditioning.ConditioningAverageScheduler,
    "ConditioningMultiply": conditioning.ConditioningMultiply,
    "Conditioning Recast FP64": conditioning.Conditioning_Recast64,
    "StableCascade_StageB_Conditioning64": conditioning.StableCascade_StageB_Conditioning64,
    "ConditioningZeroAndTruncate": conditioning.ConditioningZeroAndTruncate,
    "ConditioningTruncate": conditioning.ConditioningTruncate,
    "StyleModelApplyAdvanced": conditioning.StyleModelApplyAdvanced,

    "ConditioningToBase64": conditioning.ConditioningToBase64,
    "Base64ToConditioning": conditioning.Base64ToConditioning,

    "Set Precision": latents.set_precision,
    "Set Precision Universal": latents.set_precision_universal,
    "Set Precision Advanced": latents.set_precision_advanced,
    
    "Latent Match Channelwise": latent_images.latent_channelwise_match,

    "LatentNoised": latents.LatentNoised,
    "LatentNoiseList": latents.LatentNoiseList,
    "LatentBatch_channels": latents.LatentBatch_channels,
    "LatentBatch_channels_16": latents.LatentBatch_channels_16,
    "LatentNoiseBatch_perlin": latents.LatentNoiseBatch_perlin,
    "LatentNoiseBatch_fractal": latents.LatentNoiseBatch_fractal,
    "LatentNoiseBatch_gaussian": latents.LatentNoiseBatch_gaussian,
    "LatentNoiseBatch_gaussian_channels": latents.LatentNoiseBatch_gaussian_channels,
    
    "Latent to Cuda": latents.latent_to_cuda,
    "Latent Batcher": latents.latent_batch,
    "Latent Normalize Channels": latents.latent_normalize_channels,

    "EmptyLatentImage64": latents.EmptyLatentImage64,
    "EmptyLatentImageCustom": latents.EmptyLatentImageCustom,
    "StableCascade_StageC_VAEEncode_Exact": latents.StableCascade_StageC_VAEEncode_Exact,

    "LatentPhaseMagnitude": latents.LatentPhaseMagnitude,
    "LatentPhaseMagnitudeMultiply": latents.LatentPhaseMagnitudeMultiply,
    "LatentPhaseMagnitudeOffset": latents.LatentPhaseMagnitudeOffset,
    "LatentPhaseMagnitudePower": latents.LatentPhaseMagnitudePower,

    "Sigmas Recast": sigmas.set_precision_sigmas,
    "Sigmas Noise Inversion": sigmas.sigmas_noise_inversion,
    "Sigmas From Text": sigmas.sigmas_from_text, 

    "Sigmas Variance Floor": sigmas.sigmas_variance_floor,
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
    
    "Sigmas Rescale": sigmas.sigmas_rescale,

    "Sigmas Math1": sigmas.sigmas_math1,
    "Sigmas Math3": sigmas.sigmas_math3,

    "Sigmas Iteration Karras": sigmas.sigmas_iteration_karras,
    "Sigmas Iteration Polyexp": sigmas.sigmas_iteration_polyexp,

    "Tan Scheduler": sigmas.tan_scheduler,
    "Tan Scheduler 2": sigmas.tan_scheduler_2stage,
    "Tan Scheduler 2 Simple": sigmas.tan_scheduler_2stage_simple,
    
    #"VGG19StyleTransfer": images.VGG19StyleTransfer,
    "Image Channels LAB": images.Image_Channels_LAB,
    "Image Median Blur": images.ImageMedianBlur,
    "Image Pair Split": images.Image_Pair_Split,
    "Image Crop Location Exact": images.Image_Crop_Location_Exact,
    "Film Grain": images.Film_Grain,
    "Frequency Separation Hard Light": images.Frequency_Separation_Hard_Light,
    "Frequency Separation Hard Light LAB": images.Frequency_Separation_Hard_Light_LAB,
    
    "UNetSave": models.UNetSave,
    
    "PrepForUnsampling": helper_sigma_preview_image_preproc.PrepForUnsampling,
    
    "SigmasPreview": helper_sigma_preview_image_preproc.SigmasPreview,
}

if flags["test_samplers"]:
    NODE_CLASS_MAPPINGS.update({
        "SamplerRK_Test": test_samplers.SamplerRK_Test,
    })
    extra_samplers.update({
        "rk_test":  test_samplers.sample_rk_test,
    })

WEB_DIRECTORY = "./web/js"
__all__ = ["NODE_CLASS_MAPPINGS",  "WEB_DIRECTORY"]


add_samplers()

