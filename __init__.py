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

#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

flags = {
    "test_samplers": False,
    "beta_samplers": False,
}
try:
    from . import test_samplers
    flags["test_samplers"] = True
    print("Importing test_samplers.py")
except ImportError:
    pass
try:
    from . import rk_sampler_beta
    flags["beta_samplers"] = True
    print("Importing rk_sampler_beta.py")
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
    "res_6s": rk_sampler.sample_res_6s,
    "res_2m_sde": rk_sampler.sample_res_2m_sde,
    "res_2s_sde": rk_sampler.sample_res_2s_sde,
    "res_3s_sde": rk_sampler.sample_res_3s_sde,
    "res_5s_sde": rk_sampler.sample_res_5s_sde,
    "res_6s_sde": rk_sampler.sample_res_6s_sde,
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
    "ClownSamplerAdvanced": samplers.ClownSamplerAdvanced,
    "SharkSampler": samplers.SharkSampler,
    "ClownsharKSampler": samplers.ClownsharKSampler,
    "ClownsharKSamplerGuides": samplers_extensions.ClownsharKSamplerGuides,
    "ClownsharKSamplerGuide": samplers_extensions.ClownsharKSamplerGuide,
    
    "ClownsharKSamplerGuides_Beta": samplers_extensions.ClownsharKSamplerGuides_Beta,
    "ClownsharKSamplerGuide_Beta": samplers_extensions.ClownsharKSamplerGuide_Beta,
    
    "ClownsharKSamplerGuidesMisc_Beta": samplers_extensions.ClownsharKSamplerGuidesMisc_Beta,
    "ClownsharKSamplerGuideMisc_Beta": samplers_extensions.ClownsharKSamplerGuideMisc_Beta,
    
    "ClownInpaint": samplers_extensions.ClownInpaint,
    "ClownInpaintSimple": samplers_extensions.ClownInpaintSimple,

    
    "ClownsharKSamplerOptions": samplers_extensions.ClownsharKSamplerOptions,
    "ClownsharKSamplerOptions_SDE_Noise": samplers_extensions.ClownsharKSamplerOptions_SDE_Noise,
    "ClownsharkSamplerOptions_FrameWeights": samplers_extensions.ClownsharKSamplerOptions_FrameWeights,
    "ClownsharKSamplerAutomation": samplers_extensions.ClownsharKSamplerAutomation,
    "ClownsharKSamplerAutomation_Beta": samplers_extensions.ClownsharKSamplerAutomation_Beta,
    "ClownsharKSamplerAutomation_Advanced": samplers_extensions.ClownsharKSamplerAutomation_Advanced,

    "UltraSharkSampler": samplers.UltraSharkSampler,
    "UltraSharkSampler Tiled": samplers_tiled.UltraSharkSampler_Tiled,

    "SamplerOptions_TimestepScaling": samplers_extensions.SamplerOptions_TimestepScaling,
    "SamplerOptions_GarbageCollection": samplers_extensions.SamplerOptions_GarbageCollection,
    "ModelSamplingAdvanced": models.ModelSamplingAdvanced,
    "ModelTimestepPatcher": models.ModelSamplingAdvanced,
    "TorchCompileModelFluxAdv": models.TorchCompileModelFluxAdvanced,

    "ModelSamplingAdvancedResolution": models.ModelSamplingAdvancedResolution,
    "FluxGuidanceDisable": models.FluxGuidanceDisable,
    
    "AdvancedNoise": latents.AdvancedNoise,
    
    "FluxLoader": loaders.FluxLoader,
    "SD35Loader": loaders.SD35Loader,
    
    "TextBox1": nodes_misc.TextBox1,
    "TextBox3": nodes_misc.TextBox3,
    
    
    
    "CLIPTextEncodeFluxUnguided": conditioning.CLIPTextEncodeFluxUnguided,
    "ConditioningOrthoCollin": conditioning.ConditioningOrthoCollin,

    "ConditioningAverageScheduler": conditioning.ConditioningAverageScheduler,
    "ConditioningMultiply": conditioning.ConditioningMultiply,
    "ConditioningAdd": conditioning.ConditioningAdd,
    "Conditioning Recast FP64": conditioning.Conditioning_Recast64,
    "StableCascade_StageB_Conditioning64": conditioning.StableCascade_StageB_Conditioning64,
    "ConditioningZeroAndTruncate": conditioning.ConditioningZeroAndTruncate,
    "ConditioningTruncate": conditioning.ConditioningTruncate,
    "StyleModelApplyAdvanced": conditioning.StyleModelApplyAdvanced,
    
    "FluxRegionalPrompt": conditioning.FluxRegionalPrompt,
    
    "FluxRegionalConditioning": conditioning.FluxRegionalConditioning,

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
    
    "MaskToggle": latents.MaskToggle,

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
    "Constant Scheduler": sigmas.constant_scheduler,
    "Linear Quadratic Advanced": sigmas.linear_quadratic_advanced,
    
    #"VGG19StyleTransfer": images.VGG19StyleTransfer,
    "Image Channels LAB": images.Image_Channels_LAB,
    "Image Median Blur": images.ImageMedianBlur,
    "Image Pair Split": images.Image_Pair_Split,
    "Image Crop Location Exact": images.Image_Crop_Location_Exact,
    "Film Grain": images.Film_Grain,
    "Frequency Separation Hard Light": images.Frequency_Separation_Hard_Light,
    "Frequency Separation Hard Light LAB": images.Frequency_Separation_Hard_Light_LAB,
    
    "UNetSave": models.UNetSave,
    
    "PrepForUnsampling": helper_sigma_preview_image_preproc.VAEEncodeAdvanced,
    "VAEEncodeAdvanced": helper_sigma_preview_image_preproc.VAEEncodeAdvanced,
    
    "SigmasPreview": helper_sigma_preview_image_preproc.SigmasPreview,
    "SigmasSchedulePreview": helper_sigma_preview_image_preproc.SigmasSchedulePreview,

    
    "ReFluxPatcher": models.ReFluxPatcher,
    "FluxOrthoCFGPatcher": models.FluxOrthoCFGPatcher,
}

if flags["test_samplers"]:
    NODE_CLASS_MAPPINGS.update({
        "SamplerRK_Test": test_samplers.SamplerRK_Test,
        "Zampler_Test": test_samplers.Zampler_Test,
        "UltraSharkSamplerRBTest": test_samplers.UltraSharkSamplerRBTest,
    })
    extra_samplers.update({
        "rk_test":  test_samplers.sample_rk_test,
        "rk_sphere":  test_samplers.sample_rk_sphere,
        "rk_vpsde":  test_samplers.sample_rk_vpsde,
        "rk_vpsde_ddpm":  test_samplers.sample_rk_vpsde_ddpm,
        "rk_vpsde_csbw":  test_samplers.sample_rk_vpsde_csbw,
        "rk_momentum":  test_samplers.sample_rk_momentum,
        "rk_ralston_2s":  test_samplers.sample_rk_ralston_2s,
        "rk_implicit_res_2s":  test_samplers.sample_rk_implicit_res_2s,
        "res_multistep": test_samplers.sample_res_multistep,
        "rk_res_2m": test_samplers.sample_rk_res_2m,
        "rk_res_2s": test_samplers.sample_rk_res_2s,
        "rk_res_2s_prenoise": test_samplers.sample_rk_res_2s_prenoise,
        "rk_ralston_2s_prenoise": test_samplers.sample_rk_ralston_2s_prenoise,

        "rk_ralston_2s_prenoise_alt": test_samplers.sample_rk_ralston_2s_prenoise_alt,
        "rk_ralston_2s_prenoise_alt2": test_samplers.sample_rk_ralston_2s_prenoise_alt2,
        "rk_ralston_3s_prenoise_alt2": test_samplers.sample_rk_ralston_3s_prenoise_alt2,

        "rk_res_2m_nonstandard_prenoise_alt": test_samplers.sample_rk_res_2m_nonstandard_prenoise_alt,

        "rk_res_2m_prenoise_alt": test_samplers.sample_rk_res_2m_prenoise_alt,
        "rk_res_2m_prenoise_alt2": test_samplers.sample_rk_res_2m_prenoise_alt2,

        "rk_res_2s_prenoise_alt": test_samplers.sample_rk_res_2s_prenoise_alt,
        "rk_res_2s_prenoise_data": test_samplers.sample_rk_res_2s_prenoise_data,

        "rk_res_2s_overstep": test_samplers.sample_rk_res_2s_overstep,
        "rk_res_2s_downswap": test_samplers.sample_rk_res_2s_downswap,

        
        "rk_crazy": test_samplers.sample_rk_crazy,
        "rk_crazy2": test_samplers.sample_rk_crazy2,
        "rk_crazymod43": test_samplers.sample_rk_crazymod43,
        "rk_crazymod44": test_samplers.sample_rk_crazymod44,
        "rk_crazymod45": test_samplers.sample_rk_crazymod45,
        "rk_pec423": test_samplers.sample_rk_pec423,
        "rk_pec433": test_samplers.sample_rk_pec433,
        "rk_gausslang_full": test_samplers.sample_rk_gausslang_full,
        "rk_gausslang_3s_full": test_samplers.sample_rk_gausslang_3s_full,
        "rk_gausslang_3s_full_guide": test_samplers.sample_rk_gausslang_3s_full_guide,
        "rk_radau_iia_alt_lang_3s_full": test_samplers.sample_rk_radau_iia_alt_lang_3s_full,
        "rk_implicit": test_samplers.sample_rk_implicit,


        "rk_gausslang": test_samplers.sample_rk_gausslang,
        "rk_gausslangeps": test_samplers.sample_rk_gausslangeps,
        "rk_ralradau": test_samplers.sample_rk_ralradau,


        "rk_gausscycle": test_samplers.sample_rk_gausscycle,
        "rk_gausscycle2": test_samplers.sample_rk_gausscycle2,

        "rk_radaucycle": test_samplers.sample_rk_radaucycle,
        "rk_radaucycle_ia": test_samplers.sample_rk_radaucycle_ia,
        "rk_radaucycle_3s": test_samplers.sample_rk_radaucycle_3s,
        "rk_radaucycle_retry": test_samplers.sample_rk_radaucycle_retry,
        "rk_radaucycle_staggered": test_samplers.sample_rk_radaucycle_staggered,
        
        "rk_implicit_euler_von_svd": test_samplers.sample_rk_implicit_euler_von_svd,
        "rk_implicit_cycloeuler": test_samplers.sample_rk_implicit_cycloeuler,
        "rk_salmon": test_samplers.sample_rk_salmon,


        "rk_radau_iia_2s_BS": test_samplers.sample_rk_radau_iia_2s_BS,

        "rk_radau_ia_2s_lang_full": test_samplers.sample_rk_radau_ia_2s_lang_full,

        "rk_radau_iia_2s_lang_full": test_samplers.sample_rk_radau_iia_2s_lang_full,

        "rk_radau_iia_2s": test_samplers.sample_rk_radau_iia_2s,
        "rk_radau_iia_3s": test_samplers.sample_rk_radau_iia_3s,


        "rk_abnorsett4": test_samplers.sample_rk_abnorsett4,
        "rk_euler_lowranksvd": test_samplers.sample_rk_euler_lowranksvd,
        "rk_randomized_svd": test_samplers.sample_rk_randomized_svd,
        "rk_implicit_euler_fd": test_samplers.sample_rk_implicit_euler_fd,

        "rk_euler_banana_alphaupdown_test": test_samplers.sample_rk_euler_banana_alphaupdown_test,
        "rk_euler_hamberder": test_samplers.sample_rk_euler_hamberder,

        "rk_euler_banana": test_samplers.sample_rk_euler_banana,
        "rk_res_2s_banana": test_samplers.sample_rk_res_2s_banana,
        "rk_res_3s_banana": test_samplers.sample_rk_res_3s_banana,

        "rk_euler": test_samplers.sample_rk_euler,

        "rk_euler_prenoise": test_samplers.sample_rk_euler_prenoise,


        "rk_ddim_test": test_samplers.sample_rk_ddim_test,
        "rk_res_denoise_eps": test_samplers.sample_rk_res_denoise_eps,
        "rk_exp_euler_denoise_eps": test_samplers.sample_rk_exp_euler_denoise_eps,
        "rk_euler_alt_sde": test_samplers.sample_rk_euler_alt_sde,

        "rk_implicit_euler":  test_samplers.sample_rk_implicit_euler,

        "rk_vpsde_trivial":  test_samplers.sample_rk_vpsde_trivial,
        "zample": test_samplers.sample_zsample,
        "zample_paper": test_samplers.sample_zample_paper,
        "zample_inversion": test_samplers.sample_zample_inversion,
        "sample_zample_edit": test_samplers.sample_zample_edit,
        
    })
    

if flags["beta_samplers"]:
    NODE_CLASS_MAPPINGS.update({
        "ClownSamplerAdvanced_Beta": samplers.ClownSamplerAdvanced_Beta,
    })
    extra_samplers.update({
        "rk_beta": rk_sampler_beta.sample_rk_beta,
    })

WEB_DIRECTORY = "./web/js"
__all__ = ["NODE_CLASS_MAPPINGS",  "WEB_DIRECTORY"]


add_samplers()



