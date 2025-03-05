import importlib

from . import loaders
from . import sigmas
from . import conditioning
from . import images
from . import models
from . import helper_sigma_preview_image_preproc
from . import nodes_misc

from . import nodes_latents
from . import nodes_precision

from .res4lyf import RESplain

#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

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

extra_samplers = {}

extra_samplers = dict(reversed(extra_samplers.items()))

NODE_CLASS_MAPPINGS = {

    "FluxLoader"                          : loaders.FluxLoader,
    "SD35Loader"                          : loaders.SD35Loader,
    
    
    
    "TextBox1"                            : nodes_misc.TextBox1,
    "TextBox3"                            : nodes_misc.TextBox3,
    
    "TextConcatenate"                     : nodes_misc.TextConcatenate,
    "TextBoxConcatenate"                  : nodes_misc.TextBoxConcatenate,
    
    "TextLoadFile"                        : nodes_misc.TextLoadFile,
    "TextShuffle"                         : nodes_misc.TextShuffle,
    "TextShuffleAndTruncate"              : nodes_misc.TextShuffleAndTruncate,
    "TextTruncateTokens"                  : nodes_misc.TextTruncateTokens,

    "SeedGenerator"                       : nodes_misc.SeedGenerator,
    
    "ClownRegionalConditioning"           : conditioning.ClownRegionalConditioning,
    
    "CLIPTextEncodeFluxUnguided"          : conditioning.CLIPTextEncodeFluxUnguided,
    "ConditioningOrthoCollin"             : conditioning.ConditioningOrthoCollin,

    "ConditioningAverageScheduler"        : conditioning.ConditioningAverageScheduler,
    "ConditioningMultiply"                : conditioning.ConditioningMultiply,
    "ConditioningAdd"                     : conditioning.ConditioningAdd,
    "Conditioning Recast FP64"            : conditioning.Conditioning_Recast64,
    "StableCascade_StageB_Conditioning64" : conditioning.StableCascade_StageB_Conditioning64,
    "ConditioningZeroAndTruncate"         : conditioning.ConditioningZeroAndTruncate,
    "ConditioningTruncate"                : conditioning.ConditioningTruncate,
    "StyleModelApplyAdvanced"             : conditioning.StyleModelApplyAdvanced,
    
    "RectifiedFlow_RegionalPrompt"        : conditioning.RectifiedFlow_RegionalPrompt,
    "RectifiedFlow_RegionalConditioning"  : conditioning.RectifiedFlow_RegionalConditioning,

    "ConditioningToBase64"                : conditioning.ConditioningToBase64,
    "Base64ToConditioning"                : conditioning.Base64ToConditioning,



    "Set Precision"                       : nodes_precision.set_precision,
    "Set Precision Universal"             : nodes_precision.set_precision_universal,
    "Set Precision Advanced"              : nodes_precision.set_precision_advanced,
    
    
    
    "LatentNoised"                        : nodes_latents.LatentNoised,
    "LatentNoiseList"                     : nodes_latents.LatentNoiseList,
    "AdvancedNoise"                       : nodes_latents.AdvancedNoise,

    "LatentNoiseBatch_perlin"             : nodes_latents.LatentNoiseBatch_perlin,
    "LatentNoiseBatch_fractal"            : nodes_latents.LatentNoiseBatch_fractal,
    "LatentNoiseBatch_gaussian"           : nodes_latents.LatentNoiseBatch_gaussian,
    "LatentNoiseBatch_gaussian_channels"  : nodes_latents.LatentNoiseBatch_gaussian_channels,
    
    "LatentBatch_channels"                : nodes_latents.LatentBatch_channels,
    "LatentBatch_channels_16"             : nodes_latents.LatentBatch_channels_16,
    
    "Latent Match Channelwise"            : nodes_latents.latent_channelwise_match,
    
    "Latent to RawX"                      : nodes_latents.latent_to_raw_x,
    "Latent to Cuda"                      : nodes_latents.latent_to_cuda,
    "Latent Batcher"                      : nodes_latents.latent_batch,
    "Latent Normalize Channels"           : nodes_latents.latent_normalize_channels,



    "LatentPhaseMagnitude"                : nodes_latents.LatentPhaseMagnitude,
    "LatentPhaseMagnitudeMultiply"        : nodes_latents.LatentPhaseMagnitudeMultiply,
    "LatentPhaseMagnitudeOffset"          : nodes_latents.LatentPhaseMagnitudeOffset,
    "LatentPhaseMagnitudePower"           : nodes_latents.LatentPhaseMagnitudePower,
    
    "MaskToggle"                          : nodes_latents.MaskToggle,
    
    
    
    "EmptyLatentImage64"                  : nodes_latents.EmptyLatentImage64,
    "EmptyLatentImageCustom"              : nodes_latents.EmptyLatentImageCustom,
    "StableCascade_StageC_VAEEncode_Exact": nodes_latents.StableCascade_StageC_VAEEncode_Exact,

    
    
    "PrepForUnsampling"                   : helper_sigma_preview_image_preproc.VAEEncodeAdvanced,
    "VAEEncodeAdvanced"                   : helper_sigma_preview_image_preproc.VAEEncodeAdvanced,
    
    "SigmasPreview"                       : helper_sigma_preview_image_preproc.SigmasPreview,
    "SigmasSchedulePreview"               : helper_sigma_preview_image_preproc.SigmasSchedulePreview,


    "ModelSamplingAdvanced"               : models.ModelSamplingAdvanced,
    "ModelTimestepPatcher"                : models.ModelSamplingAdvanced,
    "TorchCompileModelFluxAdv"            : models.TorchCompileModelFluxAdvanced,

    "ModelSamplingAdvancedResolution"     : models.ModelSamplingAdvancedResolution,
    "FluxGuidanceDisable"                 : models.FluxGuidanceDisable,

    "ReFluxPatcher"                       : models.ReFluxPatcher,
    "FluxOrthoCFGPatcher"                 : models.FluxOrthoCFGPatcher,
    "ReSD35Patcher"                       : models.ReSD35Patcher,
    "ReAuraPatcher"                       : models.ReAuraPatcher,
    
    
    "UNetSave"                            : models.UNetSave,



    "Sigmas Recast"                       : sigmas.set_precision_sigmas,
    "Sigmas Noise Inversion"              : sigmas.sigmas_noise_inversion,
    "Sigmas From Text"                    : sigmas.sigmas_from_text, 

    "Sigmas Variance Floor"               : sigmas.sigmas_variance_floor,
    "Sigmas Truncate"                     : sigmas.sigmas_truncate,
    "Sigmas Start"                        : sigmas.sigmas_start,
    "Sigmas Split"                        : sigmas.sigmas_split,
    "Sigmas Concat"                       : sigmas.sigmas_concatenate,
    "Sigmas Pad"                          : sigmas.sigmas_pad,
    "Sigmas Unpad"                        : sigmas.sigmas_unpad,
    
    "Sigmas SetFloor"                     : sigmas.sigmas_set_floor,
    "Sigmas DeleteBelowFloor"             : sigmas.sigmas_delete_below_floor,
    "Sigmas DeleteDuplicates"             : sigmas.sigmas_delete_consecutive_duplicates,
    "Sigmas Cleanup"                      : sigmas.sigmas_cleanup,
    
    "Sigmas Mult"                         : sigmas.sigmas_mult,
    "Sigmas Modulus"                      : sigmas.sigmas_modulus,
    "Sigmas Quotient"                     : sigmas.sigmas_quotient,
    "Sigmas Add"                          : sigmas.sigmas_add,
    "Sigmas Power"                        : sigmas.sigmas_power,
    "Sigmas Abs"                          : sigmas.sigmas_abs,
    
    "Sigmas2 Mult"                        : sigmas.sigmas2_mult,
    "Sigmas2 Add"                         : sigmas.sigmas2_add,
    
    "Sigmas Rescale"                      : sigmas.sigmas_rescale,

    "Sigmas Math1"                        : sigmas.sigmas_math1,
    "Sigmas Math3"                        : sigmas.sigmas_math3,

    "Sigmas Iteration Karras"             : sigmas.sigmas_iteration_karras,
    "Sigmas Iteration Polyexp"            : sigmas.sigmas_iteration_polyexp,

    "Tan Scheduler"                       : sigmas.tan_scheduler,
    "Tan Scheduler 2"                     : sigmas.tan_scheduler_2stage,
    "Tan Scheduler 2 Simple"              : sigmas.tan_scheduler_2stage_simple,
    "Constant Scheduler"                  : sigmas.constant_scheduler,
    "Linear Quadratic Advanced"           : sigmas.linear_quadratic_advanced,
    
    
    
    "Image Sharpen FS"                    : images.ImageSharpenFS,
    "Image Channels LAB"                  : images.Image_Channels_LAB,
    "Image Median Blur"                   : images.ImageMedianBlur,
    "Image Gaussian Blur"                 : images.ImageGaussianBlur,

    "Image Pair Split"                    : images.Image_Pair_Split,
    "Image Crop Location Exact"           : images.Image_Crop_Location_Exact,
    "Film Grain"                          : images.Film_Grain,
    "Frequency Separation Linear Light"   : images.Frequency_Separation_Linear_Light,
    "Frequency Separation Hard Light"     : images.Frequency_Separation_Hard_Light,
    "Frequency Separation Hard Light LAB" : images.Frequency_Separation_Hard_Light_LAB,
}



WEB_DIRECTORY = "./web/js"
__all__ = ["NODE_CLASS_MAPPINGS",  "WEB_DIRECTORY"]



flags = {
    "zampler"        : False,
    "beta_samplers"  : False,
    "legacy_samplers": False,
}



try:
    zampler_module = importlib.import_module("RES4LYF.zampler")
    from .zampler import add_zamplers
    NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
    flags["zampler"] = True
    RESplain("Importing zampler.")
except ImportError:
    RESplain("Failed to import zampler.", debug=True)
    #print(f"Failed to import zamplers: {e}")
    pass



try:
    legacy_module = importlib.import_module("RES4LYF.legacy")
    from .legacy import add_legacy
    NODE_CLASS_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, extra_samplers)
    flags["legacy_samplers"] = True
    RESplain("Importing legacy samplers.")
except Exception as e:
    #RESplain("Failed to import legacy samplers", debug=False)
    print(f"(RES4LYF) Failed to import legacy samplers: {e}")



try:
    beta_module = importlib.import_module("RES4LYF.beta")
    from .beta import add_beta
    NODE_CLASS_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, extra_samplers)
    flags["beta_samplers"] = True
    RESplain("Importing beta samplers.")
except Exception as e:
    #RESplain("Failed to import legacy samplers", debug=False)
    print(f"(RES4LYF) Failed to import beta samplers: {e}")



add_samplers()


