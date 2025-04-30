import importlib
import os

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
    "TextBox2"                            : nodes_misc.TextBox2,
    "TextBox3"                            : nodes_misc.TextBox3,
    
    "TextConcatenate"                     : nodes_misc.TextConcatenate,
    "TextBoxConcatenate"                  : nodes_misc.TextBoxConcatenate,
    
    "TextLoadFile"                        : nodes_misc.TextLoadFile,
    "TextShuffle"                         : nodes_misc.TextShuffle,
    "TextShuffleAndTruncate"              : nodes_misc.TextShuffleAndTruncate,
    "TextTruncateTokens"                  : nodes_misc.TextTruncateTokens,

    "SeedGenerator"                       : nodes_misc.SeedGenerator,
    
    "ClownRegionalConditioning"           : conditioning.ClownRegionalConditioning,
    "ClownRegionalConditionings"          : conditioning.ClownRegionalConditionings,
    
    "ClownRegionalConditioning2"          : conditioning.ClownRegionalConditioning2,
    "ClownRegionalConditioning3"          : conditioning.ClownRegionalConditioning3,
    
    "ClownRegionalConditioning_AB"        : conditioning.ClownRegionalConditioning_AB,
    "ClownRegionalConditioning_ABC"       : conditioning.ClownRegionalConditioning_ABC,

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

    "ConditioningDownsample (T5)"         : conditioning.ConditioningDownsampleT5,

    "ConditioningToBase64"                : conditioning.ConditioningToBase64,
    "Base64ToConditioning"                : conditioning.Base64ToConditioning,
    
    "ConditioningBatch4"                  : conditioning.ConditioningBatch4,
    "ConditioningBatch8"                  : conditioning.ConditioningBatch8,
    
    "TemporalMaskGenerator"               : conditioning.TemporalMaskGenerator,
    "TemporalSplitAttnMask"               : conditioning.TemporalSplitAttnMask,
    "TemporalSplitAttnMask (Midframe)"    : conditioning.TemporalSplitAttnMask_Midframe,
    "TemporalCrossAttnMask"               : conditioning.TemporalCrossAttnMask,



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
    
    "Latent Get Channel Means"            : nodes_latents.latent_get_channel_means,
    
    "Latent Match Channelwise"            : nodes_latents.latent_channelwise_match,
    
    "Latent to RawX"                      : nodes_latents.latent_to_raw_x,
    "Latent Clear State Info"             : nodes_latents.latent_clear_state_info,
    "Latent Replace State Info"           : nodes_latents.latent_replace_state_info,
    "Latent Display State Info"           : nodes_latents.latent_display_state_info,
    "Latent Transfer State Info"          : nodes_latents.latent_transfer_state_info,
    "Latent to Cuda"                      : nodes_latents.latent_to_cuda,
    "Latent Batcher"                      : nodes_latents.latent_batch,
    "Latent Normalize Channels"           : nodes_latents.latent_normalize_channels,
    "Latent Channels From To"             : nodes_latents.latent_mean_channels_from_to,



    "LatentPhaseMagnitude"                : nodes_latents.LatentPhaseMagnitude,
    "LatentPhaseMagnitudeMultiply"        : nodes_latents.LatentPhaseMagnitudeMultiply,
    "LatentPhaseMagnitudeOffset"          : nodes_latents.LatentPhaseMagnitudeOffset,
    "LatentPhaseMagnitudePower"           : nodes_latents.LatentPhaseMagnitudePower,
    
    "MaskToggle"                          : nodes_latents.MaskToggle,
    "Frames Masks Uninterpolate"          : nodes_latents.Frames_Masks_Uninterpolate,
    "Frames Masks ZeroOut"                : nodes_latents.Frames_Masks_ZeroOut,
    "Frames Latent ReverseOrder"          : nodes_latents.Frames_Latent_ReverseOrder,

    
    "EmptyLatentImage64"                  : nodes_latents.EmptyLatentImage64,
    "EmptyLatentImageCustom"              : nodes_latents.EmptyLatentImageCustom,
    "StableCascade_StageC_VAEEncode_Exact": nodes_latents.StableCascade_StageC_VAEEncode_Exact,
    
    
    
    "PrepForUnsampling"                   : helper_sigma_preview_image_preproc.VAEEncodeAdvanced,
    "VAEEncodeAdvanced"                   : helper_sigma_preview_image_preproc.VAEEncodeAdvanced,
    
    "SigmasPreview"                       : helper_sigma_preview_image_preproc.SigmasPreview,
    "SigmasSchedulePreview"               : helper_sigma_preview_image_preproc.SigmasSchedulePreview,


    "TorchCompileModelFluxAdv"            : models.TorchCompileModelFluxAdvanced,
    "TorchCompileModelAura"               : models.TorchCompileModelAura,
    "TorchCompileModelSD35"               : models.TorchCompileModelSD35,
    "TorchCompileModels"                  : models.TorchCompileModels,
    "ClownpileModelWanVideo"              : models.ClownpileModelWanVideo,


    "ModelTimestepPatcher"                : models.ModelSamplingAdvanced,
    "ModelSamplingAdvanced"               : models.ModelSamplingAdvanced,
    "ModelSamplingAdvancedResolution"     : models.ModelSamplingAdvancedResolution,
    "FluxGuidanceDisable"                 : models.FluxGuidanceDisable,

    "ReWanPatcher"                        : models.ReWanPatcher,
    "ReFluxPatcher"                       : models.ReFluxPatcher,
    "ReSD35Patcher"                       : models.ReSD35Patcher,
    "ReAuraPatcher"                       : models.ReAuraPatcher,
    "ReHiDreamPatcher"                    : models.ReHiDreamPatcher,
    
    "ReWanPatcherAdvanced"                : models.ReWanPatcherAdvanced,
    "ReFluxPatcherAdvanced"               : models.ReFluxPatcherAdvanced,
    "ReSD35PatcherAdvanced"               : models.ReSD35PatcherAdvanced,
    "ReAuraPatcherAdvanced"               : models.ReAuraPatcherAdvanced,
    
    "ReHiDreamPatcherAdvanced"            : models.ReHiDreamPatcherAdvanced,
    
    "FluxOrthoCFGPatcher"                 : models.FluxOrthoCFGPatcher,

    
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

    "ClownScheduler"                      : sigmas.ClownScheduler, # for modulating parameters
    "Tan Scheduler"                       : sigmas.tan_scheduler,
    "Tan Scheduler 2"                     : sigmas.tan_scheduler_2stage,
    "Tan Scheduler 2 Simple"              : sigmas.tan_scheduler_2stage_simple,
    "Constant Scheduler"                  : sigmas.constant_scheduler,
    "Linear Quadratic Advanced"           : sigmas.linear_quadratic_advanced,
    
    "Image Get Color Swatches"            : images.Image_Get_Color_Swatches,
    "Masks From Color Swatches"           : images.Masks_From_Color_Swatches,
    "Masks From Colors"                   : images.Masks_From_Colors,
    
    "Masks Unpack 4"                      : images.Masks_Unpack4,
    "Masks Unpack 8"                      : images.Masks_Unpack8,
    "Masks Unpack 16"                     : images.Masks_Unpack16,

    
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
    
    "Frame Select"                        : images.Frame_Select,
    "Frames Slice"                        : images.Frames_Slice,
    "Frames Concat"                       : images.Frames_Concat,
    
    "Mask Sketch"                         : images.MaskSketch,

    "Frames Concat Masks"                 : nodes_latents.Frames_Concat_Masks,


    "Frame Select Latent"                 : nodes_latents.Frame_Select_Latent,
    "Frames Slice Latent"                 : nodes_latents.Frames_Slice_Latent,
    "Frames Concat Latent"                : nodes_latents.Frames_Concat_Latent,


    "Frame Select Latent Raw"                 : nodes_latents.Frame_Select_Latent_Raw,
    "Frames Slice Latent Raw"                 : nodes_latents.Frames_Slice_Latent_Raw,
    "Frames Concat Latent Raw"                : nodes_latents.Frames_Concat_Latent_Raw,



}



WEB_DIRECTORY = "./web/js"
__all__ = ["NODE_CLASS_MAPPINGS",  "WEB_DIRECTORY"]



flags = {
    "zampler"        : False,
    "beta_samplers"  : False,
    "legacy_samplers": False,
}


file_path = os.path.join(os.path.dirname(__file__), "zampler_test_code.txt")
if os.path.exists(file_path):
    try:
        from .zampler import add_zamplers
        NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
        flags["zampler"] = True
        RESplain("Importing zampler.")
    except ImportError:
        try:
            import importlib
            for module_name in ["RES4LYF.zampler", "res4lyf.zampler"]:
                try:
                    zampler_module = importlib.import_module(module_name)
                    add_zamplers = zampler_module.add_zamplers
                    NODE_CLASS_MAPPINGS, extra_samplers = add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers)
                    flags["zampler"] = True
                    RESplain(f"Importing zampler via {module_name}.")
                    break
                except ImportError:
                    continue
            else:
                raise ImportError("Zampler module not found in any path")
        except Exception as e:
            print(f"(RES4LYF) Failed to import zamplers: {e}")


try:
    from .legacy import add_legacy
    NODE_CLASS_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, extra_samplers)
    flags["legacy_samplers"] = True
    RESplain("Importing legacy samplers.")
except ImportError:
    try:
        import importlib
        for module_name in ["RES4LYF.legacy", "res4lyf.legacy"]:
            try:
                legacy_module = importlib.import_module(module_name)
                add_legacy = legacy_module.add_legacy
                NODE_CLASS_MAPPINGS, extra_samplers = add_legacy(NODE_CLASS_MAPPINGS, extra_samplers)
                flags["legacy_samplers"] = True
                RESplain(f"Importing legacy samplers via {module_name}.")
                break
            except ImportError:
                continue
        else:
            raise ImportError("Legacy module not found in any path")
    except Exception as e:
        print(f"(RES4LYF) Failed to import legacy samplers: {e}")


try:
    from .beta import add_beta
    NODE_CLASS_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, extra_samplers)
    flags["beta_samplers"] = True
    RESplain("Importing beta samplers.")
except ImportError:
    try:
        import importlib
        for module_name in ["RES4LYF.beta", "res4lyf.beta"]:
            try:
                beta_module = importlib.import_module(module_name)
                add_beta = beta_module.add_beta
                NODE_CLASS_MAPPINGS, extra_samplers = add_beta(NODE_CLASS_MAPPINGS, extra_samplers)
                flags["beta_samplers"] = True
                RESplain(f"Importing beta samplers via {module_name}.")
                break
            except ImportError:
                continue
        else:
            raise ImportError("Beta module not found in any path")
    except Exception as e:
        print(f"(RES4LYF) Failed to import beta samplers: {e}")





add_samplers()




