
from . import rk_sampler_beta
from . import samplers
from . import samplers_extensions


def add_beta(NODE_CLASS_MAPPINGS, extra_samplers):
    
    NODE_CLASS_MAPPINGS.update({
        "SharkSampler"                    : samplers.SharkSampler,
        "SharkSamplerAdvanced_Beta"       : samplers.SharkSampler, #SharkSamplerAdvanced_Beta,

        "SharkSampler_Beta"               : samplers.SharkSampler_Beta,
        
        "SharkChainsampler_Beta"          : samplers.SharkChainsampler_Beta,
        #"SharkUnsampler_Beta"             : samplers.SharkUnsampler_Beta,


        "BongSampler"                     : samplers.BongSampler,

        "ClownsharKSampler_Beta"          : samplers.ClownsharKSampler_Beta,
        "ClownsharkChainsampler_Beta"      : samplers.ClownsharkChainsampler_Beta,
        #"ClownsharkUnsampler_Beta"        : samplers.ClownsharkUnsampler_Beta,

        
        
        "ClownSampler_Beta"               : samplers.ClownSampler_Beta,
        "ClownSamplerAdvanced_Beta"       : samplers.ClownSamplerAdvanced_Beta,

        "ClownGuide_Mean_Beta"            : samplers_extensions.ClownGuide_Mean_Beta,

        "ClownGuide_Beta"                 : samplers_extensions.ClownGuide_Beta,
        "ClownGuides_Beta"                : samplers_extensions.ClownGuides_Beta,
        "ClownGuidesAB_Beta"              : samplers_extensions.ClownGuidesAB_Beta,
        
        "ClownGuide_Misc_Beta"            : samplers_extensions.ClownGuide_Misc_Beta,
        
        "ClownSamplerSelector_Beta"       : samplers_extensions.ClownSamplerSelector_Beta,

        "ClownOptions_SDE_Mask_Beta"      : samplers_extensions.ClownOptions_SDE_Mask_Beta,
        "ClownOptions_SDE_Beta"           : samplers_extensions.ClownOptions_SDE_Beta,
        
        "ClownOptions_StepSize_Beta"      : samplers_extensions.ClownOptions_StepSize_Beta,
        "ClownOptions_DetailBoost_Beta"   : samplers_extensions.ClownOptions_DetailBoost_Beta,
        "ClownOptions_Momentum_Beta"      : samplers_extensions.ClownOptions_Momentum_Beta,
        "ClownOptions_ImplicitSteps_Beta" : samplers_extensions.ClownOptions_ImplicitSteps_Beta,
        "ClownOptions_SwapSampler_Beta"   : samplers_extensions.ClownOptions_SwapSampler_Beta,
        #"ClownOptions_StepsToRun_Beta"    : samplers_extensions.ClownOptions_StepsToRun_Beta,
        
        "ClownOptions_ExtraOptions_Beta"  : samplers_extensions.ClownOptions_ExtraOptions_Beta,
        "ClownOptions_Automation_Beta"    : samplers_extensions.ClownOptions_Automation_Beta,
        "ClownOptions_Combine"            : samplers_extensions.ClownOptions_Combine,
        "ClownOptions_Frameweights"       : samplers_extensions.ClownOptions_Frameweights,
        "SharkOptions_GuideCond_Beta"     : samplers_extensions.SharkOptions_GuideCond_Beta,


        "SharkOptions_Beta"               : samplers_extensions.SharkOptions_Beta,
        #"SharkOptions_UltraCascade_Beta"  : samplers_extensions.SharkOptions_UltraCascade_Beta,
        "SharkOptions_UltraCascade_Latent_Beta"  : samplers_extensions.SharkOptions_UltraCascade_Latent_Beta,
    })

    extra_samplers.update({
        "rk_beta": rk_sampler_beta.sample_rk_beta,
    })
    
    return NODE_CLASS_MAPPINGS, extra_samplers


