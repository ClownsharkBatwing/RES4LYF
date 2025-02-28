
from . import rk_sampler_beta
from . import samplers
from . import samplers_extensions


def add_beta(NODE_CLASS_MAPPINGS, extra_samplers):
    NODE_CLASS_MAPPINGS.update({
        "SharkSampler": samplers.SharkSampler,

        "ClownsharKSamplerSimple_Beta": samplers.ClownsharKSamplerSimple_Beta,
        
        "ClownSamplerAdvanced_Beta": samplers.ClownSamplerAdvanced_Beta,

        "ClownGuidesAB_Beta": samplers_extensions.ClownGuidesAB_Beta,
        "ClownGuides_Beta": samplers_extensions.ClownGuides_Beta,
        "ClownGuide_Beta": samplers_extensions.ClownGuide_Beta,
        
        "ClownSamplerSelector_Beta": samplers_extensions.ClownSamplerSelector_Beta,

        "ClownOptions_SDE_Beta": samplers_extensions.ClownOptions_SDE_Beta,
        "ClownOptions_StepSize_Beta": samplers_extensions.ClownOptions_StepSize_Beta,
        "ClownOptions_DetailBoost_Beta": samplers_extensions.ClownOptions_DetailBoost_Beta,
        "ClownOptions_ImplicitSteps_Beta": samplers_extensions.ClownOptions_ImplicitSteps_Beta,
        "ClownOptions_ExtraOptions_Beta": samplers_extensions.ClownOptions_ExtraOptions_Beta,
        "ClownOptions_Automation_Beta": samplers_extensions.ClownOptions_Automation_Beta,

        "SharkOptions_Beta": samplers_extensions.SharkOptions_Beta,
        
        "ClownsharKSamplerGuidesMisc_Beta": samplers_extensions.ClownsharKSamplerGuidesMisc_Beta,
        "ClownsharKSamplerGuideMisc_Beta": samplers_extensions.ClownsharKSamplerGuideMisc_Beta,
        
        "ClownGuidesFluxAdvanced_Beta": samplers_extensions.ClownGuidesFluxAdvanced_Beta,
        

        "ClownsharKSamplerAutomation_Beta": samplers_extensions.ClownsharKSamplerAutomation_Beta,
        
        "ClownOptions_Combine": samplers_extensions.ClownOptions_Combine,
    })

    extra_samplers.update({
        "rk_beta": rk_sampler_beta.sample_rk_beta,
    })
    
    return NODE_CLASS_MAPPINGS, extra_samplers


