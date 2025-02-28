from . import legacy_samplers
#from .legacy_sampler_rk import legacy_sample_rk
from . import legacy_sampler_rk

from . import rk_sampler
from . import samplers
from . import samplers_extensions
from . import samplers_tiled



def add_legacy(NODE_CLASS_MAPPINGS, extra_samplers):
    NODE_CLASS_MAPPINGS.update({
        "Legacy_ClownSampler": legacy_samplers.Legacy_SamplerRK,
        "Legacy_SharkSampler": legacy_samplers.Legacy_SharkSampler,
        "Legacy_ClownsharKSampler": legacy_samplers.Legacy_ClownsharKSampler,
        "Legacy_ClownsharKSamplerGuides": legacy_samplers.Legacy_ClownsharKSamplerGuides,
        
        "ClownSampler": samplers.ClownSampler,
        "ClownSamplerAdvanced": samplers.ClownSamplerAdvanced,
        "ClownsharKSampler": samplers.ClownsharKSampler,
        
        
        "ClownsharKSamplerGuides": samplers_extensions.ClownsharKSamplerGuides,
        "ClownsharKSamplerGuide": samplers_extensions.ClownsharKSamplerGuide,
    
        "ClownOptions_SDE_Noise": samplers_extensions.ClownOptions_SDE_Noise,
        "ClownOptions_FrameWeights": samplers_extensions.ClownOptions_FrameWeights,
    
        "ClownInpaint": samplers_extensions.ClownInpaint,
        "ClownInpaintSimple": samplers_extensions.ClownInpaintSimple,

        "ClownsharKSamplerOptions": samplers_extensions.ClownsharKSamplerOptions,

        "ClownsharKSamplerAutomation": samplers_extensions.ClownsharKSamplerAutomation,
        "ClownsharKSamplerAutomation_Advanced": samplers_extensions.ClownsharKSamplerAutomation_Advanced,
        "SamplerOptions_TimestepScaling": samplers_extensions.SamplerOptions_TimestepScaling,
        "SamplerOptions_GarbageCollection": samplers_extensions.SamplerOptions_GarbageCollection,

        "UltraSharkSampler": samplers.UltraSharkSampler,

        "UltraSharkSampler Tiled": samplers_tiled.UltraSharkSampler_Tiled,

    })

    extra_samplers.update({
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
    })
    
    return NODE_CLASS_MAPPINGS, extra_samplers

