
from . import rk_sampler_beta
from . import samplers
from . import samplers_extensions


def add_beta(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers):
    
    NODE_CLASS_MAPPINGS.update({
        #"SharkSampler"                    : samplers.SharkSampler,
        #"SharkSamplerAdvanced_Beta"       : samplers.SharkSampler, #SharkSamplerAdvanced_Beta,
        "SharkOptions_Beta"               : samplers_extensions.SharkOptions_Beta,
        "ClownOptions_SDE_Beta"           : samplers_extensions.ClownOptions_SDE_Beta,
        "ClownOptions_DetailBoost_Beta"   : samplers_extensions.ClownOptions_DetailBoost_Beta,
        "ClownGuide_Style_Beta"           : samplers_extensions.ClownGuide_Style_Beta,
        "ClownGuide_Beta"                 : samplers_extensions.ClownGuide_Beta,
        "ClownGuides_Beta"                : samplers_extensions.ClownGuides_Beta,
        "ClownGuidesAB_Beta"              : samplers_extensions.ClownGuidesAB_Beta,
        
        "SharkOptions_GuiderInput"        : samplers_extensions.SharkOptions_GuiderInput,
        "ClownOptions_ImplicitSteps_Beta" : samplers_extensions.ClownOptions_ImplicitSteps_Beta,
        "ClownOptions_Cycles_Beta"        : samplers_extensions.ClownOptions_Cycles_Beta,

        "SharkOptions_GuideCond_Beta"     : samplers_extensions.SharkOptions_GuideCond_Beta,
        "SharkOptions_GuideConds_Beta"    : samplers_extensions.SharkOptions_GuideConds_Beta,
        
        "ClownOptions_Tile_Beta"          : samplers_extensions.ClownOptions_Tile_Beta,
        "ClownOptions_Tile_Advanced_Beta" : samplers_extensions.ClownOptions_Tile_Advanced_Beta,


        "ClownGuide_Mean_Beta"            : samplers_extensions.ClownGuide_Mean_Beta,
        "ClownGuide_AdaIN_MMDiT_Beta"     : samplers_extensions.ClownGuide_AdaIN_MMDiT_Beta,
        "ClownGuide_AttnInj_MMDiT_Beta"   : samplers_extensions.ClownGuide_AttnInj_MMDiT_Beta,

        "ClownOptions_SDE_Mask_Beta"      : samplers_extensions.ClownOptions_SDE_Mask_Beta,
        
        "ClownOptions_StepSize_Beta"      : samplers_extensions.ClownOptions_StepSize_Beta,
        "ClownOptions_SigmaScaling_Beta"  : samplers_extensions.ClownOptions_SigmaScaling_Beta,

        "ClownOptions_Momentum_Beta"      : samplers_extensions.ClownOptions_Momentum_Beta,
        "ClownOptions_SwapSampler_Beta"   : samplers_extensions.ClownOptions_SwapSampler_Beta,
        "ClownOptions_ExtraOptions_Beta"  : samplers_extensions.ClownOptions_ExtraOptions_Beta,
        "ClownOptions_Automation_Beta"    : samplers_extensions.ClownOptions_Automation_Beta,

        "SharkOptions_UltraCascade_Latent_Beta"  : samplers_extensions.SharkOptions_UltraCascade_Latent_Beta,
        "SharkOptions_StartStep_Beta"     : samplers_extensions.SharkOptions_StartStep_Beta,
        
        "ClownOptions_Combine"            : samplers_extensions.ClownOptions_Combine,
        "ClownOptions_Frameweights"       : samplers_extensions.ClownOptions_Frameweights,
        

        "ClownSamplerSelector_Beta"       : samplers_extensions.ClownSamplerSelector_Beta,

        "SharkSampler_Beta"               : samplers.SharkSampler_Beta,
        
        "SharkChainsampler_Beta"          : samplers.SharkChainsampler_Beta,

        "ClownsharKSampler_Beta"          : samplers.ClownsharKSampler_Beta,
        "ClownsharkChainsampler_Beta"     : samplers.ClownsharkChainsampler_Beta,
        
        "ClownSampler_Beta"               : samplers.ClownSampler_Beta,
        "ClownSamplerAdvanced_Beta"       : samplers.ClownSamplerAdvanced_Beta,
        
        "BongSampler"                     : samplers.BongSampler,

    })

    extra_samplers.update({
        "res_2m"     : sample_res_2m,
        "res_3m"     : sample_res_3m,
        "res_2s"     : sample_res_2s,
        "res_3s"     : sample_res_3s,
        "res_5s"     : sample_res_5s,
        "res_6s"     : sample_res_6s,
        "res_2m_ode" : sample_res_2m_ode,
        "res_3m_ode" : sample_res_3m_ode,
        "res_2s_ode" : sample_res_2s_ode,
        "res_3s_ode" : sample_res_3s_ode,
        "res_5s_ode" : sample_res_5s_ode,
        "res_6s_ode" : sample_res_6s_ode,

        "deis_2m"    : sample_deis_2m,
        "deis_3m"    : sample_deis_3m,
        "deis_2m_ode": sample_deis_2m_ode,
        "deis_3m_ode": sample_deis_3m_ode,
        "rk_beta": rk_sampler_beta.sample_rk_beta,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
            #"SharkSampler"                          : "SharkSampler",
            #"SharkSamplerAdvanced_Beta"             : "SharkSamplerAdvanced",
            "SharkSampler_Beta"                     : "SharkSampler",
            "SharkChainsampler_Beta"                : "SharkChainsampler",
            "BongSampler"                           : "BongSampler",
            "ClownsharKSampler_Beta"                : "ClownsharKSampler",
            "ClownsharkChainsampler_Beta"           : "ClownsharkChainsampler",
            "ClownSampler_Beta"                     : "ClownSampler",
            "ClownSamplerAdvanced_Beta"             : "ClownSamplerAdvanced",
            "ClownGuide_Mean_Beta"                  : "ClownGuide Mean",
            "ClownGuide_AdaIN_MMDiT_Beta"           : "ClownGuide AdaIN (MMDiT)",
            "ClownGuide_AttnInj_MMDiT_Beta"         : "ClownGuide AttnInj (MMDiT)",
            "ClownGuide_Style_Beta"                 : "ClownGuide Style",
            "ClownGuide_Beta"                       : "ClownGuide",
            "ClownGuides_Beta"                      : "ClownGuides",
            "ClownGuidesAB_Beta"                    : "ClownGuidesAB",
            "ClownSamplerSelector_Beta"             : "ClownSamplerSelector",
            "ClownOptions_SDE_Mask_Beta"            : "ClownOptions SDE Mask",
            "ClownOptions_SDE_Beta"                 : "ClownOptions SDE",
            "ClownOptions_StepSize_Beta"            : "ClownOptions Step Size",
            "ClownOptions_DetailBoost_Beta"         : "ClownOptions Detail Boost",
            "ClownOptions_SigmaScaling_Beta"        : "ClownOptions Sigma Scaling",
            "ClownOptions_Momentum_Beta"            : "ClownOptions Momentum",
            "ClownOptions_ImplicitSteps_Beta"       : "ClownOptions Implicit Steps",
            "ClownOptions_Cycles_Beta"              : "ClownOptions Cycles",
            "ClownOptions_SwapSampler_Beta"         : "ClownOptions Swap Sampler",
            "ClownOptions_ExtraOptions_Beta"        : "ClownOptions Extra Options",
            "ClownOptions_Automation_Beta"          : "ClownOptions Automation",
            "SharkOptions_GuideCond_Beta"           : "SharkOptions Guide Cond",
            "SharkOptions_GuideConds_Beta"          : "SharkOptions Guide Conds",
            "SharkOptions_Beta"                     : "SharkOptions",
            "SharkOptions_StartStep_Beta"           : "SharkOptions Start Step",
            "SharkOptions_UltraCascade_Latent_Beta" : "SharkOptions UltraCascade Latent",
            "ClownOptions_Combine"                  : "ClownOptions Combine",
            "ClownOptions_Frameweights"             : "ClownOptions Frameweights",
            "SharkOptions_GuiderInput"              : "SharkOptions Guider Input",
            "ClownOptions_Tile_Beta"                : "ClownOptions Tile",
            "ClownOptions_Tile_Advanced_Beta"       : "ClownOptions Tile Advanced",

    })
    
    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, extra_samplers



def sample_res_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_2m",)
def sample_res_3m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_3m",)
def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_2s",)
def sample_res_3s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_3s",)
def sample_res_5s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_5s",)
def sample_res_6s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_6s",)

def sample_res_2m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_2m", eta=0.0, eta_substep=0.0, )
def sample_res_3m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_3m", eta=0.0, eta_substep=0.0, )
def sample_res_2s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_2s", eta=0.0, eta_substep=0.0, )
def sample_res_3s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_3s", eta=0.0, eta_substep=0.0, )
def sample_res_5s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_5s", eta=0.0, eta_substep=0.0, )
def sample_res_6s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="res_6s", eta=0.0, eta_substep=0.0, )

def sample_deis_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="deis_2m",)
def sample_deis_3m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="deis_3m",)

def sample_deis_2m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="deis_2m", eta=0.0, eta_substep=0.0, )
def sample_deis_3m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable, rk_type="deis_3m", eta=0.0, eta_substep=0.0, )

