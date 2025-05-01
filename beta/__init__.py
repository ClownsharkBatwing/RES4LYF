
from . import rk_sampler_beta
from . import samplers
from . import samplers_extensions


def add_beta(NODE_CLASS_MAPPINGS, extra_samplers):
    
    NODE_CLASS_MAPPINGS.update({
        "SharkSampler"                    : samplers.SharkSampler,
        "SharkSamplerAdvanced_Beta"       : samplers.SharkSampler, #SharkSamplerAdvanced_Beta,

        "SharkSampler_Beta"               : samplers.SharkSampler_Beta,
        
        "SharkChainsampler_Beta"          : samplers.SharkChainsampler_Beta,

        "BongSampler"                     : samplers.BongSampler,

        "ClownsharKSampler_Beta"          : samplers.ClownsharKSampler_Beta,
        "ClownsharkChainsampler_Beta"      : samplers.ClownsharkChainsampler_Beta,
        
        
        "ClownSampler_Beta"               : samplers.ClownSampler_Beta,
        "ClownSamplerAdvanced_Beta"       : samplers.ClownSamplerAdvanced_Beta,

        "ClownGuide_Mean_Beta"            : samplers_extensions.ClownGuide_Mean_Beta,
        "ClownGuide_AdaIN_MMDiT_Beta"     : samplers_extensions.ClownGuide_AdaIN_MMDiT_Beta,
        "ClownGuide_AttnInj_MMDiT_Beta"   : samplers_extensions.ClownGuide_AttnInj_MMDiT_Beta,
        "ClownGuide_Style_Beta"           : samplers_extensions.ClownGuide_Style_Beta,

        "ClownGuide_Beta"                 : samplers_extensions.ClownGuide_Beta,
        "ClownGuides_Beta"                : samplers_extensions.ClownGuides_Beta,
        "ClownGuidesAB_Beta"              : samplers_extensions.ClownGuidesAB_Beta,
        
        "ClownGuide_Misc_Beta"            : samplers_extensions.ClownGuide_Misc_Beta,
        
        "ClownSamplerSelector_Beta"       : samplers_extensions.ClownSamplerSelector_Beta,

        "ClownOptions_SDE_Mask_Beta"      : samplers_extensions.ClownOptions_SDE_Mask_Beta,
        "ClownOptions_SDE_Beta"           : samplers_extensions.ClownOptions_SDE_Beta,
        
        "ClownOptions_StepSize_Beta"      : samplers_extensions.ClownOptions_StepSize_Beta,
        "ClownOptions_DetailBoost_Beta"   : samplers_extensions.ClownOptions_DetailBoost_Beta,
        "ClownOptions_SigmaScaling_Beta"  : samplers_extensions.ClownOptions_SigmaScaling_Beta,

        "ClownOptions_Momentum_Beta"      : samplers_extensions.ClownOptions_Momentum_Beta,
        "ClownOptions_ImplicitSteps_Beta" : samplers_extensions.ClownOptions_ImplicitSteps_Beta,
        "ClownOptions_SwapSampler_Beta"   : samplers_extensions.ClownOptions_SwapSampler_Beta,
        
        "ClownOptions_ExtraOptions_Beta"  : samplers_extensions.ClownOptions_ExtraOptions_Beta,
        "ClownOptions_Automation_Beta"    : samplers_extensions.ClownOptions_Automation_Beta,
        "ClownOptions_Combine"            : samplers_extensions.ClownOptions_Combine,
        "ClownOptions_Frameweights"       : samplers_extensions.ClownOptions_Frameweights,
        "SharkOptions_GuideCond_Beta"     : samplers_extensions.SharkOptions_GuideCond_Beta,
        "SharkOptions_GuideConds_Beta"    : samplers_extensions.SharkOptions_GuideConds_Beta,
        "ClownOptions_GuiderInput"        : samplers_extensions.ClownOptions_GuiderInput,


        "SharkOptions_Beta"               : samplers_extensions.SharkOptions_Beta,
        "SharkOptions_UltraCascade_Latent_Beta"  : samplers_extensions.SharkOptions_UltraCascade_Latent_Beta,
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
    
    return NODE_CLASS_MAPPINGS, extra_samplers



def sample_res_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m",)
def sample_res_3m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_3m",)
def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2s",)
def sample_res_3s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_3s",)
def sample_res_5s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_5s",)
def sample_res_6s(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_6s",)

def sample_res_2m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", eta=0.0, eta_substep=0.0, )
def sample_res_3m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_3m", eta=0.0, eta_substep=0.0, )
def sample_res_2s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_2s", eta=0.0, eta_substep=0.0, )
def sample_res_3s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_3s", eta=0.0, eta_substep=0.0, )
def sample_res_5s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_5s", eta=0.0, eta_substep=0.0, )
def sample_res_6s_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="res_6s", eta=0.0, eta_substep=0.0, )

def sample_deis_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_2m",)
def sample_deis_3m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_3m",)

def sample_deis_2m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_2m", eta=0.0, eta_substep=0.0, )
def sample_deis_3m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return sample_rk_beta(model, x, sigmas, extra_args, callback, disable, noise_sampler_type="gaussian", noise_mode="hard", noise_seed=-1, rk_type="deis_3m", eta=0.0, eta_substep=0.0, )

