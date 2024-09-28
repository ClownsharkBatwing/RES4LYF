import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc

from comfy.k_diffusion.sampling import get_ancestral_step, to_d


import functools

from .noise_classes import *
from .extra_samplers_helpers import get_deis_coeff_list

#from comfy_extras.nodes_advanced_samplers import sample_euler_cfgpp_alt

def get_ancestral_step_RF_var(sigma, sigma_next, eta):
    dtype = sigma.dtype #calculate variance adjusted sigma up... sigma_up = sqrt(dt)
    eps = 1e-10

    sigma, sigma_next = sigma.to(torch.float64), sigma_next.to(torch.float64)

    sigma_diff = (sigma - sigma_next).abs() + eps 
    sigma_up = torch.sqrt(sigma_diff).to(torch.float64) * eta

    #print(f"sigma_up dtype: {sigma_up.dtype}")

    sigma_down_num = (sigma_next**2 - sigma_up**2).to(torch.float64)
    sigma_down = torch.sqrt(sigma_down_num) / ((1 - sigma_next).to(torch.float64) + torch.sqrt(sigma_down_num).to(torch.float64))

    #print(f"sigma_down dtype: {sigma_down.dtype}")

    alpha_ratio = (1 - sigma_next).to(torch.float64) / (1 - sigma_down).to(torch.float64)
    #print(f"alpha_ratio dtype: {alpha_ratio.dtype}")
    return sigma_up.to(dtype), sigma_down.to(dtype), alpha_ratio.to(dtype)

def get_ancestral_step(sigma, sigma_next, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_next, 0.
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def get_ancestral_step2(sigma, sigma_next, eta=1.):
    #Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing an ancestral sampling step. Only works for eta = 1.0. Intellectual curiousity only.
    if not eta:
        return sigma_next, 0.
    sigma_down = sigma_next ** 2 / sigma
    sigma_up = sigma_next ** 2 - sigma_down ** 2
    #sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    #sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def get_ancestral_step_backwards(sigma, sigma_next, eta=1.):
    """Calculates sigma_down using the new backward-derived formula and preserves variance."""
    if not eta:
        return sigma_next, 0.
    
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
    sigma_up = torch.sqrt(sigma_next**2 - sigma_down**2)
    return sigma_down, sigma_up

"""def get_RF_step(sigma, sigma_next, eta, alpha_ratio=None):
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)   #maybe this is the most appropriate target for scaling?
    #sigma_down = sigma_next * down_ratio
    sigma_up = eta
    alpha_ratio = 1 - sigma_up ** 2 / sigma_next ** 2 
    sigma_down = sigma_next

    return sigma_down, sigma_up, alpha_ratio"""

def get_RF_step_traditional(sigma, sigma_next, eta, scale=0.0, alpha_ratio=None):
    # uses math similar to what is used for the get ancestral step code in comfyui. WORKS!
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)   #maybe this is the most appropriate target for scaling?
    #sigma_down = sigma_next * down_ratio
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_down, sigma_up, alpha_ratio

def get_RF_step_orig(sigma, sigma_next, eta, alpha_ratio=None):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition."""
    down_ratio = (1 - eta) + eta * (sigma_next / sigma)   #maybe this is the most appropriate target for scaling?
    sigma_down = sigma_next * down_ratio

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_down, sigma_up, alpha_ratio


def get_RF_step(sigma, sigma_next, eta, noise_scale=1.0, alpha_ratio=None):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition. Scale is to shape the sigma_down curve."""
    sigma_minus = sigma - sigma_next
    #down_ratio = (1 - eta) + eta * ((sigma - sigma_minus * noise_scale) / sigma)
    down_ratio = (1 - eta) + eta * ((sigma - sigma_minus) / sigma)
    #sigma_down = sigma_next * down_ratio ** noise_scale
    sigma_down = sigma_next ** noise_scale * down_ratio

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_down, sigma_up, alpha_ratio


def get_RF_step2(sigma_up, sigma_next):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition."""
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)
    device = sigma_next.device
    dtype = sigma_next.dtype
    #sqrt_term = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)  #THIS IS THE SAME AS SIGMA_DOWN FOR ANCESTRAL STEP
    sigma_next = sigma_next.to(torch.complex64)
    sigma_up = sigma_up.to(torch.complex64)
    sqrt_term = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)
    
    sigma_down = sqrt_term / ((1 - sigma_next) + sqrt_term)
    
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down)

    #return (sigma_down.real.to(device=device, dtype=dtype), alpha_ratio.real.to(device=device, dtype=dtype), )
    return (torch.abs(sigma_down).to(device=device, dtype=dtype), torch.abs(alpha_ratio).to(device=device, dtype=dtype), )

import torch

def get_scaled_ancestral_step_RF(sigma_next, eta, scale_exponent):
    #this shit be fucked
    eta_scaled = torch.sqrt(1 - eta**2)
    
    sigma_up = sigma_next * eta
    sigma_down = sigma_next * eta_scaled / (1-sigma_next + sigma_next * eta_scaled) 
    
    #alpha_ratio = 1-sigma_next + sigma_next * eta_scaled
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down)

    # a^x * b^y = c, solve for y
    # y = (ln(c) - x*ln(a)) / ln(b)
    c = sigma_next * eta_scaled
    a = alpha_ratio
    b = sigma_down
    
    if a != 0 and b != 0:
        y = (torch.log(c) - scale_exponent * torch.log(a)) / torch.log(b)
    else:
        y = 1 

    scaled_alpha_ratio = alpha_ratio ** scale_exponent
    scaled_sigma_down = sigma_down ** y

    return sigma_up, scaled_sigma_down, scaled_alpha_ratio

def get_sigma_down_RF(sigma_next, eta):
    eta_scale = torch.sqrt(1 - eta**2)
    sigma_down = (sigma_next * eta_scale) / (1 - sigma_next + sigma_next * eta_scale)
    return sigma_down

def get_sigma_up_RF(sigma_next, eta):
    return sigma_next * eta

def get_ancestral_step_RF(sigma_next, eta):
    sigma_up = sigma_next * eta
    eta_scaled = (1 - eta**2)**0.5
    sigma_down = (sigma_next * eta_scaled) / (1 - sigma_next + sigma_next * eta_scaled)
    alpha_ratio = (1 - sigma_next) + sigma_next * eta_scaled
    #alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    return sigma_up, sigma_down, alpha_ratio

"""def get_ancestral_step_RF_trash(sigma_next, eta):
    #Modified trash-like RF step function where sigma_down = sigma_next, and alpha_ratio is used to control variance preservation.
    sigma_up = eta * sigma_next
    sigma_down = sigma_next
    alpha_ratio = torch.sqrt((sigma_next**2 - sigma_up**2) / sigma_next**2)
    return sigma_down, sigma_up, alpha_ratio"""


"""def get_ancestral_step_RF(sigma_next, eta):
    sigma_up = sigma_next * eta
    eta_scaled = (1 - eta**2)**0.5
    sigma_down = (sigma_next * eta_scaled) / (1 - sigma_next + sigma_next * eta_scaled)
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    #alpha_ratio = 1 - sigma_next + sigma_next * (1 - eta**2) ** 0.5
    return sigma_up, sigma_down, alpha_ratio"""


def get_ancestral_step_RF2(sigma_next, sigma, eta):
    sigma_up = eta * sigma_next / sigma

    sigma_down_squared = (sigma_next ** 2) / (eta ** 2) - (sigma_next ** 2) / (sigma ** 2)
    sigma_down = torch.sqrt(sigma_down_squared)

    alpha_ratio = eta

    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF3(sigma_next, sigma, eta):
    sigma_up = eta * sigma_next
    sigma_down = sigma_next / (1 + eta * (sigma_next / sigma))
    alpha_ratio = 1 - eta**2 * (sigma_next / sigma)**2
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF4(sigma_next, eta):
    # Simplified alpha_ratio
    alpha_ratio = 1 - sigma_next/2
    
    # Sigma up is still proportional to sigma_next
    sigma_up = eta * sigma_next
    
    # Calculate sigma_down assuming alpha_ratio is preserved as 1 - sigma_next
    sigma_down = (sigma_next - sigma_up * alpha_ratio)
    
    return sigma_up, sigma_down, alpha_ratio

def get_RF_step3(sigma, sigma_next, eta, noise_mode="hard"):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition."""
    
    eta = eta * sigma_next / sigma #keep things safe to avoid the need for complex numbers
    sigma_up = sigma * eta
    device = sigma_next.device
    dtype = sigma_next.dtype
    sqrt_term = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)  #THIS IS THE SAME AS SIGMA_DOWN FOR ANCESTRAL STEP

    sqrt_term = torch.sqrt(sigma_next ** 2 - sigma_up ** 2)
    
    sigma_down = sqrt_term / ((1 - sigma_next) + sqrt_term)
    
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down)

    #return (sigma_down.real.to(device=device, dtype=dtype), alpha_ratio.real.to(device=device, dtype=dtype), )
    return (torch.abs(sigma_down).to(device=device, dtype=dtype), torch.abs(alpha_ratio).to(device=device, dtype=dtype), )


@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_sde_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    momentum=1.0, noise_sampler=None, r=1/2, noise_sampler_type="brownian", noise_mode="hard", noise_scale=1.0, k=1.0, scale=0.1, momentums=None, etas1=None, etas2=None, s_noises1=None, s_noises2=None, rs=None, alphas=None, denoise_boosts=None,
):
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_dpmpp_sde_advanced_RF(model, x, sigmas, extra_args, callback, disable, momentum, noise_sampler, r, noise_sampler_type, noise_mode, noise_scale, k, scale,
                                            momentums, etas1, etas2, s_noises1, s_noises2, rs, alphas, denoise_boosts)
    
    #DPM-Solver++ (stochastic with eta parameter).20
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel
    seed = extra_args.get("seed", None) + 1

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        x_hat = x
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        diff_2 = momentum_func(denoised, vel_2, sigmas[i], -momentums[i]/2.0)
        vel_2 = diff_2
        denoised = diff_2
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alphas[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), etas1[i])
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            #x_2 = (sigma_sd_s / sigmas[i]) * x + (1 - (sigma_sd_s / sigmas[i])) * denoised
            x_2 = x_2 + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(s)) * s_noises1[i] * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            
            diff = momentum_func(denoised_2, vel, sigmas[i], -momentums[i]/2.0)
            vel = diff
            denoised_2 = diff

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), etas2[i])
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d

            d = to_d(x_hat, sigmas[i], x)  #what the hell is this second euler's momma about?
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt

            x = x + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * s_noises2[i] * su


@torch.no_grad()
def sample_dpmpp_sde_advanced_RF(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    momentum=0.0, noise_sampler=None, r=1/2, noise_sampler_type="brownian", noise_mode="hard", noise_scale=1.0, k=1.0, scale=0.1, 
    momentums=None, etas1=None, etas2=None, s_noises1=None, s_noises2=None, rs=None, alphas=None,denoise_boosts=None,
):
    """DPM-Solver++ (stochastic with eta parameter) adapted for Rectified Flow."""
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel
    
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        get_down_step = get_RF_step
        sigma_fn = lambda t: (t.exp() + 1) ** -1
        t_fn = lambda sigma: ((1-sigma)/sigma).log()
    else:
        get_down_step = lambda sigma, sigma_next, eta: (*get_ancestral_step(sigma, sigma_next, eta), 1.0) # wrap function so it returns a third fixed value: alpha_ratio = 1.0 
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

    vel = None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        vel = denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i]/2.0)
        
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alphas[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
            
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * rs[i]
            fac = 1 / (2 * rs[i])
            
            sigma_s = sigma_fn(s)

            if sigma_s.isnan():
                sigma_s = 0.9999

            # Step 1
            #sd, su, alpha_ratio = get_down_step(sigmas[i], sigma_s, etas1[i])
            sd, su, alpha_ratio = None, None, None
            if noise_mode == "soft": 
                sd, su, alpha_ratio = get_down_step(sigmas[i], sigma_s, etas1[i], noise_scale)
            elif noise_mode == "softer":
                sd, su, alpha_ratio = get_RF_step_traditional(sigmas[i], sigma_s, etas1[i], noise_scale)
            elif noise_mode == "hard":
                su = sigmas[i] * etas1[i]
                if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                    su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], etas1[i])
                    #sd, alpha_ratio = get_RF_step2(su, sigmas[i+1])
                else: 
                    sd = sigmas[i+1] #this may not work well...
            elif noise_mode == "hard_var":
                su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], etas1[i])
                sigma_var = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
                if sigmas[i+1] > sigma_var:
                    su, sd, alpha_ratio = get_ancestral_step_RF_var(sigmas[i], sigmas[i+1], etas1[i])
                
                
            sigma_sd_s = denoise_boosts[i] * sd + (1 - denoise_boosts[i]) * sigma_s #interpolate between sd and sigma_s; default is to step down to sigma_s, sd is farther down
            
            x_2 = (sigma_sd_s / sigmas[i]) * x + (1 - (sigma_sd_s / sigmas[i])) * denoised
            x_2 = alpha_ratio * x_2 + noise_sampler(sigma=sigmas[i], sigma_next=sigma_s) * s_noises1[i] * su
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)

            # Step 2
            #sd, su, alpha_ratio = get_down_step(sigmas[i], sigmas[i+1], etas2[i])
            if noise_mode == "soft": 
                sd, su, alpha_ratio = get_down_step(sigmas[i], sigmas[i+1], etas2[i], noise_scale)
            elif noise_mode == "softer":
                sd, su, alpha_ratio = get_RF_step_traditional(sigmas[i], sigmas[i+1], etas2[i], noise_scale)
            elif noise_mode == "hard":
                su = sigmas[i] * etas2[i]
                if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                    su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], etas2[i])
                    #sd, alpha_ratio = get_RF_step2(su, sigmas[i+1])
                else: 
                    sd = sigmas[i+1] #this may not work well...
            elif noise_mode == "hard_var":
                su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], etas2[i])
                sigma_var = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
                if sigmas[i+1] > sigma_var:
                    su, sd, alpha_ratio = get_ancestral_step_RF_var(sigmas[i], sigmas[i+1], etas2[i])
                
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sd / sigmas[i]) * x + (1 - (sd / sigmas[i]))  * denoised_d
            x = alpha_ratio * x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i+1]) * s_noises2[i] * su
            
            del denoised, denoised_d, denoised_2, x_2
            gc.collect()
            torch.cuda.empty_cache()
    return x



@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_sde_cfgpp_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eulers_mom=None,
    eta=1., s_noise=1., noise_sampler=None, r=1/2, k=1.0, scale=0.1, noise_sampler_type="brownian", cfgpp: FloatTensor = torch.zeros((1,)), alpha: FloatTensor = torch.zeros((1,))
):
    #DPM-Solver++ (stochastic with eta parameter).
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args

    #import pdb; pdb.set_trace()
    if cfgpp.sum().item() != 0.0:
        temp = [0]
        def post_cfg_function(args):
            temp[0] = args["uncond_denoised"]
            return args["denoised"]

        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        x_hat = x
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

        if sigmas[i + 1] == 0:
            # Euler method
            if cfgpp.sum().item() == 0.0:
                d = to_d(x, sigmas[i], denoised)
            else:
                d = to_d(x - cfgpp[i]*denoised + cfgpp[i]*temp[0], sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            if cfgpp.sum().item() == 0.0:
                x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            else:
                x_2 = (sigma_fn(s_) / sigma_fn(t)) * (x + cfgpp[i]*(denoised - temp[0])) - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            if cfgpp.sum().item() == 0.0:
                x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            else:
                x = (sigma_fn(t_next_) / sigma_fn(t)) * (x + cfgpp[i]*(denoised - temp[0])) - (t - t_next_).expm1() * denoised_d

            if eulers_mom is not None:
                d = to_d(x_hat, sigmas[i], x)
                dt = sigmas[i + 1] - sigmas[i]
                x = x + eulers_mom[i].item() * d * dt

            x = x + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * s_noise * su
    return x

@precision_tool.cast_tensor
def sample_dpmpp_dualsdemomentum_advanced(model, x, sigmas, seed=42, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, r=1/2, momentum=0.0, momentums=None, etas=None, s_noises=None,rs=None,scheduled_r=False):
    return sample_dpmpp_dualsde_momentum_advanced(model, x, sigmas, seed=seed, extra_args=extra_args, callback=callback, disable=disable, eta=etas, s_noise=s_noises, noise_sampler_type=noise_sampler_type, noise_sampler=noise_sampler, r=rs, momentum=momentums, scheduled_r=False)

@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_dualsde_momentum_advanced (
    model, 
    x, 
    sigmas, 
    seed=42,
    extra_args=None, 
    callback=None, 
    disable=None,
    noise_sampler=None, 
    noise_sampler_type=None,
    momentum: FloatTensor = torch.zeros((1,)),
    eta: FloatTensor = torch.zeros((1,)),
    s_noise: FloatTensor = torch.zeros((1,)),
    r: FloatTensor = torch.zeros((1,)),
    scheduled_r=False
):
    """DPM-Solver++ (Stochastic with Momentum). Personal modified sampler by Clybius"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    denoisedsde_1, denoisedsde_2, denoisedsde_3 = None, None, None
    h_1, h_2, h_3 = None, None, None

    def momentum_func(diff, velocity, timescale=1.0, current_momentum=0): # Diff is current diff, vel is previous diff
        offset=-current_momentum / 2.0
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = current_momentum * (timescale + offset) * velocity + (1 - current_momentum * (timescale + offset)) * diff
        return momentum_vel

    vel = None
    vel_2 = None
    vel_sde = None
    sigma_down, sigma_up, alpha_ratio = None, None, 1.0
    
    sigma_fn_RF = lambda t: (t.exp() + 1) ** -1
    t_fn_RF = lambda sigma: ((1-sigma)/sigma).log()

    current_r = r[0]

    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max

        current_momentum = momentum[i]
        current_eta = eta[i]
        current_s_noise = s_noise[i]
        if scheduled_r == True:
            current_r = r[i]

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                t, t_next = t_fn_RF(sigmas[i]), t_fn_RF(sigmas[i + 1])
            else:
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            h_eta = h * (current_eta + 1)
            s = t + h * current_r
            fac = 1 / (2 * current_r)
            sigma_s = sigma_fn(s)
            if sigmas[i] == 1.0:
                sigma_s = 0.9999

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), current_eta)
            s_ = t_fn(sd)
            if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                sd, su, alpha_ratio = get_RF_step(sigma_fn_RF(t), sigma_fn_RF(s), current_eta)
                s_ = t_fn_RF(sd)
                t = t_fn_RF(sigmas[i])

            #diff_2 = momentum_func((t - s_).expm1() * denoised, vel_2, time, current_momentum)
            diff_2 = momentum_func((t_fn(sigmas[i]) - t_fn(sd)).expm1() * denoised, vel_2, time, current_momentum)
            vel_2 = diff_2
            #x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - diff_2
            x_2 = (sd / sigmas[i]) * x - diff_2
            
            if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                x_2 = x_2 * alpha_ratio + noise_sampler(sigma=sigma_fn_RF(t), sigma_next=sigma_fn_RF(s)) * current_s_noise * su
            else:
                x_2 = x_2 * alpha_ratio + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(s)) * current_s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), current_eta)
            t_next_ = t_fn(sd)
            if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                sd, su, alpha_ratio = get_RF_step(sigma_fn_RF(t), sigma_fn_RF(t_next), current_eta)
                t_next_ = t_fn_RF(sd)
                t = t_fn_RF(sigmas[i])

            denoised_d = (1 - fac) * denoised + fac * denoised_2
            #diff = momentum_func((t - t_next_).expm1() * denoised_d, vel, time, current_momentum)
            diff = momentum_func((t_fn(sigmas[i]) - t_fn(sd)).expm1() * denoised_d, vel, time, current_momentum)
            vel = diff
            #x = (sigma_fn(t_next_) / sigma_fn(t)) * x - diff
            x = (sd / sigmas[i]) * x - diff

            if h_3 is not None:
                r0 = h_3 / h_2
                r1 = h_2 / h
                r2 = h / h_1
                d1_0 = (denoised_d - denoisedsde_1) / r2
                d1_1 = (denoisedsde_1 - denoisedsde_2) / r1
                d1_2 = (denoisedsde_2 - denoisedsde_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time, current_momentum)
                vel_sde = diff
                x = x + diff
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised_d - denoisedsde_1) / r0
                d1_1 = (denoisedsde_1 - denoisedsde_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time, current_momentum)
                vel_sde = diff
                x = x + diff
            elif h_1 is not None:
                current_r = h_1 / h
                d = (denoised_d - denoisedsde_1) / current_r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = diff
                x = x + diff

            if current_eta:
                #x = x * alpha_ratio + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * current_s_noise * su
                if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                    x = x * alpha_ratio + noise_sampler(sigma=sigma_fn_RF(t), sigma_next=sigma_fn_RF(t_next)) * current_s_noise * su
                else:
                    x = x * alpha_ratio + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * current_s_noise * su
            #if 'denoised_d' in locals():
            denoisedsde_1, denoisedsde_2, denoisedsde_3 = denoised_d, denoisedsde_1, denoisedsde_2 
            #if 'h' in locals():
            h_1, h_2, h_3 = h, h_1, h_2
            
        gc.collect()
        torch.cuda.empty_cache()
    return x

# Many thanks to Kat + Birch-San for this wonderful sampler implementation! https://github.com/Birch-san/sdxl-play/commits/res/
from .refined_exp_solver import sample_refined_exp_s_advanced, sample_refined_exp_s_advanced_RF

@precision_tool.cast_tensor
def sample_res_solver_advanced(model, 
                               x, 
                               sigmas, etas1, etas2, s_noises1, s_noises2, c2s, momentums, eulers_moms, offsets, branch_mode, branch_depth, branch_width,
                               guides_1, guides_2, latent_guide_1, latent_guide_2, guide_mode_1, guide_mode_2, guide_1_channels,
                               k, clownseed=0, cfgpps=0.0, alphas=None, latent_noise=None, latent_self_guide_1=False,latent_shift_guide_1=False,
                               extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_mode="hard", noise_scale=1.0, ancestral_noise=False, alpha_ratios=None, noise_sampler=None, 
                               denoise_to_zero=True, simple_phi_calc=False, c2=0.5, momentum=0.0, eulers_mom=0.0, offset=0.0):
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_refined_exp_s_advanced_RF(
            model=model, 
            x=x, 
            clownseed=clownseed,
            sigmas=sigmas, 
            branch_mode=branch_mode,
            branch_depth=branch_depth,
            branch_width=branch_width,
            latent_guide_1=latent_guide_1,
            latent_guide_2=latent_guide_2,
            guide_1=guides_1,
            guide_2=guides_2,
            guide_mode_1=guide_mode_1,
            guide_mode_2=guide_mode_2,
            guide_1_channels=guide_1_channels,
            extra_args=extra_args, 
            callback=callback, 
            disable=disable, 
            noise_sampler=noise_sampler,
            noise_mode=noise_mode,
            noise_scale=noise_scale,
            ancestral_noise=ancestral_noise,
            alpha_ratios=alpha_ratios,
            denoise_to_zero=denoise_to_zero, 
            simple_phi_calc=simple_phi_calc, 
            cfgpp=cfgpps,
            c2=c2s, 
            etas1=etas1,
            etas2=etas2,
            s_noises1=s_noises1,
            s_noises2=s_noises2,
            momentum=momentums,
            eulers_mom=eulers_moms,
            offset=offsets,
            alpha=alphas,
            noise_sampler_type=noise_sampler_type,
            k=k,
            latent_noise=latent_noise,
            latent_self_guide_1=latent_self_guide_1,
            latent_shift_guide_1=latent_shift_guide_1
        )
    else:
        return sample_refined_exp_s_advanced(
            model=model, 
            x=x, 
            clownseed=clownseed,
            sigmas=sigmas, 
            branch_mode=branch_mode,
            branch_depth=branch_depth,
            branch_width=branch_width,
            latent_guide_1=latent_guide_1,
            latent_guide_2=latent_guide_2,
            guide_1=guides_1,
            guide_2=guides_2,
            guide_mode_1=guide_mode_1,
            guide_mode_2=guide_mode_2,
            guide_1_channels=guide_1_channels,
            extra_args=extra_args, 
            callback=callback, 
            disable=disable, 
            noise_sampler=noise_sampler,
            denoise_to_zero=denoise_to_zero, 
            simple_phi_calc=simple_phi_calc, 
            cfgpp=cfgpps,
            c2=c2s, 
            eta=etas1,
            s_noises=s_noises1,
            momentum=momentums,
            eulers_mom=eulers_moms,
            offset=offsets,
            alpha=alphas,
            noise_sampler_type=noise_sampler_type,
            k=k,
            latent_noise=latent_noise,
            latent_self_guide_1=latent_self_guide_1,
            latent_shift_guide_1=latent_shift_guide_1
        )


from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x - denoised + temp[0], sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic) with post-configuration."""
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x - denoised + temp[0], sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * (x + (denoised - temp[0])) - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * (x + (denoised - temp[0])) - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M) with post-configuration."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE with post-configuration."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None) + 1
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * (x + (denoised - temp[0])) + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x
@torch.no_grad()
def sample_dpmpp_3m_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE with post-configuration."""

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None) + 1
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * (x + (denoised - temp[0])) + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_euler_ancestral_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", noise_mode="soft", k=1.0, scale=0.1, alpha=None):
    """Ancestral sampling with Euler method steps."""
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        if noise_mode == "soft": 
            return sample_euler_ancestral_advanced_RF_soft(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, noise_sampler_type, k, scale, alpha)
        if noise_mode == "hard": 
            return sample_euler_ancestral_advanced_RF_hard(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, noise_sampler_type, k, scale, alpha)
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_euler_ancestral_advanced_RF_hard(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        #sigma_up = sigmas[i] * eta
        #sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])
        #sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta)
        if sigmas[i+1] > 0:
            #sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF_var(sigmas[i], sigmas[i+1], eta)
            sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta)
        else:
            sigma_up = 0.0
            sigma_down = sigmas[i+1]
            alpha_ratio = 1.0
            
        print(i, sigma_up, sigma_down, alpha_ratio, sigmas[i], sigmas[i+1])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x   +   noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
            
        gc.collect()
        torch.cuda.empty_cache()
    return x


@torch.no_grad()
def sample_euler_ancestral_advanced_RF_hard_old(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        #sigma_up = sigmas[i] * eta
        #sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])
        #sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta)
        if sigmas[i+1] > 0:
            sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta)
        else:
            sigma_up = 0.0
            sigma_down = sigmas[i+1]
            alpha_ratio = 1.0
            
        print(i, sigma_up, sigma_down, alpha_ratio, sigmas[i], sigmas[i+1])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x   +   noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
            
        gc.collect()
        torch.cuda.empty_cache()
    return x


@torch.no_grad()
def sample_euler_ancestral_advanced_RF_soft(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        sigma_down, sigma_up, alpha_ratio = get_RF_step(sigmas[i], sigmas[i+1], eta)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x   +   noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        gc.collect()
        torch.cuda.empty_cache()
            
    return x


@torch.no_grad()
def sample_euler_ancestral_advanced_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        sigma_up = sigmas[i] * eta
        sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x   +   noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
            
        gc.collect()
        torch.cuda.empty_cache()
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    extra_args = {} if extra_args is None else extra_args

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        gc.collect()
        torch.cuda.empty_cache()
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint', noise_sampler_type="brownian", ):
    """DPM-Solver++(2M) SDE."""
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_dpmpp_2m_sde_advanced_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, solver_type, noise_sampler_type)
 
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None) + 1
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None # step size

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    #       C2
                    x = x + 1.0 * ( (-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 *   (-h - eta_h).expm1().neg()                     * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_advanced_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint', noise_sampler_type="brownian", ):
    """DPM-Solver++(2M) SDE."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None) + 1
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None # step size

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        sigma_down, sigma_up, alpha_ratio = get_RF_step(sigma, sigma_next, eta)
        sigma_ratio = (sigma_down - sigma) / (sigma_next - sigma)
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        #denoised = denoised * sigma_ratio
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    #       C2
                    x = x + 1.0 * ( (-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 *   (-h - eta_h).expm1().neg()                     * (1 / r) * (denoised - old_denoised)

            if eta:
                #x = x * alpha_ratio + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise * sigma_up
                x = x * alpha_ratio + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        old_denoised = denoised
        h_last = h
        
        gc.collect()
        torch.cuda.empty_cache()
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_advanced_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="brownian", 
                                    momentums=None, etas=None, s_noises=None, k=1.0, scale=0.1, alpha=None,
                                    ):
    """DPM-Solver++(3M) SDE."""
    vel = None
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentums[0] / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentums[0] * (timescale + offset) * velocity + (1 - momentums[0] * (timescale + offset)) * diff
        return momentum_vel
    
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None) + 1
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        eta = etas[i]
        sigma_down, sigma_up, alpha_ratio = get_RF_step(sigmas[i], sigmas[i+1], eta)
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i] / 2.0)
        vel = denoised
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step0
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                #x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise
                x = x * alpha_ratio + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noises[i] * sigma_up

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
        
        gc.collect()
        torch.cuda.empty_cache()

    return x



@torch.no_grad()
def sample_dpmpp_3m_sde_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="brownian", 
                                    momentums=None, etas=None, s_noises=None, k=1.0, scale=0.1, alpha=None,
                                    ):
    """DPM-Solver++(3M) SDE."""
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_dpmpp_3m_sde_advanced_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, noise_sampler_type, momentums, etas, s_noises, k, scale, alpha)
    
    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None) + 1
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

def sample_rk4(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements the fourth-order Runge-Kutta method."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        dt = sigma_next - sigma
        
        # Runge-Kutta 4th order calculations
        k1 = to_d(x, sigma, model(x, sigma * s_in, **extra_args))
        k2 = to_d(x + 0.5 * dt * k1, sigma + 0.5 * dt, model(x + 0.5 * dt * k1, (sigma + 0.5 * dt) * s_in, **extra_args))
        k3 = to_d(x + 0.5 * dt * k2, sigma + 0.5 * dt, model(x + 0.5 * dt * k2, (sigma + 0.5 * dt) * s_in, **extra_args))
        k4 = to_d(x + dt * k3, sigma_next, model(x + dt * k3, sigma_next * s_in, **extra_args))
        
        dx = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        x = x + dx * dt
        
        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_next': sigma_next, 'dx': dx})
    return x

def sample_rk4_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements the fourth-order Runge-Kutta method."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    seed = torch.initial_seed()
    print("seed: ", seed)
    noise_sampler = NOISE_GENERATOR_CLASSES.get("brownian")(x=x, seed=seed, sigma_min=0.0, sigma_max=1.0)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        dt = sigma_next - sigma
        
        # Runge-Kutta 4th order calculations
        sd, su, alpha_ratio = get_RF_step(sigma, sigma_next, 1.0)
        sigma_ratio = (sd - sigma) / (sigma_next - sigma)
        
        x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * su
        
        k1 = to_d(x, sigma, model(x, sigma * s_in, **extra_args))
        k2 = to_d(x + 0.5 * dt * k1, sigma + 0.5 * dt, model(x + 0.5 * dt * k1, (sigma + 0.5 * dt) * s_in, **extra_args))
        k3 = to_d(x + 0.5 * dt * k2, sigma + 0.5 * dt, model(x + 0.5 * dt * k2, (sigma + 0.5 * dt) * s_in, **extra_args))
        k4 = to_d(x + dt * k3, sigma_next, model(x + dt * k3, sigma_next * s_in, **extra_args))
        
        dx = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        x = x + dx * dt * sigma_ratio
        
        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_next': sigma_next, 'dx': dx})
        gc.collect()
        torch.cuda.empty_cache()
    return x

@torch.no_grad()
def sample_euler_ancestral_recursive(model, x, sigmas, i=0, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if noise_sampler is None:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed", None) + 1
        noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    denoised = model(x, sigmas[i] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    d = to_d(x, sigmas[i], denoised)
    # Euler method
    dt = sigma_down - sigmas[i]
    x = x + d * dt
    if sigmas[i + 1] > 0:
        x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

    i += 1

    if i == len(sigmas) - 1:
        return x
    else:
        x = sample_euler_ancestral_recursive(model, x, sigmas, i=i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

    return x

from comfy.k_diffusion.sampling import deis
from .refined_exp_solver import _refined_exp_sosu_step_RF, _refined_exp_sosu_step_RF_hard, _refined_exp_sosu_step_RF_hard_deis

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
@torch.no_grad()
def sample_deis_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3, deis_mode='rhoab', step_type='res_a', denoised_type="1_2",
                    momentums=None, etas=None, s_noises=None, noise_sampler_type="gaussian", noise_mode="hard", noise_scale=0.0, k=1.0, scale=0.1, alpha=None,):

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    print("DEIS_SDE seed set to: ", seed)

    vel = None
    vel, vel_2, x_n, denoised, denoised2 = None, None, None, None, None
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentums[0] / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentums[0] * (timescale + offset) * velocity + (1 - momentums[0] * (timescale + offset)) * diff
        return momentum_vel
    
    x_next = x

    coeff_list = get_deis_coeff_list(sigmas, max_order, deis_mode=deis_mode)

    buffer_model = []
    if step_type == "res_a":
        sigmas = torch.cat((sigmas, torch.tensor([0.0]).to(sigmas.device).to(sigmas.dtype) ))
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        time = sigmas[i] / sigma_max
            
        sigma_up, sigma_down, alpha_ratio = None, None, None
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        if noise_mode == "soft": 
            sigma_down, sigma_up, alpha_ratio = get_RF_step(sigma, sigma_next, etas[i], noise_scale)
        elif noise_mode == "softer":
            sigma_down, sigma_up, alpha_ratio = get_RF_step_traditional(sigma, sigma_next, etas[i], noise_scale)
        elif noise_mode == "hard":
            sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF(sigma_next, etas[i])
        elif noise_mode == "hard_var":
            sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF(sigma_next, etas[i])
            sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
            if sigma_next > sigma_var:
                sigma_up, sigma_down, alpha_ratio = get_ancestral_step_RF_var(sigma, sigma_next, etas[i])
                  

            #sigma_up, sigma_down, alpha_ratio = get_scaled_ancestral_step_RF(sigma_next, etas[i], 1.1)
            #sigma_up = sigmas[i] * etas[i]
            #sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])
        
        sigma_ratio = (sigma_down - sigma) / (sigma_next - sigma) #sigma_minus_full / sigma_minus_orig... alternative to using sigma_down in the equations directly

        #print(sigma_up, sigma_down, alpha_ratio, sigma_ratio, alpha_ratio * sigma_ratio)
        #print(sigma_next ** 2 - sigma_up ** 2 - sigma_down ** 2 * alpha_ratio ** 2)

        x_cur = x_next

        if step_type == "simple": 
            denoised = model(x_cur, sigma * s_in, **extra_args)
            denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i] / 2.0)
        elif step_type == "res_a":
            x, denoised, denoised2, denoised1_2, vel, vel_2 = _refined_exp_sosu_step_RF_hard_deis(model, x_cur, sigma, sigma_next, sigmas[i+2], c2=0.5,eta=etas[i], noise_sampler=noise_sampler, s_noise=s_noises[i], noise_mode=noise_mode, ancestral_noise=True,
                                                                                extra_args=extra_args, pbar=None, simple_phi_calc=False,
                                                                                momentum = momentums[i], vel = vel, vel_2 = vel_2, time = time, eulers_mom = 0.0, cfgpp = 0.0
                                                                                )
            if denoised_type == "2":
                denoised = denoised2
            elif denoised_type == "1_2":
                denoised = denoised1_2
        #denoised = vel * (sigma/sigma_down)
        #denoised = -vel
        #vel = denoised
        
        elif step_type == "dpmpp_sde":
            if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                get_down_step = get_RF_step
                sigma_fn = lambda t: (t.exp() + 1) ** -1
                t_fn = lambda sigma: ((1-sigma)/sigma).log()
            else:
                get_down_step = lambda sigma, sigma_next, eta: (*get_ancestral_step(sigma, sigma_next, eta), 1.0) # wrap function so it returns a third fixed value: alpha_ratio = 1.0 
                sigma_fn = lambda t: t.neg().exp()
                t_fn = lambda sigma: sigma.log().neg()

            vel = None
            eta2 = etas[i]
            eta1 = eta2 / 2
            s_noise1 = s_noise2 = s_noises[i]
            denoise_boost = 0.0
            #for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            
            vel = denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i]/2.0)
                
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigmas[i + 1] == 0:
                # Euler method
                d = to_d(x, sigmas[i], denoised)
                dt = sigmas[i + 1] - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver++
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = t_next - t
                s = t + h * 0.5
                fac = 1 / (2 * 0.5)
                
                sigma_s = sigma_fn(s)

                if sigma_s.isnan():
                    sigma_s = 0.9999

                # Step 1
                #sd, su, alpha_ratio = get_down_step(sigmas[i], sigma_s, etas1[i])
                sd, su, alpha_ratio = None, None, None
                if noise_mode == "soft": 
                    sd, su, alpha_ratio = get_down_step(sigmas[i], sigma_s, eta1, noise_scale)
                elif noise_mode == "softer":
                    sd, su, alpha_ratio = get_RF_step_traditional(sigmas[i], sigma_s, eta1, noise_scale)
                elif noise_mode == "hard":
                    su = sigmas[i] * eta1
                    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                        su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta1)
                        #sd, alpha_ratio = get_RF_step2(su, sigmas[i+1])
                    else: 
                        sd = sigmas[i+1] #this may not work well...
                elif noise_mode == "hard_var":
                    su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta1)
                    sigma_var = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
                    if sigmas[i+1] > sigma_var:
                        su, sd, alpha_ratio = get_ancestral_step_RF_var(sigmas[i], sigmas[i+1], eta1)
                    
                    
                sigma_sd_s = denoise_boost * sd + (1 - denoise_boost) * sigma_s #interpolate between sd and sigma_s; default is to step down to sigma_s, sd is farther down
                
                x_2 = (sigma_sd_s / sigmas[i]) * x + (1 - (sigma_sd_s / sigmas[i])) * denoised
                x_2 = alpha_ratio * x_2 + noise_sampler(sigma=sigmas[i], sigma_next=sigma_s) * s_noise1 * su
                denoised_2 = model(x_2, sigma_s * s_in, **extra_args)

                # Step 2
                #sd, su, alpha_ratio = get_down_step(sigmas[i], sigmas[i+1], etas2[i])
                if noise_mode == "soft": 
                    sd, su, alpha_ratio = get_down_step(sigmas[i], sigmas[i+1], eta2, noise_scale)
                elif noise_mode == "softer":
                    sd, su, alpha_ratio = get_RF_step_traditional(sigmas[i], sigmas[i+1], eta2, noise_scale)
                elif noise_mode == "hard":
                    su = sigmas[i] * eta2
                    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                        su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta2)
                        #sd, alpha_ratio = get_RF_step2(su, sigmas[i+1])
                    else: 
                        sd = sigmas[i+1] #this may not work well...
                elif noise_mode == "hard_var":
                    su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta2)
                    sigma_var = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
                    if sigmas[i+1] > sigma_var:
                        su, sd, alpha_ratio = get_ancestral_step_RF_var(sigmas[i], sigmas[i+1], eta2)
                    
                denoised_d = (1 - fac) * denoised + fac * denoised_2
                x = (sd / sigmas[i]) * x + (1 - (sd / sigmas[i]))  * denoised_d
                x = alpha_ratio * x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i+1]) * s_noise2 * su
                
                #del denoised, denoised_d, denoised_2, x_2
                denoised = denoised_d
                gc.collect()
                torch.cuda.empty_cache()
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            
        d_cur = ((x_cur - denoised) / sigma) * sigma_ratio    ###### convert to alt sigma scaling "sigma space" so remaining code does not need modification
        #d_cur = (x / sigma) # * sigma_ratio
        #d_cur = (((sigma_down/sigma) * x - vel) / sigma) * sigma_ratio

        order = min(max_order, i+1)
        if sigma_next <= 0:
            order = 1

        if order == 1:          # First Euler step.
            dt = sigma_next - sigma  #from the euler ancestral RF sampler
            x_next = x_cur + dt * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]

        if max_order > 1:
            if len(buffer_model) == max_order - 1:
                for k in range(max_order - 2):
                    buffer_model[k] = buffer_model[k+1]
                buffer_model[-1] = d_cur.detach()
            else:
                buffer_model.append(d_cur.detach())
            
        if sigma_next > 0 and etas[i] > 0:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noises[i] * sigma_up   
            x_next = x_next * alpha_ratio   +   noise                                            
            
        gc.collect() #necessary after every step to minimize OOM errors with flux dev
        torch.cuda.empty_cache()
        
        if step_type == "res_a":
            if sigmas[i+2] == 0:
                return x_next

    return x_next


# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
    added = 0
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

# The following function adds the samplers during initialization, in __init__.py
def add_schedulers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    added = 0
    for scheduler in extra_schedulers: #getattr(self, "sample_{}".format(extra_samplers))
        if scheduler not in KSampler.SCHEDULERS:
            try:
                idx = KSampler.SCHEDULERS.index("ddim_uniform") # Last item in the samplers list
                KSampler.SCHEDULERS.insert(idx+1, scheduler) # Add our custom samplers
                setattr(k_diffusion_sampling, "get_sigmas_{}".format(scheduler), extra_schedulers[scheduler])
                added += 1
            except ValueError as err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

# Add any extra samplers to the following dictionary
#from .refined_exp_solver import sample_refined_exp_s, sample_refined_exp_s_advanced
extra_samplers = {
    "euler_ancestral_advanced": sample_euler_ancestral_advanced,
    #"euler_ancestral_advanced2": sample_euler_ancestral_advanced_RF2,

    "dpmpp_2s_ancestral_advanced": sample_dpmpp_2s_ancestral_advanced,
    "res_momentumized_advanced": sample_res_solver_advanced,
    "dpmpp_dualsde_momentumized_advanced": sample_dpmpp_dualsdemomentum_advanced,
    "dpmpp_sde_advanced": sample_dpmpp_sde_advanced,
    "dpmpp_2m_sde_advanced": sample_dpmpp_2m_sde_advanced,
    "dpmpp_3m_sde_advanced": sample_dpmpp_3m_sde_advanced,
    "dpmpp_sde_cfgpp_advanced": sample_dpmpp_sde_cfgpp_advanced,

    "dpmpp_2s_a_cfg++": sample_dpmpp_2s_ancestral_cfgpp,
    "dpmpp_2m_cfg++": sample_dpmpp_2m_cfgpp,
    "dpmpp_sde_cfg++": sample_dpmpp_sde_cfgpp,
    "dpmpp_2m_sde_cfg++": sample_dpmpp_2m_sde_cfgpp,
    "dpmpp_3m_sde_cfg++": sample_dpmpp_3m_sde_cfgpp,
    "deis_sde": sample_deis_sde,

    "rk4":  sample_rk4,
    "rk4_RF":  sample_rk4_RF,
}

discard_penultimate_sigma_samplers = set((
    "dpmpp_dualsde_momentumized",
    "clyb_4m_sde_momentumized"
))

def get_sigmas_simple_exponential(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    sigs = torch.FloatTensor(sigs)
    exp = torch.exp(torch.log(torch.linspace(1, 0, steps + 1)))
    return sigs * exp

extra_schedulers = {
    "simple_exponential": get_sigmas_simple_exponential
}
