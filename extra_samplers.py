import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc
import kornia

from comfy.k_diffusion.sampling import get_ancestral_step, to_d

import torch.nn.functional as F
import torchvision.transforms as T


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
    return sigma_up, sigma_down

def get_ancestral_step_backwards(sigma, sigma_next, eta=1.):
    """Calculates sigma_down using the new backward-derived formula and preserves variance."""
    if not eta:
        return sigma_next, 0.
    
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
    sigma_up = torch.sqrt(sigma_next**2 - sigma_down**2)
    return sigma_up, sigma_down

"""def get_RF_step(sigma, sigma_next, eta, alpha_ratio=None):
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)   #maybe this is the most appropriate target for scaling?
    #sigma_down = sigma_next * down_ratio
    sigma_up = eta
    alpha_ratio = 1 - sigma_up ** 2 / sigma_next ** 2 
    sigma_down = sigma_next

    return sigma_down, sigma_up, alpha_ratio"""

def get_RF_step_traditional(sigma, sigma_next, eta, scale=0.0, alpha_ratio=None):
    # uses math similar to what is used for the get ancestral step code in comfyui. WORKS!
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_up, sigma_down, alpha_ratio

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


import torch

def get_sigma_down_RF(sigma_next, eta):
    eta_scale = torch.sqrt(1 - eta**2)
    sigma_down = (sigma_next * eta_scale) / (1 - sigma_next + sigma_next * eta_scale)
    return sigma_down

def get_sigma_up_RF(sigma_next, eta):
    return sigma_next * eta

"""def get_ancestral_step_RF_maybe_works_with_variance_exploding(sigma_next, eta):
    sigma_up = sigma_next * eta
    eta_scaled = (1 - eta**2)**0.5
    #sigma_down = (sigma_next * eta_scaled) / (1 - sigma_next + sigma_next * eta_scaled)
    #alpha_ratio = (1 - sigma_next) + sigma_next * eta_scaled
    #alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    sigma_down = (sigma_next * eta_scaled)
    alpha_ratio = 1.0
    return sigma_up, sigma_down, alpha_ratio"""


def get_ancestral_step_RF_correct_working(sigma_next, eta):
    sigma_up = sigma_next * eta
    eta_scaled = (1 - eta**2)**0.5
    sigma_down = (sigma_next * eta_scaled) / (1 - sigma_next + sigma_next * eta_scaled)
    alpha_ratio =                             1 - sigma_next + sigma_next * eta_scaled
    #alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    return sigma_up, sigma_down, alpha_ratio



def get_ancestral_step_RF(sigma_next, eta, sigma_max=1.0):
    sigma_up = sigma_next * eta #or whatever the f
    
    sigma_signal = sigma_max - sigma_next
    sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)

    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    
    return sigma_up, sigma_down, alpha_ratio





@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_sde_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None, noisy_cfg=False,
    momentum=1.0, noise_sampler=None, r=1/2, noise_sampler_type="brownian", noise_mode="hard", noise_scale=1.0, k=1.0, scale=0.1, momentums=None, etas1=None, etas2=None, s_noises1=None, s_noises2=None, 
    rs=None, auto_r=False, alphas=None, denoise_boosts=None, t_fn_formula=None, sigma_fn_formula=None, order=2,
):
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_dpmpp_sde_advanced_RF(model, x, sigmas, extra_args, callback, disable, momentum, noise_sampler, r, noise_sampler_type, noise_mode, noise_scale, k, scale,
                                            momentums, etas1, etas2, s_noises1, s_noises2, rs, auto_r, alphas, denoise_boosts, noisy_cfg, order, t_fn_formula, sigma_fn_formula)
    
    #DPM-Solver++ (stochastic with eta parameter).20
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel
    
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))
    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)
        
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





def get_res4lyf_step(sigma, sigma_next, eta=0.0, eta_var=1.0, noise_mode="hard"):
  sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
  if eta_var > 0.0 and sigma_next > sigma_var:
    su, sd, alpha_ratio = get_ancestral_step_RF_var(sigma, sigma_next, eta_var)
  else:
    if   noise_mode == "soft":
      su, sd, alpha_ratio = get_RF_step(sigma, sigma_next, eta)
    elif noise_mode == "softer":
      su, sd, alpha_ratio = get_RF_step_traditional(sigma, sigma_next, eta)
    elif noise_mode == "hard":
      su, sd, alpha_ratio = get_ancestral_step_RF(sigma_next, eta)
    elif noise_mode == "e**(-2*eta*h)":
      su, sd, alpha_ratio = get_ancestral_step_RF(sigma_next, eta)
  return su, sd, alpha_ratio


def get_res4lyf_half_step(sigma, sigma_next, c2=0.5, auto_c2=False, h_last=None, t_fn_formula="", sigma_fn_formula="", ):
  sigma_fn = lambda t: t.neg().exp()
  t_fn_x     = t_fn     = lambda sigma: sigma.log().neg()
  
  sigma_fn_x = eval(f"lambda t: {sigma_fn_formula}", {"t": None}) if sigma_fn_formula else sigma_fn
  t_fn_x = eval(f"lambda sigma: {t_fn_formula}", {"sigma": None}) if t_fn_formula else t_fn
      
  t, t_next = t_fn_x(sigma), t_fn_x(sigma_next)
  h = t_next - t
  if h_last is not None and auto_c2 == True:
    c2 = h_last / h 
  s = t + h * c2
  sigma_s = sigma_fn_x(s)

  h = (t_fn(sigma_s) - t_fn(sigma)) / c2 # h = (s - t) / c2    #remapped timestep-space
    
  print("sigma:", sigma.item(), "sigma_s:", sigma_s.item(), "sigma_next:", sigma_next.item(),)
  print("t:", t.item(), "s:", s.item(), "t_next:", t_next.item(), "h:", h.item(), "c2:", c2.item())
  
  return sigma_s, h, c2



@torch.no_grad()
def _step_RF_sde(model, x, sigma, sigma_next, c2 = 0.5, eta1=0.25, eta2=0.5, eta_var1=0.0, eta_var2=0.0, noise_sampler=None, noise_mode="hard", order=2, 
                                   s_noise1=1.0, s_noise2=1.0, denoised1_2=None, h_last=None, auto_c2=False, denoise_boost=0.0, extra_args=None,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  t_fn_formula="", sigma_fn_formula="", 
):
  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
  sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, )
  fac = 1 / (2 * c2)
  
  s_in = x.new_ones([x.shape[0]])
  
  if order == 1 or h_last is None:
    denoised = model(x, sigma * s_in, **extra_args)
  else:
    denoised = denoised1_2
    
  diff_2 = vel_2 = momentum_func(denoised, vel_2, time)
    
  sigma_sd_s = denoise_boost * sd + (1 - denoise_boost) * sigma_s #interpolate between sd and sigma_s; default is to step down to sigma_s, sd is farther down
            
  x_2 = (sigma_sd_s / sigma) * x + (1 - (sigma_sd_s / sigma)) * diff_2 #denoised

  if sigma_next > 0.00001:
    su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1, eta_var1, noise_mode)
    noise = noise_sampler(sigma=sigma, sigma_next=sigma_s)
    noise = (noise - noise.mean()) / noise.std()
    x_2 = alpha_ratio_2 * x_2 + noise * s_noise2 * su_2
    #x_2 = alpha_ratio_2 * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * s_noise2 * su_2
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)
  else: 
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)   #last step!
    
  diff = vel = momentum_func(denoised2, vel, time)

  denoised1_2 = (1 - fac) * denoised + fac * diff #denoised2
  x = (sd / sigma) * x + (1 - (sd / sigma))  * denoised1_2
  
  noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
  noise = (noise - noise.mean()) / noise.std()
  x = alpha_ratio * x + noise * s_noise1 * su
  #x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise1 * su

  return x, denoised, denoised2, denoised1_2, vel, vel_2, h



@torch.no_grad()
def sample_dpmpp_sde_advanced_RF(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    momentum=0.0, noise_sampler=None, r=1/2, noise_sampler_type="brownian", noise_mode="hard", noise_scale=1.0, k=1.0, scale=0.1, 
    momentums=None, etas1=None, etas2=None, s_noises1=None, s_noises2=None, rs=None, auto_r=False, alphas=None,denoise_boosts=None, noisy_cfg=False, order=2, t_fn_formula=None, sigma_fn_formula=None, 
):
    """DPM-Solver++ (stochastic with eta parameter) adapted for Rectified Flow."""   ###^^^swapped sigma_fn and t_fn by accident!
    
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))
    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)
        
    seed = extra_args.get("seed", None) + 1

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args

    vel, vel_2 = None, None
    denoised1_2=None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=disable):

        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        time = sigmas[i] / sigma_max
        
        sigma_next = torch.tensor(0.00001) if sigma_next == 0.0 else sigma_next
        
        x, denoised, denoised2, denoised1_2, vel, vel_2, h_last = _step_RF_sde(model, x, sigma=sigma, sigma_next= sigma_next, c2=rs[i], eta1=etas1[i], eta2=etas2[i], eta_var1= torch.tensor(0.0), eta_var2=torch.tensor(0.0), noise_sampler=noise_sampler, noise_mode=noise_mode, order=order, 
                                                                          s_noise1=s_noises1[i], s_noise2=s_noises2[i], denoised1_2=denoised1_2, h_last=h_last, auto_c2=auto_r, denoise_boost=denoise_boosts[i],  extra_args=extra_args, time=time,
                                                                          vel = vel, vel_2=vel_2,
                                                                          )
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1_2})

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
                               sigmas, etas1, etas2, eta_vars1, eta_vars2, s_noises1, s_noises2, c2s, c3s, momentums, eulers_moms, offsets, branch_mode, branch_depth, branch_width,
                               guides_1, guides_2, latent_guide_1, latent_guide_2, guide_mode_1, guide_mode_2, guide_1_channels,
                               k, clownseed=0, cfgpps=0.0, alphas=None, latent_noise=None, latent_self_guide_1=False,latent_shift_guide_1=False,
                               extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_mode="hard", noise_scale=1.0, ancestral_noise=False, noisy_cfg=False, alpha_ratios=None, noise_sampler=None, 
                               denoise_to_zero=True, simple_phi_calc=False, c2=0.5, momentum=0.0, eulers_mom=0.0, offset=0.0, t_fn_formula=None, sigma_fn_formula=None, skip_corrector=False,corrector_is_predictor=False,
                               step_type="res_a", order="2b", auto_c2=False,
                               ):
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
            noisy_cfg=noisy_cfg,
            alpha_ratios=alpha_ratios,
            denoise_to_zero=denoise_to_zero, 
            simple_phi_calc=simple_phi_calc, 
            cfgpp=cfgpps,
            c2=c2s, 
            c3=c3s,
            etas1=etas1,
            etas2=etas2,
            eta_vars1=eta_vars1,
            eta_vars2=eta_vars2,
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
            latent_shift_guide_1=latent_shift_guide_1,
            t_fn_formula=t_fn_formula,
            sigma_fn_formula=sigma_fn_formula,
            skip_corrector=skip_corrector,
            corrector_is_predictor=corrector_is_predictor,
            step_type=step_type, 
            order=order,
            auto_c2=auto_c2,
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
def sample_euler_implicit_advanced_RF(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., 
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", k=1.0, scale=0.1, 
    alpha=None, iter=3, tol=1e-5):
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
        
        x_next = x.clone()

        for _ in range(iter): 
            sigma_up, sigma_down, alpha_ratio = (0.0, sigmas[i + 1], 1.0) if sigmas[i + 1] <= 0 else get_ancestral_step_RF(sigmas[i + 1], eta)
            
            denoised_next = model(x_next, sigmas[i] * s_in, **extra_args)
            #denoised_next = model(x_next, sigma_down * s_in, **extra_args)

            d_next = to_d(x_next, sigmas[i], denoised_next)

            dt = sigma_down - sigmas[i]
            x_next = x + d_next * dt

        x = x_next
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_next})

        gc.collect()
        torch.cuda.empty_cache()

    return x




from .refined_exp_solver import _de_second_order, get_res4lyf_half_step, get_res4lyf_step, _refined_exp_sosu_step_RF2

@torch.no_grad()
def sample_RES_implicit_advanced_RF(
    model, x, sigmas, extra_args=None, callback=None, disable=None, c2=0.5, eta=1., s_noise=1., 
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", k=1.0, scale=0.1, 
    alpha=None, iter=3, tol=1e-5):
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    denoised_next = None
    vel, vel_2, denoised, denoised2, denoised1_2 = None, None, None, None, None
    
    #t_fn = lambda sigma: 1/((sigma).exp()+1)
    #sigma_fn = lambda t: ((1-t)/t).log()
    
    for i in trange(len(sigmas) - 2, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t    
    
        su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, torch.tensor(0.0).to(x.dtype).to(x.device), torch.tensor(0.0).to(x.dtype).to(x.device), noise_mode)
        sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, False, None, "", "", remap_t_to_exp_space=True)
        a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=False)

        denoised = model(x, sigma * s_in, **extra_args)
        x_2 = ((sd/sigma)**c2)*x + h*a2_1*denoised
        denoised2 = model(x_2, sigma_s * s_in, **extra_args)
        x_next = (sd/sigma)*x + h*(b1*denoised + b2*denoised2)
        denoised_next = (b1*denoised + b2*denoised2) / (b1 + b2)

        for iteration in range(iter):  

            time = sigmas[i] / sigma_max
            
            
            x_next_next, denoised, denoised2, denoised1_2, vel, vel_2, h_step = _refined_exp_sosu_step_RF2(model, x_next, sigma_next, sigmas[i+2], eta1=0.0, eta2=0.0, eta_var1=0.0, eta_var2=0.0, noise_sampler=noise_sampler, noise_mode="hard", order="1", extra_args=extra_args, 
                                                                                                     vel = vel, vel_2 = vel_2, time = time,)
            
            denoised_next = denoised1_2

            x_new = (sd/sigma)*x + h*denoised1_2*(b1 + b2)
            
            # x = (x_new - h*denoised1_2*(b1 + b2)) / (sd/sigma)   # projection back to x
            
            #x_new = (sigma_next/sigma) * x + (1 - sigma_next/sigma) * denoised_next
            error = torch.norm(x_new - x_next)
            #error = torch.norm(denoised_next - denoised_prev)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_new
                break

            x_next = x_new
            denoised_next = (b1*denoised + b2*denoised2) / (b1 + b2)
            
            #x  = (x_new - (1 - sigma_next/sigma) * denoised_next) / (sigma_next/sigma) # projection back to x with alternative math

        x = x_next
        
        #if sigmas[i + 1] > 0 and eta > 0:
        #    x = alpha_ratio * x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        denoised_next = denoised if denoised_next is None else denoised_next
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_next})

        gc.collect()
        torch.cuda.empty_cache()

    return x


import torch.fft

def frequency_separation_noise_fft(noise, alpha=0.5, cutoff_radius=30):
    """
    Separates the noise into low and high-frequency components using a Fourier transform on PyTorch tensors,
    applies a single weighting factor between the low and high frequencies, and returns the adjusted noise.

    Parameters:
    - noise: The input noise tensor to be separated (shape [batch, channels, height, width] or [height, width]).
    - alpha: A scalar weight to apply to the low-frequency components. (default 0.5).
             High-frequency components will be weighted by (1 - alpha).
    - cutoff_radius: The radius of the low-pass filter in the frequency domain (default 30).

    Returns:
    - adjusted_noise: The noise after applying frequency separation and weighting.
    """
    
    device = noise.device  # Ensure everything is on the same device as 'noise'

    # Step 1: Apply Fourier transform to convert noise to the frequency domain
    noise_fft = torch.fft.fft2(noise)
    noise_fft_shifted = torch.fft.fftshift(noise_fft)  # Shift the zero frequency component to the center

    # Step 2: Create a low-pass filter (circular mask)
    # Get dimensions
    if len(noise.shape) == 2:  # If it's a 2D tensor (no batch or channels)
        rows, cols = noise.shape
    else:  # If it's a 4D tensor with [batch, channels, height, width]
        _, _, rows, cols = noise.shape
    
    crow, ccol = rows // 2, cols // 2  # Center of the frequency image

    # Generate the low-frequency mask
    y, x = torch.meshgrid(torch.arange(0, rows, device=device), torch.arange(0, cols, device=device), indexing='ij')
    mask = ((x - ccol)**2 + (y - crow)**2 <= cutoff_radius**2).float().to(device)

    if len(noise.shape) == 4:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Step 3: Separate low and high-frequency components
    low_freq_fft = noise_fft_shifted * mask
    high_freq_fft = noise_fft_shifted * (1 - mask)

    # Step 4: Inverse Fourier transform to convert back to the spatial domain
    low_freq_noise = torch.real(torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft)))
    high_freq_noise = torch.real(torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft)))

    # Step 5: Apply single weighting factor, then multiply by 2 to maintain the original amplitude
    adjusted_noise = 2 * (alpha * low_freq_noise + (1 - alpha) * high_freq_noise)

    # Step 6: Optionally normalize the adjusted noise to a specific range (e.g., -1 to 1)
    #adjusted_noise = torch.clamp(adjusted_noise, -1, 1)

    return adjusted_noise


def frequency_equalization(image):
    device = image.device  # Ensure everything is on the same device
    dtype = image.dtype  # Ensure the same dtype

    # Apply Fourier transform
    image_fft = torch.fft.fft2(image.to(device=device, dtype=dtype))
    image_fft_shifted = torch.fft.fftshift(image_fft)

    # Calculate magnitude
    magnitude = torch.abs(image_fft_shifted)

    # Define a target distribution (average magnitude)
    target_distribution = torch.ones_like(magnitude, device=device, dtype=dtype) * magnitude.mean()

    # Equalize the frequency content
    equalized_fft = image_fft_shifted * (target_distribution / (magnitude + 1e-8))

    # Inverse Fourier transform to get the equalized image
    equalized_image = torch.real(torch.fft.ifft2(torch.fft.ifftshift(equalized_fft)))

    return equalized_image

def local_frequency_equalization(image, patch_size=64, stride=32):
    """
    Performs local frequency equalization by dividing the image into patches, equalizing
    the frequency content of each patch, and recombining them.

    Parameters:
    - image: The input image tensor (shape [batch, channels, height, width] or [height, width]).
    - patch_size: The size of each patch for local frequency equalization.
    - stride: The step size for the sliding window. A lower stride introduces overlap between patches.

    Returns:
    - equalized_image: The image after local frequency equalization.
    """

    # Ensure the image is 4D (batch, channels, height, width)
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, height, width]
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)  # Convert to [1, channels, height, width]

    # Ensure we are working with the same device and dtype
    device = image.device
    dtype = image.dtype

    # Get the image dimensions
    batch_size, channels, height, width = image.shape

    # Extract patches using unfold
    patches = F.unfold(image, kernel_size=patch_size, stride=stride)
    patches = patches.permute(0, 2, 1)  # Rearrange to [batch, num_patches, patch_size*patch_size]

    # Reshape to have individual patches of size [batch, num_patches, channels, patch_height, patch_width]
    patches = patches.view(batch_size, patches.shape[1], channels, patch_size, patch_size)

    # Initialize a tensor to hold the equalized patches, on the same device and dtype as the input
    equalized_patches = torch.zeros_like(patches, device=device, dtype=dtype)

    # Perform frequency equalization on each patch
    for i in range(patches.shape[1]):
        equalized_patches[:, i] = frequency_equalization(patches[:, i])

    # Recombine the patches back into the full image using fold
    equalized_patches = equalized_patches.view(batch_size, -1, patch_size * patch_size)
    equalized_patches = equalized_patches.permute(0, 2, 1)
    equalized_image = F.fold(equalized_patches, output_size=(height, width), kernel_size=patch_size, stride=stride)

    return equalized_image

def frequency_weighted_sum(image, cutoff_radius=30):
    """
    Splits the image into low and high-frequency components, creates a density map representing
    the relative contribution of low vs high-frequency content, and combines them using
    weighted contributions that sum to 1.

    Parameters:
    - image: The input image tensor (shape [batch, channels, height, width] or [height, width]).
    - cutoff_radius: The radius of the low-pass filter in the frequency domain (default 30).

    Returns:
    - weighted_image: The image after weighting low and high-frequency contributions.
    """

    device = image.device  # Ensure everything is on the same device
    dtype = image.dtype  # Ensure the same dtype

    # Step 1: Apply Fourier transform to convert image to the frequency domain
    image_fft = torch.fft.fft2(image)
    image_fft_shifted = torch.fft.fftshift(image_fft)  # Shift the zero frequency component to the center

    # Get dimensions
    if len(image.shape) == 2:  # If it's a 2D tensor (no batch or channels)
        rows, cols = image.shape
    else:  # If it's a 4D tensor with [batch, channels, height, width]
        _, _, rows, cols = image.shape

    crow, ccol = rows // 2, cols // 2  # Center of the frequency image

    # Generate meshgrid on the same device as the image tensor
    y, x = torch.meshgrid(torch.arange(0, rows, device=device), torch.arange(0, cols, device=device), indexing='ij')
    distance_map = ((x - ccol)**2 + (y - crow)**2).float().sqrt()  # Distance from the center (frequency magnitude)

    # Step 2: Create low-pass and high-pass masks based on cutoff radius
    low_freq_mask = (distance_map <= cutoff_radius).float()
    high_freq_mask = 1.0 - low_freq_mask  # Complement of the low-pass mask

    # Step 3: Apply the masks to the frequency content
    low_freq_fft = image_fft_shifted * low_freq_mask
    high_freq_fft = image_fft_shifted * high_freq_mask

    # Step 4: Inverse Fourier transform to get low and high frequency components
    low_freq_image = torch.real(torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft)))
    high_freq_image = torch.real(torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft)))

    # Step 5: Compute the magnitude of the low and high-frequency components
    low_freq_magnitude = torch.abs(low_freq_image)
    high_freq_magnitude = torch.abs(high_freq_image)

    # Step 6: Create a density map that shows the relative contribution of low vs high-frequency content
    # Normalize the magnitudes to avoid division by zero
    total_magnitude = low_freq_magnitude + high_freq_magnitude + 1e-8  # Add a small value to prevent division by zero
    low_freq_weight = low_freq_magnitude / total_magnitude
    high_freq_weight = high_freq_magnitude / total_magnitude

    # Step 7: Combine low and high frequency components using weights
    weighted_image = low_freq_weight * low_freq_image + high_freq_weight * high_freq_image

    # Optionally normalize the result
    #weighted_image = torch.clamp(weighted_image, 0, 1)  # Normalize between 0 and 1 if necessary

    return weighted_image

def full_frequency_rebalance(image):
    """
    Iterates through all frequencies in the image, isolates each frequency band, calculates its magnitude,
    and applies a weighted contribution to each, summing the contributions so they effectively sum to 1.

    Parameters:
    - image: The input image tensor (shape [batch, channels, height, width] or [height, width]).

    Returns:
    - rebalanced_image: The image after rebalancing the contributions of all frequencies.
    """

    device = image.device  # Ensure everything is on the same device
    dtype = image.dtype  # Ensure the same dtype

    # Step 1: Apply Fourier transform to convert image to the frequency domain
    image_fft = torch.fft.fft2(image)
    image_fft_shifted = torch.fft.fftshift(image_fft)  # Shift the zero frequency component to the center

    # Get dimensions
    if len(image.shape) == 2:  # If it's a 2D tensor (no batch or channels)
        rows, cols = image.shape
    else:  # If it's a 4D tensor with [batch, channels, height, width]
        _, _, rows, cols = image.shape

    crow, ccol = rows // 2, cols // 2  # Center of the frequency image

    # Generate meshgrid on the same device as the image tensor
    y, x = torch.meshgrid(torch.arange(0, rows, device=device), torch.arange(0, cols, device=device), indexing='ij')
    distance_map = ((x - ccol)**2 + (y - crow)**2).float().sqrt()  # Distance from the center (frequency magnitude)

    # Initialize rebalanced frequency components
    rebalanced_fft = torch.zeros_like(image_fft_shifted, device=device, dtype=torch.complex128)

    # Maximum frequency is the diagonal frequency from the center (theoretical max frequency)
    max_distance = distance_map.max().item()

    # Step 2: Iterate through every possible frequency band
    for i in range(int(max_distance) + 1):
        # Create a band mask for the current frequency band (a single frequency at distance 'i')
        band_mask = (distance_map >= i) & (distance_map < i + 1)
        band_mask = band_mask.float()

        # Isolate the current frequency band
        current_band_fft = image_fft_shifted * band_mask

        # Compute the magnitude of the current frequency band
        current_band_magnitude = torch.abs(current_band_fft)

        # Normalize the magnitude of the current frequency band to create a weight
        total_magnitude = current_band_magnitude.sum() + 1e-8  # Sum all magnitudes (add small value to avoid zero division)
        current_band_weight = current_band_magnitude / total_magnitude

        # Apply the weight to the current frequency band (keep as complex)
        rebalanced_fft += current_band_weight * current_band_fft

    # Step 3: Inverse Fourier transform to convert back to the spatial domain
    rebalanced_image = torch.real(torch.fft.ifft2(torch.fft.ifftshift(rebalanced_fft)))

    return rebalanced_image

from .refined_exp_solver import get_res4lyf_step_with_model, calculate_third_order_coeffs

@torch.no_grad()
def sample_RES_implicit_advanced_RF_PC(
    model, x, sigmas, extra_args=None, callback=None, disable=None, c2=1.0, auto_c2=False, eta1=0.0, eta2=0.0, eta_var1=0.0, eta_var2=0.0, s_noise1=1.0, s_noise2=1.0,
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard", k=1.0, scale=0.1, 
    alpha=None, iter_c2=0, iter=3, tol=1e-5, reverse_weight_c2=0.0, reverse_weight=0.0,
    latent_guide=None, latent_guide_weight=0.0, latent_guide_weights=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))
    
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    denoised_next = None
    vel, vel_2, denoised, denoised2, denoised1_2, h_last = None, None, None, None, None, None
    
    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)
    
    if mask is None:
        mask = torch.ones_like(x)
    else:
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 16, 1, 1) 
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        mask = mask.to(x.dtype).to(x.device)
        
    #t_fn = lambda sigma: 1/((sigma).exp()+1)
    #sigma_fn = lambda t: ((1-t)/t).log()
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t    
    
        #sigma_up, sigma_down, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta, eta_var, noise_mode)
        sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta2, eta_var2, noise_mode)
        if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard":
            sigma = sigma * (1 + eta2)
        
        sigma_s, h, c2 = get_res4lyf_half_step(sigma, sigma_down, c2, auto_c2, h_last, "", "", remap_t_to_exp_space=True)
        a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=False)

        denoised = model(x, sigma * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            #k2 = lg_weight*latent_guide + (1-lg_weight)*k2
            hard_light_blend_1 = hard_light_blend(latent_guide, denoised)
            denoised = denoised - lg_weight * sigma_next * denoised  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
        x_2 = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised
        gc.collect()
        torch.cuda.empty_cache()
        
        for iteration in range(iter_c2):  
            time = sigmas[i] / sigma_max
            
            denoised_next = model(x_2, sigma_s * s_in, **extra_args)
            x_2_new = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised_next
            
            # x = (x_new - h*denoised1_2*(b1 + b2)) / (sd/sigma)   # projection back to x

            error = torch.norm(x_2_new - x_2)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_2 = x_2_new
                denoised = denoised_next
                break
            
            if reverse_weight_c2 > 0.0:
                #x_2_new = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised_next
                #(x_2_new - h*a2_1*denoised_next) / ((sigma_down/sigma)**c2) = x 
                x_reverse_new = (x_2 - h*a2_1*denoised_next) / ((sigma_down/sigma)**c2)
                x = reverse_weight_c2 * x_reverse_new + (1-reverse_weight_c2) * x
                
            """if reverse_weight_c2 > 0.0:
                #x_2_new = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised_next
                #(x_2_new - h*a2_1*denoised_next) / ((sigma_down/sigma)**c2) = x 
                x_reverse_new = (x_2_new - h*a2_1*denoised) / ((sigma_down/sigma)**c2)
                x = reverse_weight_c2 * x_reverse_new + (1-reverse_weight_c2) * x
                denoised = denoised_next
                
            if reverse_weight_c2 > 0.0:
                #x_2_new = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised_next
                #(x_2_new - h*a2_1*denoised_next) / ((sigma_down/sigma)**c2) = x 
                x_reverse_new = x + (denoised_next - denoised) * sigma
                x = reverse_weight_c2 * x_reverse_new + (1-reverse_weight_c2) * x
                denoised = denoised_next"""

            x_2 = x_2_new
            denoised = denoised_next
            
            gc.collect()
            torch.cuda.empty_cache()
            #denoised_next = (b1*denoised + b2*denoised2_next) / (b1 + b2)

            #x  = (x_new - (1 - sigma_next/sigma) * denoised_next) / (sigma_next/sigma) # projection back to x with alternative math
        su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1, eta_var1, noise_mode)
        noise1 = noise_sampler(sigma=sigma, sigma_next=sigma_s)
        noise1 = (noise1 - noise1.mean()) / noise1.std()
        x_2 = alpha_ratio_2 * x_2 + noise1 * s_noise1 * su_2
        #x_2 = alpha_ratio_2 * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * s_noise1 * su_2
        
        denoised2 = model(x_2, sigma_s * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            #k2 = lg_weight*latent_guide + (1-lg_weight)*k2
            hard_light_blend_1 = hard_light_blend(latent_guide, denoised2)
            denoised2 = denoised2 - lg_weight * sigma_next * denoised2  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
        x_next = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2)
        denoised_next = (b1*denoised + b2*denoised2) / (b1 + b2)
        
        gc.collect()
        torch.cuda.empty_cache()

        for iteration in range(iter):  
            time = sigmas[i] / sigma_max
            
            denoised2_next = model(x_next, sigma_down * s_in, **extra_args)
            x_new = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2_next)
            
            # x = (x_new - h*denoised1_2*(b1 + b2)) / (sd/sigma)   # projection back to x

            error = torch.norm(x_new - x_next)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_new
                denoised2 = denoised2_next
                denoised_next = (b1*denoised + b2*denoised2_next) / (b1 + b2)
                break
            
            if reverse_weight > 0.0:
                #x_new = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2_next)
                #(x_new - h*(b1*denoised + b2*denoised2_next)) / (sigma_down/sigma) = x 
                x_reverse_new = (x_next - h*(b1*denoised + b2*denoised2_next)) / (sigma_down/sigma)
                x = reverse_weight * x_reverse_new + (1-reverse_weight) * x

            """if reverse_weight > 0.0:
                #x_new = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2_next)
                #(x_new - h*(b1*denoised + b2*denoised2_next)) / (sigma_down/sigma) = x 
                x_reverse_new = (x_new - h*(b1*denoised + b2*denoised2)) / (sigma_down/sigma)
                x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
                denoised2 = denoised2_next

            if reverse_weight > 0.0:
                #x_new = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2_next)
                #(x_new - h*(b1*denoised + b2*denoised2_next)) / (sigma_down/sigma) = x 
                x_reverse_new = x + (denoised2_next - denoised2) * sigma
                x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
                denoised2 = denoised2_next"""

            x_next = x_new
            denoised2 = denoised2_next
            denoised_next = (b1*denoised + b2*denoised2_next) / (b1 + b2)
            
            gc.collect()
            torch.cuda.empty_cache()

            #x  = (x_new - (1 - sigma_next/sigma) * denoised_next) / (sigma_next/sigma) # projection back to x with alternative math

        x = x_next
        h_last = h
        
        #if sigmas[i + 1] > 0 and eta > 0 and isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) == True:
        noise2 = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise2 = (noise2 - noise2.mean()) / noise2.std()
        x = alpha_ratio * x + noise2 * s_noise2 * sigma_up
        #x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise2 * sigma_up

        denoised_next = denoised if denoised_next is None else denoised_next
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma, 'denoised': denoised_next})

        gc.collect()
        torch.cuda.empty_cache()

    return x




def compute_laplacian(image):
    laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
    
    laplacian_per_channel = []
    for channel in range(image.shape[1]):
        laplacian = F.conv2d(image[:, channel:channel+1, :, :], laplacian_filter, padding=1)
        laplacian_per_channel.append(laplacian)
    
    laplacian = torch.cat(laplacian_per_channel, dim=1)
    return laplacian



def sharpen(input_image):

    sharpen_kernel = torch.tensor([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=input_image.dtype)

    sharpen_kernel = sharpen_kernel.view(1, 1, 3, 3).to(input_image.device)
    return F.conv2d(input_image, sharpen_kernel.repeat(input_image.shape[1], 1, 1, 1), padding=1, groups=input_image.shape[1])

from .refined_exp_solver import hard_light_blend

@torch.no_grad()
def sample_RES_implicit_advanced_RF_PC_3rd_order(
    model, x, sigmas, extra_args=None, callback=None, disable=None, c2=0.5, c3=1.5, auto_c2=False, eta1=0.0, eta2=0.0, eta3=0.0, eta_var1=0.0, eta_var2=0.0, eta_var3=0.0, s_noise1=1.0, s_noise2=1.0, s_noise3=0.0,
    noise_sampler=None, noise_sampler_type1="gaussian", noise_sampler_type2="gaussian", noise_sampler_type3="gaussian",noise_mode="hard", k=1.0, scale=0.1, 
    alpha=None, iter_c2=0, iter_c3=0, iter=3, tol=1e-5, reverse_weight_c2=0.0, reverse_weight_c3=0.0, reverse_weight=0.0,
    latent_guide=None, latent_guide_weight=0.0, latent_guide_weights=None, mask=None):

    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha 
    sigma_min, sigma_max = sigmas[sigmas > 0].min()**2, sigmas.max() #squaring sigma_min to avoid sigma_next == sigma_min issue with brownian...
    
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))
    
    noise_sampler1 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type1)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type2)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler3 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type3)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    vel, vel_2, denoised, denoised2, denoised1_2, h_last = None, None, None, None, None, None
    
    if mask is None:
        mask = torch.ones_like(x)
    else:
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 16, 1, 1) 
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        mask = mask.to(x.dtype).to(x.device)

    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)
    

    """sigmas_reversed = torch.flip(sigmas, dims=[0])
    
    epsilons = []

    for i in trange(len(sigmas_reversed) - 1, disable=disable):

        denoised = model(latent_guide, sigmas_reversed[i] * s_in, **extra_args)
        
        epsilon = latent_guide - denoised
        epsilons.append(epsilon)
        d = to_d(latent_guide, sigmas_reversed[i], denoised)

        if callback is not None:
            callback({'x': latent_guide, 'i': i, 'sigma': sigmas_reversed[i], 'sigma_hat': sigmas_reversed[i], 'denoised': denoised, 'epsilon': epsilon})

        dt = sigmas_reversed[i + 1] - sigmas_reversed[i]
        latent_guide = latent_guide + d * dt"""
        

    #t_fn = lambda sigma: 1/((sigma).exp()+1)
    #sigma_fn = lambda t: ((1-t)/t).log()
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type1 == "fractal":
            noise_sampler1.alpha = alpha[i]
            noise_sampler1.k = k
            noise_sampler1.scale = scale
        if noise_sampler_type2 == "fractal":
            noise_sampler2.alpha = alpha[i]
            noise_sampler2.k = k
            noise_sampler2.scale = scale
        if noise_sampler_type3 == "fractal":
            noise_sampler3.alpha = alpha[i]
            noise_sampler3.k = k
            noise_sampler3.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        
        
        t      = t_fn(sigma)
        t_next = t_fn(sigma_next)
        
        h = t_next - t
        print("sigma_next: ", sigma_next)
        if h >= 0.9999:
            h = torch.tensor(0.9999).to(h.dtype).to(h.device) 
            t_next = h + t
            sigma_next = sigma_fn(t_next)
        
        s2 = t + h * c2
        s3 = t + h * c3
        
        a21, a31, a32, b1, b2, b3 = calculate_third_order_coeffs(h, c2, c3)

        sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta3[i], eta_var3[i], noise_mode, h)
            
        
        if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard":
            sigma = sigma * (1 + eta3[i])
        
        t      = t_fn(sigma)
        t_next = t_fn(sigma_down)
        
        h = t_next - t
        
        print("sigma_next: ", sigma_next)
        if h >= 0.9999:
            h = torch.tensor(0.9999).to(h.dtype).to(h.device) 
            t_next = h + t
            sigma_next = sigma_fn(t_next)
        
        s2 = t + h * c2
        s3 = t + h * c3
        sigma_2 = sigma_fn(s2)
        sigma_3 = sigma_fn(s3)


        su_2, sd_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_next, eta1[i], eta_var1[i], noise_mode, h)
        su_3, sd_3, alpha_ratio_3 = get_res4lyf_step_with_model(model, sigma, sigma_next, eta2[i], eta_var2[i], noise_mode, h)
        
        print("sigma_down orig: ", sigma_down.item())
        su_total = sigma_up #+ su_2 + su_3
        
        #s_d = (-sigma_next**2 + su_total**2 + sigma_next * torch.sqrt((su_total+sigma_next)*(sigma_next - su_total)) - torch.sqrt((su_total+sigma_next)*(sigma_next-su_total)) )   /   (-2*sigma_next + 1 + su_total**2) #values end up being > 1.0
        s_d = (-sigma_next**2 + su_total**2 - sigma_next * torch.sqrt((su_total+sigma_next)*(sigma_next - su_total)) + torch.sqrt((su_total+sigma_next)*(sigma_next-su_total)) )   /   (-2*sigma_next + 1 + su_total**2) #this one works and should be very general. this assumes alpha_ratio = (1 - sigma_down)/(1 - sigma_next)
        s_u = torch.sqrt(sigma_next**2 - 2*sigma_down*sigma_next**2 + sigma_next**2 * sigma_down**2 - sigma_down**2 + 2*sigma_down**2 * sigma_next - sigma_down**2 * sigma_next**2 ) / (1 - sigma_down)
        
        print(su_total.item(), sigma_next.item(), sigma_down.item(), s_d.item(), s_u.item())
        sigma_down = s_d
        
        """noise1 = noise_sampler1(sigma=sigma, sigma_next=sigma_2)
        noise2 = noise_sampler2(sigma=sigma, sigma_next=sigma_3)
        noise3 = noise_sampler3(sigma=sigma, sigma_next=sigma_next)
        
        noise1 = (noise1 - noise1.mean()) / noise1.std()
        noise2 = (noise2 - noise2.mean()) / noise2.std()
        noise3 = (noise3 - noise3.mean()) / noise3.std()"""

        k1 = model(x, sigma * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            #k1 = lg_weight*latent_guide + (1-lg_weight)*k1            
            hard_light_blend_1 = hard_light_blend(latent_guide, k1)
            k1 = k1 - lg_weight * sigma_next * k1  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
        
        #su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1[i], eta_var1[i], noise_mode)
        x_2 = ((sigma_down/sigma)**c2)*x + h*(a21*k1)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        for iteration in range(iter_c2):  
            time = sigmas[i] / sigma_max
            
            k1_new = model(x_2, sigma_2 * s_in, **extra_args)
            x_2_new = ((sigma_down/sigma)**c2)*x + h*(a21*k1_new)
            
            error = torch.norm(x_2_new - x_2)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_2 = x_2_new
                k1 = k1_new   #check second order method for this, may have forgotten
                break
            
            if reverse_weight_c2 > 0.0:
                x_reverse_new = (x_2 - h*(a21*k1_new)) / ((sigma_down/sigma)**c2)
                x = reverse_weight_c2 * x_reverse_new + (1-reverse_weight_c2) * x

            x_2 = x_2_new
            k1 = k1_new
            
            gc.collect()
            torch.cuda.empty_cache()

        su_2, sd_2, alpha_ratio_2 = get_res4lyf_step_with_model(model, sigma, sigma_next, eta1[i], eta_var1[i], noise_mode, h)
        #su_2, sd_2, alpha_ratio_2 =                   get_res4lyf_step(sigma, sigma_next, eta1[i], eta_var1[i], noise_mode)
        #x_2 = alpha_ratio_2 * x_2 + noise_sampler1(sigma=sigma, sigma_next=sigma_2) * s_noise1[i] * su_2
        noise1 = noise_sampler1(sigma=sigma, sigma_next=sigma_2)
        noise1 = (noise1 - noise1.mean()) / noise1.std()
        x_2 = alpha_ratio_2 * x_2 + noise1 * s_noise1[i] * su_2
        
        
        
        k2 = model(x_2, sigma_2 * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            #k2 = lg_weight*latent_guide + (1-lg_weight)*k2
            hard_light_blend_1 = hard_light_blend(latent_guide, k2)
            k2 = k2 - lg_weight * sigma_next * k2  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
        #su_3, sd_3, alpha_ratio_3 = get_res4lyf_step(sigma, sigma_next, eta2[i], eta_var2[i], noise_mode)
        x_3 = ((sigma_down/sigma)**c3)*x + h*(a31*k1 + a32*k2)
        
        gc.collect()
        torch.cuda.empty_cache()

        for iteration in range(iter_c3):  
            time = sigmas[i] / sigma_max
            
            k2_new = model(x_3, sigma_3 * s_in, **extra_args)
            x_3_new = ((sigma_down/sigma)**c3)*x + h*(a31*k1 + a32*k2_new)
            
            error = torch.norm(x_3_new - x_3)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_3 = x_3_new 
                k2 = k2_new
                break
            
            if reverse_weight_c3 > 0.0:
                x_reverse_new = (x_3 - h*(a31*k1 + a32*k2_new)) / ((sigma_down/sigma)**c3)
                x = reverse_weight_c3*x_reverse_new + (1-reverse_weight_c3)*x

            x_3 = x_3_new 
            k2 = k2_new
            
            gc.collect()
            torch.cuda.empty_cache()
            
        su_3, sd_3, alpha_ratio_3 = get_res4lyf_step_with_model(model, sigma, sigma_next, eta2[i], eta_var2[i], noise_mode, h)
        #su_3, sd_3, alpha_ratio_3 =                   get_res4lyf_step(sigma, sigma_next, eta2[i], eta_var2[i], noise_mode)
        #x_3 = alpha_ratio_3 * x_3 + noise_sampler2(sigma=sigma, sigma_next=sigma_3) * s_noise2[i] * su_3
        noise2 = noise_sampler2(sigma=sigma, sigma_next=sigma_3)
        noise2 = (noise2 - noise2.mean()) / noise2.std()
        x_3 = alpha_ratio_3 * x_3 + noise2 * s_noise2[i] * su_3



        k3 = model(x_3, sigma_3 * s_in, **extra_args)

        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            #k3 = lg_weight*latent_guide + (1-lg_weight)*k3
            hard_light_blend_1 = hard_light_blend(latent_guide, k3)
            k3 = k3 - lg_weight * sigma_next * k3  + (lg_weight * sigma_next * hard_light_blend_1 * mask)

        x_next =  ((sigma_down/sigma))*x + h*(b1*k1 + b2*k2 + b3*k3)
        
        gc.collect()
        torch.cuda.empty_cache()

        for iteration in range(iter):  
            time = sigmas[i] / sigma_max

            k3_new = model(x_next, sigma_down * s_in, **extra_args)
            x_next_new = ((sigma_down/sigma))*x + h*(b1*k1 + b2*k2 + b3*k3_new)
            
            error = torch.norm(x_next_new - x_next)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_next_new 
                k3 = k3_new
                break
            
            if reverse_weight > 0.0:
                x_reverse_new = (x_next - h*(b1*k1 + b2*k2 + b3*k3_new)) / ((sigma_down/sigma))
                x = reverse_weight*x_reverse_new + (1-reverse_weight)*x

            x_next = x_next_new 
            k3 = k3_new
            
            gc.collect()
            torch.cuda.empty_cache()

        x = x_next
        
        
        #if sigmas[i + 1] > 0 and eta > 0 and isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) == True:
        #x = alpha_ratio * x + noise_sampler3(sigma=sigma, sigma_next=sigma_next) * s_noise3[i] * sigma_up
        noise3 = noise_sampler3(sigma=sigma, sigma_next=sigma_next)
        noise3 = (noise3 - noise3.mean()) / noise3.std()
        x = alpha_ratio * x + noise3 * s_noise3[i] * sigma_up

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma, 'denoised': k3})

        gc.collect()
        torch.cuda.empty_cache()

    return x



@torch.no_grad()
def sample_SDE_implicit_advanced_RF(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., eta_var=1., s_noise=1., 
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard",reverse_weight=0.0, k=1.0, scale=0.1, 
    alpha=None, iter=3, tol=1e-5):
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    x_reverse_new = None
    
    #t_fn = lambda sigma: 1/((sigma).exp()+1)
    #sigma_fn = lambda t: ((1-t)/t).log()
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        sigma_up, sigma_down, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta, eta_var, noise_mode)
        #sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, )
        #t, t_next = t_fn(sigma), t_fn(sigma_next)
        #h = t_next - t
           
        sigma_next = sigma_down
        denoised = model(x, sigma * s_in, **extra_args)

        x_next = (sigma_next/sigma) * x + (1 - sigma_next/sigma) * denoised 
        denoised_next = denoised

        for iteration in range(iter):  
            #x = x_reverse_new if x_reverse_new is not None else x
            denoised_next = model(x_next, sigma_next * s_in, **extra_args)
            if reverse_weight > 0.0:
                x_reverse_new = (x_next - (1 - sigma_next/sigma) * denoised_next) / (sigma_next/sigma)
            x_new = (sigma_next/sigma) * x + (1 - sigma_next/sigma) * denoised_next

            error = torch.norm(x_new - x_next)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_new
                break

            x_next = x_new
            x = reverse_weight * x_reverse_new + (1-reverse_weight) * x

        x = x_next
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_next})

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
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_next': sigma_next, 'dx': dx})
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
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_next': sigma_next, 'dx': dx})
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
from .refined_exp_solver import _refined_exp_sosu_step_RF

#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
@torch.no_grad()
def sample_deis_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3, deis_mode='rhoab', step_type='res_a', denoised_type="1_2",
                    momentums=None, etas=None, s_noises=None, noise_sampler_type="gaussian", noise_mode="hard", noise_scale=0.0, k=1.0, scale=0.1, alpha=None,   
                    latent_guide=None, latent_guide_weight=0.0, latent_guide_weights=None, mask=None):

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))
        
    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)

    if mask is None:
        mask = torch.ones_like(x)
    else:
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 16, 1, 1) 
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        mask = mask.to(x.dtype).to(x.device)
        
    seed = extra_args.get("seed", None) + 1
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    print("DEIS_SDE seed set to: ", seed)

    vel = None
    vel, vel_2, x_n, denoised, denoised2, denoised1_2, h_last = None, None, None, None, None, None, None
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
        elif noise_mode == "exp": 
            h = sigma_next.log().neg() - sigma.log().neg()
            sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, etas[i], torch.full_like(etas[i], 0.0), noise_mode, h)


            #sigma_up, sigma_down, alpha_ratio = get_scaled_ancestral_step_RF(sigma_next, etas[i], 1.1)
            #sigma_up = sigmas[i] * etas[i]
            #sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])
        
        sigma_ratio = (sigma_down - sigma) / (sigma_next - sigma) #sigma_minus_full / sigma_minus_orig... alternative to using sigma_down in the equations directly

        #print(sigma_up, sigma_down, alpha_ratio, sigma_ratio, alpha_ratio * sigma_ratio)
        #print(sigma_next ** 2 - sigma_up ** 2 - sigma_down ** 2 * alpha_ratio ** 2)

        x_cur = x_next

        if step_type == "simple": 
            denoised = model(x_cur, sigma * s_in, **extra_args)
            
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                #k2 = lg_weight*latent_guide + (1-lg_weight)*k2
                hard_light_blend_1 = hard_light_blend(latent_guide, denoised)
                denoised = denoised - lg_weight * sigma_next * denoised  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
            denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i] / 2.0)
        elif step_type == "res_a":
            """x, denoised, denoised2, denoised1_2, vel, vel_2 = _refined_exp_sosu_step_RF_hard_deis(model, x_cur, sigma, sigma_next, sigmas[i+2], c2=0.5,eta=etas[i], noise_sampler=noise_sampler, s_noise=s_noises[i], noise_mode=noise_mode, ancestral_noise=True,
                                                                                extra_args=extra_args, pbar=None, simple_phi_calc=False,
                                                                                momentum = momentums[i], vel = vel, vel_2 = vel_2, time = time, eulers_mom = 0.0, cfgpp = 0.0
                                                                                )"""
            x, denoised, denoised2, denoised1_2, vel, vel_2, h_last = _refined_exp_sosu_step_RF(model, x_cur, sigma, sigma_next, c2=torch.tensor(0.5),eta1=etas[i]/2, eta2=etas[i], noise_sampler=noise_sampler, s_noise1=s_noises[i], s_noise2=s_noises[i], noise_mode=noise_mode,
                                                                                extra_args=extra_args, pbar=None, simple_phi_calc=False,
                                                                                momentum = momentums[i], vel = vel, vel_2 = vel_2, time = time, order=2, denoised1_2=denoised, h_last=h_last,
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







@torch.no_grad()
def sample_deis_sde_implicit(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3, deis_mode='rhoab', step_type='res_a', denoised_type="1_2",
                    momentums=None, etas=None, s_noises=None, noise_sampler_type="gaussian", noise_mode="hard", noise_scale=0.0, k=1.0, scale=0.1, alpha=None, iter=3, tol=1e-5, reverse_weight=0.0):

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    if sigmas[-1] == 0.0:
        sigmas[-1] = min(sigmas[-2]**2, 0.00001)
    
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    print("DEIS_SDE seed set to: ", seed)

    vel = None
    vel, vel_2, x_n, denoised, denoised2, denoised1_2, h_last = None, None, None, None, None, None, None
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentums[0] / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentums[0] * (timescale + offset) * velocity + (1 - momentums[0] * (timescale + offset)) * diff
        return momentum_vel
    
    x_next = x

    coeff_list = get_deis_coeff_list(sigmas, max_order, deis_mode=deis_mode)

    buffer_model = []
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
        elif noise_mode == "exp": 
            h = sigma_next.log().neg() - sigma.log().neg()
            sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, etas[i], torch.full_like(etas[i], 0.0), noise_mode, h)

            #sigma_up, sigma_down, alpha_ratio = get_scaled_ancestral_step_RF(sigma_next, etas[i], 1.1)
            #sigma_up = sigmas[i] * etas[i]
            #sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])
        
        sigma_ratio = (sigma_down - sigma) / (sigma_next - sigma) #sigma_minus_full / sigma_minus_orig... alternative to using sigma_down in the equations directly

        #print(sigma_up, sigma_down, alpha_ratio, sigma_ratio, alpha_ratio * sigma_ratio)
        #print(sigma_next ** 2 - sigma_up ** 2 - sigma_down ** 2 * alpha_ratio ** 2)

        x_cur = x_next

        denoised = model(x_cur, sigma * s_in, **extra_args)
        denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i] / 2.0)
        
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
            
        gc.collect()
        torch.cuda.empty_cache()
            
        
        for iteration in range(iter):  
            denoised_new = model(x_next, sigma_down * s_in, **extra_args)
            denoised_new = momentum_func(denoised_new, vel, sigmas[i], -momentums[i] / 2.0)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_new})
            d_new = ((x_cur - denoised_new) / sigma) * sigma_ratio
            
            if sigma_next <= 0:
                order = 1
            if order == 1:          # First Euler step.
                dt = sigma_next - sigma  #from the euler ancestral RF sampler
                x_new = x_cur + dt * d_new
            elif order == 2:        # Use one history point.
                coeff_cur, coeff_prev1 = coeff_list[i]
                x_new = x_cur + coeff_cur * d_new + coeff_prev1 * buffer_model[-1]
            elif order == 3:        # Use two history points.
                coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
                x_new = x_cur + coeff_cur * d_new + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
            elif order == 4:        # Use three history points.
                coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
                x_new = x_cur + coeff_cur * d_new + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
                
            error = torch.norm(x_new - x_next)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_new
                break
            
            if reverse_weight > 0.0:
                if order == 1:          # First Euler step.
                    #dt = sigma_next - sigma  #from the euler ancestral RF sampler
                    #x_new = x_cur + dt * d_new
                    x_cur = x_next - (dt * d_new)
                elif order == 2:        # Use one history point.
                    #coeff_cur, coeff_prev1 = coeff_list[i]
                    #x_new = x_cur + coeff_cur * d_new + coeff_prev1 * buffer_model[-1]
                    x_cur = x_next - (coeff_cur * d_new + coeff_prev1 * buffer_model[-1])
                elif order == 3:        # Use two history points.
                    #coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
                    #x_new = x_cur + coeff_cur * d_new + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
                    x_cur = x_next - (coeff_cur * d_new + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2])
                elif order == 4:        # Use three history points.
                    #coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
                    #x_new = x_cur + coeff_cur * d_new + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
                    x_cur = x_next - (coeff_cur * d_new + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3])

            x_next = x_new
            denoised = denoised_new
            
            gc.collect()
            torch.cuda.empty_cache()

        if max_order > 1:
            if len(buffer_model) == max_order - 1:
                for k in range(max_order - 2):
                    buffer_model[k] = buffer_model[k+1]
                buffer_model[-1] = d_cur.detach()
            else:
                buffer_model.append(d_cur.detach())
            
        if sigma_next > 0 and etas[i] > 0:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            noise = noise * s_noises[i] * sigma_up   
            x_next = x_next * alpha_ratio   +   noise                                            
            
        gc.collect() #necessary after every step to minimize OOM errors with flux dev
        torch.cuda.empty_cache()
        
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

#from .refined_exp_solver import sample_refined_exp_s, sample_refined_exp_s_advanced
extra_samplers = {
    "euler_ancestral_advanced": sample_euler_ancestral_advanced,
    "RES_implicit_advanced_RF": sample_RES_implicit_advanced_RF, 
    "RES_implicit_advanced_RF_PC": sample_RES_implicit_advanced_RF_PC, 
    "RES_implicit_advanced_RF_PC_3rd_order": sample_RES_implicit_advanced_RF_PC_3rd_order,
    "SDE_implicit_advanced_RF": sample_SDE_implicit_advanced_RF,
    "euler_implicit_advanced_RF": sample_euler_implicit_advanced_RF,
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
    "deis_sde_implicit": sample_deis_sde_implicit,


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
