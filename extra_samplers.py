import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc
import kornia

from comfy.k_diffusion.sampling import get_ancestral_step, to_d

from comfy.k_diffusion.sampling import deis
from .refined_exp_solver import _refined_exp_sosu_step_RF

import torch.nn.functional as F
import torchvision.transforms as T
from .refined_exp_solver import hard_light_blend


import functools

from .noise_classes import *
from .extra_samplers_helpers import get_deis_coeff_list


from .refined_exp_solver import _de_second_order, get_res4lyf_half_step, get_res4lyf_step, \
    get_res4lyf_step_with_model, calculate_third_order_coeffs

from .noise_sigmas_timesteps_scaling import *


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


def get_res4lyf_half_step(sigma, sigma_next, c2=0.5, auto_c2=False, h_last=None, t_fn_formula="", sigma_fn_formula="", remap_t_to_exp_space=True ):
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
    
  #print("sigma:", sigma.item(), "sigma_s:", sigma_s.item(), "sigma_next:", sigma_next.item(),)
  #print("t:", t.item(), "s:", s.item(), "t_next:", t_next.item(), "h:", h.item(), "c2:", c2.item())
  
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

from numpy.polynomial.chebyshev import chebfit, chebval

def chebyshev_fit(x, nodes, N):
    """
    Fit a Chebyshev polynomial to the function values using PyTorch.
    x: Function values at the Chebyshev nodes (PyTorch tensor).
    nodes: The Chebyshev nodes (PyTorch tensor).
    N: Degree of the Chebyshev polynomial.
    
    Returns: Chebyshev coefficients (PyTorch tensor).
    """
    # gen chebyshev polynomials T_k at nodes
    T = [torch.ones_like(nodes), nodes]  # T_0 = 1, T_1 = x
    for k in range(2, N + 1):
        T_k = 2 * nodes * T[k - 1] - T[k - 2]
        T.append(T_k)
    
    # stack polynomials T_0, T_1, ..., T_N
    T = torch.stack(T, dim=1)
    
    # calc chebyshev coefficients... least-squares fit
    coeffs = torch.linalg.lstsq(T, x).solution
    
    return coeffs

def chebyshev_eval(coeffs, x, N):
    """
    Evaluate a Chebyshev polynomial at point x using the Chebyshev coefficients.
    coeffs: Chebyshev coefficients (PyTorch tensor).
    x: The point at which to evaluate the polynomial (PyTorch tensor).
    N: Degree of the Chebyshev polynomial.
    
    Returns: Evaluated polynomial at x (PyTorch tensor).
    """
    T_0 = torch.ones_like(x)
    T_1 = x
    
    # Evaluate using the recurrence relation
    result = coeffs[0] * T_0 + coeffs[1] * T_1
    for k in range(2, N + 1):
        T_k = 2 * x * T_1 - T_0
        result = result + coeffs[k] * T_k
        T_0, T_1 = T_1, T_k
    
    return result



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
                su, sd, alpha_ratio = get_RF_step(sigma_fn_RF(t), sigma_fn_RF(s), current_eta)
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
                su, sd, alpha_ratio = get_RF_step(sigma_fn_RF(t), sigma_fn_RF(t_next), current_eta)
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

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t    
    
        sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta2, eta_var2, noise_mode)
        if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard":
            sigma = sigma * (1 + eta2)
        
        h = t_fn(sigma_down) - t_fn(sigma)
        sigma_s, h_half, c2 = get_res4lyf_half_step(sigma, sigma_down, c2, auto_c2, h_last, "", "", remap_t_to_exp_space=True)
        a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=False)

        denoised = model(x, sigma * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            hard_light_blend_1 = hard_light_blend(latent_guide, denoised)
            denoised = denoised - lg_weight * sigma_next * denoised  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
        x_2 = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised
        gc.collect()
        torch.cuda.empty_cache()
        
        for iteration in range(iter_c2):  
            time = sigmas[i] / sigma_max
            
            denoised_next = model(x_2, sigma_s * s_in, **extra_args)
            
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                hard_light_blend_1 = hard_light_blend(latent_guide, denoised_next)
                denoised_next = denoised_next - lg_weight * sigma_next * denoised_next  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
                
            x_2_new = ((sigma_down/sigma)**c2)*x + h*a2_1*denoised_next
            
            error = torch.norm(x_2_new - x_2)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_2 = x_2_new
                denoised = denoised_next
                break
            
            if reverse_weight_c2 > 0.0:
                x_reverse_new = (x_2 - h*a2_1*denoised_next) / ((sigma_down/sigma)**c2)
                x = reverse_weight_c2 * x_reverse_new + (1-reverse_weight_c2) * x
                
            x_2 = x_2_new
            denoised = denoised_next
            
            gc.collect()
            torch.cuda.empty_cache()

        su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1, eta_var1, noise_mode)
        noise1 = noise_sampler(sigma=sigma, sigma_next=sigma_s)
        noise1 = (noise1 - noise1.mean()) / noise1.std()
        x_2 = alpha_ratio_2 * x_2 + noise1 * s_noise1 * su_2
        
        denoised2 = model(x_2, sigma_s * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            hard_light_blend_1 = hard_light_blend(latent_guide, denoised2)
            denoised2 = denoised2 - lg_weight * sigma_next * denoised2  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
        x_next = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2)
        denoised_next = (b1*denoised + b2*denoised2) / (b1 + b2)
        
        gc.collect()
        torch.cuda.empty_cache()

        for iteration in range(iter):  
            time = sigmas[i] / sigma_max
            
            denoised2_next = model(x_next, sigma_down * s_in, **extra_args)
            
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                hard_light_blend_1 = hard_light_blend(latent_guide, denoised2_next)
                denoised2_next = denoised2_next - lg_weight * sigma_next * denoised2_next  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
                
            x_new = (sigma_down/sigma)*x + h*(b1*denoised + b2*denoised2_next)
            
            error = torch.norm(x_new - x_next)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_new
                denoised2 = denoised2_next
                denoised_next = (b1*denoised + b2*denoised2_next) / (b1 + b2)
                break
            
            if reverse_weight > 0.0:
                x_reverse_new = (x_next - h*(b1*denoised + b2*denoised2_next)) / (sigma_down/sigma)
                x = reverse_weight * x_reverse_new + (1-reverse_weight) * x

            x_next = x_new
            denoised2 = denoised2_next
            denoised_next = (b1*denoised + b2*denoised2_next) / (b1 + b2)
            
            gc.collect()
            torch.cuda.empty_cache()

        x = x_next
        h_last = h
        
        noise2 = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise2 = (noise2 - noise2.mean()) / noise2.std()
        x = alpha_ratio * x + noise2 * s_noise2 * sigma_up

        denoised_next = denoised if denoised_next is None else denoised_next
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma, 'denoised': denoised_next})

        gc.collect()
        torch.cuda.empty_cache()

    return x




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
        
        s_d = (-sigma_next**2 + su_total**2 - sigma_next * torch.sqrt((su_total+sigma_next)*(sigma_next - su_total)) + torch.sqrt((su_total+sigma_next)*(sigma_next-su_total)) )   /   (-2*sigma_next + 1 + su_total**2) #this one works and should be very general. this assumes alpha_ratio = (1 - sigma_down)/(1 - sigma_next)
        s_u = torch.sqrt(sigma_next**2 - 2*sigma_down*sigma_next**2 + sigma_next**2 * sigma_down**2 - sigma_down**2 + 2*sigma_down**2 * sigma_next - sigma_down**2 * sigma_next**2 ) / (1 - sigma_down)
        
        print(su_total.item(), sigma_next.item(), sigma_down.item(), s_d.item(), s_u.item())
        sigma_down = s_d

        k1 = model(x, sigma * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            hard_light_blend_1 = hard_light_blend(latent_guide, k1)
            k1 = k1 - lg_weight * sigma_next * k1  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
        
        x_2 = ((sigma_down/sigma)**c2)*x + h*(a21*k1)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        for iteration in range(iter_c2):  
            time = sigmas[i] / sigma_max
            
            k1_new = model(x_2, sigma_2 * s_in, **extra_args)
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                hard_light_blend_1 = hard_light_blend(latent_guide, k1_new)
                k1_new = k1_new - lg_weight * sigma_next * k1_new  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
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
        noise1 = noise_sampler1(sigma=sigma, sigma_next=sigma_2)
        noise1 = (noise1 - noise1.mean()) / noise1.std()
        x_2 = alpha_ratio_2 * x_2 + noise1 * s_noise1[i] * su_2
        
        
        
        k2 = model(x_2, sigma_2 * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            hard_light_blend_1 = hard_light_blend(latent_guide, k2)
            k2 = k2 - lg_weight * sigma_next * k2  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
        x_3 = ((sigma_down/sigma)**c3)*x + h*(a31*k1 + a32*k2)
        
        gc.collect()
        torch.cuda.empty_cache()

        for iteration in range(iter_c3):  
            time = sigmas[i] / sigma_max
            
            k2_new = model(x_3, sigma_3 * s_in, **extra_args)
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                hard_light_blend_1 = hard_light_blend(latent_guide, k2_new)
                k2_new = k2_new - lg_weight * sigma_next * k2_new  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
                
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

        noise2 = noise_sampler2(sigma=sigma, sigma_next=sigma_3)
        noise2 = (noise2 - noise2.mean()) / noise2.std()
        x_3 = alpha_ratio_3 * x_3 + noise2 * s_noise2[i] * su_3



        k3 = model(x_3, sigma_3 * s_in, **extra_args)

        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            hard_light_blend_1 = hard_light_blend(latent_guide, k3)
            k3 = k3 - lg_weight * sigma_next * k3  + (lg_weight * sigma_next * hard_light_blend_1 * mask)

        x_next =  ((sigma_down/sigma))*x + h*(b1*k1 + b2*k2 + b3*k3)
        
        gc.collect()
        torch.cuda.empty_cache()

        for iteration in range(iter):  
            time = sigmas[i] / sigma_max

            k3_new = model(x_next, sigma_down * s_in, **extra_args)
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                hard_light_blend_1 = hard_light_blend(latent_guide, k3_new)
                k3_new = k3_new - lg_weight * sigma_next * k3_new  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
                
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
    alpha=None, iter=3, tol=1e-5, latent_guide=None, latent_guide_weight=0.0, latent_guide_weights=None, mask=None, loop_weight=0.0):
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
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
        
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    x_reverse_new = None

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        sigma_up, sigma_down, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta, eta_var, noise_mode)

        sigma_next = sigma_down
        denoised = model(x, sigma * s_in, **extra_args)
        
        if latent_guide is not None:
            lg_weight = latent_guide_weights[i] * sigma
            hard_light_blend_1 = hard_light_blend(latent_guide, denoised)
            denoised = denoised - lg_weight * sigma_next * denoised  + (lg_weight * sigma_next * hard_light_blend_1 * mask)

        x_next = (sigma_next/sigma) * x + (1 - sigma_next/sigma) * denoised 
        denoised_next = denoised

        for iteration in range(iter):  
            denoised_next = model(x_next, sigma_next * s_in, **extra_args)
            
            if latent_guide is not None:
                lg_weight = latent_guide_weights[i] * sigma
                hard_light_blend_1 = hard_light_blend(latent_guide, denoised_next)
                denoised_next = denoised_next - lg_weight * sigma_next * denoised_next  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
            
            x_new = (sigma_next/sigma) * x + (1 - sigma_next/sigma) * denoised_next

            error = torch.norm(x_new - x_next)
            print(f"Iteration {iteration + 1}, Error: {error.item()}")

            if error < tol:
                print(f"Converged after {iteration + 1} iterations with error {error.item()}")
                x_next = x_new
                break
            
            if reverse_weight > 0.0:
                x_reverse_new = (x_next - (1 - sigma_next/sigma) * denoised_next) / (sigma_next/sigma)
                x = reverse_weight * x_reverse_new + (1-reverse_weight) * x
                
                if loop_weight > 0.0:
                    denoised = model(x, sigma * s_in, **extra_args)
                    if latent_guide is not None:
                        lg_weight = latent_guide_weights[i] * sigma
                        hard_light_blend_1 = hard_light_blend(latent_guide, denoised)
                        denoised = denoised - lg_weight * sigma_next * denoised  + (lg_weight * sigma_next * hard_light_blend_1 * mask)
                    x_new2 = (sigma_next/sigma) * x + (1 - sigma_next/sigma) * denoised 
                    x_new = loop_weight * x_new2 + (1-loop_weight) * x_new
                
            x_next = x_new
            if loop_weight == 0.0:
                denoised = denoised_next
            

        x = x_next
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = alpha_ratio * x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_next})

        gc.collect()
        torch.cuda.empty_cache()

    return x






@torch.no_grad()
def get_sigma_up_SA_solver(sigma, sigma_next): 
    sigma_up = sigma_next * torch.sqrt(1 - torch.exp(sigma_next**2 - sigma**2))
    return sigma_up

@torch.no_grad()
def get_common_integral_SA_solver(sigma, sigma_next):
    common_integral = torch.exp( (1/2) * (sigma_next**2 - sigma**2) )
    return common_integral

@torch.no_grad()
def get_special_integral_SA_solver(sigma, sigma_next):
    special_integral = ((1-sigma_next)/sigma_next - (1-sigma)/sigma)
    return special_integral

@torch.no_grad()
def sample_corona(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., eta_var=1., s_noise=1., 
    noise_sampler=None, noise_sampler_type="gaussian", noise_mode="hard",reverse_weight=0.0, k=1.0, scale=0.1, 
    alpha=None, iter=3, tol=1e-5, latent_guide=None, latent_guide_weight=0.0, latent_guide_weights=None, mask=None):
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
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
        
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        h = t_next - t
        
        noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = (noise - noise.mean()) / noise.std()
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        sigma_up = get_sigma_up_SA_solver(sigma, sigma_next)
        common_integral = get_common_integral_SA_solver(sigma, sigma_next)
        special_integral = get_special_integral_SA_solver(sigma, sigma_next)
        x_next = (sigma_next / sigma) * common_integral * x   +  s_noise * sigma_next * common_integral * special_integral * denoised   +   sigma_up * noise * eta
        
        x = x_next
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

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

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None # step size

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        sigma_up, sigma_down, alpha_ratio = get_RF_step(sigma, sigma_next, eta)
        sigma_ratio = (sigma_down - sigma) / (sigma_next - sigma)
        
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
        sigma_up, sigma_down, alpha_ratio = get_RF_step(sigmas[i], sigmas[i+1], eta)
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i] / 2.0)
        vel = denoised
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
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
            sigma_up, sigma_down, alpha_ratio = get_RF_step(sigma, sigma_next, etas[i], noise_scale)
        elif noise_mode == "softer":
            sigma_up, sigma_down, alpha_ratio = get_RF_step_traditional(sigma, sigma_next, etas[i], noise_scale)
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

        sigma_ratio = (sigma_down - sigma) / (sigma_next - sigma) #sigma_minus_full / sigma_minus_orig... alternative to using sigma_down in the equations directly

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
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            
            vel = denoised = momentum_func(denoised, vel, sigmas[i], -momentums[i]/2.0)
                
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigmas[i + 1] == 0:
                d = to_d(x, sigmas[i], denoised)
                dt = sigmas[i + 1] - sigmas[i]
                x = x + d * dt
            else:
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = t_next - t
                s = t + h * 0.5
                fac = 1 / (2 * 0.5)
                
                sigma_s = sigma_fn(s)

                if sigma_s.isnan():
                    sigma_s = 0.9999

                # Step 1
                sd, su, alpha_ratio = None, None, None
                if noise_mode == "soft": 
                    su, sd, alpha_ratio = get_down_step(sigmas[i], sigma_s, eta1, noise_scale)
                elif noise_mode == "softer":
                    su, sd, alpha_ratio = get_RF_step_traditional(sigmas[i], sigma_s, eta1, noise_scale)
                elif noise_mode == "hard":
                    su = sigmas[i] * eta1
                    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                        su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta1)
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
                if noise_mode == "soft": 
                    su, sd, alpha_ratio = get_down_step(sigmas[i], sigmas[i+1], eta2, noise_scale)
                elif noise_mode == "softer":
                    su, sd, alpha_ratio = get_RF_step_traditional(sigmas[i], sigmas[i+1], eta2, noise_scale)
                elif noise_mode == "hard":
                    su = sigmas[i] * eta2
                    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                        su, sd, alpha_ratio = get_ancestral_step_RF(sigmas[i+1], eta2)
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


from .sampler_rk import phi


@torch.no_grad()
def sample_noise_inversion_rev(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, c2=0.5, c3=1.0, gamma=0.25, eta_value=0.5, eta=0.0, eta_var=0.0, 
                               noise_mode="hard", eta_values=None, t_is=None, etas=None, s_noises=None, alpha=-1.0, k=1.0, scale=0.1, order=2, 
                               cfgpp=0.0, latent_guide=None, latent_guide_weight=1.0, noise_sampler_type="brownian", sde_seed=-1.0):
    
    #model_sampling = model.get_model_object("model_sampling")
    sigma_min_model=model.inner_model.inner_model.model_sampling.sigma_min 
    sigma_max_model=model.inner_model.inner_model.model_sampling.sigma_max 
    sigmax = sigma_max_model
    
    temp = [0]
    temp[0] = torch.full_like(x, 0.0)
    if cfgpp != 0.0:
        def post_cfg_function(args):
            temp[0] = args["uncond_denoised"]
            return args["denoised"]
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    if sde_seed < 0:
        seed = torch.initial_seed() + 1
    else:
        seed = sde_seed
    sigmas = sigmas.clone()
    
    if sigmas[0] == 0.0:      #remove padding used to avoid need for model patch with noise inversion
        sigmas = sigmas[1:]
    if sigmas[-1] == 0.0:
        sigmas = sigmas[:-1]
        
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        sigma_max = torch.full_like(sigma_max, sigmax) # 1.0)
        sigma_min = torch.full_like(sigma_min, min(sigma_min.item(), 0.00001))

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigmax) #sigma_max)
    if noise_sampler_type == "fractal":
        noise_sampler.alpha = alpha
        noise_sampler.k = k
        noise_sampler.scale = scale

    t_fn = lambda sigma: sigma.log().neg()
    sigma_fn = lambda t: t.neg().exp() 

    if t_is is None:
        if sigmas[1] > sigmas[0]:
            #t_is = 1 - sigmas
            t_is = sigmas - sigmax # 1       ########### FOR FLIPPING SIGNS! equal to -1 * (commented out value for t_is)
            #t_is = torch.clamp(t_is, min=-1.0, max=-0.001)
            t_is = torch.clamp(t_is, min=-sigmax, max=-0.001)
        else:
            t_is = sigmas
            #t_is = torch.clamp(t_is, min=0.001, max=1.0)
            t_is = torch.clamp(t_is, min=0.001, max=sigmax)
    
    
    if sigmas[1] > sigmas[2]:
        sample_rev = True
    else:
        sample_rev = False

    if latent_guide is None:
        noise = noise_sampler(sigma=sigma_max, sigma_next=sigma_min)
        y0 = (noise - noise.mean()) / noise.std()
    else: #REV MODE, initialize image guide
        y0 = latent_guide.clone().to(x.device)
    
        
    for i in trange(len(sigmas)-1, disable=disable):            
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        if eta > 0.0 or eta_var > 0.0:
            if sigma_next == sigmax: #1.0:
                alpha_ratio = 1.0
                sigma_up = 0.0
                sigma_down = sigma_next   
            else:
                sigma_up, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode)
        else:
            alpha_ratio = 1.0
            sigma_up = 0.0
            sigma_down = sigma_next

        t_down, t = t_fn(sigma_down), t_fn(sigma)
        h = t_down - t
        
        #h_inv = t_fn(sigma_down - 1) - t_fn(sigma - 1)
        
        #sigma_2 = (sigma_down/sigma)**c2
        #sigma_3 = (sigma_down/sigma)**c3
        s2 = t + h * c2
        s3 = t + h * c3
        sigma_2 = sigma_fn(s2)
        sigma_3 = sigma_fn(s3)
        dt_2 = sigma_2 - sigma
        dt_3 = sigma_3 - sigma
        dt = sigma_down - sigma
            
        etz = eta_values[i]
        sds = sigma_down/sigma
        #sdsm1 = ((sigma_down-1) / (sigma-1))
        sdsm1 = ((sigma_down-sigmax) / (sigma-sigmax))
        
        if sample_rev == True:
            if sigma_down < 0.001 and order == 3:
                order = 2
                print("Dropping to 2nd order step to avoid numerical instability.")
        
        if sample_rev == False:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            x = alpha_ratio * x + noise * s_noise * sigma_up
        
        #denoised = model(x, sigma * s_in, **extra_args)
        if order == 1:
            denoised = model(x, sigma * s_in, **extra_args)
            eps = (x - denoised) / sigma
            if sample_rev == True:
                x_next = x   +   (1 - eta_values[i]) * eps * dt   +   eta_values[i] * ((y0 - x) / t_is[i]) * (dt**2)**(1/2) 
            else:
                noise = noise_sampler(sigma=sigmax, sigma_next=sigma_min) 
                y0 = (noise - noise.mean()) / noise.std()
            
                #x_next = x   +   (1 - eta_values[i]) * eps * dt   +   eta_values[i] * ((x - y0) / t_is[i]) * dt    #WORKS INVERTIBLE ti = sigma - 1
                
                #_next = x   +     (1 - eta_values[i]) * (  (1 - sigma_down/sigma) * (denoised - x) )      +        eta_values[i] * ((x - y0) / t_is[i]) * dt 
                
                #x_next = x   +     (sigmax - eta_values[i]) * (  (sigmax - sigma_down/sigma) * (denoised - x) )      +        eta_values[i] * ((x - y0) / t_is[i]) * (sigma_down - sigma)

                x_next = (1-etz) * sds * x + etz * sdsm1 * x    +    (1-etz)*(sigmax-sds)*denoised      +     etz * (sigmax-sdsm1) * y0
                
                #x_next = x   +   (1 - eta_values[i]) * (  (1 - sigma_down/sigma) * (denoised - x) )   +   eta_values[i] * (  (1 - sigma_down/t_is[i]) * (x - y0) ) 
                #x = x - (1-eta_values[i]) * (1 - sigma_next/sigma) * x    +   (1-eta_values[i])*(1-sigma_next/sigma)*denoised    -   eta_values[i] * sigma_next * (y0 - x) ** 2
        
        if order == 2:
            a2_1 = c2 * phi(1, -h*c2)
            b1 =        phi(1, -h) - phi(2, -h)/c2
            b2 =        phi(2, -h)/c2
            if sample_rev == True:

                k1 = model(x, sigma * s_in, **extra_args)
                k1u = temp[0]
                cfgpp_term = cfgpp * h * ( (a2_1*k1) - (a2_1*k1u))
                #x_2 = x   +   (1 - eta_values[i]) * eps_2 * dt_2   +   eta_values[i] * ((y0 - x) / (sigma_2)) * (dt_2**2)**(1/2)     #this def didn't work so hot with the forward direction
                #x_2 = ((sigma_down/sigma)**c2)*x + h*(a2_1*k1)
                x_2 = ((sigma_down/sigma)**c2)*(x + cfgpp_term) + (1-eta_values[i]) * h*(a2_1*k1)     +   eta_values[i] * h * (a2_1*y0) 
                
                #x_2 = ((sigma_down/sigma)**c2)*x + (1-eta_values[i]) * h*(a2_1*k1)     +    eta_values[i] * h * (y0 - x) / 2 #* (b1 + b2)# / 2
                
                #dt2 = ((sigma_next/sigma)**c2) - sigma
                #x_2 = x + dt2 * (a2_1*k1)
                k2 = model(x_2, sigma_fn(t + h*c2) * s_in, **extra_args)
                k2u = temp[0]
                cfgpp_term = cfgpp * h * ( (b1*k1 + b2*k2) - (b1*k1u + b2*k2u))
                #x_next = (sigma_down/sigma) * x + h*(b1*k1 + b2*k2)
                x_next = (sigma_down/sigma) * (x + cfgpp_term)   +    (1-eta_values[i]) * h*(b1*k1 + b2*k2)    +   eta_values[i] * h * (b1*y0 + b2*y0) 
                
                #x_next = (sigma_down/sigma) * x +    (1-eta_values[i]) * h*(b1*k1 + b2*k2)     +    eta_values[i] * h * (y0 - x) / 2 #* (b1 + b2)# / 2
                
                
                denoised1_2 = (b1*k1 + b2*k2) / (b1 + b2)
                denoised = denoised1_2
                
                eps_next = (x - denoised) / sigma
                #x_next = x   +   (1 - eta_values[i]) * eps_next * dt   +   eta_values[i] * ((y0 - x) / t_is[i]) * (dt**2)**(1/2)  
                #x = x_next
            elif sample_rev == False: #forward mode  
                noise = noise_sampler(sigma=sigmax, sigma_next=sigma_min) 
                y0 = (noise - noise.mean()) / noise.std()              
                k1 = model(x, sigma * s_in, **extra_args)
                k1u = temp[0]
                cfgpp_term = cfgpp * h * ( (a2_1*k1) - (a2_1*k1u))
                #x_2 = ((sigma_down/sigma)**c2)*x + h*(a2_1*k1)
                
                x_2 = ((1-etz) * (sigma_2/sigma)  +  etz * (((sigma_2-sigmax)/sigmax)/((sigma-sigmax)/sigmax)) ) * (x + cfgpp_term)     +     (1-etz)* h*(a2_1*k1)      +     etz * ((sigmax-(((sigma_2-sigmax)/sigmax)/((sigma-sigmax)/sigmax)))/sigmax) * y0
                
                #x_next = ((1-etz) * sds  +  etz * sdsm1) * x     +     (1-etz)*(1-sds) * denoised      +     etz * (1-sdsm1) * y0
                
                #x_2 = ((sigma_down/sigma)**c2)*x + (1-eta_values[i]) * h*(a2_1*k1)     +   eta_values[i] * h * (a2_1*y0) 
                k2 = model(x_2, sigma_fn(t + h*c2) * s_in, **extra_args)
                k2u = temp[0]
                cfgpp_term = cfgpp * h * ( (b1*k1 + b2*k2) - (b1*k1u + b2*k2u))
                #x_next = (sigma_down/sigma) * x   +    (1-eta_values[i]) * h*(b1*k1 + b2*k2)    +   eta_values[i] * h * (b1*y0 + b2*y0) 
                #x_next = ((1-etz) * sds  +  etz * sdsm1) * x     +     (1-etz)*  h*(b1*k1 + b2*k2)      +     etz * h_inv * (b1*y0 + b2*y0) 
                
                denoised1_2 = (b1*k1 + b2*k2) / (b1 + b2)
                denoised = denoised1_2

                #x_next = ((1-etz) * sds  +  etz * sdsm1) * x     +     (1-etz)*(1-sds) * denoised      +     etz * (1-sdsm1) * y0
                #x_next = ((1-etz) * sds  +  etz * sdsm1) * (x + cfgpp_term)     +     (1-etz)* h*(b1*k1 + b2*k2)      +     etz * (1-sdsm1) * y0
                x_next = ((1-etz) * (sigma_down/sigma)  +  etz * (((sigma_down-sigmax)/sigmax)/((sigma-sigmax)/sigmax)) ) * (x + cfgpp_term)     +     (1-etz)* h*(b1*k1 + b2*k2)      +     etz * ((sigmax-(((sigma_down-sigmax)/sigmax)/((sigma-sigmax)/sigmax)))/sigmax) * y0

                
            
        if order == 3:
            gamma = (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = gamma * c2 * phi(2, -h*c2) + (c3 ** 2 / c2) * phi(2, -h*c3) #phi_2_c3_h  # a32 from k2 to k3
            a3_1 = c3 * phi(1, -h*c3) - a3_2 # a31 from k1 to k3
            b3 = (1 / (gamma * c2 + c3)) * phi(2, -h)      
            b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
            b1 = phi(1, -h) - b2 - b3 
            if sample_rev == True:

                k1 = model(x, sigma * s_in, **extra_args)
                k1u = temp[0]
                cfgpp_term = cfgpp * h * ( (a2_1*k1) - (a2_1*k1u))
                #x_2 = ((sigma_down/sigma)**c2)*x + h*(a2_1*k1)      
                x_2 = ((sigma_down/sigma)**c2) * (x + cfgpp_term) + (1-eta_values[i]) * h*(a2_1*k1)   +   eta_values[i] * h*(a2_1*y0)
                
                k2 = model(x_2, sigma_fn(t + h*c2) * s_in, **extra_args)
                k2u = temp[0]
                cfgpp_term = cfgpp * h * ( (a3_1*k1 + a3_2*k2) - (a3_1*k1u + a3_2*k2u))
                #x_3 = ((sigma_down/sigma)**c3)*x_2 + h*(a3_1*k1 + a3_2*k2)        
                x_3 = ((sigma_down/sigma)**c3) * (x + cfgpp_term) + (1-eta_values[i]) * h*(a3_1*k1 + a3_2*k2)   + eta_values[i] * h*(a3_1*y0 + a3_2*y0)
            
                k3 = model(x_3, sigma_fn(t + h*c3) * s_in, **extra_args)
                k3u = temp[0]
                cfgpp_term = cfgpp * h * ( (b1*k1 + b2*k2 + b3*k3) - (b1*k1u + b2*k2u + b3*k3u))
                #x_next = ((sigma_down/sigma))*x + h*(b1*k1 + b2*k2 + b3*k3)      
                x_next = (sigma_down/sigma) * (x + cfgpp_term)   +   (1-eta_values[i]) * h*(b1*k1 + b2*k2 + b3*k3)   +   eta_values[i] * h * (b1*y0 + b2*y0 + b3*y0)   
                
                denoised = (b1*k1 + b2*k2 + b3*k3) / (b1 + b2 + b3)
            elif sample_rev == False: #forward mode                
                noise = noise_sampler(sigma=sigmax, sigma_next=sigma_min) 
                y0 = (noise - noise.mean()) / noise.std()
                k1 = model(x, sigma * s_in, **extra_args)
                k1u = temp[0]
                cfgpp_term = cfgpp * h * ( (a2_1*k1) - (a2_1*k1u))
                #x_2 = ((sigma_down/sigma)**c2)*x + h*(a2_1*k1)      
                x_2 = ((1-etz) * (sigma_2/sigma)  +  etz * (((sigma_2-sigmax)/sigmax)/((sigma-sigmax)/sigmax)) ) * (x + cfgpp_term)     +     (1-etz)* h*(a2_1*k1)      +     etz * ((sigmax-(((sigma_2-sigmax)/sigmax)/((sigma-sigmax)/sigmax)))/sigmax)  * y0
                #x_2 = ((1-etz) * (sigma_2/sigma)  +  etz * ((sigma_2-1)/(sigma-1)) ) * x     +     (1-etz)* (1-(sigma_2/sigma)) * k1      +     etz * (1-((sigma_2-1)/(sigma-1))) * y0
                
                k2 = model(x_2, sigma_fn(t + h*c2) * s_in, **extra_args)
                k2u = temp[0]
                cfgpp_term = cfgpp * h * ( (a3_1*k1 + a3_2*k2) - (a3_1*k1u + a3_2*k2u))
                #x_3 = ((sigma_down/sigma)**c3)*x_2 + h*(a3_1*k1 + a3_2*k2)        
                #x_3 = ((sigma_down/sigma)**c3)*x_2 + (1-eta_values[i]) * h*(a3_1*k1 + a3_2*k2)   + eta_values[i] * h*(a3_1*y0 + a3_2*y0)
                #denoised = (a3_1*k1 + a3_2*k2) / (a3_1 + a3_2)
                
                #x_3 = ((1-etz) * (sigma_3/sigma)  +  etz * ((sigma_3-1)/(sigma-1)) ) * x_2     +     (1-etz)* (1-(sigma_3/sigma)) * denoised      +     etz * (1-((sigma_3-1)/(sigma-1))) * y0
                x_3 = ((1-etz) * (sigma_3/sigma)  +  etz * (((sigma_3-sigmax)/sigmax)/((sigma-sigmax)/sigmax)) ) * (x + cfgpp_term)     +     (1-etz)* h*(a3_1*k1 + a3_2*k2)      +     etz * ((sigmax-(((sigma_3-sigmax)/sigmax)/((sigma-sigmax)/sigmax)))/sigmax)  * y0
            
                k3 = model(x_3, sigma_fn(t + h*c3) * s_in, **extra_args)
                k3u = temp[0]
                cfgpp_term = cfgpp * h * ( (b1*k1 + b2*k2 + b3*k3) - (b1*k1u + b2*k2u + b3*k3u))
                #x_next = ((sigma_down/sigma))*x + h*(b1*k1 + b2*k2 + b3*k3)      
                #x_next = (sigma_down/sigma) * x   +   (1-eta_values[i]) * h*(b1*k1 + b2*k2 + b3*k3)   +   eta_values[i] * h * (b1*y0 + b2*y0 + b3*y0)   
                
                x_next = ((1-etz) * (sigma_down/sigma)  +  etz * (((sigma_down-sigmax)/sigmax)/((sigma-sigmax)/sigmax))  ) * (x + cfgpp_term)     +     (1-etz)* h*(b1*k1 + b2*k2 + b3*k3)      +     etz * ((sigmax-(((sigma_down-sigmax)/sigmax)/((sigma-sigmax)/sigmax)))/sigmax) * y0
                #x_next = ((1-etz) * sds  +  etz * sdsm1) * x     +     (1-etz)* h*(b1*k1 + b2*k2 + b3*k3)      +     etz * (1-sdsm1) * y0
                
                denoised = (b1*k1 + b2*k2 + b3*k3) / (b1 + b2 + b3)
                #denoised = k3
                eps = (x - denoised) / sigma
                #x_next = ((1-etz) * sds  +  etz * sdsm1) * x     +     (1-etz)* (1-(sigma_down/sigma)) * denoised      +     etz * (1-sdsm1) * y0
                #x_next = x   +   (1 - eta_values[i]) * eps * dt   +   eta_values[i] * ((y0 - x) / (1-sigma)) * (dt**2)**(1/2)  

                
            
            
        ratio_madness = ((1-etz) * (sigma_down/sigma)  +  etz * (((sigma_down-sigmax)/sigmax)/((sigma-sigmax)/sigmax))  )     /    (sigma_down/sigma)
        ratio_madness = 1 / ratio_madness
        x = x_next
        if sample_rev == True:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            x = alpha_ratio * x + noise * s_noise * sigma_up * ratio_madness
        #x = x - (1-eta_values[i]) * (1 - sigma_next/sigma) * x    +   (1-eta_values[i])*(1-sigma_next/sigma)*denoised    -   eta_values[i] * sigma_next * (y0 - x) ** 2

        if callback is not None:
            callback({'x': x, 'denoised': denoised, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        #gc.collect()
        #torch.cuda.empty_cache()
        
    return x



#from .refined_exp_solver import sample_refined_exp_s, sample_refined_exp_s_advanced
from .sampler_rk import sample_rk, sample_rk_multistep, sample_rk_test

extra_samplers = {
    "RES_implicit_advanced_RF_PC": sample_RES_implicit_advanced_RF_PC, 
    "RES_implicit_advanced_RF_PC_3rd_order": sample_RES_implicit_advanced_RF_PC_3rd_order,
    "SDE_implicit_advanced_RF": sample_SDE_implicit_advanced_RF,
    "corona": sample_corona,
    "noise_inversion_rev": sample_noise_inversion_rev,

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

    "rk":  sample_rk,
    "rk_test": sample_rk_test,
    "rk_multistep": sample_rk_multistep,
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
