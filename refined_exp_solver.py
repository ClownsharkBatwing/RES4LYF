"""Algorithm 1 "RES Second order Single Update Step with c2"
https://arxiv.org/abs/2308.02157"""

import torch
from torch import no_grad, FloatTensor
from tqdm import tqdm
from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple
import math
import gc
from .noise_classes import *

from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher


class DenoiserModel(Protocol):
  def __call__(self, x: FloatTensor, t: FloatTensor, *args, **kwargs) -> FloatTensor: ...

class RefinedExpCallbackPayload(TypedDict):
  x: FloatTensor
  i: int
  sigma: FloatTensor
  sigma_hat: FloatTensor
  

class RefinedExpCallback(Protocol):
  def __call__(self, payload: RefinedExpCallbackPayload) -> None: ...

class NoiseSampler(Protocol):
  def __call__(self, x: FloatTensor) -> FloatTensor: ...

class StepOutput(NamedTuple):
  x_next: FloatTensor
  denoised: FloatTensor
  denoised2: FloatTensor
  vel: FloatTensor
  vel_2: FloatTensor

"""def get_RF_step(sigma, sigma_next, eta):
    downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_ip1 = 1 - sigma_next
    alpha_down = 1 - sigma_down
    alpha_ratio = alpha_ip1 / alpha_down
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2).abs() ** 0.5
    return sigma_down, sigma_up, alpha_ratio """
  
"""def get_RF_step(sigma, sigma_next, eta, alpha_ratio=None):
    #Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    #and a mixing ratio (alpha_ratio) for scaling the latent during noise addition.
    down_ratio = (1 - eta) + eta * (sigma_next / sigma)   ###NOTE THE CHANGE HERE WITH SQUARING THE SHIT
    sigma_down = sigma_next * down_ratio

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    else:
      sigma_down = - (1-sigma_next-alpha_ratio)/alpha_ratio
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_down, sigma_up, alpha_ratio"""
  
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

def get_RF_step_traditional(sigma, sigma_next, eta, scale=0.0,  alpha_ratio=None):
    # uses math similar to what is used for the get ancestral step code in comfyui. WORKS!
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)   #maybe this is the most appropriate target for scaling?
    #sigma_down = sigma_next * down_ratio
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_down, sigma_up, alpha_ratio

"""def get_RF_step(sigma, sigma_next, eta, alpha_ratio=None):
    #Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    #and a mixing ratio (alpha_ratio) for scaling the latent during noise addition.
    down_ratio = (1 - eta) + eta * (sigma_next / sigma)   ###NOTE THE CHANGE HERE WITH SQUARING THE SHIT
    sigma_down = sigma_next * down_ratio

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_down, sigma_up, alpha_ratio"""


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
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    return sigma_up, sigma_down, alpha_ratio

def _gamma(n: int,) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(s: int, x: float, gamma_s: Optional[int] = None) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
  if s is a positive integer,
  Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
  """
  if gamma_s is None:
    gamma_s = _gamma(s)

  sum_: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    quotient: float = numerator/denom
    sum_ += quotient
  incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
  return incomplete_gamma_

# by Katherine Crowson
def _phi_1(neg_h: FloatTensor):
  return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

# by Katherine Crowson
def _phi_2(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)

# by Katherine Crowson
def _phi_3(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h - neg_h**2 / 2) / neg_h**3, nan=1 / 6)

def _phi(neg_h: float, j: int,):
  """
  For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = _gamma(j)
  incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)
  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)
  return phi_

class RESDECoeffsSecondOrder(NamedTuple):
  a2_1: float
  b1: float
  b2: float

def _de_second_order(h: float, c2: float, simple_phi_calc = False,) -> RESDECoeffsSecondOrder:
  """
  Table 3
  https://arxiv.org/abs/2308.02157
  ϕi,j := ϕi,j(-h) = ϕi(-cj*h)
  a2_1 = c2ϕ1,2
       = c2ϕ1(-c2*h)
  b1 = ϕ1 - ϕ2/c2
  """
  if simple_phi_calc:
    # Kat computed simpler expressions for phi for cases j={1,2,3}
    a2_1: float = c2 * _phi_1(-c2*h)
    phi1: float = _phi_1(-h)
    phi2: float = _phi_2(-h)
  else:
    # I computed general solution instead.
    # they're close, but there are slight differences. not sure which would be more prone to numerical error.
    a2_1: float = c2 * _phi(j=1, neg_h=-c2*h)
    phi1: float = _phi(j=1, neg_h=-h)
    phi2: float = _phi(j=2, neg_h=-h)
  phi2_c2: float = phi2/c2
  b1: float = phi1 - phi2_c2
  b2: float = phi2_c2
  
  return RESDECoeffsSecondOrder(a2_1=a2_1, b1=b1, b2=b2,)  
  
  
# by Katherine Crowson
def _phi_1(neg_h: FloatTensor):
  return torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)

# by Katherine Crowson
def _phi_2(neg_h: FloatTensor):
  return torch.nan_to_num((torch.expm1(neg_h) - neg_h) / neg_h**2, nan=0.5)
  

def _refined_exp_sosu_step(model, x, sigma, sigma_next, c2 = 0.5,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  eulers_mom = 0.0,
  cfgpp = 0.0,
) -> StepOutput:

  """Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigma (`FloatTensor`): timestep to denoise
    sigma_next (`FloatTensor`): timestep+1 to denoise
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences."""
  if cfgpp != 0.0:
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  sigma_fn = lambda t: t.neg().exp()
  t_fn = lambda sigma: sigma.log().neg()
  lam, lam_next = t_fn(sigma), t_fn(sigma_next)
  
  s_in = x.new_ones([x.shape[0]])
  h = lam_next - lam
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  denoised = model(x, sigma * s_in, **extra_args)
  
  if pbar is not None:
    pbar.update(0.5)

  c2_h = c2*h

  diff_2 = vel_2 = momentum_func(a2_1*h*denoised, vel_2, time)

  x_2 = math.exp(-c2_h)*x + diff_2
  if cfgpp == False:
    x_2 = math.exp(-c2_h)*x + diff_2
  else:
    x_2 = math.exp(-c2_h) * (x + cfgpp*denoised - cfgpp*temp[0]) + diff_2
  lam_2 = lam + c2_h
  sigma_2 = sigma_fn(lam_2)

  denoised2 = model(x_2, sigma_2 * s_in, **extra_args)

  if pbar is not None:
    pbar.update(0.5)

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)

  if cfgpp == False:
    x_next = math.exp(-h)*x + diff
  else:
    x_next = math.exp(-h) * (x + cfgpp*denoised - cfgpp*temp[0]) + diff

  return StepOutput(x_next=x_next, denoised=denoised, denoised2=denoised2, vel=vel, vel_2=vel_2,)


def _refined_exp_sosu_step_RF_hard(model, x, sigma, sigma_next, sigma_next2, c2 = 0.5, eta=1.0, noise_sampler=None, noise_mode="hard", ancestral_noise=True, s_noise=1.0, #COMFY 
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  eulers_mom = 0.0,
  cfgpp = 0.0,
) -> StepOutput:

  """Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigma (`FloatTensor`): timestep to denoise
    sigma_next (`FloatTensor`): timestep+1 to denoise
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences."""
  if cfgpp != 0.0:
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  sigma_fn = lambda t: t.neg().exp()
  t_fn = lambda sigma: sigma.log().neg()
  h_fn = lambda sigma, sigma_next: t_fn(sigma_next) - t_fn(sigma)
  sigma_s_fn = lambda sigma, sigma_next, c2: torch.nan_to_num(sigma_fn(t_fn(sigma) + c2 * h_fn(sigma, sigma_next)), 0.9999) #if sigma == 1.0, and nan results with RF sigma_fn_RF(), set to 0.9999

  sigma_fn_RF = lambda t: (t.exp() + 1) ** -1
  t_fn_RF = lambda sigma: ((1-sigma)/sigma).log()
  h_fn_RF = lambda sigma, sigma_next: t_fn_RF(sigma_next) - t_fn_RF(sigma)
  sigma_s_fn_RF = lambda sigma, sigma_next, c2: torch.nan_to_num(sigma_fn_RF(t_fn_RF(sigma) + c2 * h_fn_RF(sigma, sigma_next)), 0.9999) #if sigma == 1.0, and nan results with RF sigma_fn_RF(), set to 0.9999

  s_in = x.new_ones([x.shape[0]])
  
  su, sd, alpha_ratio = get_ancestral_step_RF(sigma_next, eta)

  if ancestral_noise == False: # add noise before first step, results in a very clean image, great for some styles but looks fake with photography
    x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise * su
  
  denoised = model(x, sigma * s_in, **extra_args)

  h = h_fn(sigma, sd) 
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  #sigma_s = sigma_s_fn(sigma, sd, c2)
  sigma_s = sigma_s_fn_RF(sigma, sd, c2)

  if pbar is not None:
    pbar.update(0.5)

  diff_2 = vel_2 = momentum_func(h*a2_1*denoised, vel_2, time)
  x_2 = ((sd/sigma)**c2)*x + diff_2 
  
  denoised2 = model(x_2, sigma_s * s_in, **extra_args)

  if pbar is not None:
    pbar.update(0.5)

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)

  x_next =  (sd/sigma) * x + diff
  
  if ancestral_noise == True and sigma_next > 0.00001: # very good for photography styles
    su_2, sd_2, alpha_ratio_2 = get_ancestral_step_RF(sigma_next, eta)
    x_next = alpha_ratio_2 * x_next + noise_sampler(sigma=sigma_s, sigma_next=sigma_next) * s_noise * su_2
    
  if sigma_next2 == 0.0:
    sigma_tiny = torch.tensor(min(0.00001, (sigma_next**2).item()), dtype=sigma_next.dtype).to(sigma_next.device)
    print("denoise from: ", sigma_next.item(), "   denoise_to: ", sigma_tiny.item())
    gc.collect(); torch.cuda.empty_cache()
    return _refined_exp_sosu_step_RF_hard(model, x_next, sigma_next, sigma_tiny, sigma_tiny, c2=c2, eta=eta, noise_sampler=noise_sampler, s_noise=s_noise,
                                      extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, vel=vel, vel_2=vel_2, time=time, eulers_mom=eulers_mom, cfgpp=cfgpp) 

  return StepOutput(x_next=x_next, denoised=denoised, denoised2=denoised2, vel=vel, vel_2=vel_2,)


def _refined_exp_sosu_step_RF_hard_deis(model, x, sigma, sigma_next, sigma_next2, c2 = 0.5, eta=1.0, noise_sampler=None, noise_mode="hard", ancestral_noise=True, s_noise=1.0, #COMFY 
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  eulers_mom = 0.0,
  cfgpp = 0.0,
) -> StepOutput:

  """Algorithm 1 "RES Second order Single Update Step with c2"
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigma (`FloatTensor`): timestep to denoise
    sigma_next (`FloatTensor`): timestep+1 to denoise
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences."""
  if cfgpp != 0.0:
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  sigma_fn = lambda t: t.neg().exp()
  t_fn = lambda sigma: sigma.log().neg()
  h_fn = lambda sigma, sigma_next: t_fn(sigma_next) - t_fn(sigma)
  sigma_s_fn = lambda sigma, sigma_next, c2: torch.nan_to_num(sigma_fn(t_fn(sigma) + c2 * h_fn(sigma, sigma_next)), 0.9999) #if sigma == 1.0, and nan results with RF sigma_fn_RF(), set to 0.9999

  sigma_fn_RF = lambda t: (t.exp() + 1) ** -1
  t_fn_RF = lambda sigma: ((1-sigma)/sigma).log()
  h_fn_RF = lambda sigma, sigma_next: t_fn_RF(sigma_next) - t_fn_RF(sigma)
  sigma_s_fn_RF = lambda sigma, sigma_next, c2: torch.nan_to_num(sigma_fn_RF(t_fn_RF(sigma) + c2 * h_fn_RF(sigma, sigma_next)), 0.9999) #if sigma == 1.0, and nan results with RF sigma_fn_RF(), set to 0.9999

  s_in = x.new_ones([x.shape[0]])
   
  alpha_ratio = 1.0
  if   noise_mode == "soft":
    sd, su, alpha_ratio = get_RF_step(sigma, sigma_next, eta)
  elif noise_mode == "softer":
    sd, su, alpha_ratio = get_RF_step_traditional(sigma, sigma_next, eta)
  elif noise_mode == "hard":
    su, sd, alpha_ratio = get_ancestral_step_RF(sigma_next, eta)

  if ancestral_noise == False: # add noise before first step, results in a very clean image, great for some styles but looks fake with photography
    x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise * su
  
  denoised = model(x, sigma * s_in, **extra_args)

  h = h_fn(sigma, sd) 
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  #sigma_s = sigma_s_fn(sigma, sd, c2)
  sigma_s = sigma_s_fn_RF(sigma, sd, c2)

  if pbar is not None:
    pbar.update(0.5)

  diff_2 = vel_2 = momentum_func(h*a2_1*denoised, vel_2, time)
  x_2 = ((sd/sigma)**c2)*x + diff_2 
  
  denoised2 = model(x_2, sigma_s * s_in, **extra_args)

  if pbar is not None:
    pbar.update(0.5)

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  denoised1_2 = momentum_func((b1*denoised + b2*denoised2), vel, time) / (b1 + b2)
  #print(b1, b2, h)

  x_next =  (sd/sigma) * x + diff
  
  if ancestral_noise == True and sigma_next > 0.00001: # very good for photography styles
    if   noise_mode == "soft":
      sd_2, su_2, alpha_ratio_2 = get_RF_step(sigma, sigma_next, eta)
    elif noise_mode == "softer":
      sd_2, su_2, alpha_ratio_2 = get_RF_step_traditional(sigma, sigma_next, eta)
    elif noise_mode == "hard":
      su_2, sd_2, alpha_ratio_2 = get_ancestral_step_RF(sigma_next, eta)
    #su_2, sd_2, alpha_ratio_2 = get_ancestral_step_RF(sigma_next, eta)
    x_next = alpha_ratio_2 * x_next + noise_sampler(sigma=sigma_s, sigma_next=sigma_next) * s_noise * su_2
    
  if sigma_next2 == 0.0:
    sigma_tiny = torch.tensor(min(0.00001, (sigma_next**2).item()), dtype=sigma_next.dtype).to(sigma_next.device)
    print("denoise from: ", sigma_next.item(), "   denoise_to: ", sigma_tiny.item())
    gc.collect(); torch.cuda.empty_cache()
    return _refined_exp_sosu_step_RF_hard_deis(model, x_next, sigma_next, sigma_tiny, sigma_tiny, c2=c2, eta=eta, noise_sampler=noise_sampler, s_noise=s_noise,
                                      extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, vel=vel, vel_2=vel_2, time=time, eulers_mom=eulers_mom, cfgpp=cfgpp) 
  return x_next, denoised, denoised2, denoised1_2, vel, vel_2,
  #return StepOutput(x_next=x_next, denoised=denoised, denoised2=denoised2, vel=vel, vel_2=vel_2,)


def _refined_exp_sosu_step_RF(model, x, sigma, sigma_next, sigma_next2, c2 = 0.5, eta=1.0, noise_sampler=None, noise_mode="soft", s_noise=1.0, noise_scale=1.0, ancestral_noise=True, 
                              alpha_ratios=None, #COMFY 08407 PNG
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  eulers_mom = 0.0,
  cfgpp = 0.0,
) -> StepOutput:


  if cfgpp != 0.0:
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  sigma_fn = lambda t: t.neg().exp()
  t_fn = lambda sigma: sigma.log().neg()
  h_fn = lambda sigma, sigma_next: t_fn(sigma_next) - t_fn(sigma)
  sigma_s_fn = lambda sigma, sigma_next, c2: torch.nan_to_num(sigma_fn(t_fn(sigma) + c2 * h_fn(sigma, sigma_next)), 0.9999) #if sigma == 1.0, and nan results with RF sigma_fn_RF(), set to 0.9999

  sigma_fn_RF = lambda t: (t.exp() + 1) ** -1
  t_fn_RF = lambda sigma: ((1-sigma)/sigma).log()
  h_fn_RF = lambda sigma, sigma_next: t_fn_RF(sigma_next) - t_fn_RF(sigma)
  sigma_s_fn_RF = lambda sigma, sigma_next, c2: torch.nan_to_num(sigma_fn_RF(t_fn_RF(sigma) + c2 * h_fn_RF(sigma, sigma_next)), 0.9999) #if sigma == 1.0, and nan results with RF sigma_fn_RF(), set to 0.9999


  s_in = x.new_ones([x.shape[0]])
  
  if   noise_mode == "soft":
    sd, su, alpha_ratio = get_RF_step(sigma, sigma_next, eta, noise_scale, alpha_ratios)
  elif noise_mode == "softer":
    sd, su, alpha_ratio = get_RF_step_traditional(sigma, sigma_next, eta, noise_scale, alpha_ratios)
  if ancestral_noise == False: #add noise before first step, results in a very clean image, great for some styles but looks fake with photography

    #x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_s) * su #####################
    x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise * su ##################### TEST LINE FOR CFG W/O GUIDANCE
  
  denoised = model(x, sigma * s_in, **extra_args)
    
  h = h_fn(sigma, sd) 
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  #sigma_s = sigma_s_fn(sigma, sd, c2)
  sigma_s = sigma_s_fn_RF(sigma, sd, c2)

  if pbar is not None:
    pbar.update(0.5)

  diff_2 = vel_2 = momentum_func(h*a2_1*denoised, vel_2, time)

  x_2 = ((sd/sigma)**c2)*x + diff_2 
  #x_2 = alpha_ratio_sigma_s * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * su_sigma_s #COMMENTED OUT FOR TEST LINE FOR CFG W/O GUIDANCE
  #x_2 = alpha_ratio * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * su
  
  denoised2 = model(x_2, sigma_s * s_in, **extra_args)

  if pbar is not None:
    pbar.update(0.5)

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)

  x_next =  (sd/sigma) * x + diff
  
  if ancestral_noise == True and sigma_next > 0.00001: #results in leftover noise with low step counts (<= 30), but very good for photography styles
    if   noise_mode == "soft":
      sd_2, su_2, alpha_ratio_2 = get_RF_step(sigma, sigma_next, eta, noise_scale, alpha_ratios)
    elif noise_mode == "softer":
      sd_2, su_2, alpha_ratio_2 = get_RF_step_traditional(sigma, sigma_next, eta, noise_scale, alpha_ratios)
      
    sd_2, su_2, alpha_ratio_2 = get_RF_step(sigma, sigma_next, eta, noise_scale, alpha_ratios)
    x_next = alpha_ratio_2 * x_next + noise_sampler(sigma=sigma_s, sigma_next=sigma_next) * s_noise * su_2
    #x_next = alpha_ratio * x_next + noise_sampler(sigma=sigma, sigma_next=sigma_next) * su
    
  if sigma_next2 == 0.0:
    sigma_tiny = torch.tensor(min(0.00001, (sigma_next**2).item()), dtype=sigma_next.dtype).to(sigma_next.device)
    print("denoise from: ", sigma_next.item(), "   denoise_to: ", sigma_tiny.item())
    gc.collect(); torch.cuda.empty_cache()
    return _refined_exp_sosu_step_RF(model, x_next, sigma_next, sigma_tiny, sigma_tiny, c2=c2, eta=eta, noise_sampler=noise_sampler, s_noise=s_noise,
                                      extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, vel=vel, vel_2=vel_2, time=time, eulers_mom=eulers_mom, cfgpp=cfgpp) 

  return StepOutput(x_next=x_next, denoised=denoised, denoised2=denoised2, vel=vel, vel_2=vel_2,)


@no_grad()
def sample_refined_exp_s_advanced_RF(
  model,
  x,
  sigmas,
  branch_mode,
  branch_depth,
  branch_width,
  guide_1=None,
  guide_2=None,
  guide_mode_1 = 0,
  guide_mode_2 = 0,
  guide_1_channels=None,
  denoise_to_zero: bool = True,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  eta=None,
  s_noises=None,
  momentum=None,
  eulers_mom=None,
  c2=None,
  cfgpp=None,
  offset=None,
  alpha=None,
  latent_guide_1=None,
  latent_guide_2=None,
  noise_sampler: NoiseSampler = torch.randn_like,
  noise_sampler_type=None,
  noise_mode="hard",
  noise_scale=1.0,
  ancestral_noise=True,
  alpha_ratios=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None,
  latent_self_guide_1=False,
  latent_shift_guide_1=False,
): 
  """
  
  Refined Exponential Solver (S).
  Algorithm 2 "RES Single-Step Sampler" with Algorithm 1 second-order step
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigmas (`FloatTensor`): sigmas (ideally an exponential schedule!) e.g. get_sigmas_exponential(n=25, sigma_min=model.sigma_min, sigma_max=model.sigma_max)
    denoise_to_zero (`bool`, *optional*, defaults to `True`): whether to finish with a first-order step down to 0 (rather than stopping at sigma_min). True = fully denoise image. False = match Algorithm 2 in paper
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    callback (`RefinedExpCallback`, *optional*, defaults to `None`): you can supply this callback to see the intermediate denoising results, e.g. to preview each step of the denoising process
    disable (`bool`, *optional*, defaults to `False`): whether to hide `tqdm`'s progress bar animation from being printed
    eta (`FloatTensor`, *optional*, defaults to 0.): degree of stochasticity, η, for each timestep. tensor shape must be broadcastable to 1-dimensional tensor with length `len(sigmas) if denoise_to_zero else len(sigmas)-1`. each element should be from 0 to 1.
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    noise_sampler (`NoiseSampler`, *optional*, defaults to `torch.randn_like`): method used for adding noise
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """

  s_in = x.new_ones([x.shape[0]])
  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)

  dt = None
  vel, vel_2 = None, None
  x_hat = None
  
  x_n        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  x_h        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel_2      = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised2  = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised_  = None
  denoised2_ = None
  denoised2_prev = None
  
  i=0
  with tqdm(disable=disable, total=len(sigmas)-(1 if denoise_to_zero else 2)) as pbar:
    #for i, (sigma, sigma_next) in enumerate(pairwise(sigmas[:-1].split(1))):
    while i < len(sigmas) - 1 and sigmas[i+1] > 0.0:

      sigma = sigmas[i]
      sigma_next = sigmas[i+1]
      time = sigmas[i] / sigma_max

      if 'sigma' not in locals():
        sigma = sigmas[i]

      if latent_noise is not None:
        if latent_noise.size()[0] == 1:
          eps = latent_noise[0]
        else:
          eps = latent_noise[i]
      else:
        if noise_sampler_type == "fractal":
          noise_sampler.alpha = alpha[i]
          noise_sampler.k = k

      sigma_hat = sigma * (1 + eta[i])

      x_n[0][0] = x
      for depth in range(1, branch_depth+1):
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        sigma_hat = sigma * (1 + eta[i])
        
        for m in range(branch_width**(depth-1)):
          for n in range(branch_width):
            idx = m * branch_width + n

            """sigma_hat = torch.clamp(sigma_hat, max=1.0)

            sigma_up = sigmas[i] * eta[i]
            sigma_hat = sigmas[i] * (1 + eta[i])
            sigma_down, alpha_ratio = get_RF_step2(sigma_up, sigmas[i+1])
            x_n[depth-1][m] = alpha_ratio * x_n[depth-1][m]   +   noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noises[i] * sigma_up"""
            x_h[depth][idx] = x_n[depth-1][m]

            """print(sigmas[i], sigmas[i+1], sigma_up, sigma_down, alpha_ratio)
            print(sigmas[i] + sigma_up)
            x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], vel[depth][idx], vel_2[depth][idx] = _refined_exp_sosu_step(model, x_h[depth][idx], sigma_hat, sigma_down, c2=c2[i],
                                                                          extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc,
                                                                          momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time, eulers_mom = eulers_mom[i].item(), cfgpp = cfgpp[i].item()
                                                                          )"""
            if noise_mode == "hard":
              x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], vel[depth][idx], vel_2[depth][idx] = _refined_exp_sosu_step_RF_hard(model, x_h[depth][idx], sigma, sigma_next, sigmas[i+2], c2=c2[i],eta=eta[i], noise_sampler=noise_sampler, s_noise=s_noises[i], noise_mode=noise_mode, ancestral_noise=ancestral_noise,
                                                                            extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc,
                                                                            momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time, eulers_mom = eulers_mom[i].item(), cfgpp = cfgpp[i].item()
                                                                            )
            elif noise_mode == "soft" or noise_mode == "softer":           
              if alpha_ratios == None:
                alpha_ratios_ = None
              else:
                alpha_ratios_ = alpha_ratios[i]                                                                                                                
              x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], vel[depth][idx], vel_2[depth][idx] = _refined_exp_sosu_step_RF(model, x_h[depth][idx], sigma, sigma_next, sigmas[i+2], c2=c2[i],eta=eta[i], noise_sampler=noise_sampler, s_noise=s_noises[i], noise_scale=noise_scale, alpha_ratios=alpha_ratios_,
                                                                            extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc,
                                                                            momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time, eulers_mom = eulers_mom[i].item(), cfgpp = cfgpp[i].item()
                                                                            )
                                                               
            denoised_ = denoised[depth][idx]
            denoised2_ = denoised2[depth][idx]

            gc.collect()
            torch.cuda.empty_cache()
        i += 1
        
      if denoised2_prev is not None:
        x_n[0][0] = denoised2_prev
      x_next, x_hat, denoised2_prev = branch_mode_proc(x_n, x_h, denoised2, latent_guide_2, branch_mode, branch_depth, branch_width)
      
      d = to_d(x_hat, sigma_hat, x_next) #this is the "euler's momma" method. it effectively projects more noise removal from a noised state.
      dt = sigma_next - sigma_hat
      x_next = x_next + eulers_mom[i].item() * d * dt
      
      if callback is not None:
        payload = RefinedExpCallbackPayload(x=x, i=i, sigma=sigma, sigma_hat=sigma_hat, denoised=denoised_, denoised2=denoised2_prev,)         # added updated denoised2_prev that's selected from the same slot as x_next                      
        callback(payload)

      x = x_next - sigma_next*offset[i]
      
      x = guide_mode_proc(x, i, guide_mode_1, guide_mode_2, sigma_next, guide_1, guide_2,  latent_guide_1, latent_guide_2, guide_1_channels)
      
    if denoise_to_zero:
      final_eta = eta[-1]
      eps = noise_sampler(sigma=sigma, sigma_next=sigma_next).double()
      sigma_hat = sigma * (1 + final_eta)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** .5 * eps
      
      s_in = x.new_ones([x.shape[0]])
      x_next = model(x_hat, torch.zeros_like(sigma).to(x_hat.device) * s_in, **extra_args)
      pbar.update()
      x = x_next

  return x






@no_grad()
def sample_refined_exp_s_advanced(
  model,
  x,
  sigmas,
  branch_mode,
  branch_depth,
  branch_width,
  guide_1=None,
  guide_2=None,
  guide_mode_1 = 0,
  guide_mode_2 = 0,
  guide_1_channels=None,
  denoise_to_zero: bool = True,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  eta=None,
  s_noises=None,
  momentum=None,
  eulers_mom=None,
  c2=None,
  cfgpp=None,
  offset=None,
  alpha=None,
  latent_guide_1=None,
  latent_guide_2=None,
  noise_sampler: NoiseSampler = torch.randn_like,
  noise_sampler_type=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None,
  latent_self_guide_1=False,
  latent_shift_guide_1=False,
): 
  """
  
  Refined Exponential Solver (S).
  Algorithm 2 "RES Single-Step Sampler" with Algorithm 1 second-order step
  https://arxiv.org/abs/2308.02157

  Parameters:
    model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
    x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
    sigmas (`FloatTensor`): sigmas (ideally an exponential schedule!) e.g. get_sigmas_exponential(n=25, sigma_min=model.sigma_min, sigma_max=model.sigma_max)
    denoise_to_zero (`bool`, *optional*, defaults to `True`): whether to finish with a first-order step down to 0 (rather than stopping at sigma_min). True = fully denoise image. False = match Algorithm 2 in paper
    extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
    callback (`RefinedExpCallback`, *optional*, defaults to `None`): you can supply this callback to see the intermediate denoising results, e.g. to preview each step of the denoising process
    disable (`bool`, *optional*, defaults to `False`): whether to hide `tqdm`'s progress bar animation from being printed
    eta (`FloatTensor`, *optional*, defaults to 0.): degree of stochasticity, η, for each timestep. tensor shape must be broadcastable to 1-dimensional tensor with length `len(sigmas) if denoise_to_zero else len(sigmas)-1`. each element should be from 0 to 1.
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    noise_sampler (`NoiseSampler`, *optional*, defaults to `torch.randn_like`): method used for adding noise
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """

  s_in = x.new_ones([x.shape[0]])

  #assert sigmas[-1] == 0
  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)

  b, c, h, w = x.shape

  dt = None
  vel, vel_2 = None, None
  x_hat = None
  
  x_n   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  x_h   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel_2 = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised2 = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised_ = None
  denoised2_ = None
  denoised2_prev = None
  
  i=0
  with tqdm(disable=disable, total=len(sigmas)-(1 if denoise_to_zero else 2)) as pbar:
    #for i, (sigma, sigma_next) in enumerate(pairwise(sigmas[:-1].split(1))):
    while i < len(sigmas) - 1 and sigmas[i+1] > 0.0:

      sigma = sigmas[i]
      sigma_next = sigmas[i+1]
      time = sigmas[i] / sigma_max

      if 'sigma' not in locals():
        sigma = sigmas[i]

      if latent_noise is not None:
        if latent_noise.size()[0] == 1:
          eps = latent_noise[0]
        else:
          eps = latent_noise[i]
      else:
        if noise_sampler_type == "fractal":
          noise_sampler.alpha = alpha[i]
          noise_sampler.k = k

      sigma_hat = sigma * (1 + eta[i])

      x_n[0][0] = x
      for depth in range(1, branch_depth+1):
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        sigma_hat = sigma * (1 + eta[i])
        
        for m in range(branch_width**(depth-1)):
          for n in range(branch_width):
            idx = m * branch_width + n
            x_h[depth][idx] = x_n[depth-1][m] + (sigma_hat ** 2 - sigma ** 2).sqrt() * noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noises[i]
            x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], vel[depth][idx], vel_2[depth][idx] = _refined_exp_sosu_step(model, x_h[depth][idx], sigma_hat, sigma_next, c2=c2[i],
                                                                          extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc,
                                                                          momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time, eulers_mom = eulers_mom[i].item(), cfgpp = cfgpp[i].item()
                                                                          )
            denoised_ = denoised[depth][idx]
            denoised2_ = denoised2[depth][idx]
        i += 1
        
      if denoised2_prev is not None:
        x_n[0][0] = denoised2_prev
      x_next, x_hat, denoised2_prev = branch_mode_proc(x_n, x_h, denoised2, latent_guide_2, branch_mode, branch_depth, branch_width)
      
      d = to_d(x_hat, sigma_hat, x_next) #this is the "euler's momma" method. it effectively projects more noise removal from a noised state.
      dt = sigma_next - sigma_hat
      x_next = x_next + eulers_mom[i].item() * d * dt
      
      if callback is not None:
        payload = RefinedExpCallbackPayload(x=x, i=i, sigma=sigma, sigma_hat=sigma_hat, denoised=denoised_, denoised2=denoised2_prev,)         # added updated denoised2_prev that's selected from the same slot as x_next                      
        callback(payload)

      x = x_next - sigma_next*offset[i]
      
      x = guide_mode_proc(x, i, guide_mode_1, guide_mode_2, sigma_next, guide_1, guide_2,  latent_guide_1, latent_guide_2, guide_1_channels)
      
    if denoise_to_zero:
      final_eta = eta[-1]
      eps = noise_sampler(sigma=sigma, sigma_next=sigma_next).double()
      sigma_hat = sigma * (1 + final_eta)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** .5 * eps
      
      s_in = x.new_ones([x.shape[0]])
      x_next = model(x_hat, torch.zeros_like(sigma).to(x_hat.device) * s_in, **extra_args)
      pbar.update()
      x = x_next

  return x


from torch.nn.functional import cosine_similarity

@no_grad()
def branch_mode_proc(
  x_n, x_h,
  denoised2,
  latent,
  branch_mode,
  branch_depth,
  branch_width,

):
  if branch_mode == 'cos_reversal':
    x_next, x_hat, d_next = select_trajectory_with_reversal(x_n, x_h, branch_depth)
  if branch_mode == 'cos_similarity':
    x_next, x_hat, d_next = select_trajectory_based_on_cosine_similarity(x_n, x_h, branch_depth, branch_width)
  if branch_mode == 'cos_similarity_d':
    x_next, x_hat, d_next = select_trajectory_based_on_cosine_similarity_d(x_n, x_h, denoised2, branch_depth, branch_width)
  if branch_mode == 'cos_linearity':
    x_next, x_hat, d_next = select_most_linear_trajectory(x_n, x_h, branch_depth, branch_width) 
  if branch_mode == 'cos_linearity_d':
    x_next, x_hat, d_next = select_most_linear_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width) 
  if branch_mode == 'cos_perpendicular':
    x_next, x_hat, d_next = select_perpendicular_cosine_trajectory(x_n, x_h, branch_depth, branch_width) 
  if branch_mode == 'cos_perpendicular_d':
    x_next, x_hat, d_next = select_perpendicular_cosine_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width) 
    
  if branch_mode == 'latent_match':
    distances = [torch.norm(tensor - latent).item() for tensor in x_n[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'latent_match_d':
    distances = [torch.norm(tensor - latent).item() for tensor in denoised2[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'latent_match_sdxl_color_d':
      relevant_latent = latent[:, 1:3, :, :] 
      denoised2_relevant = [tensor[:, 1:3, :, :] for tensor in denoised2[branch_depth]]

      distances = [torch.norm(tensor - relevant_latent).item() for tensor in denoised2_relevant]
      closest_index = distances.index(min(distances))
      
      x_next = x_n[branch_depth][closest_index]
      x_hat = x_h[branch_depth][closest_index]
      d_next = denoised2[branch_depth][closest_index]
      
  if branch_mode == 'latent_match_sdxl_luminosity_d':
      relevant_latent = latent[:, 0:1, :, :] 
      denoised2_relevant = [tensor[:, 0:1, :, :] for tensor in denoised2[branch_depth]]

      distances = [torch.norm(tensor - relevant_latent).item() for tensor in denoised2_relevant]
      closest_index = distances.index(min(distances))
      
      x_next = x_n[branch_depth][closest_index]
      x_hat = x_h[branch_depth][closest_index]
      d_next = denoised2[branch_depth][closest_index]
      
  if branch_mode == 'latent_match_sdxl_pattern_d':
      relevant_latent = latent[:, 3:4, :, :] 
      denoised2_relevant = [tensor[:, 3:4, :, :] for tensor in denoised2[branch_depth]]

      distances = [torch.norm(tensor - relevant_latent).item() for tensor in denoised2_relevant]
      closest_index = distances.index(min(distances))
      
      x_next = x_n[branch_depth][closest_index]
      x_hat = x_h[branch_depth][closest_index]
      d_next = denoised2[branch_depth][closest_index]

    
  if branch_mode == 'mean':
    x_mean = torch.mean(torch.stack(x_n[branch_depth]), dim=0)
    distances = [torch.norm(tensor - x_mean).item() for tensor in x_n[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'mean_d':
    d_mean = torch.mean(torch.stack(denoised2[branch_depth]), dim=0)
    distances = [torch.norm(tensor - d_mean).item() for tensor in denoised2[branch_depth]]
    closest_index = distances.index(min(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'median': #minimum median distance
    d_n_3 = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_3 = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_3 = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    num_tensors = len(x_n_3)
    distance_matrix = torch.zeros(num_tensors, num_tensors)

    for m in range(num_tensors):
        for n in range(num_tensors):
            if m != n:
                distance_matrix[m, n] = torch.norm(x_n_3[m] - x_n_3[n])
    median_distances = torch.median(distance_matrix, dim=1).values
    min_median_distance_index = torch.argmin(median_distances).item()
    x_next = x_n_3[min_median_distance_index]
    x_hat = x_h_3[min_median_distance_index]
    d_next = d_n_3[min_median_distance_index]
    
  if branch_mode == 'median_d': #minimum median distance
    d_n_3 = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_3 = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_3 = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    num_tensors = len(x_n_3)
    distance_matrix = torch.zeros(num_tensors, num_tensors)

    for m in range(num_tensors):
        for n in range(num_tensors):
            if m != n:
                distance_matrix[m, n] = torch.norm(d_n_3[m] - d_n_3[n])
    median_distances = torch.median(distance_matrix, dim=1).values
    min_median_distance_index = torch.argmin(median_distances).item()
    
    x_next = x_n_3[min_median_distance_index]
    x_hat = x_h_3[min_median_distance_index]
    d_next = d_n_3[min_median_distance_index]
    
  if branch_mode == 'zmean_d':
    d_mean = torch.mean(torch.stack(denoised2[branch_depth]), dim=0)
    distances = [torch.norm(tensor - d_mean).item() for tensor in denoised2[branch_depth]]
    closest_index = distances.index(max(distances))
    x_next = x_n[branch_depth][closest_index]
    x_hat = x_h[branch_depth][closest_index]
    d_next = denoised2[branch_depth][closest_index]
    
  if branch_mode == 'zmedian_d': #minimum median distance
    d_n_3 = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_3 = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_3 = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    num_tensors = len(x_n_3)
    distance_matrix = torch.zeros(num_tensors, num_tensors)

    for m in range(num_tensors):
        for n in range(num_tensors):
            if m != n:
                distance_matrix[m, n] = torch.norm(d_n_3[m] - d_n_3[n])
    median_distances = torch.median(distance_matrix, dim=1).values
    min_median_distance_index = torch.argmax(median_distances).item()
    
    x_next = x_n_3[min_median_distance_index]
    x_hat = x_h_3[min_median_distance_index]
    d_next = d_n_3[min_median_distance_index]    
    
  if branch_mode == 'gradient_max_full_d': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in denoised2[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(max(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min_full_d': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in denoised2[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(min(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]

  if branch_mode == 'gradient_max_full': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in x_n[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(max(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min_full': # greatest gradient descent
    start_point = x_n[0][0]
    norms = [torch.norm(tensor - start_point).item() for tensor in x_n[branch_depth] if tensor is not None]
    
    greatest_norm_index = norms.index(min(norms))
    
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]

  if branch_mode == 'gradient_max': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in x_n[branch_depth] if tensor is not None]
    greatest_norm_index = norms.index(max(norms))
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat  = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in x_n[branch_depth] if tensor is not None]
    min_norm_index = norms.index(min(norms))
    x_next = x_n[branch_depth][min_norm_index]
    x_hat  = x_h[branch_depth][min_norm_index]
    d_next = denoised2[branch_depth][min_norm_index]
    
  if branch_mode == 'gradient_max_d': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in denoised2[branch_depth] if tensor is not None]
    greatest_norm_index = norms.index(max(norms))
    x_next = x_n[branch_depth][greatest_norm_index]
    x_hat  = x_h[branch_depth][greatest_norm_index]
    d_next = denoised2[branch_depth][greatest_norm_index]
    
  if branch_mode == 'gradient_min_d': #greatest gradient descent
    norms = [torch.norm(tensor).item() for tensor in denoised2[branch_depth] if tensor is not None]
    min_norm_index = norms.index(min(norms))
    x_next = x_n[branch_depth][min_norm_index]
    x_hat  = x_h[branch_depth][min_norm_index]
    d_next = denoised2[branch_depth][min_norm_index]
    
  return x_next, x_hat, d_next
    
def select_trajectory_with_reversal(x_n, x_h, denoised2, branch_depth):
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    num_tensors = len(x_n_depth)

    negative_cos_sim_indices = []
    cos_sim_values = []

    for i in range(num_tensors):
        trajectory_cos_sims = []
        for j in range(1, len(x_n_depth[i]) - 1): 
            cos_sim = cosine_similarity(x_n_depth[i][j].unsqueeze(0), x_n_depth[i][j + 1].unsqueeze(0))
            trajectory_cos_sims.append(cos_sim.item())
        # check for reversal (negative cosine similarity)
        if any(cos_sim < 0 for cos_sim in trajectory_cos_sims):
            negative_cos_sim_indices.append(i)
            cos_sim_values.append(min(trajectory_cos_sims))

    if not negative_cos_sim_indices:
        # no reversal? fall back to the first available trajectory
        selected_index = 0
    else:
        # choose trajectory with most negative cosine similarity
        selected_index = negative_cos_sim_indices[torch.argmin(torch.tensor(cos_sim_values)).item()]

    x_next = x_n_depth[selected_index]
    x_hat = x_h_depth[selected_index]
    d_next = d_n_depth[selected_index]

    return x_next, x_hat, d_next

def select_trajectory_based_on_cosine_similarity(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    max_cosine_similarity = float('-inf')
    best_idx = -1

    for n in range(len(x_n_depth)):
        direction_vector = x_n_depth[n] - x_n[0][0]
        total_cosine_similarity = 0.0

        for depth in range(1, branch_depth):
            for j in range(len(x_n[depth])):
                x1_direction = x_n[depth][j] - x_n[depth - 1][j // branch_width]
                x1_to_x3_direction = x_n_depth[n] - x_n[depth][j]
                cosine_similarity = torch.dot(x1_direction.flatten(), x1_to_x3_direction.flatten()) / (torch.norm(x1_direction) * torch.norm(x1_to_x3_direction))
                total_cosine_similarity += cosine_similarity

        if total_cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  
def select_trajectory_based_on_cosine_similarity_d(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]
    
    denoised2[0][0] = x_n[0][0]

    max_cosine_similarity = float('-inf')
    best_idx = -1

    for n in range(len(d_n_depth)):
        direction_vector = d_n_depth[n] - x_n[0][0]
        total_cosine_similarity = 0.0

        for depth in range(1, branch_depth):
            for j in range(len(denoised2[depth])):
                x1_direction = denoised2[depth][j] - denoised2[depth - 1][j // branch_width]
                x1_to_x3_direction = d_n_depth[n] - denoised2[depth][j]
                cosine_similarity = torch.dot(x1_direction.flatten(), x1_to_x3_direction.flatten()) / (torch.norm(x1_direction) * torch.norm(x1_to_x3_direction))
                total_cosine_similarity += cosine_similarity

        if total_cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  

def select_most_linear_trajectory(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    max_cosine_similarity_sum = float('-inf')
    best_idx = -1

    base_vector = x_n[0][0]

    # sum up  absolute cosine similarities for each trajectory
    for n in range(len(x_n_depth)):
        total_cosine_similarity = 0.0
        current_vector = x_n_depth[n]

        #cormpare trajectory's endpoint vs all intermediate steps
        for depth in range(1, branch_depth + 1):
            for j in range(len(x_n[depth])):
                if depth == 1:
                    previous_vector = base_vector
                else:
                    previous_vector = x_n[depth - 1][j // branch_width]

                direction_vector = x_n[depth][j] - previous_vector
                cosine_similarity = torch.dot(direction_vector.flatten(), (current_vector - previous_vector).flatten()) / (
                    torch.norm(direction_vector) * torch.norm(current_vector - previous_vector))

                total_cosine_similarity += torch.abs(cosine_similarity) #abs val is key here... allows reversals (180 degree swap in direction, i.e., convergence)

        if total_cosine_similarity > max_cosine_similarity_sum:
            max_cosine_similarity_sum = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  
  
def select_most_linear_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    max_cosine_similarity_sum = float('-inf')
    best_idx = -1

    base_vector = x_n[0][0]

    # sum up  absolute cosine similarities for each trajectory
    for n in range(len(d_n_depth)):
        total_cosine_similarity = 0.0
        current_vector = d_n_depth[n]

        #cormpare trajectory's endpoint vs all intermediate steps
        for depth in range(1, branch_depth + 1):
            for j in range(len(denoised2[depth])):
                if depth == 1:
                    previous_vector = base_vector
                else:
                    previous_vector = denoised2[depth - 1][j // branch_width]

                direction_vector = denoised2[depth][j] - previous_vector
                cosine_similarity = torch.dot(direction_vector.flatten(), (current_vector - previous_vector).flatten()) / (
                    torch.norm(direction_vector) * torch.norm(current_vector - previous_vector))

                total_cosine_similarity += torch.abs(cosine_similarity) #abs val is key here... allows reversals (180 degree swap in direction, i.e., convergence)

        if total_cosine_similarity > max_cosine_similarity_sum:
            max_cosine_similarity_sum = total_cosine_similarity
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next


def select_perpendicular_cosine_trajectory(x_n, x_h, denoised2, branch_depth, branch_width):
    d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
    x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
    x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

    min_cosine_deviation_from_zero = float('inf')
    best_idx = -1

    # Calculate cosine similarities aiming for orthogonality at each step
    for n in range(len(x_n_depth)):
        total_cosine_deviation = 0.0

        # Iterate through the trajectory path
        for depth in range(1, branch_depth):
            for j in range(len(x_n[depth])):
                if depth == 1:
                    previous_vector = x_n[0][0]
                else:
                    previous_vector = x_n[depth - 1][j // branch_width]

                current_vector = x_n[depth][j] - previous_vector
                target_vector = x_n_depth[n] - x_n[depth][j]
                
                cosine_similarity = torch.dot(current_vector.flatten(), target_vector.flatten()) / (
                    torch.norm(current_vector) * torch.norm(target_vector))

                # Accumulate deviation from zero (ideal for perpendicular direction)
                total_cosine_deviation += (cosine_similarity ** 2)  # Squaring to emphasize smaller values

        # Update to select the trajectory with the minimum deviation from zero cosine similarity (most perpendicular)
        if total_cosine_deviation < min_cosine_deviation_from_zero:
            min_cosine_deviation_from_zero = total_cosine_deviation
            best_idx = n

    x_next = x_n_depth[best_idx]
    x_hat = x_h_depth[best_idx]
    d_next = d_n_depth[best_idx]

    return x_next, x_hat, d_next

  
  
def select_perpendicular_cosine_trajectory_d(x_n, x_h, denoised2, branch_depth, branch_width):
  d_n_depth = [tensor for tensor in denoised2[branch_depth] if tensor is not None]
  x_n_depth = [tensor for tensor in x_n[branch_depth] if tensor is not None]
  x_h_depth = [tensor for tensor in x_h[branch_depth] if tensor is not None]

  min_cosine_deviation_from_zero = float('inf')
  best_idx = -1

  # Calculate cosine similarities aiming for orthogonality at each step
  for n in range(len(d_n_depth)):
      total_cosine_deviation = 0.0

      # Iterate through the trajectory path
      for depth in range(1, branch_depth):
          for j in range(len(denoised2[depth])):
              if depth == 1:
                  previous_vector = x_n[0][0] #did i do this right???
              else:
                  previous_vector = denoised2[depth - 1][j // branch_width]

              current_vector = denoised2[depth][j] - previous_vector
              target_vector = d_n_depth[n] - denoised2[depth][j]
              
              cosine_similarity = torch.dot(current_vector.flatten(), target_vector.flatten()) / (
                  torch.norm(current_vector) * torch.norm(target_vector))

              # Accumulate deviation from zero (ideal for perpendicular direction)
              total_cosine_deviation += (cosine_similarity ** 2)  # Squaring to emphasize smaller values

      # Update to select the trajectory with the minimum deviation from zero cosine similarity (most perpendicular)
      if total_cosine_deviation < min_cosine_deviation_from_zero:
          min_cosine_deviation_from_zero = total_cosine_deviation
          best_idx = n

  x_next = x_n_depth[best_idx]
  x_hat = x_h_depth[best_idx]
  d_next = d_n_depth[best_idx]

  return x_next, x_hat, d_next





def guide_mode_proc(x, i, guide_mode_1, guide_mode_2, sigma_next, guide_1, guide_2,  latent_guide_1, latent_guide_2, guide_1_channels):
  if latent_guide_1 is not None:
    latent_guide_crushed_1 = (latent_guide_1 - latent_guide_1.min()) / (latent_guide_1 - latent_guide_1.min()).max()
  if latent_guide_2 is not None:
    latent_guide_crushed_2 = (latent_guide_2 - latent_guide_2.min()) / (latent_guide_2 - latent_guide_2.min()).max()

  b, c, h, w = x.shape
  
  if latent_guide_1 is not None:
    if(guide_mode_1 == 1):
      x = x - sigma_next * guide_1[i] * latent_guide_1 * guide_1_channels.view(1,c,1,1)

    if(guide_mode_1 == 2):
      x = x - sigma_next * guide_1[i] * latent_guide_crushed_1 * guide_1_channels.view(1,c,1,1)

    if(guide_mode_1 == 3):
      x = (1 - guide_1[i]) * x * guide_1_channels.view(1,c,1,1) + (guide_1[i] * latent_guide_1 * guide_1_channels.view(1,c,1,1))

    if(guide_mode_1 == 4):
      x = (1 - guide_1[i]) * x * guide_1_channels.view(1,c,1,1) + (guide_1[i] * latent_guide_crushed_1 * guide_1_channels.view(1,c,1,1))   

    if(guide_mode_1 == 5):
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * latent_guide_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 6):
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * latent_guide_crushed_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 7):
      hard_light_blend_1 = hard_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 8):
      hard_light_blend_1 = hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 9):
      soft_light_blend_1 = soft_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * soft_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 10):
      soft_light_blend_1 = soft_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * soft_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 11):
      linear_light_blend_1 = linear_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * linear_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 12):
      linear_light_blend_1 = linear_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * linear_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 13):
      vivid_light_blend_1 = vivid_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * vivid_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 14):
      vivid_light_blend_1 = vivid_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * vivid_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 801):
      hard_light_blend_1 = bold_hard_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 802):
      hard_light_blend_1 = bold_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 803):
      hard_light_blend_1 = fix_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 804):
      hard_light_blend_1 = fix2_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 805):
      hard_light_blend_1 = fix3_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 806):
      hard_light_blend_1 = fix4_hard_light_blend(latent_guide_1, x)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))
    if(guide_mode_1 == 807):
      hard_light_blend_1 = fix4_hard_light_blend(x, latent_guide_1)
      x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,c,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,c,1,1))

  if latent_guide_2 is not None:
    if(guide_mode_2 == 1):
      x = x - sigma_next * guide_2[i] * latent_guide_2
    if(guide_mode_2 == 2):
      x = x - sigma_next * guide_2[i] * latent_guide_crushed_2
    if(guide_mode_2 == 3):
      x = (1 - guide_2[i]) * x + (guide_2[i] * latent_guide_2)
    if(guide_mode_2 == 4):
      x = (1 - guide_2[i]) * x + (guide_2[i] * latent_guide_crushed_2)   
    if(guide_mode_2 == 5):
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * latent_guide_2)
    if(guide_mode_2 == 6):
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * latent_guide_crushed_2)   
    if(guide_mode_2 == 7):
      hard_light_blend_2 = hard_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * hard_light_blend_2)
    if(guide_mode_2 == 8):
      hard_light_blend_2 = hard_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * hard_light_blend_2)
    if(guide_mode_2 == 9):
      soft_light_blend_2 = soft_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * soft_light_blend_2)
    if(guide_mode_2 == 10):
      soft_light_blend_2 = soft_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * soft_light_blend_2)
    if(guide_mode_2 == 11):
      linear_light_blend_2 = linear_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * linear_light_blend_2)
    if(guide_mode_2 == 12):
      linear_light_blend_2 = linear_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * linear_light_blend_2)
    if(guide_mode_2 == 13):
      vivid_light_blend_2 = vivid_light_blend(x, latent_guide_2)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * vivid_light_blend_2)
    if(guide_mode_2 == 14):
      vivid_light_blend_2 = vivid_light_blend(latent_guide_2, x)
      x = (x - guide_2[i] * sigma_next * x) + (guide_2[i] * sigma_next * vivid_light_blend_2)
  return x




def fix4_hard_light_blend(base_latent, blend_latent):

    multiply_effect = 2 * base_latent * blend_latent
    screen_effect = base_latent + blend_latent - base_latent * blend_latent
    result = torch.where(blend_latent < 0, multiply_effect, screen_effect)
    return result

def fix3_hard_light_blend(base_latent, blend_latent):
    blend_mid = (blend_latent.max() + blend_latent.min()) / 2

    multiply_effect = 2 * base_latent * ((blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min()))
    screen_effect = (blend_latent.max() - blend_latent.min()) + base_latent - 2 * (base_latent * (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min()))

    result = torch.where(blend_latent <= blend_mid, multiply_effect, screen_effect)
    return result

def fix2_hard_light_blend(base_latent, blend_latent):

    blend_range = blend_latent.max() - blend_latent.min()
    blend_mid = blend_latent.min() + blend_range / 2

    result = torch.where(blend_latent <= blend_mid,
                         2 * (blend_latent - blend_latent.min()) / blend_range * base_latent,
                         1 - 2 * (1 - (blend_latent - blend_latent.min()) / blend_range) * (1 - base_latent))
    return result

def fix_hard_light_blend(base_latent, blend_latent):

    blend_latent = blend_latent - blend_latent.min()
    base_latent = base_latent - base_latent.min()

    blend_max = blend_latent.max()
    blend_min = blend_latent.min()
    blend_half = blend_max/2

    result = torch.where(blend_latent < blend_half,
                                  2 * base_latent * blend_latent,
                                  blend_max - 2 * (blend_max - blend_latent) * (blend_max - blend_latent))

    result = result + base_latent.min()
    return result

def bold_hard_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())
    blend_latent = blend_latent - blend_latent.min()

    blend_max = blend_latent.max()
    blend_min = blend_latent.min()
    blend_half = blend_max/2
    
    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0
    
    positive_latent = base_latent * positive_mask.float()
    negative_latent = base_latent * negative_mask.float()
    
    positive_result = torch.where(blend_latent < blend_half,
                                  2 * positive_latent * blend_latent,
                                  1 - 2 * (1 - positive_latent) * (1 - blend_latent))

    negative_result = torch.where(blend_latent < blend_half,
                                  2 * negative_latent.abs() * blend_latent,
                                  1 - 2 * (1 - negative_latent.abs()) * (1 - blend_latent))
    negative_result = -negative_result 
    
    combined_result = positive_result * positive_mask.float() + negative_result * negative_mask.float()

    return combined_result

def bold_soft_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - base_latent) * (1 - (blend_latent - 0.5) * 2)),
                                  base_latent * (blend_latent * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - negative_base) * (1 - (blend_latent - 0.5) * 2)),
                                  negative_base * (blend_latent * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result

    return positive_result + negative_result

def bold_vivid_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0,
                                  1 - (1 - base_latent) / ((blend_latent - 0.5) * 2),
                                  base_latent / (1 - (blend_latent - 0.5) * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  1 - (1 - negative_base) / ((blend_latent - 0.5) * 2),
                                  negative_base / (1 - (blend_latent - 0.5) * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result 

    return positive_result + negative_result

def hard_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0
    
    positive_latent = base_latent * positive_mask.float()
    negative_latent = base_latent * negative_mask.float()

    positive_result = torch.where(blend_latent < 0.5,
                                  2 * positive_latent * blend_latent,
                                  1 - 2 * (1 - positive_latent) * (1 - blend_latent))

    negative_result = torch.where(blend_latent < 0.5,
                                  2 * negative_latent.abs() * blend_latent,
                                  1 - 2 * (1 - negative_latent.abs()) * (1 - blend_latent))
    negative_result = -negative_result

    combined_result = positive_result * positive_mask.float() + negative_result * negative_mask.float()

    return combined_result

def soft_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - base_latent) * (1 - (blend_latent - 0.5) * 2)),
                                  base_latent * (blend_latent * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  (1 - (1 - negative_base) * (1 - (blend_latent - 0.5) * 2)),
                                  negative_base * (blend_latent * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result  

    return positive_result + negative_result

def vivid_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = torch.where(blend_latent > 0.5,
                                  1 - (1 - base_latent) / ((blend_latent - 0.5) * 2),
                                  base_latent / (1 - (blend_latent - 0.5) * 2))
    positive_result *= positive_mask.float()

    negative_base = base_latent.abs() * negative_mask.float()
    negative_result = torch.where(blend_latent > 0.5,
                                  1 - (1 - negative_base) / ((blend_latent - 0.5) * 2),
                                  negative_base / (1 - (blend_latent - 0.5) * 2))
    negative_result *= negative_mask.float()
    negative_result = -negative_result 

    return positive_result + negative_result

def linear_light_blend(base_latent, blend_latent):
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0

    positive_result = base_latent + 2 * blend_latent - 1
    positive_result *= positive_mask.float()

    negative_result = -base_latent.abs() + 2 * blend_latent - 1
    negative_result *= negative_mask.float()
    negative_result = -negative_result 

    return positive_result + negative_result

"""blend_modes = {
    'hard_light': hard_light_blend,
    'soft_light': soft_light_blend,
    'vivid_light': vivid_light_blend,
    'linear_light': linear_light_blend,
    'subtractive': subtractive_blend,
    'average': average_blend,
    'multiply': multiply_blend,
    'screen': screen_blend,
    'color_burn': color_burn_blend,
    'color_dodge': color_dodge_blend,
}
"""