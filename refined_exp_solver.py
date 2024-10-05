"""Algorithm 1 "RES Second order Single Update Step with c2"
https://arxiv.org/abs/2308.02157"""

import torch
from torch import no_grad, FloatTensor
from tqdm import tqdm
from tqdm.auto import trange

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

def get_ancestral_step_RF_var(sigma, sigma_next, eta):
    dtype = sigma.dtype #calculate variance adjusted sigma up... sigma_up = sqrt(dt)
    eps = 1e-10

    sigma, sigma_next = sigma.to(torch.float64), sigma_next.to(torch.float64)
    sigma_diff = (sigma - sigma_next).abs() + eps 
    sigma_up = torch.sqrt(sigma_diff).to(torch.float64) * eta

    sigma_down_num = (sigma_next**2 - sigma_up**2).to(torch.float64)
    sigma_down = torch.sqrt(sigma_down_num) / ((1 - sigma_next).to(torch.float64) + torch.sqrt(sigma_down_num).to(torch.float64))

    alpha_ratio = (1 - sigma_next).to(torch.float64) / (1 - sigma_down).to(torch.float64)
    return sigma_up.to(dtype), sigma_down.to(dtype), alpha_ratio.to(dtype)

  
def get_RF_step(sigma, sigma_next, eta, noise_scale=1.0, alpha_ratio=None):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition. Scale is to shape the sigma_down curve."""
    sigma_minus = sigma - sigma_next
    down_ratio = (1 - eta) + eta * ((sigma - sigma_minus) / sigma)
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)   ###NOTE THE CHANGE HERE WITH SQUARING THE SHIT
    sigma_down = sigma_next ** noise_scale * down_ratio

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 
    return sigma_up, sigma_down, alpha_ratio

def get_RF_step_traditional(sigma, sigma_next, eta, scale=0.0,  alpha_ratio=None):
    # uses math similar to what is used for the get ancestral step code in comfyui. WORKS!
    #down_ratio = (1 - eta) + eta * (sigma_next / sigma)   #maybe this is the most appropriate target for scaling?
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_up, sigma_down, alpha_ratio

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



def calculate_second_order_multistep_coeffs(sigma, sigma_next, sigma_prev):
    """
    Calculate coefficients for the second-order multistep solver using Algorithm 4 from the paper.

    Args:
        sigma: Current noise level (float or tensor)
        sigma_next: Next noise level (float or tensor)
        sigma_prev: Previous noise level (float or tensor)

    Returns:
        A dictionary of coefficients: h, b1, b2
    """
    lam_n_plus_1 = -torch.log(sigma_next)
    lam_n = -torch.log(sigma)
    lam_n_minus_1 = -torch.log(sigma_prev)
    
    h = lam_n_plus_1 - lam_n
    
    c2 = h / (lam_n - lam_n_minus_1)
    
    phi1 = _phi_1(-h)
    phi2 = _phi_2(-h) / c2
    
    b1 = phi1 - phi2
    b2 = phi2

    return h, c2, b1, b2
  
  

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

# by CSBW
def _phi_1(neg_h: FloatTensor):
  phi_1_csbw = torch.nan_to_num(torch.exp(neg_h)*torch.expm1(-neg_h) / -neg_h, nan=1.0)
  phi_1_kc = torch.nan_to_num(torch.expm1(neg_h) / neg_h, nan=1.0)
  print("phi1 csbw, kc: ", phi_1_csbw, phi_1_kc)
  return phi_1_csbw
  return torch.nan_to_num(torch.exp(neg_h)*torch.expm1(-neg_h) / -neg_h, nan=1.0)


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

def _phi_csbw(j, h):
  remainder = torch.zeros_like(h)
  
  for k in range(j): 
    remainder += (-h)**k / math.factorial(k)
  phi_j_h = ((-h).exp() - remainder) / (-h)**j
  
  return phi_j_h
  

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
    #a2_1: float = c2 * _phi_1(-c2*h)
    #phi1: float = _phi_1(-h)
    #phi2: float = _phi_2(-h)
    
    a2_1: float = c2 * _phi_csbw(1, c2*h)
    phi1: float = _phi_csbw(1, h)
    phi2: float = _phi_csbw(2, h)
    
    
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
  

"""def calculate_gamma(c2: float, c3: float) -> float:
    numerator = 3 * (c2 ** 2) + 3 * (c3 ** 3)
    denominator = 2 * (c2 + c3)
    gamma = numerator / denominator
    return gamma"""


def calculate_gamma(c2: float, c3: float) -> float:
    """Calculate gamma based on c2 and c3 to satisfy the third-order condition."""
    numerator = 3 * (c3 ** 3) - 2 * c3
    denominator = c2 * (2 - 3 * c2)
    if denominator == 0:
        raise ValueError("Invalid values for c2 leading to division by zero.")
    gamma = numerator / denominator
    print("gamma half1: ", 2*(gamma*c2+c3))
    print("gamma half2: ", 3*(gamma*c2**2 + c3**3))
    return gamma
  

class RESDECoeffsThirdOrder(NamedTuple):
    a3_1: float
    b1: float
    b2: float
    b3: float
    gamma: float

def _de_third_order(h: float, c2: float, c3: float, simple_phi_calc=False) -> RESDECoeffsThirdOrder:
    """
    Calculate third-order coefficients.
    """
    gamma = calculate_gamma(c2, c3)
    
    if simple_phi_calc:
        a3_1 = c3 * _phi_1(-c3 * h)
        phi1 = _phi_1(-h)
        phi2 = _phi_2(-h)
        phi3 = _phi_3(-h)
    else:
        a3_1 = c3 * _phi(j=1, neg_h=-c3 * h)
        phi1 = _phi(j=1, neg_h=-h)
        phi2 = _phi(j=2, neg_h=-h)
        phi3 = _phi(j=3, neg_h=-h)

    b1 = phi1 - (phi2 / c2) - (phi3 / c3)
    b2 = (phi2 / c2) - (phi3 / (c2 * c3))
    b3 = phi3 / (c2 * c3)

    return RESDECoeffsThirdOrder(a3_1=a3_1, b1=b1, b2=b2, b3=b3, gamma=gamma)

"""def calculate_third_order_coeffs(h, c2, c3): #, gamma):
    gamma = calculate_gamma(c2, c3)
    
    # Calculate the step sizes
    neg_h = -h
    neg_h_c2 = -c2 * h
    neg_h_c3 = -c3 * h
    
    # Calculate the phi values
    phi_1_h = _phi_1(neg_h)
    phi_2_h = _phi_2(neg_h)
    
    # Phi for scaled step sizes
    phi_1_c2_h = _phi_1(neg_h_c2)
    phi_1_c3_h = _phi_1(neg_h_c3)
    
    phi_2_c2_h = _phi_2(neg_h_c2)
    phi_2_c3_h = _phi_2(neg_h_c3)
    
    # Step 1: Compute a21 for second stage
    a21 = c2 * phi_1_c2_h
    
    # Step 2: Compute a31 and a32 for third stage
    a31 = c3 * phi_1_c3_h  # a31 from k1 to k3
    a32 = gamma * c2 * phi_2_c2_h + (c3 ** 2 / c2) * phi_2_c3_h  # a32 from k2 to k3
    
    # Step 3: Compute b1, b2, b3 (final combination coefficients)
    b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  # Middle term for bottom row
    b3 = (1 / (gamma * c2 + c3)) * phi_2_h      # Right term for bottom row
    b1 = phi_1_h - b2 - b3                      # First term from balancing the bottom row
    
    print("gamma half1: ", 2*(gamma*c2+c3))
    print("gamma half2: ", 3*(gamma*c2**2 + c3**2))
    return a21, a31, a32, b1, b2, b3
    
    return {
        "a21": a21,
        "a31": a31,
        "a32": a32,
        "b1": b1,
        "b2": b2,
        "b3": b3
    }"""

def calculate_third_order_coeffs(h, c2, c3):
    """
    Calculate all coefficients for the third-order solver using the Butcher tableau.
    
    Args:
        h: Step size (float)
        c2: Coefficient for second stage (from Butcher tableau)
        c3: Coefficient for third stage (from Butcher tableau)
    
    Returns:
        A tuple of coefficients: a21, a31, a32, b1, b2, b3
    """
    gamma = calculate_gamma(c2, c3)
    
    neg_h = -h
    neg_h_c2 = -c2 * h
    neg_h_c3 = -c3 * h
    
    phi_1_h = _phi(neg_h, j=1)
    phi_2_h = _phi(neg_h, j=2)
    
    phi_1_c2_h = _phi(neg_h_c2, j=1)
    phi_1_c3_h = _phi(neg_h_c3, j=1)
    
    phi_2_c2_h = _phi(neg_h_c2, j=2)
    phi_2_c3_h = _phi(neg_h_c3, j=2)
    
    a21 = c2 * phi_1_c2_h
    
    a32 = gamma * c2 * phi_2_c2_h + (c3 ** 2 / c2) * phi_2_c3_h  # a32 from k2 to k3
    #a31 = c3 * phi_1_c3_h  # a31 from k1 to k3
    a31 = c3 * phi_1_c3_h - a32 # a31 from k1 to k3
    
    b3 = (1 / (gamma * c2 + c3)) * phi_2_h      
    b2 = gamma * b3
    #b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  # 
    b1 = phi_1_h - b2 - b3                      
    
    print("a21 31 32: ", a21.item(), a31.item(), a32.item(), "b: ", b1.item(), b2.item(), b3.item(), "h: ", h.item(), c2.item(), c3.item())
    return a21, a31, a32, b1, b2, b3


def _refined_exp_sosu_step_RF_third_order(
    model, x, sigma, sigma_next, c2=0.5, c3=2./3., eta1=0.0, eta2=0.0, eta_var1=0.0, eta_var2=0.0, noise_sampler=None, noise_mode="hard", order=3,
    s_noise1=1.0, s_noise2=1.0, s_noise3=1.0, denoised1_2_3=None, h_last=None, auto_c2=False,
    extra_args: Dict[str, Any] = {},
    pbar: Optional[tqdm] = None,
    momentum=0.0, vel=None, vel_2=None, vel_3=None,
    time=None,
    t_fn_formula="", sigma_fn_formula="",
    simple_phi_calc=False,
):
    t_fn = lambda sigma: sigma.log().neg() if not t_fn_formula else eval(f"lambda sigma: {t_fn_formula}", {"torch": torch})
    sigma_fn = lambda t: t.neg().exp() if not sigma_fn_formula else eval(f"lambda t: {sigma_fn_formula}", {"torch": torch})

    t_fn = eval(f"lambda sigma: {t_fn_formula}", {"torch": torch}) if t_fn_formula else (lambda sigma: sigma.log().neg())
    sigma_fn = eval(f"lambda t: {sigma_fn_formula}", {"torch": torch}) if sigma_fn_formula else (lambda t: t.neg().exp())
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0):
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel

    gamma = calculate_gamma(c2, c3)
    
    #sd = sigma_next
    
    #t, t_next = sigma.log().neg(), sigma_next.log().neg()
    t = t_fn(sigma)
    t_next = t_fn(sigma_next)
    
    h = t_next - t
    
    print("sigma_next: ", sigma_next)
    if h >= 0.9999:
      h = torch.tensor(0.9999).to(h.dtype).to(h.device) # dtype=h.dtype, device=h.device)
      t_next = h + t
      sigma_next = sigma_fn(t_next)
      
      

    s2 = t + h * c2
    s3 = t + h * c3
    #sigma_2 = s2.neg().exp()
    #sigma_3 = s3.neg().exp()
    sigma_2 = sigma_fn(s2)
    sigma_3 = sigma_fn(s3)
    
    #su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
    #sigma_2, h, c2 = get_res4lyf_half_step(sigma, sd, c2, False, None, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
    #sigma_3, h, c3 = get_res4lyf_half_step(sigma, sd, c3, False, None, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
    #sigma_next = sd
    
    a21, a31, a32, b1, b2, b3 = calculate_third_order_coeffs(h, c2, c3)
    print("sigmas: ", sigma.item(), sigma_next.item(), sigma_2.item(), sigma_3.item())
    
    #sigma_2 = sigma * torch.exp(-c2 * h)
    #sigma_3 = sigma * torch.exp(-c3 * h)
    s_in = x.new_ones([x.shape[0]])
    
    denoised1 = model(x, sigma * s_in, **extra_args)
    k1 = denoised1

    x_2 = torch.exp(-c2*h)*x + h * (a21 * k1)
    #x_2 = ((sd/sigma)**c2)*x + h * (a21 * k1)
    
    denoised2 = model(x_2, sigma_2 * s_in, **extra_args)
    k2 = denoised2

    x_3 = torch.exp(-c3*h)*x + h * (a31 * k1 + a32 * k2)
    #x_3 = ((sd/sigma)**c3)*x + h * (a31 * k1 + a32 * k2)

    denoised3 = model(x_3, sigma_3 * s_in, **extra_args)
    k3 = denoised3

    x_next = torch.exp(-h)*x + h * (b1 * k1 + b2 * k2 + b3 * k3)
    #x_next = ((sd/sigma))*x + h * (b1 * k1 + b2 * k2 + b3 * k3)

    #x_next = alpha_ratio * x_next + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise2 * su

    gc.collect()
    torch.cuda.empty_cache()
    if pbar is not None:
        pbar.update(1.0)

    denoised1_2_3 = (b1 * k1 + b2 * k2 + b3 * k3)
    return x_next, denoised1, denoised2, denoised3, denoised1_2_3, vel, vel_2, vel_3, h, sigma_next







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

  gc.collect()
  torch.cuda.empty_cache()

  return StepOutput(x_next=x_next, denoised=denoised, denoised2=denoised2, vel=vel, vel_2=vel_2,)





def _refined_exp_sosu_step_RF(model, x, sigma, sigma_next, c2 = 0.5, eta1=0.25, eta2=0.5, eta_var1=0.0, eta_var2=0.0, noise_sampler=None, noise_mode="hard", order="2b", 
                                   s_noise1=1.0, s_noise2=1.0, denoised1_2=None, h_last=None, auto_c2=False,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  t_fn_formula="", sigma_fn_formula="", 
  simple_phi_calc = False,
):
  t_fn_formula = "sigma.log().neg()" if not t_fn_formula else t_fn_formula
  sigma_fn_formula = "t.neg().exp()" if not sigma_fn_formula else sigma_fn_formula
  
  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
  sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  s_in = x.new_ones([x.shape[0]])
  
  if order == "1" or h_last is None:
    denoised = model(x, sigma * s_in, **extra_args)
  else:
    denoised = denoised1_2
  
  diff_2 = vel_2 = momentum_func(h*a2_1*denoised, vel_2, time)
  x_2 = ((sd/sigma)**c2)*x + diff_2 
    
  if sigma_next > 0.00001:
    su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1, eta_var1, noise_mode)
    x_2 = alpha_ratio_2 * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * s_noise2 * su_2
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)
  else: 
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)   #last step!

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  x_next =  (sd/sigma) * x + diff
  x_next = alpha_ratio * x_next + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise1 * su
  
  denoised1_2 = momentum_func((b1*denoised + b2*denoised2), vel, time) / (b1 + b2)

  if pbar is not None:
    pbar.update(1.0)

  gc.collect()
  torch.cuda.empty_cache()
  
  return x_next, denoised, denoised2, denoised1_2, vel, vel_2, h



@no_grad()
def sample_refined_exp_s_advanced_RF_rolling(
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
  etas1=None,
  etas2=None,
  eta_vars1=None,
  eta_vars2=None,
  s_noises1=None,
  s_noises2=None,
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
  noisy_cfg=False,
  noise_scale=1.0,
  ancestral_noise=True,
  alpha_ratios=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None,
  latent_self_guide_1=False,
  latent_shift_guide_1=False,
  t_fn_formula="",
  sigma_fn_formula="",
  skip_corrector=False,
  corrector_is_predictor=False,
  order=2,
  auto_c2=False,
  step_type="res_a"
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
  
  if clownseed < 0.0:
    clownseed = extra_args.get("seed", None) + 1
  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)
  print("ClownSeed set to: ", clownseed)

  dt = None
  vel, vel_2 = None, None
  x_hat = None
  
  x_n        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  x_h        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel_2      = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised2  = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised1_2= [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised_  = None
  denoised2_ = None
  denoised1_2= None
  denoised2_prev = None
  h_last = None
  
  
      
  if len(sigmas) <= 1:
      return x

  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
  seed = extra_args.get("seed", None) + 1

  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

  extra_args = {} if extra_args is None else extra_args

  vel, vel_2 = None, None
  denoised1_2=None
  h_last = None
  sigma_prev = None
  #denoised = [None] * len(sigmas)
  #x_next   = [None] * len(sigmas)
  
  i=0
  sigma = sigmas[i]
  sigma_next = sigmas[i+1]
  time = sigmas[i] / sigma_max
  x_next, denoised, denoised2, denoised1_2, vel, vel_2, h_last  = _refined_exp_sosu_step_RF(model, x, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                  noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                  extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                  momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                  t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                  )

  x_prev = x
  for i in trange(1, len(sigmas) - 1, disable=disable):

      sigma_prev = sigmas[i-1]
      sigma = sigmas[i]
      sigma_next = sigmas[i+1]
      time = sigmas[i] / sigma_max
      
      sigma_next = torch.tensor(0.00001) if sigma_next == 0.0 else sigma_next
      

      x_next, denoised, denoised2, denoised1_2, vel, vel_2, h_last, x_prev  = _refined_exp_sosu_step_RF_midpoint(model, x_prev, sigma_prev, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                        noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                        extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                        momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                        t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                        )


      if callback is not None:
          callback({'x': x_next, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1_2})

      gc.collect()
      torch.cuda.empty_cache()

  return x_next



def _refined_exp_sosu_step_RF_midpoint(model, x_prev, sigma_prev, sigma, sigma_next, c2 = 0.5, eta1=0.25, eta2=0.5, eta_var1=0.0, eta_var2=0.0, noise_sampler=None, noise_mode="hard", order=2, 
                                   s_noise1=1.0, s_noise2=1.0, denoised1_2=None, h_last=None, auto_c2=False,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  t_fn_formula="", sigma_fn_formula="", 
  simple_phi_calc = False,
):
  su, sd, alpha_ratio, sigma_s, h, c2, denoised = None, None, None, None, None, None, None
  t_fn_formula = "sigma.log().neg()" if not t_fn_formula else t_fn_formula
  sigma_fn_formula = "t.neg().exp()" if not sigma_fn_formula else sigma_fn_formula
  
  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  
  s_in = x_prev.new_ones([x_prev.shape[0]])
  
  """if order == 1 or h_last is None:
    denoised = model(x, sigma * s_in, **extra_args)
    su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
    sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
  else:"""
  denoised = denoised1_2
  sigma_s = sigma
  sigma = sigma_prev
  su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
  
  t, t_sd = sigma.log().neg(), sd.log().neg()
  h = t_sd - t
  s = sigma_s.log().neg()
  
  c2 = (s - t) / h
  
  #su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
  #sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  diff_2 = vel_2 = momentum_func(h*a2_1*denoised, vel_2, time)
  x_2 = ((sd/sigma)**c2)*x_prev + diff_2 
    
  if sigma_next > 0.00001:
    su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1, eta_var1, noise_mode)
    x_2 = alpha_ratio_2 * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * s_noise2 * su_2
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)
  else: 
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)   #last step!

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  x_next =  (sd/sigma) * x_prev + diff
  x_next = alpha_ratio * x_next + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise1 * su
  
  denoised1_2 = momentum_func((b1*denoised + b2*denoised2), vel, time) / (b1 + b2)

  if pbar is not None:
    pbar.update(1.0)

  gc.collect()
  torch.cuda.empty_cache()

  return x_next, denoised, denoised2, denoised1_2, vel, vel_2, h, x_2  #x_2 is the next x_prev... it is the "midpoint" x, which we will set to the "start" x for the next cycle




def _refined_exp_sosu_step_RF_midpoint2(model, x_prev, sigma_prev, sigma, sigma_next, c2 = 0.5, eta1=0.25, eta2=0.5, eta_var1=0.0, eta_var2=0.0, noise_sampler=None, noise_mode="hard", order=2, 
                                   s_noise1=1.0, s_noise2=1.0, denoised1_2=None, h_last=None, auto_c2=False,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  t_fn_formula="", sigma_fn_formula="", 
  simple_phi_calc = False,
):
  su, sd, alpha_ratio, sigma_s, h, c2, denoised = None, None, None, None, None, None, None
  t_fn_formula = "sigma.log().neg()" if not t_fn_formula else t_fn_formula
  sigma_fn_formula = "t.neg().exp()" if not sigma_fn_formula else sigma_fn_formula
  
  su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  
  s_in = x_prev.new_ones([x_prev.shape[0]])
  h, c2, b1, b2 = calculate_second_order_multistep_coeffs(sigma, sigma_next, sigma_prev)
  
  denoised = model(x_prev, sigma * s_in, **extra_args) 
  
  x_next = math.exp(-h) * x_prev + h * (b1 * denoised + b2 * denoised1_2)
  #x_next = (sd/sigma) * x_prev + h * (b1 * denoised + b2 * denoised1_2)
  
  x_next = alpha_ratio * x_next + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise2 * su
  
  return x_next, denoised, vel, vel_2
  """if order == 1 or h_last is None:
  
    denoised = model(x, sigma * s_in, **extra_args)
    su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
    sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
  else:"""
  denoised = denoised1_2
  sigma_s = sigma
  sigma = sigma_prev
  su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)

  #su, sd, alpha_ratio = get_res4lyf_step(sigma, sigma_next, eta2, eta_var2, noise_mode)
  #sigma_s, h, c2 = get_res4lyf_half_step(sigma, sd, c2, auto_c2, h_last, t_fn_formula, sigma_fn_formula, remap_t_to_exp_space=True)
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  diff_2 = vel_2 = momentum_func(h*a2_1*denoised, vel_2, time)
  x_2 = ((sd/sigma)**c2)*x_prev + diff_2 
    
  if sigma_next > 0.00001:
    su_2, sd_2, alpha_ratio_2 = get_res4lyf_step(sigma, sigma_next, eta1, eta_var1, noise_mode)
    x_2 = alpha_ratio_2 * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * s_noise2 * su_2
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)
  else: 
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)   #last step!

  diff = vel = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  x_next =  (sd/sigma) * x_prev + diff
  x_next = alpha_ratio * x_next + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise1 * su
  
  denoised1_2 = momentum_func((b1*denoised + b2*denoised2), vel, time) / (b1 + b2)

  if pbar is not None:
    pbar.update(1.0)

  gc.collect()
  torch.cuda.empty_cache()

  return x_next, denoised, denoised2, denoised1_2, vel, vel_2, h, x_2  #x_2 is the next x_prev... it is the "midpoint" x, which we will set to the "start" x for the next cycle





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
  return su, sd, alpha_ratio


def get_res4lyf_half_step(sigma, sigma_next, c2=0.5, auto_c2=False, h_last=None, t_fn_formula="", sigma_fn_formula="", remap_t_to_exp_space=False):
  sigma_fn = lambda t: t.neg().exp()
  t_fn     = lambda sigma: sigma.log().neg()
  
  sigma_fn_x = eval(f"lambda t: {sigma_fn_formula}", {"t": None}) if sigma_fn_formula else sigma_fn
  t_fn_x = eval(f"lambda sigma: {t_fn_formula}", {"sigma": None}) if t_fn_formula else t_fn
  
  if not remap_t_to_exp_space: 
    sigma_fn = sigma_fn_x
    t_fn = t_fn_x
      
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



def _dpmpp_sde_step_RF(model, x, sigma, sigma_next, c2 = 0.5, eta1=0.25, eta2=0.5, eta_var1=0.0, eta_var2=0.0, noise_sampler=None, noise_mode="hard", order=2, 
                                   s_noise1=1.0, s_noise2=1.0, denoised1_2=None, h_last=None, auto_c2=False, denoise_boost=0.0, 
  extra_args=None,
  pbar: Optional[tqdm] = None,
  momentum = 0.0, vel = None, vel_2 = None,
  time = None,
  t_fn_formula="", sigma_fn_formula="", 
):
  t_fn_formula = "1/((sigma).exp()+1)" if not t_fn_formula else t_fn_formula
  sigma_fn_formula = "((1-t)/t).log()" if not sigma_fn_formula else sigma_fn_formula
  
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
    x_2 = alpha_ratio_2 * x_2 + noise_sampler(sigma=sigma, sigma_next=sigma_s) * s_noise2 * su_2
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)
  else: 
    denoised2 = model(x_2, sigma_s * s_in, **extra_args)   #last step!
    
  diff = vel = momentum_func(denoised2, vel, time)

  denoised1_2 = (1 - fac) * denoised + fac * diff #denoised2
  x = (sd / sigma) * x + (1 - (sd / sigma))  * denoised1_2
  
  x = alpha_ratio * x + noise_sampler(sigma=sigma, sigma_next=sigma_next) * s_noise1 * su

  if pbar is not None:
    pbar.update(1.0)
    
  gc.collect()
  torch.cuda.empty_cache()
  
  return x, denoised, denoised2, denoised1_2, vel, vel_2, h



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
  etas1=None,
  etas2=None,
  eta_vars1=None,
  eta_vars2=None,
  s_noises1=None,
  s_noises2=None,
  momentum=None,
  eulers_mom=None,
  c2=None,
  c3=None,
  cfgpp=None,
  offset=None,
  alpha=None,
  latent_guide_1=None,
  latent_guide_2=None,
  noise_sampler: NoiseSampler = torch.randn_like,
  noise_sampler_type=None,
  noise_mode="hard",
  noisy_cfg=False,
  noise_scale=1.0,
  ancestral_noise=True,
  alpha_ratios=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None,
  latent_self_guide_1=False,
  latent_shift_guide_1=False,
  t_fn_formula="",
  sigma_fn_formula="",
  skip_corrector=False,
  corrector_is_predictor=False,
  order="2b",
  auto_c2=False,
  step_type="res_a"
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
  
  if clownseed < 0.0:
    clownseed = extra_args.get("seed", None) + 1
  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)
  print("ClownSeed set to: ", clownseed)

  dt = None
  vel, vel_2 = None, None
  x_hat = None
  
  x_n        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  x_h        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel_2      = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised2  = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised1_2= [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised_  = None
  denoised2_ = None
  denoised1_2= None
  denoised2_prev = None
  h_last = None
  
  
      
  if len(sigmas) <= 1:
      return x

  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
  seed = extra_args.get("seed", None) + 1

  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

  extra_args = {} if extra_args is None else extra_args

  vel, vel_2, vel_3 = None, None, None
  denoised1_2=None
  denoised1_2_3 = None
  h_last = None

  if order == "1" or order == "2a":
    for i in trange(len(sigmas) - 1, disable=disable):

        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        time = sigmas[i] / sigma_max
        
        sigma_next = torch.tensor(0.00001) if sigma_next == 0.0 else sigma_next
        
        if step_type == "res_a":
          x, denoised, denoised2, denoised1_2, vel, vel_2, h_last  = _refined_exp_sosu_step_RF(model, x, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                          noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                          extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                          momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                          t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                          )
        if step_type == "dpmpp_sde_alt":
          x, denoised, denoised2, denoised1_2, vel, vel_2, h_last  = _dpmpp_sde_step_RF(model, x, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                          noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                          extra_args=extra_args,
                                                          momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                          t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2, denoise_boost=0.0,
                                                          )

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1_2})

        gc.collect()
        torch.cuda.empty_cache()
    return x
  
  elif order == "3":
    new_sigma_next = torch.tensor(0.0).to(sigmas.dtype).to(sigmas.device)
    for i in trange(len(sigmas) - 1, disable=disable):

        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        
        if new_sigma_next > sigma:
          sigma = new_sigma_next
        
        time = sigma / sigma_max
        
        sigma_next = torch.tensor(0.001) if sigma_next == 0.0 else sigma_next
        
        if step_type == "res_a":
          x, denoised, denoised2, denoised3, denoised1_2_3, vel, vel_2, vel_3, h_last, new_sigma_next = _refined_exp_sosu_step_RF_third_order(model, x, sigma, sigma_next, c2=c2[i], c3=c3[i], eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                          noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                          extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                          momentum = momentum[i], vel = vel, vel_2 = vel_2, vel_3 = vel_3, time = time,
                                                          t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2_3=denoised1_2_3, h_last=h_last, order=order, auto_c2=auto_c2,
                                                          )
        """if step_type == "dpmpp_sde_alt":
          x, denoised, denoised2, denoised1_2, vel, vel_2, h_last  = _dpmpp_sde_step_RF(model, x, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                          noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                          extra_args=extra_args,
                                                          momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                          t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2, denoise_boost=0.0,
                                                          )"""

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1_2_3})

        gc.collect()
        torch.cuda.empty_cache()
    return x
        
  elif order == "2c":
    sigma_prev = None
    #denoised = [None] * len(sigmas)
    #x_next   = [None] * len(sigmas)
    
    i=0
    sigma = sigmas[i]
    sigma_next = sigmas[i+1]
    time = sigmas[i] / sigma_max
    x_next, denoised, denoised2, denoised1_2, vel, vel_2, h_last  = _refined_exp_sosu_step_RF(model, x, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                    noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                    extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                    momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                    t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                    )

    for i in trange(1, len(sigmas) - 1, disable=disable):

        sigma_prev = sigmas[i-1]
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        time = sigmas[i] / sigma_max
        
        sigma_next = torch.tensor(0.00001) if sigma_next == 0.0 else sigma_next
        

        x, denoised1_2, vel, vel_2 = _refined_exp_sosu_step_RF_midpoint2(model, x, sigma_prev, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                          noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                          extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                          momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                          t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                          )


        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1_2})

        gc.collect()
        torch.cuda.empty_cache()
    return x

  elif order == "2b":
    sigma_prev = None
    #denoised = [None] * len(sigmas)
    #x_next   = [None] * len(sigmas)
    
    i=0
    sigma = sigmas[i]
    sigma_next = sigmas[i+1]
    time = sigmas[i] / sigma_max
    x_next, denoised, denoised2, denoised1_2, vel, vel_2, h_last  = _refined_exp_sosu_step_RF(model, x, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                    noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                    extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                    momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                    t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                    )

    x_prev = x
    for i in trange(1, len(sigmas) - 1, disable=disable):

        sigma_prev = sigmas[i-1]
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        time = sigmas[i] / sigma_max
        
        sigma_next = torch.tensor(0.00001) if sigma_next == 0.0 else sigma_next
        

        x_next, denoised, denoised2, denoised1_2, vel, vel_2, h_last, x_prev  = _refined_exp_sosu_step_RF_midpoint(model, x_prev, sigma_prev, sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                          noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                          extra_args=extra_args, simple_phi_calc=simple_phi_calc,
                                                          momentum = momentum[i], vel = vel, vel_2 = vel_2, time = time,
                                                          t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                          )


        if callback is not None:
            callback({'x': x_next, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1_2})

        gc.collect()
        torch.cuda.empty_cache()
    return x_next




@no_grad()
def sample_refined_exp_s_advanced_RF_branch(
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
  etas1=None,
  etas2=None,
  eta_vars1=None,
  eta_vars2=None,
  s_noises1=None,
  s_noises2=None,
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
  noisy_cfg=False,
  noise_scale=1.0,
  ancestral_noise=True,
  alpha_ratios=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None,
  latent_self_guide_1=False,
  latent_shift_guide_1=False,
  t_fn_formula="",
  sigma_fn_formula="",
  skip_corrector=False,
  corrector_is_predictor=False,
  order=1,
  auto_c2=False,
  step_type="res_a"
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
  
  if clownseed < 0.0:
    clownseed = extra_args.get("seed", None) + 1
  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)
  print("ClownSeed set to: ", clownseed)

  dt = None
  vel, vel_2 = None, None
  x_hat = None
  
  x_n        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  x_h        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel        = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  vel_2      = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised   = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised2  = [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised1_2= [[None for _ in range(branch_width ** depth)] for depth in range(branch_depth + 1)]
  denoised_  = None
  denoised2_ = None
  denoised1_2= None
  denoised2_prev = None
  h_last = None
  

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

      sigma_hat = sigma * (1 + etas2[i])  #using etas2 as the "main" eta

      x_n[0][0] = x
      for depth in range(1, branch_depth+1):
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        sigma_hat = sigma * (1 + etas2[i])
        
        for m in range(branch_width**(depth-1)):
          for n in range(branch_width):
            idx = m * branch_width + n

            sigma_next = torch.tensor(0.00001) if sigma_next == 0.0 else sigma_next
            x_h[depth][idx] = x_n[depth-1][m]
            if step_type == "res_a":
              x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], denoised1_2, vel[depth][idx], vel_2[depth][idx], h_last = _refined_exp_sosu_step_RF(model, x_h[depth][idx], sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                                            noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                                            extra_args=extra_args, pbar=pbar, simple_phi_calc=simple_phi_calc,
                                                                            momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time,
                                                                            t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2,
                                                                            )
            if step_type == "dpmpp_sde_alt":
              x_n[depth][idx], denoised[depth][idx], denoised2[depth][idx], denoised1_2, vel[depth][idx], vel_2[depth][idx], h_last = _dpmpp_sde_step_RF(model, x_h[depth][idx], sigma, sigma_next, c2=c2[i],eta1=etas1[i], eta2=etas2[i], eta_var1=eta_vars1[i], eta_var2=eta_vars2[i], 
                                                              noise_sampler=noise_sampler, s_noise1=s_noises1[i], s_noise2=s_noises2[i], noise_mode=noise_mode,
                                                              extra_args=extra_args, pbar=pbar, 
                                                              momentum = momentum[i], vel = vel[depth][idx], vel_2 = vel_2[depth][idx], time = time,
                                                              t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula, denoised1_2=denoised1_2, h_last=h_last, order=order, auto_c2=auto_c2, denoise_boost=0.0,
                                                              )

            denoised_  = denoised [depth][idx]
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
      final_eta = etas2[-1] #defaulting to "main" etas2
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