import torch
from torch import no_grad, FloatTensor
from tqdm import tqdm
from itertools import pairwise
from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple
import math

from .noise_classes import *

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

def _gamma(
  n: int,
) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(
  s: int,
  x: float,
  gamma_s: Optional[int] = None
) -> float:
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

def _phi(
  neg_h: float,
  j: int,
):
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

def _de_second_order(
  h: float,
  c2: float,
  simple_phi_calc = False,
) -> RESDECoeffsSecondOrder:
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
  return RESDECoeffsSecondOrder(
    a2_1=a2_1,
    b1=b1,
    b2=b2,
  )  

def _refined_exp_sosu_step(
  model: DenoiserModel,
  x: FloatTensor,
  sigma: FloatTensor,
  sigma_next: FloatTensor,
  c2 = 0.5,
  extra_args: Dict[str, Any] = {},
  pbar: Optional[tqdm] = None,
  simple_phi_calc = False,
  momentum = 0.0,
  vel = None,
  vel_2 = None,
  time = None
) -> StepOutput:

  #Algorithm 1 "RES Second order Single Update Step with c2"
  #https://arxiv.org/abs/2308.02157

  #Parameters:
  #  model (`DenoiserModel`): a k-diffusion wrapped denoiser model (e.g. a subclass of DiscreteEpsDDPMDenoiser)
  #  x (`FloatTensor`): noised latents (or RGB I suppose), e.g. torch.randn((B, C, H, W)) * sigma[0]
  #  sigma (`FloatTensor`): timestep to denoise
  #  sigma_next (`FloatTensor`): timestep+1 to denoise
  #  c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
  #  extra_args (`Dict[str, Any]`, *optional*, defaults to `{}`): kwargs to pass to `model#__call__()`
  #  pbar (`tqdm`, *optional*, defaults to `None`): progress bar to update after each model call
  #  simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.

  def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
    if velocity is None:
        momentum_vel = diff
    else:
        momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
    return momentum_vel

  lam_next, lam = (s.log().neg() for s in (sigma_next, sigma))

  s_in = x.new_ones([x.shape[0]])
  h: float = lam_next - lam
  a2_1, b1, b2 = _de_second_order(h=h, c2=c2, simple_phi_calc=simple_phi_calc)
  
  denoised: FloatTensor = model(x, sigma * s_in, **extra_args)
  if pbar is not None:
    pbar.update(0.5)

  c2_h: float = c2*h

  diff_2 = momentum_func(a2_1*h*denoised, vel_2, time)
  vel_2 = diff_2
  x_2: FloatTensor = math.exp(-c2_h)*x + diff_2
  lam_2: float = lam + c2_h
  sigma_2: float = lam_2.neg().exp()

  denoised2: FloatTensor = model(x_2, sigma_2 * s_in, **extra_args)
  if pbar is not None:
    pbar.update(0.5)

  diff = momentum_func(h*(b1*denoised + b2*denoised2), vel, time)
  vel = diff

  x_next: FloatTensor = math.exp(-h)*x + diff
  
  return StepOutput(
    x_next=x_next,
    denoised=denoised,
    denoised2=denoised2,
    vel=vel,
    vel_2=vel_2,
  )

@cast_fp64
@no_grad()
def sample_refined_exp_s_advanced(
  model: FloatTensor,
  x: FloatTensor,
  sigmas: FloatTensor,
  guide_1: FloatTensor = torch.zeros((1,)),
  guide_2: FloatTensor = torch.zeros((1,)),
  guide_mode_1 = 0,
  guide_mode_2 = 0,
  guide_1_channels=None,
  denoise_to_zero: bool = True,
  extra_args: Dict[str, Any] = {},
  callback: Optional[RefinedExpCallback] = None,
  disable: Optional[bool] = None,
  ita: FloatTensor = torch.zeros((1,)),
  momentum: FloatTensor = torch.zeros((1,)),
  c2: FloatTensor = torch.zeros((1,)),
  offset: FloatTensor = torch.zeros((1,)),
  alpha: FloatTensor = torch.zeros((1,)),
  latent_guide_1: FloatTensor = torch.zeros((1,)),  
  latent_guide_2: FloatTensor = torch.zeros((1,)),  
  noise_sampler: NoiseSampler = torch.randn_like,
  noise_sampler_type=None,
  simple_phi_calc = False,
  k=1.0,
  clownseed=0,
  latent_noise=None
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
    ita (`FloatTensor`, *optional*, defaults to 0.): degree of stochasticity, η, for each timestep. tensor shape must be broadcastable to 1-dimensional tensor with length `len(sigmas) if denoise_to_zero else len(sigmas)-1`. each element should be from 0 to 1.
    c2 (`float`, *optional*, defaults to .5): partial step size for solving ODE. .5 = midpoint method
    noise_sampler (`NoiseSampler`, *optional*, defaults to `torch.randn_like`): method used for adding noise
    simple_phi_calc (`bool`, *optional*, defaults to `True`): True = calculate phi_i,j(-h) via simplified formulae specific to j={1,2}. False = Use general solution that works for any j. Mathematically equivalent, but could be numeric differences.
  """

  #import pdb; pdb.set_trace()
  #assert sigmas[-1] == 0
  sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
  noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=clownseed, sigma_min=sigma_min, sigma_max=sigma_max)

  if latent_guide_1 is not None:
    latent_guide_crushed_1 = (latent_guide_1 - latent_guide_1.min()) / (latent_guide_1 - latent_guide_1.min()).max()
  if latent_guide_2 is not None:
    latent_guide_crushed_2 = (latent_guide_2 - latent_guide_2.min()) / (latent_guide_2 - latent_guide_2.min()).max()

  vel, vel_2 = None, None
  with tqdm(disable=disable, total=len(sigmas)-(1 if denoise_to_zero else 2)) as pbar:
    for i, (sigma, sigma_next) in enumerate(pairwise(sigmas[:-1].split(1))):
      time = sigmas[i] / sigma_max

      if 'sigma' not in locals():
        sigma = sigmas[i]

      if latent_noise is not None:
        if latent_noise.size()[0] == 1:
          eps = latent_noise[0]
        else:
          eps = latent_noise[i]
      else:
        if noise_sampler_type == "power":
          noise_sampler.alpha = alpha[i]
          noise_sampler.k = k

        eps = noise_sampler(sigma=sigma, sigma_next=sigma_next)

      sigma_hat = sigma * (1 + ita[i])
      x_hat = x + ((sigma_hat ** 2 - sigma ** 2).sqrt() * eps)

      x_next, denoised, denoised2, vel, vel_2 = _refined_exp_sosu_step(
        model,
        x_hat,
        sigma_hat,
        sigma_next,
        c2=c2[i],
        extra_args=extra_args,
        pbar=pbar,
        simple_phi_calc=simple_phi_calc,
        momentum = momentum[i],
        vel = vel,
        vel_2 = vel_2,
        time = time
      )
      if callback is not None:
        payload = RefinedExpCallbackPayload(
          x=x,
          i=i,
          sigma=sigma,
          sigma_hat=sigma_hat,
          denoised=denoised,
          denoised2=denoised2,
        )
        callback(payload)

      x = x_next - sigma_next*offset[i]

      if latent_guide_1 is not None:
        if(guide_mode_1 == 1):
          x = x - sigma_next * guide_1[i] * latent_guide_1 * guide_1_channels.view(1,4,1,1)

        if(guide_mode_1 == 2):
          x = x - sigma_next * guide_1[i] * latent_guide_crushed_1 * guide_1_channels.view(1,4,1,1)

        if(guide_mode_1 == 3):
          x = (1 - guide_1[i]) * x * guide_1_channels.view(1,4,1,1) + (guide_1[i] * latent_guide_1 * guide_1_channels.view(1,4,1,1))

        if(guide_mode_1 == 4):
          x = (1 - guide_1[i]) * x * guide_1_channels.view(1,4,1,1) + (guide_1[i] * latent_guide_crushed_1 * guide_1_channels.view(1,4,1,1))   

        if(guide_mode_1 == 5):
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * latent_guide_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 6):
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * latent_guide_crushed_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 7):
          hard_light_blend_1 = hard_light_blend(x, latent_guide_1)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 8):
          hard_light_blend_1 = hard_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 9):
          soft_light_blend_1 = soft_light_blend(x, latent_guide_1)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * soft_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 10):
          soft_light_blend_1 = soft_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * soft_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 11):
          linear_light_blend_1 = linear_light_blend(x, latent_guide_1)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * linear_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 12):
          linear_light_blend_1 = linear_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * linear_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 13):
          vivid_light_blend_1 = vivid_light_blend(x, latent_guide_1)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * vivid_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 14):
          vivid_light_blend_1 = vivid_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * vivid_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 801):
          hard_light_blend_1 = bold_hard_light_blend(x, latent_guide_1)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 802):
          hard_light_blend_1 = bold_hard_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 803):
          hard_light_blend_1 = fix_hard_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 804):
          hard_light_blend_1 = fix2_hard_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 805):
          hard_light_blend_1 = fix3_hard_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 806):
          hard_light_blend_1 = fix4_hard_light_blend(latent_guide_1, x)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))
        if(guide_mode_1 == 807):
          hard_light_blend_1 = fix4_hard_light_blend(x, latent_guide_1)
          x = (x - guide_1[i] * sigma_next * x * guide_1_channels.view(1,4,1,1)) + (guide_1[i] * sigma_next * hard_light_blend_1 * guide_1_channels.view(1,4,1,1))

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

    if denoise_to_zero:
      final_ita = ita[-1]
      eps = noise_sampler(sigma=sigma, sigma_next=sigma_next).double()
      sigma_hat = sigma * (1 + final_ita)
      x_hat = x + (sigma_hat ** 2 - sigma ** 2) ** .5 * eps
      
      s_in = x.new_ones([x.shape[0]])
      x_next: FloatTensor = model(x_hat, torch.zeros_like(sigma).to(x_hat.device) * s_in, **extra_args)
      #x_next: FloatTensor = model(x_hat, sigma.to(x_hat.device), **extra_args)
      pbar.update()
      x = x_next

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
