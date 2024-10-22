import torch
from .noise_classes import *
import comfy.model_patcher


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

def get_ancestral_step(sigma, sigma_next, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_next, 0.
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def get_ancestral_step_backwards(sigma, sigma_next, eta=1.):
    """Calculates sigma_down using the new backward-derived formula and preserves variance."""
    if not eta:
        return sigma_next, 0.
    
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
    sigma_up = torch.sqrt(sigma_next**2 - sigma_down**2)
    return sigma_up, sigma_down

def get_RF_step_traditional(sigma, sigma_next, eta, scale=0.0, alpha_ratio=None):
    # uses math similar to what is used for the get ancestral step code in comfyui. WORKS!
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 

    return sigma_up, sigma_down, alpha_ratio

def get_RF_step(sigma, sigma_next, eta, alpha_ratio=None):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition. Scale is to shape the sigma_down curve."""
    down_ratio = (1 - eta) + eta * ((sigma_next) / sigma)
    sigma_down = down_ratio * sigma_next

    if alpha_ratio is None:
        alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
        
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 # variance preservation is required with RF
    return sigma_up, sigma_down, alpha_ratio


def get_sigma_down_RF(sigma_next, eta):
    eta_scale = torch.sqrt(1 - eta**2)
    sigma_down = (sigma_next * eta_scale) / (1 - sigma_next + sigma_next * eta_scale)
    return sigma_down

def get_sigma_up_RF(sigma_next, eta):
    return sigma_next * eta

def get_ancestral_step_RF_exp(sigma_next, eta, sigma_max=1.0, h=None):
    sigma_up = sigma_next * (1 - (-2*eta*h).exp())**0.5 #or whatever the f
    
    sigma_signal = sigma_max - sigma_next
    sigma_residual = (sigma_next**2 - sigma_up**2)**0.5

    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_correct_working(sigma_next, eta):
    sigma_up = sigma_next * eta
    eta_scaled = (1 - eta**2)**0.5
    sigma_down = (sigma_next * eta_scaled) / (1 - sigma_next + sigma_next * eta_scaled)
    alpha_ratio =                             1 - sigma_next + sigma_next * eta_scaled
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF(sigma_next, eta, sigma_max=1.0):
    sigma_up = sigma_next * eta #or whatever the f
    
    sigma_signal = sigma_max - sigma_next
    sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)

    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_down(sigma_next, sigma_up, sigma_max=1.0, h=None):

    sigma_signal = sigma_max - sigma_next
    sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)

    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    
    return sigma_up, sigma_down, alpha_ratio

def get_res4lyf_step_with_model(model, sigma, sigma_next, eta=0.0, eta_var=1.0, noise_mode="hard", h=None):
  if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
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
      elif noise_mode == "exp": 
        su, sd, alpha_ratio = get_ancestral_step_RF_exp(sigma_next, eta, h)
  else:
    alpha_ratio = 1.0
    if noise_mode == "hard":
      sd = sigma_next
      sigma_hat = sigma * (1 + eta)
      su = (sigma_hat ** 2 - sigma ** 2) ** .5
    if noise_mode == "soft" or noise_mode == "softer": 
      sd, su = get_ancestral_step(sigma, sigma_next, eta)
  return su, sd, alpha_ratio


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

