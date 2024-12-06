import torch
from .noise_classes import *
import comfy.model_patcher

def get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max=1.0):
    if sigma_up >= sigma_next and sigma_next > 0:
      print("Maximum VPSDE noise level exceeded: falling back to hard noise mode.")
      if eta >= 1:
        sigma_up = sigma_next * 0.9999 #avoid sqrt(neg_num) later 
      else:
        sigma_up = sigma_next * eta 
        
    sigma_signal = sigma_max - sigma_next
    sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)

    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    return alpha_ratio, sigma_up, sigma_down

def get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max=1.0):
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down) 
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 
    
    if sigma_up >= sigma_next:
      alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
      
    return alpha_ratio, sigma_up, sigma_down
  

def get_ancestral_step_RF_var(sigma, sigma_next, eta, sigma_max=1.0):
    dtype = sigma.dtype #calculate variance adjusted sigma up... sigma_up = sqrt(dt)

    sigma, sigma_next = sigma.to(torch.float64), sigma_next.to(torch.float64) # float64 is very important to avoid numerical precision issues

    sigma_diff = (sigma - sigma_next).abs() + 1e-10 
    sigma_up = torch.sqrt(sigma_diff).to(torch.float64) * eta

    sigma_down_num = (sigma_next**2 - sigma_up**2).to(torch.float64)
    sigma_down = torch.sqrt(sigma_down_num) / ((1 - sigma_next).to(torch.float64) + torch.sqrt(sigma_down_num).to(torch.float64))

    alpha_ratio = (1 - sigma_next).to(torch.float64) / (1 - sigma_down).to(torch.float64)
    
    return sigma_up.to(dtype),  sigma_down.to(dtype), alpha_ratio.to(dtype)
  
def get_ancestral_step_RF_lorentzian(sigma, sigma_next, eta, sigma_max=1.0):
    dtype = sigma.dtype
    alpha = 1 / ((sigma.to(torch.float64))**2 + 1)
    sigma_up = eta * (1 - alpha) ** 0.5
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up.to(dtype),  sigma_down.to(dtype), alpha_ratio.to(dtype)

def get_ancestral_step_EPS(sigma, sigma_next, eta=1.):
    # Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing an ancestral sampling step.
    alpha_ratio = torch.full_like(sigma, 1.0)
    
    if not eta or not sigma_next:
        return torch.full_like(sigma, 0.0), sigma_next, alpha_ratio
      
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF_softer(sigma, sigma_next, eta, sigma_max=1.0):
    # math adapted from get_ancestral_step_EPS to work with RF
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF_soft(sigma, sigma_next, eta, sigma_max=1.0):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition. Scale is to shape the sigma_down curve."""
    down_ratio = (1 - eta) + eta * ((sigma_next) / sigma)
    sigma_down = down_ratio * sigma_next
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF_exp(sigma_next, eta, h=None, sigma_max=1.0): # TODO: fix black image issue with linear RK
    sigma_up = sigma_next * (1 - (-2*eta*h).exp())**0.5 
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF_sqrd(sigma, sigma_next, eta, sigma_max=1.0):
    sigma_hat = sigma * (1 + eta)
    sigma_up = (sigma_hat ** 2 - sigma ** 2) ** .5
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio

def get_ancestral_step_RF_hard(sigma_next, eta, sigma_max=1.0):
    sigma_up = sigma_next * eta 
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_res4lyf_step_with_model(model, sigma, sigma_next, eta=0.0, eta_var=1.0, noise_mode="hard", h=None):
  if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
    sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
    if noise_mode == "hard_var" and eta_var > 0.0 and sigma_next > sigma_var:
      su, sd, alpha_ratio = get_ancestral_step_RF_var(sigma, sigma_next, eta_var)
    else:
      if   noise_mode == "soft":
        su, sd, alpha_ratio = get_ancestral_step_RF_soft(sigma, sigma_next, eta)
      elif noise_mode == "softer":
        su, sd, alpha_ratio = get_ancestral_step_RF_softer(sigma, sigma_next, eta)
      elif noise_mode == "hard":
        su, sd, alpha_ratio = get_ancestral_step_RF_hard(sigma_next, eta)
      elif noise_mode == "hard_sq": 
        su, sd, alpha_ratio = get_ancestral_step_RF_sqrd(sigma, sigma_next, eta)
      elif noise_mode == "exp": 
        su, sd, alpha_ratio = get_ancestral_step_RF_exp(sigma_next, eta, h)
      elif noise_mode == "lorentzian":
        su, sd, alpha_ratio = get_ancestral_step_RF_lorentzian(sigma, sigma_next, eta)
      else: #fall back to hard noise from hard_var
        su, sd, alpha_ratio = get_ancestral_step_RF_hard(sigma_next, eta)
  else:
    alpha_ratio = torch.full_like(sigma, 1.0)
    if noise_mode == "hard":
      sd = sigma_next
      sigma_hat = sigma * (1 + eta)
      su = (sigma_hat ** 2 - sigma ** 2) ** .5
      sigma = sigma_hat
    if noise_mode == "soft" or noise_mode == "softer": 
      su, sd, alpha_ratio = get_ancestral_step_EPS(sigma, sigma_next, eta)
  
  su = torch.nan_to_num(su, 0.0)
  sd = torch.nan_to_num(sd, float(sigma_next))
  alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)
  
  return su, sigma, sd, alpha_ratio


def get_res4lyf_half_step3(sigma, sigma_next, c2=0.5, c3=1.0, t_fn=None, sigma_fn=None, t_fn_formula="", sigma_fn_formula="", ):

  t_fn_x     = eval(f"lambda sigma: {t_fn_formula}", {"torch": torch}) if t_fn_formula else t_fn
  sigma_fn_x = eval(f"lambda t: {sigma_fn_formula}", {"torch": torch}) if sigma_fn_formula else sigma_fn
      
  t_x, t_next_x = t_fn_x(sigma), t_fn_x(sigma_next)
  h_x = t_next_x - t_x

  s2 = t_x + h_x * c2
  s3 = t_x + h_x * c3
  sigma_2 = sigma_fn_x(s2)
  sigma_3 = sigma_fn_x(s3)

  h = t_fn(sigma_next) - t_fn(sigma)
  c2 = (t_fn(sigma_2) - t_fn(sigma)) / h    
  c3 = (t_fn(sigma_3) - t_fn(sigma)) / h    
  
  return c2, c3


