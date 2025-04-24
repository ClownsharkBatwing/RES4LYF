import torch

from torch  import Tensor
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .rk_method_beta import RK_Method_Exponential, RK_Method_Linear

import comfy.model_patcher
import comfy.supported_models

from .noise_classes import NOISE_GENERATOR_CLASSES, NOISE_GENERATOR_CLASSES_SIMPLE
from .constants     import MAX_STEPS

from ..helper       import ExtraOptions, has_nested_attr 
from ..latents      import normalize_zscore, get_orthogonal, get_collinear
from ..res4lyf      import RESplain




NOISE_MODE_NAMES = ["none",
                    #"hard_sq",
                    "hard",
                    "lorentzian", 
                    "soft", 
                    "soft-linear",
                    "softer",
                    "eps",
                    "sinusoidal",
                    "exp", 
                    "vpsde",
                    "er4",
                    "hard_var", 
                    ]



def get_data_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (x - x_next) / h



class RK_NoiseSampler:
    def __init__(self,
                RK            : Union["RK_Method_Exponential", "RK_Method_Linear"],
                model,
                step          : int=0,
                device        : str='cuda',
                dtype         : torch.dtype=torch.float64,
                extra_options : str=""
                ):
        
        self.device                 = device
        self.dtype                  = dtype
        
        self.model                  = model

        if has_nested_attr(model, "inner_model.inner_model.model_sampling"):
            model_sampling = model.inner_model.inner_model.model_sampling
        elif has_nested_attr(model, "model.model_sampling"):
            model_sampling = model.model.model_sampling
            
        self.sigma_max              = model_sampling.sigma_max.to(dtype=self.dtype, device=self.device)
        self.sigma_min              = model_sampling.sigma_min.to(dtype=self.dtype, device=self.device)
        
                        
        self.sigma_fn               = RK.sigma_fn
        self.t_fn                   = RK.t_fn
        self.h_fn                   = RK.h_fn

        self.row_offset             = 1 if not RK.IMPLICIT else 0
        
        self.step                   = step
        
        self.noise_sampler          = None
        self.noise_sampler2         = None
        
        self.noise_mode_sde         = None
        self.noise_mode_sde_substep = None
        
        self.LOCK_H_SCALE           = True
        
        self.CONST                  = isinstance(model_sampling, comfy.model_sampling.CONST)
        self.VARIANCE_PRESERVING    = isinstance(model_sampling, comfy.model_sampling.CONST)
        
        self.extra_options          = extra_options
        self.EO                     = ExtraOptions(extra_options)
        
        self.DOWN_SUBSTEP           = self.EO("down_substep")
        self.DOWN_STEP              = self.EO("down_step")
        
        self.init_noise             = None




    def init_noise_samplers(self,
                            x                      : Tensor,              
                            noise_seed             : int,
                            noise_seed_substep     : int,
                            noise_sampler_type     : str,
                            noise_sampler_type2    : str,
                            noise_mode_sde         : str,
                            noise_mode_sde_substep : str,
                            overshoot_mode         : str,
                            overshoot_mode_substep : str,
                            noise_boost_step       : float,
                            noise_boost_substep    : float,
                            alpha                  : float,
                            alpha2                 : float,
                            k                      : float = 1.0,
                            k2                     : float = 1.0,
                            scale                  : float = 0.1,
                            scale2                 : float = 0.1,
                            last_rng                       = None,
                            last_rng_substep               = None,
                            ) -> None:
        
        self.noise_sampler_type     = noise_sampler_type
        self.noise_sampler_type2    = noise_sampler_type2
        self.noise_mode_sde         = noise_mode_sde
        self.noise_mode_sde_substep = noise_mode_sde_substep
        self.overshoot_mode         = overshoot_mode
        self.overshoot_mode_substep = overshoot_mode_substep
        self.noise_boost_step       = noise_boost_step
        self.noise_boost_substep    = noise_boost_substep
        self.s_in                   = x.new_ones([1], dtype=self.dtype, device=self.device)
        
        if noise_seed < 0 and last_rng is None:
            seed = torch.initial_seed()+1 
            RESplain("SDE noise seed: ", seed, " (set via torch.initial_seed()+1)", debug=True)
        if noise_seed < 0 and last_rng is not None:
            seed = torch.initial_seed() 
            RESplain("SDE noise seed: ", seed, " (set via torch.initial_seed())", debug=True)
        else:
            seed = noise_seed
            RESplain("SDE noise seed: ", seed, debug=True)

            
        #seed2 = seed + MAX_STEPS #for substep noise generation. offset needed to ensure seeds are not reused
            
        if noise_sampler_type == "fractal":
            self.noise_sampler        = NOISE_GENERATOR_CLASSES.get(noise_sampler_type )(x=x, seed=seed,               sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler.alpha  = alpha
            self.noise_sampler.k      = k
            self.noise_sampler.scale  = scale
        if noise_sampler_type2 == "fractal":
            self.noise_sampler2       = NOISE_GENERATOR_CLASSES.get(noise_sampler_type2)(x=x, seed=noise_seed_substep, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler2.alpha = alpha2
            self.noise_sampler2.k     = k2
            self.noise_sampler2.scale = scale2
        else:
            self.noise_sampler  = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type )(x=x, seed=seed,               sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler2 = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type2)(x=x, seed=noise_seed_substep, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            
        if last_rng is not None:
            self.noise_sampler .generator.set_state(last_rng)
            self.noise_sampler2.generator.set_state(last_rng_substep)
            
            
    def set_substep_list(self, RK:Union["RK_Method_Exponential", "RK_Method_Linear"]) -> None:
        
        self.multistep_stages = RK.multistep_stages
        self.rows = RK.rows
        self.C    = RK.C
        self.s_ = self.sigma_fn(self.t_fn(self.sigma) + self.h * self.C)
    
    
    def get_substep_list(self, RK:Union["RK_Method_Exponential", "RK_Method_Linear"], sigma, h) -> None:
        s_ = RK.sigma_fn(RK.t_fn(sigma) + h * RK.C)
        return s_
    
    
    def get_sde_coeff(self, sigma_next:Tensor, sigma_down:Tensor=None, sigma_up:Tensor=None, eta:float=0.0, VP_OVERRIDE=None) -> Tuple[Tensor,Tensor,Tensor]:
        VARIANCE_PRESERVING = VP_OVERRIDE if VP_OVERRIDE is not None else self.VARIANCE_PRESERVING

        if VARIANCE_PRESERVING:
            if sigma_down is not None:
                alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
                sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 
                
            elif sigma_up is not None:
                if sigma_up >= sigma_next:
                    RESplain("Maximum VPSDE noise level exceeded: falling back to hard noise mode.")
                    if eta >= 1:
                        sigma_up = sigma_next * 0.9999 #avoid sqrt(neg_num) later 
                    else:
                        sigma_up = sigma_next * eta 
                    
                if VP_OVERRIDE is not None:
                    sigma_signal   =              1 - sigma_next
                else:
                    sigma_signal   = self.sigma_max - sigma_next
                sigma_residual = (sigma_next ** 2 - sigma_up ** 2) ** .5
                alpha_ratio    = sigma_signal + sigma_residual
                sigma_down     = sigma_residual / alpha_ratio     
                
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            
            if sigma_down is not None:
                sigma_up   = (sigma_next ** 2 - sigma_down ** 2) ** .5   # not sure this is correct
            elif sigma_up is not None:
                sigma_down = (sigma_next ** 2 - sigma_up   ** 2) ** .5    
        
        return alpha_ratio, sigma_down, sigma_up



    def set_sde_step(self, sigma:Tensor, sigma_next:Tensor, eta:float, overshoot:float, s_noise:float) -> None:
        self.sigma_0    = sigma
        self.sigma_next = sigma_next
        
        self.s_noise    = s_noise
        self.eta        = eta
        self.overshoot  = overshoot
        
        self.sigma_up_eta, self.sigma_eta, self.sigma_down_eta, self.alpha_ratio_eta \
            = self.get_sde_step(sigma, sigma_next, eta, self.noise_mode_sde, self.DOWN_STEP, SUBSTEP=False)
            
        self.sigma_up, self.sigma, self.sigma_down, self.alpha_ratio \
            = self.get_sde_step(sigma, sigma_next, overshoot, self.overshoot_mode, self.DOWN_STEP, SUBSTEP=False)
        
        self.h          = self.h_fn(self.sigma_down, self.sigma)
        self.h_no_eta   = self.h_fn(self.sigma_next, self.sigma)
        self.h          = self.h + self.noise_boost_step * (self.h_no_eta - self.h)
        

        
        
        
    def set_sde_substep(self,
                        row                 : int,
                        multistep_stages    : int,
                        eta_substep         : float,
                        overshoot_substep   : float,
                        s_noise_substep     : float,
                        full_iter           : int = 0,
                        diag_iter           : int = 0,
                        implicit_steps_full : int = 0,
                        implicit_steps_diag : int = 0
                        ) -> None:    
        
        # start with stepsizes for no overshoot/noise addition/noise swapping
        self.sub_sigma_up_eta    = self.sub_sigma_up                          = 0.0
        self.sub_sigma_eta       = self.sub_sigma                             = self.s_[row]
        self.sub_sigma_down_eta  = self.sub_sigma_down  = self.sub_sigma_next = self.s_[row+self.row_offset+multistep_stages]
        self.sub_alpha_ratio_eta = self.sub_alpha_ratio                       = 1.0
        
        self.s_noise_substep     = s_noise_substep
        self.eta_substep         = eta_substep
        self.overshoot_substep   = overshoot_substep


        if row < self.rows   and   self.s_[row+self.row_offset+multistep_stages] > 0:
            if   diag_iter > 0 and diag_iter == implicit_steps_diag and self.EO("implicit_substep_skip_final_eta"):
                pass
            elif diag_iter > 0 and                                      self.EO("implicit_substep_only_first_eta"):
                pass
            elif full_iter > 0 and full_iter == implicit_steps_full and self.EO("implicit_step_skip_final_eta"):
                pass
            elif full_iter > 0 and                                      self.EO("implicit_step_only_first_eta"):
                pass
            elif (full_iter > 0 or diag_iter > 0)                   and self.noise_sampler_type2 == "brownian":
                pass # brownian noise does not increment its seed when generated, deactivate on implicit repeats to avoid burn
            elif full_iter > 0 and                                      self.EO("implicit_step_only_first_all_eta"):
                self.sigma_down_eta   = self.sigma_next
                self.sigma_up_eta    *= 0
                self.alpha_ratio_eta /= self.alpha_ratio_eta
                
                self.sigma_down       = self.sigma_next
                self.sigma_up        *= 0
                self.alpha_ratio     /= self.alpha_ratio
                
                self.h_new = self.h = self.h_no_eta
            
            elif (row < self.rows-self.row_offset-multistep_stages   or   diag_iter < implicit_steps_diag)   or   self.EO("substep_eta_use_final"):
                self.sub_sigma_up,     self.sub_sigma,     self.sub_sigma_down,     self.sub_alpha_ratio     = self.get_sde_substep(sigma               = self.s_[row],
                                                                                                                                    sigma_next          = self.s_[row+self.row_offset+multistep_stages],
                                                                                                                                    eta                 = overshoot_substep,
                                                                                                                                    noise_mode_override = self.overshoot_mode_substep,
                                                                                                                                    DOWN                = self.DOWN_SUBSTEP)
                
                self.sub_sigma_up_eta, self.sub_sigma_eta, self.sub_sigma_down_eta, self.sub_alpha_ratio_eta = self.get_sde_substep(sigma               = self.s_[row],
                                                                                                                                    sigma_next          = self.s_[row+self.row_offset+multistep_stages],
                                                                                                                                    eta                 = eta_substep,
                                                                                                                                    noise_mode_override = self.noise_mode_sde_substep,
                                                                                                                                    DOWN                = self.DOWN_SUBSTEP)

        if self.h_fn(self.sub_sigma_next, self.sigma) != 0:
            self.h_new      = self.h * self.h_fn(self.sub_sigma_down,     self.sigma) / self.h_fn(self.sub_sigma_next, self.sigma) 
            self.h_eta      = self.h * self.h_fn(self.sub_sigma_down_eta, self.sigma) / self.h_fn(self.sub_sigma_next, self.sigma) 
            self.h_new_orig = self.h_new.clone()
            self.h_new      = self.h_new + self.noise_boost_substep * (self.h - self.h_eta)
        else:
            self.h_new = self.h_eta = self.h
            self.h_new_orig = self.h_new.clone()
        
        
        

    def get_sde_substep(self,
                        sigma               :Tensor,
                        sigma_next          :Tensor,
                        eta                 :float         = 0.0  ,
                        noise_mode_override :Optional[str] = None ,
                        DOWN                :bool          = False,
                        ) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        
        return self.get_sde_step(sigma=sigma, sigma_next=sigma_next, eta=eta, noise_mode_override=noise_mode_override, DOWN=DOWN, SUBSTEP=True,)

    def get_sde_step(self,
                        sigma               :Tensor,
                        sigma_next          :Tensor,
                        eta                 :float         = 0.0  ,
                        noise_mode_override :Optional[str] = None ,
                        DOWN                :bool          = False,
                        SUBSTEP             :bool          = False,
                        VP_OVERRIDE                        = None,
                        ) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        
        VARIANCE_PRESERVING = VP_OVERRIDE if VP_OVERRIDE is not None else self.VARIANCE_PRESERVING
            
        if noise_mode_override is not None:
            noise_mode = noise_mode_override
        elif SUBSTEP:
            noise_mode = self.noise_mode_sde_substep
        else:
            noise_mode = self.noise_mode_sde
        
        if DOWN: #calculates noise level by first scaling sigma_down from sigma_next, instead of sigma_up from sigma_next
            eta_fn = lambda eta_scale: 1-eta_scale
            sud_fn = lambda sd: (sd, None)
        else:
            eta_fn = lambda eta_scale:   eta_scale
            sud_fn = lambda su: (None, su)
        
        su, sd, sud = None, None, None
        eta_ratio   = None
        sigma_base  = sigma_next
        
        sigmax      = self.sigma_max if VP_OVERRIDE is None else 1
        
        match noise_mode:
            case "hard":
                eta_ratio = eta
            case "exp": 
                h = -(sigma_next/sigma).log()
                eta_ratio = (1 - (-2*eta*h).exp())**.5
            case "soft":
                eta_ratio = 1-(1 - eta) + eta * ((sigma_next) / sigma)
            case "softer":
                eta_ratio = 1-torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
            case "soft-linear":
                eta_ratio = 1-eta * (sigma_next - sigma)
            case "sinusoidal":
                eta_ratio = eta * torch.sin(torch.pi * (sigma_next / sigmax)) ** 2
            case "eps":
                eta_ratio = eta * torch.sqrt((sigma_next/sigma) ** 2 * (sigma ** 2 - sigma_next ** 2) ) 
                
            case "lorentzian":
                eta_ratio  = eta
                alpha      = 1 / ((sigma_next.to(sigma.dtype))**2 + 1)
                sigma_base = ((1 - alpha) ** 0.5).to(sigma.dtype)
                
            case "hard_var":
                sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
                if sigma_next > sigma_var:
                    eta_ratio  = 0
                    sigma_base = sigma_next
                else:
                    eta_ratio  = eta
                    sigma_base = torch.sqrt((sigma - sigma_next).abs() + 1e-10)
            
            case "hard_sq":
                sigma_hat = sigma * (1 + eta)
                su        = (sigma_hat ** 2 - sigma ** 2) ** .5    #su
                
                if VARIANCE_PRESERVING:
                    alpha_ratio, sd, su = self.get_sde_coeff(sigma_next, None, su, eta, VARIANCE_PRESERVING)
                else:
                    sd          = sigma_next
                    sigma       = sigma_hat
                    alpha_ratio = torch.ones_like(sigma)
                    
            case "vpsde":
                alpha_ratio, sd, su = self.get_vpsde_step_RF(sigma, sigma_next, eta)
                
            case "er4":
                #def noise_scaler(sigma):
                #    return sigma * ((sigma ** 0.3).exp() + 10.0)
                noise_scaler = lambda sigma: sigma * ((sigma ** eta).exp() + 10.0)
                alpha_ratio = noise_scaler(sigma_next) / noise_scaler(sigma)
                sigma_up    = (sigma_next ** 2 - sigma ** 2 * alpha_ratio ** 2) ** 0.5
                eta_ratio = sigma_up / sigma_next

                
        if eta_ratio is not None:
            sud = sigma_base * eta_fn(eta_ratio)
            alpha_ratio, sd, su = self.get_sde_coeff(sigma_next, *sud_fn(sud), eta, VARIANCE_PRESERVING)
        
        su          = torch.nan_to_num(su,          0.0)
        sd          = torch.nan_to_num(sd,    float(sigma_next))
        alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

        return su, sigma, sd, alpha_ratio
    
    def get_vpsde_step_RF(self, sigma:Tensor, sigma_next:Tensor, eta:float) -> Tuple[Tensor,Tensor,Tensor]:
        dt          = sigma - sigma_next
        sigma_up    = eta * sigma * dt**0.5
        alpha_ratio = 1 - dt * (eta**2/4) * (1 + sigma)
        sigma_down  = sigma_next - (eta/4)*sigma*(1-sigma)*(sigma - sigma_next)
        return sigma_up, sigma_down, alpha_ratio
        
    def linear_noise_init(self, y:Tensor, sigma_curr:Tensor, x_base:Optional[Tensor]=None, x_curr:Optional[Tensor]=None, mask:Optional[Tensor]=None) -> Tensor: 

        y_noised = (self.sigma_max - sigma_curr) * y + sigma_curr * self.init_noise

        if x_curr is not None:
            x_curr = x_curr + sigma_curr * (self.init_noise - y)
            x_base = x_base + self.sigma * (self.init_noise - y)
            return y_noised, x_base, x_curr

        if mask is not None:
            y_noised = mask * y_noised + (1-mask) * y
        
        return y_noised

    def linear_noise_step(self, y:Tensor, sigma_curr:Optional[Tensor]=None, x_base:Optional[Tensor]=None, x_curr:Optional[Tensor]=None, brownian_sigma:Optional[Tensor]=None, brownian_sigma_next:Optional[Tensor]=None, mask:Optional[Tensor]=None) -> Tensor:
        if self.sigma_up_eta == 0   or   self.sigma_next == 0:
            return y, x_base, x_curr
        
        sigma_curr = self.sub_sigma if sigma_curr is None else sigma_curr

        brownian_sigma      = sigma_curr              if brownian_sigma      is None else brownian_sigma
        brownian_sigma_next = self.sigma_next.clone() if brownian_sigma_next is None else brownian_sigma_next
        
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
            
        if brownian_sigma_next > brownian_sigma and not self.EO("disable_brownian_swap"): # should this really be done?
            brownian_sigma, brownian_sigma_next = brownian_sigma_next, brownian_sigma
        
        noise = self.noise_sampler(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = normalize_zscore(noise, channelwise=True, inplace=True)

        y_noised = (self.sigma_max - sigma_curr) * y + sigma_curr * noise
        
        if x_curr is not None:
            x_curr = x_curr + sigma_curr * (noise - y)
            x_base = x_base + self.sigma * (noise - y)
            return y_noised, x_base, x_curr
        
        if mask is not None:
            y_noised = mask * y_noised + (1-mask) * y
        
        return y_noised


    def linear_noise_substep(self, y:Tensor, sigma_curr:Optional[Tensor]=None, x_base:Optional[Tensor]=None, x_curr:Optional[Tensor]=None, brownian_sigma:Optional[Tensor]=None, brownian_sigma_next:Optional[Tensor]=None, mask:Optional[Tensor]=None) -> Tensor:
        if self.sub_sigma_up_eta == 0   or   self.sub_sigma_next == 0:
            return y, x_base, x_curr
        
        sigma_curr = self.sub_sigma if sigma_curr is None else sigma_curr

        brownian_sigma      = sigma_curr                  if brownian_sigma      is None else brownian_sigma
        brownian_sigma_next = self.sub_sigma_next.clone() if brownian_sigma_next is None else brownian_sigma_next

        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999

        if brownian_sigma_next > brownian_sigma and not self.EO("disable_brownian_swap"): # should this really be done?
            brownian_sigma, brownian_sigma_next = brownian_sigma_next, brownian_sigma
        
        noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = normalize_zscore(noise, channelwise=True, inplace=True)

        y_noised = (self.sigma_max - sigma_curr) * y + sigma_curr * noise
        
        if x_curr is not None:
            x_curr = x_curr + sigma_curr * (noise - y)
            x_base = x_base + self.sigma * (noise - y)
            return y_noised, x_base, x_curr
        
        if mask is not None:
            y_noised = mask * y_noised + (1-mask) * y
        
        return y_noised


    def swap_noise_step(self, x_0:Tensor, x_next:Tensor, brownian_sigma:Optional[Tensor]=None, brownian_sigma_next:Optional[Tensor]=None, mask:Optional[Tensor]=None) -> Tensor:
        if self.sigma_up_eta == 0   or   self.sigma_next == 0:
            return x_next

        brownian_sigma      = self.sigma.clone()      if brownian_sigma      is None else brownian_sigma
        brownian_sigma_next = self.sigma_next.clone() if brownian_sigma_next is None else brownian_sigma_next
        
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
            
        eps_next      = (x_0 - x_next) / (self.sigma - self.sigma_next)
        denoised_next = x_0 - self.sigma * eps_next
        
        if brownian_sigma_next > brownian_sigma and not self.EO("disable_brownian_swap"): # should this really be done?
            brownian_sigma, brownian_sigma_next = brownian_sigma_next, brownian_sigma
        
        noise = self.noise_sampler(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = normalize_zscore(noise, channelwise=True, inplace=True)

        x_noised = self.alpha_ratio_eta * (denoised_next + self.sigma_down_eta * eps_next) + self.sigma_up_eta * noise * self.s_noise

        if mask is not None:
            x = mask * x_noised + (1-mask) * x_next
        else:
            x = x_noised
        
        return x


    def swap_noise_substep(self, x_0:Tensor, x_next:Tensor, brownian_sigma:Optional[Tensor]=None, brownian_sigma_next:Optional[Tensor]=None, mask:Optional[Tensor]=None, guide:Optional[Tensor]=None) -> Tensor:
        if self.sub_sigma_up_eta == 0   or   self.sub_sigma_next == 0:
            return x_next
        
        brownian_sigma      = self.sub_sigma.clone()      if brownian_sigma      is None else brownian_sigma
        brownian_sigma_next = self.sub_sigma_next.clone() if brownian_sigma_next is None else brownian_sigma_next

        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
            
        eps_next      = (x_0 - x_next) / (self.sigma - self.sub_sigma_next)
        denoised_next = x_0 - self.sigma * eps_next
        
        if brownian_sigma_next > brownian_sigma and not self.EO("disable_brownian_swap"): # should this really be done?
            brownian_sigma, brownian_sigma_next = brownian_sigma_next, brownian_sigma
        
        noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = normalize_zscore(noise, channelwise=True, inplace=True)

        x_noised = self.sub_alpha_ratio_eta * (denoised_next + self.sub_sigma_down_eta * eps_next) + self.sub_sigma_up_eta * noise * self.s_noise_substep

        if mask is not None:
            x = mask * x_noised + (1-mask) * x_next
        else:
            x = x_noised
        
        return x




    def swap_noise_inv_substep(self, x_0:Tensor, x_next:Tensor, eta_substep:float, row:int, row_offset_multistep_stages:int, brownian_sigma:Optional[Tensor]=None, brownian_sigma_next:Optional[Tensor]=None, mask:Optional[Tensor]=None, guide:Optional[Tensor]=None) -> Tensor:
        if self.sub_sigma_up_eta == 0   or   self.sub_sigma_next == 0:
            return x_next
        
        brownian_sigma      = self.sub_sigma.clone()      if brownian_sigma      is None else brownian_sigma
        brownian_sigma_next = self.sub_sigma_next.clone() if brownian_sigma_next is None else brownian_sigma_next

        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
            
        eps_next      = (x_0 - x_next) / ((1-self.sigma) - (1-self.sub_sigma_next))
        denoised_next = x_0 - (1-self.sigma) * eps_next
        
        if brownian_sigma_next > brownian_sigma and not self.EO("disable_brownian_swap"): # should this really be done?
            brownian_sigma, brownian_sigma_next = brownian_sigma_next, brownian_sigma
        
        noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = normalize_zscore(noise, channelwise=True, inplace=True)
        
        sub_sigma_up,     sub_sigma,     sub_sigma_down,     sub_alpha_ratio     = self.get_sde_substep(sigma               = 1-self.s_[row],
                                                                                                                            sigma_next          = 1-self.s_[row_offset_multistep_stages],
                                                                                                                            eta                 = eta_substep,
                                                                                                                            noise_mode_override = self.noise_mode_sde_substep,
                                                                                                                            DOWN                = self.DOWN_SUBSTEP)

        x_noised = sub_alpha_ratio * (denoised_next + sub_sigma_down * eps_next) + sub_sigma_up * noise * self.s_noise_substep

        if mask is not None:
            x = mask * x_noised + (1-mask) * x_next
        else:
            x = x_noised
        
        return x


    def swap_noise(self,
                    x_0                 :Tensor,
                    x_next              :Tensor,
                    sigma_0             :Tensor,
                    sigma               :Tensor,
                    sigma_next          :Tensor,
                    sigma_down          :Tensor,
                    sigma_up            :Tensor,
                    alpha_ratio         :Tensor,
                    s_noise             :float,
                    SUBSTEP             :bool             = False,
                    brownian_sigma      :Optional[Tensor] = None,
                    brownian_sigma_next :Optional[Tensor] = None,
                    ) -> Tensor:
        
        if sigma_up == 0:
            return x_next
        
        if brownian_sigma is None:
            brownian_sigma = sigma.clone()
        if brownian_sigma_next is None:
            brownian_sigma_next = sigma_next.clone()
        if sigma_next == 0:
            return x_next
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
        eps_next      = (x_0 - x_next) / (sigma_0 - sigma_next)
        denoised_next = x_0 - sigma_0 * eps_next
        
        if brownian_sigma_next > brownian_sigma:
            s_tmp               = brownian_sigma
            brownian_sigma      = brownian_sigma_next
            brownian_sigma_next = s_tmp
        
        if not SUBSTEP:
            noise = self.noise_sampler(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        else:
            noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
            
        noise = normalize_zscore(noise, channelwise=True, inplace=True)

        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise * s_noise
        return x

    # not used. WARNING: some parameters have a different order than swap_noise!
    def add_noise_pre(self,
                        x_0                :Tensor,
                        x                  :Tensor,
                        sigma_up           :Tensor,
                        sigma_0            :Tensor,
                        sigma              :Tensor,
                        sigma_next         :Tensor,
                        real_sigma_down    :Tensor,
                        alpha_ratio        :Tensor,
                        s_noise            :float,
                        noise_mode         :str,
                        SDE_NOISE_EXTERNAL :bool             = False,
                        sde_noise_t        :Optional[Tensor] = None,
                        SUBSTEP            :bool             = False,
                        ) -> Tensor:
        
        if not self.CONST and noise_mode == "hard_sq": 
            if self.LOCK_H_SCALE:
                x = self.swap_noise(x_0             = x_0,
                                    x               = x,
                                    sigma           = sigma,
                                    sigma_0         = sigma_0,
                                    sigma_next      = sigma_next,
                                    real_sigma_down = real_sigma_down,
                                    sigma_up        = sigma_up,
                                    alpha_ratio     = alpha_ratio,
                                    s_noise         = s_noise,
                                    SUBSTEP         = SUBSTEP,
                                    )
            else:
                x = self.add_noise( x                  = x,
                                    sigma_up           = sigma_up,
                                    sigma              = sigma,
                                    sigma_next         = sigma_next,
                                    alpha_ratio        = alpha_ratio,
                                    s_noise            = s_noise,
                                    SDE_NOISE_EXTERNAL = SDE_NOISE_EXTERNAL,
                                    sde_noise_t        = sde_noise_t,
                                    SUBSTEP            = SUBSTEP,
                                    )
                
        return x
        
    # only used for handle_tiled_etc_noise_steps() in rk_guide_func_beta.py
    def add_noise_post(self,
                        x_0                :Tensor,
                        x                  :Tensor,
                        sigma_up           :Tensor,
                        sigma_0            :Tensor,
                        sigma              :Tensor,
                        sigma_next         :Tensor,
                        real_sigma_down    :Tensor,
                        alpha_ratio        :Tensor,
                        s_noise            :float,
                        noise_mode         :str,
                        SDE_NOISE_EXTERNAL :bool             = False,
                        sde_noise_t        :Optional[Tensor] = None,
                        SUBSTEP            :bool             = False,
                        ) -> Tensor:
        
        if self.CONST   or   (not self.CONST and noise_mode != "hard_sq"):
            if self.LOCK_H_SCALE:
                x = self.swap_noise(x_0             = x_0,
                                    x               = x,
                                    sigma           = sigma,
                                    sigma_0         = sigma_0,
                                    sigma_next      = sigma_next,
                                    real_sigma_down = real_sigma_down,
                                    sigma_up        = sigma_up,
                                    alpha_ratio     = alpha_ratio,
                                    s_noise         = s_noise,
                                    SUBSTEP         = SUBSTEP,
                                    )
            else:
                x = self.add_noise( x                  = x,
                                    sigma_up           = sigma_up,
                                    sigma              = sigma,
                                    sigma_next         = sigma_next,
                                    alpha_ratio        = alpha_ratio,
                                    s_noise            = s_noise,
                                    SDE_NOISE_EXTERNAL = SDE_NOISE_EXTERNAL,
                                    sde_noise_t        = sde_noise_t,
                                    SUBSTEP            = SUBSTEP,
                                    )
        return x

    def add_noise(self,
                    x                 :Tensor,
                    sigma_up          :Tensor,
                    sigma             :Tensor,
                    sigma_next        :Tensor,
                    alpha_ratio       :Tensor,
                    s_noise           :float,
                    SDE_NOISE_EXTERNAL :bool             = False,
                    sde_noise_t        :Optional[Tensor] = None,
                    SUBSTEP            :bool             = False,
                    ) -> Tensor:

        if sigma_next > 0.0 and sigma_up > 0.0:
            if sigma_next > sigma:
                sigma, sigma_next = sigma_next, sigma
            
            if sigma == sigma_next:
                sigma_next = sigma * 0.9999
            if not SUBSTEP:
                noise = self.noise_sampler (sigma=sigma, sigma_next=sigma_next)
            else:
                noise = self.noise_sampler2(sigma=sigma, sigma_next=sigma_next)

            #noise_ortho = get_orthogonal(noise, x)
            #noise_ortho = noise_ortho / noise_ortho.std()model,
            noise = normalize_zscore(noise, channelwise=True, inplace=True)

            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            x_next = alpha_ratio * x + noise * sigma_up * s_noise
            
            return x_next
        
        else:
            return x
    
    def sigma_from_to(self,
                        x_0        : Tensor,
                        x_down     : Tensor,
                        sigma      : Tensor,
                        sigma_down : Tensor,
                        sigma_next : Tensor) -> Tensor:   #sigma, sigma_from, sigma_to
        
        eps      = (x_0 - x_down) / (sigma - sigma_down)
        denoised =  x_0 - sigma * eps
        x_next   = denoised + sigma_next * eps
        return x_next

    def rebound_overshoot_step(self, x_0:Tensor, x:Tensor) -> Tensor:
        eps      = (x_0 - x) / (self.sigma - self.sigma_down)
        denoised =  x_0 - self.sigma * eps
        x        = denoised + self.sigma_next * eps
        return x
    
    def rebound_overshoot_substep(self, x_0:Tensor, x:Tensor) -> Tensor:
        if self.sigma - self.sub_sigma_down > 0:
            sub_eps      = (x_0 - x) / (self.sigma - self.sub_sigma_down)
            sub_denoised =  x_0 - self.sigma * sub_eps
            x            = sub_denoised + self.sub_sigma_next * sub_eps
        return x

    def prepare_sigmas(self,
                        sigmas             : Tensor,
                        sigmas_override    : Tensor,
                        d_noise            : float,
                        d_noise_start_step : int,
                        sampler_mode       : str) -> Tuple[Tensor,bool]:
        
        if sigmas_override is not None:
            sigmas = sigmas_override.clone().to(sigmas.device).to(sigmas.dtype)
            
        if d_noise_start_step == 0:
            sigmas = sigmas.clone() * d_noise
            
        UNSAMPLE_FROM_ZERO = False
        if sigmas[0] == 0.0:      #remove padding used to prevent comfy from adding noise to the latent (for unsampling, etc.)
            UNSAMPLE = True
            if sigmas[-1] == 0.0:
                UNSAMPLE_FROM_ZERO = True
            sigmas   = sigmas[1:-1]
        else:
            UNSAMPLE = False
        
        if hasattr(self.model, "sigmas"):
            self.model.sigmas = sigmas
            
        if sampler_mode == "standard":
            UNSAMPLE = False
        
        consecutive_duplicate_mask = torch.cat((torch.tensor([True], device=sigmas.device), torch.diff(sigmas) != 0))
        sigmas = sigmas[consecutive_duplicate_mask]
                
        if sigmas[-1] == 0:
            if sigmas[-2] < self.sigma_min:
                sigmas[-2] = self.sigma_min
            elif (sigmas[-2] - self.sigma_min).abs() > 1e-4:
                sigmas = torch.cat((sigmas[:-1], self.sigma_min.unsqueeze(0), sigmas[-1:]))
                
        elif UNSAMPLE_FROM_ZERO and not torch.isclose(sigmas[0], self.sigma_min):
            sigmas = torch.cat([self.sigma_min.unsqueeze(0), sigmas])
        
        self.sigmas       = sigmas
        self.UNSAMPLE     = UNSAMPLE
        self.d_noise      = d_noise
        self.sampler_mode = sampler_mode
        
        return sigmas, UNSAMPLE
    
    