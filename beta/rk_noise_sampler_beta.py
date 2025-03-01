import torch

import comfy.model_patcher
import comfy.supported_models

from .noise_classes import NOISE_GENERATOR_CLASSES, NOISE_GENERATOR_CLASSES_SIMPLE
from .constants     import MAX_STEPS

from ..helper       import get_extra_options_list, has_nested_attr, extra_options_flag, get_extra_options_kv
from ..latents      import get_orthogonal, get_collinear
from ..res4lyf      import RESplain




NOISE_MODE_NAMES = ["none",
                    "hard_sq",
                    "hard",
                    "lorentzian", 
                    "soft", 
                    "soft-linear",
                    "softer",
                    "eps",
                    "sinusoidal",
                    "exp", 
                    "vpsde",
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
                RK,
                model,
                step=0,
                device='cuda',
                dtype=torch.float64,
                extra_options=""
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
        
        self.DOWN_SUBSTEP           = extra_options_flag("down_substep", extra_options)  
        self.DOWN_STEP              = extra_options_flag("down_step",    extra_options)  



    def init_noise_samplers(self,
                            x,
                            noise_seed,
                            noise_seed_substep,
                            noise_sampler_type,
                            noise_sampler_type2,
                            noise_mode_sde,
                            noise_mode_sde_substep,
                            overshoot_mode,
                            overshoot_mode_substep,
                            noise_boost_step,
                            noise_boost_substep,
                            alpha,
                            alpha2,
                            k      = 1.0,
                            k2     = 1.0,
                            scale  = 0.1,
                            scale2 = 0.1,
                            ):
        
        self.noise_sampler_type     = noise_sampler_type
        self.noise_sampler_type2    = noise_sampler_type2
        self.noise_mode_sde         = noise_mode_sde
        self.noise_mode_sde_substep = noise_mode_sde_substep
        self.overshoot_mode         = overshoot_mode
        self.overshoot_mode_substep = overshoot_mode_substep
        self.noise_boost_step       = noise_boost_step
        self.noise_boost_substep    = noise_boost_substep
        self.s_in                   = x.new_ones([1], dtype=self.dtype, device=self.device)
        
        if noise_seed < 0:
            seed = torch.initial_seed()+1 
            RESplain("SDE noise seed: ", seed, " (set via torch.initial_seed()+1)")
        else:
            seed = noise_seed
            RESplain("SDE noise seed: ", seed)
            
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
            
            
            
    def set_substep_list(self, RK):
        
        self.multistep_stages = RK.multistep_stages
        self.rows = RK.rows
        self.C    = RK.C
        #self.s_ = [(self.sigma_fn(self.t_fn(self.sigma) + self.h*c_)) * self.s_in for c_ in self.C]
        self.s_ = self.sigma_fn(self.t_fn(self.sigma) + self.h * self.C)
    
    
    
    def get_sde_coeff(self, sigma_next, sigma_down=None, sigma_up=None, eta=0.0):
        if self.VARIANCE_PRESERVING:
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



    def set_sde_step(self, sigma, sigma_next, eta, overshoot, s_noise):
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
                        row,
                        multistep_stages,
                        eta_substep,
                        overshoot_substep,
                        s_noise_substep,
                        full_iter           = 0,
                        diag_iter           = 0,
                        implicit_steps_full = 0,
                        implicit_steps_diag = 0
                        ):    
        
        self.sub_sigma_up_eta    = self.sub_sigma_up                          = 0.0
        self.sub_sigma_eta       = self.sub_sigma                             = self.s_[row]
        self.sub_sigma_down_eta  = self.sub_sigma_down  = self.sub_sigma_next = self.s_[row+self.row_offset+multistep_stages]
        self.sub_alpha_ratio_eta = self.sub_alpha_ratio                       = 1.0
        
        self.s_noise_substep     = s_noise_substep
        self.eta_substep         = eta_substep
        self.overshoot_substep   = overshoot_substep


        if row < self.rows   and   self.s_[row+self.row_offset+multistep_stages] > 0:
            if   diag_iter > 0 and diag_iter == implicit_steps_diag and extra_options_flag("implicit_substep_skip_final_eta",  self.extra_options):
                pass
            elif diag_iter > 0 and                                      extra_options_flag("implicit_substep_only_first_eta",  self.extra_options):
                pass
            elif full_iter > 0 and full_iter == implicit_steps_full and extra_options_flag("implicit_step_skip_final_eta",     self.extra_options):
                pass
            elif full_iter > 0 and                                      extra_options_flag("implicit_step_only_first_eta",     self.extra_options):
                pass
            elif (full_iter > 0 or diag_iter > 0)                   and self.noise_sampler_type2 == "brownian":
                pass # brownian noise does not increment its seed when generated, deactivate on implicit repeats to avoid burn
            elif full_iter > 0 and                                      extra_options_flag("implicit_step_only_first_all_eta", self.extra_options):
                self.sigma_down_eta   = self.sigma_next
                self.sigma_up_eta    *= 0
                self.alpha_ratio_eta /= self.alpha_ratio_eta
                
                self.sigma_down       = self.sigma_next
                self.sigma_up        *= 0
                self.alpha_ratio     /= self.alpha_ratio
                
                self.h_new = self.h = self.h_no_eta
                
            elif (row < self.rows-self.row_offset-multistep_stages   or   diag_iter < implicit_steps_diag)   or   extra_options_flag("substep_eta_use_final", self.extra_options):
                self.sub_sigma_up,     self.sub_sigma,     self.sub_sigma_down,     self.sub_alpha_ratio     = self.get_sde_substep(self.s_[row], self.s_[row+self.row_offset+multistep_stages], overshoot_substep, noise_mode_override=self.overshoot_mode_substep, DOWN=self.DOWN_SUBSTEP)
                self.sub_sigma_up_eta, self.sub_sigma_eta, self.sub_sigma_down_eta, self.sub_alpha_ratio_eta = self.get_sde_substep(self.s_[row], self.s_[row+self.row_offset+multistep_stages], eta_substep,       noise_mode_override=self.noise_mode_sde_substep, DOWN=self.DOWN_SUBSTEP)

        self.h_new      = self.h * self.h_fn(self.sub_sigma_down,     self.sigma) / self.h_fn(self.sub_sigma_next, self.sigma) 
        self.h_eta      = self.h * self.h_fn(self.sub_sigma_down_eta, self.sigma) / self.h_fn(self.sub_sigma_next, self.sigma) 
        self.h_new_orig = self.h_new.clone()
        self.h_new      = self.h_new + self.noise_boost_substep * (self.h - self.h_eta)
        
        
        

    def get_sde_substep(self,
                        sigma,
                        sigma_next,
                        eta                 = 0.0,
                        noise_mode_override = None,
                        DOWN                = False,
                        ):
        
        return self.get_sde_step(sigma, sigma_next, eta, noise_mode_override=noise_mode_override, DOWN=DOWN, SUBSTEP=True,)

    def get_sde_step(self,
                    sigma,
                    sigma_next,
                    eta                 = 0.0,
                    noise_mode_override = None,
                    DOWN                = False,
                    SUBSTEP             = False,
                    ):
            
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
                eta_ratio = torch.sin(torch.pi * sigma_next) ** 2
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
                
                if self.VARIANCE_PRESERVING:
                    alpha_ratio, sd, su = self.get_sde_coeff(sigma_next, None, su, eta)
                else:
                    sd          = sigma_next
                    sigma       = sigma_hat
                    alpha_ratio = torch.ones_like(sigma)
                    
            case "vpsde":
                alpha_ratio, sd, su = self.get_vpsde_step_RF(sigma, sigma_next, eta)
                
        if eta_ratio is not None:
            sud = sigma_base * eta_fn(eta_ratio)
            alpha_ratio, sd, su = self.get_sde_coeff(sigma_next, *sud_fn(sud), eta)
        
        su          = torch.nan_to_num(su,          0.0)
        sd          = torch.nan_to_num(sd,    float(sigma_next))
        alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

        return su, sigma, sd, alpha_ratio
    
    def get_vpsde_step_RF(self, sigma, sigma_next, eta):
        dt          = sigma - sigma_next
        sigma_up    = eta * sigma * dt**0.5
        alpha_ratio = 1 - dt * (eta**2/4) * (1 + sigma)
        sigma_down  = sigma_next - (eta/4)*sigma*(1-sigma)*(sigma - sigma_next)
        return sigma_up, sigma_down, alpha_ratio
        

    def swap_noise_step(self, x_0, x_next, brownian_sigma=None, brownian_sigma_next=None, ):
        if self.sigma_up_eta == 0:
            return x_next
        
        if brownian_sigma is None:
            brownian_sigma = self.sigma.clone()
        if brownian_sigma_next is None:
            brownian_sigma_next = self.sigma_next.clone()
        if self.sigma_next == 0:
            return x_next
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
            
        eps_next      = (x_0 - x_next) / (self.sigma - self.sigma_next)
        denoised_next = x_0 - self.sigma * eps_next
        
        if brownian_sigma_next > brownian_sigma:
            s_tmp               = brownian_sigma
            brownian_sigma      = brownian_sigma_next
            brownian_sigma_next = s_tmp
        
        noise = self.noise_sampler(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

        x = self.alpha_ratio_eta * (denoised_next + self.sigma_down_eta * eps_next) + self.sigma_up_eta * noise * self.s_noise
        return x


    def swap_noise_substep(self, x_0, x_next, brownian_sigma=None, brownian_sigma_next=None, ):
        if self.sub_sigma_up_eta == 0:
            return x_next
        
        if brownian_sigma is None:
            brownian_sigma = self.sub_sigma.clone()
        if brownian_sigma_next is None:
            brownian_sigma_next = self.sub_sigma_next.clone()
        if self.sigma_next == 0:
            return x_next
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
        eps_next = (x_0 - x_next) / (self.sigma - self.sub_sigma_next)
        denoised_next = x_0 - self.sigma * eps_next
        
        if brownian_sigma_next > brownian_sigma:
            s_tmp               = brownian_sigma
            brownian_sigma      = brownian_sigma_next
            brownian_sigma_next = s_tmp
        
        noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

        x = self.sub_alpha_ratio_eta * (denoised_next + self.sub_sigma_down_eta * eps_next) + self.sub_sigma_up_eta * noise * self.s_noise_substep
        return x


    def swap_noise(self,
                    x_0,
                    x_next,
                    sigma_0,
                    sigma,
                    sigma_next,
                    sigma_down,
                    sigma_up,
                    alpha_ratio,
                    s_noise,
                    SUBSTEP             = False,
                    brownian_sigma      = None,
                    brownian_sigma_next = None,
                    ):
        
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
            
        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise * s_noise
        return x

    # not used
    def add_noise_pre(self,
                        x_0,
                        x,
                        sigma_up,
                        sigma_0,
                        sigma,
                        sigma_next,
                        real_sigma_down,
                        alpha_ratio,
                        s_noise,
                        noise_mode,
                        SDE_NOISE_EXTERNAL = False,
                        sde_noise_t        = None,
                        SUBSTEP            = False,
                        ):
        
        if not self.CONST and noise_mode == "hard_sq": 
            if self.LOCK_H_SCALE:
                x = self.swap_noise(x_0,
                                    x,
                                    sigma,
                                    sigma_0,
                                    sigma_next,
                                    real_sigma_down,
                                    sigma_up,
                                    alpha_ratio,
                                    s_noise,
                                    SUBSTEP,
                                    )
            else:
                x = self.add_noise(x,
                                    sigma_up,
                                    sigma,
                                    sigma_next,
                                    alpha_ratio,
                                    s_noise,
                                    SDE_NOISE_EXTERNAL,
                                    sde_noise_t,
                                    SUBSTEP,
                                    )
        return x
        
    # only used for handle_tiled_etc_noise_steps() in rk_guide_func_beta.py
    def add_noise_post(self,
                        x_0,
                        x,
                        sigma_up,
                        sigma_0,
                        sigma,
                        sigma_next,
                        real_sigma_down,
                        alpha_ratio,
                        s_noise,
                        noise_mode,
                        SDE_NOISE_EXTERNAL = False,
                        sde_noise_t        = None,
                        SUBSTEP            = False
                        ):
        
        if self.CONST   or   (not self.CONST and noise_mode != "hard_sq"):
            if self.LOCK_H_SCALE:
                x = self.swap_noise(x_0,
                                    x,
                                    sigma_0,
                                    sigma,
                                    sigma_next,
                                    real_sigma_down,
                                    sigma_up,
                                    alpha_ratio,
                                    s_noise,
                                    SUBSTEP,
                                    )
            else:
                x = self.add_noise(x,
                                    sigma_up,
                                    sigma,
                                    sigma_next,
                                    alpha_ratio,
                                    s_noise,
                                    SDE_NOISE_EXTERNAL,
                                    sde_noise_t,
                                    SUBSTEP,
                                    )
        return x

    def add_noise(self,
                    x,
                    sigma_up,
                    sigma,
                    sigma_next,
                    alpha_ratio,
                    s_noise,
                    SDE_NOISE_EXTERNAL,
                    sde_noise_t,
                    SUBSTEP,
                    ):

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
            noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            x_next = alpha_ratio * x + noise * sigma_up * s_noise
            
            return x_next
        
        else:
            return x
    
    def sigma_from_to(self, x_0, x_down, sigma, sigma_down, sigma_next):   #sigma, sigma_from, sigma_to
        eps      = (x_0 - x_down) / (sigma - sigma_down)
        denoised =  x_0 - sigma * eps
        x_next   = denoised + sigma_next * eps
        return x_next

    def rebound_overshoot_step(self, x_0, x):
        eps      = (x_0 - x) / (self.sigma - self.sigma_down)
        denoised =  x_0 - self.sigma * eps
        x        = denoised + self.sigma_next * eps
        return x
    
    def rebound_overshoot_substep(self, x_0, x):
        sub_eps      = (x_0 - x) / (self.sigma - self.sub_sigma_down)
        sub_denoised =  x_0 - self.sigma * sub_eps
        x            = sub_denoised + self.sub_sigma_next * sub_eps
        return x

    def prepare_sigmas(self, sigmas, sigmas_override, d_noise, sampler_mode):
        if sigmas_override is not None:
            sigmas = sigmas_override.clone()
        sigmas = sigmas.clone() * d_noise
        
        if sigmas[0] == 0.0:      #remove padding used to prevent comfy from adding noise to the latent (for unsampling, etc.)
            UNSAMPLE = True
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
        
        self.sigmas       = sigmas
        self.UNSAMPLE     = UNSAMPLE
        self.d_noise      = d_noise
        self.sampler_mode = sampler_mode
        
        return sigmas, UNSAMPLE
    

