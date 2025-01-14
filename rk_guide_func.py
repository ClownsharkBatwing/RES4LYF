import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re

from einops import rearrange

from .noise_classes import *
from .latents import hard_light_blend, normalize_latent, initialize_or_scale
from .rk_method import RK_Method
from .helper import get_extra_options_kv, extra_options_flag, get_cosine_similarity


import itertools


def normalize_inputs(x, y0, y0_inv, guide_mode,  extra_options):
    
    if guide_mode == "epsilon_guide_mean_std_from_bkg":
        y0 = normalize_latent(y0, y0_inv)
        
    input_norm = get_extra_options_kv("input_norm", "", extra_options)
    input_std = float(get_extra_options_kv("input_std", "1.0", extra_options))
    
    if input_norm == "input_ch_mean_set_std_to":
        x = normalize_latent(x, set_std=input_std)

    if input_norm == "input_ch_set_std_to":
        x = normalize_latent(x, set_std=input_std, mean=False)
            
    if input_norm == "input_mean_set_std_to":
        x = normalize_latent(x, set_std=input_std, channelwise=False)
        
    if input_norm == "input_std_set_std_to":
        x = normalize_latent(x, set_std=input_std, mean=False, channelwise=False)
    
    return x, y0, y0_inv


class LatentGuide:
    def __init__(self, guides, x, model, sigmas, UNSAMPLE, LGW_MASK_RESCALE_MIN, extra_options, device='cuda', dtype=torch.float64, max_steps=10000):
        self.model    = model
        self.sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(dtype)
        self.sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(dtype)
        self.sigmas   = sigmas
        self.UNSAMPLE = UNSAMPLE
        self.SAMPLE = (sigmas[0] > sigmas[1])
        self.extra_options = extra_options
        self.y0     = torch.zeros_like(x)
        self.y0_inv = torch.zeros_like(x)
        self.guide_mode = ""
        self.mask = None
        self.mask_inv = None
        
        self.latent_guide = None
        self.latent_guide_inv = None

        self.lgw_masks = []
        self.lgw_masks_inv = []
        self.lgw, self.lgw_inv = [torch.full_like(sigmas, 0.) for _ in range(2)]
        
        self.guide_cossim_cutoff_, self.guide_bkg_cossim_cutoff_ = 1.0, 1.0
                
        latent_guide_weight, latent_guide_weight_inv = 0.,0.
        latent_guide_weights, latent_guide_weights_inv = None, None
        if guides is not None:
            self.guide_mode, latent_guide_weight, latent_guide_weight_inv, latent_guide_weights, latent_guide_weights_inv, self.latent_guide, self.latent_guide_inv, latent_guide_mask, latent_guide_mask_inv, scheduler_, scheduler_inv_, steps_, steps_inv_, denoise_, denoise_inv_ = guides
            
            self.mask, self.mask_inv                                 = latent_guide_mask, latent_guide_mask_inv
            self.guide_cossim_cutoff_, self.guide_bkg_cossim_cutoff_ = denoise_, denoise_inv_
            
        latent_guide_weights     = initialize_or_scale(latent_guide_weights,     latent_guide_weight,     max_steps).to(dtype)
        latent_guide_weights_inv = initialize_or_scale(latent_guide_weights_inv, latent_guide_weight_inv, max_steps).to(dtype)

        latent_guide_weights     = F.pad(latent_guide_weights,     (0, max_steps), value=0.0)
        latent_guide_weights_inv = F.pad(latent_guide_weights_inv, (0, max_steps), value=0.0)
        
        
        if latent_guide_weights is not None:
            self.lgw = latent_guide_weights.to(x.device)
        if latent_guide_weights_inv is not None:
            self.lgw_inv = latent_guide_weights_inv.to(x.device)
            
        self.mask, LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask, LGW_MASK_RESCALE_MIN)
        if self.mask_inv is not None:
            self.mask_inv, LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_inv, LGW_MASK_RESCALE_MIN)
        elif not self.SAMPLE:
            self.mask_inv = (1-self.mask)
            
        for step in range(len(self.sigmas)-1):
            lgw_mask, lgw_mask_inv = prepare_weighted_masks(self.mask, self.mask_inv, self.lgw[step], self.lgw_inv[step], self.latent_guide, self.latent_guide_inv, LGW_MASK_RESCALE_MIN)
            self.lgw_masks.append(lgw_mask)
            self.lgw_masks_inv.append(lgw_mask_inv)


    def init_guides(self, x, noise_sampler, latent_guide=None, latent_guide_inv=None):
        self.y0, self.y0_inv = torch.zeros_like(x), torch.zeros_like(x)
        latent_guide = self.latent_guide if latent_guide is None else latent_guide
        latent_guide_inv = self.latent_guide_inv if latent_guide_inv is None else latent_guide_inv


        if latent_guide is not None:
            if type(latent_guide) == dict:
                latent_guide_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
            else:
                latent_guide_samples = latent_guide
            if self.SAMPLE:
                self.y0 = latent_guide_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                x = (1-self.mask) * x + self.mask * latent_guide_samples
            else:
                x = latent_guide_samples

        if latent_guide_inv is not None:
            if type(latent_guide_inv) == dict:
                latent_guide_inv_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)
            else:
                latent_guide_inv_samples = latent_guide_inv
            if self.SAMPLE:
                self.y0_inv = latent_guide_inv_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                x = (1-self.mask_inv) * x + self.mask_inv * latent_guide_inv_samples #fixed old approach, which was mask, (1-mask)
            else:
                x = latent_guide_inv_samples   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_INV AFTER SETTING TO LG above!
                
        if self.UNSAMPLE and not self.SAMPLE: #sigma_next > sigma:
            self.y0 = noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min)
            self.y0 = (self.y0 - self.y0.mean()) / self.y0.std()
            self.y0_inv = noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min)
            self.y0_inv = (self.y0_inv - self.y0_inv.mean()) / self.y0_inv.std()
            
        x, self.y0, self.y0_inv = normalize_inputs(x, self.y0, self.y0_inv, self.guide_mode, self.extra_options)

        return x
    


    def process_guides_substep(self, x_0, x_, eps_, data_, row, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, extra_options, frame_weights=None):

        y0 = self.y0
        if self.y0.shape[0] > 1:
            y0 = self.y0[min(step, self.y0.shape[0]-1)].unsqueeze(0)  
        y0_inv = self.y0_inv
        
        lgw_mask = self.lgw_masks[step].clone()
        lgw_mask_inv = self.lgw_masks_inv[step].clone() if self.lgw_masks_inv is not None else None
        
        lgw = self.lgw[step]
        lgw_inv = self.lgw_inv[step]
        
        latent_guide = self.latent_guide
        latent_guide_inv = self.latent_guide_inv
        guide_mode = self.guide_mode
        UNSAMPLE = self.UNSAMPLE

        if self.guide_mode: 
            data_norm   = data_[row] - data_[row].mean(dim=(-2,-1), keepdim=True)
            y0_norm     = y0         -         y0.mean(dim=(-2,-1), keepdim=True)
            y0_inv_norm = y0_inv     -     y0_inv.mean(dim=(-2,-1), keepdim=True)

            y0_cossim     = get_cosine_similarity(data_norm*lgw_mask,     y0_norm    *lgw_mask)
            y0_cossim_inv = get_cosine_similarity(data_norm*lgw_mask_inv, y0_inv_norm*lgw_mask_inv)
            
            if y0_cossim < self.guide_cossim_cutoff_ or y0_cossim_inv < self.guide_bkg_cossim_cutoff_:
                lgw_mask_cossim, lgw_mask_cossim_inv = lgw_mask, lgw_mask_inv
                if y0_cossim     >= self.guide_cossim_cutoff_:
                    lgw_mask_cossim     = torch.zeros_like(lgw_mask)
                if y0_cossim_inv >= self.guide_bkg_cossim_cutoff_:
                    lgw_mask_cossim_inv = torch.zeros_like(lgw_mask_inv)
                lgw_mask = lgw_mask_cossim
                lgw_mask_inv = lgw_mask_cossim_inv
            else:
                return eps_, x_ 
        else:
            return eps_, x_ 
        
        if self.UNSAMPLE and RK_Method.is_exponential(rk_type):
            if not (extra_options_flag("disable_power_unsample", extra_options) or extra_options_flag("disable_power_resample", extra_options)):
                extra_options += "\npower_unsample\npower_resample\n"
            if not extra_options_flag("disable_lgw_scaling_substep_ch_mean_std", extra_options):
                extra_options += "\nsubstep_eps_ch_mean_std\n"
                


        s_in = x_0.new_ones([x_0.shape[0]])
        eps_orig = eps_.clone()
        
        if extra_options_flag("dynamic_guides_mean_std", extra_options):
            y_shift, y_inv_shift = normalize_latent([y0, y0_inv], [data_, data_])
            y0 = y_shift
            if extra_options_flag("dynamic_guides_inv", extra_options):
                y0_inv = y_inv_shift

        if extra_options_flag("dynamic_guides_mean", extra_options):
            y_shift, y_inv_shift = normalize_latent([y0, y0_inv], [data_, data_], std=False)
            y0 = y_shift
            if extra_options_flag("dynamic_guides_inv", extra_options):
                y0_inv = y_inv_shift


        if frame_weights is not None and x_0.dim() == 5:
            for f in range(lgw_mask.shape[2]):
                frame_weight = frame_weights[f]
                lgw_mask[..., f:f+1, :, :] *= frame_weight
                if lgw_mask_inv is not None:
                    lgw_mask_inv[..., f:f+1, :, :] *= frame_weight

        if "data" == guide_mode:
            y0_tmp = y0.clone()
            if latent_guide_inv is not None:
                y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
                y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
            x_[row+1] = y0_tmp + eps_[row]
            
        if guide_mode == "data_projection":

            d_lerp = data_[row]   +   lgw_mask * (y0-data_[row])   +   lgw_mask_inv * (y0_inv-data_[row])
            
            d_collinear_d_lerp = get_collinear(data_[row], d_lerp)  
            d_lerp_ortho_d     = get_orthogonal(d_lerp, data_[row])  
            
            data_[row] = d_collinear_d_lerp + d_lerp_ortho_d
            
            x_[row+1] = data_[row] + eps_[row] * sigma
            

        elif "epsilon" in guide_mode:
            if sigma > sigma_next:
                    
                tol_value = float(get_extra_options_kv("tol", "-1.0", extra_options))
                if tol_value >= 0 and (lgw > 0 or lgw_inv > 0):           
                    for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                        current_diff     = torch.norm(data_[row][b][c] - y0    [b][c])
                        current_diff_inv = torch.norm(data_[row][b][c] - y0_inv[b][c])
                        
                        lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                        lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                        
                        lgw_tmp     = min(lgw    , lgw_scaled)
                        lgw_tmp_inv = min(lgw_inv, lgw_scaled_inv)

                        lgw_mask_clamp     = torch.clamp(lgw_mask,     max=lgw_tmp)
                        lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)
                        
                        eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type, b, c)
                        eps_[row][b][c] = eps_[row][b][c] + lgw_mask_clamp[b][c] * (eps_row - eps_[row][b][c]) + lgw_mask_clamp_inv[b][c] * (eps_row_inv - eps_[row][b][c])


                elif guide_mode == "epsilon_projection":
                    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                    
                    if extra_options_flag("eps_proj_v2", extra_options):
                        
                        eps_row_lerp_fg = eps_[row]   +   lgw_mask * (eps_row-eps_[row])
                        eps_row_lerp_bg = eps_[row]   +   lgw_mask_inv * (eps_row_inv-eps_[row])
                        
                        eps_collinear_eps_lerp_fg = get_collinear(eps_[row], eps_row_lerp_fg)  
                        eps_lerp_ortho_eps_fg     = get_orthogonal(eps_row_lerp_fg, eps_[row])  
                        
                        eps_collinear_eps_lerp_bg = get_collinear(eps_[row], eps_row_lerp_bg)  
                        eps_lerp_ortho_eps_bg     = get_orthogonal(eps_row_lerp_bg, eps_[row])  
                        
                        eps_[row] = eps_[row] + lgw_mask * (eps_collinear_eps_lerp_fg + eps_lerp_ortho_eps_fg - eps_[row]) + lgw_mask_inv * (eps_collinear_eps_lerp_bg + eps_lerp_ortho_eps_bg - eps_[row]) 
                        
                    elif extra_options_flag("eps_proj_v3", extra_options):

                        eps_collinear_eps_lerp_fg = get_collinear(eps_[row], eps_row)  
                        eps_lerp_ortho_eps_fg     = get_orthogonal(eps_row, eps_[row])  
                        
                        eps_collinear_eps_lerp_bg = get_collinear(eps_[row], eps_row_inv)  
                        eps_lerp_ortho_eps_bg     = get_orthogonal(eps_row_inv, eps_[row])  
                        
                        eps_[row] = eps_[row] + lgw_mask * (eps_collinear_eps_lerp_fg + eps_lerp_ortho_eps_fg - eps_[row]) + lgw_mask_inv * (eps_collinear_eps_lerp_bg + eps_lerp_ortho_eps_bg - eps_[row]) 
                       
                    elif extra_options_flag("eps_proj_v5", extra_options):

                        eps2g_collin = get_collinear(eps_[row], eps_row)  
                        g2eps_ortho  = get_orthogonal(eps_row, eps_[row])  
                        
                        g2eps_collin = get_collinear(eps_row, eps_[row])  
                        eps2g_ortho  = get_orthogonal(eps_[row], eps_row)  
                        
                        eps2i_collin = get_collinear(eps_[row], eps_row_inv)  
                        i2eps_ortho  = get_orthogonal(eps_row_inv, eps_[row])  
                        
                        i2eps_collin = get_collinear(eps_row_inv, eps_[row])  
                        eps2i_ortho  = get_orthogonal(eps_[row], eps_row_inv)  
                        
                        #eps_[row] = (eps2g_collin+g2eps_ortho)   +   (g2eps_collin+eps2g_ortho)       +       (eps2i_collin+i2eps_ortho)   +   (i2eps_collin+eps2i_ortho)
                        #eps_[row] = eps_[row] + lgw_mask * (eps2g_collin+g2eps_ortho)   +   (1-lgw_mask) * (g2eps_collin+eps2g_ortho)       +      lgw_mask_inv * (eps2i_collin+i2eps_ortho)   +   (1-lgw_mask_inv) * (i2eps_collin+eps2i_ortho)

                        eps_[row] = lgw_mask * (eps2g_collin+g2eps_ortho)   -   lgw_mask * (g2eps_collin+eps2g_ortho)       +      lgw_mask_inv * (eps2i_collin+i2eps_ortho)   -   lgw_mask_inv * (i2eps_collin+eps2i_ortho)
                        
                        #eps_[row] = eps_[row] + lgw_mask * (eps_collinear_eps_lerp_fg + eps_lerp_ortho_eps_fg - eps_[row]) + lgw_mask_inv * (eps_collinear_eps_lerp_bg + eps_lerp_ortho_eps_bg - eps_[row]) 
                       
                       
                    elif extra_options_flag("eps_proj_v4a", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_[row] = (1 - torch.clamp(lgw_mask + lgw_mask_inv, max=1.0)) * eps_[row]   +   torch.clamp((lgw_mask + lgw_mask_inv), max=1.0) * (eps_collinear_eps_lerp + eps_lerp_ortho_eps)


                    elif extra_options_flag("eps_proj_v4b", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_[row] = (1 - (lgw_mask + lgw_mask_inv)/2) * eps_[row]   +   ((lgw_mask + lgw_mask_inv)/2) * (eps_collinear_eps_lerp + eps_lerp_ortho_eps)

                    elif extra_options_flag("eps_proj_v4c", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        lgw_mask_sum = (lgw_mask + lgw_mask_inv)


                        eps_[row] = (1 - (lgw_mask + lgw_mask_inv)/2) * eps_[row]   +   ((lgw_mask + lgw_mask_inv)/2) * (eps_collinear_eps_lerp + eps_lerp_ortho_eps)

                    elif extra_options_flag("eps_proj_v4e", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_sum = eps_collinear_eps_lerp + eps_lerp_ortho_eps

                        eps_[row] = eps_[row] + self.mask * (eps_sum - eps_[row]) + self.mask_inv * (eps_sum - eps_[row])

                    elif extra_options_flag("eps_proj_self1", extra_options):
                        eps_row_lerp = eps_[row]   +   self.mask * (eps_row-eps_[row])   +   self.mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_[row])
                        eps_lerp_ortho_eps     = get_orthogonal(eps_[row], eps_[row])

                        eps_[row] = eps_collinear_eps_lerp + eps_lerp_ortho_eps

                    elif extra_options_flag("eps_proj_v4z", extra_options):
                        eps_row_lerp = eps_[row]   +   self.mask * (eps_row-eps_[row])   +   self.mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        peak = max(lgw, lgw_inv)
                        lgw_mask_sum = (lgw_mask + lgw_mask_inv)

                        eps_sum = eps_collinear_eps_lerp + eps_lerp_ortho_eps
                        #NOT FINISHED!!!
                        #eps_[row] = eps_[row] + lgw_mask * (eps_sum - eps_[row]) + lgw_mask_inv * (eps_sum - eps_[row])

                    elif extra_options_flag("eps_proj_v5", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_[row] = ((lgw_mask + lgw_mask_inv)==0) * eps_[row]   +   ((lgw_mask + lgw_mask_inv)>0) * (eps_collinear_eps_lerp + eps_lerp_ortho_eps)

                    elif extra_options_flag("eps_proj_v6", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_[row] = ((lgw_mask * lgw_mask_inv)==0) * eps_[row]   +   ((lgw_mask * lgw_mask_inv)>0) * (eps_collinear_eps_lerp + eps_lerp_ortho_eps)


                    elif extra_options_flag("eps_proj_old_default", extra_options):
                        eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   lgw_mask_inv * (eps_row_inv-eps_[row])
                        #eps_row_lerp = eps_[row]   +   lgw_mask * (eps_row-eps_[row])   +   (1-lgw_mask) * (eps_row_inv-eps_[row])
                        
                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)  
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])  
                        
                        eps_[row] = eps_collinear_eps_lerp + eps_lerp_ortho_eps
                        
                    else: #elif extra_options_flag("eps_proj_v4d", extra_options):
                        eps_row_lerp = eps_[row]   +   self.mask * (eps_row-eps_[row])   +   (1-self.mask) * (eps_row_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_sum = eps_collinear_eps_lerp + eps_lerp_ortho_eps

                        eps_[row] = eps_[row] + lgw_mask * (eps_sum - eps_[row]) + lgw_mask_inv * (eps_sum - eps_[row])



                elif extra_options_flag("disable_lgw_scaling", extra_options):
                    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                    eps_[row] = eps_[row]      +     lgw_mask * (eps_row - eps_[row])    +    lgw_mask_inv * (eps_row_inv - eps_[row])
                    

                elif (lgw > 0 or lgw_inv > 0): # default old channelwise epsilon
                    avg, avg_inv = 0, 0
                    for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                        avg     += torch.norm(data_[row][b][c] - y0    [b][c])
                        avg_inv += torch.norm(data_[row][b][c] - y0_inv[b][c])
                    avg     /= x_0.shape[1]
                    avg_inv /= x_0.shape[1]
                    
                    for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                        ratio     = torch.nan_to_num(torch.norm(data_[row][b][c] - y0    [b][c])   /   avg,     0)
                        ratio_inv = torch.nan_to_num(torch.norm(data_[row][b][c] - y0_inv[b][c])   /   avg_inv, 0)
                        
                        eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type, b, c)
                        eps_[row][b][c] = eps_[row][b][c]      +     ratio * lgw_mask[b][c] * (eps_row - eps_[row][b][c])    +    ratio_inv * lgw_mask_inv[b][c] * (eps_row_inv - eps_[row][b][c])
                        
                temporal_smoothing = float(get_extra_options_kv("temporal_smoothing", "0.0", extra_options))
                if temporal_smoothing > 0:
                    eps_[row] = apply_temporal_smoothing(eps_[row], temporal_smoothing)
                



        elif (UNSAMPLE or guide_mode in {"resample", "unsample"}) and (lgw > 0 or lgw_inv > 0):
                
            cvf = rk.get_epsilon(x_0, x_[row+1], y0, sigma, s_[row], sigma_down, unsample_resample_scale, extra_options)
            if UNSAMPLE and sigma > sigma_next and latent_guide_inv is not None:
                cvf_inv = rk.get_epsilon(x_0, x_[row+1], y0_inv, sigma, s_[row], sigma_down, unsample_resample_scale, extra_options)      
            else:
                cvf_inv = torch.zeros_like(cvf)

            tol_value = float(get_extra_options_kv("tol", "-1.0", extra_options))
            if tol_value >= 0:
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    current_diff     = torch.norm(data_[row][b][c] - y0    [b][c]) 
                    current_diff_inv = torch.norm(data_[row][b][c] - y0_inv[b][c]) 
                    
                    lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                    lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                    
                    lgw_tmp     = min(lgw    , lgw_scaled)
                    lgw_tmp_inv = min(lgw_inv, lgw_scaled_inv)

                    lgw_mask_clamp     = torch.clamp(lgw_mask,     max=lgw_tmp)
                    lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)

                    eps_[row][b][c] = eps_[row][b][c] + lgw_mask_clamp[b][c] * (cvf[b][c] - eps_[row][b][c]) + lgw_mask_clamp_inv[b][c] * (cvf_inv[b][c] - eps_[row][b][c])
                    
            elif extra_options_flag("disable_lgw_scaling", extra_options):
                eps_[row] = eps_[row] + lgw_mask * (cvf - eps_[row]) + lgw_mask_inv * (cvf_inv - eps_[row])
                
            else:
                avg, avg_inv = 0, 0
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    avg     += torch.norm(lgw_mask    [b][c] * data_[row][b][c]   -   lgw_mask    [b][c] * y0    [b][c])
                    avg_inv += torch.norm(lgw_mask_inv[b][c] * data_[row][b][c]   -   lgw_mask_inv[b][c] * y0_inv[b][c])
                avg     /= x_0.shape[1]
                avg_inv /= x_0.shape[1]
                
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    ratio     = torch.nan_to_num(torch.norm(lgw_mask    [b][c] * data_[row][b][c] - lgw_mask    [b][c] * y0    [b][c])   /   avg,     0)
                    ratio_inv = torch.nan_to_num(torch.norm(lgw_mask_inv[b][c] * data_[row][b][c] - lgw_mask_inv[b][c] * y0_inv[b][c])   /   avg_inv, 0)
                            
                    eps_[row][b][c] = eps_[row][b][c]      +     ratio * lgw_mask[b][c] * (cvf[b][c] - eps_[row][b][c])    +    ratio_inv * lgw_mask_inv[b][c] * (cvf_inv[b][c] - eps_[row][b][c])
                    
        if extra_options_flag("substep_eps_ch_mean_std", extra_options):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row])
        if extra_options_flag("substep_eps_ch_mean", extra_options):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], std=False)
        if extra_options_flag("substep_eps_ch_std", extra_options):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], mean=False)
        if extra_options_flag("substep_eps_mean_std", extra_options):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], channelwise=False)
        if extra_options_flag("substep_eps_mean", extra_options):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], std=False, channelwise=False)
        if extra_options_flag("substep_eps_std", extra_options):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], mean=False, channelwise=False)
        return eps_, x_



    @torch.no_grad
    def process_guides_poststep(self, x, denoised, eps, step, extra_options):
        x_orig = x.clone()
        mean_weight = float(get_extra_options_kv("mean_weight", "0.01", extra_options))
        
        y0 = self.y0
        if self.y0.shape[0] > 1:
            y0 = self.y0[min(step, self.y0.shape[0]-1)].unsqueeze(0)  
        y0_inv = self.y0_inv
        
        lgw_mask = self.lgw_masks[step].clone()
        lgw_mask_inv = self.lgw_masks_inv[step].clone() if self.lgw_masks_inv is not None else None
        mask = self.mask #needed for bitwise mask below
        
        lgw = self.lgw[step]
        lgw_inv = self.lgw_inv[step]
        
        latent_guide = self.latent_guide
        latent_guide_inv = self.latent_guide_inv
        guide_mode = self.guide_mode
        UNSAMPLE = self.UNSAMPLE
        
        if self.guide_mode: 
            data_norm   = denoised - denoised.mean(dim=(-2,-1), keepdim=True)
            y0_norm     = y0         -         y0.mean(dim=(-2,-1), keepdim=True)
            y0_inv_norm = y0_inv     -     y0_inv.mean(dim=(-2,-1), keepdim=True)

            y0_cossim     = get_cosine_similarity(data_norm*lgw_mask,     y0_norm    *lgw_mask)
            y0_cossim_inv = get_cosine_similarity(data_norm*lgw_mask_inv, y0_inv_norm*lgw_mask_inv)
            
            if y0_cossim < self.guide_cossim_cutoff_ or y0_cossim_inv < self.guide_bkg_cossim_cutoff_:
                lgw_mask_cossim, lgw_mask_cossim_inv = lgw_mask, lgw_mask_inv
                if y0_cossim     >= self.guide_cossim_cutoff_:
                    lgw_mask_cossim     = torch.zeros_like(lgw_mask)
                if y0_cossim_inv >= self.guide_bkg_cossim_cutoff_:
                    lgw_mask_cossim_inv = torch.zeros_like(lgw_mask_inv)
                lgw_mask = lgw_mask_cossim
                lgw_mask_inv = lgw_mask_cossim_inv
            else:
                return x
        
        if guide_mode in {"epsilon_dynamic_mean_std", "epsilon_dynamic_mean", "epsilon_dynamic_std", "epsilon_dynamic_mean_from_bkg"}:
        
            denoised_masked     = denoised * ((mask==1)*mask)
            denoised_masked_inv = denoised * ((mask==0)*(1-mask))
            
            
            d_shift, d_shift_inv = torch.zeros_like(x), torch.zeros_like(x)
            
            for b, c in itertools.product(range(x.shape[0]), range(x.shape[1])):
                denoised_mask     = denoised[b][c][mask[b][c] == 1]
                denoised_mask_inv = denoised[b][c][mask[b][c] == 0]
                
                if guide_mode == "epsilon_dynamic_mean_std":
                    d_shift[b][c] = (denoised_masked[b][c] - denoised_mask.mean()) / denoised_mask.std()
                    d_shift[b][c] = (d_shift[b][c] * denoised_mask_inv.std()) + denoised_mask_inv.mean()
                    
                elif guide_mode == "epsilon_dynamic_mean":
                    d_shift[b][c]     = denoised_masked[b][c]     - denoised_mask.mean()     + denoised_mask_inv.mean()
                    d_shift_inv[b][c] = denoised_masked_inv[b][c] - denoised_mask_inv.mean() + denoised_mask.mean()

                elif guide_mode == "epsilon_dynamic_mean_from_bkg":
                    d_shift[b][c] = denoised_masked[b][c] - denoised_mask.mean() + denoised_mask_inv.mean()

            if guide_mode in {"epsilon_dynamic_mean_std", "epsilon_dynamic_mean_from_bkg"}:
                denoised_shifted = denoised   +   mean_weight * lgw_mask * (d_shift - denoised_masked) 
            elif guide_mode == "epsilon_dynamic_mean":
                denoised_shifted = denoised   +   mean_weight * lgw_mask * (d_shift - denoised_masked)   +   mean_weight * lgw_mask_inv * (d_shift_inv - denoised_masked_inv)
                
            x = denoised_shifted + eps
        
        
        if UNSAMPLE == False and (latent_guide is not None or latent_guide_inv is not None) and guide_mode in ("hard_light", "blend", "blend_projection", "mean_std", "mean", "mean_tiled", "std"):
            if guide_mode == "hard_light":
                d_shift, d_shift_inv = hard_light_blend(y0, denoised), hard_light_blend(y0_inv, denoised)
            elif guide_mode == "blend":
                d_shift, d_shift_inv = y0, y0_inv
                
            elif guide_mode == "blend_projection":
                #d_shift     = get_collinear(denoised, y0) 
                #d_shift_inv = get_collinear(denoised, y0_inv) 
                
                d_lerp = denoised   +   lgw_mask * (y0-denoised)   +   lgw_mask_inv * (y0_inv-denoised)
                
                d_collinear_d_lerp = get_collinear(denoised, d_lerp)  
                d_lerp_ortho_d     = get_orthogonal(d_lerp, denoised)  
                
                denoised_shifted = d_collinear_d_lerp + d_lerp_ortho_d
                x = denoised_shifted + eps
                return x


            elif guide_mode == "mean_std":
                d_shift, d_shift_inv = normalize_latent([denoised, denoised], [y0, y0_inv])
            elif guide_mode == "mean":
                d_shift, d_shift_inv = normalize_latent([denoised, denoised], [y0, y0_inv], std=False)
            elif guide_mode == "std":
                d_shift, d_shift_inv = normalize_latent([denoised, denoised], [y0, y0_inv], mean=False)
            elif guide_mode == "mean_tiled":
                mean_tile_size = int(get_extra_options_kv("mean_tile", "8", extra_options))
                y0_tiled       = rearrange(y0,       "b c (h t1) (w t2) -> (t1 t2) b c h w", t1=mean_tile_size, t2=mean_tile_size)
                y0_inv_tiled   = rearrange(y0_inv,   "b c (h t1) (w t2) -> (t1 t2) b c h w", t1=mean_tile_size, t2=mean_tile_size)
                denoised_tiled = rearrange(denoised, "b c (h t1) (w t2) -> (t1 t2) b c h w", t1=mean_tile_size, t2=mean_tile_size)
                d_shift_tiled, d_shift_inv_tiled = torch.zeros_like(y0_tiled), torch.zeros_like(y0_tiled)
                for i in range(y0_tiled.shape[0]):
                    d_shift_tiled[i], d_shift_inv_tiled[i] = normalize_latent([denoised_tiled[i], denoised_tiled[i]], [y0_tiled[i], y0_inv_tiled[i]], std=False)
                d_shift     = rearrange(d_shift_tiled,     "(t1 t2) b c h w -> b c (h t1) (w t2)", t1=mean_tile_size, t2=mean_tile_size)
                d_shift_inv = rearrange(d_shift_inv_tiled, "(t1 t2) b c h w -> b c (h t1) (w t2)", t1=mean_tile_size, t2=mean_tile_size)


            if guide_mode in ("hard_light", "blend", "mean_std", "mean", "mean_tiled", "std"):
                if latent_guide_inv is None:
                    denoised_shifted = denoised   +   lgw_mask * (d_shift - denoised)
                else:
                    denoised_shifted = denoised   +   lgw_mask * (d_shift - denoised)   +   lgw_mask_inv * (d_shift_inv - denoised)
            
                if extra_options_flag("poststep_denoised_ch_mean_std", extra_options):
                    denoised_shifted = normalize_latent(denoised_shifted, denoised)
                if extra_options_flag("poststep_denoised_ch_mean", extra_options):
                    denoised_shifted = normalize_latent(denoised_shifted, denoised, std=False)
                if extra_options_flag("poststep_denoised_ch_std", extra_options):
                    denoised_shifted = normalize_latent(denoised_shifted, denoised, mean=False)
                if extra_options_flag("poststep_denoised_mean_std", extra_options):
                    denoised_shifted = normalize_latent(denoised_shifted, denoised, channelwise=False)
                if extra_options_flag("poststep_denoised_mean", extra_options):
                    denoised_shifted = normalize_latent(denoised_shifted, denoised, std=False, channelwise=False)
                if extra_options_flag("poststep_denoised_std", extra_options):
                    denoised_shifted = normalize_latent(denoised_shifted, denoised, mean=False, channelwise=False)

                x = denoised_shifted + eps

        if extra_options_flag("poststep_x_ch_mean_std", extra_options):
            x = normalize_latent(x, x_orig)
        if extra_options_flag("poststep_x_ch_mean", extra_options):
            x = normalize_latent(x, x_orig, std=False)
        if extra_options_flag("poststep_x_ch_std", extra_options):
            x = normalize_latent(x, x_orig, mean=False)
        if extra_options_flag("poststep_x_mean_std", extra_options):
            x = normalize_latent(x, x_orig, channelwise=False)
        if extra_options_flag("poststep_x_mean", extra_options):
            x = normalize_latent(x, x_orig, std=False, channelwise=False)
        if extra_options_flag("poststep_x_std", extra_options):
            x = normalize_latent(x, x_orig, mean=False, channelwise=False)
        return x







def prepare_mask(x, mask, LGW_MASK_RESCALE_MIN):
    if mask is None:
        mask = torch.ones_like(x)
        LGW_MASK_RESCALE_MIN = False
    else:
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, x.shape[1], 1, 1) 
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        mask = mask.to(x.dtype).to(x.device)
    return mask, LGW_MASK_RESCALE_MIN
    
def prepare_weighted_masks(mask, mask_inv, lgw_, lgw_inv_, latent_guide, latent_guide_inv, LGW_MASK_RESCALE_MIN):
    if LGW_MASK_RESCALE_MIN: 
        lgw_mask     =    mask  * (1-lgw_) + lgw_
        lgw_mask_inv = (1-mask) * (1-lgw_inv_) + lgw_inv_
    else:
        if latent_guide is not None:
            lgw_mask = mask * lgw_
        else:
            lgw_mask = torch.zeros_like(mask)
        if latent_guide_inv is not None:
            if mask_inv is not None:
                lgw_mask_inv = torch.minimum(1-mask_inv, (1-mask) * lgw_inv_)
            else:
                lgw_mask_inv = (1-mask) * lgw_inv_
        else:
            lgw_mask_inv = torch.zeros_like(mask)
    return lgw_mask, lgw_mask_inv

def apply_temporal_smoothing(tensor, temporal_smoothing):
    if temporal_smoothing <= 0 or tensor.dim() != 5:
        return tensor

    kernel_size = 5
    padding = kernel_size // 2
    temporal_kernel = torch.tensor(
        [0.1, 0.2, 0.4, 0.2, 0.1],
        device=tensor.device, dtype=tensor.dtype
    ) * temporal_smoothing
    temporal_kernel[kernel_size//2] += (1 - temporal_smoothing)
    temporal_kernel = temporal_kernel / temporal_kernel.sum()

    # resahpe for conv1d
    b, c, f, h, w = tensor.shape
    data_flat = tensor.permute(0, 1, 3, 4, 2).reshape(-1, f)

    # apply smoohting
    data_smooth = F.conv1d(
        data_flat.unsqueeze(1),
        temporal_kernel.view(1, 1, -1),
        padding=padding
    ).squeeze(1)

    return data_smooth.view(b, c, h, w, f).permute(0, 1, 4, 2, 3)

def get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type, b=None, c=None):
    s_in = x_0.new_ones([x_0.shape[0]])
    
    if b is not None and c is not None:  
        index = (b, c)
    elif b is not None: 
        index = (b,)
    else: 
        index = ()

    if RK_Method.is_exponential(rk_type):
        eps_row     = y0    [index] - x_0[index]
        eps_row_inv = y0_inv[index] - x_0[index]
    else:
        eps_row     = (x_[row+1][index] - y0    [index]) / (s_[row] * s_in)
        eps_row_inv = (x_[row+1][index] - y0_inv[index]) / (s_[row] * s_in)
    
    return eps_row, eps_row_inv

def get_guide_epsilon(x_0, x_, y0, sigma, rk_type, b=None, c=None):
    s_in = x_0.new_ones([x_0.shape[0]])
    
    if b is not None and c is not None:  
        index = (b, c)
    elif b is not None: 
        index = (b,)
    else: 
        index = ()

    if RK_Method.is_exponential(rk_type):
        eps     = y0    [index] - x_0[index]
    else:
        eps     = (x_[index] - y0    [index]) / (sigma * s_in)
    
    return eps




@torch.no_grad
def noise_cossim_guide_tiled(x_list, guide, cossim_mode="forward", tile_size=2, step=0):

    guide_tiled = rearrange(guide, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)

    x_tiled_list = [
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for x in x_list
    ]
    x_tiled_stack = torch.stack([x_tiled[0] for x_tiled in x_tiled_list])  # [n_x, n_tiles, c, h, w]

    guide_flat = guide_tiled[0].view(guide_tiled.shape[1], -1).unsqueeze(0)  # [1, n_tiles, c*h*w]
    x_flat = x_tiled_stack.view(x_tiled_stack.size(0), x_tiled_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]

    cossim_tmp_all = F.cosine_similarity(x_flat, guide_flat, dim=-1)  # [n_x, n_tiles]

    if cossim_mode == "forward":
        indices = cossim_tmp_all.argmax(dim=0) 
    elif cossim_mode == "reverse":
        indices = cossim_tmp_all.argmin(dim=0) 
    elif cossim_mode == "orthogonal":
        indices = torch.abs(cossim_tmp_all).argmin(dim=0) 
    elif cossim_mode == "forward_reverse":
        if step % 2 == 0:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_forward":
        if step % 2 == 1:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "orthogonal_reverse":
        if step % 2 == 0:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_orthogonal":
        if step % 2 == 1:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    else:
        target_value = float(cossim_mode)
        indices = torch.abs(cossim_tmp_all - target_value).argmin(dim=0)  

    x_tiled_out = x_tiled_stack[indices, torch.arange(indices.size(0))]  # [n_tiles, c, h, w]

    x_tiled_out = x_tiled_out.unsqueeze(0) 
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)

    return x_detiled


@torch.no_grad
def noise_cossim_eps_tiled(x_list, eps, noise_list, cossim_mode="forward", tile_size=2, step=0):

    eps_tiled = rearrange(eps, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    x_tiled_list = [
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for x in x_list
    ]
    noise_tiled_list = [
        rearrange(noise, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for noise in noise_list
    ]

    noise_tiled_stack = torch.stack([noise_tiled[0] for noise_tiled in noise_tiled_list])  # [n_x, n_tiles, c, h, w]
    eps_expanded = eps_tiled[0].view(eps_tiled.shape[1], -1).unsqueeze(0)  # [1, n_tiles, c*h*w]
    noise_flat = noise_tiled_stack.view(noise_tiled_stack.size(0), noise_tiled_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]
    cossim_tmp_all = F.cosine_similarity(noise_flat, eps_expanded, dim=-1)  # [n_x, n_tiles]

    if cossim_mode == "forward":
        indices = cossim_tmp_all.argmax(dim=0)  
    elif cossim_mode == "reverse":
        indices = cossim_tmp_all.argmin(dim=0) 
    elif cossim_mode == "orthogonal":
        indices = torch.abs(cossim_tmp_all).argmin(dim=0) 
    elif cossim_mode == "orthogonal_pos":
        positive_mask = cossim_tmp_all > 0
        positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
        indices = positive_tmp.argmin(dim=0)
    elif cossim_mode == "orthogonal_neg":
        negative_mask = cossim_tmp_all < 0
        negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
        indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_posneg":
        if step % 2 == 0:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_negpos":
        if step % 2 == 1:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "forward_reverse":
        if step % 2 == 0:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_forward":
        if step % 2 == 1:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "orthogonal_reverse":
        if step % 2 == 0:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_orthogonal":
        if step % 2 == 1:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    else:
        target_value = float(cossim_mode)
        indices = torch.abs(cossim_tmp_all - target_value).argmin(dim=0)
    #else:
    #    raise ValueError(f"Unknown cossim_mode: {cossim_mode}")

    x_tiled_stack = torch.stack([x_tiled[0] for x_tiled in x_tiled_list])  # [n_x, n_tiles, c, h, w]
    x_tiled_out = x_tiled_stack[indices, torch.arange(indices.size(0))]  # [n_tiles, c, h, w]

    x_tiled_out = x_tiled_out.unsqueeze(0)  # restore batch dim
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)
    return x_detiled



@torch.no_grad
def noise_cossim_guide_eps_tiled(x_0, x_list, y0, noise_list, cossim_mode="forward", tile_size=2, step=0, sigma=None, rk_type=None):

    x_tiled_stack = torch.stack([
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)[0]
        for x in x_list
    ])  # [n_x, n_tiles, c, h, w]
    eps_guide_stack = torch.stack([
        rearrange(x - y0, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)[0]
        for x in x_list
    ])  # [n_x, n_tiles, c, h, w]
    del x_list

    noise_tiled_stack = torch.stack([
        rearrange(noise, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)[0]
        for noise in noise_list
    ])  # [n_x, n_tiles, c, h, w]
    del noise_list

    noise_flat = noise_tiled_stack.view(noise_tiled_stack.size(0), noise_tiled_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]
    eps_guide_flat = eps_guide_stack.view(eps_guide_stack.size(0), eps_guide_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]

    cossim_tmp_all = F.cosine_similarity(noise_flat, eps_guide_flat, dim=-1)  # [n_x, n_tiles]
    del noise_tiled_stack, noise_flat, eps_guide_stack, eps_guide_flat

    if cossim_mode == "forward":
        indices = cossim_tmp_all.argmax(dim=0) 
    elif cossim_mode == "reverse":
        indices = cossim_tmp_all.argmin(dim=0) 
    elif cossim_mode == "orthogonal":
        indices = torch.abs(cossim_tmp_all).argmin(dim=0) 
    elif cossim_mode == "orthogonal_pos":
        positive_mask = cossim_tmp_all > 0
        positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
        indices = positive_tmp.argmin(dim=0)
    elif cossim_mode == "orthogonal_neg":
        negative_mask = cossim_tmp_all < 0
        negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
        indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_posneg":
        if step % 2 == 0:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_negpos":
        if step % 2 == 1:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "forward_reverse":
        if step % 2 == 0:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_forward":
        if step % 2 == 1:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "orthogonal_reverse":
        if step % 2 == 0:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_orthogonal":
        if step % 2 == 1:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    else:
        target_value = float(cossim_mode)
        indices = torch.abs(cossim_tmp_all - target_value).argmin(dim=0)  

    x_tiled_out = x_tiled_stack[indices, torch.arange(indices.size(0))]  # [n_tiles, c, h, w]
    del x_tiled_stack

    x_tiled_out = x_tiled_out.unsqueeze(0)  
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)

    return x_detiled






def get_collinear(x, y):

    y_flat = y.view(y.size(0), -1).clone()
    x_flat = x.view(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat

    return x_proj_y.view_as(x)


def get_orthogonal(x, y):

    y_flat = y.view(y.size(0), -1).clone()
    x_flat = x.view(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat
    
    x_ortho_y = x_flat - x_proj_y 

    return x_ortho_y.view_as(x)



def get_orthogonal_noise_from_channelwise(*refs, max_iter=500, max_score=1e-15):
    noise, *refs = refs
    noise_tmp = noise.clone()
    #b,c,h,w = noise.shape
    if (noise.dim() == 4):
        b,ch,h,w = noise.shape
    elif (noise.dim() == 5):
        b,ch,t,h,w = noise.shape
    
    for i in range(max_iter):
        noise_tmp = gram_schmidt_channels_optimized(noise_tmp, *refs)
        
        cossim_scores = []
        for ref in refs:
            #for c in range(noise.shape[-3]):
            for c in range(ch):
                cossim_scores.append(get_cosine_similarity(noise_tmp[0][c], ref[0][c]).abs())
            cossim_scores.append(get_cosine_similarity(noise_tmp[0], ref[0]).abs())
            
        if max(cossim_scores) < max_score:
            break
    
    return noise_tmp



def gram_schmidt_channels_optimized(A, *refs):
    if (A.dim() == 4):
        b,c,h,w = A.shape
    elif (A.dim() == 5):
        b,c,t,h,w = A.shape

    A_flat = A.view(b, c, -1)  
    
    for ref in refs:
        ref_flat = ref.view(b, c, -1).clone()  

        ref_flat /= ref_flat.norm(dim=-1, keepdim=True) 

        proj_coeff = torch.sum(A_flat * ref_flat, dim=-1, keepdim=True)  
        projection = proj_coeff * ref_flat 

        A_flat -= projection

    return A_flat.view_as(A)



class NoiseStepHandlerOSDE:
    def __init__(self, x, eps=None, data=None, x_init=None, guide=None, guide_bkg=None):
        self.noise = None
        self.x = x
        self.eps = eps
        self.data = data
        self.x_init = x_init
        self.guide = guide
        self.guide_bkg = guide_bkg
        
        self.eps_list = None

        self.noise_cossim_map = {
            "eps_orthogonal":              [self.noise, self.eps],
            "eps_data_orthogonal":         [self.noise, self.eps, self.data],

            "data_orthogonal":             [self.noise, self.data],
            "xinit_orthogonal":            [self.noise, self.x_init],
            
            "x_orthogonal":                [self.noise, self.x],
            "x_data_orthogonal":           [self.noise, self.x, self.data],
            "x_eps_orthogonal":            [self.noise, self.x, self.eps],

            "x_eps_data_orthogonal":       [self.noise, self.x, self.eps, self.data],
            "x_eps_data_xinit_orthogonal": [self.noise, self.x, self.eps, self.data, self.x_init],
            
            "x_eps_guide_orthogonal":      [self.noise, self.x, self.eps, self.guide],
            "x_eps_guide_bkg_orthogonal":  [self.noise, self.x, self.eps, self.guide_bkg],
            
            "noise_orthogonal":            [self.noise, self.x_init],
            
            "guide_orthogonal":            [self.noise, self.guide],
            "guide_bkg_orthogonal":        [self.noise, self.guide_bkg],
        }

    def check_cossim_source(self, source):
        return source in self.noise_cossim_map

    def get_ortho_noise(self, noise, prev_noises=None, max_iter=100, max_score=1e-7, NOISE_COSSIM_SOURCE="eps_orthogonal"):
        
        if NOISE_COSSIM_SOURCE not in self.noise_cossim_map:
            raise ValueError(f"Invalid NOISE_COSSIM_SOURCE: {NOISE_COSSIM_SOURCE}")
        
        self.noise_cossim_map[NOISE_COSSIM_SOURCE][0] = noise

        params = self.noise_cossim_map[NOISE_COSSIM_SOURCE]
        
        noise = get_orthogonal_noise_from_channelwise(*params, max_iter=max_iter, max_score=max_score)
        
        return noise



def handle_tiled_etc_noise_steps(x_0, x, x_prenoise, x_init, eps, denoised, y0, y0_inv, step, 
                                 rk_type, rk, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t,
                                 NOISE_COSSIM_SOURCE, NOISE_COSSIM_MODE, noise_cossim_tile_size, noise_cossim_iterations,
                                 extra_options):
    
    x_tmp, cossim_tmp, noise_tmp_list = [], [], []
    if step > int(get_extra_options_kv("noise_cossim_end_step", "10000", extra_options)):
        NOISE_COSSIM_SOURCE = get_extra_options_kv("noise_cossim_takeover_source", "eps", extra_options)
        NOISE_COSSIM_MODE   = get_extra_options_kv("noise_cossim_takeover_mode", "forward", extra_options)
        noise_cossim_tile_size   = int(get_extra_options_kv("noise_cossim_takeover_tile", str(noise_cossim_tile_size), extra_options))
        noise_cossim_iterations   = int(get_extra_options_kv("noise_cossim_takeover_iterations", str(noise_cossim_iterations), extra_options))
        
    for i in range(noise_cossim_iterations):
        x_tmp.append(rk.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)    )#y0, lgw, sigma_down are currently unused
        noise_tmp = x_tmp[i] - x
        if extra_options_flag("noise_noise_zscore_norm", extra_options):
            noise_tmp = (noise_tmp - noise_tmp.mean()) / noise_tmp.std()
        if extra_options_flag("noise_eps_zscore_norm", extra_options):
            eps = (eps - eps.mean()) / eps.std()
        if   NOISE_COSSIM_SOURCE in ("eps_tiled", "guide_epsilon_tiled", "guide_bkg_epsilon_tiled", "iig_tiled"):
            noise_tmp_list.append(noise_tmp)
        if   NOISE_COSSIM_SOURCE == "eps":
            cossim_tmp.append(get_cosine_similarity(eps, noise_tmp))
        if   NOISE_COSSIM_SOURCE == "eps_ch":
            cossim_total = torch.zeros_like(eps[0][0][0][0])
            for ch in range(eps.shape[1]):
                cossim_total += get_cosine_similarity(eps[0][ch], noise_tmp[0][ch])
            cossim_tmp.append(cossim_total)
        elif NOISE_COSSIM_SOURCE == "data":
            cossim_tmp.append(get_cosine_similarity(denoised, noise_tmp))
        elif NOISE_COSSIM_SOURCE == "latent":
            cossim_tmp.append(get_cosine_similarity(x_prenoise, noise_tmp))
        elif NOISE_COSSIM_SOURCE == "x_prenoise":
            cossim_tmp.append(get_cosine_similarity(x_prenoise, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "x":
            cossim_tmp.append(get_cosine_similarity(x, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "x_data":
            cossim_tmp.append(get_cosine_similarity(denoised, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "x_init_vs_noise":
            cossim_tmp.append(get_cosine_similarity(x_init, noise_tmp))
        elif NOISE_COSSIM_SOURCE == "mom":
            cossim_tmp.append(get_cosine_similarity(denoised, x + sigma_next*noise_tmp))
        elif NOISE_COSSIM_SOURCE == "guide":
            cossim_tmp.append(get_cosine_similarity(y0, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "guide_bkg":
            cossim_tmp.append(get_cosine_similarity(y0_inv, x_tmp[i]))
            
    if step < int(get_extra_options_kv("noise_cossim_start_step", "0", extra_options)):
        x = x_tmp[0]

    elif (NOISE_COSSIM_SOURCE == "eps_tiled"):
        x = noise_cossim_eps_tiled(x_tmp, eps, noise_tmp_list, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step)
    elif (NOISE_COSSIM_SOURCE == "guide_epsilon_tiled"):
        x = noise_cossim_guide_eps_tiled(x_0, x_tmp, y0, noise_tmp_list, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step, sigma=sigma, rk_type=rk_type)
    elif (NOISE_COSSIM_SOURCE == "guide_bkg_epsilon_tiled"):
        x = noise_cossim_guide_eps_tiled(x_0, x_tmp, y0_inv, noise_tmp_list, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step, sigma=sigma, rk_type=rk_type)
    elif (NOISE_COSSIM_SOURCE == "guide_tiled"):
        x = noise_cossim_guide_tiled(x_tmp, y0, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step)
    elif (NOISE_COSSIM_SOURCE == "guide_bkg_tiled"):
        x = noise_cossim_guide_tiled(x_tmp, y0_inv, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size)
    else:
        for i in range(len(x_tmp)):
            if   (NOISE_COSSIM_MODE == "forward") and (cossim_tmp[i] == max(cossim_tmp)):
                x = x_tmp[i]
                break
            elif (NOISE_COSSIM_MODE == "reverse") and (cossim_tmp[i] == min(cossim_tmp)):
                x = x_tmp[i]
                break
            elif (NOISE_COSSIM_MODE == "orthogonal") and (abs(cossim_tmp[i]) == min(abs(val) for val in cossim_tmp)):
                x = x_tmp[i]
                break
            elif (NOISE_COSSIM_MODE != "forward") and (NOISE_COSSIM_MODE != "reverse") and (NOISE_COSSIM_MODE != "orthogonal"):
                x = x_tmp[0]
                break
    return x



