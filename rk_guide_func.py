import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re

from .noise_classes import *
from .latents import hard_light_blend, normalize_latent
from .rk_method import RK_Method
from .helper import get_extra_options_kv, extra_options_flag


import itertools



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


def get_guide_epsilon(x_0, x_, y0, y0_inv, s_, row, rk_type, b=None, c=None):
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



@torch.no_grad()
def process_guides_substep(x_0, x_, eps_, data_, row, y0, y0_inv, lgw, lgw_inv, lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, guide_mode, latent_guide_inv, UNSAMPLE, extra_options):
    
    if UNSAMPLE and RK_Method.is_exponential(rk_type):
        if not (extra_options_flag("disable_power_unsample", extra_options) or extra_options_flag("disable_power_resample", extra_options)):
            extra_options += "\npower_unsample\npower_resample\n"
        if not extra_options_flag("disable_lgw_scaling_substep_ch_mean_std", extra_options):
            extra_options += "\nsubstep_eps_ch_mean_std\n"
            
    s_in = x_0.new_ones([x_0.shape[0]])
    y0_orig, y0_inv_orig = y0.clone(), y0_inv.clone()
    eps_orig = eps_.clone()
    
    if extra_options_flag("dynamic_guides_mean_std", extra_options):
        y_shift, y_inv_shift = normalize_latent([y0_orig, y0_inv_orig], [data_, data_])
        y0 = y_shift
        if extra_options_flag("dynamic_guides_inv", extra_options):
            y0_inv = y_inv_shift

    if extra_options_flag("dynamic_guides_mean", extra_options):
        y_shift, y_inv_shift = normalize_latent([y0_orig, y0_inv_orig], [data_, data_], std=False)
        y0 = y_shift
        if extra_options_flag("dynamic_guides_inv", extra_options):
            y0_inv = y_inv_shift

    if "data" in guide_mode:
        y0_tmp = y0
        if latent_guide_inv is not None:
            y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
            y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
        x_[row+1] = y0_tmp + eps_[row]

    elif "epsilon" in guide_mode:
        if sigma > sigma_next:
                
            if extra_options_flag("disable_lgw_scaling", extra_options):
                eps_row, eps_row_inv = get_guide_epsilon(x_0, x_, y0, y0_inv, s_, row, rk_type)
                eps_[row] = eps_[row]      +     lgw_mask * (eps_row - eps_[row])    +    lgw_mask_inv * (eps_row_inv - eps_[row])
                     
            tol_value = float(get_extra_options_kv("tol", "-1.0", extra_options))
            if tol_value >= 0 and (lgw > 0 or lgw_inv > 0):           
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    current_diff     = torch.norm(data_[row][b][c] - y0    [b][c])
                    current_diff_inv = torch.norm(data_[row][b][c] - y0_inv[b][c])
                    
                    lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                    lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                    
                    lgw_tmp     = min(lgw    , lgw_scaled)
                    lgw_tmp_inv = min(lgw_inv, lgw_scaled_inv)

                    lgw_mask_clamp = torch.clamp(lgw_mask, max=lgw_tmp)
                    lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)
                    
                    eps_row, eps_row_inv = get_guide_epsilon(x_0, x_, y0, y0_inv, s_, row, rk_type, b, c)
                    eps_[row][b][c] = eps_[row][b][c] + lgw_mask_clamp[b][c] * (eps_row - eps_[row][b][c]) + lgw_mask_clamp_inv[b][c] * (eps_row_inv - eps_[row][b][c])
                    
            elif (lgw > 0 or lgw_inv > 0):
                avg, avg_inv = 0, 0
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    avg     += torch.norm(data_[row][b][c] - y0    [b][c])
                    avg_inv += torch.norm(data_[row][b][c] - y0_inv[b][c])
                avg     /= x_0.shape[1]
                avg_inv /= x_0.shape[1]
                
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    ratio     = torch.nan_to_num(torch.norm(data_[row][b][c] - y0    [b][c])   /   avg,     0)
                    ratio_inv = torch.nan_to_num(torch.norm(data_[row][b][c] - y0_inv[b][c])   /   avg_inv, 0)
                    
                    eps_row, eps_row_inv = get_guide_epsilon(x_0, x_, y0, y0_inv, s_, row, rk_type, b, c)
                    eps_[row][b][c] = eps_[row][b][c]      +     ratio * lgw_mask[b][c] * (eps_row - eps_[row][b][c])    +    ratio_inv * lgw_mask_inv[b][c] * (eps_row_inv - eps_[row][b][c])

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

                lgw_mask_clamp = torch.clamp(lgw_mask, max=lgw_tmp)
                lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)

                eps_[row][b][c] = eps_[row][b][c] + lgw_mask_clamp[b][c] * (cvf[b][c] - eps_[row][b][c]) + lgw_mask_clamp_inv[b][c] * (cvf_inv[b][c] - eps_[row][b][c])
                
        elif extra_options_flag("disable_lgw_scaling", extra_options):
            eps_[row] = eps_[row] + lgw_mask * (cvf - eps_[row]) + lgw_mask_inv * (cvf_inv - eps_[row])
            
        else:
            avg, avg_inv = 0, 0
            for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                avg     += torch.norm(lgw_mask[b][c]     * data_[row][b][c]   -   lgw_mask[b][c]     * y0[b][c])
                avg_inv += torch.norm(lgw_mask_inv[b][c] * data_[row][b][c]   -   lgw_mask_inv[b][c] * y0_inv[b][c])
            avg     /= x_0.shape[1]
            avg_inv /= x_0.shape[1]
            
            for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                ratio     = torch.nan_to_num(torch.norm(lgw_mask[b][c]     * data_[row][b][c] - lgw_mask[b][c]     * y0    [b][c])   /   avg,     0)
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
def process_guides_poststep(x, denoised, eps, y0, y0_inv, mask, lgw_mask, lgw_mask_inv, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options):
    x_orig = x.clone()
    mean_weight = float(get_extra_options_kv("mean_weight", "0.01", extra_options))
    
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
    
    
    if UNSAMPLE == False and (latent_guide is not None or latent_guide_inv is not None) and guide_mode in ("hard_light", "blend", "mean_std", "mean", "std"):
        if guide_mode == "hard_light":
            d_shift, d_shift_inv = hard_light_blend(y0, denoised), hard_light_blend(y0_inv, denoised)
        elif guide_mode == "blend":
            d_shift, d_shift_inv = y0, y0_inv
        elif guide_mode == "mean_std":
            d_shift, d_shift_inv = normalize_latent([denoised, denoised], [y0, y0_inv])
        elif guide_mode == "mean":
            d_shift, d_shift_inv = normalize_latent([denoised, denoised], [y0, y0_inv], std=False)
        elif guide_mode == "std":
            d_shift, d_shift_inv = normalize_latent([denoised, denoised], [y0, y0_inv], mean=False)

        if guide_mode in ("hard_light", "blend", "mean_std", "mean", "std"):
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

