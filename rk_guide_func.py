import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re

from .noise_classes import *
from .latents import hard_light_blend, normalize_latent
from .rk_method import RK_Method

import itertools


@torch.no_grad()
def process_guides_substep(x_0, x_, eps_, data_, row, y0, y0_inv, lgw, lgw_inv, lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_, t_i, rk, rk_type, guide_mode, latent_guide_inv, UNSAMPLE, extra_options):
    if UNSAMPLE and RK_Method.is_exponential(rk_type):
        if not (re.search(r"\bdisable_power_unsample\b", extra_options) or re.search(r"\bdisable_power_resample\b", extra_options)):
            extra_options += "\npower_unsample\npower_resample\n"
        if not (re.search(r"\bdisable_lgw_scaling_substep_ch_mean_std\b", extra_options)):
            extra_options += "\nsubstep_eps_ch_mean_std\n"
            
    
    s_in = x_0.new_ones([x_0.shape[0]])
    y0_orig, y0_inv_orig = y0.clone(), y0_inv.clone()
    eps_orig = eps_.clone()
    
    if re.search(r"\bdynamic_guides_mean_std\b", extra_options):
        
        #for b, c in itertools.product(range(y0.shape[0]), range(y0.shape[1])):
        for n in range(y0.shape[1]):
            y_norm     [0][n] = (y0_orig[0][n] - y0_orig[0][n].mean()) / y0_orig[0][n].std()
            y_shift    [0][n] = (y_norm[0][n] * data_[row]    [0][n].std()) + data_[row]    [0][n].mean()
            
            y_inv_norm [0][n] = (y0_inv_orig[0][n] - y0_inv_orig[0][n].mean()) / y0_inv_orig[0][n].std()
            y_inv_shift[0][n] = (y_inv_norm[0][n] * data_[row][0][n].std()) + data_[row][0][n].mean()
        y0 = y_shift
        if re.search(r"\bdynamic_guides_inv\b", extra_options):
            y0_inv = y_inv_shift
            
    if re.search(r"\bdynamic_guides_mean\b", extra_options):
        y_norm,     y_shift     = torch.zeros_like(x_0), torch.zeros_like(x_0)
        y_inv_norm, y_inv_shift = torch.zeros_like(x_0), torch.zeros_like(x_0)
        
        for n in range(y0.shape[1]):
            y_norm     [0][n] = (y0_orig[0][n] - y0_orig[0][n].mean())
            y_shift    [0][n] = (y_norm[0][n]) + data_[row][0][n].mean()
            
            y_inv_norm [0][n] = (y0_inv_orig[0][n] - y0_inv_orig[0][n].mean())
            y_inv_shift[0][n] = (y_inv_norm[0][n]) + data_[row][0][n].mean()
        y0 = y_shift
        if re.search(r"\bdynamic_guides_inv\b", extra_options):
            y0_inv = y_inv_shift

    if "data" in guide_mode:
        x_[row+1] = y0_tmp + eps_[row]

    elif "epsilon" in guide_mode:
        if sigma > sigma_next:
                
            if re.search(r"\bdisable_lgw_scaling\b", extra_options): # and lgw[_] > 0:

                if RK_Method.is_exponential(rk_type):
                    eps_m   = y0     - x_0
                    eps_inv = y0_inv - x_0
                else:
                    eps_m   = (x_[row+1] - y0)     / (s_[row] * s_in)
                    eps_inv = (x_[row+1] - y0_inv) / (s_[row] * s_in)
                    
                eps_[row] = eps_[row]      +     lgw_mask * (eps_m - eps_[row])    +    lgw_mask_inv * (eps_inv - eps_[row])
                    
            elif re.search(r"tol\s*=\s*([\d.]+)", extra_options) and (lgw > 0 or lgw_inv > 0):
                tol_value = float(re.search(r"tol\s*=\s*([\d.]+)", extra_options).group(1))                    
                for i4 in range(x_0.shape[1]):
                    current_diff     = torch.norm(data_[row][0][i4] - y0    [0][i4]) 
                    current_diff_inv = torch.norm(data_[row][0][i4] - y0_inv[0][i4]) 
                    
                    lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                    lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                    
                    lgw_tmp     = min(lgw    , lgw_scaled)
                    lgw_tmp_inv = min(lgw_inv, lgw_scaled_inv)

                    lgw_mask_clamp = torch.clamp(lgw_mask, max=lgw_tmp)
                    lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)
                    
                    #rk.process_guide_row(x_0, x_, y0, y0_inv, lgw_mask_clamp[0][i4], lgw_mask_clamp_inv[0][i4] )


                    if RK_Method.is_exponential(rk_type):
                        eps_row     = y0    [0][i4] - x_0[0][i4]
                        eps_row_inv = y0_inv[0][i4] - x_0[0][i4]
                    else:
                        eps_row     = (x_[row+1][0][i4] - y0    [0][i4]) / (s_[row] * s_in)
                        eps_row_inv = (x_[row+1][0][i4] - y0_inv[0][i4]) / (s_[row] * s_in)
                    eps_[row][0][i4] = eps_[row][0][i4] + lgw_mask_clamp[0][i4] * (eps_row - eps_[row][0][i4]) + lgw_mask_clamp_inv[0][i4] * (eps_row_inv - eps_[row][0][i4])
                    
            elif (lgw > 0 or lgw_inv > 0):
                avg, avg_inv = 0, 0
                for i4 in range(x_0.shape[1]):
                    avg     += torch.norm(data_[row][0][i4] - y0    [0][i4])
                    avg_inv += torch.norm(data_[row][0][i4] - y0_inv[0][i4])
                avg     /= x_0.shape[1]
                avg_inv /= x_0.shape[1]
                
                for i4 in range(x_0.shape[1]):
                    ratio     = torch.nan_to_num(torch.norm(data_[row][0][i4] - y0    [0][i4])   /   avg,     0)
                    ratio_inv = torch.nan_to_num(torch.norm(data_[row][0][i4] - y0_inv[0][i4])   /   avg_inv, 0)
                    
                    if RK_Method.is_exponential(rk_type):
                        eps_row     = y0    [0][i4] - x_0[0][i4]
                        eps_row_inv = y0_inv[0][i4] - x_0[0][i4]
                    else:
                        eps_row     = (x_[row+1][0][i4] - y0    [0][i4]) / (s_[row] * s_in)
                        eps_row_inv = (x_[row+1][0][i4] - y0_inv[0][i4]) / (s_[row] * s_in)            
                    eps_[row][0][i4] = eps_[row][0][i4]      +     ratio * lgw_mask[0][i4] * (eps_row - eps_[row][0][i4])    +    ratio_inv * lgw_mask_inv[0][i4] * (eps_row_inv - eps_[row][0][i4])

    elif (UNSAMPLE or guide_mode == "resample" or guide_mode == "unsample") and (lgw > 0 or lgw_inv > 0):
        y0_tmp = y0
        if latent_guide_inv is not None:
            y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
            y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
            
        cvf = rk.get_epsilon(x_0, x_[row+1], y0, sigma, s_[row], sigma_down, t_i, extra_options)
        if UNSAMPLE and sigma > sigma_next and latent_guide_inv is not None:
            cvf_inv = rk.get_epsilon(x_0, x_[row+1], y0_inv, sigma, s_[row], sigma_down, t_i, extra_options)      
        else:
            cvf_inv = torch.zeros_like(cvf)

        if re.search(r"tol\s*=\s*([\d.]+)", extra_options):
            tol_value = float(re.search(r"tol\s*=\s*([\d.]+)", extra_options).group(1))                    
            for i4 in range(x_0.shape[1]):
                current_diff     = torch.norm(data_[row][0][i4] - y0    [0][i4]) 
                current_diff_inv = torch.norm(data_[row][0][i4] - y0_inv[0][i4]) 
                
                lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                
                lgw_tmp     = min(lgw    , lgw_scaled)
                lgw_tmp_inv = min(lgw_inv, lgw_scaled_inv)

                lgw_mask_clamp = torch.clamp(lgw_mask, max=lgw_tmp)
                lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)

                eps_[row][0][i4] = eps_[row][0][i4] + lgw_mask_clamp[0][i4] * (cvf[0][i4] - eps_[row][0][i4]) + lgw_mask_clamp_inv[0][i4] * (cvf_inv[0][i4] - eps_[row][0][i4])
                
        elif re.search(r"\bdisable_lgw_scaling\b", extra_options):
            eps_[row] = eps_[row] + lgw_mask * (cvf - eps_[row]) + lgw_mask_inv * (cvf_inv - eps_[row])
            
            
        else:
            avg, avg_inv = 0, 0
            for i4 in range(x_0.shape[1]):
                avg     += torch.norm(lgw_mask[0][i4]     * data_[row][0][i4]   -   lgw_mask[0][i4]     * y0[0][i4])
                avg_inv += torch.norm(lgw_mask_inv[0][i4] * data_[row][0][i4]   -   lgw_mask_inv[0][i4] * y0_inv[0][i4])
            avg     /= x_0.shape[1]
            avg_inv /= x_0.shape[1]
            
            for i4 in range(x_0.shape[1]):
                ratio     = torch.nan_to_num(torch.norm(lgw_mask[0][i4]     * data_[row][0][i4] - lgw_mask[0][i4]     * y0    [0][i4])   /   avg,     0)
                ratio_inv = torch.nan_to_num(torch.norm(lgw_mask_inv[0][i4] * data_[row][0][i4] - lgw_mask_inv[0][i4] * y0_inv[0][i4])   /   avg_inv, 0)
                         
                eps_[row][0][i4] = eps_[row][0][i4]      +     ratio * lgw_mask[0][i4] * (cvf[0][i4] - eps_[row][0][i4])    +    ratio_inv * lgw_mask_inv[0][i4] * (cvf_inv[0][i4] - eps_[row][0][i4])
                
    if re.search(r"\bsubstep_eps_ch_mean_std\b", extra_options):
        eps_[row] = normalize_latent(eps_[row], eps_orig[row])
    if re.search(r"\bsubstep_eps_ch_mean\b", extra_options):
        eps_[row] = normalize_latent(eps_[row], eps_orig[row], std=False)
    if re.search(r"\bsubstep_eps_ch_std\b", extra_options):
        eps_[row] = normalize_latent(eps_[row], eps_orig[row], mean=False)
    if re.search(r"\bsubstep_eps_mean_std\b", extra_options):
        eps_[row] = normalize_latent(eps_[row], eps_orig[row], channelwise=False)
    if re.search(r"\bsubstep_eps_mean\b", extra_options):
        eps_[row] = normalize_latent(eps_[row], eps_orig[row], std=False, channelwise=False)
    if re.search(r"\bsubstep_eps_std\b", extra_options):
        eps_[row] = normalize_latent(eps_[row], eps_orig[row], mean=False, channelwise=False)
    return eps_



@torch.no_grad
def process_guides_poststep(x, denoised, eps, y0, y0_inv, mask, lgw_mask, lgw_mask_inv, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options):
    x_orig = x.clone()
    if re.search(r"mean_weight\s*=\s*([\d.]+)", extra_options):
        mean_weight = float(re.search(r"mean_weight\s*=\s*([\d.]+)", extra_options).group(1))           
    else:
        mean_weight = 0.01
    
    if guide_mode == "epsilon_mean_std" or guide_mode == "epsilon_mean" or guide_mode == "epsilon_std" or guide_mode == "epsilon_mean_use_inv":
        denoised_masked     = denoised * ((mask==1)*mask)
        denoised_masked_inv = denoised * ((mask==0)*(1-mask))
        denoised_masked_intermediate = denoised - denoised_masked - denoised_masked_inv
        
        if guide_mode == "epsilon_mean_std":

            ks3 = torch.zeros_like(x)
            for n in range(denoised.shape[1]):
                denoised_mask     = denoised[0][n][mask[0][n] == 1]
                denoised_mask_inv = denoised[0][n][mask[0][n] == 0]
                
                ks3[0][n] = (denoised_masked[0][n] - denoised_mask.mean()) / denoised_mask.std()
                ks3[0][n] = (ks3[0][n] * denoised_mask_inv.std()) + denoised_mask_inv.mean()
                
            x = mean_weight * ks3 + (1-mean_weight) * denoised_masked           + denoised_masked_intermediate +  denoised_masked_inv + eps
        elif guide_mode == "epsilon_mean":
            denoised_masked     = denoised * ((mask==1)*mask)
            denoised_masked_inv = denoised * ((mask==0)*(1-mask))
            denoised_masked_intermediate = denoised - denoised_masked - denoised_masked_inv
            
            d_shift, d_shift_inv = torch.zeros_like(x), torch.zeros_like(x)
            for n in range(denoised.shape[1]):
                denoised_mask     = denoised[0][n][mask[0][n] == 1]
                denoised_mask_inv = denoised[0][n][mask[0][n] == 0]
                
                d_shift[0][n] = (denoised_masked[0][n] - denoised_mask.mean())
                d_shift[0][n] = (d_shift[0][n]) + denoised_mask_inv.mean()
                
                d_shift_inv[0][n] = (denoised_masked_inv[0][n] - denoised_mask_inv.mean())
                d_shift_inv[0][n] = (d_shift_inv[0][n]) + denoised_mask.mean()

            denoised_shifted = denoised   +   mean_weight * lgw_mask * (d_shift - denoised_masked)   +   mean_weight * lgw_mask_inv * (d_shift_inv - denoised_masked_inv)
            x = denoised_shifted + eps
            
        elif guide_mode == "epsilon_mean_use_inv":
            denoised_masked     = denoised * ((mask==1)*mask)
            denoised_masked_inv = denoised * ((mask==0)*(1-mask))
            denoised_masked_intermediate = denoised - denoised_masked - denoised_masked_inv
            
            d_shift, d_shift_inv = torch.zeros_like(x), torch.zeros_like(x)
            for n in range(denoised.shape[1]):
                denoised_mask     = denoised[0][n][mask[0][n] == 1]
                denoised_mask_inv = denoised[0][n][mask[0][n] == 0]
                
                d_shift[0][n] = (denoised_masked[0][n] - denoised_mask.mean())
                d_shift[0][n] = (d_shift[0][n]) + denoised_mask_inv.mean()

            denoised_shifted = denoised   +   mean_weight * lgw_mask * (d_shift - denoised_masked)  # +   mean_weight * lgw_mask_inv * (d_shift_inv - denoised_masked_inv)
            x = denoised_shifted + eps
    
    
    if UNSAMPLE == False and (latent_guide is not None or latent_guide_inv is not None):
        d_norm, d_shift, d_shift_inv = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
        
        if guide_mode == "hard_light":
            d_shift     = hard_light_blend(y0,     denoised)
            d_shift_inv = hard_light_blend(y0_inv, denoised)
            
        elif guide_mode == "blend":
            d_shift     = y0
            d_shift_inv = y0_inv               
            
        elif guide_mode == "mean_std":
            for n in range(y0.shape[1]):
                d_norm     [0][n] = (denoised[0][n] - denoised[0][n].mean()) / denoised[0][n].std()
                d_shift    [0][n] = (d_norm[0][n] * y0    [0][n].std()) + y0    [0][n].mean()
                d_shift_inv[0][n] = (d_norm[0][n] * y0_inv[0][n].std()) + y0_inv[0][n].mean()
        elif guide_mode == "mean":
            for n in range(y0.shape[1]):
                d_norm     [0][n] = denoised[0][n] - denoised[0][n].mean()
                d_shift    [0][n] = d_norm[0][n] + y0    [0][n].mean()
                d_shift_inv[0][n] = d_norm[0][n] + y0_inv[0][n].mean()
        elif guide_mode == "std":
            for n in range(y0.shape[1]):
                d_norm     [0][n] = denoised[0][n] / denoised[0][n].std()
                d_shift    [0][n] = d_norm[0][n] * y0    [0][n].std()
                d_shift_inv[0][n] = d_norm[0][n] * y0_inv[0][n].std()

        if guide_mode in ("hard_light", "blend", "mean_std", "mean", "std"):
            denoised_shifted = denoised   +   lgw_mask * (d_shift - denoised)   +   lgw_mask_inv * (d_shift_inv - denoised)
            
            if re.search(r"\bpoststep_denoised_ch_mean_std\b", extra_options):
                denoised_shifted = normalize_latent(denoised_shifted, denoised)
            if re.search(r"\bpoststep_denoised_ch_mean\b", extra_options):
                denoised_shifted = normalize_latent(denoised_shifted, denoised, std=False)
            if re.search(r"\bpoststep_denoised_ch_std\b", extra_options):
                denoised_shifted = normalize_latent(denoised_shifted, denoised, mean=False)
            if re.search(r"\bpoststep_denoised_mean_std\b", extra_options):
                denoised_shifted = normalize_latent(denoised_shifted, denoised, channelwise=False)
            if re.search(r"\bpoststep_denoised_mean\b", extra_options):
                denoised_shifted = normalize_latent(denoised_shifted, denoised, std=False, channelwise=False)
            if re.search(r"\bpoststep_denoised_std\b", extra_options):
                denoised_shifted = normalize_latent(denoised_shifted, denoised, mean=False, channelwise=False)
            
            x = denoised_shifted + eps


    if re.search(r"\bpoststep_x_ch_mean_std\b", extra_options):
        x = normalize_latent(x, x_orig)
    if re.search(r"\bpoststep_x_ch_mean\b", extra_options):
        x = normalize_latent(x, x_orig, std=False)
    if re.search(r"\bpoststep_x_ch_std\b", extra_options):
        x = normalize_latent(x, x_orig, mean=False)
    if re.search(r"\bpoststep_x_mean_std\b", extra_options):
        x = normalize_latent(x, x_orig, channelwise=False)
    if re.search(r"\bpoststep_x_mean\b", extra_options):
        x = normalize_latent(x, x_orig, std=False, channelwise=False)
    if re.search(r"\bpoststep_x_std\b", extra_options):
        x = normalize_latent(x, x_orig, mean=False, channelwise=False)
    return x

