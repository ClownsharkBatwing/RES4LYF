import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re

from einops import rearrange

from .noise_classes import *
from .latents import hard_light_blend, normalize_latent
from .rk_method import RK_Method
from .helper import get_extra_options_kv, extra_options_flag, get_cosine_similarity


import itertools

def apply_guided_denoising(denoised, y0, y0_inv, lgw_mask, lgw_mask_inv, guide_mode, extra_options=None):
    """
    Applies guided denoising for data-based modes, with optional hybrid features.
    Handles 'data' and 'hybrid_data' modes only.
    
    Returns denoised predictions with guidance applied.
    """
    if extra_options is None:
        extra_options = ""
        
    if guide_mode == "data":
        # Standard data mode spatial guidance
        if y0_inv is None:
            denoised_shifted = (1-lgw_mask) * denoised + lgw_mask * y0
        else:
            denoised_shifted = (1-lgw_mask) * denoised + lgw_mask * y0
            denoised_shifted = (1-lgw_mask_inv) * denoised_shifted + lgw_mask_inv * y0_inv
            
    elif guide_mode == "hybrid_data":
        # Get hybrid parameters
        hybrid_spatial_weight = float(get_extra_options_kv("hybrid_spatial_weight", "1.0", extra_options))
        hybrid_stats_weight = float(get_extra_options_kv("hybrid_stats_weight", "0.3", extra_options))
        hybrid_mean_weight = float(get_extra_options_kv("hybrid_mean_weight", "1.0", extra_options))
        hybrid_std_weight = float(get_extra_options_kv("hybrid_std_weight", "0.7", extra_options))
        hybrid_temporal_smooth = float(get_extra_options_kv("hybrid_temporal_smooth", "0.2", extra_options))
        hybrid_clamp_mult = float(get_extra_options_kv("hybrid_clamp_mult", "1.5", extra_options))

        # First apply spatial guidance with controllable strength
        spatial_mask = lgw_mask * hybrid_spatial_weight
        denoised_shifted = (1-spatial_mask) * denoised + spatial_mask * y0
        if y0_inv is not None:
            spatial_mask_inv = lgw_mask_inv * hybrid_spatial_weight
            denoised_shifted = (1-spatial_mask_inv) * denoised_shifted + spatial_mask_inv * y0_inv

        # Apply statistical matching if enabled
        if hybrid_stats_weight > 0:
            data_shifted = denoised_shifted.clone()
            
            # Calculate guide statistics
            guide_stats = {
                'mean': torch.tensor([y0[0][c].mean() for c in range(y0.shape[1])],
                                   device=denoised.device, dtype=denoised.dtype),
                'std': torch.tensor([y0[0][c].std() for c in range(y0.shape[1])],
                                  device=denoised.device, dtype=denoised.dtype)
            }

            # Apply per-channel statistical matching
            for b, c in itertools.product(range(denoised.shape[0]), range(denoised.shape[1])):
                curr_mean = data_shifted[b][c].mean()
                curr_std = data_shifted[b][c].std()
                
                stats_weight = lgw_mask[b][c].mean() * hybrid_stats_weight

                # Apply mean matching
                if hybrid_mean_weight > 0:
                    mean_weight = stats_weight * hybrid_mean_weight
                    target_mean = guide_stats['mean'][c] * mean_weight + curr_mean * (1 - mean_weight)
                    data_shifted[b][c] = data_shifted[b][c] - curr_mean + target_mean

                # Apply std matching
                if hybrid_std_weight > 0:
                    std_weight = stats_weight * hybrid_std_weight
                    target_std = guide_stats['std'][c] * std_weight + curr_std * (1 - std_weight)
                    
                    centered = data_shifted[b][c] - data_shifted[b][c].mean()
                    scale_factor = target_std / (curr_std + 1e-6)
                    scale_factor = torch.clamp(scale_factor, 
                                             min=1/hybrid_clamp_mult,
                                             max=hybrid_clamp_mult)
                    data_shifted[b][c] = centered * scale_factor + data_shifted[b][c].mean()

            # Apply temporal smoothing if we have video frames
            if hybrid_temporal_smooth > 0 and data_shifted.dim() == 5:
                kernel_size = 5
                padding = kernel_size // 2
                temporal_kernel = torch.tensor(
                    [0.1, 0.2, 0.4, 0.2, 0.1],
                    device=denoised.device, dtype=denoised.dtype
                ) * hybrid_temporal_smooth
                temporal_kernel[kernel_size//2] += (1 - hybrid_temporal_smooth)
                temporal_kernel = temporal_kernel / temporal_kernel.sum()
                
                # Reshape for conv1d
                b, c, f, h, w = data_shifted.shape
                data_flat = data_shifted.permute(0, 1, 3, 4, 2).reshape(-1, f)
                
                # Apply smoothing
                data_smooth = F.conv1d(
                    data_flat.unsqueeze(1),
                    temporal_kernel.view(1, 1, -1),
                    padding=padding
                ).squeeze(1)
                
                # Reshape back
                data_shifted = data_smooth.view(b, c, h, w, f).permute(0, 1, 4, 2, 3)

            # Final blend between spatial and statistical guidance
            denoised_shifted = denoised_shifted + spatial_mask * (data_shifted - denoised_shifted)
    
    else:
        denoised_shifted = denoised
        
    return denoised_shifted

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


def get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type, b=None, c=None):
    s_in = x_0.new_ones([x_0.shape[0]])
    
    if b is not None and c is not None:  
        index = (b, c)
    elif b is not None: 
        index = (b,)
    else: 
        index = ()

    if RK_Method.is_exponential(rk_type):
        # Calculate full-strength epsilon values
        eps_row     = y0    [index] - x_0[index]
        eps_row_inv = y0_inv[index] - x_0[index] if y0_inv is not None else None
    else:
        # Calculate full-strength epsilon values
        eps_row     = (x_[row+1][index] - y0    [index]) / (s_[row] * s_in)
        eps_row_inv = (x_[row+1][index] - y0_inv[index]) / (s_[row] * s_in) if y0_inv is not None else None
    
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
        #eps_inv = y0_inv[index] - x_0[index]
    else:
        eps     = (x_[index] - y0    [index]) / (sigma * s_in)
        #eps_inv = (x_[index] - y0_inv[index]) / (sigma * s_in)
    
    return eps#, eps_inv




@torch.no_grad()
def process_guides_substep(x_0, x_, eps_, data_, row, y0, y0_inv, lgw, lgw_inv, lgw_mask, lgw_mask_inv, step, sigma, sigma_next, sigma_down, s_, unsample_resample_scale, rk, rk_type, guide_mode, latent_guide, latent_guide_inv, UNSAMPLE, extra_options, frame_weights=None):
    
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

    if guide_mode == "data":
        y0_tmp = y0
        if latent_guide_inv is not None:
            y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
            y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
        x_[row+1] = y0_tmp + eps_[row]

    elif guide_mode == "hybrid_data":
        y0_tmp = apply_guided_denoising(data_[row], y0, y0_inv, lgw_mask, lgw_mask_inv, 
                                      guide_mode, extra_options=extra_options)
        x_[row+1] = y0_tmp + eps_[row]

    elif "epsilon" in guide_mode:
        if sigma > sigma_next:
            # Get hybrid-specific parameters
            hybrid_stats_weight = float(get_extra_options_kv("hybrid_stats_weight", "0.0", extra_options))
            hybrid_temporal_smooth = float(get_extra_options_kv("hybrid_temporal_smooth", "0.0", extra_options))
            hybrid_noise_normalize = float(get_extra_options_kv("hybrid_noise_normalize", "0.0", extra_options))

            # Modify the lgw masks with frame weights before any guidance
            lgw_mask_frames = lgw_mask.clone()
            lgw_mask_inv_frames = lgw_mask_inv.clone() if lgw_mask_inv is not None else None
            
            for f in range(lgw_mask_frames.shape[2]):  # Iterate over frames dimension
                frame_weight = frame_weights[f] if frame_weights is not None else 1.0
                lgw_mask_frames[..., f:f+1, :, :] *= frame_weight
                if lgw_mask_inv_frames is not None:
                    lgw_mask_inv_frames[..., f:f+1, :, :] *= frame_weight
            
            # Now proceed with either disabled or enabled lgw scaling using modified masks
            if extra_options_flag("disable_lgw_scaling", extra_options):
                eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type)
                eps_[row] = eps_[row] + lgw_mask_frames * (eps_row - eps_[row])
                if eps_row_inv is not None and lgw_mask_inv_frames is not None:
                    eps_[row] = eps_[row] + lgw_mask_inv_frames * (eps_row_inv - eps_[row])
            
            eps_guided = eps_[row].clone()
                    
            tol_value = float(get_extra_options_kv("tol", "-1.0", extra_options))
                
            if tol_value >= 0 and (lgw > 0 or lgw_inv > 0):           
                # Use lgw_mask_frames and lgw_mask_inv_frames in the L2 norm calculations
                for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                    current_diff     = torch.norm(data_[row][b][c] - y0    [b][c])
                    current_diff_inv = torch.norm(data_[row][b][c] - y0_inv[b][c])
                    
                    lgw_scaled     = torch.nan_to_num(1-(tol_value/current_diff),     0)
                    lgw_scaled_inv = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                    
                    lgw_tmp     = min(lgw    , lgw_scaled)
                    lgw_tmp_inv = min(lgw_inv, lgw_scaled_inv)

                    lgw_mask_clamp = torch.clamp(lgw_mask_frames, max=lgw_tmp)
                    lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv_frames, max=lgw_tmp_inv)
                    
                    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type, b, c)
                    eps_guided[b][c] = eps_[row][b][c] + lgw_mask_clamp[b][c] * (eps_row - eps_[row][b][c]) + lgw_mask_clamp_inv[b][c] * (eps_row_inv - eps_[row][b][c])
            
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
                    
                    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, rk_type, b, c)
                    eps_guided[b][c] = eps_[row][b][c]      +     ratio * lgw_mask_frames[b][c] * (eps_row - eps_[row][b][c])    +    ratio_inv * lgw_mask_inv_frames[b][c] * (eps_row_inv - eps_[row][b][c])

            # If hybrid mode is active (stats_weight > 0), apply statistical matching
            if guide_mode == "hybrid_epsilon" and hybrid_stats_weight > 0:
                eps_shifted = eps_guided.clone()
                
                # Calculate target noise statistics
                noise_stats = {
                    'mean': torch.tensor([eps_guided[0][c].mean() for c in range(eps_guided.shape[1])],
                                       device=eps_guided.device, dtype=eps_guided.dtype),
                    'std': torch.tensor([eps_guided[0][c].std() for c in range(eps_guided.shape[1])],
                                      device=eps_guided.device, dtype=eps_guided.dtype)
                }
                
                # Apply noise normalization if enabled
                if hybrid_noise_normalize > 0:
                    noise_mean_target = float(get_extra_options_kv("hybrid_noise_mean_target", "0.0", extra_options))
                    noise_std_target = float(get_extra_options_kv("hybrid_noise_std_target", "1.0", extra_options))
                    
                    for c in range(eps_guided.shape[1]):
                        noise_stats['mean'][c] = (1 - hybrid_noise_normalize) * noise_stats['mean'][c] + \
                                               hybrid_noise_normalize * noise_mean_target
                        noise_stats['std'][c] = (1 - hybrid_noise_normalize) * noise_stats['std'][c] + \
                                              hybrid_noise_normalize * noise_std_target
                
                # Apply per-channel statistical matching
                for b, c in itertools.product(range(eps_guided.shape[0]), range(eps_guided.shape[1])):
                    curr_mean = eps_shifted[b][c].mean()
                    curr_std = eps_shifted[b][c].std()
                    
                    # Center the noise
                    centered = eps_shifted[b][c] - curr_mean
                    
                    # Scale to target statistics
                    target_std = noise_stats['std'][c]
                    scale_factor = target_std / (curr_std + 1e-6)
                    eps_shifted[b][c] = centered * scale_factor + noise_stats['mean'][c]
                
                # Apply temporal smoothing if enabled and we have video frames
                if hybrid_temporal_smooth > 0 and eps_shifted.dim() == 5:
                    kernel_size = 5
                    padding = kernel_size // 2
                    temporal_kernel = torch.tensor(
                        [0.1, 0.2, 0.4, 0.2, 0.1],
                        device=eps_shifted.device, dtype=eps_shifted.dtype
                    ) * hybrid_temporal_smooth
                    temporal_kernel[kernel_size//2] += (1 - hybrid_temporal_smooth)
                    temporal_kernel = temporal_kernel / temporal_kernel.sum()
                    
                    # Reshape for conv1d
                    b, c, f, h, w = eps_shifted.shape
                    data_flat = eps_shifted.permute(0, 1, 3, 4, 2).reshape(-1, f)
                    
                    # Apply smoothing
                    data_smooth = F.conv1d(
                        data_flat.unsqueeze(1),
                        temporal_kernel.view(1, 1, -1),
                        padding=padding
                    ).squeeze(1)
                    
                    # Reshape back
                    eps_shifted = data_smooth.view(b, c, h, w, f).permute(0, 1, 4, 2, 3)
                
                # Blend statistical matching results back with original guided epsilon
                eps_guided = eps_guided + lgw_mask * hybrid_stats_weight * (eps_shifted - eps_guided)
            
            # Update epsilon values
            eps_[row] = eps_guided

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
    
    
    if UNSAMPLE == False and (latent_guide is not None or latent_guide_inv is not None) and guide_mode in ("hard_light", "blend", "mean_std", "mean", "mean_tiled", "std"):
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
def noise_cossim_guide_tiled_works(x_list, guide, cossim_mode="forward", tile_size=2):
    #tiles = F.unfold(x, kernel_size=tile_size, stride=tile_size)
    guide_tiled = rearrange(guide, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    
    cossim_tmp, x_tiled_list = [], []
    x_tiled_out = torch.zeros_like(guide_tiled)
    
    for i in range (len(x_list)):
        x_tiled     = rearrange(x_list[i],     "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        x_tiled_list.append(x_tiled)
        
    for j in range(guide_tiled.shape[1]):
        cossim_tmp = []
        for i in range(len(x_tiled_list)):
            cossim_tmp.append(get_cosine_similarity(x_tiled_list[i][0][j], guide_tiled[0][j]))
        for i in range(len(x_tiled_list)):
            if cossim_tmp[i] == max(cossim_tmp):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
    
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
def noise_cossim_eps_tiled_works(x_list, eps, noise_list, cossim_mode="forward", tile_size=2, step=0):

    eps_tiled = rearrange(eps, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    
    x_tiled_list, noise_tiled_list = [], []
    x_tiled_out = torch.zeros_like(eps_tiled)

    x_tiled_list = [
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for x in x_list
    ]
    noise_tiled_list = [
        rearrange(noise, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for noise in noise_list
    ]

    noise_tiled_stack2 = torch.stack([noise_tiled_list[i][0] for i in range(len(x_tiled_list))])
    thing1 = noise_tiled_stack2.view(*noise_tiled_stack2.shape[:-3],-1)
    eps_expanded = eps_tiled[0].view(*eps_tiled.shape[:-3],-1).expand_as(thing1)
    cossim_tmp_all = F.cosine_similarity(thing1, eps_expanded, dim=-1)
        
    for j in range(eps_tiled.shape[1]): #iterate over tiles (j)

        for i in range(len(x_tiled_list)):
            if   (cossim_mode == "forward") and (cossim_tmp_all[i][j] == max(cossim_tmp_all[:,j])):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
            elif (cossim_mode == "reverse") and (cossim_tmp_all[i][j] == min(cossim_tmp_all[:,j])):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
            elif (cossim_mode == "orthogonal") and (abs(cossim_tmp_all[i][j]) == min(abs(val) for val in cossim_tmp_all[:,j])):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
            elif (cossim_mode == "forward_reverse"): # and (abs(cossim_tmp[i]) == min(abs(val) for val in cossim_tmp)):
                if   (step % 2 == 0) and (cossim_tmp_all[i][j] == max(cossim_tmp_all[:,j])):
                    x_tiled_out[0][j] = x_tiled_list[i][0][j]
                elif (step % 2 == 1) and (cossim_tmp_all[i][j] == min(cossim_tmp_all[:,j])):
                    x_tiled_out[0][j] = x_tiled_list[i][0][j]
            elif (cossim_mode == "orthogonal_reverse"): # and (abs(cossim_tmp[i]) == min(abs(val) for val in cossim_tmp)):
                if   (step % 2 == 0) and (abs(cossim_tmp_all[i][j]) == min(abs(val) for val in cossim_tmp_all[:,j])):
                    x_tiled_out[0][j] = x_tiled_list[i][0][j]
                elif (step % 2 == 1) and (cossim_tmp_all[i][j] == min(cossim_tmp_all[:,j])):
                    x_tiled_out[0][j] = x_tiled_list[i][0][j]
    
    if x_tiled_out.sum() == 0:
        x_tiled_out = x_tiled_list[0]
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)
    
    return x_detiled


@torch.no_grad
def noise_cossim_guide_eps_tiled(x_0, x_list, y0, noise_list, cossim_mode="forward", tile_size=2, step=0, sigma=None, rk_type=None):

    #y0_tiled = rearrange(y0, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    #x_0_tiled = rearrange(x_0, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)

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



@torch.no_grad
def noise_cossim_guide_eps_tiled_works(x_0, x_list, y0, noise_list, cossim_mode="forward", tile_size=2, sigma=None, rk_type=None):
    #tiles = F.unfold(x, kernel_size=tile_size, stride=tile_size)
    y0_tiled = rearrange(y0, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    
    cossim_tmp, x_tiled_list, noise_tiled_list, eps_guide_tiled_list = [], [], [], []
    x_tiled_out = torch.zeros_like(y0_tiled)
    
    x_0_tiled     = rearrange(x_0,     "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    
    for i in range (len(x_list)):
        x_tiled     = rearrange(x_list[i],     "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        x_tiled_list.append(x_tiled)
        
    for i in range (len(noise_list)):
        noise_tiled     = rearrange(noise_list[i],     "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        noise_tiled_list.append(noise_tiled)
        
    for i in range (len(x_list)):
        #eps_guide = get_guide_epsilon(x_0, x_list[i], y0, sigma, rk_type)
        eps_guide = x_list[i] - y0
        eps_guide_tiled     = rearrange(eps_guide,     "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        eps_guide_tiled_list.append(eps_guide_tiled)
        
        
        
    for j in range(x_tiled.shape[1]):
        cossim_tmp = []
        for i in range(len(x_tiled_list)):
            cossim_tmp.append(get_cosine_similarity(noise_tiled_list[i][0][j], eps_guide_tiled_list[i][0][j]))
        for i in range(len(x_tiled_list)):
            if   (cossim_mode == "forward") and (cossim_tmp[i] == max(cossim_tmp)):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
            elif (cossim_mode == "reverse") and (cossim_tmp[i] == min(cossim_tmp)):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
            elif (cossim_mode == "orthogonal") and (abs(cossim_tmp[i]) == min(abs(val) for val in cossim_tmp)):
                x_tiled_out[0][j] = x_tiled_list[i][0][j]
    
    if x_tiled_out.sum() == 0:
        x_tiled_out = x_tiled_list[0]
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)
    
    return x_detiled


