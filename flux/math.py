import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.modules.attention import attention_pytorch

import comfy.model_management

import math

USE_LOG_BOOST = False

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, cross_self_mask=None, sigma=None, lamb_t_factor=0.1) -> Tensor:
    
    #derp_mask = torch.ones_like(q)
    #derp_mask_inv = torch.ones_like(q)
    #derp_mask[:,:,512:,:] = (cross_self_mask) * 0.25 + 1 
    #derp_mask_inv[:,:,512:,:] = (1-cross_self_mask) * 0.25 + 1 
    #q *= derp_mask
    ##k *= derp_mask_inv
    #v *= derp_mask_inv
    
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    
    #x = attention_pytorch(q, k, v, heads, skip_reshape=True, mask=mask)
    
    if cross_self_mask is None or cross_self_mask.sum() == 0.0:
        x = attention_pytorch(q, k, v, heads, skip_reshape=True, mask=mask)
    else:
        x = attention_rescale(q, k, v, heads, skip_reshape=True, mask=mask, cross_self_mask=cross_self_mask, sigma=sigma, lamb_t_factor=lamb_t_factor)
        
        
    #if mask is not None:
    #    x = attention_pytorch(q, k, v, heads, skip_reshape=True, mask=mask)
    #else:
    #    from comfy.ldm.modules.attention import optimized_attention
    #    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=None)
    return x


"""def attention_rescale(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = attention_pytorch(q, k, v, heads, skip_reshape=True, mask=mask)
    #if mask is not None:
    #    x = attention_pytorch(q, k, v, heads, skip_reshape=True, mask=mask)
    #else:
    #    from comfy.ldm.modules.attention import optimized_attention
    #    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=None)
    return x"""



def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu() or comfy.model_management.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)



def attention_rescale(
    q, 
    k, 
    v,
    heads,
    skip_reshape=False,
    skip_output_reshape=False,
    mask=None,
    cross_self_mask=None,
    sigma=None,
    lamb_t_factor=0.1,
) -> torch.Tensor:
    
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )
    
    L, S = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1))


    txt_len = q.shape[-2] - cross_self_mask.shape[-2]

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    
    attn_w_img = attn_weight[:,:,txt_len:, txt_len:]
    
    if mask is not None:
        R = mask[txt_len:, txt_len:]
        #M_pos = attn_w_img.max() - attn_w_img
        #M_neg = attn_w_img - attn_w_img.min() 
        
        M_pos = attn_w_img[0].max(-1)[0].unsqueeze(-1) - attn_w_img
        M_neg = attn_w_img - attn_w_img[0].min(-1)[0].unsqueeze(-1)   #watch out for batches...
        
        #lam_t = sigma.item() * lamb_t_factor
        lam_t = lamb_t_factor
        
        #M = lam_t * (1-R.to(attn_weight)) * M_neg    -   lam_t * R.to(attn_weight) * M_pos
        
        #M =  lam_t * R.to(attn_w_img) * M_pos        -   lam_t * (1-R.to(attn_w_img)) * M_neg
        
        #M =  lam_t * (1-R.to(attn_w_img)) * M_neg * cross_self_mask    -   lam_t * R.to(attn_w_img) * M_pos * cross_self_mask
        M =  lam_t * (1-R.to(attn_w_img)) * M_neg    -   lam_t * R.to(attn_w_img) * M_pos
        
        del M_pos, M_neg
        
        attn_w_img += M
        
        attn_weight[:,:,txt_len:, txt_len:] = attn_w_img
    
    
    elif mask is not None:
        R = mask #[txt_len:, txt_len:]
        #M_pos = attn_w_img.max() - attn_w_img
        #M_neg = attn_w_img - attn_w_img.min() 
        
        M_pos = attn_weight[0].max(-1)[0].unsqueeze(-1) - attn_weight
        M_neg = attn_weight - attn_weight[0].min(-1)[0].unsqueeze(-1)   #watch out for batches...
        
        #lam_t = sigma.item() * lamb_t_factor
        lam_t = lamb_t_factor
        
        #M = lam_t * (1-R.to(attn_weight)) * M_neg    -   lam_t * R.to(attn_weight) * M_pos
        
        #M =  lam_t * R.to(attn_w_img) * M_pos        -   lam_t * (1-R.to(attn_w_img)) * M_neg
        
        M =  lam_t * (1-R.to(attn_weight)) * M_neg   -   lam_t * R.to(attn_weight) * M_pos
        
        del M_pos, M_neg
        
        attn_weight += M
        
        #attn_weight[:,:,txt_len:, txt_len:] = attn_weight
    
    
    
    
    
    if False: #mask is not None:
        if not USE_LOG_BOOST:
            #attn_weight *= mask.to(attn_weight)
            
            min_value = attn_weight.min()

            # Shift to ≥0 space
            attn_weight -= min_value  # [B, H, L, S]

            # Apply mask boost (multiplicative)
            attn_weight *= mask.to(attn_weight)

            # Shift back to original range
            attn_weight += min_value
                        
            
        else:
            #safe_mask = mask.clamp(min=1e-9)
            pos_mask = attn_weight > 0
            #attn_weight[pos_mask] *= mask[pos_mask]
            for i in range(attn_weight.shape[1]):  # loop over 24 heads
                attn_slice = attn_weight[:, i, :, :]        # [1, 4608, 4608]
                mask_slice = mask                          # [4608, 4608]
                pos_mask_slice = pos_mask[:, i, :, :]      # [1, 4608, 4608]

                # Expand mask if needed
                mask_exp = mask_slice.unsqueeze(0)         # [1, 4608, 4608]

                attn_slice = torch.where(
                    pos_mask_slice,
                    attn_slice * mask_exp,
                    attn_slice
                )

                attn_weight[:, i, :, :] = attn_slice

            """attn_weight = torch.where(
                attn_weight > 0,
                attn_weight * mask,
                attn_weight  # leave negative values untouched
            )

            attn_weight += torch.log(mask.to(attn_weight).clamp(min=1e-9))"""

    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    
    """txt_len = q.shape[-2] - cross_self_mask.shape[-2]
    #derp_mask = torch.ones_like(img_q)
    v_boost = v.clone()
    v_boost[:,:,txt_len:,:] = v_boost[:,:,txt_len:,:] * (1 + 0.25 * (1-cross_self_mask))


    out = attn_weight @ v_boost"""
    #out = attn_weight @ v
    #out_cross_self = attn_weight @ v_boost
    #out_cross_self = attn_weight @ (v * (1 + 0.25 * (1-cross_self_mask)))
    
    #out[:,:,txt_len:,:] = (1-cross_self_mask) * out[:,:,txt_len:,:] + cross_self_mask * out_cross_self[:,:,txt_len:,:]

    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
    
    return out



