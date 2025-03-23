# Adapted from: https://github.com/black-forest-labs/flux

import math
import torch
from torch import Tensor, nn

from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

import torch.nn.functional as F
import einops
from einops import rearrange
from torch import Tensor
from dataclasses import dataclass

from .math import attention, rope, apply_rope
import comfy.ldm.common_dit

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim      = dim
        self.theta    = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def attention_weights(q, k):
    # implementation of in-place softmax to reduce memory req
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores.div_(math.sqrt(q.size(-1)))
    torch.exp(scores, out=scores)
    summed = torch.sum(scores, dim=-1, keepdim=True)
    scores /= summed
    return scores.nan_to_num_(0.0, 65504., -65504.)

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element. 
                    These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_layer  = operations.Linear(    in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.silu      = nn.SiLU()
        self.out_layer = operations.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.scale = nn.Parameter(torch.empty((dim), dtype=dtype, device=device))    # self.scale.shape = 128

    def forward(self, x: Tensor):
        return comfy.ldm.common_dit.rms_norm(x, self.scale, 1e-6)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)
        self.key_norm   = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads    # 24
        head_dim  = dim // num_heads   # 128 = 3072 / 24

        self.qkv  = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim,                               dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim,                    dtype=dtype, device=device)    # dim is usually 3072


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, dtype=None, device=None, operations=None):
        super().__init__()
        self.is_double  = double
        self.multiplier = 6 if double else 3
        self.lin        = operations.Linear(dim, self.multiplier * dim, bias=True, dtype=dtype, device=device)

    def forward(self, vec: Tensor) -> tuple:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]),    ModulationOut(*out[3:]) if self.is_double else None,)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, dtype=None, device=None, operations=None, idx=-1):
        super().__init__()

        self.idx         = idx

        mlp_hidden_dim   = int(hidden_size * mlp_ratio)
        self.num_heads   = num_heads
        self.hidden_size = hidden_size
        
        self.img_mod     = Modulation(hidden_size, double=True,                                   dtype=dtype, device=device, operations=operations) # in_features=3072, out_features=18432 (3072*6)
        self.txt_mod     = Modulation(hidden_size, double=True,                                   dtype=dtype, device=device, operations=operations) # in_features=3072, out_features=18432 (3072*6)

        self.img_attn    = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations) # .qkv: in_features=3072, out_features=9216   .proj: 3072,3072
        self.txt_attn    = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations) # .qkv: in_features=3072, out_features=9216   .proj: 3072,3072

        self.img_norm1   = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6,  dtype=dtype, device=device)
        self.txt_norm1   = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6,  dtype=dtype, device=device)

        self.img_norm2   = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6,  dtype=dtype, device=device)
        self.txt_norm2   = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6,  dtype=dtype, device=device)

        self.img_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        ) # 3072->12288, 12288->3072  (3072*4)
        
        self.txt_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        ) # 3072->12288, 12288->3072  (3072*4)
    
    def img_attn_preproc(self, img, img_mod1):
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv       = self.img_attn.qkv(img_modulated)
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k        = self.img_attn.norm(img_q, img_k, img_v)
        return img_q, img_k, img_v
    
    def txt_attn_preproc(self, txt, txt_mod1):
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv       = self.txt_attn.qkv(txt_modulated)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)    # Batch SeqLen (9216==3*3072) -> 3*1 24 SeqLen 128
        txt_q, txt_k        = self.txt_attn.norm(txt_q, txt_k, txt_v)
        return txt_q, txt_k, txt_v
    
    

    
    # ADDED THIS TIMESTEP = NONE     2-28-25          mask.shape 4608,4608
    #def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, timestep=None, transformer_options={}, mask=None, weight=1): # vec 1,3072      #img_attn.shape 1,4096,3072    txt_attn.shape 1,512,3072
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, mask=None, reg_cond_mask=None, idx=0) -> Tuple[Tensor, Tensor]: # vec 1,3072

        img_mod1, img_mod2  = self.img_mod(vec) # -> 3072, 3072
        txt_mod1, txt_mod2  = self.txt_mod(vec)
        
        img_q, img_k, img_v = self.img_attn_preproc(img, img_mod1)
        txt_q, txt_k, txt_v = self.txt_attn_preproc(txt, txt_mod1)

        q, k, v = torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2)
        
        """if idx % 2 == 1:
            trimask = torch.tril(torch.ones(img.shape[1], img.shape[1])).to(mask.dtype).to(mask.device)
            attn_mask = mask.clone()
            attn_mask[txt.shape[1]:,txt.shape[1]:] = torch.logical_or(trimask, mask[txt.shape[1]:,txt.shape[1]:])
        if idx % 2 == 0:
            trimask = ~torch.tril(torch.ones(img.shape[1], img.shape[1])).to(mask.dtype).to(mask.device)
            attn_mask = mask.clone()
            attn_mask[txt.shape[1]:,txt.shape[1]:] = torch.logical_or(trimask, mask[txt.shape[1]:,txt.shape[1]:])"""
        
        if reg_cond_mask is None:
            attn = attention(q, k, v, pe=pe, mask=mask)
            
            txt_attn = attn[:, : txt.shape[1]   ]                         # 1, 768,3072
            img_attn = attn[:,   txt.shape[1] : ]  
            
            img += img_mod1.gate * self.img_attn.proj(img_attn)
            txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
            
            img += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
            txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
            
        else:
            #reg_cond_mask = reg_cond_mask.unsqueeze(0)
            mask_inv_selfattn = mask.clone()
            #mask_inv_selfattn[txt.shape[1]:,txt.shape[1]:] = torch.clamp(mask_inv_selfattn[txt.shape[1]:,txt.shape[1]:], max=0.0)
            mask_inv_selfattn[txt.shape[1]:,txt.shape[1]:] = True
                        
            attn = attention(q, k, v, pe=pe, mask=mask)
            
            txt_attn = attn[:, : txt.shape[1]   ]                         # 1, 768,3072
            img_attn = attn[:,   txt.shape[1] : ]  
                        
            attn_maskless = attention(q, k, v, pe=pe, mask=mask_inv_selfattn)
            
            txt_attn_maskless = attn_maskless[:, : txt.shape[1]   ]                         # 1, 768,3072
            img_attn_maskless = attn_maskless[:,   txt.shape[1] : ]  
            
            img_attn = reg_cond_mask * img_attn + (1-reg_cond_mask) * img_attn_maskless

            img += img_mod1.gate * self.img_attn.proj(img_attn)
            txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
            
            img += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
            txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        # Compute a new mask from cross-attention heatmaps when idx==5
        attn_mask = None
        
        return img, txt, attn_mask
        
        if idx == 5:
            # --- Compute a new mask from cross-attention over image tokens ---
            # Use only the first 256 tokens from txt_q to form concept vectors.
            ca_cv = einops.rearrange(txt_q[..., :256, :], "b h hw d -> b hw (h d)")
            # Rearrange image keys (for image tokens) to get image vectors.
            ca_iv = einops.rearrange(img_k, "b h hw d -> b hw (h d)")
            
            # Compute heatmaps: shape becomes (batch, concepts, patches).
            heatmaps = einops.einsum(
                ca_iv,
                ca_cv,
                "batch patches dim, batch concepts dim -> batch concepts patches",
            )
            # For simplicity, assume batch size = 1.
            # Average over the "concepts" dimension to get one heat value per image token.
            heatmap = heatmaps.mean(dim=1).squeeze(0)  # shape: (4096,)
            
            # Create a binary vector: tokens with nonnegative heat become True.
            binary_vec = heatmap >= 0  # shape: (4096,)
            # Form a symmetric mask for the image block.
            new_image_mask = binary_vec.unsqueeze(1) & binary_vec.unsqueeze(0)  # shape: (4096, 4096)
            
            # Smooth the new image mask to encourage contiguous regions.
            smoothed_new_image_mask = smooth_binary_mask(new_image_mask, kernel_size=5, sigma=2.0, threshold=0.5)
            
            # --- Merge new mask with the old mask ---
            # Assume 'mask' is the old self-attention mask of shape (4608, 4608),
            # with the first txt.shape[1] tokens corresponding to text and the remaining 4096 to image.
            old_mask = mask.clone()
            final_mask = old_mask.clone()
            # Replace the image-image block with our smoothed new image mask.
            final_mask[txt.shape[1]:, txt.shape[1]:] = smoothed_new_image_mask
            
            # --- Override based on a threshold ---
            # For image tokens, if the absolute heatmap value is > 100, use the new (smoothed) mask.
            cond = heatmap.abs() > 100  # shape: (4096,)
            cond_2d = cond.unsqueeze(1) | cond.unsqueeze(0)  # shape: (4096, 4096)
            
            image_region = final_mask[txt.shape[1]:, txt.shape[1]:]  # shape: (4096, 4096)
            image_region = torch.where(cond_2d, smoothed_new_image_mask, image_region)
            final_mask[txt.shape[1]:, txt.shape[1]:] = image_region
            
            attn_mask = final_mask

        return img, txt, attn_mask
                
        
        
        if idx == 5:
            # --- Compute a new mask from cross-attention over image tokens ---
            # Use only the first 256 tokens from txt_q to form concept vectors
            ca_cv = einops.rearrange(txt_q[..., :256, :], "b h hw d -> b hw (h d)")
            # Rearrange image keys (for image tokens) to get image vectors
            ca_iv = einops.rearrange(img_k, "b h hw d -> b hw (h d)")
            
            # Compute heatmaps: shape becomes (batch, concepts, patches)
            heatmaps = einops.einsum(
                ca_iv,
                ca_cv,
                "batch patches dim, batch concepts dim -> batch concepts patches",
            )
            # Average over the "concepts" dimension to get one heat value per image token.
            # (Assume batch size is 1)
            heatmap = heatmaps.mean(dim=1).squeeze(0)  # shape: (4096,)
            
            # Create a binary vector: tokens with nonnegative heat become True.
            binary_vec = heatmap >= 0  # shape: (4096,)
            # Form a symmetric mask for the image block:
            new_image_mask = binary_vec.unsqueeze(1) & binary_vec.unsqueeze(0)  # shape: (4096, 4096)
            
            # --- Merge new mask with the old mask ---
            # Assume 'mask' is the old self-attention mask of shape (4608, 4608),
            # where the first 512 tokens correspond to text and the remaining 4096 to image.
            final_mask = mask.clone()
            # Replace the image portion with the new mask
            final_mask[txt.shape[1]:, txt.shape[1]:] = new_image_mask
            
            # --- Override based on a threshold ---
            # For the image tokens only, if the absolute heatmap value is > 100, use the new mask.
            cond = heatmap.abs() > 100  # shape: (4096,)
            cond_2d = cond.unsqueeze(1) | cond.unsqueeze(0)  # shape: (4096, 4096)
            
            # Update the image block of the final mask accordingly:
            image_region = final_mask[txt.shape[1]:, txt.shape[1]:]  # shape: (4096, 4096)
            image_region = torch.where(cond_2d, new_image_mask, image_region)
            final_mask[txt.shape[1]:, txt.shape[1]:] = image_region

            attn_mask = final_mask

        return img, txt, attn_mask


        """attn_mask = None
        if idx == 5:
            ca_cv = einops.rearrange(txt_q[...,:256,:], "b h hw d -> b hw (h d)")
            ca_iv = einops.rearrange(img_k, "b h hw d -> b hw (h d)")

            heatmaps = einops.einsum(       # CONCEPT                               CROSS-ATTN
                ca_iv,              # img_attn     (txt+img)      or        img_q     (    img)
                ca_cv,            # concept_attn (txt)          or        concept_q (txt+img)   
                "batch patches dim, batch concepts dim -> batch concepts patches",
            )
            
            # Assume heatmap is a tensor of shape (1, 4096)
            heatmaps = heatmaps.mean(dim=1)  # now shape (4096,)
            binary_vec = heatmaps >= 0      # shape (4096,), boolean

            # Create a symmetric self-attention mask (4096 x 4096)
            attn_mask = binary_vec.unsqueeze(1) & binary_vec.unsqueeze(0)
            new_mask = mask.clone()
            new_mask[txt.shape[1]:, txt.shape[1]:] = attn_mask
            attn_mask = new_mask
            
            cond = heatmap.abs() > 100  # shape (4096,)

            # Lift the condition to a 2D mask.
            # For a self-attention mask, you might want the pair (i,j) to be new if either token meets the condition.
            cond_2d = cond.unsqueeze(1) | cond.unsqueeze(0)  # shape (4096, 4096)

            # Now, choose: if cond_2d is True, use the new_mask value; otherwise, use the old_mask value.
            final_mask = torch.where(cond_2d, new_mask, old_mask)
            
            
            new_mask = mask.clone()
            new_mask[txt.shape[1]:, txt.shape[1]:] = attn_mask
            attn_mask = new_mask"""



        return img, txt, attn_mask
    
    

    
    # ADDED THIS TIMESTEP = NONE     2-28-25          mask.shape 4608,4608
    #def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, timestep=None, transformer_options={}, mask=None, weight=1): # vec 1,3072      #img_attn.shape 1,4096,3072    txt_attn.shape 1,512,3072
    def forward_mega(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, mask=None, reg_cond_mask=None, reg_vec=None, vec_a=None,vec_b=None, txt_a=None,txt_b=None,pe_a=None,pe_b=None) -> Tuple[Tensor, Tensor]: # vec 1,3072

        img_mod1, img_mod2  = self.img_mod(vec) # -> 3072, 3072
        txt_mod1, txt_mod2  = self.txt_mod(vec)
        
        img_q, img_k, img_v = self.img_attn_preproc(img, img_mod1)
        txt_q, txt_k, txt_v = self.txt_attn_preproc(txt, txt_mod1)
        
        #img_mod1z, img_mod2z  = self.img_mod(torch.zeros_like(vec)) # -> 3072, 3072
        #img_qz, img_kz, img_vz = self.img_attn_preproc(img, img_mod1z)
        #txt_mod1z, txt_mod2z  = self.txt_mod(torch.zeros_like(vec))
        
        #txt_qa, txt_ka, txt_va = self.txt_attn_preproc(txt_a, txt_mod1z)
        #txt_qb, txt_kb, txt_vb = self.txt_attn_preproc(txt_b, txt_mod1z)

        q, k, v = torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2)
        #q, k = apply_rope(q, k, pe)
        
        #reg_cond_mask = None
        
        if reg_cond_mask is None:
            attn = attention(q, k, v, pe=pe, mask=mask)
            
            txt_attn = attn[:, : txt.shape[1]   ]                         # 1, 768,3072
            img_attn = attn[:,   txt.shape[1] : ]  
            
            img += img_mod1.gate * self.img_attn.proj(img_attn)
            txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
            
            img += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
            txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
            
        else:
            #reg_cond_mask = reg_cond_mask.unsqueeze(0)
            mask_inv_selfattn = mask.clone()
            #mask_inv_selfattn[txt.shape[1]:,txt.shape[1]:] = torch.clamp(mask_inv_selfattn[txt.shape[1]:,txt.shape[1]:], max=0.0)
            mask_inv_selfattn[txt.shape[1]:,txt.shape[1]:] = True
            
            #mask_inv_selfattn[txt.shape[1]:, txt.shape[1]:] = mask[txt.shape[1]:, txt.shape[1]:] == False
            
            attn = attention(q, k, v, pe=pe, mask=mask)
            
            txt_attn = attn[:, : txt.shape[1]   ]                         # 1, 768,3072
            img_attn = attn[:,   txt.shape[1] : ]  
            
            #mask_inv_selfattn = mask[]
            
            attn_maskless = attention(q, k, v, pe=pe, mask=mask_inv_selfattn)
            
            txt_attn_maskless = attn_maskless[:, : txt.shape[1]   ]                         # 1, 768,3072
            img_attn_maskless = attn_maskless[:,   txt.shape[1] : ]  
            
            img_attn = reg_cond_mask * img_attn + (1-reg_cond_mask) * img_attn_maskless
            #img_attn = (1-reg_cond_mask) * img_attn + reg_cond_mask * img_attn_maskless
            
            #img_maskless = img.clone()
            #txt_maskless = txt.clone()
        
            img += img_mod1.gate * self.img_attn.proj(img_attn)
            txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
            
            img += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
            txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
            
            
            
            #img_maskless += img_mod1.gate * self.img_attn.proj(img_attn_maskless)
            #txt_maskless += txt_mod1.gate * self.txt_attn.proj(txt_attn_maskless)
            
            #img_maskless += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img_maskless) + img_mod2.shift)
            #txt_maskless += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt_maskless) + txt_mod2.shift)
            
            #img = reg_cond_mask * img + (1-reg_cond_mask) * img_maskless
        heatmap=None
        heatmap_a, heatmap_b = None, None
        if mask is not None and False:
            attn_unmasked = attention(q, k, v, pe=pe, mask=None)
            heatmap = einops.einsum(       # CONCEPT                               CROSS-ATTN
                attn_unmasked[:,512:,:],                  # img_attn     (txt+img)      or        img_q     (    img)
                attn[:,512:,:],            # concept_attn (txt)          or        concept_q (txt+img)   
                "batch patches dim, batch concepts dim -> batch concepts patches",
            ).mean(dim=1,keepdim=True).detach().cpu().to(torch.float32).reshape(1,64,64).squeeze(0)
            
        if mask is not None and False:
            q_rope, k_rope = apply_rope(q, k, pe)

            txt_q_rope = q_rope[...,:512,:]
            txt_k_rope = k_rope[...,:512,:]
            img_q_rope = q_rope[...,512:,:]
            img_k_rope = k_rope[...,512:,:]
            
            heatmap = attention_weights_alt(txt_q_rope, img_k_rope).detach().to('cpu')
            
        
        if mask is not None:

            img_mod1a, img_mod2a  = self.img_mod(vec_a) # -> 3072, 3072
            img_mod1b, img_mod2b  = self.img_mod(vec_b) # -> 3072, 3072

            img_qa, img_ka, img_va = self.img_attn_preproc(img, img_mod1a)
            img_qb, img_kb, img_vb = self.img_attn_preproc(img, img_mod1b)

            txt_mod1a, txt_mod2a  = self.txt_mod(vec_a)
            txt_mod1b, txt_mod2b  = self.txt_mod(vec_b)

            txt_qa, txt_ka, txt_va = self.txt_attn_preproc(txt_a, txt_mod1a)
            txt_qb, txt_kb, txt_vb = self.txt_attn_preproc(txt_b, txt_mod1b)

            qa = torch.cat((txt_qa, img_qa), dim=2)
            qb = torch.cat((txt_qb, img_qb), dim=2)
            ka = torch.cat((txt_ka, img_qa), dim=2)
            kb = torch.cat((txt_kb, img_qb), dim=2)
            va = torch.cat((txt_va, img_qa), dim=2)
            vb = torch.cat((txt_vb, img_qb), dim=2)

            qa_rope, ka_rope = apply_rope(qa, ka, pe_a)
            qb_rope, kb_rope = apply_rope(qb, kb, pe_a)
            q_rope, k_rope = apply_rope(q, k, pe)

            txt_q_rope = q_rope[...,:512,:]
            txt_k_rope = k_rope[...,:512,:]
            img_q_rope = q_rope[...,512:,:]
            img_k_rope = k_rope[...,512:,:]

            txt_qa_rope = qa_rope[...,:256,:]
            txt_ka_rope = ka_rope[...,:256,:]
            img_qa_rope = qa_rope[...,256:,:]
            img_ka_rope = ka_rope[...,256:,:]

            txt_qb_rope = qb_rope[...,:256,:]
            txt_kb_rope = kb_rope[...,:256,:]
            img_qb_rope = qb_rope[...,256:,:]
            img_kb_rope = kb_rope[...,256:,:]
            
            #heatmap_a = attention_weights_alt(txt_qa_rope, img_k_rope).detach().mean(dim=1).to('cpu')
            #heatmap_b = attention_weights_alt(txt_qb_rope, img_k_rope).detach().mean(dim=1).to('cpu')
            
            attn_a = attention(qa, ka, va, pe=pe_a, mask=None)
            attn_b = attention(qb, kb, va, pe=pe_b, mask=None)
            
            #attn_a = rearrange(attn_a, "b h l d -> b l (h d)")
            #attn_b = rearrange(attn_b, "b h l d -> b l (h d)")
            
            heatmap_a = einops.einsum(       # CONCEPT                               CROSS-ATTN
                img_attn, #[:,512:,:],                  # img_attn     (txt+img)      or        img_q     (    img)
                attn_a[:,:256,:],            # concept_attn (txt)          or        concept_q (txt+img)   
                "batch patches dim, batch concepts dim -> batch concepts patches",
            ).to('cpu')
            heatmap_b = einops.einsum(       # CONCEPT                               CROSS-ATTN
                img_attn, #[:,512:,:],                  # img_attn     (txt+img)      or        img_q     (    img)
                attn_b[:,:256,:],            # concept_attn (txt)          or        concept_q (txt+img)   
                "batch patches dim, batch concepts dim -> batch concepts patches",
            ).to('cpu')
            
            
            
        return img, txt, txt_a, txt_b #heatmap_a, heatmap_b #, mask_resized



class SingleStreamBlock(nn.Module):      #attn.shape = 1,4608,3072       mlp.shape = 1,4608,12288     4096*3 = 12288
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """
    def __init__(self, hidden_size: int,  num_heads: int, mlp_ratio: float = 4.0, qk_scale: float = None, dtype=None, device=None, operations=None, idx=-1):
        super().__init__()
        self.idx            = idx
        self.hidden_dim     = hidden_size #3072
        self.num_heads      = num_heads    #24
        head_dim            = hidden_size // num_heads
        self.scale          = qk_scale or head_dim**-0.5   #0.08838834764831845

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)    #12288== 3072 * 4
        # qkv and mlp_in
        self.linear1        = operations.Linear(hidden_size, 3*hidden_size + self.mlp_hidden_dim, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2        = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size,     dtype=dtype, device=device)

        self.norm           = QKNorm(head_dim,                                                      dtype=dtype, device=device, operations=operations)

        self.hidden_size    = hidden_size #3072
        self.pre_norm       = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.mlp_act        = nn.GELU(approximate="tanh")
        self.modulation     = Modulation(hidden_size, double=False,                                 dtype=dtype, device=device, operations=operations)
        
    def img_attn(self, img, mod, pe, mask, reg_cond_mask=None, txt_len=None):
        img_mod  = (1 + mod.scale) * self.pre_norm(img) + mod.shift   # mod => vec
        qkv, mlp = torch.split(self.linear1(img_mod), [3*self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v  = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k     = self.norm(q, k, v)

        attn     = attention(q, k, v, pe=pe, mask=mask)
        
        #reg_cond_mask = None
        
        if reg_cond_mask is not None:
            
            mask_inv_selfattn = mask.clone()
            #mask_inv_selfattn[txt_len:, txt_len:] = mask[txt_len:, txt_len:] == False
            #mask_inv_selfattn[txt_len:, txt_len:] = torch.clamp(mask_inv_selfattn[txt_len:, txt_len:], max=0.0)
            mask_inv_selfattn[txt_len:,txt_len:] = True
            
            attn_maskless = attention(q, k, v, pe=pe, mask=mask_inv_selfattn)
            
            txt_attn_maskless = attn_maskless[:, : txt_len   ]                         # 1, 768,3072
            img_attn_maskless = attn_maskless[:,   txt_len : ]
            
            img_attn = attn[:,   txt_len : ]

            img_attn = reg_cond_mask * img_attn + (1-reg_cond_mask) * img_attn_maskless
            #img_attn = (1-reg_cond_mask) * img_attn + reg_cond_mask * img_attn_maskless
            
            attn[:,   txt_len : ] = img_attn

        return attn, mlp

    # vec 1,3072    x 1,9984,3072
    def forward(self, img: Tensor, vec: Tensor, pe: Tensor, mask=None, reg_cond_mask=None, reg_vec=None, idx=0) -> Tensor:   # x 1,9984,3072 if 2 reg embeds, 1,9472,3072 if none    # 9216x4096 = 16x1536x1536
        mod, _    = self.modulation(vec)
        
        txt_len = None
        if reg_cond_mask is not None:
            txt_len = img.shape[-2] - reg_cond_mask.shape[-2]
            
        """if idx % 2 == 1:
            trimask = torch.tril(torch.ones(4096,4096)).to(mask.dtype).to(mask.device)
            attn_mask = mask.clone()
            attn_mask[512:,512:] = torch.logical_or(trimask, mask[512:,512:])
        if idx % 2 == 0:
            trimask = ~torch.tril(torch.ones(4096, 4096)).to(mask.dtype).to(mask.device)
            attn_mask = mask.clone()
            attn_mask[512:,512:] = torch.logical_or(trimask, mask[512:,512:])
        attn, mlp = self.img_attn(img, mod, pe, attn_mask, reg_cond_mask, txt_len)"""
        
        attn, mlp = self.img_attn(img, mod, pe, mask, reg_cond_mask, txt_len)
        output    = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        img      += mod.gate * output
        return img



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,                                        dtype=None,  device=None, operations=None):
        super().__init__()
        self.norm_final       = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6,               dtype=dtype, device=device)
        self.linear           = operations.Linear(hidden_size, patch_size * patch_size * out_channels,   bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
    
    
    
    
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention_alt(
    query, 
    key, 
    value,
    attn_mask=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value





    
# Efficient implementation equivalent to the following:
def attention_weights_alt(
    query, 
    key, 
    attn_mask=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight



    
def smooth_binary_mask(binary_mask: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0, threshold: float = 0.5) -> torch.Tensor:
    """
    Smooths a binary mask (values 0/1) using a Gaussian blur and re-thresholds it.
    
    Args:
        binary_mask: Tensor of shape (H, W) with values 0 or 1.
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian.
        threshold: Threshold value for binarizing the smoothed output.
        
    Returns:
        A binary mask of shape (H, W) after smoothing.
    """
    # Create a Gaussian kernel.
    ax = torch.arange(kernel_size, dtype=torch.float32, device=binary_mask.device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(- (xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, kernel_size, kernel_size)
    
    # Prepare the mask for convolution: shape (1, 1, H, W)
    bm = binary_mask.float().unsqueeze(0).unsqueeze(0)
    smoothed = F.conv2d(bm, kernel, padding=kernel_size // 2)
    smoothed = smoothed.squeeze(0).squeeze(0)
    
    # Re-threshold: values >= threshold become 1, otherwise 0.
    return (smoothed >= threshold).float()

    