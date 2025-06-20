import torch
import torch.nn.functional as F
import math

import torch.nn as nn
from torch import Tensor, FloatTensor
from typing import Optional, Callable, Tuple, List, Dict, Any, Union, TYPE_CHECKING, TypeVar
from dataclasses import dataclass

import einops
from einops import repeat, rearrange

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
import torch.nn.functional as F

from comfy.ldm.flux.math import apply_rope, rope
#from comfy.ldm.flux.layers import LastLayer
from ..flux.layers import LastLayer

from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
import comfy.model_management
import comfy.ldm.common_dit

from ..helper  import ExtraOptions
from ..latents import slerp_tensor, interpolate_spd, tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch, adain_seq, apply_scattersort

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate : Tensor
    


class BlockType:
    Double = 2
    Single = 1
    Zero   = 0


#########################################################################################################################################################################
class HDBlock(nn.Module):
    def __init__(
        self,
        dim                   : int,
        heads                 : int,
        head_dim              : int,
        num_routed_experts    : int = 4,
        num_activated_experts : int = 2,
        block_type            : BlockType = BlockType.Zero,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        block_classes = {
            BlockType.Double : HDBlockDouble,
            BlockType.Single : HDBlockSingle,
        }
        self.block = block_classes[block_type](dim, heads, head_dim, num_routed_experts, num_activated_experts, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        img       :          FloatTensor,
        img_masks : Optional[FloatTensor]  = None,
        txt       : Optional[FloatTensor]  = None,
        clip      :          FloatTensor   = None,
        rope      :          FloatTensor   = None,
        mask      : Optional[FloatTensor]  = None,
        update_cross_attn : Optional[Dict] = None,
        cache_v   :                  bool  = False,
        attninj_opts :      Optional[Dict] = {},
        CACHE_ATTN: bool=False,
    ) -> FloatTensor:
        return self.block(img, img_masks, txt, clip, rope, mask, update_cross_attn, cache_v, attninj_opts, CACHE_ATTN=CACHE_ATTN)



# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta    = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([  rope(ids[..., i], self.axes_dim[i], self.theta)   for i in range(n_axes)],    dim=-3,)
        return emb.unsqueeze(2)

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size   = 2,
        in_channels  = 4,
        out_channels = 1024,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.proj         = operations.Linear(in_channels * patch_size * patch_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, latent):
        latent = self.proj(latent)
        return latent

class PooledEmbed(nn.Module):
    def __init__(self, text_emb_dim, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, pooled_embed):
        return self.pooled_embedder(pooled_embed)

class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj         = Timesteps       (num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, t, wdtype):
        t_emb = self.time_proj(t).to(dtype=wdtype)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb

class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(in_features=in_features, out_features=hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states






class HDFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim                : int,
        hidden_dim         : int,
        multiple_of        : int             = 256,
        ffn_dim_multiplier : Optional[float] = None,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        
        if ffn_dim_multiplier is not None:  # custom dim factor multiplier
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, cache_v=False): # 1,4096,2560 -> 
        x1 = self.w1(x)
        x3 = self.w3(x)
        
        if hasattr(self, "x1"):
            x1 = ScatterSort.apply(x1, self.x1.to("cuda", non_blocking=True))
            del self.x1
        elif cache_v:
            self.x1 = x1.to("cpu", non_blocking=True).pin_memory()

        if hasattr(self, "x3"):
            x3 = ScatterSort.apply(x3, self.x3.to("cuda", non_blocking=True))
            del self.x3
        elif cache_v:
            self.x3 = x3.to("cpu", non_blocking=True).pin_memory()
        
        x1 = torch.nn.functional.silu(x1)
        x13 = x1 * x3
        
        if hasattr(self, "x13"):
            x13 = ScatterSort.apply(x13, self.x13.to("cuda", non_blocking=True))
            del self.x13
        elif cache_v:
            self.x13 = x13.to("cpu", non_blocking=True).pin_memory()
        
        x2 = self.w2(x13)
        
        if hasattr(self, "x2"):
            x2 = ScatterSort.apply(x2, self.x2.to("cuda", non_blocking=True))
            del self.x2
        elif cache_v:
            self.x2 = x2.to("cpu", non_blocking=True).pin_memory()
        
        return x2
        #return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class HDMoEGate(nn.Module):
    def __init__(self, dim, num_routed_experts=4, num_activated_experts=2, dtype=None, device=None):
        super().__init__()
        self.top_k            = num_activated_experts # 2
        self.n_routed_experts = num_routed_experts    # 4
        self.gating_dim       = dim                   # 2560
        self.weight           = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), dtype=dtype, device=device))

    def forward(self, x):
        x      = x.view(-1, x.shape[-1]) # 4032,2560    # below is just matmul... hidden_states @ self.weight.T
        logits = F.linear(x, comfy.model_management.cast_to(self.weight, dtype=x.dtype, device=x.device), None)
        scores = logits.softmax(dim=-1)       # logits.shape == 4032,4   scores.shape == 4032,4
        return torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

class HDMOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim                   : int,
        hidden_dim            : int,
        num_routed_experts    : int,
        num_activated_experts : int,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.shared_experts =                HDFeedForwardSwiGLU(dim, hidden_dim // 2,  dtype=dtype, device=device, operations=operations)
        self.experts        = nn.ModuleList([HDFeedForwardSwiGLU(dim, hidden_dim     ,  dtype=dtype, device=device, operations=operations) for i in range(num_routed_experts)])
        self.gate           = HDMoEGate(dim, num_routed_experts, num_activated_experts, dtype=dtype, device=device)
        self.num_activated_experts = num_activated_experts

    def forward(self, x, cache_v=False):
        y_shared = self.shared_experts(x)   # 1,4096,2560 -> 1,4096,2560 

        topk_weight, topk_idx = self.gate(x) # -> 4096,2   4096,2
        flat_topk_idx         = topk_idx.view(-1) # -> 8192,
        
        x = x.view(-1, x.shape[-1]) # -> 4096,2560
        x = x.repeat_interleave(self.num_activated_experts, dim=0) # -> 8192,2560
        
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(x.dtype)
        y = torch.einsum('bk,bkd->bd', topk_weight, y.view(*topk_weight.shape, -1))
        
        return y.view_as(y_shared) + y_shared


class ScatterSort:
    buffer = {}

    @staticmethod
    def apply(denoised_embed, y0_adain_embed):
        buf = ScatterSort.buffer
        buf['src_idx']    = denoised_embed.argsort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y0_adain_embed.sort(dim=-2)

        return denoised_embed.scatter_(
            dim=-2, 
            index=buf['src_idx'], 
            src=buf['ref_sorted'].expand_as(buf['ref_sorted'])
        )

class AttentionBuffer:
    buffer = {}


def attention(q: Tensor, k: Tensor, v: Tensor, rope: Tensor, mask: Optional[Tensor] = None):
    q, k = apply_rope(q, k, rope)
    if mask is not None:
        AttentionBuffer.buffer = attention_pytorch(
            q.view(q.shape[0], -1, q.shape[-1] * q.shape[-2]), 
            k.view(k.shape[0], -1, k.shape[-1] * k.shape[-2]), 
            v.view(v.shape[0], -1, v.shape[-1] * v.shape[-2]), 
            q.shape[2],
            mask=mask,
            )
    else:
        AttentionBuffer.buffer = optimized_attention(
            q.view(q.shape[0], -1, q.shape[-1] * q.shape[-2]), 
            k.view(k.shape[0], -1, k.shape[-1] * k.shape[-2]), 
            v.view(v.shape[0], -1, v.shape[-1] * v.shape[-2]), 
            q.shape[2],
            mask=mask,
            )
    return AttentionBuffer.buffer

class HDAttention(nn.Module):
    def __init__(
        self,
        query_dim        : int,
        heads            : int   = 8,
        dim_head         : int   = 64,

        eps              : float = 1e-5,
        out_dim          : int   = None,
        single           : bool  = False,
        dtype=None, device=None, operations=None
    ):

        super().__init__()
        self.inner_dim          = out_dim if out_dim is not None else dim_head * heads
        self.query_dim          = query_dim
        self.out_dim            = out_dim if out_dim is not None else query_dim

        self.heads              = out_dim // dim_head if out_dim is not None else heads
        self.single             = single

        self.to_q               = operations.Linear (self.query_dim, self.inner_dim, dtype=dtype, device=device)
        self.to_k               = operations.Linear (self.inner_dim, self.inner_dim, dtype=dtype, device=device)
        self.to_v               = operations.Linear (self.inner_dim, self.inner_dim, dtype=dtype, device=device)
        self.to_out             = operations.Linear (self.inner_dim, self.out_dim,   dtype=dtype, device=device)
        self.q_rms_norm         = operations.RMSNorm(self.inner_dim, eps,            dtype=dtype, device=device)
        self.k_rms_norm         = operations.RMSNorm(self.inner_dim, eps,            dtype=dtype, device=device)

        if not single:
            self.to_q_t         = operations.Linear (self.query_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_k_t         = operations.Linear (self.inner_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_v_t         = operations.Linear (self.inner_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_out_t       = operations.Linear (self.inner_dim, self.out_dim,   dtype=dtype, device=device)
            self.q_rms_norm_t   = operations.RMSNorm(self.inner_dim, eps,            dtype=dtype, device=device)
            self.k_rms_norm_t   = operations.RMSNorm(self.inner_dim, eps,            dtype=dtype, device=device)

    def forward(
        self,
        img       :          FloatTensor,
        img_masks : Optional[FloatTensor] = None,
        txt       : Optional[FloatTensor] = None,
        rope      :          FloatTensor  = None,
        mask      : Optional[FloatTensor] = None,
        update_cross_attn : Optional[Dict]= None,
        cache_v : bool = False,
        attninj_opts : Optional[Dict] = {},
    ) -> Tensor:
        
        attn_cache = {}
        for name in ["img_q_cache", "img_k_cache", "img_v_cache", "txt_q_cache", "txt_k_cache", "txt_v_cache"]:
            if hasattr(self, name):
                attn_cache[name] = getattr(self, name).to("cuda", non_blocking=True)
                
        norm_seq = adain_seq
        if self.EO("attninj_scattersort"):
            norm_seq = ScatterSort.apply

        bsz = img.shape[0]

        img_q = self.q_rms_norm(self.to_q(img))
        img_k = self.k_rms_norm(self.to_k(img))
        img_v =                 self.to_v(img)

        weight_img_q = attninj_opts.get("img_q", 0.0)
        weight_img_k = attninj_opts.get("img_k", 0.0)
        weight_img_v = attninj_opts.get("img_v", 0.0)
        
        weight_img_q_norm = attninj_opts.get("img_q_norm", 0.0)
        weight_img_k_norm = attninj_opts.get("img_k_norm", 0.0)
        weight_img_v_norm = attninj_opts.get("img_v_norm", 0.0)

        if cache_v:
            if weight_img_q != 0 or weight_img_q_norm != 0:
                self.img_q_cache = img_q.to("cpu", non_blocking=True).pin_memory()
            if weight_img_k != 0 or weight_img_k_norm != 0:
                self.img_k_cache = img_k.to("cpu", non_blocking=True).pin_memory()
            if weight_img_v != 0 or weight_img_v_norm != 0:
                self.img_v_cache = img_v.to("cpu", non_blocking=True).pin_memory()
        else: 
            if self.EO("slerp"):
                interp_fn = slerp_tensor
            else:
                interp_fn = lambda weight, A, B: (1-weight) * A + weight * B

            if weight_img_q != 0 and "img_q_cache" in attn_cache:
                img_q = interp_fn(weight_img_q, img_q, attn_cache["img_q_cache"])
            if weight_img_k != 0 and "img_k_cache" in attn_cache:
                img_k = interp_fn(weight_img_k, img_k, attn_cache["img_k_cache"])
            if weight_img_v != 0 and "img_v_cache" in attn_cache:
                img_v = interp_fn(weight_img_v, img_v, attn_cache["img_v_cache"])

            if weight_img_q_norm != 0 and "img_q_cache" in attn_cache:
                img_q = interp_fn(weight_img_q_norm, img_q, norm_seq(img_q, attn_cache["img_q_cache"]))
            if weight_img_k_norm != 0 and "img_k_cache" in attn_cache:
                img_k = interp_fn(weight_img_k_norm, img_k, norm_seq(img_k, attn_cache["img_k_cache"]))
            if weight_img_v_norm != 0 and "img_v_cache" in attn_cache:
                img_v = interp_fn(weight_img_v_norm, img_v, norm_seq(img_v, attn_cache["img_v_cache"]))
                
            for name in ["img_q_cache", "img_k_cache", "img_v_cache"]:
                if hasattr(self, name):
                    delattr(self, name)
                attn_cache.pop(name, None)


        inner_dim = img_k.shape[-1]
        head_dim  = inner_dim // self.heads

        img_q = img_q.view(bsz, -1, self.heads, head_dim)
        img_k = img_k.view(bsz, -1, self.heads, head_dim)
        img_v = img_v.view(bsz, -1, self.heads, head_dim)
        
        if img_masks is not None:
            img_k = img_k * img_masks.view(bsz, -1, 1, 1)


        if self.single:
            attn = attention(img_q, img_k, img_v, rope=rope, mask=mask)                 # TODO: scattersort here!
            
            if cache_v == True and self.EO("single_attn_adain"):
                self.single_attn_cache = img_attn.to("cpu", non_blocking=True).pin_memory()
            elif hasattr(self, "single_attn_cache") and self.EO("single_attn_adain"):
                attn_adain_weight = self.EO("single_attn_adain_weight", 1.0)
                img_attn = (1-attn_adain_weight) * img_attn + attn_adain_weight * adain_seq(img_attn, self.single_attn_cache.to("cuda", non_blocking=True))
                del self.single_attn_cache
                
            if cache_v == True and self.EO("single_attn_scattersort"):
                self.single_attn_cache = img_attn.to("cpu", non_blocking=True).pin_memory()
            elif hasattr(self, "single_attn_cache") and self.EO("single_attn_scattersort"):
                attn_adain_weight = self.EO("single_attn_scattersort_weight", 1.0)
                img_attn = (1-attn_adain_weight) * img_attn + attn_adain_weight * ScatterSort.apply(img_attn, self.single_attn_cache.to("cuda", non_blocking=True))
                del self.single_attn_cache
            
            return self.to_out(attn)
        else:
            
            txt_q   = self.q_rms_norm_t(self.to_q_t(txt))
            txt_k   = self.k_rms_norm_t(self.to_k_t(txt))
            txt_v   =                   self.to_v_t(txt)

            weight_txt_q = attninj_opts.get("txt_q", 0.0)
            weight_txt_k = attninj_opts.get("txt_k", 0.0)
            weight_txt_v = attninj_opts.get("txt_v", 0.0)
            
            weight_txt_q_norm = attninj_opts.get("txt_q_norm", 0.0)
            weight_txt_k_norm = attninj_opts.get("txt_k_norm", 0.0)
            weight_txt_v_norm = attninj_opts.get("txt_v_norm", 0.0)

            if cache_v:
                if weight_txt_q != 0 or weight_txt_q_norm != 0:
                    self.txt_q_cache = txt_q.to("cpu", non_blocking=True).pin_memory()
                if weight_txt_k != 0 or weight_txt_k_norm != 0:
                    self.txt_k_cache = txt_k.to("cpu", non_blocking=True).pin_memory()
                if weight_txt_v != 0 or weight_txt_v_norm != 0:
                    self.txt_v_cache = txt_v.to("cpu", non_blocking=True).pin_memory()
            else: 
                if self.EO("slerp"):
                    interp_fn = slerp_tensor
                else:
                    interp_fn = lambda weight, A, B: (1-weight) * A + weight * B

                if weight_txt_q != 0 and "txt_q_cache" in attn_cache:
                    txt_q = interp_fn(weight_txt_q, txt_q, attn_cache["txt_q_cache"])
                if weight_txt_k != 0 and "txt_k_cache" in attn_cache:
                    txt_k = interp_fn(weight_txt_k, txt_k, attn_cache["txt_k_cache"])
                if weight_txt_v != 0 and "txt_v_cache" in attn_cache:
                    txt_v = interp_fn(weight_txt_v, txt_v, attn_cache["txt_v_cache"])

                if weight_txt_q_norm != 0 and "txt_q_cache" in attn_cache:
                    txt_q = interp_fn(weight_txt_q_norm, txt_q, norm_seq(txt_q, attn_cache["txt_q_cache"]))
                if weight_txt_k_norm != 0 and "txt_k_cache" in attn_cache:
                    txt_k = interp_fn(weight_txt_k_norm, txt_k, norm_seq(txt_k, attn_cache["txt_k_cache"]))
                if weight_txt_v_norm != 0 and "txt_v_cache" in attn_cache:
                    txt_v = interp_fn(weight_txt_v_norm, txt_v, norm_seq(txt_v, attn_cache["txt_v_cache"]))
                    
                for name in ["txt_q_cache", "txt_k_cache", "txt_v_cache"]:
                    if hasattr(self, name):
                        delattr(self, name)
                    attn_cache.pop(name, None)

            txt_q   = txt_q.view(bsz, -1, self.heads, head_dim)
            txt_k   = txt_k.view(bsz, -1, self.heads, head_dim)
            txt_v   = txt_v.view(bsz, -1, self.heads, head_dim)
            
            img_len = img_q.shape[1]
            txt_len = txt_q.shape[1]
            
            attn    = attention(torch.cat([img_q, txt_q], dim=1), 
                                torch.cat([img_k, txt_k], dim=1), 
                                torch.cat([img_v, txt_v], dim=1), rope=rope, mask=mask)
            
            img_attn, txt_attn = torch.split(attn, [img_len, txt_len], dim=1)   #1, 4480, 2560
            
            if cache_v == True and self.EO("img_attn_adain"):
                self.img_attn_cache = img_attn.to("cpu", non_blocking=True).pin_memory()
            elif hasattr(self, "img_attn_cache") and self.EO("img_attn_adain"):
                attn_adain_weight = self.EO("img_attn_adain_weight", 1.0)
                img_attn = (1-attn_adain_weight) * img_attn + attn_adain_weight * adain_seq(img_attn, self.img_attn_cache.to("cuda", non_blocking=True))
                del self.img_attn_cache
                
            if cache_v == True and self.EO("img_attn_scattersort"):
                self.img_attn_cache = img_attn.to("cpu", non_blocking=True).pin_memory()
            elif hasattr(self, "img_attn_cache") and self.EO("img_attn_scattersort"):
                attn_adain_weight = self.EO("img_attn_scattersort_weight", 1.0)
                img_attn = (1-attn_adain_weight) * img_attn + attn_adain_weight * ScatterSort.apply(img_attn, self.img_attn_cache.to("cuda", non_blocking=True))
                del self.img_attn_cache
                
            if cache_v == True and self.EO("txt_attn_adain"):
                self.txt_attn_cache = txt_attn.to("cpu", non_blocking=True).pin_memory()
            elif hasattr(self, "txt_attn_cache") and self.EO("txt_attn_adain"):
                attn_adain_weight = self.EO("txt_attn_adain_weight", 1.0)
                txt_attn = (1-attn_adain_weight) * txt_attn + attn_adain_weight * adain_seq(txt_attn, self.txt_attn_cache.to("cuda", non_blocking=True))
                del self.txt_attn_cache
                
            if cache_v == True and self.EO("txt_attn_scattersort"):
                self.txt_attn_cache = txt_attn.to("cpu", non_blocking=True).pin_memory()
            elif hasattr(self, "txt_attn_cache") and self.EO("txt_attn_scattersort"):
                attn_adain_weight = self.EO("txt_attn_scattersort_weight", 1.0)
                txt_attn = (1-attn_adain_weight) * txt_attn + attn_adain_weight * ScatterSort.apply(txt_attn, self.txt_attn_cache.to("cuda", non_blocking=True))
                del self.txt_attn_cache
            
            #if anticond == "uncond":
            #    txt_src    = torch.cat([txt[:,1:3,:], txt[:,129:131,:], txt[:,257:259],], dim=-2).float()
            #    self.c_src = txt_src.transpose(-2,-1).squeeze(0)    # shape [C,1]

            if update_cross_attn is not None:
                if not update_cross_attn['skip_cross_attn']:
                    UNCOND      = update_cross_attn['UNCOND']
                    
                    if UNCOND:
                        llama_start = update_cross_attn['src_llama_start']
                        llama_end   = update_cross_attn['src_llama_end']
                        t5_start    = update_cross_attn['src_t5_start']
                        t5_end      = update_cross_attn['src_t5_end']
                    
                        txt_src    = torch.cat([txt[:,t5_start:t5_end,:], txt[:,128+llama_start:128+llama_end,:], txt[:,256+llama_start:256+llama_end],], dim=-2).float()
                        self.c_src = txt_src.transpose(-2,-1).squeeze(0)    # shape [C,1]
                    else:
                        llama_start = update_cross_attn['tgt_llama_start']
                        llama_end   = update_cross_attn['tgt_llama_end']
                        t5_start    = update_cross_attn['tgt_t5_start']
                        t5_end      = update_cross_attn['tgt_t5_end']
                        
                        lamb  = update_cross_attn['lamb']
                        erase = update_cross_attn['erase']
                        
                        txt_guide = torch.cat([txt[:,t5_start:t5_end,:], txt[:,128+llama_start:128+llama_end,:], txt[:,256+llama_start:256+llama_end],], dim=-2).float()
                        c_guide   = txt_guide.transpose(-2,-1).squeeze(0)  # [C,1]
                        
                        Wv_old       = self.to_v_t.weight.data.float()              # [C,C]
                        Wk_old       = self.to_k_t.weight.data.float()              # [C,C]

                        v_star       = Wv_old @ c_guide                             # [C,1]
                        k_star       = Wk_old @ c_guide                             # [C,1]

                        c_src        = self.c_src                                   # [C,1]

                        erase_scale  = erase
                        d            = c_src.shape[0]

                        C            = c_src @ c_src.T                              # [C,C]
                        I            = torch.eye(d, device=C.device, dtype=C.dtype)

                        mat1_v       = lamb*Wv_old + erase_scale*(v_star @ c_src.T)     # [C,C]
                        mat2_v       = lamb*I      + erase_scale*(C)                    # [C,C]
                        Wv_new       = mat1_v @ torch.inverse(mat2_v)                   # [C,C]

                        mat1_k       = lamb*Wk_old + erase_scale*(k_star @ c_src.T)     # [C,C]
                        mat2_k       = lamb*I      + erase_scale*(C)                    # [C,C]
                        Wk_new       = mat1_k @ torch.inverse(mat2_k)                   # [C,C]

                        self.to_v_t.weight.data.copy_(Wv_new.to(self.to_v_t.weight.data.dtype))
                        self.to_k_t.weight.data.copy_(Wk_new.to(self.to_k_t.weight.data.dtype))
                
            return self.to_out(img_attn), self.to_out_t(txt_attn)

    
    
    
#########################################################################################################################################################################
class HDBlockDouble(nn.Module):
    buffer = {}
    
    def __init__(
        self,
        dim                   : int,
        heads                 : int,
        head_dim              : int,
        num_routed_experts    : int = 4,
        num_activated_experts : int = 2,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 12*dim, bias=True,                                               dtype=dtype, device=device)
        )

        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)
        self.norm1_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)
        
        self.attn1   = HDAttention         (dim, heads, head_dim, single=False,                     dtype=dtype, device=device, operations=operations)

        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)
        self.ff_i    = HDMOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts,  dtype=dtype, device=device, operations=operations)
        
        self.norm3_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)                                 
        self.ff_t    =  HDFeedForwardSwiGLU(dim, 4*dim,                                             dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        img       :          FloatTensor,
        img_masks : Optional[FloatTensor] = None,
        txt       : Optional[FloatTensor] = None,
        clip      : Optional[FloatTensor] = None,    # clip = t + p_embedder (from pooled)
        rope      :          FloatTensor  = None,
        mask      : Optional[FloatTensor] = None,
        update_cross_attn : Optional[Dict]= None,
        cache_v   :                  bool = True,
        attninj_opts :     Optional[Dict] = {},
        CACHE_ATTN: bool = False,
        CACHE_TXT: bool = False
    ) -> FloatTensor:
        
        buf = HDBlockDouble.buffer
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate, \
        txt_msa_shift, txt_msa_scale, txt_msa_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(12, dim=-1)      # 1,1,2560           

        img_norm = self.norm1_i(img) * (1+img_msa_scale) + img_msa_shift
        txt_norm = self.norm1_t(txt) * (1+txt_msa_scale) + txt_msa_shift
        
        if hasattr(self, "img_norm"):
            img_norm = ScatterSort.apply(img_norm, self.img_norm.to("cuda", non_blocking=True))
            del self.img_norm
            txt_norm = ScatterSort.apply(txt_norm, self.txt_norm.to("cuda", non_blocking=True))
            del self.txt_norm
        elif cache_v:
            self.img_norm = img_norm.to("cpu", non_blocking=True).pin_memory()
            self.txt_norm = txt_norm.to("cpu", non_blocking=True).pin_memory()
        
        img_attn, txt_attn = self.attn1(img_norm, img_masks, txt_norm, rope=rope, mask=mask, update_cross_attn=update_cross_attn, cache_v=cache_v, attninj_opts=attninj_opts)
        
        #if hasattr(self, "img_attn"):
        #    img_attn = apply_scattersort(img_attn, self.img_attn.to("cuda", non_blocking=True))
        #    del self.img_attn
        #    txt_attn = apply_scattersort(txt_attn, self.txt_attn.to("cuda", non_blocking=True))
        #    del self.txt_attn
        #elif cache_v:
        #    self.img_attn = img_attn.to("cpu", non_blocking=True).pin_memory()
        #    self.txt_attn = txt_attn.to("cpu", non_blocking=True).pin_memory()
        
        img     += img_attn            *    img_msa_gate
        
        if hasattr(self, "img"):
            img = ScatterSort.apply(img, self.img.to("cuda", non_blocking=True))
            del self.img
        elif cache_v:
            self.img = img.to("cpu", non_blocking=True).pin_memory()
        
        img_norm = self.norm3_i(img) * (1+img_mlp_scale) + img_mlp_shift
        
        #if hasattr(self, "img_norm"):
        #    img_norm = apply_scattersort(img_norm, self.img_norm.to("cuda", non_blocking=True))
        #    del self.img_norm
        #elif cache_v:
        #    self.img_norm = img_norm.to("cpu", non_blocking=True).pin_memory()
        
        img     += self.ff_i(img_norm, cache_v) *    img_mlp_gate
        
        txt     += txt_attn            *    txt_msa_gate
        if hasattr(self, "txt"):
            txt = ScatterSort.apply(txt, self.txt.to("cuda", non_blocking=True))
            del self.txt
        elif cache_v:
            self.txt = txt.to("cpu", non_blocking=True).pin_memory()
        
        txt_norm = self.norm3_t(txt) * (1+txt_mlp_scale) + txt_mlp_shift
        
        #if hasattr(self, "txt_norm"):
        #    txt_norm = apply_scattersort(txt_norm, self.txt_norm.to("cuda", non_blocking=True))
        #    del self.txt_norm
        #elif cache_v:
        #    self.txt_norm = txt_norm.to("cpu", non_blocking=True).pin_memory()
        
        txt     += self.ff_t(txt_norm, cache_v) *    txt_mlp_gate 
        
        return img, txt


#########################################################################################################################################################################
class HDBlockSingle(nn.Module):
    buffer = {}
    
    def __init__(
        self,
        dim                   : int,
        heads                 : int,
        head_dim              : int,
        num_routed_experts    : int = 4,
        num_activated_experts : int = 2,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True,                                              dtype=dtype, device=device)
        )

        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)
        self.attn1   = HDAttention         (dim, heads, head_dim, single=True,                      dtype=dtype, device=device, operations=operations)

        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)
        self.ff_i    = HDMOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts,  dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        img        :          FloatTensor,
        img_masks  : Optional[FloatTensor]  = None,
        txt        : Optional[FloatTensor]  = None,
        clip       : Optional[FloatTensor]  = None,
        rope       :          FloatTensor   = None,
        mask       : Optional[FloatTensor]  = None,
        update_cross_attn : Optional[Dict] = None,
        cache_v   :                  bool  = True,
        attninj_opts :      Optional[Dict]  = None,
        CACHE_ATTN: bool = False,

    ) -> FloatTensor:
        
        buf = HDBlockSingle.buffer
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(6, dim=-1)

        img_norm = self.norm1_i(img) * (1+img_msa_scale) + img_msa_shift
        
        if hasattr(self, "img_norm"):
            img_norm = ScatterSort.apply(img_norm, self.img_norm.to("cuda", non_blocking=True))
            del self.img_norm
        elif cache_v:
            self.img_norm = img_norm.to("cpu", non_blocking=True).pin_memory()
        
        img_attn = self.attn1(img_norm, img_masks, rope=rope, mask=mask, cache_v=cache_v, attninj_opts=attninj_opts)
        
        #if hasattr(self, "img_attn"):
        #    img_attn = apply_scattersort(img_attn, self.img_attn.to("cuda", non_blocking=True))
        #    del self.img_attn
        #elif cache_v:
        #    self.img_attn = img_attn.to("cpu", non_blocking=True).pin_memory()
        
        img     += img_attn            *    img_msa_gate
        
        
        if hasattr(self, "img"):
            img = ScatterSort.apply(img, self.img.to("cuda", non_blocking=True))
            del self.img
        elif cache_v:
            self.img = img.to("cpu", non_blocking=True).pin_memory()
        
        
        
        img_norm = self.norm3_i(img) * (1+img_mlp_scale) + img_mlp_shift
        
        #if hasattr(self, "img_norm"):
        #    img_norm = apply_scattersort(img_norm, self.img_norm.to("cuda", non_blocking=True))
        #    del self.img_norm
        #elif cache_v:
        #    self.img_norm = img_norm.to("cpu", non_blocking=True).pin_memory()
        
        img     += self.ff_i(img_norm, cache_v) *    img_mlp_gate
        
        return img


#########################################################################################################################################################################
class HDModel(nn.Module):
    def __init__(
        self,
        patch_size            : Optional[int]   = None,
        in_channels           : int             = 64,
        out_channels          : Optional[int]   = None,
        num_layers            : int             = 16,
        num_single_layers     : int             = 32,
        attention_head_dim    : int             = 128,
        num_attention_heads   : int             = 20,
        caption_channels      : List[int]       = None,
        text_emb_dim          : int             = 2048,
        num_routed_experts    : int             = 4,
        num_activated_experts : int             = 2,
        axes_dims_rope        : Tuple[int, int] = ( 32,  32),
        max_resolution        : Tuple[int, int] = (128, 128),
        llama_layers          : List[int]       = None,
        image_model                             = None,     # unused, what was this supposed to be??
        dtype=None, device=None, operations=None
    ):
        self.patch_size          = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim  = attention_head_dim
        self.num_layers          = num_layers
        self.num_single_layers   = num_single_layers

        self.gradient_checkpointing = False

        super().__init__()
        self.dtype        = dtype
        self.out_channels = out_channels or in_channels
        self.inner_dim    = self.num_attention_heads * self.attention_head_dim
        self.llama_layers = llama_layers

        self.t_embedder   = TimestepEmbed(              self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.p_embedder   =   PooledEmbed(text_emb_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.x_embedder   =    PatchEmbed(
            patch_size   = patch_size,
            in_channels  = in_channels,
            out_channels = self.inner_dim,
            dtype=dtype, device=device, operations=operations
        )
        self.pe_embedder = EmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HDBlock(
                    dim                   = self.inner_dim,
                    heads                 = self.num_attention_heads,
                    head_dim              = self.attention_head_dim,
                    num_routed_experts    = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type            = BlockType.Double,
                    dtype=dtype, device=device, operations=operations
                )
                for i in range(self.num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HDBlock(
                    dim                   = self.inner_dim,
                    heads                 = self.num_attention_heads,
                    head_dim              = self.attention_head_dim,
                    num_routed_experts    = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type            = BlockType.Single,
                    dtype=dtype, device=device, operations=operations
                )
                for i in range(self.num_single_layers)
            ]
        )

        self.final_layer = LastLayer(self.inner_dim, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)

        caption_channels   = [caption_channels[1], ] * (num_layers + num_single_layers) + [caption_channels[0], ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim, dtype=dtype, device=device, operations=operations))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq            = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)
        self.manual_mask        = None

    def prepare_contexts(self, llama3, context, bsz, img_num_fea):
        contexts = llama3.movedim(1, 0)
        contexts = [contexts[k] for k in self.llama_layers]    # len == 48..... of tensors that are 1,143,4096

        if self.caption_projection is not None:
            contexts_list = []
            for i, cxt in enumerate(contexts):
                cxt = self.caption_projection[i](cxt)                          # linear in_features=4096, out_features=2560      len(self.caption_projection) == 49
                cxt = cxt.view(bsz, -1, img_num_fea)
                contexts_list.append(cxt)
            contexts = contexts_list
            context  = self.caption_projection[-1](context)
            context  = context.view(bsz, -1, img_num_fea)
            
            contexts.append(context)                      # len == 49...... of tensors that are 1,143,2560.   last chunk is T5

        return contexts

    ### FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ... FORWARD ###
    def forward(
        self,
        x       :          Tensor,
        t       :          Tensor,
        y       : Optional[Tensor]   = None,
        context : Optional[Tensor]   = None,
        encoder_hidden_states_llama3 = None,    # 1,32,143,4096
        control                      = None,
        transformer_options          = {},
        mask    : Optional[Tensor]   = None,
    ) -> Tensor:
        x_orig       = x.clone()
        b, c, h, w  = x.shape
        h_len = ((h + (self.patch_size // 2)) // self.patch_size) # h_len 96
        w_len = ((w + (self.patch_size // 2)) // self.patch_size) # w_len 96
        img          = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        update_cross_attn = transformer_options.get("update_cross_attn")
        SIGMA = t[0].clone() / 1000
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        img_pre_final = []
        if EO is not None:
            EO.mute = True
            
        for block in self.double_stream_blocks:
            block.block.attn1.EO = EO
        for block in self.single_stream_blocks:
            block.block.attn1.EO = EO
        self.style_dtype = torch.float32 if self.style_dtype is None else self.style_dtype
        ADAIN_SINGLE_BLOCKS,   ADAIN_DOUBLE_BLOCKS   = [-1], [-1]
        ATTNINJ_SINGLE_BLOCKS, ATTNINJ_DOUBLE_BLOCKS = [-1], [-1]
        
        y0_adain,     img_y0_adain,     img_sizes_y0_adain     = None, None, None
        y0_attninj,   img_y0_attninj,   img_sizes_y0_attninj   = None, None, None
        y0_style_pos, img_y0_style_pos, img_sizes_y0_style_pos = None, None, None
        y0_style_neg, img_y0_style_neg, img_sizes_y0_style_neg = None, None, None
        blocks_attninj_qkv_scaled = {}
        STYLE_UNCOND = EO("STYLE_UNCOND", False)
        
        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

        y0_adain = transformer_options.get("y0_adain")
        if EO("eps_adain") and y0_adain is not None:
            x_init              = transformer_options.get("x_init").to(x)   # initial noise and/or image+noise from start of rk_sampler_beta() 
            y0_adain            = y0_adain.to(x)
            y0_adain            = (1-SIGMA) * y0_adain + SIGMA * x_init
            img_y0_adain        = comfy.ldm.common_dit.pad_to_patch_size(y0_adain, (self.patch_size, self.patch_size))
            t_y0_adain          = t[0].unsqueeze(0).clone() #torch.full_like(t, 0.0)[0].unsqueeze(0)
            blocks_adain        = transformer_options.get("blocks_adain")
            ADAIN_SINGLE_BLOCKS = blocks_adain['single_blocks']
            ADAIN_DOUBLE_BLOCKS = blocks_adain['double_blocks']
        elif y0_adain is not None:
            y0_adain            = y0_adain.to(x)
            img_y0_adain        = comfy.ldm.common_dit.pad_to_patch_size(y0_adain, (self.patch_size, self.patch_size))
            t_y0_adain          = torch.full_like(t, 0.0)[0].unsqueeze(0)
            blocks_adain        = transformer_options.get("blocks_adain")
            ADAIN_SINGLE_BLOCKS = blocks_adain['single_blocks']
            ADAIN_DOUBLE_BLOCKS = blocks_adain['double_blocks']
        
        y0_attninj = transformer_options.get("y0_attninj")
        if y0_attninj is not None:
            x_init                = transformer_options.get("x_init").to(x)   # initial noise and/or image+noise from start of rk_sampler_beta() 
            y0_attninj            = y0_attninj.to(x)
            y0_attninj            = (1-SIGMA) * y0_attninj + SIGMA * x_init
            img_y0_attninj        = comfy.ldm.common_dit.pad_to_patch_size(y0_attninj, (self.patch_size, self.patch_size))
            t_y0_attninj          = t[0].unsqueeze(0).clone() #torch.full_like(t, 0.0)[0].unsqueeze(0)
            blocks_attninj        = transformer_options.get("blocks_attninj")
            blocks_attninj_qkv    = transformer_options.get("blocks_attninj_qkv")
            ATTNINJ_SINGLE_BLOCKS = blocks_attninj['single_blocks']
            ATTNINJ_DOUBLE_BLOCKS = blocks_attninj['double_blocks']
        
        if y0_adain is not None and y0_attninj is not None and torch.norm(y0_adain - y0_attninj) == 0.0:
            IDENTICAL_ADAIN_ATTNINJ = True
        else:
            IDENTICAL_ADAIN_ATTNINJ = False

            
            
            
        img_orig, t_orig, y_orig, context_orig, llama3_orig = clone_inputs(img, t, y, context, encoder_hidden_states_llama3)
        if y0_adain is not None:
            img_y0_adain_orig, t_y0_adain_orig = clone_inputs(img_y0_adain, t_y0_adain)
        if y0_attninj is not None:
            img_y0_attninj_orig, t_y0_attninj_orig = clone_inputs(img_y0_attninj, t_y0_attninj)
        
        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        
        freqsep_lowpass_method = transformer_options.get("freqsep_lowpass_method")
        freqsep_sigma          = transformer_options.get("freqsep_sigma")
        freqsep_kernel_size    = transformer_options.get("freqsep_kernel_size")
        freqsep_inner_kernel_size    = transformer_options.get("freqsep_inner_kernel_size")
        freqsep_stride    = transformer_options.get("freqsep_stride")
        
        freqsep_lowpass_weight = transformer_options.get("freqsep_lowpass_weight")
        freqsep_highpass_weight= transformer_options.get("freqsep_highpass_weight")
        freqsep_mask           = transformer_options.get("freqsep_mask")
        
        
        #floor     = min(floor, weight)
        mask_zero = None

        out_list = []
        for cond_iter in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
            
            if update_cross_attn is not None:
                update_cross_attn['UNCOND'] = UNCOND
            
            bsz = 1

            img, t, y, context, llama3 = clone_inputs(img_orig, t_orig, y_orig, context_orig, llama3_orig, index=cond_iter)
            if y0_adain is not None:
                img_y0_adain, t_y0_adain = clone_inputs(img_y0_adain_orig, t_y0_adain_orig)
                img_sizes_y0_adain = None
            if y0_attninj is not None:
                img_y0_attninj, t_y0_attninj = clone_inputs(img_y0_attninj_orig, t_y0_attninj_orig)
                img_sizes_y0_attninj = None
            
            mask = None
            if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
                AttnMask = transformer_options['AttnMask']
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                if mask_zero is None:
                    mask_zero = torch.ones_like(mask)
                    img_len = transformer_options['AttnMask'].img_len
                    mask_zero[img_len:, img_len:] = mask[img_len:, img_len:]

                if weight == 0:
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    context = context.view(128, -1, context.shape[-1]).sum(dim=-2)                                    # 128 !!!
                    llama3  = transformer_options['RegContext'].llama3 .to(llama3 .dtype).to(llama3 .device)
                    mask = None
                else:
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    llama3  = transformer_options['RegContext'].llama3 .to(llama3 .dtype).to(llama3 .device)
                


            if UNCOND and 'AttnMask_neg' in transformer_options: # and weight != 0:
                AttnMask = transformer_options['AttnMask_neg']
                mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')
                if mask_zero is None:
                    mask_zero = torch.ones_like(mask)
                    img_len = transformer_options['AttnMask_neg'].img_len
                    mask_zero[img_len:, img_len:] = mask[img_len:, img_len:]

                if weight == 0:
                    context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                    context = context.view(128, -1, context.shape[-1]).sum(dim=-2)                                    # 128 !!!
                    llama3  = transformer_options['RegContext_neg'].llama3 .to(llama3 .dtype).to(llama3 .device)
                    mask = None

                else:
                    context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                    llama3  = transformer_options['RegContext_neg'].llama3 .to(llama3 .dtype).to(llama3 .device)

            elif UNCOND and 'AttnMask' in transformer_options:
                AttnMask = transformer_options['AttnMask']
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                
                if mask_zero is None:
                    mask_zero = torch.ones_like(mask)
                    img_len = transformer_options['AttnMask'].img_len
                    mask_zero[img_len:, img_len:] = mask[img_len:, img_len:]
                if weight == 0:                                                                             # ADDED 5/23/2025
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)  # ADDED 5/26/2025 14:53
                    context = context.view(128, -1, context.shape[-1]).sum(dim=-2)                                    # 128 !!!
                    llama3  = transformer_options['RegContext'].llama3 .to(llama3 .dtype).to(llama3 .device)
                    mask = None
                else:
                    A       = context
                    B       = transformer_options['RegContext'].context
                    context = A.repeat(1,    (B.shape[1] // A.shape[1]) + 1, 1)[:,   :B.shape[1], :]

                    A       = llama3
                    B       = transformer_options['RegContext'].llama3
                    llama3  = A.repeat(1, 1, (B.shape[2] // A.shape[2]) + 1, 1)[:,:, :B.shape[2], :]

            if self.manual_mask is not None:
                mask = self.manual_mask

            if mask is not None and not type(mask[0][0].item()) == bool:
                mask = mask.to(img.dtype)
            if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
                mask_zero = mask_zero.to(img.dtype)

            # prep embeds
            t    = self.expand_timesteps(t, bsz, img.device)
            t    = self.t_embedder      (t,      img.dtype)
            clip = t + self.p_embedder(y)
            
            if y0_adain is not None and img_sizes_y0_adain is None:
                #t_y0_adain = t_orig[0].unsqueeze(0).clone() # torch.full_like(t_orig, 10.0)[0].unsqueeze(0)
                
                t_y0_adain    = self.expand_timesteps(t_y0_adain, bsz, img.device)
                t_y0_adain    = self.t_embedder      (t_y0_adain,      img.dtype)
                y = y_orig.clone()[0].unsqueeze(0)
                clip_y0_adain = t_y0_adain + self.p_embedder(y)   

            if y0_attninj is not None and img_sizes_y0_attninj is None:
                #t_y0_attninj = t_orig[0].unsqueeze(0).clone() # torch.full_like(t_orig, 10.0)[0].unsqueeze(0)
                
                t_y0_attninj    = self.expand_timesteps(t_y0_attninj, bsz, img.device)
                t_y0_attninj    = self.t_embedder      (t_y0_attninj,      img.dtype)
                y = y_orig.clone()[0].unsqueeze(0)
                clip_y0_attninj = t_y0_attninj + self.p_embedder(y)   

            if EO("adain_swap_clip"):
                clip_y0_adain = clip.clone()
                clip_y0_attninj = clip.clone()

            img_sizes = None
            #img_prepatchify = img.clone()
            img, img_masks, img_sizes = self.patchify(img, self.max_seq, img_sizes)   # for 1024x1024: output is   1,4096,64   None   [[64,64]]     hidden_states rearranged not shrunk, patch_size 1x1???
            if img_masks is None:
                pH, pW          = img_sizes[0]
                img_ids         = torch.zeros(pH, pW, 3, device=img.device)
                img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=img.device)[:, None]
                img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=img.device)[None, :]
                img_ids         = repeat(img_ids, "h w c -> b (h w) c", b=bsz)
            img = self.x_embedder(img)
                


            if y0_adain is not None and img_sizes_y0_adain is None: #and not UNCOND 
                img_sizes_y0_adain = None
                img_y0_adain, img_masks_y0_adain, img_sizes_y0_adain = self.patchify(img_y0_adain, self.max_seq, img_sizes_y0_adain)   # for 1024x1024: output is   1,4096,64   None   [[64,64]]     hidden_states rearranged not shrunk, patch_size 1x1???
                if img_masks_y0_adain is None:
                    pH, pW          = img_sizes_y0_adain[0]
                    img_ids_y0_adain         = torch.zeros(pH, pW, 3, device=img_y0_adain.device)
                    img_ids_y0_adain[..., 1] = img_ids_y0_adain[..., 1] + torch.arange(pH, device=img_y0_adain.device)[:, None]
                    img_ids_y0_adain[..., 2] = img_ids_y0_adain[..., 2] + torch.arange(pW, device=img_y0_adain.device)[None, :]
                    img_ids_y0_adain         = repeat(img_ids_y0_adain, "h w c -> b (h w) c", b=bsz)
                img_y0_adain = self.x_embedder(img_y0_adain)  # hidden_states 1,4032,2560         for 1024x1024: -> 1,4096,2560      ,64 -> ,2560 (x40)
            
            if y0_attninj is not None and img_sizes_y0_attninj is None: #and not UNCOND 
                img_sizes_y0_attninj = None
                img_y0_attninj, img_masks_y0_attninj, img_sizes_y0_attninj = self.patchify(img_y0_attninj, self.max_seq, img_sizes_y0_attninj)   # for 1024x1024: output is   1,4096,64   None   [[64,64]]     hidden_states rearranged not shrunk, patch_size 1x1???
                if img_masks_y0_attninj is None:
                    pH, pW          = img_sizes_y0_attninj[0]
                    img_ids_y0_attninj         = torch.zeros(pH, pW, 3, device=img_y0_attninj.device)
                    img_ids_y0_attninj[..., 1] = img_ids_y0_attninj[..., 1] + torch.arange(pH, device=img_y0_attninj.device)[:, None]
                    img_ids_y0_attninj[..., 2] = img_ids_y0_attninj[..., 2] + torch.arange(pW, device=img_y0_attninj.device)[None, :]
                    img_ids_y0_attninj         = repeat(img_ids_y0_attninj, "h w c -> b (h w) c", b=bsz)
                img_y0_attninj = self.x_embedder(img_y0_attninj)  # hidden_states 1,4032,2560         for 1024x1024: -> 1,4096,2560      ,64 -> ,2560 (x40)
            
            
            #contexts_orig = self.prepare_contexts(llama3_orig, context_orig, bsz, img.shape[-1])
            contexts = self.prepare_contexts(llama3, context, bsz, img.shape[-1])

            # txt_ids -> 1,414,3
            txt_ids = torch.zeros(bsz,   contexts[-1].shape[1] + contexts[-2].shape[1] + contexts[0].shape[1],     3,    device=img_ids.device, dtype=img_ids.dtype)
            ids     = torch.cat((img_ids, txt_ids), dim=1)   # ids -> 1,4446,3
            rope    = self.pe_embedder(ids)                  # rope -> 1, 4446, 1, 64, 2, 2

            # 2. Blocks
            #txt_init_orig     = torch.cat([contexts_orig[-1], contexts_orig[-2]], dim=1)     # shape[1] == 128, 143       then on another step/call it's 128, 128...??? cuz the contexts is now 1,128,2560
            #txt_init_len_orig = txt_init_orig.shape[1]                                       # 271
            
            txt_init     = torch.cat([contexts[-1], contexts[-2]], dim=1)     # shape[1] == 128, 143       then on another step/call it's 128, 128...??? cuz the contexts is now 1,128,2560
            txt_init_len = txt_init.shape[1]                                       # 271

            if mask is not None and self.manual_mask is None:
                #txt_offset = transformer_options['AttnMask'].text_len // 3 // transformer_options['AttnMask'].num_regions
                txt_init_list = []
                
                offset_t5_start    = 0
                for i in range(transformer_options['AttnMask'].num_regions):
                    offset_t5_end   = offset_t5_start + transformer_options['AttnMask'].context_lens_list[i][0]
                    txt_init_list.append(contexts[-1][:,offset_t5_start:offset_t5_end,:])
                    offset_t5_start = offset_t5_end
                
                offset_llama_start = 0
                for i in range(transformer_options['AttnMask'].num_regions):
                    offset_llama_end   = offset_llama_start + transformer_options['AttnMask'].context_lens_list[i][1]
                    txt_init_list.append(contexts[-2][:,offset_llama_start:offset_llama_end,:])
                    offset_llama_start = offset_llama_end
                
                txt_init = torch.cat(txt_init_list, dim=1)  #T5,LLAMA3 (last block)
                txt_init_len = txt_init.shape[1]     

            img_len = img.shape[1]
            
            #if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
            if img_y0_adain is not None:
                txt_init_y0_adain = txt_init.clone()
            #if STYLE_UNCOND == UNCOND and img_y0_attninj is not None:
            if img_y0_attninj is not None:
                txt_init_y0_attninj = txt_init.clone()
            
            for bid, block in enumerate(self.double_stream_blocks):                                                              # len == 16
                txt_llama = contexts[bid]
                txt       = torch.cat([txt_init, txt_llama], dim=1)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

                #txt_llama_orig = contexts_orig[bid]
                #txt_orig       = torch.cat([txt_init_orig, txt_llama_orig], dim=1)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

                
                #if (bid == 0 or EO("eps_adain_first")) and img_y0_adain is not None:
                if EO("eps_adain_first") and img_y0_adain is not None:
                    if IDENTICAL_ADAIN_ATTNINJ:
                        img_y0_adain      = img_y0_attninj
                        txt_init_y0_adain = txt_init_y0_attninj
                    else:
                        txt_y0_adain                    = torch.cat([txt_init_y0_adain, txt_llama], dim=1)
                        #img_y0_adain, txt_init_y0_adain = block(img_y0_adain, img_masks, txt_y0_adain, clip_y0_adain, rope, mask)
                        #txt_init_y0_adain               = txt_init_y0_adain[:, :txt_init_len]
                    
                    if bid in ADAIN_DOUBLE_BLOCKS:
                        adaweight = blocks_adain['double_weights'][bid]

                        img_y0_adain_txt = txt_init_y0_adain[:, :txt_init_len]
                        img_y0_adain_img = img_y0_adain[:,:img_len]
                        
                        img_txt = txt_init[:, :txt_init_len]
                        img_img = img[:,:img_len]
                        
                        if EO("eps_adain_scattersort"):
                            if not EO("eps_adain_skip_txt"):
                                txt_init = (1-adaweight) * img_txt + adaweight * ScatterSort.apply(img_txt, img_y0_adain_txt)
                            img      = (1-adaweight) * img_img + adaweight * ScatterSort.apply(img_img, img_y0_adain_img)
                        elif EO("eps_adain_WCT"):
                            if not EO("eps_adain_skip_txt"):
                                self.StyleWCT.set(img_y0_adain_txt)
                                txt_init = (1-adaweight) * img_txt + adaweight * self.StyleWCT.get(img_txt)
                            
                            self.StyleWCT.set(img_y0_adain_img)
                            img = (1-adaweight) * img_img + adaweight * self.StyleWCT.get(img_img)
                        else:
                            if not EO("eps_adain_skip_txt"):
                                txt_init = (1-adaweight) * img_txt + adaweight * adain_seq(img_txt, img_y0_adain_txt)
                            img      = (1-adaweight) * img_img + adaweight * adain_seq(img_img, img_y0_adain_img)
                            
                    img_y0_adain, txt_init_y0_adain = block(img_y0_adain, img_masks, txt_y0_adain, clip_y0_adain, rope, mask, CACHE_ATTN=EO("eps_adain_cache_attn"))
                    txt_init_y0_adain               = txt_init_y0_adain[:, :txt_init_len]
                    
                        #img = img + (1-SIGMA) * adaweight * (adain_seq(img, img_y0_adain) - img)

                if   weight > 0 and mask is not None and     weight  <      bid/48:
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask_zero)
                    
                elif (weight < 0 and mask is not None and abs(weight) < (1 - bid/48)):
                    img_tmpZ, txt_tmpZ = img.clone(), txt.clone()

                    # more efficient than the commented lines below being used instead in the loop?
                    img_tmpZ, txt_init = block(img_tmpZ, img_masks, txt_tmpZ, clip, rope, mask)
                    img     , txt_tmpZ = block(img     , img_masks, txt     , clip, rope, mask_zero)
                    
                elif floor > 0 and mask is not None and     floor  >      bid/48:
                    mask_tmp = mask.clone()
                    mask_tmp[:img_len,:img_len] = 1.0
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask_tmp)
                    
                elif floor < 0 and mask is not None and abs(floor) > (1 - bid/48):
                    mask_tmp = mask.clone()
                    mask_tmp[:img_len,:img_len] = 1.0
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask_tmp)
                    
                elif update_cross_attn is not None and update_cross_attn['skip_cross_attn']:
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask, update_cross_attn=update_cross_attn)
                    
                else:
                    #if STYLE_UNCOND == UNCOND and img_y0_attninj is not None:
                    if img_y0_attninj is not None:
                        cache_v = False
                        if bid in ATTNINJ_DOUBLE_BLOCKS:
                            cache_v = True
                        
                        scale = blocks_attninj['double_weights'][bid]
                        blocks_attninj_qkv_scaled = {k: v * scale for k, v in blocks_attninj_qkv.items()}
                        txt_y0_attninj                      = torch.cat([txt_init_y0_attninj, txt_llama], dim=1)
                        img_y0_attninj, txt_init_y0_attninj = block(img_y0_attninj, img_masks, txt_y0_attninj, clip_y0_attninj, rope, mask, cache_v=cache_v, attninj_opts=blocks_attninj_qkv_scaled)
                        txt_init_y0_attninj                 = txt_init_y0_attninj[:, :txt_init_len]
                    
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask, update_cross_attn=update_cross_attn, attninj_opts=blocks_attninj_qkv_scaled)

                    #if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
                    if False: #EO("eps_adain_first") and img_y0_adain is not None:
                        img_y0_adain, txt_init_y0_adain = block(img_y0_adain, img_masks, txt_y0_adain, clip_y0_adain, rope, mask)
                        txt_init_y0_adain               = txt_init_y0_adain[:, :txt_init_len]
                    if not EO("eps_adain_first") and img_y0_adain is not None:
                        if IDENTICAL_ADAIN_ATTNINJ:
                            img_y0_adain      = img_y0_attninj
                            txt_init_y0_adain = txt_init_y0_attninj
                        else:
                            txt_y0_adain                    = torch.cat([txt_init_y0_adain, txt_llama], dim=1)
                            img_y0_adain, txt_init_y0_adain = block(img_y0_adain, img_masks, txt_y0_adain, clip_y0_adain, rope, mask, CACHE_ATTN=EO("eps_adain_cache_attn"))
                            txt_init_y0_adain               = txt_init_y0_adain[:, :txt_init_len]
                        
                        if bid in ADAIN_DOUBLE_BLOCKS:
                            adaweight = blocks_adain['double_weights'][bid]

                            img_y0_adain_txt = txt_init_y0_adain[:, :txt_init_len]
                            img_y0_adain_img = img_y0_adain[:,:img_len]
                            
                            img_txt = txt_init[:, :txt_init_len]
                            img_img = img[:,:img_len]
                            
                            if EO("eps_adain_scattersort"):
                                if not EO("eps_adain_skip_txt"):
                                    txt_init = (1-adaweight) * img_txt + adaweight * ScatterSort.apply(img_txt, img_y0_adain_txt)
                                img      = (1-adaweight) * img_img + adaweight * ScatterSort.apply(img_img, img_y0_adain_img)
                            elif EO("eps_adain_WCT"):
                                if not EO("eps_adain_skip_txt"):
                                    self.StyleWCT.set(img_y0_adain_txt)
                                    txt_init = (1-adaweight) * img_txt + adaweight * self.StyleWCT.get(img_txt)
                                
                                self.StyleWCT.set(img_y0_adain_img)
                                img = (1-adaweight) * img_img + adaweight * self.StyleWCT.get(img_img)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    txt_init = (1-adaweight) * img_txt + adaweight * adain_seq(img_txt, img_y0_adain_txt)
                                img      = (1-adaweight) * img_img + adaweight * adain_seq(img_img, img_y0_adain_img)
                            
                            #img = img + (1-SIGMA) * adaweight * (adain_seq(img, img_y0_adain) - img)

                txt_init = txt_init[:, :txt_init_len]

            img_len = img.shape[1]
            img     = torch.cat([img, txt_init], dim=1)   # 4032 + 271 -> 4303     # txt embed from double stream block
            
            #if STYLE_UNCOND == UNCOND and y0_adain is not None:
            if y0_adain is not None:
                img_y0_adain = torch.cat([img_y0_adain, txt_init_y0_adain], dim=1)
            #if STYLE_UNCOND == UNCOND and y0_attninj is not None:
            if y0_attninj is not None:
                img_y0_attninj = torch.cat([img_y0_attninj, txt_init_y0_attninj], dim=1)
            
            joint_len = img.shape[1]
            
            if img_masks is not None:
                img_masks_ones = torch.ones( (bsz, txt_init.shape[1] + txt_llama.shape[1]), device=img_masks.device, dtype=img_masks.dtype)   # encoder_attention_mask_ones=   padding for txt embed concatted onto end of img
                img_masks      = torch.cat([img_masks, img_masks_ones], dim=1)
            
            # SINGLE STREAM
            for bid, block in enumerate(self.single_stream_blocks): # len == 32
                txt_llama = contexts[bid+16]                        # T5 pre-embedded for single stream blocks
                img = torch.cat([img, txt_llama], dim=1)            # cat img,txt     opposite of flux which is txt,img       4303 + 143 -> 4446
                
                if False: #eight != 0:
                    mask = AttnMask.gen_edge_mask(bid+16)
                    
                #if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
                if EO("eps_adain_first") and img_y0_adain is not None:
                    if IDENTICAL_ADAIN_ATTNINJ:
                        img_y0_adain = img_y0_attninj
                    else:
                        img_y0_adain = torch.cat([img_y0_adain, txt_llama], dim=1)   
                        #img_y0_adain = block(img_y0_adain, img_masks, None, clip_y0_adain, rope, mask)
                        #img_y0_adain = img_y0_adain[:, :joint_len]
                    
                    if bid in ADAIN_SINGLE_BLOCKS:
                        adaweight = blocks_adain['single_weights'][bid]
                        
                        img_y0_adain_txt = img_y0_adain[:,img_len:]
                        img_y0_adain_img = img_y0_adain[:,:img_len]
                        
                        img_txt = img[:,img_len:]
                        img_img = img[:,:img_len]
                        
                        if EO("eps_adain_scattersort"):
                            if EO("eps_adain_joint"):
                                img = (1-adaweight) * img + adaweight * ScatterSort.apply(img, img_y0_adain)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    img_txt = (1-adaweight) * img_txt + adaweight * ScatterSort.apply(img_txt, img_y0_adain_txt)
                                img_img = (1-adaweight) * img_img + adaweight * ScatterSort.apply(img_img, img_y0_adain_img)
                        elif EO("eps_adain_WCT"):
                            if EO("eps_adain_joint"):
                                self.StyleWCT.set(img_y0_adain)
                                img = (1-adaweight) * img + adaweight * self.StyleWCT.get(img)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    self.StyleWCT.set(img_y0_adain_txt)
                                    img_txt = (1-adaweight) * img_txt + adaweight * self.StyleWCT.get(img_txt)
                                
                                self.StyleWCT.set(img_y0_adain_img)
                                img_img = (1-adaweight) * img_img + adaweight * self.StyleWCT.get(img_img)
                        else:
                            if EO("eps_adain_joint"):
                                img = (1-adaweight) * img + adaweight * adain_seq(img, img_y0_adain)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    img_txt = (1-adaweight) * img_txt + adaweight * adain_seq(img_txt, img_y0_adain_txt)
                                img_img = (1-adaweight) * img_img + adaweight * adain_seq(img_img, img_y0_adain_img)
                        
                        if not EO("eps_adain_joint"):
                            img = torch.cat([img_img, img_txt], dim=-2)
                            
                    img_y0_adain = block(img_y0_adain, img_masks, None, clip_y0_adain, rope, mask, CACHE_ATTN=EO("eps_adain_cache_attn"))
                    img_y0_adain = img_y0_adain[:, :joint_len]

                        
                if   weight > 0 and mask is not None and     weight  <      (bid+16)/48:
                    img = block(img, img_masks, None, clip, rope, mask_zero)
                    
                elif weight < 0 and mask is not None and abs(weight) < (1 - (bid+16)/48):
                    img = block(img, img_masks, None, clip, rope, mask_zero)
                    
                elif floor > 0 and mask is not None and     floor  >      (bid+16)/48:
                    mask_tmp = mask.clone()
                    mask_tmp[:img_len,:img_len] = 1.0
                    img = block(img, img_masks, None, clip, rope, mask_tmp)
                    
                elif floor < 0 and mask is not None and abs(floor) > (1 - (bid+16)/48):
                    mask_tmp = mask.clone()
                    mask_tmp[:img_len,:img_len] = 1.0
                    img = block(img, img_masks, None, clip, rope, mask_tmp)
                    
                else:
                    #if STYLE_UNCOND == UNCOND and img_y0_attninj is not None:
                    if img_y0_attninj is not None:
                        cache_v = False
                        if bid in ATTNINJ_SINGLE_BLOCKS:
                            cache_v = True
                        
                        scale = blocks_attninj['single_weights'][bid]
                        blocks_attninj_qkv_scaled = {k: v * scale for k, v in blocks_attninj_qkv.items()}
                        img_y0_attninj = torch.cat([img_y0_attninj, txt_llama], dim=1)   
                        img_y0_attninj = block(img_y0_attninj, img_masks, None, clip_y0_attninj, rope, mask, cache_v=cache_v, attninj_opts=blocks_attninj_qkv_scaled)
                        img_y0_attninj = img_y0_attninj[:, :joint_len]

                    img = block(img, img_masks, None, clip, rope, mask, attninj_opts=blocks_attninj_qkv_scaled)

                img = img[:, :joint_len]   # slice off txt_llama
                
                #if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
                if False: #EO("eps_adain_first") and img_y0_adain is not None:
                    img_y0_adain = block(img_y0_adain, img_masks, None, clip_y0_adain, rope, mask)
                    img_y0_adain = img_y0_adain[:, :joint_len]
                if not EO("eps_adain_first") and img_y0_adain is not None:
                    if IDENTICAL_ADAIN_ATTNINJ:
                        img_y0_adain = img_y0_attninj
                    else:
                        img_y0_adain = torch.cat([img_y0_adain, txt_llama], dim=1)   
                        img_y0_adain = block(img_y0_adain, img_masks, None, clip_y0_adain, rope, mask, CACHE_ATTN=EO("eps_adain_cache_attn"))
                        img_y0_adain = img_y0_adain[:, :joint_len]
                    
                    if bid in ADAIN_SINGLE_BLOCKS:
                        adaweight = blocks_adain['single_weights'][bid]
                        
                        img_y0_adain_txt = img_y0_adain[:,img_len:]
                        img_y0_adain_img = img_y0_adain[:,:img_len]
                        
                        img_txt = img[:,img_len:]
                        img_img = img[:,:img_len]
                        
                        if EO("eps_adain_scattersort"):
                            if EO("eps_adain_joint"):
                                img = (1-adaweight) * img + adaweight * ScatterSort.apply(img, img_y0_adain)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    img_txt = (1-adaweight) * img_txt + adaweight * ScatterSort.apply(img_txt, img_y0_adain_txt)
                                img_img = (1-adaweight) * img_img + adaweight * ScatterSort.apply(img_img, img_y0_adain_img)
                        elif EO("eps_adain_WCT"):
                            if EO("eps_adain_joint"):
                                self.StyleWCT.set(img_y0_adain)
                                img = (1-adaweight) * img + adaweight * self.StyleWCT.get(img)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    self.StyleWCT.set(img_y0_adain_txt)
                                    img_txt = (1-adaweight) * img_txt + adaweight * self.StyleWCT.get(img_txt)
                                
                                self.StyleWCT.set(img_y0_adain_img)
                                img_img = (1-adaweight) * img_img + adaweight * self.StyleWCT.get(img_img)
                        else:
                            if EO("eps_adain_joint"):
                                img = (1-adaweight) * img + adaweight * adain_seq(img, img_y0_adain)
                            else:
                                if not EO("eps_adain_skip_txt"):
                                    img_txt = (1-adaweight) * img_txt + adaweight * adain_seq(img_txt, img_y0_adain_txt)
                                img_img = (1-adaweight) * img_img + adaweight * adain_seq(img_img, img_y0_adain_img)
                        
                        if not EO("eps_adain_joint"):
                            img = torch.cat([img_img, img_txt], dim=-2)

                        #img = img + (1-SIGMA) * adaweight * (adain_seq(img, img_y0_adain) - img)
            
            img = img[:, :img_len, ...]
            
            img_pre_final.append(img)
            
            img = self.final_layer(img, clip)   # 4096,2560 -> 4096,64
            img = self.unpatchify (img, img_sizes)
            
            out_list.append(img)
            
            
        output = torch.stack(out_list, dim=0).squeeze(dim=1)

        
        eps = -output[:, :, :h, :w]
        
        
        
        
        
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        
        if y0_style_pos is not None:
            y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            y0_style_pos_mask = transformer_options.get("y0_style_pos_mask")
            y0_style_pos_mask_edge = transformer_options.get("y0_style_pos_mask_edge")

            y0_style_pos = y0_style_pos.to(dtype)
            x   = x_orig.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps
            
            denoised_embed = self.Retrojector.embed(denoised)
            y0_adain_embed = self.Retrojector.embed(y0_style_pos)
            
            if transformer_options['y0_style_method'] == "scattersort":
                tile_h, tile_w = transformer_options.get('y0_style_tile_height'), transformer_options.get('y0_style_tile_width')
                pad = transformer_options.get('y0_style_tile_padding')
                if pad is not None and tile_h is not None and tile_w is not None:
                    
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if EO("scattersort_median_LP"):
                        denoised_spatial_LP = median_blur_2d(denoised_spatial, kernel_size=EO("scattersort_median_LP",7))
                        y0_adain_spatial_LP = median_blur_2d(y0_adain_spatial, kernel_size=EO("scattersort_median_LP",7))
                        
                        denoised_spatial_HP = denoised_spatial - denoised_spatial_LP
                        y0_adain_spatial_HP = y0_adain_spatial - y0_adain_spatial_LP
                        
                        denoised_spatial_LP = apply_scattersort_tiled(denoised_spatial_LP, y0_adain_spatial_LP, tile_h, tile_w, pad)
                        
                        denoised_spatial = denoised_spatial_LP + denoised_spatial_HP
                        denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")
                    else:
                        denoised_spatial = apply_scattersort_tiled(denoised_spatial, y0_adain_spatial, tile_h, tile_w, pad)
                    
                    denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")
                    
                else:
                    denoised_embed = apply_scattersort_masked(denoised_embed, y0_adain_embed, y0_style_pos_mask, y0_style_pos_mask_edge, h_len, w_len)



            elif transformer_options['y0_style_method'] == "AdaIN":
                if freqsep_mask is not None:
                    freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                    freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                
                if hasattr(self, "adain_tile"):
                    tile_h, tile_w = self.adain_tile
                    
                    denoised_pretile = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_pretile = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if self.adain_flag:
                        h_off = tile_h // 2
                        w_off = tile_w // 2
                        denoised_pretile = denoised_pretile[:,:,h_off:-h_off, w_off:-w_off]
                        self.adain_flag = False
                    else:
                        h_off = 0
                        w_off = 0
                        self.adain_flag = True
                    
                    tiles,    orig_shape, grid, strides = tile_latent(denoised_pretile, tile_size=(tile_h,tile_w))
                    y0_tiles, orig_shape, grid, strides = tile_latent(y0_adain_pretile, tile_size=(tile_h,tile_w))
                    
                    tiles_out = []
                    for i in range(tiles.shape[0]):
                        tile = tiles[i].unsqueeze(0)
                        y0_tile = y0_tiles[i].unsqueeze(0)
                        
                        tile    = rearrange(tile,    "b c h w -> b (h w) c", h=tile_h, w=tile_w)
                        y0_tile = rearrange(y0_tile, "b c h w -> b (h w) c", h=tile_h, w=tile_w)
                        
                        tile = adain_seq_inplace(tile, y0_tile)
                        tiles_out.append(rearrange(tile, "b (h w) c -> b c h w", h=tile_h, w=tile_w))
                    
                    tiles_out_tensor = torch.cat(tiles_out, dim=0)
                    tiles_out_tensor = untile_latent(tiles_out_tensor, orig_shape, grid, strides)

                    if h_off == 0:
                        denoised_pretile = tiles_out_tensor
                    else:
                        denoised_pretile[:,:,h_off:-h_off, w_off:-w_off] = tiles_out_tensor
                    denoised_embed = rearrange(denoised_pretile, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):

                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    if   freqsep_lowpass_method == "median_pw":
                        denoised_spatial_new = adain_patchwise_row_batch_med(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True, lowpass_weight=freqsep_lowpass_weight, highpass_weight=freqsep_highpass_weight)
                    elif freqsep_lowpass_method == "gaussian_pw": 
                        denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None: 
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if   freqsep_lowpass_method == "median":
                        denoised_spatial_LP = median_blur_2d(denoised_spatial, kernel_size=freqsep_kernel_size)
                        y0_adain_spatial_LP = median_blur_2d(y0_adain_spatial, kernel_size=freqsep_kernel_size)
                    elif freqsep_lowpass_method == "gaussian":
                        denoised_spatial_LP = gaussian_blur_2d(denoised_spatial, sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                        y0_adain_spatial_LP = gaussian_blur_2d(y0_adain_spatial, sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_spatial_HP = denoised_spatial - denoised_spatial_LP
                    
                    if EO("adain_fs_uhp"):
                        y0_adain_spatial_HP = y0_adain_spatial - y0_adain_spatial_LP
                        
                        denoised_spatial_ULP = gaussian_blur_2d(denoised_spatial, sigma=EO("adain_fs_uhp_sigma", 1.0), kernel_size=EO("adain_fs_uhp_kernel_size", 3))
                        y0_adain_spatial_ULP = gaussian_blur_2d(y0_adain_spatial, sigma=EO("adain_fs_uhp_sigma", 1.0), kernel_size=EO("adain_fs_uhp_kernel_size", 3))
                        
                        denoised_spatial_UHP = denoised_spatial_HP  - denoised_spatial_ULP
                        y0_adain_spatial_UHP = y0_adain_spatial_HP  - y0_adain_spatial_ULP
                        
                        #denoised_spatial_HP  = y0_adain_spatial_ULP + denoised_spatial_UHP
                        denoised_spatial_HP  = denoised_spatial_ULP + y0_adain_spatial_UHP
                    
                    denoised_spatial_new = freqsep_lowpass_weight * y0_adain_spatial_LP + freqsep_highpass_weight * denoised_spatial_HP
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                else:
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = self.Retrojector.embed(self.Retrojector.unembed(denoised_embed))
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
            elif transformer_options['y0_style_method'] == "WCT":
                self.StyleWCT.set(y0_adain_embed)
                denoised_embed = self.StyleWCT.get(denoised_embed)
                
                if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    y0_standard_guide_embed = self.Retrojector.embed(y0_standard_guide)
                    f_cs = self.StyleWCT.get(y0_standard_guide_embed)
                    self.y0_standard_guide = self.Retrojector.unembed(f_cs)

                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')

                    y0_inv_standard_guide_embed = self.Retrojector.embed(y0_inv_standard_guide)
                    f_cs = self.StyleWCT.get(y0_inv_standard_guide_embed)
                    self.y0_inv_standard_guide = self.Retrojector.unembed(f_cs)

            denoised_approx = self.Retrojector.unembed(denoised_embed)
            
            eps = (x - denoised_approx) / sigma

            if not UNCOND:
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
            elif eps.shape[0] == 1 and UNCOND:
                eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
            
            eps = eps.float()
        
        if y0_style_neg is not None:
            y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight")
            y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight")
            y0_style_neg_synweight *= y0_style_neg_weight
            y0_style_neg_mask = transformer_options.get("y0_style_neg_mask")
            y0_style_neg_mask_edge = transformer_options.get("y0_style_neg_mask_edge")
            
            y0_style_neg = y0_style_neg.to(dtype)
            x   = x_orig.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps

            denoised_embed = self.Retrojector.embed(denoised)
            y0_adain_embed = self.Retrojector.embed(y0_style_neg)
            
            if transformer_options['y0_style_method'] == "scattersort":
                tile_h, tile_w = transformer_options.get('y0_style_tile_height'), transformer_options.get('y0_style_tile_width')
                pad = transformer_options.get('y0_style_tile_padding')
                if pad is not None and tile_h is not None and tile_w is not None:
                    
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    denoised_spatial = apply_scattersort_tiled(denoised_spatial, y0_adain_spatial, tile_h, tile_w, pad)
                    
                    denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")

                else:
                    denoised_embed = apply_scattersort_masked(denoised_embed, y0_adain_embed, y0_style_neg_mask, y0_style_neg_mask_edge, h_len, w_len)
            
            
            elif transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = self.Retrojector.embed(self.Retrojector.unembed(denoised_embed))
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
            elif transformer_options['y0_style_method'] == "WCT":
                self.StyleWCT.set(y0_adain_embed)
                denoised_embed = self.StyleWCT.get(denoised_embed)

            denoised_approx = self.Retrojector.unembed(denoised_embed)

            if UNCOND:
                eps = (x - denoised_approx) / sigma
                eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            elif eps.shape[0] == 1 and not UNCOND:
                eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
            
            eps = eps.float()
        
        return eps




    def expand_timesteps(self, t, batch_size, device):
        if not torch.is_tensor(t):
            is_mps = device.type == "mps"
            if isinstance(t, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32   if is_mps else torch.int64
            t = Tensor([t], dtype=dtype, device=device)
        elif len(t.shape) == 0:
            t = t[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = t.expand(batch_size)
        return t


    def unpatchify(self, x: Tensor, img_sizes: List[Tuple[int, int]]) -> List[Tensor]:
        x_arr = []
        for i, img_size in enumerate(img_sizes):   #  [[64,64]]
            pH, pW = img_size
            x_arr.append(
                einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)',
                    p1=self.patch_size, p2=self.patch_size)
            )
        x = torch.cat(x_arr, dim=0)
        return x


    def patchify(self, x, max_seq, img_sizes=None):
        pz2 = self.patch_size * self.patch_size
        if isinstance(x, Tensor):
            B      = x.shape[0]
            device = x.device
            dtype  = x.dtype
        else:
            B      = len(x)
            device = x[0].device
            dtype  = x[0].dtype
        x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)

        if img_sizes is not None:
            for i, img_size in enumerate(img_sizes): #  [[64,64]]
                x_masks[i, 0:img_size[0] * img_size[1]] = 1
            x         = einops.rearrange(x, 'B C S p -> B S (p C)', p=pz2)
        elif isinstance(x, Tensor):
            pH, pW    = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
            x         = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.patch_size, p2=self.patch_size)
            img_sizes = [[pH, pW]] * B
            x_masks   = None
        else:
            raise NotImplementedError
        return x, x_masks, img_sizes
    
    
def clone_inputs(*args, index: int=None):

    if index is None:
        return tuple(x.clone() for x in args)
    else:
        return tuple(x[index].unsqueeze(0).clone() for x in args)





def attention_rescale(
    query, 
    key, 
    value,
    attn_mask=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))


    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask is not None:
        attn_weight *= attn_mask

    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value


