import torch
import torch.nn.functional as F


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
from ..latents import slerp_tensor, interpolate_spd

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
    ) -> FloatTensor:
        return self.block(img, img_masks, txt, clip, rope, mask, update_cross_attn, cache_v, attninj_opts)



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






class FeedForwardSwiGLU(nn.Module):
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

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))




# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
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

class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim                   : int,
        hidden_dim            : int,
        num_routed_experts    : int,
        num_activated_experts : int,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.shared_experts =                FeedForwardSwiGLU(dim, hidden_dim // 2,  dtype=dtype, device=device, operations=operations)
        self.experts        = nn.ModuleList([FeedForwardSwiGLU(dim, hidden_dim     ,  dtype=dtype, device=device, operations=operations) for i in range(num_routed_experts)])
        self.gate           = MoEGate(dim, num_routed_experts, num_activated_experts, dtype=dtype, device=device)
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        y_shared = self.shared_experts(x)

        topk_weight, topk_idx = self.gate(x)
        flat_topk_idx         = topk_idx.view(-1)
        
        x = x.view(-1, x.shape[-1])
        x = x.repeat_interleave(self.num_activated_experts, dim=0)
        
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(x.dtype)
        y = torch.einsum('bk,bkd->bd', topk_weight, y.view(*topk_weight.shape, -1))
        
        return y.view_as(y_shared) + y_shared



def attention(q: Tensor, k: Tensor, v: Tensor, rope: Tensor, mask: Optional[Tensor] = None):
    q, k = apply_rope(q, k, rope)
    if mask is not None:
        return attention_pytorch(
            q.view(q.shape[0], -1, q.shape[-1] * q.shape[-2]), 
            k.view(k.shape[0], -1, k.shape[-1] * k.shape[-2]), 
            v.view(v.shape[0], -1, v.shape[-1] * v.shape[-2]), 
            q.shape[2],
            mask=mask,
            )
    else:
        return optimized_attention(
            q.view(q.shape[0], -1, q.shape[-1] * q.shape[-2]), 
            k.view(k.shape[0], -1, k.shape[-1] * k.shape[-2]), 
            v.view(v.shape[0], -1, v.shape[-1] * v.shape[-2]), 
            q.shape[2],
            mask=mask,
            )

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
                self.img_q_cache = img_q
            if weight_img_k != 0 or weight_img_k_norm != 0:
                self.img_k_cache = img_k
            if weight_img_v != 0 or weight_img_v_norm != 0:
                self.img_v_cache = img_v
        else: 
            if self.EO("slerp"):
                interp_fn = slerp_tensor
            else:
                interp_fn = lambda weight, A, B: (1-weight) * A + weight * B

            if weight_img_q != 0 and hasattr(self, "img_q_cache"):
                img_q = interp_fn(weight_img_q, img_q, self.img_q_cache)
            if weight_img_k != 0 and hasattr(self, "img_k_cache"):
                img_k = interp_fn(weight_img_k, img_k, self.img_k_cache)
            if weight_img_v != 0 and hasattr(self, "img_v_cache"):
                img_v = interp_fn(weight_img_v, img_v, self.img_v_cache)
                
            if weight_img_q_norm != 0 and hasattr(self, "img_q_cache"):
                img_q = interp_fn(weight_img_q_norm, img_q, adain_seq(img_q, self.img_q_cache))
            if weight_img_k_norm != 0 and hasattr(self, "img_k_cache"):
                img_k = interp_fn(weight_img_k_norm, img_k, adain_seq(img_k, self.img_k_cache))
            if weight_img_v_norm != 0 and hasattr(self, "img_v_cache"):
                img_v = interp_fn(weight_img_v_norm, img_v, adain_seq(img_v, self.img_v_cache))
                


        inner_dim = img_k.shape[-1]
        head_dim  = inner_dim // self.heads

        img_q = img_q.view(bsz, -1, self.heads, head_dim)
        img_k = img_k.view(bsz, -1, self.heads, head_dim)
        img_v = img_v.view(bsz, -1, self.heads, head_dim)
        
        if img_masks is not None:
            img_k = img_k * img_masks.view(bsz, -1, 1, 1)


        if self.single:
            attn = attention(img_q, img_k, img_v, rope=rope, mask=mask)
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
                    self.txt_q_cache = txt_q
                if weight_txt_k != 0 or weight_txt_k_norm != 0:
                    self.txt_k_cache = txt_k
                if weight_txt_v != 0 or weight_txt_v_norm != 0:
                    self.txt_v_cache = txt_v
            else: 
                if self.EO("slerp"):
                    interp_fn = slerp_tensor
                else:
                    interp_fn = lambda weight, A, B: (1-weight) * A + weight * B

                if weight_txt_q != 0 and hasattr(self, "txt_q_cache"):
                    txt_q = interp_fn(weight_txt_q, txt_q, self.txt_q_cache)
                if weight_txt_k != 0 and hasattr(self, "txt_k_cache"):
                    txt_k = interp_fn(weight_txt_k, txt_k, self.txt_k_cache)
                if weight_txt_v != 0 and hasattr(self, "txt_v_cache"):
                    txt_v = interp_fn(weight_txt_v, txt_v, self.txt_v_cache)
                    
                if weight_txt_q_norm != 0 and hasattr(self, "txt_q_cache"):
                    txt_q = interp_fn(weight_txt_q_norm, txt_q, adain_seq(txt_q, self.txt_q_cache))
                if weight_txt_k_norm != 0 and hasattr(self, "txt_k_cache"):
                    txt_k = interp_fn(weight_txt_k_norm, txt_k, adain_seq(txt_k, self.txt_k_cache))
                if weight_txt_v_norm != 0 and hasattr(self, "txt_v_cache"):
                    txt_v = interp_fn(weight_txt_v_norm, txt_v, adain_seq(txt_v, self.txt_v_cache))

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
                self.img_attn_cache = img_attn.clone()
            elif hasattr(self, "img_attn_cache") and self.EO("img_attn_adain"):
                attn_adain_weight = self.EO("img_attn_adain_weight", 0.99)
                img_attn = attn_adain_weight * img_attn + (1 - attn_adain_weight) * adain_seq(img_attn, self.img_attn_cache)
                del self.img_attn_cache
            
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
        self.ff_i    = MOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts,  dtype=dtype, device=device, operations=operations)
                
        self.norm3_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,           dtype=dtype, device=device)                                 
        self.ff_t    =    FeedForwardSwiGLU(dim, 4*dim,                                             dtype=dtype, device=device, operations=operations)

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
    ) -> FloatTensor:
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate, \
        txt_msa_shift, txt_msa_scale, txt_msa_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(12, dim=-1)      # 1,1,2560           

        img_norm = self.norm1_i(img) * (1+img_msa_scale) + img_msa_shift
        txt_norm = self.norm1_t(txt) * (1+txt_msa_scale) + txt_msa_shift

        img_attn, txt_attn = self.attn1(img_norm, img_masks, txt_norm, rope=rope, mask=mask, update_cross_attn=update_cross_attn, cache_v=cache_v, attninj_opts=attninj_opts)
        
        img     += img_attn            *    img_msa_gate
        img_norm = self.norm3_i(img) * (1+img_mlp_scale) + img_mlp_shift
        img     += self.ff_i(img_norm) *    img_mlp_gate
        
        txt     += txt_attn            *    txt_msa_gate
        txt_norm = self.norm3_t(txt) * (1+txt_mlp_scale) + txt_mlp_shift
        txt     += self.ff_t(txt_norm) *    txt_mlp_gate 
        
        return img, txt


#########################################################################################################################################################################
class HDBlockSingle(nn.Module):
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
        self.ff_i    = MOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts,  dtype=dtype, device=device, operations=operations)

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

    ) -> FloatTensor:
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(6, dim=-1)

        img_norm = self.norm1_i(img) * (1+img_msa_scale) + img_msa_shift
        
        img_attn = self.attn1(img_norm, img_masks, rope=rope, mask=mask, cache_v=cache_v, attninj_opts=attninj_opts)
        
        img     += img_attn            *    img_msa_gate
        img_norm = self.norm3_i(img) * (1+img_mlp_scale) + img_mlp_shift
        img     += self.ff_i(img_norm) *    img_mlp_gate
        
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
        if y0_adain is not None:
            y0_adain            = y0_adain.to(x)
            img_y0_adain        = comfy.ldm.common_dit.pad_to_patch_size(y0_adain, (self.patch_size, self.patch_size))
            t_y0_adain          = torch.full_like(t, 0.0)[0].unsqueeze(0)
            blocks_adain        = transformer_options.get("blocks_adain")
            ADAIN_SINGLE_BLOCKS = blocks_adain['single_blocks']
            ADAIN_DOUBLE_BLOCKS = blocks_adain['double_blocks']
        
        y0_attninj = transformer_options.get("y0_attninj")
        if y0_attninj is not None:
            y0_attninj            = y0_attninj.to(x)
            img_y0_attninj        = comfy.ldm.common_dit.pad_to_patch_size(y0_attninj, (self.patch_size, self.patch_size))
            t_y0_attninj          = torch.full_like(t, 0.0)[0].unsqueeze(0)
            blocks_attninj        = transformer_options.get("blocks_attninj")
            blocks_attninj_qkv    = transformer_options.get("blocks_attninj_qkv")
            ATTNINJ_SINGLE_BLOCKS = blocks_attninj['single_blocks']
            ATTNINJ_DOUBLE_BLOCKS = blocks_attninj['double_blocks']
        
        if y0_adain is not None and y0_attninj is not None and torch.norm(y0_adain - y0_attninj) == 0.0:
            IDENTICAL_ADAIN_ATTNINJ = True
        else:
            IDENTICAL_ADAIN_ATTNINJ = False

            
            
            
        img_orig, t_orig, y_orig, context_orig, llama3_orig = clone_inputs(img, t, y, context, encoder_hidden_states_llama3)
        
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
            
            if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
                txt_init_y0_adain = txt_init.clone()
            if STYLE_UNCOND == UNCOND and img_y0_attninj is not None:
                txt_init_y0_attninj = txt_init.clone()
            
            for bid, block in enumerate(self.double_stream_blocks):                                                              # len == 16
                txt_llama = contexts[bid]
                txt       = torch.cat([txt_init, txt_llama], dim=1)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

                #txt_llama_orig = contexts_orig[bid]
                #txt_orig       = torch.cat([txt_init_orig, txt_llama_orig], dim=1)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

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
                    if STYLE_UNCOND == UNCOND and img_y0_attninj is not None:
                        cache_v = False
                        if bid in ATTNINJ_DOUBLE_BLOCKS:
                            cache_v = True
                        
                        scale = blocks_attninj['double_weights'][bid]
                        blocks_attninj_qkv_scaled = {k: v * scale for k, v in blocks_attninj_qkv.items()}
                        txt_y0_attninj                      = torch.cat([txt_init_y0_attninj, txt_llama], dim=1)
                        img_y0_attninj, txt_init_y0_attninj = block(img_y0_attninj, img_masks, txt_y0_attninj, clip_y0_attninj, rope, mask, cache_v=cache_v, attninj_opts=blocks_attninj_qkv_scaled)
                        txt_init_y0_attninj                 = txt_init_y0_attninj[:, :txt_init_len]
                        
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask, update_cross_attn=update_cross_attn, attninj_opts=blocks_attninj_qkv_scaled)

                    if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
                        if IDENTICAL_ADAIN_ATTNINJ:
                            img_y0_adain      = img_y0_attninj
                            txt_init_y0_adain = txt_init_y0_attninj
                        else:
                            txt_y0_adain                    = torch.cat([txt_init_y0_adain, txt_llama], dim=1)
                            img_y0_adain, txt_init_y0_adain = block(img_y0_adain, img_masks, txt_y0_adain, clip_y0_adain, rope, mask)
                            txt_init_y0_adain               = txt_init_y0_adain[:, :txt_init_len]
                        
                        if bid in ADAIN_DOUBLE_BLOCKS:
                            adaweight = blocks_adain['double_weights'][bid]
                            img = (1-adaweight) * img + adaweight * adain_seq(img, img_y0_adain)
                            #img = img + (1-SIGMA) * adaweight * (adain_seq(img, img_y0_adain) - img)

                txt_init = txt_init[:, :txt_init_len]

            img_len = img.shape[1]
            img     = torch.cat([img, txt_init], dim=1)   # 4032 + 271 -> 4303     # txt embed from double stream block
            
            if STYLE_UNCOND == UNCOND and y0_adain is not None:
                img_y0_adain = torch.cat([img_y0_adain, txt_init_y0_adain], dim=1)
            if STYLE_UNCOND == UNCOND and y0_attninj is not None:
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
                    if STYLE_UNCOND == UNCOND and img_y0_attninj is not None:
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
                
                if STYLE_UNCOND == UNCOND and img_y0_adain is not None:
                    if IDENTICAL_ADAIN_ATTNINJ:
                        img_y0_adain = img_y0_attninj
                    else:
                        img_y0_adain = torch.cat([img_y0_adain, txt_llama], dim=1)   
                        img_y0_adain = block(img_y0_adain, img_masks, None, clip_y0_adain, rope, mask)
                        img_y0_adain = img_y0_adain[:, :joint_len]
                    
                    if bid in ADAIN_SINGLE_BLOCKS:
                        adaweight = blocks_adain['single_weights'][bid]
                        img = (1-adaweight) * img + adaweight * adain_seq(img, img_y0_adain)
                        #img = img + (1-SIGMA) * adaweight * (adain_seq(img, img_y0_adain) - img)
                
            img = img[:, :img_len, ...]
            
            img_pre_final.append(img)
            
            img = self.final_layer(img, clip)
            img = self.unpatchify (img, img_sizes)
            
            out_list.append(img)
            
            
        output = torch.stack(out_list, dim=0).squeeze(dim=1)

        
        eps = -output[:, :, :h, :w]
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None


        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1 and not UNCOND):
        if y0_style_pos is not None:
            y0_style_pos_weight     = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight  = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            y0_style_pos_mask       = transformer_options.get("y0_style_pos_mask")
            
            y0_style_pos = y0_style_pos.to(dtype)
            x            = x.to(dtype)
            eps          = eps.to(dtype)
            eps_orig     = eps.clone()
            
            mask = y0_style_pos_mask if y0_style_pos_mask is not None else torch.ones_like(x)
            
            sigma = t_orig[0].to(dtype) / 1000
            denoised = x - sigma * eps
            
            img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))
            img_sizes = None
            img, img_masks, img_sizes = self.patchify(img, self.max_seq, img_sizes)
            
            img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_pos, (self.patch_size, self.patch_size))
            img_sizes_y0_adain = None
            img_y0_adain, img_masks_y0_adain, img_sizes_y0_adain = self.patchify(img_y0_adain, self.max_seq, img_sizes_y0_adain) 
            
            W = self.x_embedder.proj.weight.data.to(dtype)   # shape [2560, 64]
            b = self.x_embedder.proj.bias.data.to(dtype)     # shape [2560]
            
            denoised_embed = F.linear(img         .to(W), W, b).to(img)
            y0_adain_embed = F.linear(img_y0_adain.to(W), W, b).to(img_y0_adain)
            
            if transformer_options['y0_style_method'] == "AdaIN":
                if freqsep_mask is not None:
                    freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                    freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                
                if freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):
                    
                    #if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    #    self.y0_adain_embed = y0_adain_embed
                    #    self.adain_pw_cache = None
                        
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    if freqsep_lowpass_method == "median_alt": 
                        denoised_spatial_new = adain_patchwise_row_batch_medblur(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True)
                    elif freqsep_lowpass_method == "median_pw":
                        denoised_spatial_new = adain_patchwise_row_batch_realmedblur(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True, lowpass_weight=freqsep_lowpass_weight, highpass_weight=freqsep_highpass_weight)
                    elif freqsep_lowpass_method == "gaussian_pw": 
                        denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None and freqsep_lowpass_method == "distribution": 
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    denoised_spatial_new = adain_patchwise_strict_sortmatch9(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), kernel_size=freqsep_kernel_size, inner_kernel_size=freqsep_inner_kernel_size, mask=freqsep_mask, stride=freqsep_stride)

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
                
                #denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                #for adain_iter in range(EO("style_iter", 0)):
                #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                #    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                #    denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)

            elif transformer_options['y0_style_method'] == "WCT":
                if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    self.y0_adain_embed = y0_adain_embed
                    
                    f_s          = y0_adain_embed[0].clone()
                    self.mu_s    = f_s.mean(dim=0, keepdim=True)
                    f_s_centered = f_s - self.mu_s
                    
                    cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                    S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                    
                    whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                    self.y0_color  = whiten.to(f_s_centered)

                for wct_i in range(eps.shape[0]):
                    f_c          = denoised_embed[wct_i].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T + self.mu_s
                    
                    denoised_embed[wct_i] = f_cs
                    
                if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    img_y0_standard_guide = comfy.ldm.common_dit.pad_to_patch_size(y0_standard_guide, (self.patch_size, self.patch_size))
                    img_sizes_y0_standard_guide = None
                    img_y0_standard_guide, img_masks_y0_standard_guide, img_sizes_y0_standard_guide = self.patchify(img_y0_standard_guide, self.max_seq, img_sizes_y0_standard_guide) 
                    y0_standard_guide_embed = F.linear(img_y0_standard_guide.to(W), W, b).to(img_y0_standard_guide)

                    
                    f_c          = y0_standard_guide_embed[0].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T.to(y0_standard_guide) + self.mu_s.to(y0_standard_guide)
                    
                    f_cs = (f_cs - b) @ torch.linalg.pinv(W.to(f_cs)).T.to(f_cs)
                    y0_standard_guide = self.unpatchify (f_cs.unsqueeze(0), img_sizes_y0_standard_guide)
                    self.y0_standard_guide = y0_standard_guide
                
                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')
                    
                    img_y0_standard_guide = comfy.ldm.common_dit.pad_to_patch_size(y0_inv_standard_guide, (self.patch_size, self.patch_size))
                    img_sizes_y0_standard_guide = None
                    img_y0_standard_guide, img_masks_y0_standard_guide, img_sizes_y0_standard_guide = self.patchify(img_y0_standard_guide, self.max_seq, img_sizes_y0_standard_guide) 
                    y0_standard_guide_embed = F.linear(img_y0_standard_guide.to(W), W, b).to(img_y0_standard_guide)
                    
                    f_c          = y0_standard_guide_embed[0].clone()
                    
                    #f_c          = y0_inv_standard_guide[0].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T.to(y0_inv_standard_guide) + self.mu_s.to(y0_inv_standard_guide)
                    
                    f_cs = (f_cs - b) @ torch.linalg.pinv(W.to(f_cs)).T.to(f_cs)
                    y0_inv_standard_guide = self.unpatchify (f_cs.unsqueeze(0), img_sizes_y0_standard_guide)
                    self.y0_inv_standard_guide = y0_inv_standard_guide

            denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
            denoised_embed = self.unpatchify (denoised_embed, img_sizes)
            
            eps = (x - denoised_embed) / sigma
            #if eps.shape[0] == 2:
            #    eps[1] = eps_orig[1] + mask * y0_style_pos_weight    * (eps[1] - eps_orig[1])
            #    eps[0] = eps_orig[0] + mask * y0_style_pos_synweight * (eps[0] - eps_orig[0])
            #else:
            #    eps[0] = eps_orig[0] + mask * y0_style_pos_weight    * (eps[0] - eps_orig[0])

            if not UNCOND:
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
            elif eps.shape[0] == 1 and UNCOND:
                eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                #if eps.shape[0] == 2:
                #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            
            eps = eps.float()


        if not EO("use_style_neg_as_pos"): # eps.shape[0] == 2 or (eps.shape[0] == 1 and UNCOND) 
            if y0_style_neg is not None:
                y0_style_neg_weight     = transformer_options.get("y0_style_neg_weight")
                y0_style_neg_synweight  = transformer_options.get("y0_style_neg_synweight")
                y0_style_neg_synweight *= y0_style_neg_weight
                y0_style_neg_mask       = transformer_options.get("y0_style_neg_mask")
                
                y0_style_neg = y0_style_neg.to(dtype)
                x            = x.to(dtype)
                eps          = eps.to(dtype)
                eps_orig     = eps.clone()
                
                mask = y0_style_neg_mask if y0_style_neg_mask is not None else torch.ones_like(x)
                
                sigma = t_orig[0].to(dtype) / 1000
                denoised = x - sigma * eps
                
                img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))
                img_sizes = None
                img, img_masks, img_sizes = self.patchify(img, self.max_seq, img_sizes) 
                
                img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_neg, (self.patch_size, self.patch_size))
                img_sizes_y0_adain = None
                img_y0_adain, img_masks_y0_adain, img_sizes_y0_adain = self.patchify(img_y0_adain, self.max_seq, img_sizes_y0_adain) 
                
                W = self.x_embedder.proj.weight.data.to(dtype)   # shape [2560, 64]
                b = self.x_embedder.proj.bias.data.to(dtype)     # shape [2560]
                
                denoised_embed = F.linear(img         .to(W), W, b).to(img)
                y0_adain_embed = F.linear(img_y0_adain.to(W), W, b).to(img_y0_adain)
                
                
                if transformer_options['y0_style_method'] == "AdaIN":
                    if freqsep_mask is not None:
                        freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                        freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                    
                    if freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):
                        
                        #if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        #    self.y0_adain_embed = y0_adain_embed
                        #    self.adain_pw_cache = None
                            
                        denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                        y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                        if freqsep_lowpass_method == "median_alt": 
                            denoised_spatial_new = adain_patchwise_row_batch_medblur(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True)
                        elif freqsep_lowpass_method == "median_pw":
                            denoised_spatial_new = adain_patchwise_row_batch_realmedblur(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True, lowpass_weight=freqsep_lowpass_weight, highpass_weight=freqsep_highpass_weight)
                        elif freqsep_lowpass_method == "gaussian_pw": 
                            denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                        
                        denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                    elif freqsep_lowpass_method is not None and freqsep_lowpass_method == "distribution": 
                        denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                        y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                        denoised_spatial_new = adain_patchwise_strict_sortmatch9(denoised_spatial.clone(), y0_adain_spatial.clone(), kernel_size=freqsep_kernel_size, inner_kernel_size=freqsep_inner_kernel_size, mask=freqsep_mask, stride=freqsep_stride)

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
                    
                    #denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    #for adain_iter in range(EO("style_iter", 0)):
                    #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    #    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    #    denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                    #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
                elif transformer_options['y0_style_method'] == "WCT":
                    if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        self.y0_adain_embed = y0_adain_embed
                        
                        f_s          = y0_adain_embed[0].clone()
                        self.mu_s    = f_s.mean(dim=0, keepdim=True)
                        f_s_centered = f_s - self.mu_s
                        
                        cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                        
                        whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                        self.y0_color  = whiten.to(f_s_centered)

                    for wct_i in range(eps.shape[0]):
                        f_c          = denoised_embed[wct_i].clone()
                        mu_c         = f_c.mean(dim=0, keepdim=True)
                        f_c_centered = f_c - mu_c
                        
                        cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                        S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                        
                        whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                        whiten = whiten.to(f_c_centered)

                        f_c_whitened = f_c_centered @ whiten.T
                        f_cs         = f_c_whitened @ self.y0_color.T + self.mu_s
                        
                        denoised_embed[wct_i] = f_cs

                denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                denoised_embed = self.unpatchify (denoised_embed, img_sizes)
                
                #eps = (x - denoised_embed) / sigma
                #eps[0]     = eps_orig[0] + mask * y0_style_neg_weight    * (eps[0] - eps_orig[0])
                #if eps.shape[0] == 2:
                #    eps[1] = eps_orig[1] + mask * y0_style_neg_synweight * (eps[1] - eps_orig[1])
                    
                if UNCOND:
                    eps = (x - denoised_embed) / sigma
                    eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                elif eps.shape[0] == 1 and not UNCOND:
                    eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
                    
                eps = eps.float()
        
        
        elif EO("use_style_neg_as_pos"): #eps.shape[0] == 2 or (eps.shape[0] == 1 and not UNCOND)    and 
            if y0_style_neg is not None and eps.shape[0] > 1:
                y0_style_neg_weight     = transformer_options.get("y0_style_neg_weight")
                y0_style_neg_synweight  = transformer_options.get("y0_style_neg_synweight")
                y0_style_neg_synweight *= y0_style_neg_weight
                y0_style_neg_mask       = transformer_options.get("y0_style_neg_mask")
                
                y0_style_neg = y0_style_neg.to(dtype)
                x            = x.to(dtype)
                eps          = eps.to(dtype)
                eps_orig     = eps.clone()
                
                mask = y0_style_neg_mask if y0_style_neg_mask is not None else torch.ones_like(x)
                
                sigma = t_orig[0].to(dtype) / 1000
                denoised = x - sigma * eps
                
                img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))
                img_sizes = None
                img, img_masks, img_sizes = self.patchify(img, self.max_seq, img_sizes) 
                
                img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_neg, (self.patch_size, self.patch_size))
                img_sizes_y0_adain = None
                img_y0_adain, img_masks_y0_adain, img_sizes_y0_adain = self.patchify(img_y0_adain, self.max_seq, img_sizes_y0_adain) 
                
                W = self.x_embedder.proj.weight.data.to(dtype)   # shape [2560, 64]
                b = self.x_embedder.proj.bias.data.to(dtype)     # shape [2560]
                
                denoised_embed = F.linear(img         .to(W), W, b).to(img)
                y0_adain_embed = F.linear(img_y0_adain.to(W), W, b).to(img_y0_adain)
                
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                
                denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                denoised_embed = self.unpatchify (denoised_embed, img_sizes)
                
                #eps = (x - denoised_embed) / sigma
                #if eps.shape[0] == 2:
                #    eps[1] = eps_orig[1] + mask * y0_style_neg_weight    * (eps[1] - eps_orig[1])
                #    eps[0] = eps_orig[0] + mask * y0_style_neg_synweight * (eps[0] - eps_orig[0])
                #else:
                #    eps[0] = eps_orig[0] + mask * y0_style_neg_weight    * (eps[0] - eps_orig[0])

                if UNCOND:
                    eps = (x - denoised_embed) / sigma
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



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)
    return content


def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)





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















def gaussian_blur_2d(img: torch.Tensor, sigma: float, kernel_size: int = None) -> torch.Tensor:
    B, C, H, W = img.shape
    dtype = img.dtype
    device = img.device

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)

    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, dtype=torch.float64) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()

    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d.to(dtype=dtype, device=device)

    kernel = kernel_2d.expand(C, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')

    return F.conv2d(img_padded, kernel, groups=C)


def median_blur_2d(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    B, C, H, W = img.shape
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')

    unfolded = img_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # unfolded: [B, C, H, W, kH, kW] → flatten to patches
    patches = unfolded.contiguous().view(B, C, H, W, -1)
    median = patches.median(dim=-1).values
    return median


def adain_patchwise(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:

    B, C, H, W = content.shape
    device     = content.device
    dtype      = content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad    = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss  = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss /= gauss.sum()
    kernel_2d = (gauss[:, None] * gauss[None, :]).to(dtype=dtype)

    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded   = F.pad(style,   (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    for i in range(H):
        for j in range(W):
            c_patch = content_padded[:, :, i:i + kernel_size, j:j + kernel_size]
            s_patch =   style_padded[:, :, i:i + kernel_size, j:j + kernel_size]
            w = weight.expand_as(c_patch)

            c_mean =  (c_patch              * w).sum(dim=(-1, -2), keepdim=True)
            c_std  = ((c_patch - c_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
            s_mean =  (s_patch              * w).sum(dim=(-1, -2), keepdim=True)
            s_std  = ((s_patch - s_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

            normed =  (c_patch[:, :, pad:pad+1, pad:pad+1] - c_mean) / c_std
            stylized = normed * s_std + s_mean
            result[:, :, i, j] = stylized.squeeze(-1).squeeze(-1)

    return result




def adain_patchwise_row_batch(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss = (gauss / gauss.sum()).to(dtype)
    kernel_2d = (gauss[:, None] * gauss[None, :])

    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    for i in range(H):
        c_row_patches = torch.stack([
            content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)  # [W, B, C, k, k]

        s_row_patches = torch.stack([
            style_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)

        w = weight.expand_as(c_row_patches[0])

        c_mean = (c_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        c_std  = ((c_row_patches - c_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
        s_mean = (s_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        s_std  = ((s_row_patches - s_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center:center+1, center:center+1]
        normed = (central - c_mean) / c_std
        stylized = normed * s_std + s_mean

        result[:, :, i, :] = stylized.squeeze(-1).squeeze(-1).permute(1, 2, 0)  # [B,C,W]

    return result







def adain_patchwise_row_batch_medblur(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * abs(sigma)) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    scaling = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    sigma_scale = torch.ones((H, W), device=device, dtype=torch.float32)
    if mask is not None:
        with torch.no_grad():
            padded_mask = F.pad(mask.float(), (pad, pad, pad, pad), mode="reflect")
            blurred_mask = F.avg_pool2d(padded_mask, kernel_size=kernel_size, stride=1, padding=pad)
            blurred_mask = blurred_mask[..., pad:-pad, pad:-pad]
            edge_proximity = blurred_mask * (1.0 - blurred_mask)
            scaling = 1.0 - (edge_proximity / 0.25).clamp(0.0, 1.0)
            sigma_scale = scaling[0, 0]  # assuming single-channel mask broadcasted across B, C

    if not use_median_blur:
        coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
        base_gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        base_gauss = (base_gauss / base_gauss.sum()).to(dtype)
        gaussian_table = {}
        for s in sigma_scale.unique():
            sig = float((sigma * s + eps).clamp(min=1e-3))
            gauss_local = torch.exp(-0.5 * (coords / sig) ** 2)
            gauss_local = (gauss_local / gauss_local.sum()).to(dtype)
            kernel_2d = gauss_local[:, None] * gauss_local[None, :]
            gaussian_table[s.item()] = kernel_2d

    for i in range(H):
        row_result = torch.zeros(B, C, W, dtype=dtype, device=device)
        for j in range(W):
            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch = style_padded[:, :, i:i+kernel_size, j:j+kernel_size]

            if use_median_blur:
                c_flat = c_patch.reshape(B, C, -1)
                s_flat = s_patch.reshape(B, C, -1)

                c_median = c_flat.median(dim=-1, keepdim=True).values
                s_median = s_flat.median(dim=-1, keepdim=True).values

                c_std = (c_flat - c_median).abs().mean(dim=-1, keepdim=True) + eps
                s_std = (s_flat - s_median).abs().mean(dim=-1, keepdim=True) + eps

                center = kernel_size // 2
                central = c_patch[:, :, center, center].unsqueeze(-1)

                normed = (central - c_median) / c_std
                stylized = normed * s_std + s_median
            else:
                k = gaussian_table[float(sigma_scale[i, j].item())]
                local_weight = k.view(1, 1, kernel_size, kernel_size).expand(B, C, kernel_size, kernel_size)

                c_mean = (c_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                c_std = ((c_patch - c_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
                s_mean = (s_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                s_std = ((s_patch - s_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

                center = kernel_size // 2
                central = c_patch[:, :, center:center+1, center:center+1]
                normed = (central - c_mean) / c_std
                stylized = normed * s_std + s_mean

            local_scaling = scaling[:, :, i, j].view(B, 1, 1, 1)
            stylized = central * (1 - local_scaling) + stylized * local_scaling

            row_result[:, :, j] = stylized.squeeze(-1).squeeze(-1)
        result[:, :, i, :] = row_result

    return result







def adain_patchwise_row_batch_realmedblur(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False, lowpass_weight=1.0, highpass_weight=1.0) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * abs(sigma)) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    scaling = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    sigma_scale = torch.ones((H, W), device=device, dtype=torch.float32)
    if mask is not None:
        with torch.no_grad():
            padded_mask = F.pad(mask.float(), (pad, pad, pad, pad), mode="reflect")
            blurred_mask = F.avg_pool2d(padded_mask, kernel_size=kernel_size, stride=1, padding=pad)
            blurred_mask = blurred_mask[..., pad:-pad, pad:-pad]
            edge_proximity = blurred_mask * (1.0 - blurred_mask)
            scaling = 1.0 - (edge_proximity / 0.25).clamp(0.0, 1.0)
            sigma_scale = scaling[0, 0]  # assuming single-channel mask broadcasted across B, C

    if not use_median_blur:
        coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
        base_gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        base_gauss = (base_gauss / base_gauss.sum()).to(dtype)
        gaussian_table = {}
        for s in sigma_scale.unique():
            sig = float((sigma * s + eps).clamp(min=1e-3))
            gauss_local = torch.exp(-0.5 * (coords / sig) ** 2)
            gauss_local = (gauss_local / gauss_local.sum()).to(dtype)
            kernel_2d = gauss_local[:, None] * gauss_local[None, :]
            gaussian_table[s.item()] = kernel_2d

    for i in range(H):
        row_result = torch.zeros(B, C, W, dtype=dtype, device=device)
        for j in range(W):
            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch = style_padded[:, :, i:i+kernel_size, j:j+kernel_size]

            if use_median_blur:
                # Median blur with residual restoration
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)

                c_median = unfolded_c.median(dim=-1, keepdim=True).values
                s_median = unfolded_s.median(dim=-1, keepdim=True).values

                center = kernel_size // 2
                central = c_patch[:, :, center, center].view(B, C, 1)
                residual = central - c_median
                stylized = lowpass_weight * s_median + residual * highpass_weight
            else:
                k = gaussian_table[float(sigma_scale[i, j].item())]
                local_weight = k.view(1, 1, kernel_size, kernel_size).expand(B, C, kernel_size, kernel_size)

                c_mean = (c_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                c_std = ((c_patch - c_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
                s_mean = (s_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                s_std = ((s_patch - s_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

                center = kernel_size // 2
                central = c_patch[:, :, center:center+1, center:center+1]
                normed = (central - c_mean) / c_std
                stylized = normed * s_std + s_mean

            local_scaling = scaling[:, :, i, j].view(B, 1, 1)
            stylized = central * (1 - local_scaling) + stylized * local_scaling

            row_result[:, :, j] = stylized.squeeze(-1)
        result[:, :, i, :] = row_result

    return result













def patchwise_sort_transfer9(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    src, ref: [B, C, N] where N = K*K
    Returns: [B, C, N] with values from ref permuted to match the sort-order of src.
    """
    src_sorted, src_idx  = src.sort(dim=-1)
    ref_sorted, _        = ref.sort(dim=-1)
    out = torch.zeros_like(src)
    out.scatter_(dim=-1, index=src_idx, src=ref_sorted)
    return out

def masked_patchwise_sort_transfer9(
    src       : torch.Tensor,         # [B, C, N]
    ref       : torch.Tensor,         # [B, C, N]
    mask_flat : torch.Tensor          # [B, N]  bool
) -> torch.Tensor:
    """
    Only rearrange N positions where mask_flat[b] is True... to be implemented fully later. 
    """
    B,C,N = src.shape
    out = src.clone()
    for b in range(B):
        valid = mask_flat[b]             # [N] boolean
        if valid.sum() == 0:
            continue
        sc = src[b,:,valid]              # [C, M]
        ss = ref[b,:,valid]              # [C, M]
        sc_s, idx = sc.sort(dim=-1)      # sort the channelz
        ss_s, _   = ss.sort(dim=-1)
        res = torch.zeros_like(sc)
        res.scatter_(dim=-1, index=idx, src=ss_s)
        out[b,:,valid] = res
    return out

def adain_patchwise_strict_sortmatch9(
    content           : torch.Tensor,        # [B,C,H,W]
    style             : torch.Tensor,        # [B,C,H,W]
    kernel_size       : int,
    inner_kernel_size : int = 1,
    stride            : int = 1,
    mask              : torch.Tensor = None  # [B,1,H,W]
) -> torch.Tensor:
    B,C,H,W = content.shape
    assert inner_kernel_size <= kernel_size
    pad       = kernel_size//2
    inner_off = (kernel_size - inner_kernel_size)//2

    # reflect-pad
    cp = F.pad(content, (pad,)*4, mode='reflect')
    sp = F.pad(style,   (pad,)*4, mode='reflect')
    out = content.clone()

    if mask is not None:
        mask = mask[:,0].bool()  # [B,H,W]

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            pc = cp[:, :, i:i+kernel_size, j:j+kernel_size]   # [B,C,K,K]
            ps = sp[:, :, i:i+kernel_size, j:j+kernel_size]

            Bc = pc.reshape(B, C, -1)
            Bs = ps.reshape(B, C, -1)

            matched_flat = patchwise_sort_transfer9(Bc, Bs)
            matched = matched_flat.view(B, C, kernel_size, kernel_size)

            y0, x0 = inner_off, inner_off
            y1, x1 = y0 + inner_kernel_size, x0 + inner_kernel_size
            inner = matched[:, :, y0:y1, x0:x1]  # [B,C,inner,inner]

            dst_y0 = i + y0 - pad
            dst_x0 = j + x0 - pad
            dst_y1 = dst_y0 + inner_kernel_size
            dst_x1 = dst_x0 + inner_kernel_size

            oy0 = max(dst_y0, 0); ox0 = max(dst_x0, 0)
            oy1 = min(dst_y1, H); ox1 = min(dst_x1, W)

            iy0 = oy0 - dst_y0; ix0 = ox0 - dst_x0
            iy1 = iy0 + (oy1 - oy0); ix1 = ix0 + (ox1 - ox0)

            if mask is None:
                out[:, :, oy0:oy1, ox0:ox1] = inner[:, :, iy0:iy1, ix0:ix1]
            else:
                ibm = mask[:, oy0:oy1, ox0:ox1]  # [B,inner,inner]
                for b in range(B):
                    sel = ibm[b]  # [inner,inner]   # w/ regard to kernel
                    if sel.any():
                        out[b:b+1, :, oy0:oy1, ox0:ox1][:, :,sel]   =   inner[b:b+1, :, iy0:iy1, ix0:ix1][:, :, sel]
    return out

