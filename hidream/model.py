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
#from ..flux.layers import LastLayer

from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
import comfy.model_management
import comfy.ldm.common_dit

from ..helper  import ExtraOptions
from ..latents import slerp_tensor, interpolate_spd, tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import StyleMMDiT_Model, apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch, adain_seq, apply_scattersort

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
        style_block  = None,
    ) -> FloatTensor:
        return self.block(img, img_masks, txt, clip, rope, mask, update_cross_attn, style_block=style_block)



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

    def forward(self, x, style_block=None): # 1,4096,2560 -> 
        if style_block is not None and x.shape[0] > 1 and x.ndim == 3:
            x1 = self.w1(x)
            x1 = style_block(x1, "ff_1")
            
            x1 = torch.nn.functional.silu(x1)
            x1 = style_block(x1, "ff_1_silu")
            
            x3 = self.w3(x)
            x3 = style_block(x3, "ff_3")
            
            x13 = x1 * x3
            x13 = style_block(x13, "ff_13")
            
            x2 = self.w2(x13)
            x2 = style_block(x2, "ff_2")
            
            return x2
        else:
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class HDMoEGate(nn.Module):
    def __init__(self, dim, num_routed_experts=4, num_activated_experts=2, dtype=None, device=None):
        super().__init__()
        self.top_k            = num_activated_experts # 2
        self.n_routed_experts = num_routed_experts    # 4
        self.gating_dim       = dim                   # 2560
        self.weight           = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), dtype=dtype, device=device))

    def forward(self, x):
        dtype = self.weight.dtype
        if dtype not in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
            dtype = torch.float32
            self.weight.data = self.weight.data.to(dtype)
        
        logits = F.linear(x.to(dtype), self.weight.to(x.device), None)
        scores = logits.softmax(dim=-1).to(x)       # logits.shape == 4032,4   scores.shape == 4032,4
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

    def forward(self, x, style_block=None):
        y_shared = self.shared_experts(x, style_block.FF_SHARED)   # 1,4096,2560 -> 1,4096,2560 
        y_shared = style_block(y_shared, "shared")

        topk_weight, topk_idx = self.gate(x) # -> 4096,2   4096,2
        topk_weight = style_block(topk_weight, "topk_weight")
                
        if y_shared.shape[0] > 1 and style_block.gate[0] and not HDModel.RECON_MODE:
            topk_idx[0] = topk_idx[1]
        tk_idx_flat = topk_idx.view(topk_idx.shape[0], -1) 
        
        x = x.repeat_interleave(self.num_activated_experts, dim=-2)
        y = torch.empty_like(x)
        
        if style_block.gate[0] and not HDModel.RECON_MODE and y_shared.shape[0] > 1:
            for i, expert in enumerate(self.experts): # TODO: check for empty expert lists and continue if found to avoid CUBLAS errors
                x_list = []
                for b in range(x.shape[0]):
                    x_sel = x[b][tk_idx_flat[b]==i]
                    x_list.append(x_sel)
                x_list = torch.stack(x_list, dim=0)
                x_out = expert(x_list, style_block.FF_SEPARATE).to(x.dtype)
                for b in range(y.shape[0]):
                    y[b][tk_idx_flat[b]==i] = x_out[b]
        else:
            for i, expert in enumerate(self.experts): 
                x_sel = x[tk_idx_flat == i, :]
                if x_sel.shape[0] == 0:
                    continue 
                y[tk_idx_flat == i, :] = expert(x_sel).to(x.dtype)
                
        y = style_block(y, "separate")

        y_sum = torch.einsum('abk,abkd->abd', topk_weight, y.view(*topk_weight.shape, -1))
        
        y_sum = style_block(y_sum, "sum")
        
        y_sum = y_sum.view_as(y_shared) + y_shared

        y_sum = style_block(y_sum, "out")
        
        return y_sum


def apply_passthrough(denoised_embed, *args, **kwargs):
    return denoised_embed

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
        style_block = None,
    ) -> Tensor:
        bsz = img.shape[0]
        
        img_q = self.to_q(img)
        img_k = self.to_k(img)
        img_v = self.to_v(img)
        
        img_q = style_block.img.ATTN(img_q, "q_proj")
        img_k = style_block.img.ATTN(img_k, "k_proj")
        img_v = style_block.img.ATTN(img_v, "v_proj")
        
        img_q = self.q_rms_norm(img_q)
        img_k = self.k_rms_norm(img_k)
        
        img_q = style_block.img.ATTN(img_q, "q_norm")
        img_k = style_block.img.ATTN(img_k, "k_norm")

        inner_dim = img_k.shape[-1]
        head_dim  = inner_dim // self.heads

        img_q = img_q.view(bsz, -1, self.heads, head_dim)
        img_k = img_k.view(bsz, -1, self.heads, head_dim)
        img_v = img_v.view(bsz, -1, self.heads, head_dim)
        
        if img_masks is not None:
            img_k = img_k * img_masks.view(bsz, -1, 1, 1)

        if self.single:
            attn = attention(img_q, img_k, img_v, rope=rope, mask=mask)
            attn = style_block.img.ATTN(attn, "out")
            return self.to_out(attn)
        else:
            txt_q = self.to_q_t(txt)
            txt_k = self.to_k_t(txt)
            txt_v = self.to_v_t(txt)
            
            txt_q = style_block.txt.ATTN(txt_q, "q_proj")
            txt_k = style_block.txt.ATTN(txt_k, "k_proj")
            txt_v = style_block.txt.ATTN(txt_v, "v_proj")
            
            txt_q = self.q_rms_norm_t(txt_q)
            txt_k = self.k_rms_norm_t(txt_k)
            
            txt_q = style_block.txt.ATTN(txt_q, "q_norm")
            txt_k = style_block.txt.ATTN(txt_k, "k_norm")

            txt_q   = txt_q.view(bsz, -1, self.heads, head_dim)
            txt_k   = txt_k.view(bsz, -1, self.heads, head_dim)
            txt_v   = txt_v.view(bsz, -1, self.heads, head_dim)
            
            img_len = img_q.shape[1]
            txt_len = txt_q.shape[1]
            
            attn    = attention(torch.cat([img_q, txt_q], dim=1), 
                                torch.cat([img_k, txt_k], dim=1), 
                                torch.cat([img_v, txt_v], dim=1), rope=rope, mask=mask)
            
            img_attn, txt_attn = torch.split(attn, [img_len, txt_len], dim=1)   #1, 4480, 2560
            
            img_attn = style_block.img.ATTN(img_attn, "out")
            txt_attn = style_block.txt.ATTN(txt_attn, "out")

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
        style_block = None,
    ) -> FloatTensor:
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate, \
        txt_msa_shift, txt_msa_scale, txt_msa_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(12, dim=-1)      # 1,1,2560           

        img_norm = self.norm1_i(img)
        txt_norm = self.norm1_t(txt)
        
        img_norm = style_block.img(img_norm, "attn_norm")
        txt_norm = style_block.txt(txt_norm, "attn_norm")
        
        img_norm = img_norm * (1+img_msa_scale) + img_msa_shift
        txt_norm = txt_norm * (1+txt_msa_scale) + txt_msa_shift
        
        img_norm = style_block.img(img_norm, "attn_norm_mod")
        txt_norm = style_block.txt(txt_norm, "attn_norm_mod")

        img_attn, txt_attn = self.attn1(img_norm, img_masks, txt_norm, rope=rope, mask=mask, update_cross_attn=update_cross_attn, style_block=style_block)
        
        img_attn = style_block.img(img_attn, "attn")
        txt_attn = style_block.txt(txt_attn, "attn")

        img_attn *= img_msa_gate
        txt_attn *= txt_msa_gate

        img_attn = style_block.img(img_attn, "attn_gated")
        txt_attn = style_block.txt(txt_attn, "attn_gated")

        img += img_attn
        txt += txt_attn

        img = style_block.img(img, "attn_res")
        txt = style_block.txt(txt, "attn_res")

        # FEED FORWARD

        img_norm = self.norm3_i(img)
        txt_norm = self.norm3_t(txt)

        img_norm = style_block.img(img_norm, "ff_norm")
        txt_norm = style_block.txt(txt_norm, "ff_norm")

        img_norm = img_norm * (1+img_mlp_scale) + img_mlp_shift
        txt_norm = txt_norm * (1+txt_mlp_scale) + txt_mlp_shift

        img_norm = style_block.img(img_norm, "ff_norm_mod")
        txt_norm = style_block.txt(txt_norm, "ff_norm_mod")

        img_ff_i = self.ff_i(img_norm, style_block.img.FF)
        txt_ff_t = self.ff_t(txt_norm, style_block.txt.FF)
        
        img_ff_i = style_block.img(img_ff_i, "ff")
        txt_ff_t = style_block.txt(txt_ff_t, "ff")
        
        img_ff_i *= img_mlp_gate
        txt_ff_t *= txt_mlp_gate

        img_ff_i = style_block.img(img_ff_i, "ff_gated")
        txt_ff_t = style_block.txt(txt_ff_t, "ff_gated")

        img += img_ff_i
        txt += txt_ff_t
        
        img = style_block.img(img, "ff_res")
        txt = style_block.txt(txt, "ff_res")

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
            operations.Linear(dim, 6 * dim, bias=True,                                               dtype=dtype, device=device)
        )

        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,            dtype=dtype, device=device)
        self.attn1   = HDAttention         (dim, heads, head_dim, single=True,                       dtype=dtype, device=device, operations=operations)

        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,            dtype=dtype, device=device)
        self.ff_i    = HDMOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        img        :          FloatTensor,
        img_masks  : Optional[FloatTensor]  = None,
        txt        : Optional[FloatTensor]  = None,
        clip       : Optional[FloatTensor]  = None,
        rope       :          FloatTensor   = None,
        mask       : Optional[FloatTensor]  = None,
        update_cross_attn : Optional[Dict] = None,
        style_block = None,
    ) -> FloatTensor:
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(6, dim=-1)

        img_norm = self.norm1_i(img)  
        img_norm = style_block.img(img_norm, "attn_norm")        #
        
        img_norm = img_norm * (1+img_msa_scale) + img_msa_shift
        img_norm = style_block.img(img_norm, "attn_norm_mod")    #

        img_attn = self.attn1(img_norm, img_masks, rope=rope, mask=mask, style_block=style_block)
        img_attn = style_block.img(img_attn, "attn")

        img_attn *= img_msa_gate
        img_attn = style_block.img(img_attn, "attn_gated")

        img += img_attn
        img = style_block.img(img, "attn_res")

        img_norm = self.norm3_i(img)
        img_norm = style_block.img(img_norm, "ff_norm")
        
        img_norm = img_norm * (1+img_mlp_scale) + img_mlp_shift
        img_norm = style_block.img(img_norm, "ff_norm_mod")

        img_ff_i = self.ff_i(img_norm, style_block.img.FF)
        img_ff_i = style_block.img(img_ff_i, "ff")            # fused... "ff" + "attn"
        
        img_ff_i *= img_mlp_gate
        img_ff_i = style_block.img(img_ff_i, "ff_gated")         # 

        img += img_ff_i
        img = style_block.img(img, "ff_res")       # 

        return img


#########################################################################################################################################################################
class HDModel(nn.Module):
    CHANNELS   = 2560
    RECON_MODE = False

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

        self.final_layer = HDLastLayer(self.inner_dim, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)

        caption_channels   = [caption_channels[1], ] * (num_layers + num_single_layers) + [caption_channels[0], ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim, dtype=dtype, device=device, operations=operations))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq            = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

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
        encoder_hidden_states_llama3 = None,  # 1,32,143,4096
        image_cond                   = None,  # HiDream E1
        control                      = None,
        transformer_options          = {},
        mask    : Optional[Tensor]   = None,
    ) -> Tensor:
        x_orig      = x.clone()
        b, c, h, w  = x.shape
        if image_cond is not None: # HiDream E1
            x = torch.cat([x, image_cond], dim=-1)
        h_len = ((h + (self.patch_size // 2)) // self.patch_size) # h_len 96
        w_len = ((w + (self.patch_size // 2)) // self.patch_size) # w_len 96
        img_len = h_len * w_len
        txt_slice = slice(img_len, None)
        img_slice = slice(None, img_len)
        SIGMA = t[0].clone() / 1000
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        if EO("zero_heads"):
            HEADS = 0
        else:
            HEADS = 20

        StyleMMDiT = transformer_options.get('StyleMMDiT', StyleMMDiT_Model())        
        StyleMMDiT.set_len(h_len, w_len, img_slice, txt_slice, HEADS=HEADS)
        StyleMMDiT.Retrojector = self.Retrojector if hasattr(self, "Retrojector") else None
        transformer_options['StyleMMDiT'] = None

        x_tmp = transformer_options.get("x_tmp")
        if x_tmp is not None:
            x_tmp = x_tmp.expand(x.shape[0], -1, -1, -1).clone()
            img = comfy.ldm.common_dit.pad_to_patch_size(x_tmp, (self.patch_size, self.patch_size))
        else:
            img = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        
        y0_style, img_y0_style = None, None

        img_orig, t_orig, y_orig, context_orig, llama3_orig = clone_inputs(img, t, y, context, encoder_hidden_states_llama3)
    
        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        update_cross_attn = transformer_options.get("update_cross_attn")
    
        z_ = transformer_options.get("z_")   # initial noise and/or image+noise from start of rk_sampler_beta() 
        rk_row = transformer_options.get("row") # for "smart noise"
        if z_ is not None:
            x_init = z_[rk_row].to(x)
        elif 'x_init' in transformer_options:
            x_init = transformer_options.get('x_init').to(x)

        # recon loop to extract exact noise pred for scattersort guide assembly
        HDModel.RECON_MODE = StyleMMDiT.noise_mode == "recon"
        recon_iterations = 2 if StyleMMDiT.noise_mode == "recon" else 1
        for recon_iter in range(recon_iterations):
            y0_style = StyleMMDiT.guides
            y0_style_active = True if type(y0_style) == torch.Tensor else False
            
            HDModel.RECON_MODE = True     if StyleMMDiT.noise_mode == "recon" and recon_iter == 0     else False
            
            if StyleMMDiT.noise_mode == "recon" and recon_iter == 1:
                x_recon = x_tmp if x_tmp is not None else x_orig
                noise_prediction = x_recon + (1-SIGMA.to(x_recon)) * eps.to(x_recon)
                denoised = x_recon - SIGMA.to(x_recon) * eps.to(x_recon)
                
                denoised = StyleMMDiT.apply_recon_lure(denoised, y0_style)

                new_x = (1-SIGMA.to(denoised)) * denoised + SIGMA.to(denoised) * noise_prediction
                img_orig = img = comfy.ldm.common_dit.pad_to_patch_size(new_x, (self.patch_size, self.patch_size))
                
                x_init = noise_prediction
            elif StyleMMDiT.noise_mode == "bonanza":
                x_init = torch.randn_like(x_init)

            if y0_style_active:
                SIGMA_ADAIN         = (SIGMA * EO("eps_adain_sigma_factor", 1.0)).to(y0_style)
                y0_style_noised     = (1-SIGMA_ADAIN) * y0_style + SIGMA_ADAIN * x_init[0:1].to(y0_style)   #always only use first batch of noise to avoid broadcasting
                img_y0_style_orig   = comfy.ldm.common_dit.pad_to_patch_size(y0_style_noised, (self.patch_size, self.patch_size))

            mask_zero = None
            
            out_list = []
            for cond_iter in range(len(transformer_options['cond_or_uncond'])):
                UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
                
                if update_cross_attn is not None:
                    update_cross_attn['UNCOND'] = UNCOND

                bsz_style = y0_style.shape[0] if y0_style_active else 0
                bsz       = 1 if HDModel.RECON_MODE else bsz_style + 1

                img, t, y, context, llama3 = clone_inputs(img_orig, t_orig, y_orig, context_orig, llama3_orig, index=cond_iter)
                
                mask = None
                if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
                    AttnMask = transformer_options['AttnMask']
                    mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)
                        #img_len = transformer_options['AttnMask'].img_len
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
                        #img_len = transformer_options['AttnMask'].img_len
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

                if y0_style_active and not HDModel.RECON_MODE:
                    if mask is None:
                        context, y, llama3 = StyleMMDiT.apply_style_conditioning(
                            UNCOND = UNCOND,
                            base_context       = context,
                            base_y             = y,
                            base_llama3        = llama3,
                        )
                    else:
                        context = context.repeat(bsz_style + 1, 1, 1)
                        y = y.repeat(bsz_style + 1, 1)                   if y      is not None else None
                        llama3  =  llama3.repeat(bsz_style + 1, 1, 1, 1) if llama3 is not None else None
                    img_y0_style = img_y0_style_orig.clone()

                if mask is not None and not type(mask[0][0].item()) == bool:
                    mask = mask.to(x.dtype)
                if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
                    mask_zero = mask_zero.to(x.dtype)

                # prep embeds
                t    = self.expand_timesteps(t, bsz, x.device)
                t    = self.t_embedder      (t,      x.dtype)
                clip = t + self.p_embedder(y)
                
        
                x_embedder_dtype = self.x_embedder.proj.weight.data.dtype
                if x_embedder_dtype not in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
                    x_embedder_dtype = x.dtype
                
                img_sizes = None
                img, img_masks, img_sizes = self.patchify(img, self.max_seq, img_sizes)   # for 1024x1024: output is   1,4096,64   None   [[64,64]]     hidden_states rearranged not shrunk, patch_size 1x1???
                if img_masks is None:
                    pH, pW          = img_sizes[0]
                    img_ids         = torch.zeros(pH, pW, 3, device=img.device)
                    img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=img.device)[:, None]
                    img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=img.device)[None, :]
                    img_ids         = repeat(img_ids, "h w c -> b (h w) c", b=bsz)
                img = self.x_embedder(img.to(x_embedder_dtype))
                #img_len = img.shape[-2]

                if y0_style_active and not HDModel.RECON_MODE:
                    img_y0_style, _, _ = self.patchify(img_y0_style_orig.clone(), self.max_seq, None)   # for 1024x1024: output is   1,4096,64   None   [[64,64]]     hidden_states rearranged not shrunk, patch_size 1x1???
                    img_y0_style = self.x_embedder(img_y0_style.to(x_embedder_dtype))  # hidden_states 1,4032,2560         for 1024x1024: -> 1,4096,2560      ,64 -> ,2560 (x40)
                    img = torch.cat([img, img_y0_style], dim=0)

                contexts = self.prepare_contexts(llama3, context, bsz, img.shape[-1])

                # txt_ids -> 1,414,3
                txt_ids = torch.zeros(bsz,   contexts[-1].shape[1] + contexts[-2].shape[1] + contexts[0].shape[1],     3,    device=img_ids.device, dtype=img_ids.dtype)
                ids     = torch.cat((img_ids, txt_ids), dim=-2)   # ids -> 1,4446,3
                rope    = self.pe_embedder(ids)                  # rope -> 1, 4446, 1, 64, 2, 2

                txt_init     = torch.cat([contexts[-1], contexts[-2]], dim=-2)     # shape[1] == 128, 143       then on another step/call it's 128, 128...??? cuz the contexts is now 1,128,2560
                txt_init_len = txt_init.shape[-2]                                       # 271

                if mask is not None:
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
                    
                    txt_init = torch.cat(txt_init_list, dim=-2)  #T5,LLAMA3 (last block)
                    txt_init_len = txt_init.shape[-2]
                    
                img = StyleMMDiT(img, "proj_in")
                
                img = img.to(x) if img is not None else None
                
                # DOUBLE STREAM
                for bid, (block, style_block) in enumerate(zip(self.double_stream_blocks, StyleMMDiT.double_blocks)):
                    txt_llama = contexts[bid]
                    txt = torch.cat([txt_init, txt_llama], dim=-2)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

                    if   weight > 0 and mask is not None and     weight  <      bid/48:
                        img, txt_init = block(img, img_masks, txt, clip, rope, mask_zero, style_block=style_block)
                        
                    elif (weight < 0 and mask is not None and abs(weight) < (1 - bid/48)):
                        img_tmpZ, txt_tmpZ = img.clone(), txt.clone()

                        # more efficient than the commented lines below being used instead in the loop?
                        img_tmpZ, txt_init = block(img_tmpZ, img_masks, txt_tmpZ, clip, rope, mask, style_block=style_block)
                        img     , txt_tmpZ = block(img     , img_masks, txt     , clip, rope, mask_zero, style_block=style_block)
                        
                    elif floor > 0 and mask is not None and     floor  >      bid/48:
                        mask_tmp = mask.clone()
                        mask_tmp[:img_len,:img_len] = 1.0
                        img, txt_init = block(img, img_masks, txt, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif floor < 0 and mask is not None and abs(floor) > (1 - bid/48):
                        mask_tmp = mask.clone()
                        mask_tmp[:img_len,:img_len] = 1.0
                        img, txt_init = block(img, img_masks, txt, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif update_cross_attn is not None and update_cross_attn['skip_cross_attn']:
                        img, txt_init = block(img, img_masks, txt, clip, rope, mask, update_cross_attn=update_cross_attn)
                        
                    else:
                        img, txt_init = block(img, img_masks, txt, clip, rope, mask, update_cross_attn=update_cross_attn, style_block=style_block)

                    txt_init = txt_init[..., :txt_init_len, :]
                # END DOUBLE STREAM





                img       = torch.cat([img, txt_init], dim=-2)   # 4032 + 271 -> 4303     # txt embed from double stream block
                joint_len = img.shape[-2]
                
                if img_masks is not None:
                    img_masks_ones = torch.ones( (bsz, txt_init.shape[-2] + txt_llama.shape[-2]), device=img_masks.device, dtype=img_masks.dtype)   # encoder_attention_mask_ones=   padding for txt embed concatted onto end of img
                    img_masks      = torch.cat([img_masks, img_masks_ones], dim=-2)





                # SINGLE STREAM
                for bid, (block, style_block) in enumerate(zip(self.single_stream_blocks, StyleMMDiT.single_blocks)):
                    txt_llama = contexts[bid+16]                        # T5 pre-embedded for single stream blocks
                    img = torch.cat([img, txt_llama], dim=-2)            # cat img,txt     opposite of flux which is txt,img       4303 + 143 -> 4446

                    if   weight > 0 and mask is not None and     weight  <      (bid+16)/48:
                        img = block(img, img_masks, None, clip, rope, mask_zero, style_block=style_block)
                        
                    elif weight < 0 and mask is not None and abs(weight) < (1 - (bid+16)/48):
                        img = block(img, img_masks, None, clip, rope, mask_zero, style_block=style_block)
                    
                    elif floor > 0 and mask is not None and     floor  >      (bid+16)/48:
                        mask_tmp = mask.clone()
                        mask_tmp[:img_len,:img_len] = 1.0
                        img = block(img, img_masks, None, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif floor < 0 and mask is not None and abs(floor) > (1 - (bid+16)/48):
                        mask_tmp = mask.clone()
                        mask_tmp[:img_len,:img_len] = 1.0
                        img = block(img, img_masks, None, clip, rope, mask_tmp, style_block=style_block)
                        
                    else:
                        img = block(img, img_masks, None, clip, rope, mask, style_block=style_block)
                        
                    img = img[..., :joint_len, :]   # slice off txt_llama
                # END SINGLE STREAM
                    
                img = img[..., :img_len, :]
                #img = self.final_layer(img, clip)   # 4096,2560 -> 4096,64
                shift, scale = self.final_layer.adaLN_modulation(clip).chunk(2,dim=1)
                img = (1 + scale[:, None, :]) * self.final_layer.norm_final(img) + shift[:, None, :]
                if not EO("endojector"):
                    img = StyleMMDiT(img, "proj_out")

                if y0_style_active and not HDModel.RECON_MODE:
                    img = img[0:1]
                
                if EO("endojector"):
                    if EO("dumb"):
                        eps_style = x_init[0:1].to(y0_style) - y0_style
                    else:
                        eps_style = (x_tmp[0:1].to(y0_style) - y0_style) / SIGMA.to(y0_style)
                    eps_embed = self.Endojector.embed(eps_style)
                    img = StyleMMDiT.scattersort_(img.to(eps_embed), eps_embed)
                
                img = self.final_layer.linear(img.to(self.final_layer.linear.weight.data))

                img = self.unpatchify(img, img_sizes)
                out_list.append(img)
                
            output = torch.cat(out_list, dim=0)
            eps = -output[:, :, :h, :w]
            
            if recon_iter == 1:
                denoised = new_x - SIGMA.to(new_x) * eps.to(new_x)
                if x_tmp is not None:
                    eps = (x_tmp - denoised.to(x_tmp)) / SIGMA.to(x_tmp)
                else:
                    eps = (x_orig - denoised.to(x_orig)) / SIGMA.to(x_orig)
                    












        freqsep_lowpass_method = transformer_options.get("freqsep_lowpass_method")
        freqsep_sigma          = transformer_options.get("freqsep_sigma")
        freqsep_kernel_size    = transformer_options.get("freqsep_kernel_size")
        freqsep_inner_kernel_size    = transformer_options.get("freqsep_inner_kernel_size")
        freqsep_stride    = transformer_options.get("freqsep_stride")
        
        freqsep_lowpass_weight = transformer_options.get("freqsep_lowpass_weight")
        freqsep_highpass_weight= transformer_options.get("freqsep_highpass_weight")
        freqsep_mask           = transformer_options.get("freqsep_mask")

        y0_style_pos = transformer_options.get("y0_style_pos")
        y0_style_neg = transformer_options.get("y0_style_neg")
        
        # end recon loop
        self.style_dtype = torch.float32 if self.style_dtype is None else self.style_dtype
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

            elif transformer_options['y0_style_method'] == "WCT2":
                self.WaveletStyleWCT.set(y0_adain_embed, h_len, w_len)
                denoised_embed = self.WaveletStyleWCT.get(denoised_embed, h_len, w_len)
                
                if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    y0_standard_guide_embed = self.Retrojector.embed(y0_standard_guide)
                    f_cs = self.WaveletStyleWCT.get(y0_standard_guide_embed, h_len, w_len)
                    self.y0_standard_guide = self.Retrojector.unembed(f_cs)

                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')

                    y0_inv_standard_guide_embed = self.Retrojector.embed(y0_inv_standard_guide)
                    f_cs = self.WaveletStyleWCT.get(y0_inv_standard_guide_embed, h_len, w_len)
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
            
            #eps = eps.float()
        
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

            elif transformer_options['y0_style_method'] == "WCT2":
                self.WaveletStyleWCT.set(y0_adain_embed, h_len, w_len)
                denoised_embed = self.WaveletStyleWCT.get(denoised_embed, h_len, w_len)

            denoised_approx = self.Retrojector.unembed(denoised_embed)

            if UNCOND:
                eps = (x - denoised_approx) / sigma
                eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            elif eps.shape[0] == 1 and not UNCOND:
                eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
            
            #eps = eps.float()
        
        if EO("model_eps_out"):
            self.eps_out = eps.clone()
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



class HDLastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear           = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: Tensor, vec: Tensor, modulation_dims=None) -> Tensor:
        x_dtype = x.dtype
        
        dtype = self.linear.weight.dtype
        if dtype not in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
            dtype = torch.float32
            self.linear.weight.data = self.linear.weight.data.to(dtype)
            self.linear.bias.data = self.linear.bias.data.to(dtype)
            self.adaLN_modulation[1].weight.data = self.adaLN_modulation[1].weight.data.to(dtype)
            self.adaLN_modulation[1].bias.data = self.adaLN_modulation[1].bias.data.to(dtype)
        
        x = x.to(dtype)
        vec = vec.to(dtype)
        if vec.ndim == 2:
            vec = vec[:, None, :]

        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = apply_mod(self.norm_final(x), (1 + scale), shift, modulation_dims)
        x = self.linear(x)
        return x #.to(x_dtype)

def apply_mod(tensor, m_mult, m_add=None, modulation_dims=None):
    if modulation_dims is None:
        if m_add is not None:
            return tensor * m_mult + m_add
        else:
            return tensor * m_mult
    else:
        for d in modulation_dims:
            tensor[:, d[0]:d[1]] *= m_mult[:, d[2]]
            if m_add is not None:
                tensor[:, d[0]:d[1]] += m_add[:, d[2]]
        return tensor





