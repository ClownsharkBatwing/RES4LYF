from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor

import einops
from einops import repeat

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
import torch.nn.functional as F

from comfy.ldm.flux.math import apply_rope, rope
from comfy.ldm.flux.layers import LastLayer

from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
import comfy.model_management
import comfy.ldm.common_dit


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
        img_masks : Optional[FloatTensor] = None,
        txt       : Optional[FloatTensor] = None,
        clip      :          FloatTensor  = None,
        rope      :          FloatTensor  = None,
        mask      : Optional[FloatTensor] = None,
    ) -> FloatTensor:
        return self.block(img, img_masks, txt, clip, rope, mask)



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
    ) -> Tensor:
        
        bsz = img.shape[0]

        img_q = self.q_rms_norm(self.to_q(img))
        img_k = self.k_rms_norm(self.to_k(img))
        img_v =                 self.to_v(img)

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

            txt_q   = txt_q.view(bsz, -1, self.heads, head_dim)
            txt_k   = txt_k.view(bsz, -1, self.heads, head_dim)
            txt_v   = txt_v.view(bsz, -1, self.heads, head_dim)

            img_len = img_q.shape[1]
            txt_len = txt_q.shape[1]
            
            attn    = attention(torch.cat([img_q, txt_q], dim=1), 
                                torch.cat([img_k, txt_k], dim=1), 
                                torch.cat([img_v, txt_v], dim=1), rope=rope, mask=mask)

            img_attn, txt_attn = torch.split(attn, [img_len, txt_len], dim=1)
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
            operations.Linear(dim, 12*dim, bias=True,                                                 dtype=dtype, device=device)
        )

        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,             dtype=dtype, device=device)
        self.norm1_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,             dtype=dtype, device=device)
        
        self.attn1     = HDAttention         (dim, heads, head_dim, single=False,                     dtype=dtype, device=device, operations=operations)

        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,             dtype=dtype, device=device)
        self.ff_i      = MOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts,  dtype=dtype, device=device, operations=operations)
                
        self.norm3_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,             dtype=dtype, device=device)                                 
        self.ff_t      =    FeedForwardSwiGLU(dim, 4*dim,                                             dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        img       :          FloatTensor,
        img_masks : Optional[FloatTensor] = None,
        txt       : Optional[FloatTensor] = None,
        clip      : Optional[FloatTensor] = None,    # clip = t + p_embedder (from pooled)
        rope      :          FloatTensor  = None,
        mask      : Optional[FloatTensor] = None,
    ) -> FloatTensor:
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate, \
        txt_msa_shift, txt_msa_scale, txt_msa_gate, txt_mlp_shift, txt_mlp_scale, txt_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(12, dim=-1)      # 1,1,2560           

        img_norm = self.norm1_i(img) * (1+img_msa_scale) + img_msa_shift
        txt_norm = self.norm1_t(txt) * (1+txt_msa_scale) + txt_msa_shift

        img_attn, txt_attn = self.attn1(img_norm, img_masks, txt_norm, rope=rope, mask=mask)
        
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
            operations.Linear(dim, 6 * dim, bias=True,                                                dtype=dtype, device=device)
        )

        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,             dtype=dtype, device=device)
        self.attn1     = HDAttention         (dim, heads, head_dim, single=True,                      dtype=dtype, device=device, operations=operations)

        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False,             dtype=dtype, device=device)
        self.ff_i      = MOEFeedForwardSwiGLU(dim, 4*dim, num_routed_experts, num_activated_experts,  dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        img       :          FloatTensor,
        img_masks : Optional[FloatTensor] = None,
        txt       : Optional[FloatTensor] = None,
        clip      : Optional[FloatTensor] = None,
        rope      :          FloatTensor  = None,
        mask      : Optional[FloatTensor] = None,

    ) -> FloatTensor:
        
        img_msa_shift, img_msa_scale, img_msa_gate, img_mlp_shift, img_mlp_scale, img_mlp_gate = self.adaLN_modulation(clip)[:,None].chunk(6, dim=-1)

        img_norm = self.norm1_i(img) * (1+img_msa_scale) + img_msa_shift
        
        img_attn = self.attn1(img_norm, img_masks, rope=rope, mask=mask)
        
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
        img          = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        img_orig, t_orig, y_orig, context_orig, llama3_orig = clone_inputs(img, t, y, context, encoder_hidden_states_llama3)
        
        weight    = transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = transformer_options.get("regional_conditioning_floor",  0.0)
        #floor     = min(floor, weight)

        out_list = []
        for cond_iter in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1

            img_sizes = None
            bsz = 1

            img, t, y, context, llama3 = clone_inputs(img_orig, t_orig, y_orig, context_orig, llama3_orig, index=cond_iter)
            
            mask = None
            if 'AttnMask' in transformer_options: # and weight != 0:
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')

                if False: #weight == 0:
                    mask_zero = torch.ones_like(mask)
                    img_len = transformer_options['AttnMask'].img_len
                    mask_zero[img_len:, img_len:] = mask[img_len:, img_len:]
                    mask = mask_zero
                if weight == 0 and not UNCOND:
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    context = context.view(128, -1, context.shape[-1]).sum(dim=-2)
                    llama3  = transformer_options['RegContext'].llama3 .to(llama3 .dtype).to(llama3 .device)
                    mask = None
                elif weight == 0 and UNCOND:
                    mask = None
                    pass

                elif not UNCOND:
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    llama3  = transformer_options['RegContext'].llama3 .to(llama3 .dtype).to(llama3 .device)
                
                else:
                    A       = context
                    B       = transformer_options['RegContext'].context
                    context = A.repeat(1,    (B.shape[1] // A.shape[1]) + 1, 1)[:,   :B.shape[1], :]
                    
                    A       = llama3
                    B       = transformer_options['RegContext'].llama3
                    llama3  = A.repeat(1, 1, (B.shape[2] // A.shape[2]) + 1, 1)[:,:, :B.shape[2], :]
            elif self.manual_mask is not None:
                mask = self.manual_mask

            # prep embeds
            t    = self.expand_timesteps(t, bsz, img.device)
            t    = self.t_embedder      (t,      img.dtype)
            clip = t + self.p_embedder(y)

            img, img_masks, img_sizes = self.patchify(img, self.max_seq, img_sizes)   # for 1024x1024: output is   1,4096,64   None   [[64,64]]     hidden_states rearranged not shrunk, patch_size 1x1???
            if img_masks is None:
                pH, pW          = img_sizes[0]
                img_ids         = torch.zeros(pH, pW, 3, device=img.device)
                img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=img.device)[:, None]
                img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=img.device)[None, :]
                img_ids         = repeat(img_ids, "h w c -> b (h w) c", b=bsz)
            img = self.x_embedder(img)  # hidden_states 1,4032,2560         for 1024x1024: -> 1,4096,2560      ,64 -> ,2560 (x40)
            
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
            
            for bid, block in enumerate(self.double_stream_blocks):                                                              # len == 16
                txt_llama = contexts[bid]
                txt       = torch.cat([txt_init, txt_llama], dim=1)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

                #txt_llama_orig = contexts_orig[bid]
                #txt_orig       = torch.cat([txt_init_orig, txt_llama_orig], dim=1)        # 1,384,2560       # cur_contexts = T5, LLAMA3 (last block), LLAMA3 (current block)

                if mask is not None:
                    if floor > 0 and floor > bid/48:
                        mask[:img_len,:img_len] = 1.0
                    elif weight > 0 and weight < bid/48:
                        mask = None
                
                if weight < 0 and mask is not None and abs(weight) < (1 - bid/48):
                    img, txt_init = block(img, img_masks, txt, clip, rope, None)
                elif floor < 0 and mask is not None and abs(floor) > (1 - bid/48):
                    mask_tmp = mask.clone()
                    mask_tmp[:img_len,:img_len] = 1.0
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask_tmp)
                else:
                    img, txt_init = block(img, img_masks, txt, clip, rope, mask)
                    
                txt_init = txt_init[:, :txt_init_len]
                

            img_len = img.shape[1]
            img     = torch.cat([img, txt_init], dim=1)   # 4032 + 271 -> 4303     # txt embed from double stream block
            joint_len = img.shape[1]
            
            if img_masks is not None:
                img_masks_ones = torch.ones( (bsz, txt_init.shape[1] + txt_llama.shape[1]), device=img_masks.device, dtype=img_masks.dtype)   # encoder_attention_mask_ones=   padding for txt embed concatted onto end of img
                img_masks      = torch.cat([img_masks, img_masks_ones], dim=1)
            
            # SINGLE STREAM
            for bid, block in enumerate(self.single_stream_blocks): # len == 32
                txt_llama = contexts[bid+16]                        # T5 pre-embedded for single stream blocks
                img = torch.cat([img, txt_llama], dim=1)            # cat img,txt     opposite of flux which is txt,img       4303 + 143 -> 4446

                if mask is not None:
                    if floor > 0 and floor > (bid+16)/48:
                        mask[:img_len,:img_len] = 1.0
                    elif weight > 0 and weight < (bid+16)/48:
                        mask = None
                
                if weight < 0 and mask is not None and abs(weight) < (1 - (bid+16)/48):
                    img = block(img, img_masks, None, clip, rope, None)
                elif floor < 0 and mask is not None and abs(floor) > (1 - (bid+16)/48):
                    mask_tmp = mask.clone()
                    mask_tmp[:img_len,:img_len] = 1.0
                    img = block(img, img_masks, None, clip, rope, mask_tmp)
                else:
                    img = block(img, img_masks, None, clip, rope, mask)
                
                img = img[:, :joint_len]   # slice off txt_llama

            img = img[:, :img_len, ...]
            img = self.final_layer(img, clip)
            img = self.unpatchify (img, img_sizes)
            
            out_list.append(img)
            
        output = torch.stack(out_list, dim=0).squeeze(dim=1)
        
        return -output[:, :, :h, :w]




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


