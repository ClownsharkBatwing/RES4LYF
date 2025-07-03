# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Optional, Callable, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.diffusionmodules.mmdit import RMSNorm
import comfy.ldm.common_dit
import comfy.model_management

from ..latents import interpolate_spd
from ..helper  import ExtraOptions


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half     = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x



class ReWanRawSelfAttention(nn.Module):

    def __init__(self,
                dim,
                num_heads,
                window_size        = (-1, -1),
                qk_norm            = True,
                eps                = 1e-6, 
                operation_settings = {}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.window_size = window_size
        self.qk_norm     = qk_norm
        self.eps         = eps

        # layers
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, freqs, mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q, k    = apply_rope(q, k, freqs)
        # q,k.shape = 2,14040,12,128      v.shape = 2,14040,1536

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            v,
            heads=self.num_heads,
        )

        x = self.o(x)
        return x


def attention_weights(q, k):
    # implementation of in-place softmax to reduce memory req
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores.div_(math.sqrt(q.size(-1)))
    torch.exp(scores, out=scores)
    summed = torch.sum(scores, dim=-1, keepdim=True)
    scores /= summed
    return scores.nan_to_num_(0.0, 65504., -65504.)





class ReWanSlidingSelfAttention(nn.Module):

    def __init__(self,
                dim,
                num_heads,
                window_size        = (-1, -1),
                qk_norm            = True,
                eps                = 1e-6, 
                operation_settings = {}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.window_size = window_size
        self.qk_norm     = qk_norm
        self.eps         = eps
        self.winderz     = 15
        self.winderz_type= "standard"

        # layers
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        

    def forward(self, x, freqs, mask=None, grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q, k    = apply_rope(q, k, freqs)
        # q,k.shape = 2,14040,12,128      v.shape = 2,14040,1536
    
        img_len = grid_sizes[1] * grid_sizes[2]
        total_frames = int(q.shape[1] // img_len)

        window_size = self.winderz
        half_window = window_size // 2

        q_ = q.view(b, s, n * d)
        k_ = k.view(b, s, n * d)
        x_list = []

        for i in range(total_frames):
            q_start =  i      * img_len
            q_end   = (i + 1) * img_len

            # circular frame indices for key/value window
            center = i
            #window_indices = [(center + offset) % total_frames for offset in range(-half_window, half_window + 1)]
            if self.winderz_type == "standard":
                start = max(0, center - half_window)
                end   = min(total_frames, center + half_window + 1)
                # Shift window if it would be too short
                if end - start < window_size:
                    if start == 0:
                        end = min(total_frames, start + window_size)
                    elif end == total_frames:
                        start = max(0, end - window_size)

                window_indices = list(range(start, end))
            elif self.winderz_type == "circular":
                window_indices = [(center + offset) % total_frames for offset in range(-half_window, half_window + 1)]
            
            # frame indices to token indices
            token_indices = []
            for frame in window_indices:
                start = frame * img_len
                token_indices.extend(range(start, start + img_len))

            token_indices = torch.tensor(token_indices, device=q.device)

            x = optimized_attention(
                q_[:, q_start:q_end, :],           # [B, img_len, C]
                k_.index_select(1, token_indices), # [B, window_size * img_len, C]
                v .index_select(1, token_indices),
                heads=self.num_heads,
            )

            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        del x_list, q, k, v, q_, k_

        x = self.o(x)
        return x




class ReWanT2VSlidingCrossAttention(ReWanSlidingSelfAttention):

    def forward(self, x, context, context_clip=None, mask=None, grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v =             self.v(context)

        img_len = grid_sizes[1] * grid_sizes[2]
        total_frames = int(q.shape[1] // img_len)

        window_size = self.winderz
        half_window = window_size // 2
        
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q_, k_ = q, k
        #q_ = q.view(b, s, n * d)
        #k_ = k.view(b, s, n * d)
        x_list = []

        for i in range(total_frames):
            q_start =  i      * img_len
            q_end   = (i + 1) * img_len

            # circular frame indices for key/value window
            center = i
            #window_indices = [(center + offset) % total_frames for offset in range(-half_window, half_window + 1)]
            if self.winderz_type == "standard":
                start = max(0, center - half_window)
                end   = min(total_frames, center + half_window + 1)
                # Shift window if it would be too short
                if end - start < window_size:
                    if start == 0:
                        end = min(total_frames, start + window_size)
                    elif end == total_frames:
                        start = max(0, end - window_size)

                window_indices = list(range(start, end))
            elif self.winderz_type == "circular":
                window_indices = [(center + offset) % total_frames for offset in range(-half_window, half_window + 1)]
            
            # frame indices to token indices
            token_indices = []
            for frame in window_indices:
                start = frame * img_len
                token_indices.extend(range(start, start + img_len))

            token_indices = torch.tensor(token_indices, device=q.device)

            x = optimized_attention(
                q_[:, q_start:q_end, :],           # [B, img_len, C]
                k_, #.index_select(1, token_indices), # [B, window_size * img_len, C]
                v , #.index_select(1, token_indices),
                heads=self.num_heads,
            )

            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        del x_list, q, k, v, q_, k_

        x = self.o(x)
        return x




class ReWanSelfAttention(nn.Module):

    def __init__(self,
                dim,
                num_heads,
                window_size        = (-1, -1),
                qk_norm            = True,
                eps                = 1e-6, 
                operation_settings = {}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.window_size = window_size
        self.qk_norm     = qk_norm
        self.eps         = eps

        # layers
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        

    def forward(self, x, freqs, mask=None, grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q, k    = apply_rope(q, k, freqs)
        # q,k.shape = 2,14040,12,128      v.shape = 2,14040,1536

        if mask is not None and mask.shape[-1] > 0:
            #dtype = mask.dtype if mask.dtype == torch.bool else q.dtype
            #txt_len = mask.shape[1] - mask.shape[0]
            x = attention_pytorch(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
                mask=mask#[:,txt_len:].to(dtype)
            )
        else:
            x = optimized_attention(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
            )

        x = self.o(x)
        return x


class ReWanT2VRawCrossAttention(ReWanSelfAttention):

    def forward(self, x, context, context_clip=None, mask=None, grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v =             self.v(context)

        x = optimized_attention(q, k, v, heads=self.num_heads, mask=None)

        x = self.o(x)
        return x


class ReWanT2VCrossAttention(ReWanSelfAttention):

    def forward(self, x, context, context_clip=None, mask=None, grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v =             self.v(context)
        #if mask is not None:
        #    num_repeats = q.shape[1] // mask.shape[0]
        #    mask = mask.repeat(num_repeats, 1)
        # compute attention    # x.shape 2,14040,1536     q.shape 2,14040,1536     k,v.shape = 2,512,1536       mask = 14040,512   num_heads=12
        if mask is not None: # and (mask.shape[-1] - mask.shape[-2]) == k.shape[-2]:  # need mask shape 11664,5120
            #dtype = mask.dtype if mask.dtype == torch.bool else q.dtype
            dtype = torch.bool
            x = attention_pytorch(q, k, v, heads=self.num_heads, mask=mask.to(q.device).bool())

            #x = attention_pytorch(q, k, v, heads=self.num_heads, mask=mask[:,:k.shape[-2]].to(q.device).bool())
        else:
            x = optimized_attention(q, k, v, heads=self.num_heads, mask=None)

        x = self.o(x)
        return x


class ReWanI2VCrossAttention(ReWanSelfAttention):   # image2video only

    def __init__(self,
                dim,
                num_heads,
                window_size=(-1, -1),
                qk_norm=True,
                eps=1e-6, operation_settings={}, ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)

        self.k_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, context, context_clip=None, mask=None, grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        """context_img = context[:, :257]
        context     = context[:, 257:]
        mask_clip = None"""
        
        context_img = context_clip
        
        mask_clip = None
        if mask is not None:
            mask_clip = F.interpolate(mask[None, None, ...].to(torch.float16), (mask.shape[0], 257 * mask.shape[1]//512), mode='nearest-exact').squeeze().to(mask.dtype)
            """mask_clip = []
            for i in range(mask.shape[-1]//512):
                mask_clip.append(mask[:,i*512:i*512 + 257])
            mask_clip = torch.cat(mask_clip, dim=-1)"""

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads, mask=mask_clip)
        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads, mask=mask)

        # output
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': ReWanT2VCrossAttention,
    'i2v_cross_attn': ReWanI2VCrossAttention,
}


class ReWanAttentionBlock(nn.Module):

    def __init__(self,
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                window_size        =  (-1, -1),
                qk_norm            =  True,
                cross_attn_norm    =  False,
                eps                =  1e-6, 
                operation_settings = {}):
        super().__init__()
        self.dim             = dim
        self.ffn_dim         = ffn_dim
        self.num_heads       = num_heads
        self.window_size     = window_size
        self.qk_norm         = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps             = eps

        # layers
        self.norm1     = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attn = ReWanSelfAttention(  dim, num_heads, window_size, qk_norm,
                                            eps, operation_settings=operation_settings)
        self.norm3     = operation_settings.get("operations").LayerNorm(
            dim, eps,
            elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if cross_attn_norm else nn.Identity()
        
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
                                                                        dim,
                                                                        num_heads,
                                                                        (-1, -1),
                                                                        qk_norm,
                                                                        eps, 
                                                                        operation_settings=operation_settings)
        
        self.norm2 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(dim, ffn_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operation_settings.get("operations").Linear(ffn_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(
        self,
        x,
        e,
        freqs,
        context,
        context_clip=None,
        self_mask=None,
        cross_mask=None,
        grid_sizes = None,
        #mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32
        # e = tuple with 6 elem, shape = 2,1,1536    # with length = 33 so 9 frames
        # self-attention

        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            freqs,
            grid_sizes=grid_sizes,
            mask=self_mask) # mask[:,txt_len:])

        x = x + y * e[2]

        # cross-attention & ffn   # x,y.shape 2,14040,1536   
        x = x + self.cross_attn(self.norm3(x), context, context_clip=context_clip, mask=cross_mask, grid_sizes=grid_sizes,) #mask[:,:txt_len])
        #print("before norm2 ", torch.cuda.memory_allocated() / 1024**3)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        #print("after norm2 ", torch.cuda.memory_allocated() / 1024**3)
        x = x + y * e[5]
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim        = dim
        self.out_dim    = out_dim
        self.patch_size = patch_size
        self.eps        = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.head = operation_settings.get("operations").Linear   (dim, out_dim,                       device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, operation_settings={}):
        super().__init__()

        self.proj = torch.nn.Sequential(
            operation_settings                 .get("operations").LayerNorm(in_dim,          device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), operation_settings.get("operations").Linear(in_dim, in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            torch.nn.GELU(), operation_settings.get("operations").Linear   (in_dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings                 .get("operations").LayerNorm(out_dim,         device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class ReWanModel(torch.nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                model_type      = 't2v',
                patch_size      = (1, 2, 2),
                text_len        = 512,
                in_dim          = 16,
                dim             = 2048,
                ffn_dim         = 8192,
                freq_dim        = 256,
                text_dim        = 4096,
                out_dim         = 16,
                num_heads       = 16,
                num_layers      = 32,
                window_size     = (-1, -1),
                qk_norm         = True,
                cross_attn_norm = True,
                eps             = 1e-6,
                image_model     = None,
                device          = None,
                dtype           = None,
                operations      = None,
                ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()
        self.dtype           = dtype
        operation_settings   = {"operations": operations, "device": device, "dtype": dtype}

        assert model_type in ['t2v', 'i2v']
        self.model_type      = model_type

        self.patch_size      = patch_size
        self.text_len        = text_len
        self.in_dim          = in_dim
        self.dim             = dim
        self.ffn_dim         = ffn_dim
        self.freq_dim        = freq_dim
        self.text_dim        = text_dim
        self.out_dim         = out_dim
        self.num_heads       = num_heads
        self.num_layers      = num_layers
        self.window_size     = window_size
        self.qk_norm         = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps             = eps

        # embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) #dtype=torch.float32)
        
        
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim,      device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.SiLU(), operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        
        self.blocks = nn.ModuleList([
            ReWanAttentionBlock(
                                cross_attn_type,
                                dim, 
                                ffn_dim, num_heads,
                                window_size,
                                qk_norm,
                                cross_attn_norm,
                                eps, 
                                operation_settings=operation_settings)
            
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, operation_settings=operation_settings)
        else:
            self.img_emb = None


    def invert_patch_embedding(self, z: torch.Tensor, original_shape: torch.Size, grid_sizes: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:

        import torch.nn.functional as F
        B, C_in, D, H, W = original_shape
        pD, pH, pW = self.patch_size
        sD, sH, sW = pD, pH, pW

        if z.ndim == 3:
            # [B, S, C_out] -> reshape to [B, C_out, D', H', W']
            S = z.shape[1]
            if grid_sizes is None:
                Dp = D // pD
                Hp = H // pH
                Wp = W // pW
            else:
                Dp, Hp, Wp = grid_sizes
            C_out = z.shape[2]
            z = z.transpose(1, 2).reshape(B, C_out, Dp, Hp, Wp)
        else:
            B2, C_out, Dp, Hp, Wp = z.shape
            assert B2 == B, "Batch size mismatch... ya sharked it."

        # kncokout bias
        b = self.patch_embedding.bias.view(1, C_out, 1, 1, 1)
        z_nobias = z - b

        # 2D filter -> pinv
        w3 = self.patch_embedding.weight         # [C_out, C_in, 1, pH, pW]
        w2 = w3.squeeze(2)                       # [C_out, C_in, pH, pW]
        out_ch, in_ch, kH, kW = w2.shape
        W_flat = w2.view(out_ch, -1)            # [C_out, in_ch*pH*pW]
        W_pinv = torch.linalg.pinv(W_flat)      # [in_ch*pH*pW, C_out]

        # merge depth for 2D unfold wackiness
        z2 = z_nobias.permute(0,2,1,3,4).reshape(B*Dp, C_out, Hp, Wp)

        # apply pinv ... get patch vectors
        z_flat    = z2.reshape(B*Dp, C_out, -1)  # [B*Dp, C_out, L]
        x_patches = W_pinv @ z_flat              # [B*Dp, in_ch*pH*pW, L]

        # fold -> spatial frames
        x2 = F.fold(
            x_patches,
            output_size=(H, W),
            kernel_size=(pH, pW),
            stride=(sH, sW)
        )  # → [B*Dp, C_in, H, W]

        # un-merge depth
        x2 = x2.reshape(B, Dp, in_ch, H, W)           # [B, Dp,  C_in, H, W]
        x_recon = x2.permute(0,2,1,3,4).contiguous()  # [B, C_in,   D, H, W]
        return x_recon


    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea = None,
        freqs    = None,
        transformer_options = {},
        UNCOND = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        
        
        """trash = x[:,16:,...]
        x_slice_flip = torch.cat([x[:,:16,...], torch.flip(trash, dims=[2])], dim=1)
        x_slice_flip = self.patch_embedding(x_slice_flip.float()).to(x.dtype) 
        x          = self.patch_embedding(x.float()).to(x.dtype)  
        x = torch.cat([x[:,:,:9,...], x_slice_flip[:,:,9:,...]], dim=2)"""
        
        """x1 = self.patch_embedding(x[:,:,:8,...].float()).to(x.dtype)
        
        x_slice = torch.cat([x[:,:16,8:,...], trash[:,:,0:9, ...]], dim=1)
        
        x2          = self.patch_embedding(x_slice.float()).to(x.dtype) 
        
        x = torch.cat([x1, x2], dim=2)"""
        

        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")
        SIGMA = t[0].clone() / 1000
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        
        # embeddings
        #self.patch_embedding.to(self.time_embedding[0].weight.dtype)
        x_orig     = x.clone()
        #x          = self.patch_embedding(x.float()).to(self.time_embedding[0].weight.dtype)     #next line to torch.Size([1, 5120, 17, 30, 30]) from 1,36,17,30,30
        x          = self.patch_embedding(x.float()).to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
        grid_sizes = x.shape[2:]
        x          = x.flatten(2).transpose(1, 2)      # x.shape 1,32400,5120  bfloat16   316.4 MB

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))              # e0.shape = 2,6,1536       tiny ( < 0.1 MB)

        # context
        context = self.text_embedding(context)

        context_clip = None
        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            #context      = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e       = e0,
            freqs   = freqs,              # 1,32400,1,64,2,2 bfloat16 15.8 MB
            context = context,            # 1,1536,5120      bfloat16 15.0 MB
            context_clip = context_clip,
            grid_sizes = grid_sizes)





        weight    = transformer_options['reg_cond_weight'] if 'reg_cond_weight' in transformer_options else 0.0
        floor     = transformer_options['reg_cond_floor']  if 'reg_cond_floor'  in transformer_options else 0.0
        
        floor     = min(floor, weight)
        
        if type(weight) == float or type(weight) == int:
            pass
        else:
            weight = weight.item()
        
        AttnMask = transformer_options.get('AttnMask')    # somewhere around here, jumped to 20.6GB
        mask     = None
        if AttnMask is not None and weight > 0:
            mask                      = AttnMask.get(weight=weight) #mask_obj[0](transformer_options, weight.item())         # 32400,33936  bool   1048.6 MB
            
            mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False
            if not mask_type_bool:
                mask = mask.to(x.dtype)
            
            #text_len                  = context.shape[1] # mask_obj[0].text_len
            
            #mask[text_len:,text_len:] = torch.clamp(mask[text_len:,text_len:], min=floor.to(mask.device))   #ORIGINAL SELF-ATTN REGION BLEED
            #reg_cond_mask = reg_cond_mask_expanded.unsqueeze(0).clone() if reg_cond_mask_expanded is not None else None
        
        mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False




        txt_len = context.shape[1] # mask_obj[0].text_len
        #txt_len = mask.shape[-1] - mask.shape[-2] if mask is not None else "Unlogic Condition"          #what's the point of this?
        
        #self_attn_mask  = mask[:, txt_len:]
        #cross_attn_mask = mask[:,:txt_len ].bool()
        #i = 0
        #for block in self.blocks:
        for i, block in enumerate(self.blocks):
            if mask_type_bool and weight < (i / (len(self.blocks)-1)) and mask is not None:
                mask = mask.to(x.dtype)
            
            #if mask_type_bool and weight < (i / (len(self.blocks)-1)) and mask is not None:
            #    mask = mask.to(x.dtype)
            
            if mask is not None:
                #if True:
                #    x = block(x, self_mask=None, cross_mask=mask.bool(), **kwargs)
                if mask_type_bool and floor < 0 and          (i / (len(self.blocks)-1)) < (-floor):    # use self-attn mask until block number
                    x = block(x, self_mask=mask[:,txt_len:], cross_mask=mask[:,:txt_len].bool(), **kwargs)
                elif mask_type_bool and floor > 0 and  floor < (i / (len(self.blocks)-1)):               # use self-attn mask after block number
                    x = block(x, self_mask=mask[:,txt_len:], cross_mask=mask[:,:txt_len].bool(), **kwargs)
                    #x = block(x, self_mask=None, cross_mask=mask[:,:txt_len].bool(), **kwargs)
                elif floor == 0:
                    x = block(x, self_mask=mask[:,txt_len:], cross_mask=mask[:,:txt_len].bool(), **kwargs)
                else:
                    #x = block(x, self_mask=mask[:,txt_len:], cross_mask=mask[:,:txt_len].bool(), **kwargs)
                    x = block(x, self_mask=None, cross_mask=mask[:,:txt_len].bool(), **kwargs)
            
            else:
                x = block(x, **kwargs)
            #x = block(x, mask=mask, **kwargs)
            
            #i += 1

        # head
        x = self.head(x, e)

        # unpatchify
        eps = self.unpatchify(x, grid_sizes)
        
        
        
        
        
        
        
        
        
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1 and not UNCOND):
        if y0_style_pos is not None:
            y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            
            y0_style_pos = y0_style_pos.to(torch.float32)
            x   = x_orig.clone().to(torch.float32)
            eps = eps.to(torch.float32)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps


            img = comfy.ldm.common_dit.pad_to_patch_size(denoised, self.patch_size)
            patch_size = self.patch_size

            denoised_embed          = self.patch_embedding(img.float()) #.to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
            grid_sizes = denoised_embed.shape[2:]
            denoised_embed          = denoised_embed.flatten(2).transpose(1, 2) 


            img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_pos, self.patch_size)
            patch_size = self.patch_size

            y0_adain_embed          = self.patch_embedding(img_y0_adain.float()) #.to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
            grid_sizes = y0_adain_embed.shape[2:]
            y0_adain_embed          = y0_adain_embed.flatten(2).transpose(1, 2) 


            if transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    #denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    denoised_embed = self.invert_patch_embedding(denoised_embed, x_orig.shape, grid_sizes)
                    denoised_embed = self.patch_embedding(denoised_embed.float()) #.to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
                    grid_sizes     = denoised_embed.shape[2:]
                    denoised_embed = denoised_embed.flatten(2).transpose(1, 2) 
                    
                    #denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
                    
                    
                    
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

            denoised_approx = self.invert_patch_embedding(denoised_embed, x_orig.shape, grid_sizes)
            
            denoised_approx = denoised_approx.to(eps)

            eps = (x - denoised_approx) / sigma
            #if eps.shape[0] == 2:
            #    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
            #    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
            #else:
            #    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
            
            if not UNCOND:
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
            elif eps.shape[0] == 1 and UNCOND:
                eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
            
            eps = eps.float()
        
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1 and UNCOND):
        if y0_style_neg is not None:
            y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight")
            y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight")
            y0_style_neg_synweight *= y0_style_neg_weight
            
            y0_style_neg = y0_style_neg.to(torch.float32)
            x   = x_orig.clone().to(torch.float32)
            eps = eps.to(torch.float32)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps


            img = comfy.ldm.common_dit.pad_to_patch_size(denoised, self.patch_size)
            patch_size = self.patch_size

            denoised_embed          = self.patch_embedding(img.float()) #.to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
            grid_sizes = denoised_embed.shape[2:]
            denoised_embed          = denoised_embed.flatten(2).transpose(1, 2) 


            img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_neg, self.patch_size)
            patch_size = self.patch_size

            y0_adain_embed          = self.patch_embedding(img_y0_adain.float()) #.to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
            grid_sizes = y0_adain_embed.shape[2:]
            y0_adain_embed          = y0_adain_embed.flatten(2).transpose(1, 2) 


            if transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    #denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    denoised_embed = self.invert_patch_embedding(denoised_embed, x_orig.shape, grid_sizes)
                    denoised_embed = self.patch_embedding(denoised_embed.float()) #.to(x.dtype)         # vram jumped from ~16-16.5 up to 17.98     gained 300mb with weights at torch.float8_e4m3fn
                    grid_sizes = denoised_embed.shape[2:]
                    denoised_embed = denoised_embed.flatten(2).transpose(1, 2)                         
                    
                    #denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
                    
                    
                    
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

            denoised_approx = self.invert_patch_embedding(denoised_embed, x_orig.shape, grid_sizes)
            
            denoised_approx = denoised_approx.to(eps)

            #eps = (x - denoised_approx) / sigma
            #eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
            #if eps.shape[0] == 2:
            #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                
            if UNCOND:
                eps = (x - denoised_approx) / sigma
                eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            elif eps.shape[0] == 1 and not UNCOND:
                eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
            
            eps = eps.float()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return eps
    
    
    
    
    
    
    # context.shape = 2,512,1536     x.shape = 2,14040,1536      timestep.shape      h_len=30, w_len=52    30 * 52 = 1560
    def forward(self, x, timestep, context, clip_fea=None, transformer_options={}, **kwargs):
        
        """if False: #clip_fea is not None:
            bs, c, t, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
            patch_size = self.patch_size    # tuple = 1,2,2,
            
            t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
            h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
            w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
            
            img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
            # 14040 = 9 * 1560       1560 = 1536 + 24  1560/24 = 65
            freqs = self.rope_embedder(img_ids).movedim(1, 2)
            return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs)[:, :, :t, :h, :w]"""
            
        
        
        
        #x = torch.cat([x[:,:,:8,...],   torch.flip(x[:,:,8:,...], dims=[2])], dim=2)
        
        x_orig = x.clone()      # 1,16,36,60,60   bfloat16
        timestep_orig = timestep.clone() # 1000    float32
        context_orig = context.clone() # 1,512,4096 bfloat16
        
        
        out_list = []
        for i in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][i] == 1

            x = x_orig.clone()
            timestep = timestep_orig.clone()
            context = context_orig.clone()


            bs, c, t, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
            patch_size = self.patch_size
            


            transformer_options['original_shape'] = x.shape
            transformer_options['patch_size']     = patch_size
            

            """if UNCOND:
                transformer_options['reg_cond_weight'] = 0.0 # -1
                context_tmp = context[i][None,...].clone()"""
                
            if UNCOND:
                #transformer_options['reg_cond_weight'] = -1
                #context_tmp = context[i][None,...].clone()
                
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) #transformer_options['regional_conditioning_weight']
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor",  0.0) #transformer_options['regional_conditioning_floor'] #if "regional_conditioning_floor" in transformer_options else 0.0
                transformer_options['reg_cond_mask_orig'] = transformer_options.get('regional_conditioning_mask_orig')
                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] != 0.0:
                    AttnMask.attn_mask_recast(x.dtype)
                    context_tmp = RegContext.get().to(context.dtype)
                    clip_fea    = RegContext.get_clip_fea()
                    clip_fea     = clip_fea.to(x.dtype) if clip_fea else None
                    
                    A = context[i][None,...].clone()
                    B = context_tmp
                    context_tmp = A.repeat(1, (B.shape[1] // A.shape[1]) + 1, 1)[:, :B.shape[1], :]

                else:
                    context_tmp = context[i][None,...].clone()
            
            elif UNCOND == False:
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) #transformer_options['regional_conditioning_weight']
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor", 0.0) #transformer_options['regional_conditioning_floor'] #if "regional_conditioning_floor" in transformer_options else 0.0
                transformer_options['reg_cond_mask_orig'] = transformer_options.get('regional_conditioning_mask_orig')
                
                AttnMask   = transformer_options.get('AttnMask',   None)
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] != 0.0:
                    AttnMask.attn_mask_recast(x.dtype)
                    context_tmp  = RegContext.get()
                    clip_fea     = RegContext.get_clip_fea()
                    clip_fea     = clip_fea.to(x.dtype) if clip_fea else None
                else:
                    context_tmp = context[i][None,...].clone()

            if context_tmp is None:
                context_tmp = context[i][None,...].clone()
            context_tmp = context_tmp.to(context.dtype)
            
            
            
            t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
            h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
            w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
            
            img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
            # 14040 = 9 * 1560       1560 = 1536 + 24  1560/24 = 65
            freqs = self.rope_embedder(img_ids).movedim(1, 2).to(x.dtype)
            
            
            
            
            out_x = self.forward_orig(
                                        x          [i][None,...], 
                                        timestep   [i][None,...], 
                                        context_tmp,
                                        clip_fea            = clip_fea,
                                        freqs               = freqs[i][None,...],
                                        transformer_options = transformer_options,
                                        UNCOND              = UNCOND,
                                        )[:, :, :t, :h, :w]
        
            #out_x = torch.cat([out_x[:,:,:8,...],   torch.flip(out_x[:,:,8:,...], dims=[2])], dim=2)
            out_list.append(out_x)
        
        out_stack = torch.stack(out_list, dim=0).squeeze(dim=1)
        
        
        
        
        
        
        
        
        
        

        return out_stack
        

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [L, C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
    
        
        
        return u




def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)
    return content
