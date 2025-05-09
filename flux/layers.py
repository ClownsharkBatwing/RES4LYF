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
    gate:  Tensor

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


    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, mask=None, idx=0, update_cross_attn=None) -> Tuple[Tensor, Tensor]: # vec 1,3072  # vec 1,3072    #mask.shape 4608,4608  #img_attn.shape 1,4096,3072    txt_attn.shape 1,512,3072

        img_mod1, img_mod2  = self.img_mod(vec) # -> 3072, 3072
        txt_mod1, txt_mod2  = self.txt_mod(vec)
        
        img_q, img_k, img_v = self.img_attn_preproc(img, img_mod1)
        txt_q, txt_k, txt_v = self.txt_attn_preproc(txt, txt_mod1)

        q, k, v = torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        
        txt_attn = attn[:, : txt.shape[1]   ]                         # 1, 768,3072
        img_attn = attn[:,   txt.shape[1] : ]  
        
        img += img_mod1.gate * self.img_attn.proj(img_attn)
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        
        img += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        
        attn_mask = None
        
        if update_cross_attn is not None:
            if not update_cross_attn['skip_cross_attn']:
                UNCOND      = update_cross_attn['UNCOND']
                
                #txt_orig = txt.clone()
                txt_update = self.txt_norm1(txt.cpu()).float()
                txt_update = (1 + txt_mod1.scale.to(txt_update)) * txt_update + txt_mod1.shift.to(txt_update)
                
                if UNCOND:
                    t5_start    = update_cross_attn['src_t5_start']
                    t5_end      = update_cross_attn['src_t5_end']
                
                    txt_src    = txt_update[:,t5_start:t5_end,:].cpu() #.float()
                    self.c_src = txt_src.transpose(-2,-1).squeeze(0)    # shape [C,1]
                else:
                    t5_start    = update_cross_attn['tgt_t5_start']
                    t5_end      = update_cross_attn['tgt_t5_end']
                    
                    lamb  = update_cross_attn['lamb']
                    erase = update_cross_attn['erase']
                    
                    #txt_guide = txt_update[:,t5_start:t5_end,:].clone().float()
                    #c_guide   = txt_guide.transpose(-2,-1).squeeze(0)  # [C,1]
                    
                    c_guide   = txt_update[:,t5_start:t5_end,:].transpose(-2,-1).squeeze(0)  # [C,1]
                    
                    
                    Wv_old       = self.txt_attn.qkv.weight.data.to(c_guide)              # [C,C]
                    #Wk_old       = self.to_k_t.weight.data.float()              # [C,C]

                    v_star       = Wv_old @ c_guide                             # [C,1]
                    #k_star       = Wk_old @ c_guide                             # [C,1]

                    c_src        = self.c_src  #.cpu()                                   # [C,1]

                    lamb         = lamb
                    erase_scale  = erase
                    d            = c_src.shape[0]

                    C            = c_src @ c_src.T                              # [C,C]
                    I            = torch.eye(d, device=C.device, dtype=C.dtype)

                    mat1_v       = lamb*Wv_old + erase_scale*(v_star @ c_src.T)     # [C,C]
                    mat2_v       = lamb*I      + erase_scale*(C)                    # [C,C]
                    
                    #import gc
                    I       = I.to("cpu")
                    C       = C.to("cpu")
                    c_src   = c_src.to("cpu")
                    self.c_src   = self.c_src.to("cpu")
                    v_star  = v_star.to("cpu")
                    Wv_old  = Wv_old.to("cpu")
                    c_guide = c_guide.to("cpu")
                    del I, C, c_src, self.c_src, v_star, Wv_old, c_guide
                    #torch.cuda.empty_cache()
                    #gc.collect()
                    
                    #Wv_new       = mat1_v @ torch.inverse(mat2_v.float()).to(mat1_v)                   # [C,C]
                    #Wv_new = torch.linalg.solve(mat2_v, mat1_v)
                    Wv_new = torch.linalg.solve(mat2_v.T, mat1_v.T).T

                    mat1_v = mat1_v.to("cpu")
                    mat2_v = mat2_v.to("cpu")
                    del mat1_v, mat2_v
                    #del mat1_v, mat2_v, I, C, c_src, v_star, Wv_old, c_guide
                    
                    #torch.cuda.empty_cache()
                    #gc.collect()
                    #mat1_k       = lamb*Wk_old + erase_scale*(k_star @ c_src.T)     # [C,C]
                    #mat2_k       = lamb*I      + erase_scale*(C)                    # [C,C]
                    #Wk_new       = mat1_k @ torch.inverse(mat2_k)                   # [C,C]

                    #self.to_v_t.weight.data.copy_(Wv_new.to(self.to_v_t.weight.data.dtype))
                    #self.to_k_t.weight.data.copy_(Wk_new.to(self.to_k_t.weight.data.dtype))
                    
                    update_q = update_cross_attn['update_q']
                    update_k = update_cross_attn['update_k']
                    update_v = update_cross_attn['update_v']
                    
                    if not update_q:
                        Wv_new[:3072,    :] = self.txt_attn.qkv.weight.data[:3072,    :].to(Wv_new)
                    if not update_k:
                        Wv_new[3072:6144,:] = self.txt_attn.qkv.weight.data[3072:6144,:].to(Wv_new)
                    if not update_v:
                        Wv_new[6144:    ,:] = self.txt_attn.qkv.weight.data[6144:    ,:].to(Wv_new)
                    
                    self.txt_attn.qkv.weight.data.copy_(Wv_new.to(self.txt_attn.qkv.weight.data.dtype))
                    #self.txt_attn.qkv.weight.data = Wv_new.to(self.txt_attn.qkv.weight.data.dtype)
                    
                    Wv_new = Wv_new.to("cpu")
                    del Wv_new
                    #torch.cuda.empty_cache()
                    #gc.collect()
                    
        
        return img, txt, attn_mask
        

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
        
    def img_attn(self, img, mod, pe, mask):
        img_mod  = (1 + mod.scale) * self.pre_norm(img) + mod.shift   # mod => vec
        qkv, mlp = torch.split(self.linear1(img_mod), [3*self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v  = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k     = self.norm(q, k, v)

        attn     = attention(q, k, v, pe=pe, mask=mask)
        
        return attn, mlp

    # vec 1,3072    x 1,9984,3072
    def forward(self, img: Tensor, vec: Tensor, pe: Tensor, mask=None, idx=0) -> Tensor:   # x 1,9984,3072 if 2 reg embeds, 1,9472,3072 if none    # 9216x4096 = 16x1536x1536
        mod, _    = self.modulation(vec)
        
        attn, mlp = self.img_attn(img, mod, pe, mask)
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

    def forward_scale_shift(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return x

    def forward_linear(self, x: Tensor, vec: Tensor) -> Tensor:
        x = self.linear(x)
        return x






