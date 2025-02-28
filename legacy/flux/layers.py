# Adapted from: https://github.com/black-forest-labs/flux

import math
import torch
from torch import Tensor, nn

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from dataclasses import dataclass

from .math import attention, rope, apply_rope
import comfy.ldm.common_dit

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
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
        self.silu = nn.SiLU()
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
        head_dim = dim // num_heads   # 128 = 3072 / 24

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)    # dim is usually 3072


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, dtype=None, device=None, operations=None):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=True, dtype=dtype, device=device)

    def forward(self, vec: Tensor) -> tuple:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]),    ModulationOut(*out[3:]) if self.is_double else None,)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, dtype=None, device=None, operations=None, idx=-1):
        super().__init__()

        self.idx = idx

        mlp_hidden_dim   = int(hidden_size * mlp_ratio)
        self.num_heads   = num_heads
        self.hidden_size = hidden_size
        
        self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations) # in_features=3072, out_features=18432 (3072*6)
        self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations) # in_features=3072, out_features=18432 (3072*6)

        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations) # .qkv: in_features=3072, out_features=9216   .proj: 3072,3072
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations) # .qkv: in_features=3072, out_features=9216   .proj: 3072,3072

        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

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
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        return img_q, img_k, img_v
    
    def txt_attn_preproc(self, txt, txt_mod1):
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)    # Batch SeqLen (9216==3*3072) -> 3*1 24 SeqLen 128
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        return txt_q, txt_k, txt_v
    
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, timestep, transformer_options={}, mask=None, weight=1): # vec 1,3072

        img_mod1, img_mod2 = self.img_mod(vec) # -> 3072, 3072
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_q, img_k, img_v = self.img_attn_preproc(img, img_mod1)
        txt_q, txt_k, txt_v = self.txt_attn_preproc(txt, txt_mod1)

        q, k, v = torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), torch.cat((txt_v, img_v), dim=2)
        
        """if mask is None:
            attn = attention(q, k, v, pe=pe)
        else:
            attn_false = attention(q, k, v, pe=pe)
            attn = attention(q, k, v, pe=pe, mask=mask.to(torch.bool))
            attn = attn_false + weight * (attn - attn_false)"""
        
        #I = torch.eye(q.shape[-2], q.shape[-2], dtype=q.dtype, device=q.device).expand((1,1) + (-1, -1))
        #attn_map = attention_weights(q, k)
        """mask_resized = None
        if mask is not None:
            txt_a = txt[:,:,:]
            txt_qa, txt_ka, txt_va = self.txt_attn_preproc(txt_a, txt_mod1)
            
            txt_q_rope, txt_k_rope = apply_rope(txt_q, txt_k, pe[:,:,:512,:,:])
            img_q_rope, img_k_rope = apply_rope(img_q, img_k, pe[:,:,512:,:,:])

            attn_weights = attention_weights(txt_q_rope, img_k_rope)
            attn_weights = attn_weights.permute(0,1,3,2)
            attn_weights_slice = attn_weights[:,:,:,:]
            test = attn_weights_slice.mean(dim=1)
            test2 = rearrange(test, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=64, w=64, ph=1, pw=1)
            test3 = test2.mean(dim=1)
            mask_resized = F.interpolate(test3[None,:,:,:], size=(1024,1024), mode='bilinear', align_corners=False).squeeze(1)"""
            
        attn = attention(q, k, v, pe=pe, mask=mask)
        txt_attn = attn[:, :txt.shape[1]]                         # 1, 768,3072
        img_attn = attn[:,  txt.shape[1]:]  
        
        img += img_mod1.gate * self.img_attn.proj(img_attn)
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        
        img += img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt #, mask_resized



class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """
    def __init__(self, hidden_size: int,  num_heads: int, mlp_ratio: float = 4.0, qk_scale: float = None, dtype=None, device=None, operations=None, idx=-1):
        super().__init__()
        self.idx = idx
        self.hidden_dim = hidden_size #3072
        self.num_heads = num_heads    #24
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5   #0.08838834764831845

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)    #12288 == 3072 * 4
        # qkv and mlp_in
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size #3072
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations)
        
    def img_attn(self, img, mod, pe, mask, weight):
        img_mod = (1 + mod.scale) * self.pre_norm(img) + mod.shift   # mod => vec
        qkv, mlp = torch.split(self.linear1(img_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        """if mask is None:
            attn = attention(q, k, v, pe=pe)
        else:
            attn_false = attention(q, k, v, pe=pe)
            attn = attention(q, k, v, pe=pe, mask=mask.to(torch.bool))
            attn = attn_false + weight * (attn - attn_false)"""

        attn = attention(q, k, v, pe=pe, mask=mask)
        return attn, mlp

    # vec 1,3072    x 1,9984,3072
    def forward(self, img: Tensor, vec: Tensor, pe: Tensor, timestep, transformer_options={}, mask=None, weight=1) -> Tensor:   # x 1,9984,3072 if 2 reg embeds, 1,9472,3072 if none    # 9216x4096 = 16x1536x1536
        mod, _ = self.modulation(vec)
        attn, mlp = self.img_attn(img, mod, pe, mask, weight)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        img += mod.gate * output
        return img



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


