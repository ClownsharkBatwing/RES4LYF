#AuraFlow MMDiT
#Originally written by the AuraFlow Authors

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.attention import attention_pytorch

import comfy.ops
import comfy.ldm.common_dit

from ..helper import ExtraOptions

from typing import Dict, Optional, Tuple, List
from ..latents import slerp_tensor, interpolate_spd, tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch
from einops import rearrange
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module): # not executed directly with ReAura?
    def __init__(self, dim, hidden_dim=None, dtype=None, device=None, operations=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = operations.Linear(dim, n_hidden, bias=False, dtype=dtype, device=device)
        self.c_fc2 = operations.Linear(dim, n_hidden, bias=False, dtype=dtype, device=device)
        self.c_proj = operations.Linear(n_hidden, dim, bias=False, dtype=dtype, device=device)

    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5, dtype=None, device=None):
        # Copy pasta from https://github.com/huggingface/transformers/blob/e5f71ecaae50ea476d1e12351003790273c4b2ed/src/transformers/models/cohere/modeling_cohere.py#L78

        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")
    def forward(self, hidden_states):
        input_dtype   =  hidden_states.dtype
        hidden_states =  hidden_states.to(torch.float32)
        mean          =  hidden_states.mean(-1,                keepdim=True)
        variance      = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)

class ReSingleAttention(nn.Module):
    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # this is for cond
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")              # c = 1,4552,3072      #operations.Linear = torch.nn.Linear with recast
    def forward(self, c, mask=None):

        bsz, seqlen1, _ = c.shape

        q, k, v = self.w1q(c), self.w1k(c), self.w1v(c)
        q = q.view(bsz, seqlen1, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen1, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen1, self.n_heads, self.head_dim)
        q, k = self.q_norm1(q), self.k_norm1(k)

        output = attention_pytorch(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.n_heads, skip_reshape=True, mask=mask)
        c = self.w1o(output)
        return c



class ReDoubleAttention(nn.Module):
    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        super().__init__()

        self.n_heads  = n_heads
        self.head_dim = dim // n_heads

        # this is for cond   1 (one) not l (L)
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # this is for x
        self.w2q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

        self.q_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )


    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")         # c.shape 1,264,3072    x.shape 1,4032,3072   
    def forward(self, c, x, mask=None):

        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape

        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        cq         = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
        ck         = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cv         = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cq, ck     = self.q_norm1(cq), self.k_norm1(ck)

        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        xq         = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xk         = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xv         = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xq, xk     = self.q_norm2(xq), self.k_norm2(xk)

        # concat all     q,k,v.shape 1,4299,12,256           cq 1,267,12,256   xq 1,4032,12,256      self.n_heads 12      
        q, k, v = (
            torch.cat([cq, xq], dim=1),
            torch.cat([ck, xk], dim=1),
            torch.cat([cv, xv], dim=1),
        )
        # attn mask would be 4299,4299
        if mask is not None:
            pass
        
        output = attention_pytorch(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.n_heads, skip_reshape=True, mask=mask)

        c, x = output.split([seqlen1, seqlen2], dim=1)
        c    = self.w1o(c)
        x    = self.w2o(x)

        return c, x


class ReMMDiTBlock(nn.Module):
    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False, dtype=None, device=None, operations=None):
        super().__init__()

        self.normC1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.normC2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        if not is_last:
            self.mlpC = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
            self.modC = nn.Sequential(
                nn.SiLU(),
                operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device),
            )
        else:
            self.modC = nn.Sequential(
                nn.SiLU(),
                operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device),
            )

        self.normX1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.normX2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.mlpX = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
        self.modX = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device),
        )

        self.attn = ReDoubleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.is_last = is_last

    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")                    # MAIN BLOCK
    def forward(self, c, x, global_cond, mask=None, **kwargs):

        cres, xres = c, x

        cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = (
            self.modC(global_cond).chunk(6, dim=1)
        )

        c = modulate(self.normC1(c), cshift_msa, cscale_msa)

        # xpath
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = (
            self.modX(global_cond).chunk(6, dim=1)
        )

        x = modulate(self.normX1(x), xshift_msa, xscale_msa)

        # attention    c.shape 1,520,3072   x.shape 1,6144,3072
        c, x = self.attn(c, x, mask=mask)

        c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
        c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
        c = cres + c

        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
        x = xres + x

        return c, x

class ReDiTBlock(nn.Module):
    # like MMDiTBlock, but it only has X
    def __init__(self, dim, heads=8, global_conddim=1024, dtype=None, device=None, operations=None):
        super().__init__()

        self.norm1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)

        self.modCX = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device),
        )

        self.attn = ReSingleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)

    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")         # cx.shape 1,6664,3072   global_cond.shape 1,3072   mlpout.shape 1,6664,3072       float16
    def forward(self, cx, global_cond, mask=None, **kwargs):
        cxres = cx   
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modCX(
            global_cond
        ).chunk(6, dim=1)
        cx = modulate(self.norm1(cx), shift_msa, scale_msa)
        cx = self.attn(cx, mask=mask)
        cx = self.norm2(cxres + gate_msa.unsqueeze(1) * cx)
        mlpout = self.mlp(modulate(cx, shift_mlp, scale_mlp))
        cx = gate_mlp.unsqueeze(1) * mlpout

        cx = cxres + cx    # residual connection

        return cx



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    #@torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")
    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ReMMDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        dim=3072,
        n_layers=36,
        n_double_layers=4,
        n_heads=12,
        global_conddim=3072,
        cond_seq_dim=2048,
        max_seq=32 * 32,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype

        self.t_embedder = TimestepEmbedder(global_conddim, dtype=dtype, device=device, operations=operations)

        self.cond_seq_linear = operations.Linear(
            cond_seq_dim, dim, bias=False, dtype=dtype, device=device
        )  # linear for something like text sequence.
        self.init_x_linear = operations.Linear(
            patch_size * patch_size * in_channels, dim, dtype=dtype, device=device
        )  # init linear for patchified image.

        self.positional_encoding = nn.Parameter(torch.empty(1, max_seq, dim, dtype=dtype, device=device))
        self.register_tokens = nn.Parameter(torch.empty(1, 8, dim, dtype=dtype, device=device))

        self.double_layers = nn.ModuleList([])
        self.single_layers = nn.ModuleList([])


        for idx in range(n_double_layers):
            self.double_layers.append(
                ReMMDiTBlock(dim, n_heads, global_conddim, is_last=(idx == n_layers - 1), dtype=dtype, device=device, operations=operations)
            )

        for idx in range(n_double_layers, n_layers):
            self.single_layers.append(
                ReDiTBlock(dim, n_heads, global_conddim, dtype=dtype, device=device, operations=operations)
            )


        self.final_linear = operations.Linear(
            dim, patch_size * patch_size * out_channels, bias=False, dtype=dtype, device=device
        )

        self.modF = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device),
        )

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.n_double_layers = n_double_layers
        self.n_layers = n_layers

        self.h_max = round(max_seq**0.5)
        self.w_max = round(max_seq**0.5)

    @torch.no_grad()
    def extend_pe(self, init_dim=(16, 16), target_dim=(64, 64)):
        # extend pe
        pe_data = self.positional_encoding.data.squeeze(0)[: init_dim[0] * init_dim[1]]

        pe_as_2d = pe_data.view(init_dim[0], init_dim[1], -1).permute(2, 0, 1)

        # now we need to extend this to target_dim. for this we will use interpolation.
        # we will use torch.nn.functional.interpolate
        pe_as_2d = F.interpolate(
            pe_as_2d.unsqueeze(0), size=target_dim, mode="bilinear"
        )
        pe_new = pe_as_2d.squeeze(0).permute(1, 2, 0).flatten(0, 1)
        self.positional_encoding.data = pe_new.unsqueeze(0).contiguous()
        self.h_max, self.w_max = target_dim

    def pe_selection_index_based_on_dim(self, h, w):
        h_p, w_p            = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.positional_encoding.shape[1])
        original_pe_indexes = original_pe_indexes.view(self.h_max, self.w_max)
        starth              =  self.h_max // 2 - h_p // 2
        endh                = starth + h_p
        startw              = self.w_max // 2 - w_p // 2
        endw                = startw + w_p
        original_pe_indexes = original_pe_indexes[
            starth:endh, startw:endw
        ]
        return original_pe_indexes.flatten()

    def unpatchify(self, x, h, w):
        c = self.out_channels
        p = self.patch_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        x = x.view(
            B,
            C,
            (H + 1) // self.patch_size,
            self.patch_size,
            (W + 1) // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def apply_pos_embeds(self, x, h, w):
        h = (h + 1) // self.patch_size
        w = (w + 1) // self.patch_size
        max_dim = max(h, w)

        cur_dim = self.h_max
        pos_encoding = comfy.ops.cast_to_input(self.positional_encoding.reshape(1, cur_dim, cur_dim, -1), x)

        if max_dim > cur_dim:
            pos_encoding = F.interpolate(pos_encoding.movedim(-1, 1), (max_dim, max_dim), mode="bilinear").movedim(1, -1)
            cur_dim = max_dim

        from_h = (cur_dim - h) // 2
        from_w = (cur_dim - w) // 2
        pos_encoding = pos_encoding[:,from_h:from_h+h,from_w:from_w+w]
        return x + pos_encoding.reshape(1, -1, self.positional_encoding.shape[-1])

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        
        x_orig       = x.clone()
        context_orig = context.clone()
        
        SIGMA = timestep[0].unsqueeze(0) #/ 1000
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True
            
        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

        y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight", 0.0)
        y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight", 0.0)
        y0_style_pos_synweight *= y0_style_pos_weight

        y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight", 0.0)
        y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight", 0.0)
        y0_style_neg_synweight *= y0_style_neg_weight
                
        
        out_list = []
        for i in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][i] == 1

            x       = x_orig[i][None,...].clone()
            context = context_orig.clone()

            patches_replace = transformer_options.get("patches_replace", {})
            # patchify x, add PE
            b, c, h, w = x.shape
            
            h_len = ((h + (self.patch_size // 2)) // self.patch_size) # h_len 96
            w_len = ((w + (self.patch_size // 2)) // self.patch_size) # w_len 96


            x = self.init_x_linear(self.patchify(x))  # B, T_x, D
            x = self.apply_pos_embeds(x, h, w)

            if UNCOND:

                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) 
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor",  0.0) 
                transformer_options['reg_cond_mask_orig'] = transformer_options.get('regional_conditioning_mask_orig')
                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] > 0.0:
                    AttnMask.attn_mask_recast(x.dtype)
                    context_tmp = RegContext.get().to(context.dtype)
                    #context_tmp = 0 * context_tmp.clone()
                    
                    # If it's not a perfect factor, repeat and slice:
                    A = context[i][None,...].clone()
                    B = context_tmp
                    context_tmp = A.repeat(1, (B.shape[1] // A.shape[1]) + 1, 1)[:, :B.shape[1], :]

                else:
                    context_tmp = context[i][None,...].clone()
                    
            elif UNCOND == False:
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) 
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor", 0.0) 
                transformer_options['reg_cond_mask_orig'] = transformer_options.get('regional_conditioning_mask_orig')
                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] > 0.0:
                    AttnMask.attn_mask_recast(x.dtype)
                    context_tmp = RegContext.get().to(context.dtype)
                else:
                    context_tmp = context[i][None,...].clone()
            
            if context_tmp is None:
                context_tmp = context[i][None,...].clone()
                



            # process conditions for MMDiT Blocks
            #c_seq = context  # B, T_c, D_c
            c_seq = context_tmp  # B, T_c, D_c

            t = timestep

            c = self.cond_seq_linear(c_seq)  # B, T_c, D         # 1,256,2048 -> 
            c = torch.cat([comfy.ops.cast_to_input(self.register_tokens, c).repeat(c.size(0), 1, 1), c], dim=1)   #1,256,3072 -> 1,264,3072

            global_cond = self.t_embedder(t, x.dtype)  # B, D

            global_cond = global_cond[i][None]

            

            weight    = transformer_options['reg_cond_weight'] if 'reg_cond_weight' in transformer_options else 0.0
            floor     = transformer_options['reg_cond_floor']  if 'reg_cond_floor'  in transformer_options else 0.0
            
            floor     = min(floor, weight)
            
            reg_cond_mask_expanded = transformer_options.get('reg_cond_mask_expanded')
            reg_cond_mask_expanded = reg_cond_mask_expanded.to(img.dtype).to(img.device) if reg_cond_mask_expanded is not None else None
            reg_cond_mask = None



            AttnMask = transformer_options.get('AttnMask')
            mask     = None
            if AttnMask is not None and weight > 0:
                mask                      = AttnMask.get(weight=weight) #mask_obj[0](transformer_options, weight.item())
                
                mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False
                if not mask_type_bool:
                    mask = mask.to(x.dtype)
                    
                if mask_type_bool:
                    mask = F.pad(mask, (8, 0, 8, 0), value=True)
                    #mask = F.pad(mask, (0, 8, 0, 8), value=True)
                else:
                    mask = F.pad(mask, (8, 0, 8, 0), value=1.0)
                
                text_len                  = context.shape[1] # mask_obj[0].text_len
                
                mask[text_len:,text_len:] = torch.clamp(mask[text_len:,text_len:], min=floor.to(mask.device))   #ORIGINAL SELF-ATTN REGION BLEED
                reg_cond_mask = reg_cond_mask_expanded.unsqueeze(0).clone() if reg_cond_mask_expanded is not None else None

            mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False

            total_layers = len(self.double_layers) + len(self.single_layers)

            blocks_replace = patches_replace.get("dit", {})       # context 1,259,2048      x 1,4032,3072
            if len(self.double_layers) > 0:
                for i, layer in enumerate(self.double_layers):
                    if mask_type_bool and weight < (i / (total_layers-1)) and mask is not None:
                        mask = mask.to(x.dtype)
                        
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args):
                            out = {}
                            out["txt"], out["img"] = layer( args["txt"],
                                                            args["img"],
                                                            args["vec"])
                            return out
                        out = blocks_replace[("double_block", i)]({"img": x, "txt": c, "vec": global_cond}, {"original_block": block_wrap})
                        c = out["txt"]
                        x = out["img"]
                    else:
                        c, x = layer(c, x, global_cond, mask=mask, **kwargs)

            if len(self.single_layers) > 0:
                c_len = c.size(1)
                cx = torch.cat([c, x], dim=1)
                for i, layer in enumerate(self.single_layers):
                    if mask_type_bool and weight < ((len(self.double_layers) + i) / (total_layers-1)) and mask is not None:
                        mask = mask.to(x.dtype)
                    
                    if ("single_block", i) in blocks_replace:
                        def block_wrap(args):
                            out = {}
                            out["img"] = layer(args["img"], args["vec"])
                            return out

                        out = blocks_replace[("single_block", i)]({"img": cx, "vec": global_cond}, {"original_block": block_wrap})
                        cx = out["img"]
                    else:
                        cx = layer(cx, global_cond, mask=mask, **kwargs)

                x = cx[:, c_len:]

            fshift, fscale = self.modF(global_cond).chunk(2, dim=1)

            x = modulate(x, fshift, fscale)
            x = self.final_linear(x)
            x = self.unpatchify(x, (h + 1) // self.patch_size, (w + 1) // self.patch_size)[:,:,:h,:w]
            
            out_list.append(x)
            
        eps = torch.stack(out_list, dim=0).squeeze(dim=1)
        
        
        
        
        
        
        freqsep_lowpass_method = transformer_options.get("freqsep_lowpass_method")
        freqsep_sigma          = transformer_options.get("freqsep_sigma")
        freqsep_kernel_size    = transformer_options.get("freqsep_kernel_size")
        freqsep_inner_kernel_size    = transformer_options.get("freqsep_inner_kernel_size")
        freqsep_stride    = transformer_options.get("freqsep_stride")
        
        freqsep_lowpass_weight = transformer_options.get("freqsep_lowpass_weight")
        freqsep_highpass_weight= transformer_options.get("freqsep_highpass_weight")
        freqsep_mask           = transformer_options.get("freqsep_mask")
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        
        if y0_style_pos is not None:
            y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            y0_style_pos_mask = transformer_options.get("y0_style_pos_mask")
            y0_style_pos_mask_edge = transformer_options.get("y0_style_pos_mask_edge")

            y0_style_pos = y0_style_pos.to(dtype)
            x   = x_orig.clone().to(dtype)
            #x   = x.to(dtype)
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
            x   = x.to(dtype)
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




def unpatchify2(x: torch.Tensor, H: int, W: int, patch_size: int) -> torch.Tensor:
    """
    Invert patchify:
      x: (B, N, C*p*p)
      returns: (B, C, H, W), slicing off any padding
    """
    B, N, CPP = x.shape
    p = patch_size
    Hp = math.ceil(H / p)
    Wp = math.ceil(W / p)
    C = CPP // (p * p)
    assert N == Hp * Wp, f"Expected N={Hp*Wp} patches, got {N}"

    x = x.view(B, Hp, Wp, CPP)       
    x = x.view(B, Hp, Wp, C, p, p)     
    x = x.permute(0, 3, 1, 4, 2, 5)      
    imgs = x.reshape(B, C, Hp * p, Wp * p) 
    return imgs[:, :, :H, :W]

