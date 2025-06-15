# Adapted from: https://github.com/black-forest-labs/flux

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Callable, Tuple, Dict, Any, Union

from ..helper import ExtraOptions

from dataclasses import dataclass
import copy

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from . import layers

#from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.model import Flux as Flux

import math
from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..latents import tile_latent, untile_latent

#from ..latents import interpolate_spd

@dataclass
class FluxParams:
    in_channels        : int
    out_channels       : int
    vec_in_dim         : int
    context_in_dim     : int
    hidden_size        : int
    mlp_ratio          : float
    num_heads          : int
    depth              : int
    depth_single_blocks: int
    axes_dim           : list
    theta              : int
    patch_size         : int
    qkv_bias           : bool
    guidance_embed     : bool

class ReFlux(Flux):
    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype         = dtype
        self.timestep      = -1.0
        self.threshold_inv = False
        params             = FluxParams(**kwargs)
        
        self.params        = params #self.params FluxParams(in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=False)
        self.patch_size    = params.patch_size
        self.in_channels   = params.in_channels  * params.patch_size * params.patch_size    # in_channels 64
        self.out_channels  = params.out_channels * params.patch_size * params.patch_size    # out_channels 64
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        self.hidden_size   = params.hidden_size  # 3072
        self.num_heads     = params.num_heads    # 24
        self.pe_embedder   = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
        self.img_in        = operations.Linear(     self.in_channels, self.hidden_size, bias=True,                                                    dtype=dtype, device=device)   # in_features=  64, out_features=3072
        self.txt_in        = operations.Linear(params.context_in_dim, self.hidden_size,                                                               dtype=dtype, device=device)   # in_features=4096, out_features=3072, bias=True

        self.time_in       = MLPEmbedder(           in_dim=256, hidden_dim=self.hidden_size,                                                          dtype=dtype, device=device, operations=operations)
        self.vector_in     = MLPEmbedder(params.vec_in_dim,                self.hidden_size,                                                          dtype=dtype, device=device, operations=operations) # in_features=768, out_features=3072 (first layer) second layer 3072,3072
        self.guidance_in   =(MLPEmbedder(           in_dim=256, hidden_dim=self.hidden_size,                                                          dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity())

        self.double_blocks = nn.ModuleList([DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias, dtype=dtype, device=device, operations=operations, idx=_) for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio,                           dtype=dtype, device=device, operations=operations, idx=_) for _ in range(params.depth_single_blocks)])

        if final_layer:
            self.final_layer = layers.LastLayer(self.hidden_size, 1, self.out_channels,                                                                      dtype=dtype, device=device, operations=operations)


    
    
    def forward_blocks(self,
                        img      : Tensor,
                        img_ids  : Tensor,
                        txt      : Tensor,
                        txt_ids  : Tensor,
                        timesteps: Tensor,
                        y        : Tensor,
                        guidance : Tensor   = None,
                        control             = None,
                        update_cross_attn   = None,
                        transformer_options = {},
                        UNCOND : bool = False,
                        SIGMA = None,
                        lamb_t_factor = 0.1,
                        ) -> Tensor:
        
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img   img -> 1,4096,3072
        img = self.img_in(img)    # 1,9216,64  == 768x192       # 1,9216,64   ==   1,16,128,256 + 1,16,64,64    # 1,8192,64 with uncond/cond   #:,:,64 -> :,:,3072
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype)) # 1 -> 1,3072
        
        if self.params.guidance_embed:
            if guidance is None:
                print("Guidance strength is none, not using distilled guidance.")
            else:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)  #y.shape=1,768  y==all 0s
        
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1) # img_ids.shape=1,8192,3    txt_ids.shape=1,512,3    #ids.shape=1,8704,3
        pe  = self.pe_embedder(ids)                 # pe.shape 1,1,8704,64,2,2
        
        lamb_t_factor = transformer_options.get("regional_conditioning_weight", 0.0)
        
        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        mask_zero = None
        mask = None
        cross_self_mask = None
        
        text_len = txt.shape[1] 
        
        if not UNCOND and 'AttnMask' in transformer_options: 
            AttnMask = transformer_options['AttnMask']
            mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
            cross_self_mask = transformer_options['AttnMask'].cross_self_mask.mask.to('cuda')
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask'].img_len
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None
            
        if UNCOND and 'AttnMask_neg' in transformer_options: 
            AttnMask = transformer_options['AttnMask_neg']
            mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')
            cross_self_mask = transformer_options['AttnMask_neg'].cross_self_mask.mask.to('cuda')
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask_neg'].img_len
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None
            
        elif UNCOND and 'AttnMask' in transformer_options:
            AttnMask = transformer_options['AttnMask']
            mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
            cross_self_mask = transformer_options['AttnMask'].cross_self_mask.mask.to('cuda')
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask'].img_len
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None
        
        if not hasattr(self, "cross_self_weight"):
            self.cross_self_weight = 1.0

        if mask is not None and not type(mask[0][0].item()) == bool:
            mask = mask.to(img.dtype)
        if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
            mask_zero = mask_zero.to(img.dtype)

        total_layers = len(self.double_blocks) + len(self.single_blocks)
        
        ca_idx = 0
        for i, block in enumerate(self.double_blocks):

            if   weight > 0 and mask is not None and     weight  <=      i/total_layers:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask_zero, cross_self_mask=cross_self_mask, idx=i, update_cross_attn=update_cross_attn, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            elif (weight < 0 and mask is not None and abs(weight) <= (1 - i/total_layers)):
                img_tmpZ, txt_tmpZ = img.clone(), txt.clone()
                img_tmpZ, txt = block(img=img_tmpZ, txt=txt_tmpZ, vec=vec, pe=pe, mask=mask, cross_self_mask=cross_self_mask, idx=i, update_cross_attn=update_cross_attn, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                img, txt_tmpZ = block(img=img     , txt=txt     , vec=vec, pe=pe, mask=mask_zero, cross_self_mask=cross_self_mask, idx=i, update_cross_attn=update_cross_attn, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            elif floor > 0 and mask is not None and     floor  >=      i/total_layers:
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask_tmp, cross_self_mask=cross_self_mask, idx=i, update_cross_attn=update_cross_attn, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            elif floor < 0 and mask is not None and abs(floor) >= (1 - i/total_layers):
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask_tmp, cross_self_mask=cross_self_mask, idx=i, update_cross_attn=update_cross_attn, sigma=SIGMA, lamb_t_factor=lamb_t_factor)

            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask, cross_self_mask=cross_self_mask, idx=i, update_cross_attn=update_cross_attn, sigma=SIGMA, lamb_t_factor=lamb_t_factor)


            if control is not None: 
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add
                        
            if hasattr(self, "pulid_data"):
                if self.pulid_data:
                    if i % self.pulid_double_interval == 0:
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                                img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                        ca_idx += 1

        img = torch.cat((txt, img), 1)   #first 256 is txt embed
        for i, block in enumerate(self.single_blocks):

            if   weight > 0 and mask is not None and     weight  <=      (i+len(self.double_blocks))/total_layers:
                img = block(img, vec=vec, pe=pe, mask=mask_zero, cross_self_mask=cross_self_mask, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            elif weight < 0 and mask is not None and abs(weight) <= (1 - (i+len(self.double_blocks))/total_layers):
                img = block(img, vec=vec, pe=pe, mask=mask_zero, cross_self_mask=cross_self_mask, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            elif floor > 0 and mask is not None and     floor  >=      (i+len(self.double_blocks))/total_layers:
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img = block(img, vec=vec, pe=pe, mask=mask_tmp, cross_self_mask=cross_self_mask, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            elif floor < 0 and mask is not None and abs(floor) >= (1 - (i+len(self.double_blocks))/total_layers):
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img = block(img, vec=vec, pe=pe, mask=mask_tmp, cross_self_mask=cross_self_mask, sigma=SIGMA, lamb_t_factor=lamb_t_factor)
                
            else:
                img = block(img, vec=vec, pe=pe, mask=mask, cross_self_mask=cross_self_mask, sigma=SIGMA, lamb_t_factor=lamb_t_factor)



            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:1, txt.shape[1] :, ...] += add
                        
            if hasattr(self, "pulid_data"):
                # PuLID attention
                if self.pulid_data:
                    real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                    if i % self.pulid_single_interval == 0:
                        # Will calculate influence of all nodes at once
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                                real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                        ca_idx += 1
                    img = torch.cat((txt, real_img), 1)

        img = img[:, txt.shape[1] :, ...]
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)     1,8192,3072 -> 1,8192,64 
        return img
    
    
    
    def _get_img_ids(self, x, bs, h_len, w_len, h_start, h_end, w_start, w_end):
        img_ids          = torch.zeros(  (h_len,   w_len, 3),              device=x.device, dtype=x.dtype)
        img_ids[..., 1] += torch.linspace(h_start, h_end - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] += torch.linspace(w_start, w_end - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids          = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        return img_ids

    def forward(self,
                x,
                timestep,
                context,
                y,
                guidance,
                control             = None,
                transformer_options = {},
                **kwargs
                ):
        SIGMA = timestep[0].unsqueeze(0)
        update_cross_attn = transformer_options.get("update_cross_attn")
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True
            
        if EO("adain_tile"):
            self.adain_tile = (EO("adain_tile", 4), EO("adain_tile", 4))
            self.adain_flag = False
        else:
            if hasattr(self, "adain_tile"):
                del self.adain_tile
        
        lamb_t_factor = EO("lamb_t_factor", 0.1)

        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

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

        out_list = []
        for i in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][i] == 1
            
            if update_cross_attn is not None:
                update_cross_attn['UNCOND'] = UNCOND
                
            img = x
            bs, c, h, w = x.shape
            patch_size  = 2
            img           = comfy.ldm.common_dit.pad_to_patch_size(img, (patch_size, patch_size))    # 1,16,192,192
            
            transformer_options['original_shape'] = img.shape
            transformer_options['patch_size']     = patch_size

            h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
            w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96

            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

            
            if EO("cross_self"):
                AttnMask = transformer_options.get('AttnMask')
                if AttnMask is not None:
                    AttnMask.cross_self_mask.mask *= EO("cross_self", 1.0)
            
            
            context_tmp = None
            
            if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
                AttnMask = transformer_options['AttnMask']
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')

                if weight == 0:
                    context_tmp = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    mask = None
                else:
                    context_tmp = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                
            if UNCOND and 'AttnMask_neg' in transformer_options: # and weight != 0:
                AttnMask = transformer_options['AttnMask_neg']
                mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')

                if weight == 0:
                    context_tmp = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                    mask = None
                else:
                    context_tmp = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)

            elif UNCOND and 'AttnMask' in transformer_options:
                AttnMask = transformer_options['AttnMask']
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                A       = context
                B       = transformer_options['RegContext'].context
                context_tmp = A.repeat(1,    (B.shape[1] // A.shape[1]) + 1, 1)[:,   :B.shape[1], :]

            if context_tmp is None:
                context_tmp = context[i][None,...].clone()

            
            
            
            
            txt_ids      = torch.zeros((bs, context_tmp.shape[1], 3), device=img.device, dtype=img.dtype)      # txt_ids        1, 256,3
            img_ids_orig = self._get_img_ids(img, bs, h_len, w_len, 0, h_len, 0, w_len)                  # img_ids_orig = 1,9216,3


            out_tmp = self.forward_blocks(img       [i][None,...].clone(), 
                                        img_ids_orig[i][None,...].clone(), 
                                        context_tmp,
                                        txt_ids     [i][None,...].clone(), 
                                        timestep    [i][None,...].clone(), 
                                        y           [i][None,...].clone(),
                                        guidance    [i][None,...].clone(),
                                        control, 
                                        update_cross_attn=update_cross_attn,
                                        transformer_options=transformer_options,
                                        UNCOND=UNCOND,
                                        SIGMA=SIGMA,
                                        lamb_t_factor=lamb_t_factor,
                                        )  # context 1,256,4096   y 1,768
            out_list.append(out_tmp)
            
        out = torch.stack(out_list, dim=0).squeeze(dim=1)
        #return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        eps = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1 and not UNCOND):
        if y0_style_pos is not None:
            y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            y0_style_pos_mask = transformer_options.get("y0_style_pos_mask")
            y0_style_pos_mask_edge = transformer_options.get("y0_style_pos_mask_edge")
            
            mask_flat = None
            if y0_style_pos_mask is not None:
                if y0_style_pos_mask.ndim == 3:
                    y0_style_pos_mask = y0_style_pos_mask.unsqueeze(1)
                mask_down = F.interpolate(
                    y0_style_pos_mask.to(dtype=torch.float32),
                    size=(h_len, w_len),
                    mode='nearest'
                )
                mask_flat = mask_down.view(-1)  # shape: (4096,)
                mask_flat = mask_flat > 0.5     # boolify
            
            y0_style_pos = y0_style_pos.to(dtype)
            x   = x.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps
            
            img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))

            h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
            w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

            img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_pos, (self.patch_size, self.patch_size))

            h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
            w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
            img_y0_adain = rearrange(img_y0_adain, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

            W = self.img_in.weight.data.to(dtype)   # shape [2560, 64]
            b = self.img_in.bias.data.to(dtype)     # shape [2560]
            
            denoised_embed = F.linear(img         .to(W), W, b).to(img)
            y0_adain_embed = F.linear(img_y0_adain.to(W), W, b).to(img_y0_adain)
            
            if transformer_options['y0_style_method'] == "scattersort":
                flatmask = F.interpolate(y0_style_pos_mask, size=(h_len, w_len)).bool().flatten().cpu()
                flatunmask = ~flatmask
                
                if y0_style_pos_mask_edge is not None:
                    edgemask = F.interpolate(y0_style_pos_mask_edge.unsqueeze(0), size=(h_len, w_len)).bool().flatten()
                    denoised_embed_orig = denoised_embed.clone()
                    flatmask   = flatmask   & (~edgemask)
                    flatunmask = flatunmask & (~edgemask)
                
                denoised_masked = denoised_embed[:, flatmask, :].clone()
                y0_adain_masked = y0_adain_embed[:, flatmask, :].clone()
                
                src_sorted, src_idx = denoised_masked.sort(dim=-2)
                ref_sorted, ref_idx = y0_adain_masked.sort(dim=-2)
                
                denoised_embed[:, flatmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted)
                
                denoised_unmasked = denoised_embed[:, flatunmask, :].clone()
                y0_adain_unmasked = y0_adain_embed[:, flatunmask, :].clone()
                
                src_sorted, src_idx = denoised_unmasked.sort(dim=-2)
                ref_sorted, ref_idx = y0_adain_unmasked.sort(dim=-2)
                
                denoised_embed[:, flatunmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted)
                
                if y0_style_pos_mask_edge is not None:
                    #denoised_embed[:, edgemask, :] = denoised_embed_orig[:, edgemask, :]
                    
                    denoised_edgemasked = denoised_embed[:, edgemask, :].clone()
                    y0_adain_edgemasked = y0_adain_embed[:, edgemask, :].clone()
                    
                    src_sorted, src_idx = denoised_edgemasked.sort(dim=-2)
                    ref_sorted, ref_idx = y0_adain_edgemasked.sort(dim=-2)
                    
                    denoised_embed[:, edgemask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted)
                



            elif transformer_options['y0_style_method'] == "AdaIN":
                if freqsep_mask is not None:
                    #if freqsep_mask.dim() == 3:  # [C, H, W]
                    #    freqsep_mask = freqsep_mask.unsqueeze(0)
                    #elif freqsep_mask.dim() == 2:  # [H, W]
                    #    freqsep_mask = freqsep_mask.unsqueeze(0).unsqueeze(0)
                    freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                    freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                    #patch_mask_flat = self.patch_mask.view(1, -1)  # Shape: [1, 4096]
                    #self.mask_adain = patch_mask_flat.to(denoised_embed)

                #if hasattr(self, "mask_adain"):
                
                #    denoised_embed = adain_seq_dual_region_inplace(denoised_embed, y0_adain_embed, self.mask_adain)
                
                
                
                
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
                    
                    #tiles_out_tensor[:,:,h_off:-h_off, w_off:-w_off] = tiles_out_tensor
                    #tiles_out_tensor = rearrange(tiles_out_tensor, "b c h w -> b (h w) c", h=h_len, w=w_len)
                    if h_off == 0:
                        denoised_pretile = tiles_out_tensor
                    else:
                        denoised_pretile[:,:,h_off:-h_off, w_off:-w_off] = tiles_out_tensor
                    denoised_embed = rearrange(denoised_pretile, "b c h w -> b (h w) c", h=h_len, w=w_len)
                
                elif EO("adain_tile_SOT"):
                    denoised_embed = apply_tile_sot(denoised_embed, y0_adain_embed, tile_sz=EO("adain_tile_SOT_tile_sz", 1), num_proj=EO("adain_tile_SOT_num_proj", 16))
                
                elif EO("adain_tile_sort"):

                    tile_h, tile_w = (EO("adain_tile_sort", 4), EO("adain_tile_sort", 4))
                    sort_dim = EO("adain_tile_sort_dim", -2) # channel dim

                    tiles    = rearrange(denoised_embed, 'b (h th w tw) c -> (b h w) (th tw) c', h=h_len//tile_h, w=w_len//tile_w, th=tile_h, tw=tile_w)
                    y0_tiles = rearrange(y0_adain_embed, 'b (h th w tw) c -> (b h w) (th tw) c', h=h_len//tile_h, w=w_len//tile_w, th=tile_h, tw=tile_w)
                    
                    tiles_out = []
                    for i, (tile, y0_tile) in enumerate(zip(tiles, y0_tiles)):
                        src_sorted, src_idx =    tile.sort(dim=sort_dim)
                        ref_sorted, ref_idx = y0_tile.sort(dim=sort_dim)
                    
                        tiles[i] = tile.scatter(dim=sort_dim, index=src_idx, src=ref_sorted)

                    denoised_embed = rearrange(tiles, '(b h w) (th tw) c -> b (h th w tw) c', h=h_len//tile_h, w=w_len//tile_w, th=tile_h, tw=tile_w)
                    
                elif EO("adain_mask_sort"):
                    freqsep_mask = F.interpolate(freqsep_mask, size=(h_len, w_len))
                    flatmask = freqsep_mask.bool().flatten()
                    
                    denoised_masked = denoised_embed[:, flatmask, :].clone()
                    y0_adain_masked = y0_adain_embed[:, flatmask, :].clone()
                    
                    src_sorted, src_idx = denoised_masked.sort(dim=-2)
                    ref_sorted, ref_idx = y0_adain_masked.sort(dim=-2)
                    
                    denoised_embed[:, flatmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted)
                    
                    denoised_unmasked = denoised_embed[:, ~flatmask, :].clone()
                    y0_adain_unmasked = y0_adain_embed[:, ~flatmask, :].clone()
                    
                    src_sorted, src_idx = denoised_unmasked.sort(dim=-2)
                    ref_sorted, ref_idx = y0_adain_unmasked.sort(dim=-2)
                    
                    denoised_embed[:, ~flatmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted)

                    
                elif freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):

                        
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

                    denoised_spatial_new = adain_patchwise_row_batch_sortmatch(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), kernel_size=freqsep_kernel_size, inner_kernel_size=freqsep_inner_kernel_size, mask=freqsep_mask, use_sort_match=True)

                    #denoised_spatial_new = adain_patchwise_strict_sortmatch_fixed(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), kernel_size=freqsep_kernel_size, inner_kernel_size=freqsep_inner_kernel_size, mask=freqsep_mask, stride=freqsep_stride)

                    #denoised_spatial_new = patchwise_sortmatch_nonoverlap(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), patch_size=freqsep_kernel_size, mask=freqsep_mask)

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
                    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
            elif transformer_options['y0_style_method'] == "WCT":
                if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    self.y0_adain_embed = y0_adain_embed
                    
                    f_s          = y0_adain_embed[0].clone()
                    if mask_flat is not None and (mask_flat == False).any():
                        f_s = f_s[mask_flat]
                    
                    self.mu_s    = f_s.mean(dim=0, keepdim=True)
                    f_s_centered = f_s - self.mu_s
                    
                    cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                    if EO("WCT_SVD"):
                        U_svd, S_svd, Vh_svd = torch.linalg.svd(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig = S_svd
                        U_eig = U_svd
                    else:
                        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                    
                    whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                    self.y0_color  = whiten.to(f_s_centered)

                for wct_i in range(eps.shape[0]):
                    f_c          = denoised_embed[wct_i].clone()
                    if mask_flat is not None and (mask_flat == False).any():
                        f_c_orig = f_c.clone()
                        f_c = f_c[mask_flat]
                    
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)
                    if EO("WCT_SVD"):
                        U_svd, S_svd, Vh_svd = torch.linalg.svd(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                        S_eig = S_svd
                        U_eig = U_svd
                    else:
                        S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T + self.mu_s
                    
                    if mask_flat is not None and (mask_flat == False).any():
                        std_c  = f_c.std(dim=0, keepdim=True) + 1e-6
                        std_cs = f_cs.std(dim=0, keepdim=True) + 1e-6
                        mu_cs  = f_cs.mean(dim=0, keepdim=True)
                        f_cs   = (f_cs - mu_cs) * (std_c / std_cs) + mu_c
                    
                    if mask_flat is not None and (mask_flat == False).any():
                        #f_cs_new = denoised_embed[wct_i].clone()
                        f_c_orig[mask_flat] = f_cs
                        f_cs = f_c_orig
                    
                    denoised_embed[wct_i] = f_cs




                if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    img_y0_standard_guide = comfy.ldm.common_dit.pad_to_patch_size(y0_standard_guide, (self.patch_size, self.patch_size))
                    #img_sizes_y0_standard_guide = None
                    #img_y0_standard_guide, img_masks_y0_standard_guide, img_sizes_y0_standard_guide = self.patchify(img_y0_standard_guide, self.max_seq, img_sizes_y0_standard_guide) 
                    h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
                    w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
                    img_y0_standard_guide = rearrange(img_y0_standard_guide, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
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
                    #y0_standard_guide = self.unpatchify (f_cs.unsqueeze(0), img_sizes_y0_standard_guide)
                    f_cs = f_cs.to(eps)
                    y0_standard_guide = rearrange(f_cs.unsqueeze(0), "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
                    self.y0_standard_guide = y0_standard_guide
                    
                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')
                    
                    img_y0_standard_guide = comfy.ldm.common_dit.pad_to_patch_size(y0_inv_standard_guide, (self.patch_size, self.patch_size))
                    #img_sizes_y0_standard_guide = None
                    #img_y0_standard_guide, img_masks_y0_standard_guide, img_sizes_y0_standard_guide = self.patchify(img_y0_standard_guide, self.max_seq, img_sizes_y0_standard_guide) 
                    h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
                    w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
                    img_y0_standard_guide = rearrange(img_y0_standard_guide, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
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
                    #y0_inv_standard_guide = self.unpatchify (f_cs.unsqueeze(0), img_sizes_y0_standard_guide)
                    f_cs = f_cs.to(eps)
                    y0_inv_standard_guide = rearrange(f_cs.unsqueeze(0), "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
                    self.y0_inv_standard_guide = y0_inv_standard_guide





            
            denoised_approx = (denoised_embed - b.to(denoised_embed)) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(denoised_embed)
            denoised_approx = denoised_approx.to(eps)
            
            denoised_approx = rearrange(denoised_approx, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
            
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
            y0_style_neg_mask = transformer_options.get("y0_style_neg_mask")
            
            y0_style_neg = y0_style_neg.to(dtype)
            x   = x.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps
            
            img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))

            h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
            w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
            
            img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_neg, (self.patch_size, self.patch_size))

            img_y0_adain = rearrange(img_y0_adain, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

            W = self.img_in.weight.data.to(dtype)   # shape [2560, 64]
            b = self.img_in.bias.data.to(dtype)     # shape [2560]
            
            denoised_embed = F.linear(img         .to(W), W, b).to(img)
            y0_adain_embed = F.linear(img_y0_adain.to(W), W, b).to(img_y0_adain)
            
            if transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
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

            denoised_approx = (denoised_embed - b.to(denoised_embed)) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(denoised_embed)
            denoised_approx = denoised_approx.to(eps)
            
            denoised_approx = rearrange(denoised_approx, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
            
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



def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
    return content



def adain_seq_masked_inplace(content: torch.Tensor, style: torch.Tensor, mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    In-place AdaIN on masked positions only.

    content: (B, T, C)
    style:   (B, T, C)
    mask:    (B, T) binary mask (1=use, 0=skip)
    """
    B, T, C = content.shape

    mask = mask.unsqueeze(-1)  # (B, T, 1)
    mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # no div by zero

    # masked mean and std over temporal dim
    mean_c = (content * mask).sum(1, keepdim=True) / mask_sum
    var_c  = ((content - mean_c)**2 * mask).sum(1, keepdim=True) / mask_sum
    std_c  = (var_c + eps).sqrt()

    mean_s = (style * mask).sum(1, keepdim=True) / mask_sum
    var_s  = ((style - mean_s)**2 * mask).sum(1, keepdim=True) / mask_sum
    std_s  = (var_s + eps).sqrt()

    # normalize and rescale masked tokens
    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)
    content *= mask  # keep only masked updates
    return content

def adain_seq_dual_region_inplace(content: torch.Tensor, style: torch.Tensor, mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Apply AdaIN separately to masked and unmasked regions, then combine.
    Args:
        content: Tensor of shape [B, T, C]
        style:   Tensor of shape [B, T, C]
        mask:    Tensor of shape [B, T] where 1.0 = masked, 0.0 = unmasked
    Returns:
        Modified content (in-place)
    """
    B, T, C = content.shape

    for region_value in [1.0, 0.0]:
        region_mask = (mask == region_value)                      # [B, T]
        if region_mask.any():
            idx_expand = region_mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]

            c_sub = content[idx_expand].view(-1, C)  # selected content: [N, C]
            s_sub = style  [idx_expand].view(-1, C)  # selected style:   [N, C]

            mean_c = c_sub.mean(0, keepdim=True)
            std_c  = c_sub.std (0, keepdim=True).add_(eps)
            mean_s = s_sub.mean(0, keepdim=True)
            std_s  = s_sub.std (0, keepdim=True).add_(eps)

            c_norm = (c_sub - mean_c) / std_c * std_s + mean_s  # [N, C]

            content.masked_scatter_(idx_expand, c_norm)

    return content




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



def adain_patchwise_strict_sortmatch_fixed(
    content:  torch.Tensor,   # [B,C,H,W]
    style:    torch.Tensor,   # [B,C,H,W]
    kernel_size:       int,   # can be even or odd
    inner_kernel_size: int = 1,
    stride:            int = 1,
    mask:      torch.Tensor = None  # [B,1,H,W] or None
) -> torch.Tensor:
    B,C,H,W = content.shape
    assert inner_kernel_size <= kernel_size

    # we’ll reflect-pad by the “nominal” half-kernel
    pad = kernel_size // 2

    # pad so that every center (i,j) can grab a full patch
    cp = F.pad(content, (pad,)*4, mode='reflect')
    sp = F.pad(style,   (pad,)*4, mode='reflect')
    out = content.clone()

    # squeeze mask if provided
    if mask is not None:
        mask = mask[:,0].bool()       # [B,H,W]

    def sort_transfer(src, ref):
        # src,ref = [B,C,N] → sort along N axis
        S, idx = src.sort(dim=-1)
        R,_   = ref.sort(dim=-1)
        outf = torch.zeros_like(src)
        outf.scatter_(dim=-1, index=idx, src=R)
        return outf

    # for every pixel center
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            # grab the patch out of the padded volume
            patch_c = cp[:,:, i:i+2*pad+1, j:j+2*pad+1]   # [B,C,Kh,Kw]
            patch_s = sp[:,:, i:i+2*pad+1, j:j+2*pad+1]

            Bc = patch_c.reshape(B, C, -1)  # [B,C,Kh*Kw]
            Bs = patch_s.reshape(B, C, -1)

            matched = sort_transfer(Bc, Bs)
            Kh,Kw = patch_c.shape[-2:]
            matched = matched.view(B, C, Kh, Kw)

            # compute inner block offsets in those actual Kh×Kw dims
            off_h = (Kh - inner_kernel_size) // 2
            off_w = (Kw - inner_kernel_size) // 2
            inner = matched[:, :, off_h:off_h+inner_kernel_size,
                                  off_w:off_w+inner_kernel_size]  # [B,C,ih,iw]

            # where to write it in the *unpadded* output
            y0 = i - pad + off_h
            x0 = j - pad + off_w
            y1 = y0 + inner_kernel_size
            x1 = x0 + inner_kernel_size

            # clamp to image bounds
            oy0, oy1 = max(y0,0), min(y1,H)
            ox0, ox1 = max(x0,0), min(x1,W)

            # corresponding slice in the inner block
            iy0, iy1 = oy0 - y0, oy0 - y0 + (oy1-oy0)
            ix0, ix1 = ox0 - x0, ox0 - x0 + (ox1-ox0)

            if mask is None:
                out[:, :, oy0:oy1, ox0:ox1] = inner[:, :, iy0:iy1, ix0:ix1]
            else:
                # only write centers where mask[b,i,j] is true
                for b in range(B):
                    if not mask[b,i,j]:
                        continue
                    out[b, :, oy0:oy1, ox0:ox1] = inner[b, :, iy0:iy1, ix0:ix1]

    return out



















def patchwise_sort_transfer(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    src, ref: [B, C, N]  (N = patch_h * patch_w)
    returns: [B, C, N] where ref’s values have been permuted
             to match the sort-order of src along dim=-1.
    """
    src_sorted, src_idx = src.sort(dim=-1)
    ref_sorted, _       = ref.sort(dim=-1)
    out = torch.zeros_like(src)
    out.scatter_(dim=-1, index=src_idx, src=ref_sorted)
    return out

def masked_patchwise_sort_transfer(
    src:       torch.Tensor,  # [B, C, N]
    ref:       torch.Tensor,  # [B, C, N]
    mask_flat: torch.Tensor   # [B, N] bool
) -> torch.Tensor:
    """
    Only rearrange the N positions where mask_flat[b] is True.
    """
    B, C, N = src.shape
    out = src.clone()
    for b in range(B):
        valid = mask_flat[b]        # [N]
        if not valid.any():
            continue
        sc = src[b,:,valid]         # [C, M]
        rs = ref[b,:,valid]         # [C, M]
        sc_s, idx = sc.sort(dim=-1)
        rs_s, _   = rs.sort(dim=-1)
        tmp = torch.zeros_like(sc)
        tmp.scatter_(dim=-1, index=idx, src=rs_s)
        out[b,:,valid] = tmp
    return out

def patchwise_sortmatch_nonoverlap(
    content: torch.Tensor,        # [B, C, H, W]
    style:   torch.Tensor,        # [B, C, H, W]
    patch_size: int,
    mask:    torch.Tensor = None  # [B,1,H,W] or None
) -> torch.Tensor:
    B, C, H, W = content.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"
    out = content.clone()

    if mask is not None:
        m = mask[:,0].bool()       # [B,H,W]

    # slide over non-overlapping patches
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            c_patch = content[:, :, i:i+patch_size, j:j+patch_size]  # [B,C,K,K]
            s_patch = style  [:, :, i:i+patch_size, j:j+patch_size]

            # flatten spatial → N = K*K
            Bc = c_patch.reshape(B, C, -1)
            Bs = s_patch.reshape(B, C, -1)

            # choose masked or unmasked sort-transfer
            if mask is None:
                matched = patchwise_sort_transfer(Bc, Bs)
            else:
                mask_flat = m[:, i:i+patch_size, j:j+patch_size].reshape(B, -1)
                matched   = masked_patchwise_sort_transfer(Bc, Bs, mask_flat)

            # write back
            out[:, :, i:i+patch_size, j:j+patch_size] = matched.view(B, C, patch_size, patch_size)

    return out






def adain_patchwise_row_batch_sortmatch_works(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, inner_kernel_size: int=1, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False, use_sort_match: bool = False, highpass_strength: float = 0.25, highpass_clip: float = 2.0) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * abs(sigma)) + 1)
    if kernel_size % 2 == 0 and use_sort_match == False:
        kernel_size += 1
        pad = kernel_size // 2
    else:
        pad = 0
    

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded   = F.pad(style, (pad, pad, pad, pad), mode='reflect')
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

    if not use_median_blur and not use_sort_match:
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

    def patchwise_sort_transfer(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        source_sorted, source_indices = source.sort(dim=-1)
        reference_sorted, _ = reference.sort(dim=-1)
        result = torch.zeros_like(source)
        result.scatter_(dim=-1, index=source_indices, src=reference_sorted)
        return result

    for i in range(H):
        row_result = torch.zeros(B, C, W, dtype=dtype, device=device)
        for j in range(W):
            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch = style_padded[:, :, i:i+kernel_size, j:j+kernel_size]

            if use_median_blur:
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)

                c_median = unfolded_c.median(dim=-1, keepdim=True).values
                s_median = unfolded_s.median(dim=-1, keepdim=True).values

                center = kernel_size // 2
                central = c_patch[:, :, center, center].view(B, C, 1)
                residual = central - c_median
                #residual_clipped = residual.clamp(-highpass_clip, highpass_clip)
                #stylized = s_median + residual_clipped * (1.0 + highpass_strength)
                stylized = s_median + residual
            elif use_sort_match:
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)
                sorted_transfer = patchwise_sort_transfer(unfolded_c, unfolded_s)
                center = kernel_size // 2
                idx = center * kernel_size + center
                stylized = sorted_transfer[:, :, idx].view(B, C, 1)
                central = c_patch[:, :, center, center].view(B, C, 1)
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







def adain_patchwise_row_batch_sortmatch(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, inner_kernel_size: int = 1, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False, use_sort_match: bool = False, lowpass_weight: float=1.0, highpass_weight: float=1.0, stride: int = 1) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * abs(sigma)) + 1)
    if kernel_size % 2 == 0 and use_sort_match == False:
        kernel_size += 1
        pad = kernel_size // 2
    else:
        pad = 0
        
    if inner_kernel_size == -1:
        inner_kernel_size = kernel_size

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded   = F.pad(style,   (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    scaling = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    sigma_scale = torch.ones((H, W), device=device, dtype=torch.float32)
    #if mask is not None:
    #    with torch.no_grad():
    #        padded_mask = F.pad(mask.float(), (pad, pad, pad, pad), mode="reflect")
    #        blurred_mask = F.avg_pool2d(padded_mask, kernel_size=kernel_size, stride=1, padding=pad)
    #        blurred_mask = blurred_mask[..., pad:-pad, pad:-pad]
    #        edge_proximity = blurred_mask * (1.0 - blurred_mask)
    #        scaling = 1.0 - (edge_proximity / 0.25).clamp(0.0, 1.0)
    #        sigma_scale = scaling[0, 0]  # assuming single-channel mask broadcasted across B, C

    if not use_median_blur and not use_sort_match:
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

    def patchwise_sort_transfer(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        source_sorted, source_indices = source.sort(dim=-1)
        reference_sorted, _ = reference.sort(dim=-1)
        result = torch.zeros_like(source)
        result.scatter_(dim=-1, index=source_indices, src=reference_sorted)
        return result

    for i in range(0, H - kernel_size + 1, stride):
        for j in range(0, W - kernel_size + 1, stride):
            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch =   style_padded[:, :, i:i+kernel_size, j:j+kernel_size]

            if use_median_blur:
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)

                c_median = unfolded_c.median(dim=-1, keepdim=True).values
                s_median = unfolded_s.median(dim=-1, keepdim=True).values

                residual = unfolded_c - c_median
                #residual_clipped = residual.clamp(-highpass_clip, highpass_clip)
                #stylized = s_median + residual_clipped * (1.0 + highpass_strength)
                stylized = lowpass_weight * s_median + highpass_weight * residual
                stylized = stylized.view(B, C, kernel_size, kernel_size)
                
            elif use_sort_match and inner_kernel_size == kernel_size:
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)
                
                
                if mask is not None:
                    mask_patch = mask[..., i:i+kernel_size, j:j+kernel_size]
                    mask_patch_flat = mask_patch.reshape(B, 1, -1).to(dtype=torch.bool)
                    sorted_transfer = masked_patchwise_sort_transfer(unfolded_c, unfolded_s, mask_patch_flat)
                else:
                    sorted_transfer = patchwise_sort_transfer(unfolded_c, unfolded_s)
                
                stylized = sorted_transfer.view(B, C, kernel_size, kernel_size)
                
            elif use_sort_match and inner_kernel_size < kernel_size:
                unfolded_c = c_patch.reshape(B, C, -1)
                unfolded_s = s_patch.reshape(B, C, -1)
                sorted_transfer = patchwise_sort_transfer(unfolded_c, unfolded_s)
                center = kernel_size // 2
                idx = center * kernel_size + center
                stylized = sorted_transfer[:, :, idx].view(B, C, 1)
                central = c_patch[:, :, center, center].view(B, C, 1)
                
                """ elif use_sort_match:
                        unfolded_c = c_patch.reshape(B, C, -1)
                        unfolded_s = s_patch.reshape(B, C, -1)

                        if mask is not None:
                            # Extract matching region of the mask
                            patch_mask = mask[:, :, i:i+kernel_size, j:j+kernel_size]  # [B, 1, k, k]
                            patch_mask = patch_mask.reshape(B, 1, -1).expand(-1, C, -1)  # [B, C, k*k]

                            result_flat = unfolded_c.clone()

                            for b in range(B):
                                # Get indices of valid pixels (same across channels)
                                valid = patch_mask[b, 0] > 0  # [k*k]
                                if valid.sum() == 0:
                                    continue

                                # Extract vectorized pixels at valid locations
                                c_valid = unfolded_c[b, :, valid].T  # [N_valid, C]
                                s_valid = unfolded_s[b, :, valid].T  # [N_valid, C]

                                # L2 match each pixel in c to its nearest in s
                                dists = (c_valid[:, None, :] - s_valid[None, :, :]).pow(2).sum(-1)  # [N_valid, N_valid]
                                nearest = dists.argmin(dim=1)  # [N_valid]
                                result_valid = s_valid[nearest]  # [N_valid, C]

                                result_flat[b, :, valid] = result_valid.T  # Replace

                            stylized = result_flat.view(B, C, kernel_size, kernel_size)

                        else:
                            sorted_transfer = patchwise_sort_transfer(unfolded_c, unfolded_s)
                            stylized = sorted_transfer.view(B, C, kernel_size, kernel_size)

                        if inner_kernel_size < kernel_size:
                            center = kernel_size // 2
                            idx = center * kernel_size + center
                            stylized = stylized[:, :, center:center+1, center:center+1]"""
                
            else:
                k = gaussian_table[float(sigma_scale[i, j].item())]
                local_weight = k.view(1, 1, kernel_size, kernel_size).expand(B, C, kernel_size, kernel_size)

                c_mean = (c_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                c_std = ((c_patch - c_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
                s_mean = (s_patch * local_weight).sum(dim=(-1, -2), keepdim=True)
                s_std = ((s_patch - s_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

                normed = (c_patch - c_mean) / c_std
                stylized = normed * s_std + s_mean

            #local_scaling = scaling[:, :, i:i+kernel_size, j:j+kernel_size]
            #blended = c_patch * (1 - local_scaling) + stylized * local_scaling
            #result[:, :, i:i+kernel_size, j:j+kernel_size] = blended
            result[:, :, i:i+kernel_size, j:j+kernel_size] = stylized

    return result



def sliced_ot(source, target, n_dirs=32):
    squeeze_source = False
    squeeze_target = False
    if source.ndim == 2:
        source.unsqueeze_(0)
    if target.ndim == 2:
        target.unsqueeze_(0)

    B, N, C = source.shape  # (batch, tokens, channels)
    out = torch.zeros_like(source)

    for _ in range(n_dirs):
        v = torch.randn(C, 1).to(source)
        v = v / v.norm()

        # Project
        src_proj = source @ v        # (B, N, 1)
        tgt_proj = target @ v        # (B, N, 1)

        # Sort both
        src_sorted, src_idx = src_proj.sort(dim=1)
        tgt_sorted, _ = tgt_proj.sort(dim=1)

        # Replace sorted projections
        new_proj = torch.zeros_like(src_proj)
        new_proj.scatter_(dim=1, index=src_idx, src=tgt_sorted)

        # Back-project to high dim
        out += new_proj @ v.T
        
    if squeeze_source or squeeze_target:
        out.squeeze_(0)
        
    return out / n_dirs



def sliced_optimal_transport22(
    source: torch.Tensor,  # [B, N, C]
    target: torch.Tensor,  # [B, N, C]
    num_projections: int = 64,
    preserve_mean: bool = False,
    seed: int = None
) -> torch.Tensor:
    """
    Approximate optimal transport between source and target point clouds along random 1-D projections.

    Args:
        source: Tensor of shape (B, N, C)
        target: Tensor of shape (B, N, C)
        num_projections: Number of random directions to average over.
        preserve_mean: If True, re-centers each slice to match means.
        seed: Optional random seed for reproducibility.

    Returns:
        transported: Tensor of shape (B, N, C), the approximated transported source points.
    """
    squeeze_source = False
    squeeze_target = False
    if source.ndim == 2:
        source.unsqueeze_(0)
    if target.ndim == 2:
        target.unsqueeze_(0)

    
    B, N, C = source.shape
    device = source.device
    if seed is not None:
        torch.manual_seed(seed)

    # Optional centering
    if preserve_mean:
        src_mean = source.mean(dim=1, keepdim=True)  # [B,1,C]
        tgt_mean = target.mean(dim=1, keepdim=True)
        source_centered = source - src_mean
        target_centered = target - tgt_mean
    else:
        source_centered = source
        target_centered = target
        src_mean = tgt_mean = None

    # Prepare random directions
    directions = torch.randn(num_projections, C, device=device, dtype=source.dtype)
    directions = directions / directions.norm(dim=1, keepdim=True)  # [L, C]

    # Accumulate transport
    transported = torch.zeros_like(source_centered)

    for v in directions:  # loop over L
        # project onto v: [B, N]
        src_proj = source_centered.matmul(v)  # [B, N]
        tgt_proj = target_centered.matmul(v)  # [B, N]

        # sort and rank
        src_sorted_vals, src_indices = src_proj.sort(dim=-1)      # [B, N]
        tgt_sorted_vals, _ = tgt_proj.sort(dim=-1)

        # scatter to match ranks
        replaced_proj = torch.zeros_like(src_proj)
        replaced_proj.scatter_(dim=-1, index=src_indices, src=tgt_sorted_vals)

        # lift back to C-D: replaced_proj.unsqueeze(-1) * v
        transported += replaced_proj.unsqueeze(-1) * v.view(1, 1, C)

    # average over projections
    transported = transported / float(num_projections)

    # restore mean if centered
    if preserve_mean:
        transported = transported + tgt_mean

    if squeeze_source or squeeze_target:
        transported.squeeze_(0)
        
    return transported

# Example usage:
# B, H, W, C = 1, 8, 8, 3
# source = torch.randn(B, H*W, C)
# target = torch.randn(B, H*W, C)
# out = sliced_optimal_transport(source, target, num_projections=128, preserve_mean=True)
# out has shape (B, H*W, C) and can be rearranged back to (B, C, H, W) with einops.




def sliced_optimal_transport(
    source : torch.Tensor,   # [B, N, C]
    target : torch.Tensor,   # [B, N, C]
    num_projections: int = 64,
    preserve_mean: bool = True,
    eps: float = 1e-6,
    device=None
) -> torch.Tensor:
    """
    Performs Sliced Optimal Transport (rank matching) between two point clouds
    in C-dimensional feature space, per batch.

    Args:
        source:          [B, N, C]
        target:          [B, N, C]
        num_projections: how many random directions to average over
        preserve_mean:   if True, subtract source mean before and add it back after
    Returns:
        transported:     [B, N, C]
    """
    squeeze_source = False
    squeeze_target = False
    if source.ndim == 2:
        source.unsqueeze_(0)
    if target.ndim == 2:
        target.unsqueeze_(0)

    
    B, N, C = source.shape
    device = device or source.device

    # 1) Center
    if preserve_mean:
        src_mean = source.mean(dim=1, keepdim=True)  # [B,1,C]
        source_centered = source - src_mean
    else:
        src_mean = torch.zeros_like(source[:, :1, :])
        source_centered = source

    # always center target the same way
    tgt_mean = target.mean(dim=1, keepdim=True)
    target_centered = target - tgt_mean

    out = torch.zeros_like(source_centered, device=device).to(source)

    for _ in range(num_projections):
        # 2) random directions
        v = torch.randn(C, device=device).to(source)
        v = v / (v.norm() + eps)               # unit length

        # 3) project
        # proj shape: [B, N]
        src_proj = source_centered.matmul(v)   # (B,N,C) @ (C,) -> (B,N)
        tgt_proj = target_centered.matmul(v)

        # 4) sort+replace
        src_vals, src_idx = src_proj.sort(dim=1)        # [B,N]
        tgt_sorted, _    = tgt_proj.sort(dim=1)         # [B,N]

        proj_replaced = torch.zeros_like(src_proj)      # [B,N]
        proj_replaced.scatter_(1, src_idx, tgt_sorted)

        # 5) lift back: each scalar * direction vector
        # (B,N) -> (B,N,1) broadcast to (B,N,C)
        out += proj_replaced.unsqueeze(-1) * v.view(1,1,C)

    # average projections
    out = out / float(num_projections)

    # 6) restore mean
    out = out + src_mean

    if squeeze_source or squeeze_target:
        out.squeeze_(0)

    return out





def apply_tile_sot(denoised_embed, y0_adain_embed, tile_sz=64, num_proj=32):
    B, T, C = denoised_embed.shape
    H = W = int(T**0.5)
    th = tw = tile_sz
    h = w = H//th

    # break into [B*h*w, N, C]
    tiles    = rearrange(denoised_embed, 'b (h th w tw) c -> (b h w) (th tw) c', 
                        h=h, w=w, th=th, tw=tw)
    refs     = rearrange(y0_adain_embed,    'b (h th w tw) c -> (b h w) (th tw) c', 
                        h=h, w=w, th=th, tw=tw)

    outs = []
    for src, ref in zip(tiles, refs):
        # 1) SOT with mean preservation
        out = sliced_optimal_transport(
            src.unsqueeze(0), ref.unsqueeze(0),
            num_projections=num_proj,
            preserve_mean=True
        ).squeeze(0)  # [N, C]

        # 2) global std match
        std_out = out.std()
        std_ref = ref.std()
        out = out * (std_ref / (std_out + 1e-6))

        outs.append(out)

    outs = torch.stack(outs, dim=0)  # [B*h*w, N, C]
    return rearrange(outs, '(b h w) (th tw) c -> b (h th w tw) c',
                     b=B, h=h, w=w, th=th, tw=tw)

# usage:
#denoised_embed = apply_tile_sot(denoised_embed, y0_adain_embed, tile_sz=8, num_proj=16)



