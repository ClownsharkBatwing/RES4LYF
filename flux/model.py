# Adapted from: https://github.com/black-forest-labs/flux

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Callable, Tuple, Dict, List, Any, Union

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
import einops
from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..latents import tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch, StyleMMDiT_Model
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
                        StyleMMDiT_Model = None,
                        RECON_MODE=False,
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
        
        
        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        mask_zero = None
        mask = None
        
        text_len = txt.shape[1] 
        
        if not UNCOND and 'AttnMask' in transformer_options: 
            AttnMask = transformer_options['AttnMask']
            mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
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
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask'].img_len
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None

        if mask is not None and not type(mask[0][0].item()) == bool:
            mask = mask.to(img.dtype)
        if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
            mask_zero = mask_zero.to(img.dtype)

        total_layers = len(self.double_blocks) + len(self.single_blocks)
        
        ca_idx = 0
        for i, block in enumerate(self.double_blocks):

            if   weight > 0 and mask is not None and     weight  <=      i/total_layers:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask_zero, idx=i, update_cross_attn=update_cross_attn)
                
            elif (weight < 0 and mask is not None and abs(weight) <= (1 - i/total_layers)):
                img_tmpZ, txt_tmpZ = img.clone(), txt.clone()
                img_tmpZ, txt = block(img=img_tmpZ, txt=txt_tmpZ, vec=vec, pe=pe, mask=mask, idx=i, update_cross_attn=update_cross_attn)
                img, txt_tmpZ = block(img=img     , txt=txt     , vec=vec, pe=pe, mask=mask_zero, idx=i, update_cross_attn=update_cross_attn)
                
            elif floor > 0 and mask is not None and     floor  >=      i/total_layers:
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask_tmp, idx=i, update_cross_attn=update_cross_attn)
                
            elif floor < 0 and mask is not None and abs(floor) >= (1 - i/total_layers):
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask_tmp, idx=i, update_cross_attn=update_cross_attn)

            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask, idx=i, update_cross_attn=update_cross_attn)


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
                img = block(img, vec=vec, pe=pe, mask=mask_zero)
                
            elif weight < 0 and mask is not None and abs(weight) <= (1 - (i+len(self.double_blocks))/total_layers):
                img = block(img, vec=vec, pe=pe, mask=mask_zero)
                
            elif floor > 0 and mask is not None and     floor  >=      (i+len(self.double_blocks))/total_layers:
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img = block(img, vec=vec, pe=pe, mask=mask_tmp)
                
            elif floor < 0 and mask is not None and abs(floor) >= (1 - (i+len(self.double_blocks))/total_layers):
                mask_tmp = mask.clone()
                mask_tmp[text_len:, text_len:] = 1.0
                img = block(img, vec=vec, pe=pe, mask=mask_tmp)
                
            else:
                img = block(img, vec=vec, pe=pe, mask=mask)



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
    
    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    
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
                ref_latents=None, 
                control             = None,
                transformer_options = {},
                mask                = None,
                **kwargs
                ):
        t = timestep
        self.max_seq = (128 * 128) // (2 * 2)
        x_orig      = x.clone()
        b, c, h, w  = x.shape
        h_len = ((h + (self.patch_size // 2)) // self.patch_size) # h_len 96
        w_len = ((w + (self.patch_size // 2)) // self.patch_size) # w_len 96
        img_len = h_len * w_len
        img_slice = slice(-img_len, None) #slice(None, img_len)
        txt_slice = slice(None, -img_len)
        SIGMA = t[0].clone() #/ 1000
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        if EO("zero_heads"):
            HEADS = 0
        else:
            HEADS = 24

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

        img_orig, t_orig, y_orig, context_orig = clone_inputs(img, t, y, context)
    
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
        RECON_MODE = StyleMMDiT.noise_mode == "recon"
        recon_iterations = 2 if StyleMMDiT.noise_mode == "recon" else 1
        for recon_iter in range(recon_iterations):
            y0_style = StyleMMDiT.guides
            y0_style_active = True if type(y0_style) == torch.Tensor else False
            
            RECON_MODE = True     if StyleMMDiT.noise_mode == "recon" and recon_iter == 0     else False
            
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
                if y0_style.sum() == 0.0 and y0_style.std() == 0.0:
                    y0_style = img_orig.clone()
                else:
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
                bsz       = 1 if RECON_MODE else bsz_style + 1

                img, t, y, context = clone_inputs(img_orig, t_orig, y_orig, context_orig, index=cond_iter)
                
                mask = None
                if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
                    AttnMask = transformer_options['AttnMask']
                    mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)
                        mask_zero[txt_slice, txt_slice] = mask[txt_slice, txt_slice]

                    if weight == 0:
                        context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                        mask = None
                    else:
                        context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)

                if UNCOND and 'AttnMask_neg' in transformer_options: # and weight != 0:
                    AttnMask = transformer_options['AttnMask_neg']
                    mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)
                        mask_zero[txt_slice, txt_slice] = mask[txt_slice, txt_slice]

                    if weight == 0:
                        context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                        mask = None
                    else:
                        context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)

                elif UNCOND and 'AttnMask' in transformer_options:
                    AttnMask = transformer_options['AttnMask']
                    mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                    
                    if mask_zero is None:
                        mask_zero = torch.ones_like(mask)

                        mask_zero[txt_slice, txt_slice] = mask[txt_slice, txt_slice]
                    if weight == 0:                                                                             # ADDED 5/23/2025
                        context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)  # ADDED 5/26/2025 14:53
                        mask = None
                    else:
                        A       = context
                        B       = transformer_options['RegContext'].context
                        context = A.repeat(1,    (B.shape[1] // A.shape[1]) + 1, 1)[:,   :B.shape[1], :]


                if y0_style_active and not RECON_MODE:
                    if mask is None:
                        context, y, _ = StyleMMDiT.apply_style_conditioning(
                            UNCOND = UNCOND,
                            base_context       = context,
                            base_y             = y,
                            base_llama3        = None,
                        )
                    else:
                        context = context.repeat(bsz_style + 1, 1, 1)
                        y = y.repeat(bsz_style + 1, 1)                   if y      is not None else None
                    img_y0_style = img_y0_style_orig.clone()

                if mask is not None and not type(mask[0][0].item()) == bool:
                    mask = mask.to(x.dtype)
                if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
                    mask_zero = mask_zero.to(x.dtype)

                clip = self.time_in(timestep_embedding(t, 256).to(x.dtype)) # 1 -> 1,3072
                if self.params.guidance_embed:
                    if guidance is None:
                        print("Guidance strength is none, not using distilled guidance.")
                    else:
                        clip = clip + self.guidance_in(timestep_embedding(guidance, 256).to(x.dtype))
                clip = clip + self.vector_in(y[:,:self.params.vec_in_dim])  #y.shape=1,768  y==all 0s
                clip = clip.to(x)
        
                img_in_dtype = self.img_in.weight.data.dtype
                if img_in_dtype not in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
                    img_in_dtype = x.dtype
                
                if ref_latents is not None:
                    h, w = 0, 0
                    for ref in ref_latents:
                        h_offset = 0
                        w_offset = 0
                        if ref.shape[-2] + h > ref.shape[-1] + w:
                            w_offset = w
                        else:
                            h_offset = h

                        kontext, kontext_ids = self.process_img(ref, index=1, h_offset=h_offset, w_offset=w_offset)
                        #kontext = self.img_in(kontext.to(img_in_dtype))
                        img, img_ids = self.process_img(x)
                        img = torch.cat([img, kontext], dim=1)
                        img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                        h = max(h, ref.shape[-2] + h_offset)
                        w = max(w, ref.shape[-1] + w_offset)
                    img = self.img_in(img.to(img_in_dtype))
                    
                    img_slice = slice(-2*img_len, None)
                    StyleMMDiT.KONTEXT = 1
                    for style_block in StyleMMDiT.double_blocks + StyleMMDiT.single_blocks:
                        style_block.KONTEXT = 1
                        for style_block_imgtxt in [style_block.img, getattr(style_block, "txt")]:
                            style_block_imgtxt.KONTEXT = 1
                            style_block_imgtxt.ATTN.KONTEXT = 1
                    StyleMMDiT.datashock_ref = ref_latents[0]
                else:
                    
                    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
                    img = self.img_in(img.to(img_in_dtype))
                    img_ids = self._get_img_ids(img, bsz, h_len, w_len, 0, h_len, 0, w_len)

                if y0_style_active and not RECON_MODE:
                    img_y0_style = rearrange(img_y0_style_orig, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
                    img_y0_style = self.img_in(img_y0_style.to(img_in_dtype))  # hidden_states 1,4032,2560         for 1024x1024: -> 1,4096,2560      ,64 -> ,2560 (x40)
                    if ref_latents is not None:
                        img_kontext  = self.img_in(kontext.to(img_in_dtype))
                        #img_base = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
                        #img_base = self.img_in(img_base.to(img_in_dtype))
                        #img_ids = self._get_img_ids(img, bsz, h_len, w_len, 0, h_len, 0, w_len)
                        img_ids      = img_ids     .repeat(bsz,1,1)
                        #img_y0_style = img_y0_style.repeat(1,bsz,1) # torch.cat([img, img_y0_style], dim=0)
                        img_y0_style = torch.cat([img_y0_style, img_kontext.repeat(bsz-1,1,1)], dim=1)
                        
                        StyleMMDiT.KONTEXT = 2
                        for style_block in StyleMMDiT.double_blocks + StyleMMDiT.single_blocks:
                            style_block.KONTEXT = 2
                            for style_block_imgtxt in [style_block.img, getattr(style_block, "txt")]:
                                style_block_imgtxt.KONTEXT = 2
                                style_block_imgtxt.ATTN.KONTEXT = 2
                        StyleMMDiT.datashock_ref = None
                        
                    img = torch.cat([img, img_y0_style], dim=0)


                # txt_ids -> 1,414,3
                txt_ids = torch.zeros((bsz, context.shape[-2], 3), device=img.device, dtype=x.dtype) 
                ids     = torch.cat((txt_ids, img_ids), dim=-2)   # ids -> 1,4446,3       # flipped from hidream
                rope    = self.pe_embedder(ids)                  # rope -> 1, 4446, 1, 64, 2, 2

                txt_init = self.txt_in(context)
                txt_init_len = txt_init.shape[-2]                                       # 271

                img = StyleMMDiT(img, "proj_in")
                
                img = img.to(x) if img is not None else None
                
                total_layers = len(self.double_blocks) + len(self.single_blocks)
                
                # DOUBLE STREAM
                ca_idx = 0
                for bid, (block, style_block) in enumerate(zip(self.double_blocks, StyleMMDiT.double_blocks)):
                    txt = txt_init
                    if   weight > 0 and mask is not None and     weight  <      bid/total_layers:
                        img, txt_init = block(img, txt, clip, rope, mask_zero, style_block=style_block)
                        
                    elif (weight < 0 and mask is not None and abs(weight) < (1 - bid/total_layers)):
                        img_tmpZ, txt_tmpZ = img.clone(), txt.clone()

                        # more efficient than the commented lines below being used instead in the loop?
                        img_tmpZ, txt_init = block(img_tmpZ, txt_tmpZ, clip, rope, mask, style_block=style_block)
                        img     , txt_tmpZ = block(img     , txt     , clip, rope, mask_zero, style_block=style_block)
                        
                    elif floor > 0 and mask is not None and     floor  >      bid/total_layers:
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img, txt_init = block(img, txt, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif floor < 0 and mask is not None and abs(floor) > (1 - bid/total_layers):
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img, txt_init = block(img, txt, clip, rope, mask_tmp, style_block=style_block)
                        
                    elif update_cross_attn is not None and update_cross_attn['skip_cross_attn']:
                        img, txt_init = block(img, txt, clip, rope, mask, update_cross_attn=update_cross_attn)
                        
                    else:
                        img, txt_init = block(img, txt, clip, rope, mask, update_cross_attn=update_cross_attn, style_block=style_block)

                    if control is not None: 
                        control_i = control.get("input")
                        if bid < len(control_i):
                            add = control_i[bid]
                            if add is not None:
                                img[:1] += add
                    
                    if hasattr(self, "pulid_data"):
                        if self.pulid_data:
                            if bid % self.pulid_double_interval == 0:
                                for _, node_data in self.pulid_data.items():
                                    if torch.any((node_data['sigma_start'] >= timestep) & (timestep >= node_data['sigma_end'])):
                                        img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                                ca_idx += 1

                # END DOUBLE STREAM

                #img = img[0:1]
                #txt_init = txt_init[0:1]
                img       = torch.cat([txt_init, img], dim=-2)   # 4032 + 271 -> 4303     # txt embed from double stream block   # flipped from hidream

                double_layers = len(self.double_blocks)

                # SINGLE STREAM
                for bid, (block, style_block) in enumerate(zip(self.single_blocks, StyleMMDiT.single_blocks)):

                    if   weight > 0 and mask is not None and     weight  <      (bid+double_layers)/total_layers:
                        img = block(img, clip, rope, mask_zero, style_block=style_block)
                    
                    elif weight < 0 and mask is not None and abs(weight) < (1 - (bid+double_layers)/total_layers):
                        img = block(img, clip, rope, mask_zero, style_block=style_block)
                    
                    elif floor > 0 and mask is not None and     floor  >      (bid+double_layers)/total_layers:
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img = block(img, clip, rope, mask_tmp, style_block=style_block)
                    
                    elif floor < 0 and mask is not None and abs(floor) > (1 - (bid+double_layers)/total_layers):
                        mask_tmp = mask.clone()
                        mask_tmp[img_slice,img_slice] = 1.0
                        img = block(img, clip, rope, mask_tmp, style_block=style_block)
                    
                    else:
                        img = block(img, clip, rope, mask, style_block=style_block)
                    
                    if control is not None: # Controlnet
                        control_o = control.get("output")
                        if bid < len(control_o):
                            add = control_o[bid]
                            if add is not None:
                                img[:1, txt_slice, ...] += add
                                
                    if hasattr(self, "pulid_data"):
                        # PuLID attention
                        if self.pulid_data:
                            real_img, txt = img[:, img_slice, ...], img[:, txt_slice, ...]
                            if bid % self.pulid_single_interval == 0:
                                # Will calculate influence of all nodes at once
                                for _, node_data in self.pulid_data.items():
                                    if torch.any((node_data['sigma_start'] >= timestep) & (timestep >= node_data['sigma_end'])):
                                        real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                                ca_idx += 1
                            img = torch.cat((txt, real_img), 1)
                            
                # END SINGLE STREAM
                
                img = img[..., img_slice, :]
                #img = self.final_layer(img, clip)   # 4096,2560 -> 4096,64
                shift, scale = self.final_layer.adaLN_modulation(clip).chunk(2,dim=1)
                img = (1 + scale[:, None, :]) * self.final_layer.norm_final(img) + shift[:, None, :]
                
                img = StyleMMDiT(img, "proj_out")

                if y0_style_active and not RECON_MODE:
                    img = img[0:1]
                    #img = img[1:2]
                
                #img = self.final_layer.linear(img.to(self.final_layer.linear.weight.data))
                img = self.final_layer.linear(img)

                #img = self.unpatchify(img, img_sizes)
                img = img[:,:img_len]  # accomodate kontext
                img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)
                out_list.append(img)
                
            output = torch.cat(out_list, dim=0)
            eps = output[:, :, :h, :w]
            
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

def clone_inputs(*args, index: int=None):

    if index is None:
        return tuple(x.clone() for x in args)
    else:
        return tuple(x[index].unsqueeze(0).clone() for x in args)



