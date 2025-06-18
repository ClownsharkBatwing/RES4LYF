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
from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..latents import tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch
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
        x_orig       = x.clone()
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
        eps = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        
        
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


