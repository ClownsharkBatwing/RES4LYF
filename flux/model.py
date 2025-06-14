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
        
        text_len = txt.shape[1] # mask_obj[0].text_len
        
        #AttnMask = transformer_options.get('AttnMask')
        if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
            AttnMask = transformer_options['AttnMask']
            mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
            cross_self_mask = transformer_options['AttnMask'].cross_self_mask.mask.to('cuda')
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask'].img_len
                #mask_zero[:text_len, :text_len] = mask[:text_len, :text_len]
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None
            
        if UNCOND and 'AttnMask_neg' in transformer_options: # and weight != 0:
            AttnMask = transformer_options['AttnMask_neg']
            mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')
            cross_self_mask = transformer_options['AttnMask_neg'].cross_self_mask.mask.to('cuda')
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask_neg'].img_len
                #mask_zero[:text_len, :text_len] = mask[:text_len, :text_len]
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
                #mask_zero[:text_len, :text_len] = mask[:text_len, :text_len]
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None
        
        if not hasattr(self, "cross_self_weight"):
            self.cross_self_weight = 1.0
        
        """weight    = transformer_options['reg_cond_weight'] if 'reg_cond_weight' in transformer_options else 0.0
        floor     = transformer_options['reg_cond_floor']  if 'reg_cond_floor'  in transformer_options else 0.0
        floor     = min(floor, weight)
        
        
        
        AttnMask = transformer_options.get('AttnMask')
        mask     = None
        if AttnMask is not None and weight > 0:
            mask                      = AttnMask.get(weight=weight) #mask_obj[0](transformer_options, weight.item())
            #mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
            
            mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False
            if not mask_type_bool:
                mask = mask.to(img.dtype)
            
            text_len                  = txt.shape[1] # mask_obj[0].text_len
            
            mask[text_len:,text_len:] = torch.clamp(mask[text_len:,text_len:], min=floor.to(mask.device))   #ORIGINAL SELF-ATTN REGION BLEED"""

        #mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False
        if mask is not None and not type(mask[0][0].item()) == bool:
            mask = mask.to(img.dtype)
        if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
            mask_zero = mask_zero.to(img.dtype)

        total_layers = len(self.double_blocks) + len(self.single_blocks)
        
        ca_idx = 0
        for i, block in enumerate(self.double_blocks):
            #if mask is not None and mask_type_bool and weight < (i / (total_layers-1)):
            #    mask = mask.to(img.dtype)
            #img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask, idx=i, update_cross_attn=update_cross_attn)

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


            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add
                        
            if hasattr(self, "pulid_data"):
                # PuLID attention
                if self.pulid_data:
                    if i % self.pulid_double_interval == 0:
                        # Will calculate influence of all pulid nodes at once
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                                img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                        ca_idx += 1

        img = torch.cat((txt, img), 1)   #first 256 is txt embed
        for i, block in enumerate(self.single_blocks):
            #if mask is not None and mask_type_bool and weight < ((len(self.double_blocks) + i) / (total_layers-1)):
            #    mask = mask.to(img.dtype)
            #img = block(img, vec=vec, pe=pe, mask=mask, idx=i)

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
        freqsep_lowpass_weight = transformer_options.get("freqsep_lowpass_weight")
        freqsep_highpass_weight= transformer_options.get("freqsep_highpass_weight")

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

            """ if UNCOND:
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) #transformer_options['regional_conditioning_weight']
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor",  0.0) #transformer_options['regional_conditioning_floor'] #if "regional_conditioning_floor" in transformer_options else 0.0
                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] > 0.0:
                    AttnMask.attn_mask_recast(img.dtype)
                    context_tmp = RegContext.get().to(context.dtype)
                    
                    A = context[i][None,...].clone()
                    B = context_tmp
                    context_tmp = A.repeat(1, (B.shape[1] // A.shape[1]) + 1, 1)[:, :B.shape[1], :]

                else:
                    context_tmp = context[i][None,...].clone()
            
            
            elif UNCOND == False:
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) 
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor",  0.0) 
                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] > 0.0:
                    AttnMask.attn_mask_recast(img.dtype)
                    context_tmp = RegContext.get()
                else:
                    context_tmp = context[i][None,...].clone()
            
            if context_tmp is None:
                context_tmp = context[i][None,...].clone()
            context_tmp = context_tmp.to(context.dtype)"""
            
            
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

            if transformer_options['y0_style_method'] == "AdaIN":
                if hasattr(self, "guide_mask"):
                    patch_mask = F.interpolate(self.guide_mask.float(), size=(64, 64), mode='nearest-exact')
                    patch_mask_flat = patch_mask.view(1, -1)  # Shape: [1, 4096]
                    self.mask_adain = patch_mask_flat.to(denoised_embed)

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
                    
                elif freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):
                    
                    #denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=EO("adain_pw_sigma", 1.0), kernel_size=EO("adain_pw_kernel_size", 7))
                    if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                        self.y0_adain_embed = y0_adain_embed
                        self.adain_pw_cache = None
                        
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    
                    if hasattr(self, "guide_mask"):
                        denoised_spatial_new = adain_patchwise_row_batch_mask(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=EO("adain_pw_sigma", 1.0), kernel_size=EO("adain_pw_kernel_size", 3), mask=patch_mask.to(denoised_spatial))
                    elif EO("adain_pw_adapt"):
                        
                        denoised_spatial_new = adain_patchwise_row_batch_adaptive_sigma(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=EO("adain_pw_sigma", 1.0), kernel_size=EO("adain_pw_kernel_size", 7))
                    elif freqsep_lowpass_method == "median_pw": #EO("adain_pw_median"):
                        denoised_spatial_new = adain_patchwise_row_batch_medblur(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True)
                        #denoised_spatial_new = adain_patchwise_row_batch_median(denoised_spatial.clone(), y0_adain_spatial.clone(), kernel_size=EO("adain_pw_kernel_size", 7))
                        
                    elif freqsep_lowpass_method == "gaussian_pw": 
                        denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    #denoised_spatial_new, self.adain_pw_cache = adain_patchwise_cached_rowwise(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=EO("adain_pw_sigma", 1.0), kernel_size=EO("adain_pw_kernel_size", 7), cache=self.adain_pw_cache)
                    
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)
                    
                    
                elif freqsep_lowpass_method is not None: #EO("adain_fs"):
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



























def adain_patchwise_row_batch_median(content: torch.Tensor, style: torch.Tensor, kernel_size: int = 3, eps: float = 1e-5) -> torch.Tensor:
    import torch.nn.functional as F

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded   = F.pad(style,   (pad, pad, pad, pad), mode='reflect')
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

        c_flat = c_row_patches.view(W, B, C, -1)  # [W, B, C, k*k]
        s_flat = s_row_patches.view(W, B, C, -1)

        c_median = c_flat.median(dim=-1, keepdim=True).values  # [W, B, C, 1]
        s_median = s_flat.median(dim=-1, keepdim=True).values

        c_std = (c_flat - c_median).abs().mean(dim=-1, keepdim=True) + eps 
        s_std = (s_flat - s_median).abs().mean(dim=-1, keepdim=True) + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center, center].unsqueeze(-1)  # [W, B, C, 1]

        normed = (central - c_median) / c_std
        stylized = normed * s_std + s_median

        result[:, :, i, :] = stylized.squeeze(-1).permute(1, 2, 0)  # [B,C,W]

    return result

def adain_patchwise_cached_rowwise(
    content: torch.Tensor,
    style: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: int = None,
    eps: float = 1e-5,
    cache: dict = None,
) -> torch.Tensor:

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss /= gauss.sum()
    weight = (gauss[:, None] * gauss[None, :]).to(dtype).view(1, 1, kernel_size, kernel_size)

    content_pad = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_pad   = F.pad(style,   (pad, pad, pad, pad), mode='reflect')

    result = torch.zeros_like(content)

    if cache is None:
        cache = {}

    style_key = (id(style), kernel_size, sigma)
    if style_key not in cache:
        style_means, style_stds = [], []
        for i in range(H):
            patches = torch.stack([
                style_pad[:, :, i:i+kernel_size, j:j+kernel_size] for j in range(W)
            ])
            w = weight.expand_as(patches)
            s_mean = (patches * w).sum(dim=(-1, -2), keepdim=True)
            s_std  = ((patches - s_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
            style_means.append(s_mean)
            style_stds.append(s_std)
        cache[style_key] = (style_means, style_stds)
    else:
        style_means, style_stds = cache[style_key]

    for i in range(H):
        c_patches = torch.stack([
            content_pad[:, :, i:i+kernel_size, j:j+kernel_size] for j in range(W)
        ])
        w = weight.expand_as(c_patches)

        c_mean = (c_patches * w).sum(dim=(-1, -2), keepdim=True)
        c_std  = ((c_patches - c_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

        s_mean = style_means[i]
        s_std  = style_stds[i]

        center = kernel_size // 2
        normed = (c_patches[:, :, :, center:center+1, center:center+1] - c_mean) / c_std
        stylized = normed * s_std + s_mean
        result[:, :, i, :] = stylized.squeeze(-1).squeeze(-1).permute(1, 2, 0)

    return result, cache


"""
def adain_patchwise_row_batch_adaptive_sigma(content: torch.Tensor, style: torch.Tensor, min_sigma: float = 0.5, max_sigma: float = 2.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:
    import torch.nn.functional as F
    import math

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    patch_std = content.std(dim=1, keepdim=True)  # [B, 1, H, W]
    max_std = patch_std.max()
    adaptive_sigma_map = min_sigma + (max_sigma - min_sigma) * (1.0 - patch_std / (max_std + eps))  # [B, 1, H, W]

    result = torch.zeros_like(content)
    pad_max = int(2 * math.ceil(3 * max_sigma) + 1) // 2
    content_padded = F.pad(content, (pad_max, pad_max, pad_max, pad_max), mode='reflect')
    style_padded = F.pad(style, (pad_max, pad_max, pad_max, pad_max), mode='reflect')

    for i in range(H):
        for j in range(W):
            sigma_ij = adaptive_sigma_map[0, 0, i, j].item()
            kernel_size = int(2 * math.ceil(3 * sigma_ij) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            pad = kernel_size // 2

            coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
            gauss = torch.exp(-0.5 * (coords / sigma_ij) ** 2)
            gauss = (gauss / gauss.sum()).to(dtype)
            kernel_2d = (gauss[:, None] * gauss[None, :])
            weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch = style_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            w = weight.expand_as(c_patch)

            c_mean = (c_patch * w).sum(dim=(-1, -2), keepdim=True)
            c_std  = ((c_patch - c_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
            s_mean = (s_patch * w).sum(dim=(-1, -2), keepdim=True)
            s_std  = ((s_patch - s_mean)**2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

            central = c_patch[:, :, pad:pad+1, pad:pad+1]
            normed = (central - c_mean) / c_std
            stylized = normed * s_std + s_mean

            result[:, :, i, j] = stylized.squeeze(-1).squeeze(-1)

    return result
"""


def adain_patchwise_row_batch_adaptive_sigma(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:

    base_sigma = sigma

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * base_sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2

    # gaussian weights for computing local std
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gauss = torch.exp(-0.5 * (coords / base_sigma) ** 2)
    gauss /= gauss.sum()
    kernel_2d = (gauss[:, None] * gauss[None, :]).to(dtype=dtype)
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)

    # local std map from content
    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    mean = F.conv2d(content_padded, weight, groups=C)
    mean_sq = F.conv2d(content_padded ** 2, weight, groups=C)
    var = mean_sq - mean ** 2
    std_map = torch.sqrt(var.clamp(min=eps))  # [B, C, H, W]

    sigma_map = base_sigma * (std_map.mean(dim=1, keepdim=True) / std_map.max())  # [B, 1, H, W]

    # ceuse static kernel for now for convolution ... w ill use pixelwise sigma to scale AdaIN effect
    kernel_2d = (gauss[:, None] * gauss[None, :]).to(dtype=dtype)
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)

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
        c_std = ((c_row_patches - c_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
        s_mean = (s_row_patches * w).sum(dim=(-1, -2), keepdim=True)
        s_std = ((s_row_patches - s_mean) ** 2 * w).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center:center+1, center:center+1]
        normed = (central - c_mean) / c_std
        stylized = normed * s_std + s_mean

        # adaptive strength blending based on local sigma
        local_sigma = sigma_map[:, :, i, :].permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)  # [W, B, 1, 1, 1]
        stylized = central * (1 - local_sigma) + stylized * local_sigma

        result[:, :, i, :] = stylized.squeeze(-1).squeeze(-1).permute(1, 2, 0)  # [B,C,W]

    return result



def adain_patchwise_row_batch_median(content: torch.Tensor, style: torch.Tensor, kernel_size: int = 3, eps: float = 1e-5) -> torch.Tensor:
    import torch.nn.functional as F

    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded   = F.pad(style,   (pad, pad, pad, pad), mode='reflect')
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

        c_flat = c_row_patches.view(W, B, C, -1)  # [W, B, C, k*k]
        s_flat = s_row_patches.view(W, B, C, -1)

        c_median = c_flat.median(dim=-1, keepdim=True).values  # [W, B, C, 1]
        s_median = s_flat.median(dim=-1, keepdim=True).values

        c_std = (c_flat - c_median).abs().mean(dim=-1, keepdim=True) + eps  
        s_std = (s_flat - s_median).abs().mean(dim=-1, keepdim=True) + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center, center].unsqueeze(-1)  # [W, B, C, 1]

        normed = (central - c_median) / c_std
        stylized = normed * s_std + s_median

        result[:, :, i, :] = stylized.squeeze(-1).permute(1, 2, 0)  # [B,C,W]

    return result




"""
def adain_patchwise_row_batch(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    base_gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    base_gauss = (base_gauss / base_gauss.sum()).to(dtype)
    base_kernel_2d = (base_gauss[:, None] * base_gauss[None, :])
    base_weight = base_kernel_2d.view(1, 1, kernel_size, kernel_size)

    content_padded = F.pad(content, (pad, pad, pad, pad), mode='reflect')
    style_padded = F.pad(style, (pad, pad, pad, pad), mode='reflect')
    result = torch.zeros_like(content)

    scaling = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    sigma_scale = torch.ones((H, W), device=device, dtype=torch.float32)
    if mask is not None:
        padded_mask = F.pad(mask.float(), (1, 1, 1, 1), mode="reflect")
        blurred_mask = F.avg_pool2d(padded_mask, kernel_size=3, stride=1, padding=1)
        blurred_mask = blurred_mask[..., 1:-1, 1:-1]
        edge_proximity = blurred_mask * (1.0 - blurred_mask)
        scaling = 1.0 - (edge_proximity / 0.25).clamp(0.0, 1.0)
        sigma_scale = scaling[0, 0]  # assuming single-channel mask broadcasted across B, C

    for i in range(H):
        c_row_patches = torch.stack([
            content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)

        s_row_patches = torch.stack([
            style_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            for j in range(W)
        ], dim=0)

        local_sigma = sigma * sigma_scale[i]  # [W]
        local_weight = []
        for w_sigma in local_sigma:
            coords_local = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
            gauss_local = torch.exp(-0.5 * (coords_local / (w_sigma + eps)) ** 2)
            gauss_local = (gauss_local / gauss_local.sum()).to(dtype)
            kernel_2d = gauss_local[:, None] * gauss_local[None, :]
            local_weight.append(kernel_2d.view(1, 1, kernel_size, kernel_size))
        local_weight = torch.stack(local_weight, dim=0).expand(W, C, kernel_size, kernel_size)

        c_mean = (c_row_patches * local_weight).sum(dim=(-1, -2), keepdim=True)
        c_std  = ((c_row_patches - c_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps
        s_mean = (s_row_patches * local_weight).sum(dim=(-1, -2), keepdim=True)
        s_std  = ((s_row_patches - s_mean) ** 2 * local_weight).sum(dim=(-1, -2), keepdim=True).sqrt() + eps

        center = kernel_size // 2
        central = c_row_patches[:, :, :, center:center+1, center:center+1]
        normed = (central - c_mean) / c_std
        stylized = normed * s_std + s_mean

        local_scaling = scaling[:, :, i, :].permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
        stylized = central * (1 - local_scaling) + stylized * local_scaling

        result[:, :, i, :] = stylized.squeeze(-1).squeeze(-1).permute(1, 2, 0)

    return result

"""



def adain_patchwise_row_batch_mask(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None) -> torch.Tensor:
    B, C, H, W = content.shape
    device, dtype = content.device, content.dtype

    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    base_gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
    base_gauss = (base_gauss / base_gauss.sum()).to(dtype)
    base_kernel_2d = (base_gauss[:, None] * base_gauss[None, :])
    base_weight = base_kernel_2d.view(1, 1, kernel_size, kernel_size)

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

    coords_local = torch.arange(kernel_size, dtype=torch.float64, device=device) - pad
    gaussian_table = {}
    for s in sigma_scale.unique():
        sig = float((sigma * s + eps).clamp(min=1e-3))
        gauss_local = torch.exp(-0.5 * (coords_local / sig) ** 2)
        gauss_local = (gauss_local / gauss_local.sum()).to(dtype)
        kernel_2d = gauss_local[:, None] * gauss_local[None, :]
        gaussian_table[s.item()] = kernel_2d

    for i in range(H):
        row_result = torch.zeros(B, C, W, dtype=dtype, device=device)
        for j in range(W):
            c_patch = content_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            s_patch = style_padded[:, :, i:i+kernel_size, j:j+kernel_size]
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

