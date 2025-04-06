# Adapted from: https://github.com/black-forest-labs/flux

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, Callable, Tuple, Dict, Any, Union

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

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.model import Flux as Flux

from einops import rearrange, repeat
import comfy.ldm.common_dit

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
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels,                                                                      dtype=dtype, device=device, operations=operations)


    
    
    def forward_blocks(self,
                        img      : Tensor,
                        img_ids  : Tensor,
                        txt      : Tensor,
                        txt_ids  : Tensor,
                        timesteps: Tensor,
                        y        : Tensor,
                        guidance : Tensor   = None,
                        control             = None,
                        reg_vec             = None,
                        transformer_options = {},
                        ) -> Tensor:
        
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)    # 1,9216,64  == 768x192       # 1,9216,64   ==   1,16,128,256 + 1,16,64,64    # 1,8192,64 with uncond/cond   #:,:,64 -> :,:,3072
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype)) # 1 -> 1,3072
        
        if self.params.guidance_embed:
            if guidance is None:
                print("Guidance strength is none, not using distilled guidance.")
            else:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec_orig = vec.clone()
        vec = vec + self.vector_in(y)  #y.shape=1,768  y==all 0s
        
        txt_a, txt_b = None, None
        if txt.shape[1] > 256:
            txt_a = self.txt_in(txt[:,:256,:])
            txt_b = self.txt_in(txt[:,256:,:])
            #txt_a = self.txt_in(txt[:,0:1,:])
            #txt_b = self.txt_in(txt[:,256:257:,:])
        
        txt = self.txt_in(txt)         #
        
        txt_ids_a = txt_ids[:,256:,:]
        txt_ids_b = txt_ids[:,:256,:]
        #txt_ids_a = txt_ids[:,0:1,:]
        #txt_ids_b = txt_ids[:,:256:257,:]
        
        ids_a = torch.cat((txt_ids_a, img_ids), dim=1)
        ids_b = torch.cat((txt_ids_b, img_ids), dim=1)
        pe_a  = self.pe_embedder(ids_a)
        pe_b  = self.pe_embedder(ids_b)
        
        if reg_vec is not None:
            reg_vec_a = vec_orig + self.vector_in(reg_vec[:,:768].to(torch.bfloat16))
            reg_vec_b = vec_orig + self.vector_in(reg_vec[:,768:1536].to(torch.bfloat16))
            #reg_vec_c = vec_orig + self.vector_in(reg_vec[:,1536:2304].to(torch.bfloat16))
            #reg_vec = torch.cat((reg_vec_a, reg_vec_b), dim=-1)

        ids = torch.cat((txt_ids, img_ids), dim=1) # img_ids.shape=1,8192,3    txt_ids.shape=1,512,3    #ids.shape=1,8704,3
        pe  = self.pe_embedder(ids)                 # pe.shape 1,1,8704,64,2,2
        
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
                mask = mask.to(img.dtype)
            
            text_len                  = txt.shape[1] # mask_obj[0].text_len
            
            mask[text_len:,text_len:] = torch.clamp(mask[text_len:,text_len:], min=floor.to(mask.device))   #ORIGINAL SELF-ATTN REGION BLEED
            reg_cond_mask = reg_cond_mask_expanded.unsqueeze(0).clone() if reg_cond_mask_expanded is not None else None

        #heatmaps_a, heatmaps_b = [], []
        #cdicts = []
        
        total_layers = len(self.double_blocks) + len(self.single_blocks)
        img_a, img_b = img.clone(), img.clone()
        
        for i, block in enumerate(self.double_blocks):
            if mask is not None and mask_type_bool and weight < (i / (total_layers-1)):
                mask = mask.to(img.dtype)

            img, txt, attn_mask = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask, reg_cond_mask=reg_cond_mask, idx=i) #, weight=weight) #, mask=mask)
            #img_a, txt_a, _, _ = block(img=img.clone(), txt=txt_a, vec=reg_vec_a, pe=pe_a, mask=None, reg_cond_mask=None, reg_vec=None) #, mask=mask)
            #img_b, txt_b, _, _ = block(img=img.clone(), txt=txt_b, vec=reg_vec_b, pe=pe_b, mask=None, reg_cond_mask=None, reg_vec=None) #, mask=mask)
            
            #img, txt, heatmap_a, heatmap_b = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask, reg_cond_mask=reg_cond_mask, reg_vec=reg_vec, vec_a=reg_vec_a, vec_b=reg_vec_b, txt_a=txt_a, txt_b=txt_b, pe_a=pe_a, pe_b=pe_b, ) #, mask=mask)
            
            #img, txt, txt_a, c_attention_dict = block(img=img, txt=txt, vec=vec, pe=pe, c_txt=txt_a, c_vec=reg_vec_a, c_pe=pe_a) #, mask=mask)

            

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add
            #heatmaps_a.append(heatmap_a)
            #heatmaps_b.append(heatmap_b)
            #cdicts.append(c_attention_dict)

        img = torch.cat((txt, img), 1)   #first 256 is txt embed
        for i, block in enumerate(self.single_blocks):
            if mask is not None and mask_type_bool and weight < ((len(self.double_blocks) + i) / (total_layers-1)):
                mask = mask.to(img.dtype)
            
            #img = block(img, vec=vec, pe=pe, timestep=timesteps, transformer_options=transformer_options, mask=mask, weight=weight)
            img = block(img, vec=vec, pe=pe, mask=mask, reg_cond_mask=reg_cond_mask, reg_vec=reg_vec, idx=i)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:1, txt.shape[1] :, ...] += add
                        
                        
                        
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
                cs:    Optional[Tensor] = None,
                c_ids: Optional[Tensor] = None,
                c_vec: Optional[Tensor] = None,
                **kwargs
                ):

        #y_orig = y.clone()
        out_list = []
        for i in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][i] == 1

            bs, c, h, w = x.shape
            patch_size  = 2
            x           = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))    # 1,16,192,192
            #y = y_orig.clone()
            #vec_tmp = None
            
            transformer_options['original_shape'] = x.shape
            transformer_options['patch_size']     = patch_size

            h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
            w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96

            img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
            vec_full=None
            #if UNCOND:
            #    transformer_options['reg_cond_weight'] = 0.0 # -1
            #    context_tmp = context[i][None,...].clone()
            
            if UNCOND:
                #transformer_options['reg_cond_weight'] = -1
                #context_tmp = context[i][None,...].clone()
                
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) #transformer_options['regional_conditioning_weight']
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor",  0.0) #transformer_options['regional_conditioning_floor'] #if "regional_conditioning_floor" in transformer_options else 0.0
                transformer_options['reg_cond_mask_orig'] = transformer_options.get('regional_conditioning_mask_orig')
                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] > 0.0:
                    AttnMask.attn_mask_recast(x.dtype)
                    context_tmp = RegContext.get().to(context.dtype)
                    #context_tmp = 0 * context_tmp.clone()
                    
                    A = context[i][None,...].clone()
                    B = context_tmp
                    context_tmp = A.repeat(1, (B.shape[1] // A.shape[1]) + 1, 1)[:, :B.shape[1], :]

                else:
                    context_tmp = context[i][None,...].clone()
            
            
            elif UNCOND == False:
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) 
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor",  0.0) 
                transformer_options['reg_cond_mask_orig'] = transformer_options.get('regional_conditioning_mask_orig')
                                
                AttnMask   = transformer_options.get('AttnMask',   None)                    
                RegContext = transformer_options.get('RegContext', None)
                
                if AttnMask is not None and transformer_options['reg_cond_weight'] > 0.0:
                    AttnMask.attn_mask_recast(img.dtype)
                    context_tmp = RegContext.get()
                else:
                    context_tmp = context[i][None,...].clone()
            
            if context_tmp is None:
                context_tmp = context[i][None,...].clone()
            context_tmp = context_tmp.to(context.dtype)
            txt_ids      = torch.zeros((bs, context_tmp.shape[1], 3), device=x.device, dtype=x.dtype)      # txt_ids        1, 256,3
            img_ids_orig = self._get_img_ids(x, bs, h_len, w_len, 0, h_len, 0, w_len)                  # img_ids_orig = 1,9216,3

            #vec_tmp = vec_tmp if vec_tmp is not None else y[i][None,...]

            out_tmp = self.forward_blocks(img       [i][None,...].clone(), 
                                        img_ids_orig[i][None,...].clone(), 
                                        context_tmp,
                                        txt_ids     [i][None,...].clone(), 
                                        timestep    [i][None,...].clone(), 
                                        #y,
                                        y           [i][None,...].clone(),
                                        guidance    [i][None,...].clone(),
                                        control, 
                                        vec_full, 
                                        transformer_options=transformer_options)  # context 1,256,4096   y 1,768
            out_list.append(out_tmp)
            
        out = torch.stack(out_list, dim=0).squeeze(dim=1)
        
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
    
