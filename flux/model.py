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

from einops import rearrange, repeat
import comfy.ldm.common_dit

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
        
        weight    = transformer_options['reg_cond_weight'] if 'reg_cond_weight' in transformer_options else 0.0
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
            
            mask[text_len:,text_len:] = torch.clamp(mask[text_len:,text_len:], min=floor.to(mask.device))   #ORIGINAL SELF-ATTN REGION BLEED

        total_layers = len(self.double_blocks) + len(self.single_blocks)
        
        for i, block in enumerate(self.double_blocks):
            if mask is not None and mask_type_bool and weight < (i / (total_layers-1)):
                mask = mask.to(img.dtype)

            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, mask=mask, idx=i, update_cross_attn=update_cross_attn)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add

        img = torch.cat((txt, img), 1)   #first 256 is txt embed
        for i, block in enumerate(self.single_blocks):
            if mask is not None and mask_type_bool and weight < ((len(self.double_blocks) + i) / (total_layers-1)):
                mask = mask.to(img.dtype)
            
            img = block(img, vec=vec, pe=pe, mask=mask, idx=i)

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
                **kwargs
                ):
        SIGMA = timestep[0].unsqueeze(0)
        update_cross_attn = transformer_options.get("update_cross_attn")
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")


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

            if UNCOND:
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
            context_tmp = context_tmp.to(context.dtype)
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
                                        transformer_options=transformer_options)  # context 1,256,4096   y 1,768
            out_list.append(out_tmp)
            
        out = torch.stack(out_list, dim=0).squeeze(dim=1)
        #return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        eps = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1 and not UNCOND):
            if y0_style_pos is not None:
                y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
                y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
                y0_style_pos_synweight *= y0_style_pos_weight
                
                y0_style_pos = y0_style_pos.to(torch.float32)
                x   = x.to(torch.float32)
                eps = eps.to(torch.float32)
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

                W = self.img_in.weight.data.to(torch.float32)   # shape [2560, 64]
                b = self.img_in.bias.data.to(torch.float32)     # shape [2560]
                
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

                
                denoised_approx = (denoised_embed - b.to(denoised_embed)) @ torch.linalg.pinv(W).T.to(denoised_embed)
                denoised_approx = denoised_approx.to(eps)
                
                denoised_approx = rearrange(denoised_approx, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
                
                eps = (x - denoised_approx) / sigma
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
                
                eps = eps.float()
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1 and UNCOND):
            if y0_style_neg is not None:
                y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight")
                y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight")
                y0_style_neg_synweight *= y0_style_neg_weight
                
                y0_style_neg = y0_style_neg.to(torch.float32)
                x   = x.to(torch.float32)
                eps = eps.to(torch.float32)
                eps_orig = eps.clone()
                
                sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
                denoised = x - sigma * eps
                
                img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))

                h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
                w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
                img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
                
                img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_neg, (self.patch_size, self.patch_size))

                img_y0_adain = rearrange(img_y0_adain, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

                W = self.img_in.weight.data.to(torch.float32)   # shape [2560, 64]
                b = self.img_in.bias.data.to(torch.float32)     # shape [2560]
                
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

                denoised_approx = (denoised_embed - b.to(denoised_embed)) @ torch.linalg.pinv(W).T.to(denoised_embed)
                denoised_approx = denoised_approx.to(eps)
                
                denoised_approx = rearrange(denoised_approx, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
                
                eps = (x - denoised_approx) / sigma
                eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                
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



