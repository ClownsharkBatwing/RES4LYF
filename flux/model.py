# Adapted from: https://github.com/black-forest-labs/flux

import torch
from torch import Tensor, nn
from dataclasses import dataclass

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
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: int
    qkv_bias: bool
    guidance_embed: bool

class ReFlux(Flux):
    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        
        self.params = params #self.params FluxParams(in_channels=16, out_channels=16, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=False)
        self.patch_size = params.patch_size
        self.in_channels  = params.in_channels  * params.patch_size * params.patch_size    # in_channels 64
        self.out_channels = params.out_channels * params.patch_size * params.patch_size    # out_channels 64
        
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        
        self.hidden_size = params.hidden_size  # 3072
        self.num_heads   = params.num_heads    # 24
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        
        self.img_in = operations.Linear(     self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size,            dtype=dtype, device=device)

        self.time_in      = MLPEmbedder(           in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.vector_in    = MLPEmbedder(params.vec_in_dim,                self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.guidance_in = (MLPEmbedder(           in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity())

        self.double_blocks = nn.ModuleList([DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias, dtype=dtype, device=device, operations=operations, idx=_) for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio,                           dtype=dtype, device=device, operations=operations, idx=_) for _ in range(params.depth_single_blocks)])

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

        self.mask_fn
    
    
    def forward_blocks(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, timesteps: Tensor, y: Tensor, guidance: Tensor = None, control=None, transformer_options = {},) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)    # 1,9216,64  == 768x192
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                print("Guidance strength is none, not using distilled guidance.")
            else:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        weight = transformer_options['regional_conditioning_weight'] if 'regional_conditioning_weight' in transformer_options else 0.0

        for i, block in enumerate(self.double_blocks):
            mask = None
            if "mask_fn" in transformer_options and weight >= i/56:
                mask = transformer_options['mask_fn'](transformer_options)

            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, timestep=timesteps, transformer_options=transformer_options, mask=mask) #, mask=mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img[:1] += add

        img = torch.cat((txt, img), 1)   #first 256 is txt embed
        for i, block in enumerate(self.single_blocks):
            mask = None
            if "mask_fn" in transformer_options and weight >= (i+18)/56: #/38:
                mask = transformer_options['mask_fn'](transformer_options)
            
            img = block(img, vec=vec, pe=pe, timestep=timesteps, transformer_options=transformer_options, mask=mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:1, txt.shape[1] :, ...] += add
                        
        img = img[:, txt.shape[1] :, ...]
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
    
    def _get_img_ids(self, x, bs, h_len, w_len, h_start, h_end, w_start, w_end):
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(h_start, h_end - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(w_start, w_end - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        return img_ids

    def forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        transformer_options['original_shape'] = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))    # 1,16,192,192
        transformer_options['patch_size'] = patch_size
        
        if 'regional_conditioning_weight' not in transformer_options:
            transformer_options['regional_conditioning_weight'] = timestep[0] / 1.5
        #transformer_options['unsample_resample_scale'] = transformer_options.get('patches', {}).get('unsample_resample_scale', None)
            
        h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
        w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64

        #regional_conditioning = 
        regional_conditioning_mask = transformer_options.get('patches', {}).get('regional_conditioning_mask', None)
        if regional_conditioning_mask is not None:
            transformer_options['mask_fn'] = regional_conditioning_mask[0]

        regional_conditioning_positive = transformer_options.get('patches', {}).get('regional_conditioning_positive', None)
        regional_conditioning_negative = transformer_options.get('patches', {}).get('regional_conditioning_negative', None)
        regional_conditioning = regional_conditioning_positive
        if regional_conditioning is not None:
            region_cond = regional_conditioning[0](transformer_options)
            if region_cond is not None:
                if region_cond.shape[0] < context.shape[0]:
                    newtensor    = torch.zeros((context.shape[0], region_cond.shape[1], region_cond.shape[2]), dtype=context.dtype, device=context.device)
                    newtensor[1] = region_cond[0].clone()
                    newtensor[0] = torch.zeros_like(region_cond[0])
                    region_cond  = newtensor
                context = torch.cat([context, region_cond.to(context.dtype)], dim=1)

        txt_ids      = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)      # txt_ids        1, 256,3
        img_ids_orig = self._get_img_ids(x, bs, h_len, w_len, 0, h_len, 0, w_len)                  # img_ids_orig = 1,9216,3
        out = self.forward_blocks(img, img_ids_orig, context, txt_ids, timestep, y, guidance, control, transformer_options=transformer_options)  # context 1,256,4096   y 1,768
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
