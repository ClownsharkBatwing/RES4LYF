# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from einops import repeat

from comfy.ldm.modules.attention import optimized_attention, attention_pytorch
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.diffusionmodules.mmdit import RMSNorm
import comfy.ldm.common_dit
import comfy.model_management


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half     = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class ReWanSelfAttention(nn.Module):

    def __init__(self,
                dim,
                num_heads,
                window_size        = (-1, -1),
                qk_norm            = True,
                eps                = 1e-6, 
                operation_settings = {}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.window_size = window_size
        self.qk_norm     = qk_norm
        self.eps         = eps

        # layers
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, freqs, mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q, k    = apply_rope(q, k, freqs)
        # q,k.shape = 2,14040,12,128      v.shape = 2,14040,1536
        
        if mask is not None:
            dtype = mask.dtype if mask.dtype == torch.bool else q.dtype
            txt_len = mask.shape[1] - mask.shape[0]
            x = attention_pytorch(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
                mask=mask[:,txt_len:].to(dtype)
            )
        else:
            x = optimized_attention(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
            )

        x = self.o(x)
        return x


class ReWanT2VCrossAttention(ReWanSelfAttention):

    def forward(self, x, context, mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v =             self.v(context)
        #if mask is not None:
        #    num_repeats = q.shape[1] // mask.shape[0]
        #    mask = mask.repeat(num_repeats, 1)
        # compute attention    # x.shape 2,14040,1536     q.shape 2,14040,1536     k,v.shape = 2,512,1536       mask = 14040,512   num_heads=12
        if mask is not None and (mask.shape[-1] - mask.shape[-2]) == k.shape[-2]:  # need mask shape 11664,5120
            #dtype = mask.dtype if mask.dtype == torch.bool else q.dtype
            dtype = torch.bool
            x = attention_pytorch(q, k, v, heads=self.num_heads, mask=mask[:,:k.shape[-2]].to(q.device).to(dtype))
        else:
            x = optimized_attention(q, k, v, heads=self.num_heads, mask=None)

        x = self.o(x)
        return x


class ReWanI2VCrossAttention(ReWanSelfAttention):   # image2video only

    def __init__(self,
                dim,
                num_heads,
                window_size=(-1, -1),
                qk_norm=True,
                eps=1e-6, operation_settings={}):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)

        self.k_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, context, mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        context_img = context[:, :257]
        context     = context[:, 257:]

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads)
        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)

        # output
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': ReWanT2VCrossAttention,
    'i2v_cross_attn': ReWanI2VCrossAttention,
}


class ReWanAttentionBlock(nn.Module):

    def __init__(self,
                cross_attn_type,
                dim,
                ffn_dim,
                num_heads,
                window_size        =  (-1, -1),
                qk_norm            =  True,
                cross_attn_norm    =  False,
                eps                =  1e-6, 
                operation_settings = {}):
        super().__init__()
        self.dim             = dim
        self.ffn_dim         = ffn_dim
        self.num_heads       = num_heads
        self.window_size     = window_size
        self.qk_norm         = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps             = eps

        # layers
        self.norm1     = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attn = ReWanSelfAttention(  dim, num_heads, window_size, qk_norm,
                                            eps, operation_settings=operation_settings)
        self.norm3 = operation_settings.get("operations").LayerNorm(
            dim, eps,
            elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if cross_attn_norm else nn.Identity()
        
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
                                                                        dim,
                                                                        num_heads,
                                                                        (-1, -1),
                                                                        qk_norm,
                                                                        eps, 
                                                                        operation_settings=operation_settings)
        
        self.norm2 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(dim, ffn_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operation_settings.get("operations").Linear(ffn_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(
        self,
        x,
        e,
        freqs,
        context,
        mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32

        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32
        # e = tuple with 6 elem, shape = 2,1,1536    # with length = 33 so 9 frames
        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            freqs,
            mask=mask)

        x = x + y * e[2]

        # cross-attention & ffn   # x,y.shape 2,14040,1536   
        x = x + self.cross_attn(self.norm3(x), context, mask=mask)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim        = dim
        self.out_dim    = out_dim
        self.patch_size = patch_size
        self.eps        = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.head = operation_settings.get("operations").Linear   (dim, out_dim,                       device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, operation_settings={}):
        super().__init__()

        self.proj = torch.nn.Sequential(
            operation_settings                 .get("operations").LayerNorm(in_dim,          device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), operation_settings.get("operations").Linear(in_dim, in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            torch.nn.GELU(), operation_settings.get("operations").Linear   (in_dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings                 .get("operations").LayerNorm(out_dim,         device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class ReWanModel(torch.nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                model_type      = 't2v',
                patch_size      = (1, 2, 2),
                text_len        = 512,
                in_dim          = 16,
                dim             = 2048,
                ffn_dim         = 8192,
                freq_dim        = 256,
                text_dim        = 4096,
                out_dim         = 16,
                num_heads       = 16,
                num_layers      = 32,
                window_size     = (-1, -1),
                qk_norm         = True,
                cross_attn_norm = True,
                eps             = 1e-6,
                image_model     = None,
                device          = None,
                dtype           = None,
                operations      = None,
                ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()
        self.dtype           = dtype
        operation_settings   = {"operations": operations, "device": device, "dtype": dtype}

        assert model_type in ['t2v', 'i2v']
        self.model_type      = model_type

        self.patch_size      = patch_size
        self.text_len        = text_len
        self.in_dim          = in_dim
        self.dim             = dim
        self.ffn_dim         = ffn_dim
        self.freq_dim        = freq_dim
        self.text_dim        = text_dim
        self.out_dim         = out_dim
        self.num_heads       = num_heads
        self.num_layers      = num_layers
        self.window_size     = window_size
        self.qk_norm         = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps             = eps

        # embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=operation_settings.get("device"), dtype=torch.float32)
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim,      device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.SiLU(), operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                                cross_attn_type,
                                dim, 
                                ffn_dim, num_heads,
                                window_size,
                                qk_norm,
                                cross_attn_norm,
                                eps, 
                                operation_settings=operation_settings)
            
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, operation_settings=operation_settings)
        else:
            self.img_emb = None

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea = None,
        freqs    = None,
        transformer_options = {},
        UNCOND = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # embeddings
        x          = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x          = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))              # e0.shape = 2,6,1536

        # context
        context = self.text_embedding(context)

        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context      = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e       = e0,
            freqs   = freqs,
            context = context)





        weight    = transformer_options['reg_cond_weight'] if 'reg_cond_weight' in transformer_options else 0.0
        floor     = transformer_options['reg_cond_floor']  if 'reg_cond_floor'  in transformer_options else 0.0
        
        floor     = min(floor, weight)
        
        mask_orig = None
        mask_self = None
        mask_obj  = transformer_options.get('patches', {}).get('regional_conditioning_mask', None)
        
        if type(weight) == float or type(weight) == int:
            pass
        else:
            weight = weight.item()
        
        if mask_obj is not None and weight >= 0:                 #THIS WAS WEIGHT >= 0 2-28-25

            mask_orig = mask_obj[0](transformer_options, weight)
            mask_self = mask_orig.clone()
            #mask_self[mask_obj[0].text_len:,   mask_obj[0].text_len:] = mask_self.max()

        mask     = None
        mask_obj = transformer_options.get('patches', {}).get('regional_conditioning_mask', None)
        if mask_obj is not None and weight >= 0:
            mask                      = mask_obj[0](transformer_options, weight)
            text_len                  = mask_obj[0].text_len
            if type(floor) == torch.Tensor:
                floor = floor.to(mask.device)
            mask[:,text_len:] = torch.clamp(mask[:,text_len:], min=floor)
        mask_type_bool = type(mask[0][0].item()) == bool if mask is not None else False







        i = 0
        for block in self.blocks:
            if mask_type_bool and weight < (i / (len(self.blocks)-1)) and mask is not None:
                mask = mask.to(x.dtype)
            
            x = block(x, mask=mask, **kwargs)
            
            i += 1

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x
    # context.shape = 2,512,1536     x.shape = 2,14040,1536      timestep.shape      h_len=30, w_len=52    30 * 52 = 1560
    def forward(self, x, timestep, context, clip_fea=None, transformer_options={}, **kwargs):
        
        if clip_fea is not None:
            bs, c, t, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
            patch_size = self.patch_size    # tuple = 1,2,2,
            
            t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
            h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
            w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
            
            img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
            # 14040 = 9 * 1560       1560 = 1536 + 24  1560/24 = 65
            freqs = self.rope_embedder(img_ids).movedim(1, 2)
            return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs)[:, :, :t, :h, :w]
        
        
        
        x_orig = x.clone()
        timestep_orig = timestep.clone()
        context_orig = context.clone()
        
        
        out_list = []
        for i in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][i] == 1

            x = x_orig.clone()
            timestep = timestep_orig.clone()
            context = context_orig.clone()


            bs, c, t, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
            patch_size = self.patch_size
            


            transformer_options['original_shape'] = x.shape
            transformer_options['patch_size']     = patch_size
            

            if UNCOND:
                transformer_options['reg_cond_weight'] = 0.0 # -1
                context_tmp = context[i][None,...].clone()
            
            elif UNCOND == False:
                transformer_options['reg_cond_weight'] = transformer_options.get("regional_conditioning_weight", 0.0) #transformer_options['regional_conditioning_weight']
                transformer_options['reg_cond_floor']  = transformer_options.get("regional_conditioning_floor", 0.0) #transformer_options['regional_conditioning_floor'] #if "regional_conditioning_floor" in transformer_options else 0.0
                regional_conditioning_positive         = transformer_options.get('patches', {}).get('regional_conditioning_positive', None)
                
                if regional_conditioning_positive is None or transformer_options['reg_cond_weight'] <= 0.0:
                    context_tmp = context[i][None,...].clone()
                else:
                    context_tmp = regional_conditioning_positive[0].concat_cond(context[i][None,...], transformer_options)
            
            if context_tmp is None:
                context_tmp = context[i][None,...].clone()
            context_tmp = context_tmp.to(context.dtype)
            
            
            
            t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
            h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
            w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
            
            img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
            # 14040 = 9 * 1560       1560 = 1536 + 24  1560/24 = 65
            freqs = self.rope_embedder(img_ids).movedim(1, 2)
            
            
            
            
            out_x = self.forward_orig(
                                        x[i][None,...], 
                                        timestep[i][None,...], 
                                        context_tmp,
                                        clip_fea=clip_fea,
                                        freqs=freqs[i][None,...],
                                        transformer_options=transformer_options,
                                        UNCOND=UNCOND,
                                        )[:, :, :t, :h, :w]
        
            out_list.append(out_x)
        
        out_stack = torch.stack(out_list, dim=0).squeeze(dim=1)

        return out_stack
        

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [L, C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
    
        
        
        return u
