
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor, FloatTensor
from typing import Optional, Callable, Tuple, Dict, List, Any, Union

import einops 
from einops import rearrange
import copy
import comfy


from .latents import gaussian_blur_2d, median_blur_2d

# WIP... not yet in use...
class StyleTransfer:  
    def __init__(self,
        style_method  = "WCT",
        embedder_method = None,
        patch_size    = 1,
        pinv_dtype    = torch.float64,
        dtype         = torch.float64,
    ):
        self.style_method  = style_method
        
        self.embedder_method   = None
        self.unembedder_method = None

        if embedder_method is not None:
            self.set_embedder_method(embedder_method)
        
        self.patch_size    = patch_size
        
        #if embedder_type == "conv2d":
        #    self.unembedder = self.invert_conv2d
        self.pinv_dtype = pinv_dtype
        self.dtype      = dtype
        
        self.patchify   = None
        self.unpatchify = None
        
        self.orig_shape = None
        self.grid_sizes = None
        
        #self.x_embed_ndim = 0
        
        

    def set_patchify_method(self, patchify_method=None):
        self.patchify_method = patchify_method

    def set_unpatchify_method(self, unpatchify_method=None):
        self.unpatchify_method = unpatchify_method
        
    def set_embedder_method(self, embedder_method):
        self.embedder_method = copy.deepcopy(embedder_method).to(self.pinv_dtype)
        self.W = self.embedder_method.weight
        self.B = self.embedder_method.bias    
        
        if   isinstance(embedder_method, nn.Linear):
            self.unembedder_method = self.invert_linear
        
        elif isinstance(embedder_method, nn.Conv2d):
            self.unembedder_method = self.invert_conv2d
            
        elif isinstance(embedder_method, nn.Conv3d):
            self.unembedder_method = self.invert_conv3d
            
    def set_patch_size(self, patch_size):
        self.patch_size = patch_size

    def unpatchify(self, x: Tensor) -> List[Tensor]:
        x_arr = []
        for i, img_size in enumerate(self.img_sizes):   #  [[64,64]]   , img_sizes: List[Tuple[int, int]]
            pH, pW = img_size
            x_arr.append(
                einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)',
                    p1=self.patch_size, p2=self.patch_size)
            )
        x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, x: Tensor):
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        
        pH, pW         = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        self.img_sizes = [[pH, pW]] * x.shape[0]
        x              = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.patch_size, p2=self.patch_size)
        return x
        
        
    def embedder(self, x):
        if isinstance(self.embedder_method, nn.Linear):
            x = self.patchify(x)
            
        self.orig_shape = x.shape
        x = self.embedder_method(x)
        self.grid_sizes = x.shape[2:]
        
        #self.x_embed_ndim = x.ndim
        #if x.ndim > 3:
        #    x = einops.rearrange(x, "B C H W -> B (H W) C")
        
        return x
        
    def unembedder(self, x):
        #if self.x_embed_ndim > 3:
        #    x = einops.rearrange(x, "B (H W) C -> B C H W", W=self.orig_shape[-1])
            
        x = self.unembedder_method(x)
        return x
        
        
    def invert_linear(self, x : torch.Tensor,) -> torch.Tensor:
        x = x.to(self.pinv_dtype)
        #x = (x - self.B.to(self.dtype)) @ torch.linalg.pinv(self.W.to(self.pinv_dtype)).T.to(self.dtype)
        x = (x - self.B) @ torch.linalg.pinv(self.W).T
        
        return x.to(self.dtype)

        
        
    def invert_conv2d(self, z: torch.Tensor,) -> torch.Tensor:
        z = z.to(self.pinv_dtype)
        conv = self.embedder_method
        
        B, C_in, H, W      = self.orig_shape
        C_out, _, kH, kW   = conv.weight.shape
        stride_h, stride_w = conv.stride
        pad_h,    pad_w    = conv.padding

        b = conv.bias.view(1, C_out, 1, 1).to(z)
        z_nobias = z - b

        W_flat = conv.weight.view(C_out, -1).to(z)  
        W_pinv = torch.linalg.pinv(W_flat)    

        Bz, Co, Hp, Wp = z_nobias.shape
        z_flat = z_nobias.reshape(Bz, Co, -1)  

        x_patches = W_pinv @ z_flat   

        x_sum = F.fold(
            x_patches,
            output_size=(H + 2*pad_h, W + 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )
        ones = torch.ones_like(x_patches)
        count = F.fold(
            ones,
            output_size=(H + 2*pad_h, W + 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )  

        x_recon = x_sum / count.clamp(min=1e-6)
        if pad_h > 0 or pad_w > 0:
            x_recon = x_recon[..., pad_h:pad_h+H, pad_w:pad_w+W]

        return x_recon.to(self.dtype)



    def invert_conv3d(self, z: torch.Tensor, ) -> torch.Tensor:
        z = z.to(self.pinv_dtype)
        conv = self.embedder_method
        grid_sizes = self.grid_sizes

        B, C_in, D, H, W = self.orig_shape
        pD, pH, pW = self.patch_size
        sD, sH, sW = pD, pH, pW

        if z.ndim == 3:
            # [B, S, C_out] -> reshape to [B, C_out, D', H', W']   
            S = z.shape[1]
            if grid_sizes is None:
                Dp = D // pD
                Hp = H // pH   # getting actual patchified dims
                Wp = W // pW
            else:
                Dp, Hp, Wp = grid_sizes
            C_out = z.shape[2]
            z = z.transpose(1, 2).reshape(B, C_out, Dp, Hp, Wp)
        else:
            B2, C_out, Dp, Hp, Wp = z.shape
            assert B2 == B, "Batch size mismatch... ya sharked it."

        b = conv.bias.view(1, C_out, 1, 1, 1)         # need to kncokout bias to invert via weight
        z_nobias = z - b

        # 2D filter -> pinv
        w3 = conv.weight         # [C_out, C_in, 1, pH, pW]
        w2 = w3.squeeze(2)                       # [C_out, C_in, pH, pW]
        out_ch, in_ch, kH, kW = w2.shape
        W_flat = w2.view(out_ch, -1)            # [C_out, in_ch*pH*pW]
        W_pinv = torch.linalg.pinv(W_flat)      # [in_ch*pH*pW, C_out]

        # merge depth for 2D unfold wackiness
        z2 = z_nobias.permute(0,2,1,3,4).reshape(B*Dp, C_out, Hp, Wp)

        # apply pinv ... get patch vectors
        z_flat    = z2.reshape(B*Dp, C_out, -1)  # [B*Dp, C_out, L]
        x_patches = W_pinv @ z_flat              # [B*Dp, in_ch*pH*pW, L]

        # fold -> restore spatial frames
        x2 = F.fold(
            x_patches,
            output_size=(H, W),
            kernel_size=(pH, pW),
            stride=(sH, sW)
        )  # → [B*Dp, C_in, H, W]

        # unmerge depth (de-depth charge)
        x2 = x2.reshape(B, Dp, in_ch, H, W)           # [B, Dp,  C_in, H, W]
        x_recon = x2.permute(0,2,1,3,4).contiguous()  # [B, C_in,   D, H, W]
        return x_recon.to(self.dtype)



    def adain_seq_inplace(self, content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        mean_c = content.mean(1, keepdim=True)
        std_c  = content.std (1, keepdim=True).add_(eps) 
        mean_s = style.mean  (1, keepdim=True)
        std_s  = style.std   (1, keepdim=True).add_(eps)

        content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)
        return content






class StyleWCT:  
    def __init__(self, dtype=torch.float64, use_svd=False,):
        self.dtype          = dtype
        self.use_svd        = use_svd
        self.y0_adain_embed = None
        self.mu_s           = None
        self.y0_color       = None
        
    def whiten(self, f_s_centered: torch.Tensor, set=False):
        cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

        if self.use_svd:
            U_svd, S_svd, Vh_svd = torch.linalg.svd(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
            S_eig = S_svd
            U_eig = U_svd
        else:
            S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
            
        if set:
            S_eig_root = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
        else:
            S_eig_root = S_eig.clamp(min=0).rsqrt() # inverse square root
        
        whiten = U_eig @ torch.diag(S_eig_root) @ U_eig.T
        return whiten.to(f_s_centered)

    def set(self, y0_adain_embed: torch.Tensor):
        if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
            f_s          = y0_adain_embed[0] # if y0_adain_embed.ndim > 4 else y0_adain_embed
            self.mu_s    = f_s.mean(dim=0, keepdim=True)
            f_s_centered = f_s - self.mu_s
            
            self.y0_color = self.whiten(f_s_centered, set=True)
            
    def get(self, denoised_embed: torch.Tensor):
        for wct_i in range(denoised_embed.shape[0]):
            f_c          = denoised_embed[wct_i]
            mu_c         = f_c.mean(dim=0, keepdim=True)
            f_c_centered = f_c - mu_c

            whiten = self.whiten(f_c_centered)

            f_c_whitened = f_c_centered @ whiten.T
            f_cs         = f_c_whitened @ self.y0_color.T + self.mu_s
            
            denoised_embed[wct_i] = f_cs
            
        return denoised_embed




"""

class StyleFeatures:  
    def __init__(self, dtype=torch.float64,):
        self.dtype = dtype

    def set(self, y0_adain_embed: torch.Tensor):
            
    def get(self, denoised_embed: torch.Tensor):

        return "Norpity McNerp"

"""




class Retrojector:  
    def __init__(self, proj=None, patch_size=2, pinv_dtype=torch.float64, dtype=torch.float64,):
        self.proj       = proj
        self.patch_size = patch_size
        self.pinv_dtype = pinv_dtype
        self.dtype      = dtype
        
        self.LINEAR     = isinstance(proj, nn.Linear)
        self.CONV2D     = isinstance(proj, nn.Conv2d)
        self.CONV3D     = isinstance(proj, nn.Conv3d)
        
        self.W          = proj.weight.data.to(dtype=pinv_dtype).cuda()
        
        if self.LINEAR:
            self.W_inv = torch.linalg.pinv(self.W.cuda())
        elif self.CONV2D:
            C_out, _, kH, kW = proj.weight.shape
            W_flat = proj.weight.view(C_out, -1).to(dtype=pinv_dtype)
            self.W_inv = torch.linalg.pinv(W_flat.cuda())
        
        if proj.bias is None:
            if self.LINEAR:
                bias_size = proj.out_features
            else:
                bias_size = proj.out_channels
            self.b = torch.zeros(bias_size, dtype=pinv_dtype, device=self.W_inv.device)
        else:
            self.b = proj.bias.data.to(dtype=pinv_dtype).to(self.W_inv.device)
        
    def embed(self, img: torch.Tensor):
        self.h = img.shape[-2] // self.patch_size
        self.w = img.shape[-1] // self.patch_size
        
        img = comfy.ldm.common_dit.pad_to_patch_size(img, (self.patch_size, self.patch_size))
        
        if   self.CONV2D:
            self.orig_shape = img.shape  # for unembed
            img_embed = F.conv2d(
                img.to(self.W), 
                weight=self.W, 
                bias=self.b, 
                stride=self.proj.stride, 
                padding=self.proj.padding
            )
            #img_embed = rearrange(img_embed, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size) 
            img_embed = rearrange(img_embed, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=1, pw=1) 
        
        elif self.LINEAR:
            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size) 
            img_embed = F.linear(img.to(self.W), self.W, self.b)
        
        return img_embed.to(img)
    
    def unembed(self, img_embed: torch.Tensor):
        if   self.CONV2D:
            #img_embed = rearrange(img_embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=self.patch_size, pw=self.patch_size)
            img_embed = rearrange(img_embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=1, pw=1)
            img = self.invert_conv2d(img_embed)
        
        elif self.LINEAR:
            img = F.linear(img_embed.to(self.b) - self.b, self.W_inv)
            img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=self.patch_size, pw=self.patch_size)
        
        return img.to(img_embed)
    
    def invert_conv2d(self, z: torch.Tensor,) -> torch.Tensor:
        z_dtype = z.dtype
        z = z.to(self.pinv_dtype)
        conv = self.proj
        
        B, C_in, H, W      = self.orig_shape
        C_out, _, kH, kW   = conv.weight.shape
        stride_h, stride_w = conv.stride
        pad_h,    pad_w    = conv.padding

        b = conv.bias.view(1, C_out, 1, 1).to(z)
        z_nobias = z - b

        #W_flat = conv.weight.view(C_out, -1).to(z)  
        #W_pinv = torch.linalg.pinv(W_flat)    

        Bz, Co, Hp, Wp = z_nobias.shape
        z_flat = z_nobias.reshape(Bz, Co, -1)  

        x_patches = self.W_inv @ z_flat   

        x_sum = F.fold(
            x_patches,
            output_size=(H + 2*pad_h, W+ 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )
        ones = torch.ones_like(x_patches)
        count = F.fold(
            ones,
            output_size=(H + 2*pad_h, W + 2*pad_w),
            kernel_size=(kH, kW),
            stride=(stride_h, stride_w),
        )  

        x_recon = x_sum / count.clamp(min=1e-6)
        if pad_h > 0 or pad_w > 0:
            x_recon = x_recon[..., pad_h:pad_h+H, pad_w:pad_w+W]

        return x_recon.to(z_dtype)
    
    def invert_patch_embedding(self, z: torch.Tensor, original_shape: torch.Size, grid_sizes: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:

        B, C_in, D, H, W = original_shape
        pD, pH, pW = self.patch_size
        sD, sH, sW = pD, pH, pW

        if z.ndim == 3:
            # [B, S, C_out] -> reshape to [B, C_out, D', H', W']
            S = z.shape[1]
            if grid_sizes is None:
                Dp = D // pD
                Hp = H // pH
                Wp = W // pW
            else:
                Dp, Hp, Wp = grid_sizes
            C_out = z.shape[2]
            z = z.transpose(1, 2).reshape(B, C_out, Dp, Hp, Wp)
        else:
            B2, C_out, Dp, Hp, Wp = z.shape
            assert B2 == B, "Batch size mismatch... ya sharked it."

        # kncokout bias
        b = self.patch_embedding.bias.view(1, C_out, 1, 1, 1)
        z_nobias = z - b

        # 2D filter -> pinv
        w3 = self.patch_embedding.weight         # [C_out, C_in, 1, pH, pW]
        w2 = w3.squeeze(2)                       # [C_out, C_in, pH, pW]
        out_ch, in_ch, kH, kW = w2.shape
        W_flat = w2.view(out_ch, -1)            # [C_out, in_ch*pH*pW]
        W_pinv = torch.linalg.pinv(W_flat)      # [in_ch*pH*pW, C_out]

        # merge depth for 2D unfold wackiness
        z2 = z_nobias.permute(0,2,1,3,4).reshape(B*Dp, C_out, Hp, Wp)

        # apply pinv ... get patch vectors
        z_flat    = z2.reshape(B*Dp, C_out, -1)  # [B*Dp, C_out, L]
        x_patches = W_pinv @ z_flat              # [B*Dp, in_ch*pH*pW, L]

        # fold -> spatial frames
        x2 = F.fold(
            x_patches,
            output_size=(H, W),
            kernel_size=(pH, pW),
            stride=(sH, sW)
        )  # → [B*Dp, C_in, H, W]

        # un-merge depth
        x2 = x2.reshape(B, Dp, in_ch, H, W)           # [B, Dp,  C_in, H, W]
        x_recon = x2.permute(0,2,1,3,4).contiguous()  # [B, C_in,   D, H, W]
        return x_recon






def invert_conv2d(
    conv: torch.nn.Conv2d,
    z:    torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    import torch.nn.functional as F

    B, C_in, H, W = original_shape
    C_out, _, kH, kW = conv.weight.shape
    stride_h, stride_w = conv.stride
    pad_h,    pad_w    = conv.padding

    if conv.bias is not None:
        b = conv.bias.view(1, C_out, 1, 1).to(z)
        z_nobias = z - b
    else:
        z_nobias = z

    W_flat = conv.weight.view(C_out, -1).to(z)  
    W_pinv = torch.linalg.pinv(W_flat)    

    Bz, Co, Hp, Wp = z_nobias.shape
    z_flat = z_nobias.reshape(Bz, Co, -1)  

    x_patches = W_pinv @ z_flat   

    x_sum = F.fold(
        x_patches,
        output_size=(H + 2*pad_h, W + 2*pad_w),
        kernel_size=(kH, kW),
        stride=(stride_h, stride_w),
    )
    ones = torch.ones_like(x_patches)
    count = F.fold(
        ones,
        output_size=(H + 2*pad_h, W + 2*pad_w),
        kernel_size=(kH, kW),
        stride=(stride_h, stride_w),
    )  

    x_recon = x_sum / count.clamp(min=1e-6)
    if pad_h > 0 or pad_w > 0:
        x_recon = x_recon[..., pad_h:pad_h+H, pad_w:pad_w+W]

    return x_recon



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
    return content

def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)









def apply_scattersort_tiled(
    denoised_spatial : torch.Tensor, 
    y0_adain_spatial : torch.Tensor, 
    tile_h           : int, 
    tile_w           : int, 
    pad              : int,
):
    """
    Apply spatial scattersort between denoised_spatial and y0_adain_spatial
    using local tile-wise sorted value matching.

    Args:
        denoised_spatial (Tensor): (B, C, H, W) tensor.
        y0_adain_spatial  (Tensor): (B, C, H, W) reference tensor.
        tile_h (int): tile height.
        tile_w (int): tile width.
        pad    (int): padding size to apply around tiles.

    Returns:
        denoised_embed (Tensor): (B, H*W, C) tensor after sortmatch.
    """
    denoised_padded = F.pad(denoised_spatial, (pad, pad, pad, pad), mode='reflect')
    y0_padded       = F.pad(y0_adain_spatial, (pad, pad, pad, pad), mode='reflect')

    denoised_padded_out = denoised_padded.clone()
    _, _, h_len, w_len = denoised_spatial.shape

    for ix in range(pad, h_len, tile_h):
        for jx in range(pad, w_len, tile_w):
            tile    = denoised_padded[:, :, ix - pad:ix + tile_h + pad, jx - pad:jx + tile_w + pad]
            y0_tile = y0_padded[:, :, ix - pad:ix + tile_h + pad, jx - pad:jx + tile_w + pad]

            tile    = rearrange(tile,    "b c h w -> b c (h w)", h=tile_h + pad * 2, w=tile_w + pad * 2)
            y0_tile = rearrange(y0_tile, "b c h w -> b c (h w)", h=tile_h + pad * 2, w=tile_w + pad * 2)

            src_sorted, src_idx =    tile.sort(dim=-1)
            ref_sorted, ref_idx = y0_tile.sort(dim=-1)

            new_tile = tile.scatter(dim=-1, index=src_idx, src=ref_sorted.expand(src_sorted.shape))
            new_tile = rearrange(new_tile, "b c (h w) -> b c h w", h=tile_h + pad * 2, w=tile_w + pad * 2)

            denoised_padded_out[:, :, ix:ix + tile_h, jx:jx + tile_w] = (
                new_tile if pad == 0 else new_tile[:, :, pad:-pad, pad:-pad]
            )

    denoised_padded_out = denoised_padded_out if pad == 0 else denoised_padded_out[:, :, pad:-pad, pad:-pad]
    return denoised_padded_out



def apply_scattersort_masked(
    denoised_embed         : torch.Tensor,
    y0_adain_embed         : torch.Tensor,
    y0_style_pos_mask      : torch.Tensor | None,
    y0_style_pos_mask_edge : torch.Tensor | None,
    h_len                  : int,
    w_len                  : int
):
    if y0_style_pos_mask is None:
        flatmask = torch.ones((1,1,h_len,w_len)).bool().flatten().bool()
    else:
        flatmask   = F.interpolate(y0_style_pos_mask, size=(h_len, w_len)).bool().flatten().cpu()
    flatunmask = ~flatmask

    if y0_style_pos_mask_edge is not None:
        edgemask = F.interpolate(
            y0_style_pos_mask_edge.unsqueeze(0), size=(h_len, w_len)
        ).bool().flatten()
        flatmask   = flatmask   & (~edgemask)
        flatunmask = flatunmask & (~edgemask)

    denoised_masked = denoised_embed[:, flatmask, :].clone()
    y0_adain_masked = y0_adain_embed[:, flatmask, :].clone()

    src_sorted, src_idx = denoised_masked.sort(dim=-2)
    ref_sorted, ref_idx = y0_adain_masked.sort(dim=-2)

    denoised_embed[:, flatmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))

    if (flatunmask == True).any():
        denoised_unmasked = denoised_embed[:, flatunmask, :].clone()
        y0_adain_unmasked = y0_adain_embed[:, flatunmask, :].clone()

        src_sorted, src_idx = denoised_unmasked.sort(dim=-2)
        ref_sorted, ref_idx = y0_adain_unmasked.sort(dim=-2)

        denoised_embed[:, flatunmask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))

    if y0_style_pos_mask_edge is not None:
        denoised_edgemasked = denoised_embed[:, edgemask, :].clone()
        y0_adain_edgemasked = y0_adain_embed[:, edgemask, :].clone()

        src_sorted, src_idx = denoised_edgemasked.sort(dim=-2)
        ref_sorted, ref_idx = y0_adain_edgemasked.sort(dim=-2)

        denoised_embed[:, edgemask, :] = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))

    return denoised_embed




def apply_scattersort(
    denoised_embed         : torch.Tensor,
    y0_adain_embed         : torch.Tensor,
):
    src_sorted, src_idx = denoised_embed.sort(dim=-2)
    ref_sorted, ref_idx = y0_adain_embed.sort(dim=-2)

    denoised_embed = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))
    return denoised_embed

def apply_scattersort_spatial(
    denoised_spatial         : torch.Tensor,
    y0_adain_spatial         : torch.Tensor,
):
    denoised_embed = rearrange(denoised_spatial, "b c h w -> b (h w) c")
    y0_adain_embed = rearrange(y0_adain_spatial, "b c h w -> b (h w) c")
    src_sorted, src_idx = denoised_embed.sort(dim=-2)
    ref_sorted, ref_idx = y0_adain_embed.sort(dim=-2)

    denoised_embed = src_sorted.scatter(dim=-2, index=src_idx, src=ref_sorted.expand(src_sorted.shape))
    
    return rearrange(denoised_embed, "b (h w) c -> b c h w", h=denoised_spatial.shape[-2], w=denoised_spatial.shape[-1])


























def adain_patchwise(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5) -> torch.Tensor:
    # this one is really slow
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



def adain_patchwise_row_batch_med(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False, lowpass_weight=1.0, highpass_weight=1.0) -> torch.Tensor:
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




