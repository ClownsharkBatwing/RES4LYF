
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
        self.spatial_shape  = None
        
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

    def set(self, y0_adain_embed: torch.Tensor, spatial_shape=None):
        if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
            self.y0_adain_embed = y0_adain_embed.clone()
            if spatial_shape is not None:
                self.spatial_shape = spatial_shape
            
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




class WaveletStyleWCT(StyleWCT):
    def set(self, y0_adain_embed: torch.Tensor, h_len, w_len):
        if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
            self.y0_adain_embed = y0_adain_embed.clone()
            
            B, HW, C = y0_adain_embed.shape
            LL, _, _, _ = haar_wavelet_decompose(y0_adain_embed.contiguous().view(B, C, h_len, w_len))

            B_LL, C_LL, H_LL, W_LL = LL.shape
            #flat = rearrange(LL, 'b c h w -> b (h w) c')
            flat = LL.contiguous().view(B_LL, H_LL * W_LL, C_LL)

            f_s = flat[0]  # assuming batch size 1 or using only the first
            self.mu_s = f_s.mean(dim=0, keepdim=True)
            f_s_centered = f_s - self.mu_s
            self.y0_color = self.whiten(f_s_centered, set=True)
            #self.y0_adain_embed = flat  # cache if needed
    
    def get(self, denoised_embed: torch.Tensor, h_len, w_len, stylize_highfreq=False):

        B, HW, C = denoised_embed.shape
        
        denoised_embed = denoised_embed.contiguous().view(B, C, h_len, w_len)
        
        for i in range(B):
            x = denoised_embed[i:i+1]  # [1, C, H, W]
            LL, LH, HL, HH = haar_wavelet_decompose(x)

            def process_band(band):
                Bc, Cc, Hc, Wc = band.shape
                flat = band.contiguous().view(Bc, Hc * Wc, Cc)
                
                styled = super(WaveletStyleWCT, self).get(flat)
                return styled.contiguous().view(Bc, Cc, Hc, Wc)

            LL_styled = process_band(LL)

            if stylize_highfreq:
                LH_styled = process_band(LH)
                HL_styled = process_band(HL)
                HH_styled = process_band(HH)
            else:
                LH_styled, HL_styled, HH_styled = LH, HL, HH

            recon = haar_wavelet_reconstruct(LL_styled, LH_styled, HL_styled, HH_styled)
            denoised_embed[i] = recon.squeeze(0)

        return denoised_embed.view(B, HW, C)



def haar_wavelet_decompose(x):
    """
    Orthonormal Haar decomposition.
    Input:  [B, C, H, W]
    Output: LL, LH, HL, HH with shape [B, C, H//2, W//2]
    """
    if x.dtype != torch.float32:
        x = x.float()
    
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Input must have even H, W"

    # Precompute
    norm = 1 / 2**0.5

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * norm * 0.5
    LH = (x00 - x01 + x10 - x11) * norm * 0.5
    HL = (x00 + x01 - x10 - x11) * norm * 0.5
    HH = (x00 - x01 - x10 + x11) * norm * 0.5

    return LL, LH, HL, HH

def haar_wavelet_reconstruct(LL, LH, HL, HH):
    """
    Orthonormal inverse Haar reconstruction.
    Input:  LL, LH, HL, HH [B, C, H, W]
    Output: Reconstructed [B, C, H*2, W*2]
    """
    norm = 1 / 2**0.5
    B, C, H, W = LL.shape

    x00 = (LL + LH + HL + HH) * norm
    x01 = (LL - LH + HL - HH) * norm
    x10 = (LL + LH - HL - HH) * norm
    x11 = (LL - LH - HL + HH) * norm

    out = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11

    return out








"""

class StyleFeatures:  
    def __init__(self, dtype=torch.float64,):
        self.dtype = dtype

    def set(self, y0_adain_embed: torch.Tensor):
            
    def get(self, denoised_embed: torch.Tensor):

        return "Norpity McNerp"

"""




class Retrojector:  
    def __init__(self, proj=None, patch_size=2, pinv_dtype=torch.float64, dtype=torch.float64, ENDO=False):
        self.proj       = proj
        self.patch_size = patch_size
        self.pinv_dtype = pinv_dtype
        self.dtype      = dtype
        
        self.LINEAR     = isinstance(proj, nn.Linear)
        self.CONV2D     = isinstance(proj, nn.Conv2d)
        self.CONV3D     = isinstance(proj, nn.Conv3d)
        self.ENDO       = ENDO
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
            if img.ndim == 4:
                img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size) 
            if self.ENDO:
                img_embed = F.linear(img.to(self.b) - self.b, self.W_inv)
            else:
                img_embed = F.linear(img.to(self.W), self.W, self.b)
        
        return img_embed.to(img)
    
    def unembed(self, img_embed: torch.Tensor):
        if   self.CONV2D:
            #img_embed = rearrange(img_embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=self.patch_size, pw=self.patch_size)
            img_embed = rearrange(img_embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=self.h, w=self.w, ph=1, pw=1)
            img = self.invert_conv2d(img_embed)
        
        elif self.LINEAR:
            if self.ENDO:
                img = F.linear(img_embed.to(self.W), self.W, self.b)
            else:
                img = F.linear(img_embed.to(self.b) - self.b, self.W_inv)
            if img.ndim == 3:
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



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, dim=1, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(dim, keepdim=True)
    std_c  = content.std (dim, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (dim, keepdim=True)
    std_s  = style.std   (dim, keepdim=True).add_(eps)

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
    #src_sorted, src_idx = denoised_embed.cpu().sort(dim=-2)
    src_idx    = denoised_embed.argsort(dim=-2)
    ref_sorted = y0_adain_embed.sort(dim=-2)[0]

    denoised_embed.scatter_(dim=-2, index=src_idx, src=ref_sorted.expand(ref_sorted.shape))

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





def apply_scattersort_spatial(
    x_spatial : torch.Tensor,
    y_spatial : torch.Tensor,
):
    x_emb = rearrange(x_spatial, "b c h w -> b (h w) c")
    y_emb = rearrange(y_spatial, "b c h w -> b (h w) c")
    
    x_sorted, x_idx = x_emb.sort(dim=-2)
    y_sorted, y_idx = y_emb.sort(dim=-2)

    x_emb = x_sorted.scatter(dim=-2, index=x_idx, src=y_sorted.expand(x_sorted.shape))
    
    return rearrange(x_emb, "b (h w) c -> b c h w", h=x_spatial.shape[-2], w=x_spatial.shape[-1])




def apply_adain_spatial(
    x_spatial : torch.Tensor,
    y_spatial : torch.Tensor,
):
    x_emb = rearrange(x_spatial, "b c h w -> b (h w) c")
    y_emb = rearrange(y_spatial, "b c h w -> b (h w) c")
    
    x_mean = x_emb.mean(-2, keepdim=True)
    x_std  = x_emb.std (-2, keepdim=True)
    y_mean = y_emb.mean(-2, keepdim=True)
    y_std  = y_emb.std (-2, keepdim=True)

    assert (x_std == 0).any() == 0, "Target tensor has no variance!"
    assert (y_std == 0).any() == 0, "Reference tensor has no variance!"
    
    x_emb_adain = (x_emb - x_mean) / x_std
    x_emb_adain = (x_emb_adain * y_std) + y_mean
    
    return x_emb_adain.reshape_as(x_spatial)





















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







def weighted_mix_n(tensor_list, weight_list, dim=-1, offset=0):
    assert all(t.shape == tensor_list[0].shape for t in tensor_list)
    assert len(tensor_list) == len(weight_list)

    total_weight = sum(weight_list)
    ratios = [w / total_weight for w in weight_list]

    length = tensor_list[0].shape[dim]
    idx = torch.arange(length)

    # Create a bin index tensor based on weighted slots
    float_bins = (idx + offset) * len(ratios) / length
    bin_idx = torch.floor(float_bins).long() % len(ratios)

    # Allocate slots based on ratio using a cyclic pattern
    counters = [0.0 for _ in ratios]
    slots = torch.empty_like(idx)

    for i in range(length):
        # Assign to the group that's most under-allocated
        expected = [r * (i + 1) for r in ratios]
        errors = [expected[j] - counters[j] for j in range(len(ratios))]
        k = max(range(len(errors)), key=lambda j: errors[j])
        slots[i] = k
        counters[k] += 1

    # Create mask for each tensor
    out = tensor_list[0].clone()
    for i, tensor in enumerate(tensor_list):
        mask = slots == i
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(tensor)
        out = torch.where(mask, tensor, out)
    
    return out






from torch import vmap

BLOCK_NAMES = {"double_blocks", "single_blocks", "up_blocks", "middle_blocks", "down_blocks", "input_blocks", "output_blocks"}

DEFAULT_BLOCK_WEIGHTS_MMDIT = {
    "attn_norm"    : 0.0,
    "attn_norm_mod": 0.0,
    "attn"         : 1.0,
    "attn_gated"   : 0.0,
    "attn_res"     : 1.0,
    "ff_norm"      : 0.0,
    "ff_norm_mod"  : 0.0,
    "ff"           : 1.0,
    "ff_gated"     : 0.0,
    "ff_res"       : 1.0,
    
    "h_tile"       : 8,
    "w_tile"       : 8,
}

DEFAULT_ATTN_WEIGHTS_MMDIT = {
    "q_proj": 0.0,
    "k_proj": 0.0,
    "v_proj": 1.0,
    "q_norm": 0.0,
    "k_norm": 0.0,
    "out"   : 1.0,
    
    "h_tile": 8,
    "w_tile": 8,
}

DEFAULT_BASE_WEIGHTS_MMDIT = {
    "proj_in" : 1.0,
    "proj_out": 1.0,
    
    "h_tile"  : 8,
    "w_tile"  : 8,
}

class Stylizer:
    buffer = {}
    
    CLS_WCT = StyleWCT()
    
    CLS_WCT2 = WaveletStyleWCT()
    
    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        self.dtype = dtype
        self.device = device
        self.mask  = [None]
        self.apply_to = [""]
        self.method = ["passthrough"]
        self.h_tile = [-1]
        self.w_tile = [-1]
        
        self.w_len   = 0
        self.h_len   = 0
        self.img_len = 0
        
        self.IMG_1ST = True
        self.HEADS = 0
        self.KONTEXT = 0
    def set_mode(self, mode):
        self.method = [mode] #[getattr(self, mode)]
    
    def set_weights(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, [v])
    
    def set_weights_recursive(self, **kwargs):
        for name, val in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, [val])

        for attr_name, attr_val in vars(self).items():
            if isinstance(attr_val, Stylizer):
                attr_val.set_weights_recursive(**kwargs)

        for list_name in BLOCK_NAMES:
            lst = getattr(self, list_name, None)
            if isinstance(lst, list):
                for element in lst:
                    if isinstance(element, Stylizer):
                        element.set_weights_recursive(**kwargs)
    
    def merge_weights(self, other):
        def recursive_merge(a, b, path):
            if isinstance(a, list) and isinstance(b, list):
                if path in BLOCK_NAMES:
                    out = []
                    for i in range(max(len(a), len(b))):
                        if i < len(a) and i < len(b):
                            out.append(recursive_merge(a[i], b[i], path=None))
                        elif i < len(a):
                            out.append(a[i])
                        else:
                            out.append(b[i])
                    return out
                return a + b

            if isinstance(a, dict) and isinstance(b, dict):
                merged = dict(a)
                for k, v_b in b.items():
                    if k in merged:
                        merged[k] = recursive_merge(merged[k], v_b, path=None)
                    else:
                        merged[k] = v_b
                return merged

            if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
                for attr, val_b in vars(b).items():
                    val_a = getattr(a, attr, None)
                    if val_a is not None:
                        setattr(a, attr, recursive_merge(val_a, val_b, path=attr))
                    else:
                        setattr(a, attr, val_b)
                return a
            return b

        for attr in vars(self):
            if attr in BLOCK_NAMES:
                merged = recursive_merge(getattr(self, attr), getattr(other, attr, []), path=attr)
            elif hasattr(other, attr):
                merged = recursive_merge(getattr(self, attr), getattr(other, attr), path=attr)
            else:
                continue
            setattr(self, attr, merged)
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.img_len = h_len * w_len
        self.HEADS = HEADS

    @staticmethod
    def middle_slice(length, weight):
        """
        Returns a slice object that selects the middle `weight` fraction of a dimension.
        Example: weight=1.0 → full slice; weight=0.5 → middle 50%
        """
        if weight >= 1.0:
            return slice(None)
        wr = int((length * (1 - weight)) // 2)
        return slice(wr, -wr if wr > 0 else None)

    @staticmethod
    def get_outer_slice(x, weight):
        if weight >= 0.0:
            return x
        length = x.shape[-2]
        wr = int((length * (1 - (-weight))) // 2)
        
        return torch.cat([x[...,:wr,:], x[...,-wr:,:]], dim=-2)

    @staticmethod
    def restore_outer_slice(x, x_outer, weight):
        if weight >= 0.0:
            return x
        length = x.shape[-2]
        wr = int((length * (1 - (-weight))) // 2)
        
        x[...,:wr,:]  = x_outer[...,:wr,:]
        x[...,-wr:,:] = x_outer[...,-wr:,:]
        return x

    def __call__(self, x, attr):
        if x.shape[0] == 1 and not self.KONTEXT:
            return x
        
        weight_list = getattr(self, attr)
        weights_all_zero = all(weight == 0.0 for weight in weight_list)
        if weights_all_zero:
            return x
        
        #self.HEADS=24
        #x_ndim = x.ndim
        #if x_ndim == 3:
        #    B, HW, C = x.shape
        #    if x.shape[-2] != self.HEADS and self.HEADS != 0:
        #        x = x.reshape(B,self.HEADS,HW,-1)
        
        HEAD_DIM = x.shape[1]
        if HEAD_DIM == self.HEADS:
            B, HEAD_DIM, HW, C = x.shape
            x = x.reshape(B, HW, C*HEAD_DIM)
            
        if hasattr(self, "KONTEXT") and self.KONTEXT == 1:
            x = x.reshape(2, x.shape[1] // 2, x.shape[2])
        
        txt_slice, img_slice, ktx_slice = self.txt_slice, self.img_slice, None
        if hasattr(self, "KONTEXT") and self.KONTEXT == 2:
            ktx_slice = self.img_slice # slice(2 * self.img_slice.start, None)
            img_slice = slice(2 * self.img_slice.start, self.img_slice.start)
            txt_slice = slice(None, 2 * self.txt_slice.stop)
        
        weights_all_one         = all(weight == 1.0           for weight in weight_list)
        methods_all_scattersort = all(name   == "scattersort" for name   in self.method)
        masks_all_none = all(mask is None for mask in self.mask)
        
        if weights_all_one and methods_all_scattersort and len(weight_list) > 1 and masks_all_none:
            buf = Stylizer.buffer
            buf['src_idx']   = x[0:1].argsort(dim=-2)
            buf['ref_sorted'], buf['ref_idx'] = x[1:].reshape(1, -1, x.shape[-1]).sort(dim=-2)
            buf['src'] = buf['ref_sorted'][:,::len(weight_list)].expand_as(buf['src_idx'])    #            interleave_stride = len(weight_list)
            
            x[0:1] = x[0:1].scatter_(dim=-2, index=buf['src_idx'], src=buf['src'],)
        
        else:
            for i, (weight, mask) in enumerate(zip(weight_list, self.mask)):
                if mask is not None:
                    x01 = x[0:1].clone()
                slc = Stylizer.middle_slice(x.shape[-2], weight)
                #slc = slice(None)
                    
                txt_method_name = self.method[i].removeprefix("tiled_")
                txt_method = getattr(self, txt_method_name)
                
                method_name = self.method[i].removeprefix("tiled_") if self.img_len > x.shape[-2] or self.h_len < 0 else self.method[i]
                method = getattr(self, method_name)
                apply_to = self.apply_to[i]
                if   weight == 0.0:
                    continue
                else: # if weight == 1.0:
                    if weight > 0 and weight < 1:
                        x_clone = x.clone()
                    if   self.img_len == x.shape[-2]  or apply_to == "img+txt" or self.h_len < 0:
                        x = method(x, idx=i+1, slc=slc)
                    elif   self.img_len < x.shape[-2]:
                        if "img" in apply_to:
                            x[...,img_slice,:] = method(x[...,img_slice,:], idx=i+1, slc=slc)
                            #if ktx_slice is not None:
                            #    x[...,ktx_slice,:] = method(x[...,ktx_slice,:], idx=i+1)
                            #x[:,:self.img_len,:] = method(x[:,:self.img_len,:], idx=i+1)
                        if "txt" in apply_to:
                            x[...,txt_slice,:] = txt_method(x[...,txt_slice,:], idx=i+1, slc=slc)
                            #x[:,self.img_len:,:] = method(x[:,self.img_len:,:], idx=i+1)
                        if not "img" in apply_to and not "txt" in apply_to:
                            pass
                    else:
                        x = method(x, idx=i+1, slc=slc)
                    if weight > 0 and weight < 1 and txt_method_name != "scattersort":
                        x = torch.lerp(x_clone, x, weight)
                #else:
                #    x = torch.lerp(x, method(x.clone(), idx=i+1), weight)
                
                if mask is not None:
                    x[0:1,...,img_slice,:] = torch.lerp(x01[...,img_slice,:], x[0:1,...,img_slice,:], mask.view(1, -1, 1))  
                    if ktx_slice is not None:
                        x[0:1,...,ktx_slice,:] = torch.lerp(x01[...,ktx_slice,:], x[0:1,...,ktx_slice,:], mask.view(1, -1, 1))  
                    #x[0:1,:self.img_len] = torch.lerp(x01[:,:self.img_len], x[0:1,:self.img_len], mask.view(1, -1, 1))
        
        #if x_ndim == 3:
        #    return x.view(B,HW,C)
        if hasattr(self, "KONTEXT") and  self.KONTEXT == 1:
            x = x.reshape(1, x.shape[1] * 2, x.shape[2])
        
        if HEAD_DIM == self.HEADS:
            return x.reshape(B, HEAD_DIM, HW, C)
        else:
            return x



    def WCT(self, x, idx=1):
        Stylizer.CLS_WCT.set(x[idx:idx+1])
        x[0:1] = Stylizer.CLS_WCT.get(x[0:1])
        return x
    
    def WCT2(self, x, idx=1):
        Stylizer.CLS_WCT2.set(x[idx:idx+1], self.h_len, self.w_len)
        x[0:1] = Stylizer.CLS_WCT2.get(x[0:1], self.h_len, self.w_len)
        return x

    @staticmethod
    def AdaIN_(x, y, eps: float = 1e-7) -> torch.Tensor:
        mean_c = x.mean(-2, keepdim=True)
        std_c  = x.std (-2, keepdim=True).add_(eps)  # in-place add
        mean_s = y.mean  (-2, keepdim=True)
        std_s  = y.std   (-2, keepdim=True).add_(eps)
        x.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
        return x

    def AdaIN(self, x, idx=1, eps: float = 1e-7) -> torch.Tensor:
        mean_c = x[0:1].mean(-2, keepdim=True)
        std_c  = x[0:1].std (-2, keepdim=True).add_(eps)  # in-place add
        mean_s = x[idx:idx+1].mean  (-2, keepdim=True)
        std_s  = x[idx:idx+1].std   (-2, keepdim=True).add_(eps)
        x[0:1].sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
        return x

    def injection(self, x:torch.Tensor, idx=1) -> torch.Tensor:
        x[0:1] = x[idx:idx+1]
        return x
    
    @staticmethod
    def injection_(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return y
    
    @staticmethod
    def passthrough(x:torch.Tensor, idx=1) -> torch.Tensor:
        return x
    
    @staticmethod
    def decompose_magnitude_direction(x, dim=-1, eps=1e-8):
        magnitude = x.norm(p=2, dim=dim, keepdim=True)
        direction = x / (magnitude + eps)
        return magnitude, direction

    @staticmethod
    def scattersort_dir_(x, y, dim=-2):
        #buf = Stylizer.buffer
        #buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        #buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        #mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'], dim)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'], dim)
        mag, _ = Stylizer.decompose_magnitude_direction(x.to(torch.float64), dim)
        
        buf = Stylizer.buffer
        buf['src_idx']                    = x.argsort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y   .sort(dim=-2)
        x.scatter_(dim=-2, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))
        
        
        _, dir = Stylizer.decompose_magnitude_direction(x.to(torch.float64), dim)
        
        return (mag * dir).to(x)


    @staticmethod
    def scattersort_dir2_(x, y, dim=-2):
        #buf = Stylizer.buffer
        #buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        #buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        #mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'], dim)
        #_, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'], dim)
        
        
        buf = Stylizer.buffer
        buf['src_sorted'], buf['src_idx'] = x.sort(dim=dim)
        buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=dim)
        



        buf['x_sub'], buf['x_sub_idx'] = buf['src_sorted'].sort(dim=-1)
        buf['y_sub'], buf['y_sub_idx'] = buf['ref_sorted'].sort(dim=-1)
        
        mag, _ = Stylizer.decompose_magnitude_direction(buf['x_sub'].to(torch.float64), -1)
        _, dir = Stylizer.decompose_magnitude_direction(buf['y_sub'].to(torch.float64), -1)
        
        buf['y_sub'] = (mag * dir).to(x)
        
        buf['ref_sorted'].scatter_(dim=-1, index=buf['y_sub_idx'], src=buf['y_sub'].expand_as(buf['y_sub_idx']))



        mag, _ = Stylizer.decompose_magnitude_direction(buf['src_sorted'].to(torch.float64), dim)
        _, dir = Stylizer.decompose_magnitude_direction(buf['ref_sorted'].to(torch.float64), dim)
        
        buf['ref_sorted'] = (mag * dir).to(x)
        
        x.scatter_(dim=dim, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))


        return x


    @staticmethod
    def scattersort_dir(x, idx=1):
        x[0:1] = Stylizer.scattersort_dir_(x[0:1], x[idx:idx+1])
        return x
    

    @staticmethod
    def scattersort_dir2(x, idx=1):
        x[0:1] = Stylizer.scattersort_dir2_(x[0:1], x[idx:idx+1])
        return x

    @staticmethod
    def scattersort_(x, y, slc=slice(None)):
        buf = Stylizer.buffer
        buf['src_idx']                    = x.argsort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y   .sort(dim=-2)

        return x.scatter_(dim=-2, index=buf['src_idx'][...,slc,:], src=buf['ref_sorted'][...,slc,:].expand_as(buf['src_idx'][...,slc,:]))
    

    @staticmethod
    def scattersort_double(x, y):
        buf = Stylizer.buffer
        buf['src_sorted'], buf['src_idx'] = x.sort(dim=-2)
        buf['ref_sorted'], buf['ref_idx'] = y.sort(dim=-2)
        
        buf['x_sub_idx']               = buf['src_sorted'].argsort(dim=-1)
        buf['y_sub'], buf['y_sub_idx'] = buf['ref_sorted'].sort(dim=-1)
        
        x.scatter_(dim=-1, index=buf['x_sub_idx'], src=buf['y_sub'].expand_as(buf['x_sub_idx']))

        return x.scatter_(dim=-2, index=buf['src_idx'], src=buf['ref_sorted'].expand_as(buf['src_idx']))
    
    
    def scattersort_aoeu(self, x, idx=1, slc=slice(None)):
        x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
        return x
    
    def scattersort(self, x, idx=1, slc=slice(None)):
        if x.shape[0] != 2:
            x[0:1] = Stylizer.scattersort_(x[0:1], x[idx:idx+1], slc)
            return x
        
        buf = Stylizer.buffer
        buf['sorted'], buf['idx'] = x.sort(dim=-2)

        return x.scatter_(dim=-2, index=buf['idx'][0:1][...,slc,:], src=buf['sorted'][1:2][...,slc,:].expand_as(buf['idx'][0:1][...,slc,:]))
    

    
    
    def tiled_scattersort(self, x, idx=1): #, h_tile=None, w_tile=None):
        #if HDModel.RECON_MODE:
        #    return denoised_embed
        #den   = x[0:1]      [:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        #style = x[idx:idx+1][:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        #h_tile = self.h_tile[idx-1] if h_tile is None else h_tile
        #w_tile = self.w_tile[idx-1] if w_tile is None else w_tile
        
        C = x.shape[-1]
        den   = x[0:1]      [:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        style = x[idx:idx+1][:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        
        tiles     = Stylizer.get_tiles_as_strided(den,   self.h_tile[idx-1], self.w_tile[idx-1])
        ref_tile  = Stylizer.get_tiles_as_strided(style, self.h_tile[idx-1], self.w_tile[idx-1])

        # rearrange for vmap to run on (nH, nW) ( as outer axes)
        tiles_v    = tiles   .permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)
        ref_tile_v = ref_tile.permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)

        # vmap over spatial dimms (nH, nW)... num of tiles high, num tiles wide
        vmap2   = torch.vmap(torch.vmap(Stylizer.apply_scattersort_per_tile, in_dims=0), in_dims=0)
        result  = vmap2(tiles_v, ref_tile_v)  # (nH, nW, B, C, tile_h, tile_w)

        # --> (B, C, nH, nW, tile_h, tile_w)
        result = result.permute(2, 3, 0, 1, 4, 5)  #( B, C, nH, nW, tile_h, tile_w)

        # in-place copy, werx if result has same shape/strides as tiles... overwrites same mem location "content" is using
        tiles.copy_(result)

        return x
    
    
    def tiled_AdaIN(self, x, idx=1):
        #if HDModel.RECON_MODE:
        #    return denoised_embed
        #den   = x[0:1]      [:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        #style = x[idx:idx+1][:,:self.img_len,:].view(-1, 2560, self.h_len, self.w_len)
        C = x.shape[-1]
        den   = x[0:1]      [:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        style = x[idx:idx+1][:,self.img_slice,:].reshape(-1, C, self.h_len, self.w_len)
        
        tiles     = Stylizer.get_tiles_as_strided(den,   self.h_tile[idx-1], self.w_tile[idx-1])
        ref_tile  = Stylizer.get_tiles_as_strided(style, self.h_tile[idx-1], self.w_tile[idx-1])

        # rearrange for vmap to run on (nH, nW) ( as outer axes)
        tiles_v    = tiles   .permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)
        ref_tile_v = ref_tile.permute(2, 3, 0, 1, 4, 5) # (nH, nW, B, C, tile_h, tile_w)

        # vmap over spatial dimms (nH, nW)... num of tiles high, num tiles wide
        vmap2   = torch.vmap(torch.vmap(Stylizer.apply_AdaIN_per_tile, in_dims=0), in_dims=0)
        result  = vmap2(tiles_v, ref_tile_v)  # (nH, nW, B, C, tile_h, tile_w)

        # --> (B, C, nH, nW, tile_h, tile_w)
        result = result.permute(2, 3, 0, 1, 4, 5)  #( B, C, nH, nW, tile_h, tile_w)

        # in-place copy, werx if result has same shape/strides as tiles... overwrites same mem location "content" is using
        tiles.copy_(result)

        return x
    
    
    @staticmethod
    def get_tiles_as_strided(x, tile_h, tile_w):
        B, C, H, W = x.shape
        stride = x.stride()
        nH = H // tile_h
        nW = W // tile_w

        tiles = x.as_strided(
            size=(B, C, nH, nW, tile_h, tile_w),
            stride=(stride[0], stride[1], stride[2] * tile_h, stride[3] * tile_w, stride[2], stride[3])
        )
        return tiles  # shape: (B, C, nH, nW, tile_h, tile_w)

    @staticmethod
    def apply_scattersort_per_tile(tile, ref_tile):
        flat     = tile    .flatten(-2, -1)
        ref_flat = ref_tile.flatten(-2, -1)

        sorted_ref, _ = ref_flat  .sort(dim=-1)
        src_sorted, src_idx = flat.sort(dim=-1)
        
        out = flat.scatter(dim=-1, index=src_idx, src=sorted_ref)
        return out.view_as(tile)

    @staticmethod
    def apply_AdaIN_per_tile(tile, ref_tile, eps: float = 1e-7):
        mean_c = tile.mean(-2, keepdim=True)
        std_c  = tile.std (-2, keepdim=True).add_(eps)  # in-place add
        mean_s = ref_tile.mean  (-2, keepdim=True)
        std_s  = ref_tile.std   (-2, keepdim=True).add_(eps)
        tile.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
        return tile

class StyleMMDiT_Attn(Stylizer):
    def __init__(self, mode):
        super().__init__()
        
        self.q_proj = [0.0]
        self.k_proj = [0.0]
        self.v_proj = [0.0]

        self.q_norm = [0.0]
        self.k_norm = [0.0]
        
        self.out    = [0.0]

class StyleMMDiT_FF(Stylizer): # these hit img or joint only, never txt
    def __init__(self, mode):
        super().__init__()
    
        self.ff_1      = [0.0]
        self.ff_1_silu = [0.0]
        self.ff_3      = [0.0]
        self.ff_13     = [0.0]
        self.ff_2      = [0.0]
        
class StyleMMDiT_MoE(Stylizer): # these hit img or joint only, never txt
    def __init__(self, mode):
        super().__init__()
        
        self.FF_SHARED   = StyleMMDiT_FF(mode)
        self.FF_SEPARATE = StyleMMDiT_FF(mode)
        
        self.shared      = [0.0]
        self.gate        = [False]
        self.topk_weight = [0.0]

        self.separate    = [0.0]
        self.sum         = [0.0]
        self.out         = [0.0]





class StyleMMDiT_SubBlock(Stylizer):
    def __init__(self, mode):
        super().__init__()
        
        self.ATTN = StyleMMDiT_Attn(mode)  # options for attn itself: qkv proj, qk norm, attn out

        self.attn_norm     = [0.0]
        self.attn_norm_mod = [0.0]
        self.attn          = [0.0]
        self.attn_gated    = [0.0]
        self.attn_res      = [0.0]
        
        self.ff_norm       = [0.0]
        self.ff_norm_mod   = [0.0]
        self.ff            = [0.0]
        self.ff_gated      = [0.0]
        self.ff_res        = [0.0]
        
        self.mask = [None]
        
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.ATTN.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

class StyleMMDiT_IMG_Block(StyleMMDiT_SubBlock):  # img or joint
    def __init__(self, mode):
        super().__init__(mode)
        self.FF = StyleMMDiT_MoE(mode)  # options for MoE if img or joint
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.FF.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        
class StyleMMDiT_TXT_Block(StyleMMDiT_SubBlock):   # txt only
    def __init__(self, mode):
        super().__init__(mode)
        self.FF  = StyleMMDiT_FF(mode)   # options for FF within MoE for img or joint; or for txt alone
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.FF.set_len(h_len, w_len, img_slice, txt_slice, HEADS)





class StyleMMDiT_BaseBlock:
    def __init__(self, mode="passthrough"):

        self.img = StyleMMDiT_IMG_Block(mode)
        self.txt = StyleMMDiT_TXT_Block(mode)
        
        self.mask      = [None]
        self.attn_mask = [None]
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_len = h_len * w_len
        
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.HEADS = HEADS
        
        self.img.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.txt.set_len(-1, -1, img_slice, txt_slice, HEADS)
        
        for i, mask in enumerate(self.mask):
            if mask is not None and mask.ndim > 1:
                self.mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.img.mask = self.mask
        for i, mask in enumerate(self.attn_mask):
            if mask is not None and mask.ndim > 1:
                self.attn_mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.img.ATTN.mask = self.attn_mask      

class StyleMMDiT_DoubleBlock(StyleMMDiT_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)
        self.txt = StyleMMDiT_TXT_Block(mode)
    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.txt.set_len(-1, -1, img_slice, txt_slice, HEADS)

class StyleMMDiT_SingleBlock(StyleMMDiT_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)




























class StyleUNet_Resample(Stylizer):
    def __init__(self, mode):
        super().__init__()
        self.conv = [0.0]

class StyleUNet_Attn(Stylizer):
    def __init__(self, mode):
        super().__init__()
        self.q_proj = [0.0]
        self.k_proj = [0.0]
        self.v_proj = [0.0]
        self.out    = [0.0]

class StyleUNet_FF(Stylizer):
    def __init__(self, mode):
        super().__init__()
        self.proj   = [0.0]
        self.geglu  = [0.0]
        self.linear = [0.0]
        
class StyleUNet_TransformerBlock(Stylizer): 
    def __init__(self, mode):
        super().__init__()
        
        self.ATTN1 = StyleUNet_Attn(mode)  # self-attn
        self.FF    = StyleUNet_FF  (mode)  
        self.ATTN2 = StyleUNet_Attn(mode)  # cross-attn

        self.self_attn  = [0.0]
        self.ff         = [0.0]
        self.cross_attn = [0.0]
        
        self.self_attn_res  = [0.0]
        self.cross_attn_res = [0.0]
        self.ff_res = [0.0]
        
        self.norm1 = [0.0]
        self.norm2 = [0.0]
        self.norm3 = [0.0]
        
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.ATTN1.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.ATTN2.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

class StyleUNet_SpatialTransformer(Stylizer): 
    def __init__(self, mode):
        super().__init__()
        
        self.TFMR = StyleUNet_TransformerBlock(mode)

        self.spatial_norm_in     = [0.0]
        self.spatial_proj_in     = [0.0]
        self.spatial_transformer_block = [0.0]
        self.spatial_transformer = [0.0]
        self.spatial_proj_out    = [0.0]
        self.spatial_res         = [0.0]
        
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.TFMR.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

class StyleUNet_ResBlock(Stylizer):
    def __init__(self, mode):
        super().__init__()

        self.in_norm    = [0.0]
        self.in_silu    = [0.0]
        self.in_conv    = [0.0]

        self.emb_silu   = [0.0]
        self.emb_linear = [0.0]
        self.emb_res    = [0.0]

        self.out_norm   = [0.0]
        self.out_silu   = [0.0]
        self.out_conv   = [0.0]
        
        self.residual   = [0.0]


class StyleUNet_BaseBlock(Stylizer):
    def __init__(self, mode="passthrough"):

        self.resample_block = StyleUNet_Resample(mode)
        self.res_block      = StyleUNet_ResBlock(mode)
        self.spatial_block  = StyleUNet_SpatialTransformer(mode)
        
        self.resample = [0.0]
        self.res      = [0.0]
        self.spatial  = [0.0]
        
        self.mask      = [None]
        self.attn_mask = [None]
        
        self.KONTEXT = 0

    
    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_len = h_len * w_len
        
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.HEADS = HEADS
        
        self.resample_block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.res_block     .set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        self.spatial_block .set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        
        for i, mask in enumerate(self.mask):
            if mask is not None and mask.ndim > 1:
                self.mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.resample_block.mask = self.mask
            self.res_block.mask      = self.mask
            self.spatial_block.mask  = self.mask
            self.spatial_block.TFMR.mask  = self.mask
            
        for i, mask in enumerate(self.attn_mask):
            if mask is not None and mask.ndim > 1:
                self.attn_mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()
            self.spatial_block.TFMR.ATTN1.mask = self.attn_mask     
            
    def __call__(self, x, attr):
        B, C, H, W = x.shape
        x = super().__call__(x.reshape(B, H*W, C), attr)
        return x.reshape(B,C,H,W)
        

class StyleUNet_InputBlock(StyleUNet_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)    

class StyleUNet_MiddleBlock(StyleUNet_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)

class StyleUNet_OutputBlock(StyleUNet_BaseBlock):
    def __init__(self, mode="passthrough"):
        super().__init__(mode)
















class Style_Model(Stylizer):

    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        super().__init__(dtype, device)
        self.guides = []
        self.GUIDES_INITIALIZED = False
        
        #self.double_blocks = [StyleMMDiT_DoubleBlock() for _ in range(100)]
        #self.single_blocks = [StyleMMDiT_SingleBlock() for _ in range(100)]
        
        self.h_len   = -1
        self.w_len   = -1
        self.img_len = -1
        self.h_tile  = [-1]
        self.w_tile  = [-1]
        
        self.proj_in  = [0.0]  # these are for img only! not sliced
        self.proj_out = [0.0]
        
        self.cond_pos = [None]
        self.cond_neg = [None]
        
        self.noise_mode = "update"
        self.recon_lure = "none"
        self.data_shock = "none"
        
        self.data_shock_start_step = 0
        self.data_shock_end_step   = 0
        
        self.Retrojector = None
        self.Endojector  = None
        
        self.IMG_1ST = True
        self.HEADS = 0
        self.KONTEXT = 0
    def __call__(self, x, attr):
        if x.shape[0] == 1 and not self.KONTEXT:
            return x
        
        weight_list = getattr(self, attr)
        weights_all_zero = all(weight == 0.0 for weight in weight_list)
        if weights_all_zero:
            return x
        
        """x_ndim = x.ndim
        if x_ndim == 4:
            B, HEAD, HW, C = x.shape
            
        if x_ndim == 3:
            B, HW, C = x.shape
            if x.shape[-2] != self.HEADS and self.HEADS != 0:
                x = x.reshape(B,self.HEADS,HW,-1)"""
        
        HEAD_DIM = x.shape[1]
        if HEAD_DIM == self.HEADS:
            B, HEAD_DIM, HW, C = x.shape
            x = x.reshape(B, HW, C*HEAD_DIM)
            
        if self.KONTEXT == 1:
            x = x.reshape(2, x.shape[1] // 2, x.shape[2])
            
        weights_all_one         = all(weight == 1.0           for weight in weight_list)
        methods_all_scattersort = all(name   == "scattersort" for name   in self.method)
        masks_all_none = all(mask is None for mask in self.mask)
        
        if weights_all_one and methods_all_scattersort and len(weight_list) > 1 and masks_all_none:
            buf = Stylizer.buffer
            buf['src_idx']   = x[0:1].argsort(dim=-2)
            buf['ref_sorted'], buf['ref_idx'] = x[1:].reshape(1, -1, x.shape[-1]).sort(dim=-2)
            buf['src'] = buf['ref_sorted'][:,::len(weight_list)].expand_as(buf['src_idx'])    #            interleave_stride = len(weight_list)
            
            x[0:1] = x[0:1].scatter_(dim=-2, index=buf['src_idx'], src=buf['src'],)
        else:
            for i, (weight, mask) in enumerate(zip(weight_list, self.mask)):
                if weight > 0 and weight < 1:
                    x_clone = x.clone()
                if mask is not None:
                    x01 = x[0:1].clone()
                slc = Stylizer.middle_slice(x.shape[-2], weight)
                
                method = getattr(self, self.method[i])
                if   weight == 0.0:
                    continue
                elif weight == 1.0:
                    x = method(x, idx=i+1)
                else:
                    x = method(x, idx=i+1, slc=slc)
                if weight > 0 and weight < 1 and self.method[i] != "scattersort":
                    x = torch.lerp(x_clone, x, weight)
                    
                #else:
                #    x = torch.lerp(x, method(x.clone(), idx=i), weight)
                
                if mask is not None:
                    x[0:1] = torch.lerp(x01, x[0:1], mask.view(1, -1, 1))
        
        #if x_ndim == 3:
        #    return x.view(B,HW,C)
        if self.KONTEXT == 1:
            x = x.reshape(1, x.shape[1] * 2, x.shape[2])
            
        if HEAD_DIM == self.HEADS:
            return x.reshape(B, HEAD_DIM, HW, C)
        else:
            return x

    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        self.h_len  = h_len
        self.w_len  = w_len
        self.img_len = h_len * w_len
        
        self.img_slice = img_slice
        self.txt_slice = txt_slice
        self.HEADS = HEADS
        
        #for block in self.double_blocks:
        #    block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        #for block in self.single_blocks:
        #    block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        
        for i, mask in enumerate(self.mask):
            if mask is not None and mask.ndim > 1:
                self.mask[i] = F.interpolate(mask.unsqueeze(0), size=(h_len, w_len)).flatten().to(torch.bfloat16).cuda()

    def init_guides(self, model):
        if not self.GUIDES_INITIALIZED:
            if self.guides == []:
                self.guides = None
            elif self.guides is not None:
                for i, latent in enumerate(self.guides):
                    if type(latent) is dict:
                        latent = model.inner_model.inner_model.process_latent_in(latent['samples']).to(dtype=self.dtype, device=self.device)
                    elif type(latent) is torch.Tensor:
                        latent = latent.to(dtype=self.dtype, device=self.device)
                    else:
                        latent = None
                        #raise ValueError(f"Invalid latent type: {type(latent)}")

                    #if self.VIDEO and latent.shape[2] == 1:
                    #    latent = latent.repeat(1, 1, x.shape[2], 1, 1)

                    self.guides[i] = latent
                if any(g is None for g in self.guides):
                    self.guides = None
                    print("Style guide nonetype set for Kontext.")
                else:
                    self.guides = torch.cat(self.guides, dim=0)
            self.GUIDES_INITIALIZED = True
    
    def set_conditioning(self, positive, negative):
        self.cond_pos = [positive]
        self.cond_neg = [negative] 

    def apply_style_conditioning(self, UNCOND, base_context, base_y=None, base_llama3=None):

        def get_max_token_lengths(style_conditioning, base_context, base_y=None, base_llama3=None):
            context_max_len = base_context.shape[-2]
            llama3_max_len  = base_llama3.shape[-2]  if base_llama3 is not None else -1
            y_max_len       = base_y.shape[-1]       if base_y      is not None else -1

            for style_cond in style_conditioning:
                if style_cond is None:
                    continue
                context_max_len = max(context_max_len, style_cond[0][0].shape[-2])
                if base_llama3 is not None:
                    llama3_max_len  = max(llama3_max_len,  style_cond[0][1]['conditioning_llama3'].shape[-2])
                if base_y is not None:
                    y_max_len       = max(y_max_len,       style_cond[0][1]['pooled_output'].shape[-1])

            return context_max_len, llama3_max_len, y_max_len

        def pad_to_len(x, target_len, pad_value=0.0, dim=1):
            if target_len < 0:
                return x
            cur_len = x.shape[dim]
            if cur_len == target_len:
                return x
            return F.pad(x, (0, 0, 0, target_len - cur_len), value=pad_value)

        style_conditioning = self.cond_pos if not UNCOND else self.cond_neg
        
        context_max_len, llama3_max_len, y_max_len = get_max_token_lengths(
            style_conditioning = style_conditioning,
            base_context       = base_context,
            base_y             = base_y,
            base_llama3        = base_llama3,
        )
        
        bsz_style = len(style_conditioning)
        
        context = base_context.repeat(bsz_style + 1, 1, 1)
        y = base_y.repeat(bsz_style + 1, 1)                   if base_y      is not None else None
        llama3  =  base_llama3.repeat(bsz_style + 1, 1, 1, 1) if base_llama3 is not None else None

        context = pad_to_len(context, context_max_len, dim=-2)
        llama3  = pad_to_len(llama3, llama3_max_len, dim=-2)   if base_llama3 is not None else None
        y       = pad_to_len(y,      y_max_len, dim=-1)        if base_y      is not None else None
        
        for ci, style_cond in enumerate(style_conditioning):
            if style_cond is None:
                continue
            context[ci+1:ci+2] = pad_to_len(style_cond[0][0], context_max_len, dim=-2).to(context)
            if llama3 is not None:
                llama3 [ci+1:ci+2] = pad_to_len(style_cond[0][1]['conditioning_llama3'], llama3_max_len, dim=-2).to(llama3)
            if y is not None:
                y      [ci+1:ci+2] = pad_to_len(style_cond[0][1]['pooled_output'],       y_max_len, dim=-1).to(y)
        
        return context, y, llama3
    
    def WCT_data(self, denoised_embed, y0_style_embed):
        Stylizer.CLS_WCT.set(y0_style_embed.to(denoised_embed))
        return Stylizer.CLS_WCT.get(denoised_embed)

    def WCT2_data(self, denoised_embed, y0_style_embed):
        Stylizer.CLS_WCT2.set(y0_style_embed.to(denoised_embed))
        return Stylizer.CLS_WCT2.get(denoised_embed)

    def apply_to_data(self, denoised, y0_style=None, mode="none"):
        if mode == "none":
            return denoised
        y0_style = self.guides if y0_style is None else y0_style
        
        y0_style_embed = self.Retrojector.embed(y0_style)
        denoised_embed = self.Retrojector.embed(denoised)
        B,HW,C = y0_style_embed.shape
        embed  = torch.cat([denoised_embed, y0_style_embed.view(1,B*HW,C)[:,::B,:]], dim=0)
        method = getattr(self, mode)
        if mode == "scattersort":
            slc = Stylizer.middle_slice(embed.shape[-2], self.data_shock_weight)
            embed = method(embed, slc=slc)
        else:
            embed  = method(embed)
        return self.Retrojector.unembed(embed[0:1])

    def apply_recon_lure(self, denoised, y0_style):
        if self.recon_lure == "none":
            return denoised
        for i in range(denoised.shape[0]):
            denoised[i:i+1] = self.apply_to_data(denoised[i:i+1], y0_style, self.recon_lure)
        return denoised

    def apply_data_shock(self, denoised):
        if self.data_shock == "none":
            return denoised
        datashock_ref = getattr(self, "datashock_ref", None)
        if self.data_shock == "scattersort":
            return self.apply_to_data(denoised, datashock_ref, self.data_shock)
        else:
            return torch.lerp(denoised, self.apply_to_data(denoised, datashock_ref, self.data_shock), torch.Tensor([self.data_shock_weight]).double().cuda())




class StyleMMDiT_Model(Style_Model):

    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        super().__init__(dtype, device)
        self.double_blocks = [StyleMMDiT_DoubleBlock() for _ in range(100)]
        self.single_blocks = [StyleMMDiT_SingleBlock() for _ in range(100)]

    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.double_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.single_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)


class StyleUNet_Model(Style_Model):

    def __init__(self, dtype=torch.float64, device=torch.device("cuda")):
        super().__init__(dtype, device)
        self.input_blocks  = [StyleUNet_InputBlock()  for _ in range(100)]
        self.middle_blocks = [StyleUNet_MiddleBlock() for _ in range(100)]
        self.output_blocks = [StyleUNet_OutputBlock() for _ in range(100)]

    def set_len(self, h_len, w_len, img_slice, txt_slice, HEADS):
        super().set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.input_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.middle_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)
        for block in self.output_blocks:
            block.set_len(h_len, w_len, img_slice, txt_slice, HEADS)

    def __call__(self, x, attr):
        B, C, H, W = x.shape
        x = super().__call__(x.reshape(B, H*W, C), attr)
        return x.reshape(B,C,H,W)
        
