import torch
import comfy.ops
import torch.nn
import torch.nn.functional as F

ops = comfy.ops.manual_cast

class ReReduxImageEncoder(torch.nn.Module):
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.redux_dim = redux_dim
        self.device = device
        self.dtype = dtype
        
        self.style_dtype = None

        self.redux_up = ops.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
        self.redux_down = ops.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

    def forward(self, sigclip_embeds) -> torch.Tensor:
        projected_x = self.redux_down(torch.nn.functional.silu(self.redux_up(sigclip_embeds)))
        return projected_x
    
    def feature_match(self, cond, clip_vision_output, mode="WCT"):
        sigclip_embeds = clip_vision_output.last_hidden_state
        dense_embed = torch.nn.functional.silu(self.redux_up(sigclip_embeds))
        t_sqrt = int(dense_embed.shape[-2] ** 0.5)
        dense_embed_sq = dense_embed.view(dense_embed.shape[-3], t_sqrt, t_sqrt, dense_embed.shape[-1])
        
        t_cond_sqrt = int(cond[0][0].shape[-2] ** 0.5) 
        dense_embed256 = F.interpolate(dense_embed_sq.transpose(-3,-1), size=(t_cond_sqrt,t_cond_sqrt), mode="bicubic")
        dense_embed256 = dense_embed256.flatten(-2,-1).transpose(-2,-1)
        
        dtype = self.style_dtype if hasattr(self, "style_dtype") and self.style_dtype is not None else dense_embed.dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        
        W = self.redux_down.weight.data.to(dtype)   # shape [2560, 64]
        b = self.redux_down.bias.data.to(dtype)     # shape [2560]
        
        cond_256 = cond[0][0].clone()
        
        if not hasattr(self, "W_pinv"):
            self.W_pinv = torch.linalg.pinv(W.to(pinv_dtype).cuda()).to(W)
        
        #cond_256_embed = (cond_256 - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
        cond_embed256 = (cond_256 - b.to(cond_256)) @ self.W_pinv.T.to(cond_256)
        
        
        
        
        
        if mode == "AdaIN":
            cond_embed256 = adain_seq_inplace(cond_embed256, dense_embed256)
            #for adain_iter in range(EO("style_iter", 0)):
            #    cond_embed256 = adain_seq_inplace(cond_embed256, dense_embed256)
            #    cond_embed256 = (cond_embed256 - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
            #    cond_embed256 = F.linear(cond_embed256         .to(W), W, b).to(img)
            #    cond_embed256 = adain_seq_inplace(cond_embed256, dense_embed256)

        elif mode == "WCT":
            if not hasattr(self, "dense_embed256") or self.dense_embed256 is None or self.dense_embed256.shape != dense_embed256.shape or torch.norm(self.dense_embed256 - dense_embed256) > 0:
                self.dense_embed256 = dense_embed256
                
                f_s          = dense_embed256[0].clone()
                self.mu_s    = f_s.mean(dim=0, keepdim=True)
                f_s_centered = f_s - self.mu_s
                
                cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                S_eig, U_eig = torch.linalg.eigh((cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)).cuda())
                S_eig = S_eig.to(cov)
                U_eig = U_eig.to(cov)
                
                S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                
                whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                self.y0_color  = whiten.to(f_s_centered)

            for wct_i in range(cond_embed256.shape[-3]):
                f_c          = cond_embed256[wct_i].clone()
                mu_c         = f_c.mean(dim=0, keepdim=True)
                f_c_centered = f_c - mu_c
                
                cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                S_eig, U_eig  = torch.linalg.eigh((cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device)).cuda())
                S_eig = S_eig.to(cov)
                U_eig = U_eig.to(cov)
                
                inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                
                whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                whiten = whiten.to(f_c_centered)

                f_c_whitened = f_c_centered @ whiten.T
                f_cs         = f_c_whitened @ self.y0_color.T + self.mu_s
                
                cond_embed256[wct_i] = f_cs
    
        cond[0][0] = self.redux_down(cond_embed256)
        return (cond,)
        
        
        

def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)
    return content


def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)

