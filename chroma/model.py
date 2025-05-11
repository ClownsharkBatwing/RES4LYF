#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..helper import ExtraOptions

from comfy.ldm.flux.layers import (
    EmbedND,
    timestep_embedding,
)

from .layers import (
    ReChromaDoubleStreamBlock,
    LastLayer,
    ReChromaSingleStreamBlock,
    Approximator,
    ChromaModulationOut,
)


@dataclass
class ChromaParams:
    in_channels        : int
    out_channels       : int
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
    in_dim             : int
    out_dim            : int
    hidden_dim         : int
    n_layers           : int




class ReChroma(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype        = dtype
        params            = ChromaParams(**kwargs)
        self.params       = params
        self.patch_size   = params.patch_size
        self.in_channels  = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads   = params.num_heads
        self.in_dim      = params.in_dim
        self.out_dim     = params.out_dim
        self.hidden_dim  = params.hidden_dim
        self.n_layers    = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in      = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.txt_in      = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)
        # set as nn identity for now, will overwrite it later.
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                    dtype=dtype, device=device, operations=operations
                )


        self.double_blocks = nn.ModuleList(
            [
                ReChromaDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                ReChromaSingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

        self.skip_mmdit = []
        self.skip_dit = []
        self.lite = False

    def get_modulations(self, tensor: torch.Tensor, block_type: str, *, idx: int = 0):
        # This function slices up the modulations tensor which has the following layout:
        #   single     : num_single_blocks * 3 elements
        #   double_img : num_double_blocks * 6 elements
        #   double_txt : num_double_blocks * 6 elements
        #   final      : 2 elements
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        # Double block modulations are 6 elements so we double 3 * idx.
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            # Advance past the single block modulations.
            offset += 3 * single_block_count
            if block_type == "double_txt":
                # Advance past the double block img modulations.
                offset += 6 * double_block_count
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("Bad block_type")


    def forward_blocks(
        self,
        img       : Tensor,
        img_ids   : Tensor,
        txt       : Tensor,
        txt_ids   : Tensor,
        timesteps : Tensor,
        guidance  : Tensor  = None,
        control             = None,
        update_cross_attn   = None,
        transformer_options ={},
        attn_mask : Tensor  = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        # guidance = guidance *
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length), 32).to(img.device, img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                       txt=args["txt"],
                                                       vec=args["vec"],
                                                       pe=args["pe"],
                                                       attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img,
                                                               "txt": txt,
                                                               "vec": double_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                     txt=txt,
                                     vec=double_mod,
                                     pe=pe,
                                     attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                               "vec": single_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_chroma_depr(self, x, timestep, context, guidance, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]

    
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
                #y,
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

        y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight", 0.0)
        y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight", 0.0)
        y0_style_pos_synweight *= y0_style_pos_weight

        y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight", 0.0)
        y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight", 0.0)
        y0_style_neg_synweight *= y0_style_neg_weight
                
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
                                        #y           [i][None,...].clone(),
                                        guidance    [i][None,...].clone(),
                                        control, 
                                        update_cross_attn=update_cross_attn,
                                        transformer_options=transformer_options)  # context 1,256,4096   y 1,768
            out_list.append(out_tmp)
            
        out = torch.stack(out_list, dim=0).squeeze(dim=1)
        
        eps = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1): #: and not UNCOND):
            if y0_style_pos is not None and y0_style_pos_weight != 0.0:
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
                
                if not UNCOND:
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                        eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                    else:
                        eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
                elif eps.shape[0] == 1 and UNCOND:
                    eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
                    #if eps.shape[0] == 2:
                    #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                
                eps = eps.float()
        
        if eps.shape[0] == 2 or (eps.shape[0] == 1): # and UNCOND):
            if y0_style_neg is not None and y0_style_neg_weight != 0.0:
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
                
                if UNCOND:
                    eps = (x - denoised_approx) / sigma
                    eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                    if eps.shape[0] == 2:
                        eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
                elif eps.shape[0] == 1 and not UNCOND:
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                
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


