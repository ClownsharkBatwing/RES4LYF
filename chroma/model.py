#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit

from ..helper import ExtraOptions

from ..latents import tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d
from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch

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
        UNCOND : bool = False,
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
        
        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        mask_zero = None
        mask = None
        
        text_len = txt.shape[1] # mask_obj[0].text_len
        
        if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
            AttnMask = transformer_options['AttnMask']
            mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
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
            if mask_zero is None:
                mask_zero = torch.ones_like(mask)
                img_len = transformer_options['AttnMask'].img_len
                #mask_zero[:text_len, :text_len] = mask[:text_len, :text_len]
                mask_zero[:text_len, :] = mask[:text_len, :]
                mask_zero[:, :text_len] = mask[:, :text_len]
            if weight == 0:
                mask = None

        if mask is not None and not type(mask[0][0].item()) == bool:
            mask = mask.to(img.dtype)
        if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
            mask_zero = mask_zero.to(img.dtype)

        total_layers = len(self.double_blocks) + len(self.single_blocks)
        
        attn_mask = mask if attn_mask is None else attn_mask
        
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
                        out["img"], out["txt"] = block( img       = args["img"],
                                                        txt       = args["txt"],
                                                        vec       = args["vec"],
                                                        pe        = args["pe"],
                                                        attn_mask = args.get("attn_mask"))
                        return out

                    out = blocks_replace[("double_block", i)]({ "img"             : img,
                                                                "txt"             : txt,
                                                                "vec"             : double_mod,
                                                                "pe"              : pe,
                                                                "attn_mask"       : attn_mask},
                                                                {"original_block" : block_wrap})
                    txt = out["txt"] 
                    img = out["img"]
                else:

                    if   weight > 0 and mask is not None and     weight  <=      i/total_layers:
                        img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=mask_zero)
                        
                    elif (weight < 0 and mask is not None and abs(weight) <= (1 - i/total_layers)):
                        img_tmpZ, txt_tmpZ = img.clone(), txt.clone()

                        img_tmpZ, txt = block(img=img_tmpZ, txt=txt_tmpZ, vec=double_mod, pe=pe, attn_mask=mask)
                        img, txt_tmpZ = block(img=img     , txt=txt     , vec=double_mod, pe=pe, attn_mask=mask_zero)
                        
                    elif floor > 0 and mask is not None and     floor  >=      i/total_layers:
                        mask_tmp = mask.clone()
                        mask_tmp[text_len:, text_len:] = 1.0
                        img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=mask_tmp)
                        
                    elif floor < 0 and mask is not None and abs(floor) >= (1 - i/total_layers):
                        mask_tmp = mask.clone()
                        mask_tmp[text_len:, text_len:] = 1.0
                        img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=mask_tmp)
                        
                    elif update_cross_attn is not None and update_cross_attn['skip_cross_attn']:
                        print("update_cross_attn not yet implemented for Chroma.", flush=True)
                        #img, txt_init = block(img, img_masks, txt, clip, rope, mask, update_cross_attn=update_cross_attn)
                    
                    else:
                        img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=attn_mask)

                    #img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=attn_mask)

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
                        out["img"] = block( args["img"],
                                            vec=args["vec"],
                                            pe=args["pe"],
                                            attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({ "img"             : img,
                                                                "vec"             : single_mod,
                                                                "pe"              : pe,
                                                                "attn_mask"       : attn_mask},
                                                                {"original_block" : block_wrap})
                    img = out["img"]
                else:

                    if   weight > 0 and mask is not None and     weight  <=      (i+len(self.double_blocks))/total_layers:
                        img = block(img, vec=single_mod, pe=pe, attn_mask=mask_zero)
                        
                    elif weight < 0 and mask is not None and abs(weight) <= (1 - (i+len(self.double_blocks))/total_layers):
                        img = block(img, vec=single_mod, pe=pe, attn_mask=mask_zero)
                        
                    elif floor > 0 and mask is not None and     floor  >=      (i+len(self.double_blocks))/total_layers:
                        mask_tmp = mask.clone()
                        mask_tmp[text_len:, text_len:] = 1.0
                        img = block(img, vec=single_mod, pe=pe, attn_mask=mask_tmp)
                        
                    elif floor < 0 and mask is not None and abs(floor) >= (1 - (i+len(self.double_blocks))/total_layers):
                        mask_tmp = mask.clone()
                        mask_tmp[text_len:, text_len:] = 1.0
                        img = block(img, vec=single_mod, pe=pe, attn_mask=mask_tmp)
                        
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
        x_orig = x.clone()
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
                                        #y           [i][None,...].clone(),
                                        guidance    [i][None,...].clone(),
                                        control, 
                                        update_cross_attn=update_cross_attn,
                                        transformer_options=transformer_options,
                                        UNCOND = UNCOND,
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
            x   = x_orig.clone().to(dtype)
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
            x   = x_orig.clone().to(dtype)
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



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1): #: and not UNCOND):
        if y0_style_pos is not None and y0_style_pos_weight != 0.0:
            y0_style_pos = y0_style_pos.to(dtype)
            x   = x.to(dtype)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps
            
            img = comfy.ldm.common_dit.pad_to_patch_size(denoised, (self.patch_size, self.patch_size))

            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

            img_y0_adain = comfy.ldm.common_dit.pad_to_patch_size(y0_style_pos, (self.patch_size, self.patch_size))

            img_y0_adain = rearrange(img_y0_adain, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64

            W = self.img_in.weight.data.to(dtype)   # shape [2560, 64]
            b = self.img_in.bias.data.to(dtype)     # shape [2560]
            
            denoised_embed = F.linear(img         .to(W), W, b).to(img)
            y0_adain_embed = F.linear(img_y0_adain.to(W), W, b).to(img_y0_adain)

            if transformer_options['y0_style_method'] == "AdaIN":
                if freqsep_mask is not None:
                    freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                    freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                
                if freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):
                    
                    #if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    #    self.y0_adain_embed = y0_adain_embed
                    #    self.adain_pw_cache = None
                        
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    if freqsep_lowpass_method == "median_alt": 
                        denoised_spatial_new = adain_patchwise_row_batch_medblur(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True)
                    elif freqsep_lowpass_method == "median_pw":
                        denoised_spatial_new = adain_patchwise_row_batch_realmedblur(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True, lowpass_weight=freqsep_lowpass_weight, highpass_weight=freqsep_highpass_weight)
                    elif freqsep_lowpass_method == "gaussian_pw": 
                        denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None and freqsep_lowpass_method == "distribution": 
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    denoised_spatial_new = adain_patchwise_strict_sortmatch9(denoised_spatial.clone(), y0_adain_spatial.clone().repeat(denoised_spatial.shape[0],1,1,1), kernel_size=freqsep_kernel_size, inner_kernel_size=freqsep_inner_kernel_size, mask=freqsep_mask, stride=freqsep_stride)

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
                
                #denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                #for adain_iter in range(EO("style_iter", 0)):
                #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                #    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                #    denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
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
                eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                #if eps.shape[0] == 2:
                #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            
            eps = eps.float()
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1): # and UNCOND):
        if y0_style_neg is not None and y0_style_neg_weight != 0.0:
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
                if freqsep_mask is not None:
                    freqsep_mask = freqsep_mask.view(1, 1, *freqsep_mask.shape[-2:]).float()
                    freqsep_mask = F.interpolate(freqsep_mask.float(), size=(h_len, w_len), mode='nearest-exact')
                
                if freqsep_lowpass_method is not None and freqsep_lowpass_method.endswith("pw"): #EO("adain_pw"):
                    
                    #if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    #    self.y0_adain_embed = y0_adain_embed
                    #    self.adain_pw_cache = None
                        
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    if freqsep_lowpass_method == "median_alt": 
                        denoised_spatial_new = adain_patchwise_row_batch_medblur(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True)
                    elif freqsep_lowpass_method == "median_pw":
                        denoised_spatial_new = adain_patchwise_row_batch_realmedblur(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size, use_median_blur=True, lowpass_weight=freqsep_lowpass_weight, highpass_weight=freqsep_highpass_weight)
                    elif freqsep_lowpass_method == "gaussian_pw": 
                        denoised_spatial_new = adain_patchwise_row_batch(denoised_spatial.clone(), y0_adain_spatial.clone(), sigma=freqsep_sigma, kernel_size=freqsep_kernel_size)
                    
                    denoised_embed = rearrange(denoised_spatial_new, "b c h w -> b (h w) c", h=h_len, w=w_len)

                elif freqsep_lowpass_method is not None and freqsep_lowpass_method == "distribution": 
                    denoised_spatial = rearrange(denoised_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)
                    y0_adain_spatial = rearrange(y0_adain_embed, "b (h w) c -> b c h w", h=h_len, w=w_len)

                    denoised_spatial_new = adain_patchwise_strict_sortmatch9(denoised_spatial.clone(), y0_adain_spatial.clone(), kernel_size=freqsep_kernel_size, inner_kernel_size=freqsep_inner_kernel_size, mask=freqsep_mask, stride=freqsep_stride)

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
                
                #denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                #for adain_iter in range(EO("style_iter", 0)):
                #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                #    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                #    denoised_embed = F.linear(denoised_embed         .to(W), W, b).to(img)
                #    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    
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







def adain_patchwise_row_batch_realmedblur(content: torch.Tensor, style: torch.Tensor, sigma: float = 1.0, kernel_size: int = None, eps: float = 1e-5, mask: torch.Tensor = None, use_median_blur: bool = False, lowpass_weight=1.0, highpass_weight=1.0) -> torch.Tensor:
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













def patchwise_sort_transfer9(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    src, ref: [B, C, N] where N = K*K
    Returns: [B, C, N] with values from ref permuted to match the sort-order of src.
    """
    src_sorted, src_idx  = src.sort(dim=-1)
    ref_sorted, _        = ref.sort(dim=-1)
    out = torch.zeros_like(src)
    out.scatter_(dim=-1, index=src_idx, src=ref_sorted)
    return out

def masked_patchwise_sort_transfer9(
    src       : torch.Tensor,         # [B, C, N]
    ref       : torch.Tensor,         # [B, C, N]
    mask_flat : torch.Tensor          # [B, N]  bool
) -> torch.Tensor:
    """
    Only rearrange N positions where mask_flat[b] is True... to be implemented fully later. 
    """
    B,C,N = src.shape
    out = src.clone()
    for b in range(B):
        valid = mask_flat[b]             # [N] boolean
        if valid.sum() == 0:
            continue
        sc = src[b,:,valid]              # [C, M]
        ss = ref[b,:,valid]              # [C, M]
        sc_s, idx = sc.sort(dim=-1)      # sort the channelz
        ss_s, _   = ss.sort(dim=-1)
        res = torch.zeros_like(sc)
        res.scatter_(dim=-1, index=idx, src=ss_s)
        out[b,:,valid] = res
    return out

def adain_patchwise_strict_sortmatch9(
    content           : torch.Tensor,        # [B,C,H,W]
    style             : torch.Tensor,        # [B,C,H,W]
    kernel_size       : int,
    inner_kernel_size : int = 1,
    stride            : int = 1,
    mask              : torch.Tensor = None  # [B,1,H,W]
) -> torch.Tensor:
    B,C,H,W = content.shape
    assert inner_kernel_size <= kernel_size
    pad       = kernel_size//2
    inner_off = (kernel_size - inner_kernel_size)//2

    # reflect-pad
    cp = F.pad(content, (pad,)*4, mode='reflect')
    sp = F.pad(style,   (pad,)*4, mode='reflect')
    out = content.clone()

    if mask is not None:
        mask = mask[:,0].bool()  # [B,H,W]

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            pc = cp[:, :, i:i+kernel_size, j:j+kernel_size]   # [B,C,K,K]
            ps = sp[:, :, i:i+kernel_size, j:j+kernel_size]

            Bc = pc.reshape(B, C, -1)
            Bs = ps.reshape(B, C, -1)

            matched_flat = patchwise_sort_transfer9(Bc, Bs)
            matched = matched_flat.view(B, C, kernel_size, kernel_size)

            y0, x0 = inner_off, inner_off
            y1, x1 = y0 + inner_kernel_size, x0 + inner_kernel_size
            inner = matched[:, :, y0:y1, x0:x1]  # [B,C,inner,inner]

            dst_y0 = i + y0 - pad
            dst_x0 = j + x0 - pad
            dst_y1 = dst_y0 + inner_kernel_size
            dst_x1 = dst_x0 + inner_kernel_size

            oy0 = max(dst_y0, 0); ox0 = max(dst_x0, 0)
            oy1 = min(dst_y1, H); ox1 = min(dst_x1, W)

            iy0 = oy0 - dst_y0; ix0 = ox0 - dst_x0
            iy1 = iy0 + (oy1 - oy0); ix1 = ix0 + (ox1 - ox0)

            if mask is None:
                out[:, :, oy0:oy1, ox0:ox1] = inner[:, :, iy0:iy1, ix0:ix1]
            else:
                ibm = mask[:, oy0:oy1, ox0:ox1]  # [B,inner,inner]
                for b in range(B):
                    sel = ibm[b]  # [inner,inner]   # w/ regard to kernel
                    if sel.any():
                        out[b:b+1, :, oy0:oy1, ox0:ox1][:, :,sel]   =   inner[b:b+1, :, iy0:iy1, ix0:ix1][:, :, sel]
    return out


