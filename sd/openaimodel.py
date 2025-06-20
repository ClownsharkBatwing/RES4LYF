from abc import abstractmethod
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging
import copy

from ..helper import ExtraOptions

from comfy.ldm.modules.diffusionmodules.util import (
    checkpoint,
    avg_pool_nd,
    timestep_embedding,
    AlphaBlender,
)
from comfy.ldm.modules.attention import SpatialTransformer, SpatialVideoTransformer, default
from comfy.ldm.util import exists
import comfy.patcher_extension
import comfy.ops
ops = comfy.ops.disable_weight_init

from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential, Upsample, Downsample, ResBlock, VideoResBlock
from ..latents import slerp_tensor, interpolate_spd, tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d

from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch, apply_scattersort, apply_scattersort_spatial

#This is needed because accelerate makes a copy of transformer_options which breaks "transformer_index"
def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
    for layer in ts:
        if isinstance(layer, VideoResBlock): # UNUSED
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif isinstance(layer, SpatialVideoTransformer):   # UNUSED
            x = layer(x, context, time_context, num_video_frames, image_only_indicator, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, SpatialTransformer):          # USED
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
        else:
            if "patches" in transformer_options and "forward_timestep_embed_patch" in transformer_options["patches"]:
                found_patched = False
                for class_type, handler in transformer_options["patches"]["forward_timestep_embed_patch"]:
                    if isinstance(layer, class_type):
                        x = handler(layer, x, emb, context, transformer_options, output_shape, time_context, num_video_frames, image_only_indicator)
                        found_patched = True
                        break
                if found_patched:
                    continue
            x = layer(x)
    return x



class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)

def apply_control(h, control, name):
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            try:
                h += ctrl
            except:
                logging.warning("warning control could not be applied {} {}".format(h.shape, ctrl.shape))
    return h

class ReUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=th.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        use_temporal_resblock=False,
        use_temporal_attention=False,
        time_context_dim=None,
        extra_ff_mix_layer=False,
        use_spatial_context=False,
        merge_strategy=None,
        merge_factor=0.0,
        video_kernel_size=None,
        disable_temporal_crossattention=False,
        max_ddpm_temb_period=10000,
        attn_precision=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)

        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids = n_embed is not None

        self.default_num_video_frames = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim, dtype=self.dtype, device=device)
            elif self.num_classes == "continuous":
                logging.debug("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disable_self_attn=False,
        ):
            if use_temporal_attention:
                return SpatialVideoTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=depth,
                    context_dim=context_dim,
                    time_context_dim=time_context_dim,
                    dropout=dropout,
                    ff_in=extra_ff_mix_layer,
                    use_spatial_context=use_spatial_context,
                    merge_strategy=merge_strategy,
                    merge_factor=merge_factor,
                    checkpoint=use_checkpoint,
                    use_linear=use_linear_in_transformer,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                    max_time_embed_period=max_ddpm_temb_period,
                    attn_precision=attn_precision,
                    dtype=self.dtype, device=device, operations=operations
                )
            else:
                return SpatialTransformer(
                                ch, num_heads, dim_head, depth=depth, context_dim=context_dim,
                                disable_self_attn=disable_self_attn, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=self.dtype, device=device, operations=operations
                            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_channels,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
            dtype=None,
            device=None,
            operations=ops
        ):
            if self.use_temporal_resblocks:
                return VideoResBlock(
                    merge_factor=merge_factor,
                    merge_strategy=merge_strategy,
                    video_kernel_size=video_kernel_size,
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=down,
                    up=up,
                    dtype=dtype,
                    device=device,
                    operations=operations
                )
            else:
                return ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_channels,
                    use_checkpoint=use_checkpoint,
                    dims=dims,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=down,
                    up=up,
                    dtype=dtype,
                    device=device,
                    operations=operations
                )

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(get_attention_layer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint)
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations
            )]

        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [get_attention_layer(  # always uses a self-attn
                                ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                                disable_self_attn=disable_middle_self_attn, use_checkpoint=use_checkpoint
                            ),
                get_resblock(
                    merge_factor=merge_factor,
                    merge_strategy=merge_strategy,
                    video_kernel_size=video_kernel_size,
                    ch=ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    out_channels=None,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                    device=device,
                    operations=operations
                )]
            self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            get_attention_layer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=self.dtype, device=device),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            operations.conv_nd(dims, model_channels, n_embed, 1, dtype=self.dtype, device=device),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def invert_conv2d(
        self,
        conv: torch.nn.Conv2d,
        z:    torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:

        B, C_in, H, W = original_shape
        C_out, _, kH, kW = conv.weight.shape
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

        return x_recon


    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timesteps, context, y, control, transformer_options, **kwargs)

    def _forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h_len, w_len = x.shape[-2:]
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        SIGMA = transformer_options['sigmas'].to(x) # timestep[0].unsqueeze(0) #/ 1000
        
        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

        y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight", 0.0)
        y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight", 0.0)
        y0_style_pos_synweight *= y0_style_pos_weight

        y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight", 0.0)
        y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight", 0.0)
        y0_style_neg_synweight *= y0_style_neg_weight
        
        x_orig = x.clone()

        x_orig, timesteps_orig, y_orig, context_orig = clone_inputs(x, timesteps, y, context)

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
        
        #floor     = min(floor, weight)
        mask_zero, mask_up_zero, mask_down_zero, mask_down2_zero = None, None, None, None
        txt_len = context.shape[1] # mask_obj[0].text_len
        
        y0_adain = transformer_options.get("y0_adain")
        if EO("eps_adain") and y0_adain is not None:
            x_init   = transformer_options.get("x_init").to(x)   # initial noise and/or image+noise from start of rk_sampler_beta() 
            y0_adain = y0_adain.to(x)
            #h_adain  = y0_adain + ((SIGMA ** 2 + 1) ** 0.5) * x_init
            #h_adain = (y0_adain + SIGMA * x_init) / 14.6172 #/ 14.58802395209580838323353293413173653
            #siggy = SIGMA / 14.6172
            #h_adain = ((0.13025*y0_adain + SIGMA * x_init) / ((SIGMA ** 2 + 1) ** 0.5))
            #if SIGMA > 14.61:
            #    h_adain = x_init.clone()
            h_adain = ((y0_adain + SIGMA * x_init) / ((SIGMA ** 2 + 1) ** 0.5))
            #h_adain = (1-siggy) * y0_adain + siggy * x_init
            h_adain = h_adain.expand(x.shape)
            h_adain_orig = h_adain.clone()
            #h = h_adain

        out_list = []
        for cond_iter in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
            
            x, timesteps, context = clone_inputs(x_orig[cond_iter].unsqueeze(0), timesteps_orig[cond_iter].unsqueeze(0), context_orig[cond_iter].unsqueeze(0))
            y = y_orig[cond_iter].unsqueeze(0).clone() if y_orig is not None else None
            if EO("eps_adain") and y0_adain is not None:
                h_adain = h_adain_orig.clone()[cond_iter].unsqueeze(0)
            
            mask, mask_up, mask_down, mask_down2 = None, None, None, None
            if not UNCOND and 'AttnMask' in transformer_options: # and weight != 0:
                AttnMask = transformer_options['AttnMask']
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                mask_up   = transformer_options['AttnMask'].mask_up.to('cuda')
                mask_down = transformer_options['AttnMask'].mask_down.to('cuda')
                if hasattr(transformer_options['AttnMask'], "mask_down2"):
                    mask_down2 = transformer_options['AttnMask'].mask_down2.to('cuda')
                if weight == 0:
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    mask, mask_up, mask_down, mask_down2 = None, None, None, None
                else:
                    context = transformer_options['RegContext'].context.to(context.dtype).to(context.device)
                    
                txt_len = context.shape[1]
                if mask_zero is None:
                    mask_zero = torch.ones_like(mask)
                    mask_zero[:, :txt_len] = mask[:, :txt_len]
                if mask_up_zero is None:
                    mask_up_zero = torch.ones_like(mask_up)
                    mask_up_zero[:, :txt_len] = mask_up[:, :txt_len]
                if mask_down_zero is None:
                    mask_down_zero = torch.ones_like(mask_down)
                    mask_down_zero[:, :txt_len] = mask_down[:, :txt_len]
                if mask_down2_zero is None and mask_down2 is not None:
                    mask_down2_zero = torch.ones_like(mask_down2)
                    mask_down2_zero[:, :txt_len] = mask_down2[:, :txt_len]


            if UNCOND and 'AttnMask_neg' in transformer_options: # and weight != 0:
                AttnMask = transformer_options['AttnMask_neg']
                mask = transformer_options['AttnMask_neg'].attn_mask.mask.to('cuda')
                mask_up   = transformer_options['AttnMask_neg'].mask_up.to('cuda')
                mask_down = transformer_options['AttnMask_neg'].mask_down.to('cuda')
                if hasattr(transformer_options['AttnMask_neg'], "mask_down2"):
                    mask_down2 = transformer_options['AttnMask_neg'].mask_down2.to('cuda')
                if weight == 0:
                    context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                    mask, mask_up, mask_down, mask_down2 = None, None, None, None
                else:
                    context = transformer_options['RegContext_neg'].context.to(context.dtype).to(context.device)
                    
                txt_len = context.shape[1]
                if mask_zero is None:
                    mask_zero = torch.ones_like(mask)
                    mask_zero[:, :txt_len] = mask[:, :txt_len]
                if mask_up_zero is None:
                    mask_up_zero = torch.ones_like(mask_up)
                    mask_up_zero[:, :txt_len] = mask_up[:, :txt_len]
                if mask_down_zero is None:
                    mask_down_zero = torch.ones_like(mask_down)
                    mask_down_zero[:, :txt_len] = mask_down[:, :txt_len]
                if mask_down2_zero is None and mask_down2 is not None:
                    mask_down2_zero = torch.ones_like(mask_down2)
                    mask_down2_zero[:, :txt_len] = mask_down2[:, :txt_len]

            elif UNCOND and 'AttnMask' in transformer_options:
                AttnMask = transformer_options['AttnMask']
                mask = transformer_options['AttnMask'].attn_mask.mask.to('cuda')
                mask_up   = transformer_options['AttnMask'].mask_up.to('cuda')
                mask_down = transformer_options['AttnMask'].mask_down.to('cuda')
                if hasattr(transformer_options['AttnMask'], "mask_down2"):
                    mask_down2 = transformer_options['AttnMask'].mask_down2.to('cuda')
                A       = context
                B       = transformer_options['RegContext'].context
                context = A.repeat(1,    (B.shape[1] // A.shape[1]) + 1, 1)[:,   :B.shape[1], :]
                
                txt_len = context.shape[1]
                if mask_zero is None:
                    mask_zero = torch.ones_like(mask)
                    mask_zero[:, :txt_len] = mask[:, :txt_len]
                if mask_up_zero is None:
                    mask_up_zero = torch.ones_like(mask_up)
                    mask_up_zero[:, :txt_len] = mask_up[:, :txt_len]
                if mask_down_zero is None:
                    mask_down_zero = torch.ones_like(mask_down)
                    mask_down_zero[:, :txt_len] = mask_down[:, :txt_len]
                if mask_down2_zero is None and mask_down2 is not None:
                    mask_down2_zero = torch.ones_like(mask_down2)
                    mask_down2_zero[:, :txt_len] = mask_down2[:, :txt_len]
                if weight == 0:                                                                             # ADDED 5/23/2025
                    mask, mask_up, mask_down, mask_down2 = None, None, None, None


            if mask is not None:
                if mask is not None and not type(mask[0][0].item()) == bool:
                    mask = mask.to(x.dtype)
                if mask_up is not None and not type(mask_up[0][0].item()) == bool:
                    mask_up = mask_up.to(x.dtype)
                if mask_down is not None and not type(mask_down[0][0].item()) == bool:
                    mask_down = mask_down.to(x.dtype)
                if mask_down2 is not None and not type(mask_down2[0][0].item()) == bool:
                    mask_down2 = mask_down2.to(x.dtype)
                    
                if mask_zero is not None and not type(mask_zero[0][0].item()) == bool:
                    mask_zero = mask_zero.to(x.dtype)
                if mask_up_zero is not None and not type(mask_up_zero[0][0].item()) == bool:
                    mask_up_zero = mask_up_zero.to(x.dtype)
                if mask_down_zero is not None and not type(mask_down_zero[0][0].item()) == bool:
                    mask_down_zero = mask_down_zero.to(x.dtype)
                if mask_down2_zero is not None and not type(mask_down2_zero[0][0].item()) == bool:
                    mask_down2_zero = mask_down2_zero.to(x.dtype)
                    
                transformer_options['cross_mask'] = mask[:,:txt_len]
                transformer_options['self_mask']  = mask[:,txt_len:]
                transformer_options['cross_mask_up'] = mask_up[:,:txt_len]
                transformer_options['self_mask_up']  = mask_up[:,txt_len:]
                transformer_options['cross_mask_down'] = mask_down[:,:txt_len]
                transformer_options['self_mask_down']  = mask_down[:,txt_len:]
                transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
            

            total_layers = len(self.input_blocks) + len(self.middle_block) + len(self.output_blocks)

            num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
            image_only_indicator = kwargs.get("image_only_indicator", None)
            time_context = kwargs.get("time_context", None)

            assert (y is not None) == (
                self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs, hs_adain = [], []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
            emb = self.time_embed(t_emb)

            if "emb_patch" in transformer_patches:
                patch = transformer_patches["emb_patch"]
                for p in patch:
                    emb = p(emb, self.model_channels, transformer_options)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            h = x
            for id, module in enumerate(self.input_blocks):
                transformer_options["block"] = ("input", id)

                if mask is not None:
                    transformer_options['cross_mask'] = mask[:,:txt_len]
                    transformer_options['self_mask']  = mask[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                    transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                    
                if   weight > 0 and mask is not None and     weight  <      id/total_layers:
                    transformer_options['cross_mask'] = None
                    transformer_options['self_mask']  = None
                
                elif weight < 0 and mask is not None and abs(weight) < (1 - id/total_layers):
                    transformer_options['cross_mask'] = None
                    transformer_options['self_mask']  = None
                    
                elif floor > 0 and mask is not None and       floor  >      id/total_layers:
                    transformer_options['cross_mask'] = mask_zero[:,:txt_len]
                    transformer_options['self_mask']  = mask_zero[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up_zero[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up_zero[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down_zero[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down_zero[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                    transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                
                elif floor < 0 and mask is not None and   abs(floor) > (1 - id/total_layers):
                    transformer_options['cross_mask'] = mask_zero[:,:txt_len]
                    transformer_options['self_mask']  = mask_zero[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up_zero[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up_zero[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down_zero[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down_zero[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                    transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                
                
                h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                h = apply_control(h, control, 'input')
                if "input_block_patch" in transformer_patches:
                    patch = transformer_patches["input_block_patch"]
                    for p in patch:
                        h = p(h, transformer_options)
                
                if EO("eps_adain") and y0_adain is not None:
                    h_adain = forward_timestep_embed(module, h_adain, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                    h_adain = apply_control(h_adain, control, 'input')
                    if "input_block_patch" in transformer_patches:
                        patch = transformer_patches["input_block_patch"]
                        for p in patch:
                            h_adain = p(h_adain, transformer_options)
                    hs_adain.append(h_adain)
                    h = apply_scattersort_spatial(h, h_adain)
                
                hs.append(h)
                
                if "input_block_patch_after_skip" in transformer_patches:
                    patch = transformer_patches["input_block_patch_after_skip"]
                    for p in patch:
                        h = p(h, transformer_options)

            transformer_options["block"] = ("middle", 0)
            if self.middle_block is not None:

                if mask is not None:
                    transformer_options['cross_mask'] = mask[:,:txt_len]
                    transformer_options['self_mask']  = mask[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                    transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                    
                if   weight > 0 and mask is not None and     weight  <      (len(self.input_blocks) + 1)/total_layers:
                    transformer_options['cross_mask'] = None
                    transformer_options['self_mask']  = None
                
                elif weight < 0 and mask is not None and abs(weight) < (1 - (len(self.input_blocks) + 1)/total_layers):
                    transformer_options['cross_mask'] = None
                    transformer_options['self_mask']  = None
                    
                elif floor > 0 and mask is not None and       floor  >      (len(self.input_blocks) + 1)/total_layers:
                    transformer_options['cross_mask'] = mask_zero[:,:txt_len]
                    transformer_options['self_mask']  = mask_zero[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up_zero[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up_zero[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down_zero[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down_zero[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                    transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                
                elif floor < 0 and mask is not None and   abs(floor) > (1 - (len(self.input_blocks) + 1)/total_layers):
                    transformer_options['cross_mask'] = mask_zero[:,:txt_len]
                    transformer_options['self_mask']  = mask_zero[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up_zero[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up_zero[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down_zero[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down_zero[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                    transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None

                h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                if EO("eps_adain") and y0_adain is not None:
                    h_adain = forward_timestep_embed(self.middle_block, h_adain, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                    #h = apply_scattersort_spatial(h, h_adain)
            
            h       = apply_control(h,       control, 'middle')
            if EO("eps_adain") and y0_adain is not None:
                h_adain = apply_control(h_adain, control, 'middle')
                h = apply_scattersort_spatial(h, h_adain)

            for id, module in enumerate(self.output_blocks):
                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, 'output')

                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        h, hsp = p(h, hsp, transformer_options)

                h = th.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                    
                if EO("eps_adain_out") and y0_adain is not None:
                    hsp_adain = hs_adain.pop()
                    hsp_adain = apply_control(hsp_adain, control, 'output')

                    if "output_block_patch" in transformer_patches:
                        patch = transformer_patches["output_block_patch"]
                        for p in patch:
                            h_adain, hsp_adain = p(h_adain, hsp_adain, transformer_options)

                    h_adain = th.cat([h_adain, hsp_adain], dim=1)
                    del hsp_adain
                    h = apply_scattersort_spatial(h, h_adain)
                
                if mask is not None:
                    transformer_options['cross_mask'] = mask[:,:txt_len]
                    transformer_options['self_mask']  = mask[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                    transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                    
                if   weight > 0 and mask is not None and     weight  <      (len(self.input_blocks) + 1 + id)/total_layers:
                    transformer_options['cross_mask'] = None
                    transformer_options['self_mask']  = None
                
                elif weight < 0 and mask is not None and abs(weight) < (1 - (len(self.input_blocks) + 1 + id)/total_layers):
                    transformer_options['cross_mask'] = None
                    transformer_options['self_mask']  = None
                    
                elif floor > 0 and mask is not None and       floor  >      (len(self.input_blocks) + 1 + id)/total_layers:
                    transformer_options['cross_mask'] = mask_zero[:,:txt_len]
                    transformer_options['self_mask']  = mask_zero[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up_zero[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up_zero[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down_zero[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down_zero[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                    transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                
                elif floor < 0 and mask is not None and   abs(floor) > (1 - (len(self.input_blocks) + 1 + id)/total_layers):
                    transformer_options['cross_mask'] = mask_zero[:,:txt_len]
                    transformer_options['self_mask']  = mask_zero[:,txt_len:]
                    transformer_options['cross_mask_up'] = mask_up_zero[:,:txt_len]
                    transformer_options['self_mask_up']  = mask_up_zero[:,txt_len:]
                    transformer_options['cross_mask_down'] = mask_down_zero[:,:txt_len]
                    transformer_options['self_mask_down']  = mask_down_zero[:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                    transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None

                    
                h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                if EO("eps_adain_out") and y0_adain is not None:
                    h_adain = forward_timestep_embed(module, h_adain, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
                    h = apply_scattersort_spatial(h, h_adain)
            
            h = h.type(x.dtype)
            #if EO("eps_adain_out") and y0_adain is not None:
            #    h_adain = h_adain.type(x.dtype)
            
            if self.predict_codebook_ids:
                eps = self.id_predictor(h)
            else:
                eps = self.out(h)
                
            out_list.append(eps)
            
        eps = torch.stack(out_list, dim=0).squeeze(dim=1)



        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        h_len //= self.Retrojector.patch_size
        w_len //= self.Retrojector.patch_size
        
        if y0_style_pos is not None:
            y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight")
            y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight")
            y0_style_pos_synweight *= y0_style_pos_weight
            y0_style_pos_mask = transformer_options.get("y0_style_pos_mask")
            y0_style_pos_mask_edge = transformer_options.get("y0_style_pos_mask_edge")

            y0_style_pos = y0_style_pos.to(dtype)
            #x   = x.to(dtype)
            x   = x_orig.clone().to(torch.float64) * ((SIGMA ** 2 + 1) ** 0.5)
            eps = eps.to(dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps
            
            denoised_embed = self.Retrojector.embed(denoised)     # 2,4,96,168 -> 2,16128,320
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
            #x   = x.to(dtype)
            x   = x_orig.clone().to(torch.float64) * ((SIGMA ** 2 + 1) ** 0.5)
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









def clone_inputs_unsafe(*args, index: int=None):

    if index is None:
        return tuple(x.clone() for x in args)
    else:
        return tuple(x[index].unsqueeze(0).clone() for x in args)
    
    
def clone_inputs(*args, index: int = None):
    if index is None:
        return tuple(x.clone() if x is not None else None for x in args)
    else:
        return tuple(x[index].unsqueeze(0).clone() if x is not None else None for x in args)
    
    

