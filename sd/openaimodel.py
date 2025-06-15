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
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        SIGMA = transformer_options['sigmas'] # timestep[0].unsqueeze(0) #/ 1000
        
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

        out_list = []
        for cond_iter in range(len(transformer_options['cond_or_uncond'])):
            UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
            
            x, timesteps, context = clone_inputs(x_orig[cond_iter].unsqueeze(0), timesteps_orig[cond_iter].unsqueeze(0), context_orig[cond_iter].unsqueeze(0))
            y = y_orig[cond_iter].unsqueeze(0).clone() if y_orig is not None else None
            
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
            hs = []
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
            h = apply_control(h, control, 'middle')


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
            h = h.type(x.dtype)
            
            if self.predict_codebook_ids:
                eps = self.id_predictor(h)
            else:
                eps = self.out(h)
                
            out_list.append(eps)
            
        eps = torch.stack(out_list, dim=0).squeeze(dim=1)



        
        dtype = eps.dtype if self.style_dtype is None else self.style_dtype
        pinv_dtype = torch.float32 if dtype != torch.float64 else dtype
        W_inv = None
        
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1): #: and not UNCOND):
        if y0_style_pos is not None and y0_style_pos_weight != 0.0:
            y0_style_pos = y0_style_pos.to(pinv_dtype)
            x   = x_orig.clone().to(pinv_dtype) * ((SIGMA ** 2 + 1) ** 0.5)
            eps = eps.to(pinv_dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA
            denoised = x - sigma * eps

            x_embedder = copy.deepcopy(self.input_blocks[0][0]).to(denoised)
            
            denoised_embed = x_embedder(denoised)
            y0_adain_embed = x_embedder(y0_style_pos)

            denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
            y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")
            
            h_len, w_len = eps.shape[-2], eps.shape[-1]

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
                    
            elif transformer_options['y0_style_method'] == "WCT":
                if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    self.y0_adain_embed = y0_adain_embed
                    
                    f_s          = y0_adain_embed[0].clone()
                    self.mu_s    = f_s.mean(dim=0, keepdim=True)
                    f_s_centered = f_s - self.mu_s
                    
                    cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                    S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                    
                    whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                    self.y0_color  = whiten.to(f_s_centered)

                for wct_i in range(eps.shape[0]):
                    f_c          = denoised_embed[wct_i].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T.to(f_c_whitened) + self.mu_s.to(f_c_whitened)
                    
                    denoised_embed[wct_i] = f_cs




                """if transformer_options.get('y0_standard_guide') is not None:
                    y0_standard_guide = transformer_options.get('y0_standard_guide')
                    
                    img_y0_standard_guide = comfy.ldm.common_dit.pad_to_patch_size(y0_standard_guide, (self.patch_size, self.patch_size))
                    #img_sizes_y0_standard_guide = None
                    #img_y0_standard_guide, img_masks_y0_standard_guide, img_sizes_y0_standard_guide = self.patchify(img_y0_standard_guide, self.max_seq, img_sizes_y0_standard_guide) 
                    h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
                    w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
                    img_y0_standard_guide = rearrange(img_y0_standard_guide, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
                    y0_standard_guide_embed = F.linear(img_y0_standard_guide.to(W), W, b).to(img_y0_standard_guide)
                    
                    f_c          = y0_standard_guide_embed[0].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T.to(y0_standard_guide) + self.mu_s.to(y0_standard_guide)
                    
                    f_cs = (f_cs - b) @ torch.linalg.pinv(W.to(f_cs)).T.to(f_cs)
                    #y0_standard_guide = self.unpatchify (f_cs.unsqueeze(0), img_sizes_y0_standard_guide)
                    f_cs = f_cs.to(eps)
                    y0_standard_guide = rearrange(f_cs.unsqueeze(0), "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
                    self.y0_standard_guide = y0_standard_guide
                    
                if transformer_options.get('y0_inv_standard_guide') is not None:
                    y0_inv_standard_guide = transformer_options.get('y0_inv_standard_guide')
                    
                    img_y0_standard_guide = comfy.ldm.common_dit.pad_to_patch_size(y0_inv_standard_guide, (self.patch_size, self.patch_size))
                    #img_sizes_y0_standard_guide = None
                    #img_y0_standard_guide, img_masks_y0_standard_guide, img_sizes_y0_standard_guide = self.patchify(img_y0_standard_guide, self.max_seq, img_sizes_y0_standard_guide) 
                    h_len = ((h + (patch_size // 2)) // patch_size) # h_len 96
                    w_len = ((w + (patch_size // 2)) // patch_size) # w_len 96
                    img_y0_standard_guide = rearrange(img_y0_standard_guide, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size) # img 1,9216,64     1,16,128,128 -> 1,4096,64
                    y0_standard_guide_embed = F.linear(img_y0_standard_guide.to(W), W, b).to(img_y0_standard_guide)
                    
                    f_c          = y0_standard_guide_embed[0].clone()
                    
                    #f_c          = y0_inv_standard_guide[0].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T.to(y0_inv_standard_guide) + self.mu_s.to(y0_inv_standard_guide)
                    
                    f_cs = (f_cs - b) @ torch.linalg.pinv(W.to(f_cs)).T.to(f_cs)
                    #y0_inv_standard_guide = self.unpatchify (f_cs.unsqueeze(0), img_sizes_y0_standard_guide)
                    f_cs = f_cs.to(eps)
                    y0_inv_standard_guide = rearrange(f_cs.unsqueeze(0), "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
                    self.y0_inv_standard_guide = y0_inv_standard_guide"""




            
            denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=eps.shape[-1])
            denoised_approx = self.invert_conv2d(x_embedder, denoised_embed, x_orig.shape)
            denoised_approx = denoised_approx.to(eps)

            
            eps = (x - denoised_approx) / sigma
            
            #UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1

            if eps.shape[0] == 1 and transformer_options['cond_or_uncond'][0] == 1:
                eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                #if eps.shape[0] == 2:
                #    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            else: #if not UNCOND:
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_pos_weight * (eps[1] - eps_orig[1])
                    eps[0] = eps_orig[0] + y0_style_pos_synweight * (eps[0] - eps_orig[0])
                else:
                    eps[0] = eps_orig[0] + y0_style_pos_weight * (eps[0] - eps_orig[0])
            
            eps = eps.float()
        
        #if eps.shape[0] == 2 or (eps.shape[0] == 1): # and UNCOND):
        if y0_style_neg is not None and y0_style_neg_weight != 0.0:
            y0_style_neg = y0_style_neg.to(pinv_dtype)
            x   = x_orig.clone().to(pinv_dtype)* ((SIGMA ** 2 + 1) ** 0.5)
            eps = eps.to(pinv_dtype)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps

            x_embedder = copy.deepcopy(self.input_blocks[0][0]).to(denoised)
            
            denoised_embed = x_embedder(denoised)
            y0_adain_embed = x_embedder(y0_style_neg)

            denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
            y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

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

                    
            elif transformer_options['y0_style_method'] == "WCT":
                if self.y0_adain_embed is None or self.y0_adain_embed.shape != y0_adain_embed.shape or torch.norm(self.y0_adain_embed - y0_adain_embed) > 0:
                    self.y0_adain_embed = y0_adain_embed
                    
                    f_s          = y0_adain_embed[0].clone()
                    self.mu_s    = f_s.mean(dim=0, keepdim=True)
                    f_s_centered = f_s - self.mu_s
                    
                    #cov = (f_s_centered.T.double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)
                    cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

                    S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
                    
                    whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.T
                    self.y0_color  = whiten.to(f_s_centered)

                for wct_i in range(eps.shape[0]):
                    f_c          = denoised_embed[wct_i].clone()
                    mu_c         = f_c.mean(dim=0, keepdim=True)
                    f_c_centered = f_c - mu_c
                    
                    #cov = (f_c_centered.T.double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)
                    cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

                    S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
                    inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
                    
                    whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.T
                    whiten = whiten.to(f_c_centered)

                    f_c_whitened = f_c_centered @ whiten.T
                    f_cs         = f_c_whitened @ self.y0_color.T.to(f_c_whitened) + self.mu_s.to(f_c_whitened)
                    
                    denoised_embed[wct_i] = f_cs

            denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=eps.shape[-1])
            denoised_approx = self.invert_conv2d(x_embedder, denoised_embed, x_orig.shape)
            denoised_approx = denoised_approx.to(eps)
            
            
            if eps.shape[0] == 1 and not transformer_options['cond_or_uncond'][0] == 1:
                eps[0] = eps_orig[0] + y0_style_neg_synweight * (eps[0] - eps_orig[0])
            else:
                eps = (x - denoised_approx) / sigma
                eps[0] = eps_orig[0] + y0_style_neg_weight * (eps[0] - eps_orig[0])
                if eps.shape[0] == 2:
                    eps[1] = eps_orig[1] + y0_style_neg_synweight * (eps[1] - eps_orig[1])
            
            eps = eps.float()
        
        return eps
    
    

def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
    return content


def adain_seq(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return ((content - content.mean(1, keepdim=True)) / (content.std(1, keepdim=True) + eps)) * (style.std(1, keepdim=True) + eps) + style.mean(1, keepdim=True)

    
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

