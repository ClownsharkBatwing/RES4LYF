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
from .attention import ReSpatialTransformer, ReBasicTransformerBlock
from comfy.ldm.util import exists
import comfy.patcher_extension
import comfy.ops
ops = comfy.ops.disable_weight_init

from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential, Upsample, Downsample, ResBlock, VideoResBlock
from ..latents import slerp_tensor, interpolate_spd, tile_latent, untile_latent, gaussian_blur_2d, median_blur_2d

from ..style_transfer import apply_scattersort_masked, apply_scattersort_tiled, adain_seq_inplace, adain_patchwise_row_batch_med, adain_patchwise_row_batch, apply_scattersort, apply_scattersort_spatial, StyleMMDiT_Model, StyleUNet_Model

#This is needed because accelerate makes a copy of transformer_options which breaks "transformer_index"
def forward_timestep_embed(ts, x, emb, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None, style_block=None):
    for layer in ts:
        if isinstance(layer, VideoResBlock): # UNUSED
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, TimestepBlock):  # ResBlock(TimestepBlock)
            x = layer(x, emb, style_block.res_block)
            x = style_block(x, "res")
        elif isinstance(layer, SpatialVideoTransformer):   # UNUSED
            x = layer(x, context, time_context, num_video_frames, image_only_indicator, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, ReSpatialTransformer):          # USED
            x = layer(x, context, style_block.spatial_block, transformer_options,)
            x = style_block(x, "spatial")
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample):
            x = layer(x, output_shape=output_shape)
            x = style_block(x, "resample")
        elif isinstance(layer, Downsample):
            x = layer(x)
            x = style_block(x, "resample")
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




class ReResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=ops
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample(channels, False, dims, dtype=dtype, device=device)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Downsample(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device
                ),
            )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device)
            ,
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device
            )
        else:
            self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)

    def forward(self, x, emb, style_block=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, style_block), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb, style_block=None):
        #if self.updown: # not used with sdxl?
        #    in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        #    h = in_rest(x)
        #    h = self.h_upd(h)
        #    x = self.x_upd(x)
        #    h = in_conv(h)
        #else:
        #    h = self.in_layers(x)
        
        h = self.in_layers[0](x)
        h = style_block(h, "in_norm")
        
        h = self.in_layers[1](h)
        h = style_block(h, "in_silu")
        
        h = self.in_layers[2](h)
        h = style_block(h, "in_conv")
        

        emb_out = None
        if not self.skip_t_emb:
            #emb_out = self.emb_layers(emb).type(h.dtype)
            emb_out = self.emb_layers[0](emb).type(h.dtype)
            emb_out = style_block(emb_out, "emb_silu")
            
            emb_out = self.emb_layers[1](emb_out)
            emb_out = style_block(emb_out, "emb_linear")
            
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
                
        if self.use_scale_shift_norm: # not used with sdxl?
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = emb_out.movedim(1, 2)
                h = h + emb_out
                h = style_block(h, "emb_res")
            #h = self.out_layers(h)
            h = self.out_layers[0](h)
            h = style_block(h, "out_norm")
            
            h = self.out_layers[1](h)
            h = style_block(h, "out_silu")
            
            h = self.out_layers[3](h) # [2] is dropout
            h = style_block(h, "out_conv")
            
        res_out = self.skip_connection(x) + h
        res_out = style_block(res_out, "residual")
        return res_out   
        #return self.skip_connection(x) + h




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
        dropout                         = 0,
        channel_mult                    = (1, 2, 4, 8),
        conv_resample                   = True,
        dims                            = 2,
        num_classes                     = None,
        use_checkpoint                  = False,
        dtype                           = th.float32,
        num_heads                       = -1,
        num_head_channels               = -1,
        num_heads_upsample              = -1,
        use_scale_shift_norm            = False,
        resblock_updown                 = False,
        use_new_attention_order         = False,
        use_spatial_transformer         = False,    # custom transformer support
        transformer_depth               = 1,              # custom transformer support
        context_dim                     = None,                 # custom transformer support
        n_embed                         = None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy                          = True,
        disable_self_attentions         = None,
        num_attention_blocks            = None,
        disable_middle_self_attn        = False,
        use_linear_in_transformer       = False,
        adm_in_channels                 = None,
        transformer_depth_middle        = None,
        transformer_depth_output        = None,
        use_temporal_resblock           = False,
        use_temporal_attention          = False,
        time_context_dim                = None,
        extra_ff_mix_layer              = False,
        use_spatial_context             = False,
        merge_strategy                  = None,
        merge_factor                    = 0.0,
        video_kernel_size               = None,
        disable_temporal_crossattention = False,
        max_ddpm_temb_period            = 10000,
        attn_precision                  = None,
        device                          = None,
        operations                      = ops,
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

        self.dropout                = dropout
        self.channel_mult           = channel_mult
        self.conv_resample          = conv_resample
        self.num_classes            = num_classes
        self.use_checkpoint         = use_checkpoint
        self.dtype                  = dtype
        self.num_heads              = num_heads
        self.num_head_channels      = num_head_channels
        self.num_heads_upsample     = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids   = n_embed is not None

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
                    depth                           = depth,
                    context_dim                     = context_dim,
                    time_context_dim                = time_context_dim,
                    dropout                         = dropout,
                    ff_in                           = extra_ff_mix_layer,
                    use_spatial_context             = use_spatial_context,
                    merge_strategy                  = merge_strategy,
                    merge_factor                    = merge_factor,
                    checkpoint                      = use_checkpoint,
                    use_linear                      = use_linear_in_transformer,
                    disable_self_attn               = disable_self_attn,
                    disable_temporal_crossattention = disable_temporal_crossattention,
                    max_time_embed_period           = max_ddpm_temb_period,
                    attn_precision                  = attn_precision,
                    dtype=self.dtype, device=device, operations=operations,
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
            down       = False,
            up         = False,
            dtype      = None,
            device     = None,
            operations = ops
        ):
            if self.use_temporal_resblocks:
                return VideoResBlock(
                    merge_factor         = merge_factor,
                    merge_strategy       = merge_strategy,
                    video_kernel_size    = video_kernel_size,
                    channels             = ch,
                    emb_channels         = time_embed_dim,
                    dropout              = dropout,
                    out_channels         = out_channels,
                    dims                 = dims,
                    use_checkpoint       = use_checkpoint,
                    use_scale_shift_norm = use_scale_shift_norm,
                    down                 = down,
                    up                   = up,
                    dtype=dtype, device=device, operations=operations,
                )
            else:
                return ResBlock(
                    channels             = ch,
                    emb_channels         = time_embed_dim,
                    dropout              = dropout,
                    out_channels         = out_channels,
                    use_checkpoint       = use_checkpoint,
                    dims                 = dims,
                    use_scale_shift_norm = use_scale_shift_norm,
                    down                 = down,
                    up                   = up,
                    dtype=dtype, device=device, operations=operations,
                )

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    get_resblock(
                        merge_factor         = merge_factor,
                        merge_strategy       = merge_strategy,
                        video_kernel_size    = video_kernel_size,
                        ch                   = ch,
                        time_embed_dim       = time_embed_dim,
                        dropout              = dropout,
                        out_channels         = mult * model_channels,
                        dims                 = dims,
                        use_checkpoint       = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                        dtype=self.dtype, device=device, operations=operations,
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
                            merge_factor         = merge_factor,
                            merge_strategy       = merge_strategy,
                            video_kernel_size    = video_kernel_size,
                            ch                   = ch,
                            time_embed_dim       = time_embed_dim,
                            dropout              = dropout,
                            out_channels         = out_ch,
                            dims                 = dims,
                            use_checkpoint       = use_checkpoint,
                            use_scale_shift_norm = use_scale_shift_norm,
                            down                 = True,
                            dtype=self.dtype, device=device, operations=operations,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations)
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
                merge_factor         = merge_factor,
                merge_strategy       = merge_strategy,
                video_kernel_size    = video_kernel_size,
                ch                   = ch,
                time_embed_dim       = time_embed_dim,
                dropout              = dropout,
                out_channels         = None,
                dims                 = dims,
                use_checkpoint       = use_checkpoint,
                use_scale_shift_norm = use_scale_shift_norm,
                dtype=self.dtype, device=device, operations=operations,
            )]

        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [get_attention_layer(  # always uses a self-attn
                                ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                                disable_self_attn=disable_middle_self_attn, use_checkpoint=use_checkpoint
                            ),
                get_resblock(
                    merge_factor         = merge_factor,
                    merge_strategy       = merge_strategy,
                    video_kernel_size    = video_kernel_size,
                    ch                   = ch,
                    time_embed_dim       = time_embed_dim,
                    dropout              = dropout,
                    out_channels         = None,
                    dims                 = dims,
                    use_checkpoint       = use_checkpoint,
                    use_scale_shift_norm = use_scale_shift_norm,
                    dtype=self.dtype, device=device, operations=operations,
                )]
            self.middle_block = TimestepEmbedSequential(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor         = merge_factor,
                        merge_strategy       = merge_strategy,
                        video_kernel_size    = video_kernel_size,
                        ch                   = ch + ich,
                        time_embed_dim       = time_embed_dim,
                        dropout              = dropout,
                        out_channels         = model_channels * mult,
                        dims                 = dims,
                        use_checkpoint       = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                        dtype=self.dtype, device=device, operations=operations,
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
                            merge_factor         = merge_factor,
                            merge_strategy       = merge_strategy,
                            video_kernel_size    = video_kernel_size,
                            ch                   = ch,
                            time_embed_dim       = time_embed_dim,
                            dropout              = dropout,
                            out_channels         = out_ch,
                            dims                 = dims,
                            use_checkpoint       = use_checkpoint,
                            use_scale_shift_norm = use_scale_shift_norm,
                            up                   = True,
                            dtype=self.dtype, device=device, operations=operations,
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
        img_len = h_len * w_len
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        SIGMA = transformer_options['sigmas'].to(x) # timestep[0].unsqueeze(0) #/ 1000

        img_slice = slice(None, -1) #slice(None, img_len)   # for the sake of cross attn... :-1
        txt_slice = slice(None, -1)
        
        EO = transformer_options.get("ExtraOptions", ExtraOptions(""))
        if EO is not None:
            EO.mute = True

        if EO("zero_heads"):
            HEADS = 0
        else:
            HEADS = 10 # self.input_blocks[4][1].transformer_blocks[0].attn2.heads # HEADS = 10

        StyleMMDiT = transformer_options.get('StyleMMDiT', StyleUNet_Model())        
        StyleMMDiT.set_len(h_len, w_len, img_slice, txt_slice, HEADS=HEADS)
        StyleMMDiT.Retrojector = self.Retrojector if hasattr(self, "Retrojector") else None
        transformer_options['StyleMMDiT'] = None
        
        x_tmp = transformer_options.get("x_tmp")
        if x_tmp is not None:
            x_tmp = x_tmp.clone() / ((SIGMA ** 2 + 1) ** 0.5)
            x_tmp = x_tmp.expand_as(x) # (x.shape[0], -1, -1, -1) # .clone().to(x)
        
        y0_style, img_y0_style = None, None

        
        x_orig, timesteps_orig, y_orig, context_orig = clone_inputs(x, timesteps, y, context)
        h_orig = x_orig.clone()

        weight    = -1 * transformer_options.get("regional_conditioning_weight", 0.0)
        floor     = -1 * transformer_options.get("regional_conditioning_floor",  0.0)
        
        #floor     = min(floor, weight)
        mask_zero, mask_up_zero, mask_down_zero, mask_down2_zero = None, None, None, None
        txt_len = context.shape[1] # mask_obj[0].text_len
        

        z_ = transformer_options.get("z_")   # initial noise and/or image+noise from start of rk_sampler_beta() 
        rk_row = transformer_options.get("row") # for "smart noise"
        if z_ is not None:
            x_init = z_[rk_row].to(x)
        elif 'x_init' in transformer_options:
            x_init = transformer_options.get('x_init').to(x)

        # recon loop to extract exact noise pred for scattersort guide assembly
        RECON_MODE = StyleMMDiT.noise_mode == "recon"
        recon_iterations = 2 if StyleMMDiT.noise_mode == "recon" else 1
        for recon_iter in range(recon_iterations):
            y0_style = StyleMMDiT.guides
            y0_style_active = True if type(y0_style) == torch.Tensor else False
            
            RECON_MODE = True     if StyleMMDiT.noise_mode == "recon" and recon_iter == 0     else False
            
            ISIGMA = SIGMA
            if StyleMMDiT.noise_mode == "recon" and recon_iter == 1: 
                ISIGMA = SIGMA * EO("ISIGMA_FACTOR", 1.0)
                
                model_sampling = transformer_options.get('model_sampling')     
                timesteps_orig = model_sampling.timestep(ISIGMA).expand_as(timesteps_orig)
                
                x_recon = x_tmp if x_tmp is not None else x_orig
                #noise_prediction = x_recon + (1-SIGMA.to(x_recon)) * eps.to(x_recon)
                noise_prediction = eps.to(x_recon)
                denoised = x_recon * ((SIGMA.to(x_recon) ** 2 + 1) ** 0.5)   -   SIGMA.to(x_recon) * eps.to(x_recon)
                
                denoised = StyleMMDiT.apply_recon_lure(denoised, y0_style.to(x_recon))   # .to(denoised)

                new_x = (denoised + ISIGMA.to(x_recon) * noise_prediction) / ((ISIGMA.to(x_recon) ** 2 + 1) ** 0.5)
                h_orig = new_x.clone().to(x)
                x_init = noise_prediction
            elif StyleMMDiT.noise_mode == "bonanza":
                x_init = torch.randn_like(x_init)

            if y0_style_active:
                if y0_style.sum() == 0.0 and y0_style.std() == 0.0:
                    y0_style_noised = x.clone()
                else:
                    y0_style_noised = (y0_style + ISIGMA.to(y0_style) * x_init.expand_as(x).to(y0_style)) / ((ISIGMA.to(y0_style) ** 2 + 1) ** 0.5)    #x_init.expand(x.shape[0],-1,-1,-1).to(y0_style)) 

            out_list = []
            for cond_iter in range(len(transformer_options['cond_or_uncond'])):
                UNCOND = transformer_options['cond_or_uncond'][cond_iter] == 1
                
                bsz_style = y0_style.shape[0] if y0_style_active else 0
                bsz       = 1 if RECON_MODE else bsz_style + 1
                
                h, timesteps, context = clone_inputs(h_orig[cond_iter].unsqueeze(0), timesteps_orig[cond_iter].unsqueeze(0), context_orig[cond_iter].unsqueeze(0))
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
                    if mask            is not None and not type(mask[0][0]           .item()) == bool:
                        mask            = mask           .to(x.dtype)
                    if mask_up         is not None and not type(mask_up[0][0]        .item()) == bool:
                        mask_up         = mask_up        .to(x.dtype)
                    if mask_down       is not None and not type(mask_down[0][0]      .item()) == bool:
                        mask_down       = mask_down      .to(x.dtype)
                    if mask_down2      is not None and not type(mask_down2[0][0]     .item()) == bool:
                        mask_down2      = mask_down2     .to(x.dtype)
                        
                    if mask_zero       is not None and not type(mask_zero[0][0]      .item()) == bool:
                        mask_zero       = mask_zero      .to(x.dtype)
                    if mask_up_zero    is not None and not type(mask_up_zero[0][0]   .item()) == bool:
                        mask_up_zero    = mask_up_zero   .to(x.dtype)
                    if mask_down_zero  is not None and not type(mask_down_zero[0][0] .item()) == bool:
                        mask_down_zero  = mask_down_zero .to(x.dtype)
                    if mask_down2_zero is not None and not type(mask_down2_zero[0][0].item()) == bool:
                        mask_down2_zero = mask_down2_zero.to(x.dtype)
                        
                    transformer_options['cross_mask']       = mask      [:,:txt_len]
                    transformer_options['self_mask']        = mask      [:,txt_len:]
                    transformer_options['cross_mask_up']    = mask_up   [:,:txt_len]
                    transformer_options['self_mask_up']     = mask_up   [:,txt_len:]
                    transformer_options['cross_mask_down']  = mask_down [:,:txt_len]
                    transformer_options['self_mask_down']   = mask_down [:,txt_len:]
                    transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                    transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                
                #h = x
                if y0_style_active and not RECON_MODE:
                    if mask is None:
                        context, y, _ = StyleMMDiT.apply_style_conditioning(
                            UNCOND       = UNCOND,
                            base_context = context,
                            base_y       = y,
                            base_llama3  = None,
                        )
                    else:
                        context = context.repeat(bsz_style + 1, 1, 1)
                        y = y.repeat(bsz_style + 1, 1)                   if y      is not None else None
                    h = torch.cat([h, y0_style_noised[cond_iter:cond_iter+1]], dim=0).to(h)



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
                    assert y.shape[0] == h.shape[0]
                    emb = emb + self.label_emb(y)

                #for id, module in enumerate(self.input_blocks):
                for id, (module, style_block) in enumerate(zip(self.input_blocks, StyleMMDiT.input_blocks)):
                    transformer_options["block"] = ("input", id)

                    if mask is not None:
                        transformer_options['cross_mask']       = mask      [:,:txt_len]
                        transformer_options['self_mask']        = mask      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                        transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                        
                    if   weight > 0 and mask is not None and     weight  <      id/total_layers:
                        transformer_options['cross_mask'] = None
                        transformer_options['self_mask']  = None
                    
                    elif weight < 0 and mask is not None and abs(weight) < (1 - id/total_layers):
                        transformer_options['cross_mask'] = None
                        transformer_options['self_mask']  = None
                        
                    elif floor > 0 and mask is not None and       floor  >      id/total_layers:
                        transformer_options['cross_mask']       = mask_zero      [:,:txt_len]
                        transformer_options['self_mask']        = mask_zero      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up_zero   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up_zero   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down_zero [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down_zero [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                        transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                    
                    elif floor < 0 and mask is not None and   abs(floor) > (1 - id/total_layers):
                        transformer_options['cross_mask']       = mask_zero      [:,:txt_len]
                        transformer_options['self_mask']        = mask_zero      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up_zero   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up_zero   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down_zero [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down_zero [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                        transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None

                    h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator, style_block=style_block)
                    if id == 0:
                        h = StyleMMDiT(h, "proj_in")
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
                    style_block = StyleMMDiT.middle_blocks[0]

                    if mask is not None:
                        transformer_options['cross_mask']       = mask      [:,:txt_len]
                        transformer_options['self_mask']        = mask      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                        transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                        
                    if   weight > 0 and mask is not None and     weight  <      (len(self.input_blocks) + 1)/total_layers:
                        transformer_options['cross_mask'] = None
                        transformer_options['self_mask']  = None
                    
                    elif weight < 0 and mask is not None and abs(weight) < (1 - (len(self.input_blocks) + 1)/total_layers):
                        transformer_options['cross_mask'] = None
                        transformer_options['self_mask']  = None
                        
                    elif floor > 0 and mask is not None and       floor  >      (len(self.input_blocks) + 1)/total_layers:
                        transformer_options['cross_mask']       = mask_zero      [:,:txt_len]
                        transformer_options['self_mask']        = mask_zero      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up_zero   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up_zero   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down_zero [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down_zero [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                        transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                    
                    elif floor < 0 and mask is not None and   abs(floor) > (1 - (len(self.input_blocks) + 1)/total_layers):
                        transformer_options['cross_mask']       = mask_zero      [:,:txt_len]
                        transformer_options['self_mask']        = mask_zero      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up_zero   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up_zero   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down_zero [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down_zero [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                        transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None

                    h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator, style_block=style_block)
                
                h = apply_control(h, control, 'middle')

                #for id, module in enumerate(self.output_blocks):
                for id, (module, style_block) in enumerate(zip(self.output_blocks, StyleMMDiT.output_blocks)):
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
                        transformer_options['cross_mask']       = mask      [:,:txt_len]
                        transformer_options['self_mask']        = mask      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2[:,:txt_len] if mask_down2 is not None else None
                        transformer_options['self_mask_down2']  = mask_down2[:,txt_len:] if mask_down2 is not None else None
                        
                    if   weight > 0 and mask is not None and     weight  <      (len(self.input_blocks) + 1 + id)/total_layers:
                        transformer_options['cross_mask'] = None
                        transformer_options['self_mask']  = None
                    
                    elif weight < 0 and mask is not None and abs(weight) < (1 - (len(self.input_blocks) + 1 + id)/total_layers):
                        transformer_options['cross_mask'] = None
                        transformer_options['self_mask']  = None
                        
                    elif floor > 0 and mask is not None and       floor  >      (len(self.input_blocks) + 1 + id)/total_layers:
                        transformer_options['cross_mask']       = mask_zero      [:,:txt_len]
                        transformer_options['self_mask']        = mask_zero      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up_zero   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up_zero   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down_zero [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down_zero [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                        transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None
                    
                    elif floor < 0 and mask is not None and   abs(floor) > (1 - (len(self.input_blocks) + 1 + id)/total_layers):
                        transformer_options['cross_mask']       = mask_zero      [:,:txt_len]
                        transformer_options['self_mask']        = mask_zero      [:,txt_len:]
                        transformer_options['cross_mask_up']    = mask_up_zero   [:,:txt_len]
                        transformer_options['self_mask_up']     = mask_up_zero   [:,txt_len:]
                        transformer_options['cross_mask_down']  = mask_down_zero [:,:txt_len]
                        transformer_options['self_mask_down']   = mask_down_zero [:,txt_len:]
                        transformer_options['cross_mask_down2'] = mask_down2_zero[:,:txt_len] if mask_down2_zero is not None else None
                        transformer_options['self_mask_down2']  = mask_down2_zero[:,txt_len:] if mask_down2_zero is not None else None

                    h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator, style_block=style_block)
                
                h = h.type(x.dtype)
                
                if self.predict_codebook_ids:
                    eps = self.id_predictor(h)
                else:
                    eps = self.out(h)
                    eps = StyleMMDiT(eps, "proj_out")
                    
                out_list.append(eps[0:1])
                
            eps = torch.stack(out_list, dim=0).squeeze(dim=1)

            if recon_iter == 1:
                denoised = new_x * ((ISIGMA ** 2 + 1) ** 0.5)  - ISIGMA.to(new_x) * eps.to(new_x)
                if x_tmp is not None:
                    eps = (x_tmp * ((SIGMA ** 2 + 1) ** 0.5) - denoised.to(x_tmp)) / SIGMA.to(x_tmp)
                else:
                    eps = (x_orig * ((SIGMA ** 2 + 1) ** 0.5) - denoised.to(x_orig)) / SIGMA.to(x_orig)
            





























        y0_style_pos        = transformer_options.get("y0_style_pos")
        y0_style_neg        = transformer_options.get("y0_style_neg")

        y0_style_pos_weight    = transformer_options.get("y0_style_pos_weight", 0.0)
        y0_style_pos_synweight = transformer_options.get("y0_style_pos_synweight", 0.0)
        y0_style_pos_synweight *= y0_style_pos_weight

        y0_style_neg_weight    = transformer_options.get("y0_style_neg_weight", 0.0)
        y0_style_neg_synweight = transformer_options.get("y0_style_neg_synweight", 0.0)
        y0_style_neg_synweight *= y0_style_neg_weight
        

        freqsep_lowpass_method = transformer_options.get("freqsep_lowpass_method")
        freqsep_sigma          = transformer_options.get("freqsep_sigma")
        freqsep_kernel_size    = transformer_options.get("freqsep_kernel_size")
        freqsep_inner_kernel_size    = transformer_options.get("freqsep_inner_kernel_size")
        freqsep_stride    = transformer_options.get("freqsep_stride")
        
        freqsep_lowpass_weight = transformer_options.get("freqsep_lowpass_weight")
        freqsep_highpass_weight= transformer_options.get("freqsep_highpass_weight")
        freqsep_mask           = transformer_options.get("freqsep_mask")
        


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
    
    

