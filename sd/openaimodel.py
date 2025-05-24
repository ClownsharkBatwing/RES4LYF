from abc import abstractmethod
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging
import copy

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
            y0_style_pos = y0_style_pos.to(torch.float64)
            x   = x_orig.clone().to(torch.float64) * ((SIGMA ** 2 + 1) ** 0.5)
            eps = eps.to(torch.float64)
            eps_orig = eps.clone()
            
            sigma = SIGMA
            denoised = x - sigma * eps

            x_embedder = copy.deepcopy(self.input_blocks[0][0]).to(denoised)
            
            denoised_embed = x_embedder(denoised)
            y0_adain_embed = x_embedder(y0_style_pos)

            denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
            y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

            if transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                """for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    denoised_embed = F.linear(denoised_embed.to(W), W, b).to(img)
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)"""
                    
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
            y0_style_neg = y0_style_neg.to(torch.float64)
            x   = x_orig.clone().to(torch.float64)* ((SIGMA ** 2 + 1) ** 0.5)
            eps = eps.to(torch.float64)
            eps_orig = eps.clone()
            
            sigma = SIGMA #t_orig[0].to(torch.float32) / 1000
            denoised = x - sigma * eps

            x_embedder = copy.deepcopy(self.input_blocks[0][0]).to(denoised)
            
            denoised_embed = x_embedder(denoised)
            y0_adain_embed = x_embedder(y0_style_neg)

            denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
            y0_adain_embed = rearrange(y0_adain_embed, "B C H W -> B (H W) C")

            if transformer_options['y0_style_method'] == "AdaIN":
                denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                """for adain_iter in range(EO("style_iter", 0)):
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)
                    denoised_embed = (denoised_embed - b) @ torch.linalg.pinv(W.to(pinv_dtype)).T.to(dtype)
                    denoised_embed = F.linear(denoised_embed.to(W), W, b).to(img)
                    denoised_embed = adain_seq_inplace(denoised_embed, y0_adain_embed)"""
                    
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
    
    
    