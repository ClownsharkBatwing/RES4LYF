import torch
import torch.nn.functional as F

from typing import Optional, Callable, Tuple, Dict, Any, Union

import numpy as np
import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import json
import os 
import random
import copy

from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # use the Agg backend for non-interactive rendering... prevent crashes by not using tkinter (which requires running in the main thread)
from comfy.cli_args import args

import comfy.samplers
import comfy.utils

from nodes import MAX_RESOLUTION



from .beta.rk_method_beta        import RK_Method_Beta
from .beta.rk_noise_sampler_beta import RK_NoiseSampler, NOISE_MODE_NAMES
from .helper                     import get_res4lyf_scheduler_list
from .sigmas                     import get_sigmas
from .images                     import image_resize
from .res4lyf                    import RESplain



class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",  {                      "tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self,
                    images,
                    filename_prefix = "ComfyUI",
                    prompt          = None,
                    extra_pnginfo   = None
                    ):
        
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }



# adapted from https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler
class SigmasPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "sigmas":         ("SIGMAS",),
                "print_as_list" : ("BOOLEAN", {"default": False}),
                "line_color":     ("STRING", {"default": "blue"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sigmas_preview"
    OUTPUT_NODE = True
    CATEGORY = 'RES4LYF/sigmas'

    @staticmethod
    def tensor_to_graph_image(tensor, color='blue'):
        
        plt.figure()
        plt.plot(tensor.numpy(), marker='o', linestyle='-', color=color)
        plt.title("Graph from Tensor")
        plt.xlabel("Step Number")
        plt.ylabel("Sigma Value")
        
        with BytesIO() as buf:
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).copy()
            
        plt.close()
        return image

    def sigmas_preview(self, sigmas, print_as_list, line_color):
        
        if print_as_list:
            # Convert to list with 4 decimal places
            sigmas_list = [round(float(s), 4) for s in sigmas.tolist()]
            
            # Print header using RESplain
            RESplain("\n" + "="*60)
            RESplain("SIGMAS PREVIEW - PRINT LIST")
            RESplain("="*60)
            
            # Print basic stats
            RESplain(f"Total steps: {len(sigmas_list)}")
            RESplain(f"Min sigma:   {min(sigmas_list):.4f}")
            RESplain(f"Max sigma:   {max(sigmas_list):.4f}")
          
            # Print the clean sigma values
            RESplain(f"\nSigma values ({len(sigmas_list)} steps):")
            RESplain("-" * 40)
            
            # Print in rows of 5 for readability
            for i in range(0, len(sigmas_list), 5):
                row = sigmas_list[i:i+5]
                row_str = "  ".join(f"{val:8.4f}" for val in row)
                RESplain(f"Step {i:2d}-{min(i+4, len(sigmas_list)-1):2d}: {row_str}")
            
            # Calculate and print percentages (normalized 0-1)
            sigmas_percentages = ((sigmas-sigmas.min())/(sigmas.max()-sigmas.min())).tolist()
            sigmas_percentages = [round(p, 4) for p in sigmas_percentages]
            
            RESplain(f"\nNormalized percentages (0.0-1.0):")
            RESplain("-" * 40)
            
            # Print step-by-step breakdown
            RESplain("Step | Sigma    | Normalized | Step Size")
            RESplain("-----|----------|------------|----------")
            for i, (sigma, pct) in enumerate(zip(sigmas_list, sigmas_percentages)):
                if i > 0:
                    step_size = sigmas_list[i-1] - sigma
                    RESplain(f"{i:4d} | {sigma:8.4f} | {pct:10.4f} | {step_size:8.4f}")
                else:
                    RESplain(f"{i:4d} | {sigma:8.4f} | {pct:10.4f} | {'--':>8}")
            
            RESplain("="*60 + "\n")
            
        sigmas_graph = self.tensor_to_graph_image(sigmas.cpu(), line_color)
        numpy_image = np.array(sigmas_graph)
        numpy_image = numpy_image / 255.0
        tensor_image = torch.from_numpy(numpy_image)
        tensor_image = tensor_image.unsqueeze(0)
        images_tensor = torch.cat([tensor_image], 0)
        output = self.save_images(images_tensor, "SigmasPreview")
        output["result"] = (images_tensor,)

        return output




class VAEEncodeAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resize_to_input": (["false", "image_1", "image_2", "mask", "latent"], {"default": "false"},),
                "width":           ("INT",                                             {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height":          ("INT",                                             {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "mask_channel":    (["red", "green", "blue", "alpha"],),
                "invert_mask":     ("BOOLEAN",                                         {"default": False}),
                "latent_type":     (["4_channels", "16_channels"],                     {"default": "16_channels",}),
            },
            
            "optional": {
                "image_1":         ("IMAGE",),
                "image_2":         ("IMAGE",),
                "mask":            ("IMAGE",),
                "latent":          ("LATENT",),
                "vae":             ("VAE", ),
            }
        }

    RETURN_TYPES = ("LATENT",
                    "LATENT",
                    "MASK",
                    "LATENT",
                    "INT",
                    "INT",
                    )
                    
    RETURN_NAMES = ("latent_1",
                    "latent_2",
                    "mask",
                    "empty_latent",
                    "width",
                    "height",
                    )
    
    FUNCTION = "main"
    CATEGORY = "RES4LYF/vae"

    def main(self,
            width,
            height,
            resize_to_input = "false",
            image_1         = None,
            image_2         = None,
            mask            = None,
            invert_mask     = False,
            method          = "stretch",
            interpolation   = "lanczos",
            condition       = "always",
            multiple_of     = 0,
            keep_proportion = False,
            mask_channel    = "red",
            latent          = None, 
            latent_type     = "16_channels", 
            vae             = None
            ):
        
        ratio = 8 # latent compression factor
        
        # this is unfortunately required to avoid apparent non-deterministic outputs. 
        # without setting the seed each time, the outputs of the VAE encode will change with every generation.
        torch     .manual_seed    (42)          
        torch.cuda.manual_seed_all(42)

        image_1 = image_1.clone() if image_1 is not None else None
        image_2 = image_2.clone() if image_2 is not None else None

        if latent is not None and resize_to_input == "latent":
            height, width = latent['samples'].shape[-2:]

            #height, width = latent['samples'].shape[2:4]
            height, width = height * ratio, width * ratio
            
        elif image_1 is not None and resize_to_input == "image_1":
            height, width = image_1.shape[1:3]
            
        elif image_2 is not None and resize_to_input == "image_2":
            height, width = image_2.shape[1:3]       
            
        elif mask is not None and resize_to_input == "mask":
            height, width =    mask.shape[1:3]   
            
        if latent is not None:
            c = latent['samples'].shape[1]
        else:
            if latent_type == "4_channels":
                c = 4
            else:
                c = 16
            if   image_1 is not None:
                b = image_1.shape[0]
            elif image_2 is not None:
                b = image_2.shape[0]
            else:
                b = 1
                
            latent = {"samples": torch.zeros((b, c, height // ratio, width // ratio))}
        
        latent_1, latent_2 = None, None
        if image_1 is not None:
            image_1  = image_resize(image_1, width, height, method, interpolation, condition, multiple_of, keep_proportion)
            latent_1 = {"samples": vae.encode(image_1[:,:,:,:3])}
        if image_2 is not None:
            image_2  = image_resize(image_2, width, height, method, interpolation, condition, multiple_of, keep_proportion)
            latent_2 = {"samples": vae.encode(image_2[:,:,:,:3])}
        
        if mask is not None and mask.shape[-1] > 1:
            channels = ["red", "green", "blue", "alpha"]
            mask = mask[:, :, :, channels.index(mask_channel)]
            
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
            if invert_mask:
                mask = 1.0 - mask

        return (latent_1, 
                latent_2, 
                mask, 
                latent,
                width, 
                height,
                )




class VAEStyleTransferLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method":    (["AdaIN", "WCT"], {"default": "AdaIN"}),
                "latent":    ("LATENT",),
                "style_ref": ("LATENT",),
                "vae":       ("VAE", ),
            },
            
            "optional": {
            }
        }

    RETURN_TYPES = ("LATENT",)
                    
    RETURN_NAMES = ("latent",)
    
    FUNCTION = "main"
    CATEGORY = "RES4LYF/vae"

    def main(self,
            method    = None,
            latent    = None,
            style_ref = None,
            vae       = False,
            ):
        
        from comfy.ldm.cascade.stage_c_coder import StageC_coder
        
        # this is unfortunately required to avoid apparent non-deterministic outputs. 
        # without setting the seed each time, the outputs of the VAE encode will change with every generation.
        torch     .manual_seed    (42)          
        torch.cuda.manual_seed_all(42)
        
        denoised = latent   .get('state_info', {}).get('raw_x')
        y0       = style_ref.get('state_info', {}).get('raw_x')
        
        denoised = latent['samples'] if denoised is None else denoised
        y0       = style_ref['samples'] if y0 is None else y0
            
        #denoised = latent.get('state_info', latent['samples'].get('raw_x', latent['samples']))
        #y0       = style_ref.get('state_info', style_ref['samples'].get('raw_x', style_ref['samples']))
        
        if denoised.ndim > 4:
            denoised = denoised.squeeze(0)
        if y0.ndim > 4:
            y0 = y0.squeeze(0)
        
        if   hasattr(vae.first_stage_model, "up_blocks"): # probably stable cascade stage A
            x_embedder = copy.deepcopy(vae.first_stage_model.up_blocks[0][0]).to(torch.float64)
            denoised_embed = x_embedder(denoised.to(x_embedder.weight))
            y0_embed       = x_embedder(y0.to(x_embedder.weight))
            
            denoised_embed = apply_style_to_latent(denoised_embed, y0_embed, method)
            
            denoised_styled = invert_conv2d(x_embedder, denoised_embed, denoised.shape).to(denoised)
            
            
        elif hasattr(vae.first_stage_model, "decoder"):   # probably sd15, sdxl, sd35, flux, wan, etc. vae
            x_embedder = copy.deepcopy(vae.first_stage_model.decoder.conv_in).to(torch.float64)
            denoised_embed = x_embedder(denoised.to(x_embedder.weight))
            y0_embed       = x_embedder(y0.to(x_embedder.weight))
            
            denoised_embed = apply_style_to_latent(denoised_embed, y0_embed, method)
            
            denoised_styled = invert_conv2d(x_embedder, denoised_embed, denoised.shape).to(denoised)
        
        elif type(vae.first_stage_model) == StageC_coder:
            x_embedder = copy.deepcopy(vae.first_stage_model.encoder.mapper[0]).to(torch.float64)
            #x_embedder = copy.deepcopy(vae.first_stage_model.previewer.blocks[0]).to(torch.float64) # use with strategy for decoder above, but exploding latent problem, 1.E30 etc. quick to nan

            denoised_embed = invert_conv2d(x_embedder, denoised, denoised.shape)
            y0_embed       = invert_conv2d(x_embedder, y0, y0.shape)
            
            denoised_embed = apply_style_to_latent(denoised_embed, y0_embed, method)
            
            denoised_styled = x_embedder(denoised_embed.to(x_embedder.weight))
            
            
            
        
        latent_out = latent.copy() 
        #latent_out['state_info'] = copy.deepcopy(latent['state_info'])

        if latent_out.get('state_info', {}).get('raw_x') is not None:
            latent_out['state_info']['raw_x'] = denoised_styled
        latent_out['samples'] = denoised_styled
        
        return (latent_out, )







def apply_style_to_latent(denoised_embed, y0_embed, method="WCT"):
    from einops import rearrange
    import torch.nn as nn
    
    denoised_embed_shape = denoised_embed.shape

    denoised_embed = rearrange(denoised_embed, "B C H W -> B (H W) C")
    y0_embed       = rearrange(y0_embed,       "B C H W -> B (H W) C")
    
    if method == "AdaIN":
        denoised_embed = adain_seq_inplace(denoised_embed, y0_embed)
    
    elif method == "WCT":
        f_s  = y0_embed[0].clone()           # batched style guides not supported
        mu_s = f_s.mean(dim=0, keepdim=True)
        f_s_centered = f_s - mu_s
        
        cov = (f_s_centered.transpose(-2,-1).double() @ f_s_centered.double()) / (f_s_centered.size(0) - 1)

        S_eig, U_eig = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
        S_eig_sqrt    = S_eig.clamp(min=0).sqrt() # eigenvalues -> singular values
        
        whiten = U_eig @ torch.diag(S_eig_sqrt) @ U_eig.transpose(-2,-1)
        y0_color  = whiten.to(f_s_centered)

        for wct_i in range(denoised_embed_shape[0]):
            f_c          = denoised_embed[wct_i].clone()
            mu_c         = f_c.mean(dim=0, keepdim=True)
            f_c_centered = f_c - mu_c
            
            cov = (f_c_centered.transpose(-2,-1).double() @ f_c_centered.double()) / (f_c_centered.size(0) - 1)

            S_eig, U_eig  = torch.linalg.eigh(cov + 1e-5 * torch.eye(cov.size(0), dtype=cov.dtype, device=cov.device))
            inv_sqrt_eig  = S_eig.clamp(min=0).rsqrt() 
            
            whiten = U_eig @ torch.diag(inv_sqrt_eig) @ U_eig.transpose(-2,-1)
            whiten = whiten.to(f_c_centered)

            f_c_whitened = f_c_centered @ whiten.transpose(-2,-1)
            f_cs         = f_c_whitened @ y0_color.transpose(-2,-1).to(f_c_whitened) + mu_s.to(f_c_whitened)
            
            denoised_embed[wct_i] = f_cs
    
    denoised_embed = rearrange(denoised_embed, "B (H W) C -> B C H W", W=denoised_embed_shape[-1])
    
    return denoised_embed



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


"""def invert_conv3d(conv: torch.nn.Conv3d,
                z: torch.Tensor, original_shape: torch.Size, grid_sizes: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:

    import torch.nn.functional as F
    B, C_in, D, H, W = original_shape
    pD, pH, pW = 1,2,2
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
    if conv.bias is not None:
        b = conv.bias.view(1, C_out, 1, 1, 1)
        z_nobias = z - b
    else:
        z_nobias = z

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
"""



def adain_seq_inplace(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    mean_c = content.mean(1, keepdim=True)
    std_c  = content.std (1, keepdim=True).add_(eps)  # in-place add
    mean_s = style.mean  (1, keepdim=True)
    std_s  = style.std   (1, keepdim=True).add_(eps)

    content.sub_(mean_c).div_(std_c).mul_(std_s).add_(mean_s)  # in-place chain
    return content
































class LatentUpscaleWithVAE:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent":   ("LATENT", ),      
                "width" : ("INT", {"default": 1024, "min": 8, "max": 1024 ** 2, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 1024 ** 2, "step": 8}),
                "vae": ("VAE", ),
                },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/latents"

    def main(self,
            latent,
            width,
            height,
            vae,
            method          = "stretch",
            interpolation   = "lanczos",
            condition       = "always",
            multiple_of     = 0,
            keep_proportion = False,
            ):
        
        ratio = 8 # latent compression factor
        
        # this is unfortunately required to avoid apparent non-deterministic outputs. 
        # without setting the seed each time, the outputs of the VAE encode will change with every generation.
        torch     .manual_seed    (42)          
        torch.cuda.manual_seed_all(42)
        
        images_prev_list, latent_prev_list = [], []
        
        if 'state_info' in latent:
            #images      = vae.decode(latent['state_info']['raw_x']  ) # .to(latent['samples']) )
            images      = vae.decode(latent['state_info']['denoised']  ) # .to(latent['samples']) )
            
            data_prev_ = latent['state_info']['data_prev_'].squeeze(0)
            for i in range(data_prev_.shape[0]):
                images_prev_list.append(   vae.decode(data_prev_[i])  ) # .to(latent['samples'])  )
        else:
            images = vae.decode(latent['samples'])
            
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            
        images = image_resize(images, width, height, method, interpolation, condition, multiple_of, keep_proportion)
        latent_tensor = vae.encode(images[:,:,:,:3])
        
        if images_prev_list:
            for i in range(data_prev_.shape[0]):
                image_data_p = image_resize(images_prev_list[i], width, height, method, interpolation, condition, multiple_of, keep_proportion)
                latent_prev_list.append(   vae.encode(image_data_p[:,:,:,:3])   )
            latent_prev = torch.stack(latent_prev_list).unsqueeze(0)     #.view_as(latent['state_info']['data_prev_'])
            #images_prev = image_resize(images_prev, width, height, method, interpolation, condition, multiple_of, keep_proportion)
            #latent_tensor = vae.encode(image_1[:,:,:,:3])
        
        if 'state_info' in latent:
            #latent['state_info']['raw_x']      = latent_tensor
            latent['state_info']['denoised']   = latent_tensor
            latent['state_info']['data_prev_'] = latent_prev
            
        latent['samples'] = latent_tensor.to(latent['samples'])

        return (latent,)



class SigmasSchedulePreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":       ("MODEL",),
                "noise_mode":  (NOISE_MODE_NAMES,             {"default": 'hard',                                        "tooltip": "How noise scales with the sigma schedule. Hard is the most aggressive, the others start strong and drop rapidly."}),
                "eta":         ("FLOAT",                      {"default": 0.25, "step": 0.01, "min": -1000.0, "max": 1000.0}),
                "s_noise":     ("FLOAT",                      {"default": 1.00, "step": 0.01, "min": -1000.0, "max": 1000.0}),
                "denoise":     ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                "denoise_alt": ("FLOAT",                      {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
                "scheduler":   (get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "steps":       ("INT",                        {"default": 30, "min": 1, "max": 10000}),
                "plot_max":    ("FLOAT",                      {"default": 2.1, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Set to a negative value to have the plot scale automatically."}),
                "plot_min":    ("FLOAT",                      {"default": 0.0, "min": -10000, "max": 10000, "step":0.01, "tooltip": "Set to a negative value to have the plot scale automatically."}),
            },
            "optional": {
                "sigmas":      ("SIGMAS",),
            },
        }

    FUNCTION = "plot_schedule"
    CATEGORY = "RES4LYF/sigmas"
    OUTPUT_NODE = True


    @staticmethod
    def tensor_to_graph_image(tensors, labels, colors, plot_min, plot_max, input_params):
        plt.figure(figsize=(6.4, 6.4), dpi=320) 
        ax = plt.gca()
        ax.set_facecolor("black") 
        ax.patch.set_alpha(1.0)  

        for _ in range(50):
            for tensor, color in zip(tensors, colors):
                plt.plot(tensor.numpy(), color=color, alpha=0.1)

        plt.axhline(y=1.0, color='gray', linestyle='dotted', linewidth=1.5)

        plt.xlabel("Step", color="white", weight="bold", antialiased=False)
        plt.ylabel("Value", color="white", weight="bold", antialiased=False)
        ax.tick_params(colors="white") 

        if plot_max > 0:
            plt.ylim(plot_min, plot_max)

        input_text = (
            f"noise_mode:  {input_params['noise_mode']}  |  "
            f"eta:         {input_params['eta']}  |  "
            f"s_noise:     {input_params['s_noise']}  |  "
            f"denoise:     {input_params['denoise']}  |  "
            f"denoise_alt: {input_params['denoise_alt']}  |  "
            f"scheduler:   {input_params['scheduler']}"
        )
        plt.text(0.5, 1.05, input_text, ha='center', va='center', color='white', fontsize=8, transform=ax.transAxes)

        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], color=color, lw=2, label=label) for label, color in zip(labels, colors)]
        plt.legend(handles=legend_handles, facecolor="black", edgecolor="white", labelcolor="white", framealpha=1.0)

        with BytesIO() as buf:
            plt.savefig(buf, format='png', facecolor="black")
            buf.seek(0)
            image = Image.open(buf).copy()
        plt.close()
        return image


    def plot_schedule(self, model, noise_mode, eta, s_noise, denoise, denoise_alt, scheduler, steps, plot_min, plot_max, sigmas=None):
        sigma_vals               = []
        sigma_next_vals          = []
        sigma_down_vals          = []
        sigma_up_vals            = []
        sigma_plus_up_vals       = []
        sigma_hat_vals           = []
        alpha_ratio_vals         = []
        sigma_step_size_vals     = []
        sigma_step_size_sde_vals = []
        
        eta_var = eta
        
        rk_type = "res_2s"
        noise_anchor = 1.0

        if sigmas is not None:
            sigmas = sigmas.clone()
        else: 
            sigmas = get_sigmas(model, scheduler, steps, denoise)
        sigmas *= denoise_alt

        RK = RK_Method_Beta.create(model, rk_type, noise_anchor, model_device=sigmas.device, work_device=sigmas.device, dtype=sigmas.dtype, extra_options="")
        NS = RK_NoiseSampler(RK, model, device=sigmas.device, dtype=sigmas.dtype, extra_options="")

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            su, sigma_hat, sd, alpha_ratio = NS.get_sde_step(sigma, sigma_next, eta, noise_mode_override=noise_mode, )
            #su, sigma_hat, sd, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, noise_mode)

            su = su * s_noise
            
            sigma_vals              .append(sigma)
            sigma_next_vals         .append(sigma_next)
            sigma_down_vals         .append(sd)
            sigma_up_vals           .append(su)
            sigma_plus_up_vals      .append(sigma + su)
            alpha_ratio_vals        .append(alpha_ratio)
            sigma_step_size_vals    .append(sigma - sigma_next)
            sigma_step_size_sde_vals.append(sigma + su - sd)

            if sigma_hat != sigma:
                sigma_hat_vals.append(sigma_hat)

        sigma_tensor               = torch.tensor(sigma_vals)
        sigma_next_tensor          = torch.tensor(sigma_next_vals)
        sigma_down_tensor          = torch.tensor(sigma_down_vals)
        sigma_up_tensor            = torch.tensor(sigma_up_vals)
        sigma_plus_up_tensor       = torch.tensor(sigma_plus_up_vals)
        alpha_ratio_tensor         = torch.tensor(alpha_ratio_vals)
        sigma_step_size_tensor     = torch.tensor(sigma_step_size_vals)
        sigma_step_size_sde_tensor = torch.tensor(sigma_step_size_sde_vals)

        tensors = [sigma_tensor, sigma_next_tensor, sigma_down_tensor, sigma_up_tensor]
        labels = ["$σ$", "$σ_{next}$", "$σ_{down}$", "$σ_{up}$"]
        colors = ["white", "dodgerblue", "green", "red"]
        
        if torch.norm(sigma_next_tensor - sigma_down_tensor) < 1e-2:
            tensors = [sigma_tensor, sigma_next_tensor, sigma_up_tensor]
            labels = ["$σ$", "$σ_{next,down}$", "$σ_{up}$"]
            colors = ["white", "cyan", "red"]
            
        elif torch.norm(sigma_next_tensor - sigma_up_tensor) < 1e-2:
            tensors = [sigma_tensor, sigma_next_tensor, sigma_down_tensor]
            labels = ["$σ$", "$σ_{next,up}$", "$σ_{down}$"]
            colors = ["white", "violet", "green",]
        
        if torch.norm(sigma_tensor - sigma_plus_up_tensor) > 1e-2:
            tensors.append(sigma_plus_up_tensor)
            labels.append("$σ + σ_{up}$")
            colors.append("brown")
        
        if torch.norm(sigma_step_size_tensor - sigma_step_size_sde_tensor) > 1e-2:
            tensors.append(sigma_step_size_sde_tensor)
            labels.append("$Δ \hat{t}$")
            colors.append("gold")
            
        if sigma_hat_vals:
            sigma_hat_tensor = torch.tensor(sigma_hat_vals)
            tensors.append(sigma_hat_tensor)
            labels.append("$σ̂$")
            colors.append("maroon")
            
            tensors.append(sigma_step_size_tensor)
            labels.append("$σ̂ - σ_{next}$")
            colors.append("darkorange")
        else:
            tensors.append(sigma_step_size_tensor)
            #labels.append("$σ - σ_{next}$")
            labels.append("$Δt$")
            colors.append("darkorange")
        
        tensors.append(alpha_ratio_tensor)
        labels.append("$α_{ratio}$")
        colors.append("grey")
        
        
        graph_image = self.tensor_to_graph_image(
            tensors, labels, colors, plot_min, plot_max,
            input_params={
                "noise_mode": noise_mode,
                "eta": eta,
                "s_noise": s_noise,
                "denoise": denoise,
                "denoise_alt": denoise_alt,
                "scheduler": scheduler,
            }
        )

        numpy_image   = np.array(graph_image)
        numpy_image   = numpy_image / 255.0
        tensor_image  = torch.from_numpy(numpy_image)
        tensor_image  = tensor_image.unsqueeze(0)
        images_tensor = torch.cat([tensor_image], 0)

        return self.save_images(images_tensor, "SigmasSchedulePreview")
    
    