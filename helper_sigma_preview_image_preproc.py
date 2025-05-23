import torch
import torch.nn.functional as F

import numpy as np
import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import json
import os 
import random

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
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sigmas_preview"
    OUTPUT_NODE = True
    CATEGORY = 'RES4LYF/sigmas'

    @staticmethod
    def tensor_to_graph_image(tensor):
        
        plt.figure()
        plt.plot(tensor.numpy(), marker='o', linestyle='-', color='blue')
        plt.title("Graph from Tensor")
        plt.xlabel("Step Number")
        plt.ylabel("Sigma Value")
        
        with BytesIO() as buf:
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).copy()
            
        plt.close()
        return image

    def sigmas_preview(self, sigmas, print_as_list):
        
        if print_as_list:
            print(sigmas.tolist())
            
            sigmas_percentages = ((sigmas-sigmas.min())/(sigmas.max()-sigmas.min())).tolist()
            sigmas_percentages_w_steps = [(i,round(s,4)) for i,s in enumerate(sigmas_percentages)]
            
            print(sigmas_percentages_w_steps)
            
        sigmas_graph = self.tensor_to_graph_image(sigmas.cpu())
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
    
    