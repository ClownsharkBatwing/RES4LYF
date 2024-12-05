import torch
import torch.nn.functional as F
import numpy as np
import folder_paths
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import json
import os 
import random
import matplotlib.pyplot as plt
from io import BytesIO
from comfy.cli_args import args
import comfy.utils
from nodes import MAX_RESOLUTION


class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
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

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
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
                "sigmas": ("SIGMAS",),
                "print_as_list" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "sigmas_preview"
    CATEGORY = 'sampling/custom_sampling/sigmas'
    OUTPUT_NODE = True

    @staticmethod
    def tensor_to_graph_image(tensor):
        plt.figure()
        plt.plot(tensor.numpy(), marker='o', linestyle='-', color='blue')
        plt.title("Graph from Tensor")
        plt.xlabel("Index")
        plt.ylabel("Value")
        with BytesIO() as buf:
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).copy()
        plt.close()
        return image

    def sigmas_preview(self, sigmas, print_as_list):
        # adapted from https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler
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
        
        return self.save_images(images_tensor, "SigmasPreview")






# adapted from https://github.com/cubiq/ComfyUI_essentials
def image_resize(image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
    _, oh, ow, _ = image.shape
    x = y = x2 = y2 = 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    if keep_proportion:
        method = "keep proportion"

    if multiple_of > 1:
        width = width - (width % multiple_of)
        height = height - (height % multiple_of)

    if method == 'keep proportion' or method == 'pad':
        if width == 0 and oh < height:
            width = MAX_RESOLUTION
        elif width == 0 and oh >= height:
            width = ow

        if height == 0 and ow < width:
            height = MAX_RESOLUTION
        elif height == 0 and ow >= width:
            height = oh

        ratio = min(width / ow, height / oh)
        new_width = round(ow*ratio)
        new_height = round(oh*ratio)

        if method == 'pad':
            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top

        width = new_width
        height = new_height
    elif method.startswith('fill'):
        width = width if width > 0 else ow
        height = height if height > 0 else oh

        ratio = max(width / ow, height / oh)
        new_width = round(ow*ratio)
        new_height = round(oh*ratio)
        x = (new_width - width) // 2
        y = (new_height - height) // 2
        x2 = x + width
        y2 = y + height
        if x2 > new_width:
            x -= (x2 - new_width)
        if x < 0:
            x = 0
        if y2 > new_height:
            y -= (y2 - new_height)
        if y < 0:
            y = 0
        width = new_width
        height = new_height
    else:
        width = width if width > 0 else ow
        height = height if height > 0 else oh

    if "always" in condition \
        or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
        or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

        outputs = image.permute(0,3,1,2)

        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if method == 'pad':
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0,2,3,1)

        if method.startswith('fill'):
            if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                outputs = outputs[:, y:y2, x:x2, :]
    else:
        outputs = image

    if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
        width = outputs.shape[2]
        height = outputs.shape[1]
        x = (width % multiple_of) // 2
        y = (height % multiple_of) // 2
        x2 = width - ((width % multiple_of) - x)
        y2 = height - ((height % multiple_of) - y)
        outputs = outputs[:, y:y2, x:x2, :]
    
    outputs = torch.clamp(outputs, 0, 1)

    return outputs


class PrepForUnsampling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resize_to_input": (["false", "image_1", "image_2", "mask", "latent"], {"default": "false"},),
                "width": ("INT", { "default": 1344, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 768, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "mask_channel": (["red", "green", "blue", "alpha"],),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "latent_type": (["4_channels", "16_channels"], {"default": "16_channels",}),
            },
            
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "mask": ("IMAGE",),
                "latent": ("LATENT",),
                "vae": ("VAE", ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "MASK", "LATENT", "INT", "INT",)
    RETURN_NAMES = ("latent_1", "latent_2", "mask", "empty_latent", "width", "height",)
    FUNCTION = "main"
    CATEGORY = "essentials/image manipulation"

    def main(self, width, height, resize_to_input="false", image_1=None, image_2=None, mask=None, invert_mask=False, method="stretch", interpolation="lanczos", condition="always", multiple_of=0, keep_proportion=False, mask_channel="red", latent=None, latent_type="16_channels", vae=None):
        #NOTE: VAE encode with comyfui is *non-deterministic* in that each success encode will return slightly different latent images! The difference is visible after decoding.
        ratio = 8 # latent compression factor

        if latent is not None and resize_to_input == "latent":
            height, width = latent['samples'].shape[2:4]
            height, width = height * ratio, width * ratio
        elif image_1 is not None and resize_to_input == "image_1":
            height, width = image_1.shape[1:3]
        elif image_2 is not None and resize_to_input == "image_2":
            height, width = image_2.shape[1:3]       
        elif mask is not None and resize_to_input == "mask":
            height, width = mask.shape[1:3]   
            
        if latent is not None:
            c = latent['samples'].shape[1]
        else:
            if latent_type == "4_channels":
                c = 4
            else:
                c = 16
            latent = {"samples": torch.zeros((1, c, height // ratio, width // ratio))}
        
        latent_1, latent_2 = None, None
        if image_1 is not None:
            image_1 = image_resize(image_1, width, height, method, interpolation, condition, multiple_of, keep_proportion)
            latent_1 = {"samples": vae.encode(image_1[:,:,:,:3])}
        if image_2 is not None:
            image_2 = image_resize(image_2, width, height, method, interpolation, condition, multiple_of, keep_proportion)
            latent_2 = {"samples": vae.encode(image_2[:,:,:,:3])}
        
        if mask is not None and mask.shape[-1] > 1:
            channels = ["red", "green", "blue", "alpha"]
            mask = mask[:, :, :, channels.index(mask_channel)]
            
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
            if invert_mask:
                mask = 1.0 - mask

        return (latent_1, latent_2, mask, latent, width, height,)

