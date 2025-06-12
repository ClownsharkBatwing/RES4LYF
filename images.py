import torch
import torch.nn.functional as F
import math

from torchvision import transforms

from torch  import Tensor
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar, List

import numpy as np
import kornia
import cv2

from PIL import Image, ImageFilter, ImageEnhance

import comfy

# tensor -> PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL -> tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def freq_sep_fft(img, cutoff=5, sigma=10):
    fft_img = torch.fft.fft2(img, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_img)

    _, _, h, w = img.shape

    # freq domain -> meshgrid
    y, x = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device))
    center_y, center_x = h // 2, w // 2
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # smoother low-pass filter via gaussian filter
    low_pass_filter = torch.exp(-distance**2 / (2 * sigma**2))

    low_pass_filter = low_pass_filter.unsqueeze(0).unsqueeze(0)
    low_pass_fft = fft_shifted * low_pass_filter

    high_pass_fft = fft_shifted * (1 - low_pass_filter)

    # inverse FFT -> return to spatial domain
    low_pass_img  = torch.fft.ifft2(torch.fft.ifftshift( low_pass_fft), dim=(-2, -1)).real
    high_pass_img = torch.fft.ifft2(torch.fft.ifftshift(high_pass_fft), dim=(-2, -1)).real

    return low_pass_img, high_pass_img


def color_dodge_blend(base, blend):
    return torch.clamp(base / (1 - blend + 1e-8), 0, 1)
    
def color_scorch_blend(base, blend):
    return torch.clamp(1 - (1 - base) / (1 - blend + 1e-8), 0, 1)

def divide_blend(base, blend):
    return torch.clamp(base / (blend + 1e-8), 0, 1)

def color_burn_blend(base, blend):
    return torch.clamp(1 - (1 - base) / (blend + 1e-8), 0, 1)

def hard_light_blend(base, blend):
    return torch.where(blend <= 0.5, 
                       2 * base * blend, 
                       1 - 2 * (1 - base) * (1 - blend))

def hard_light_freq_sep(original, low_pass):
    high_pass = (color_burn_blend(original, (1 - low_pass)) + divide_blend(original, low_pass)) / 2
    return high_pass

def linear_light_blend(base, blend):
    return torch.where(blend <= 0.5,
                       base + 2 *  blend - 1,
                       base + 2 * (blend - 0.5))

def linear_light_freq_sep(base, blend):
    return (base + (1-blend)) / 2

def scale_to_range(value, min_old, max_old, min_new, max_new):
    return (value - min_old) / (max_old - min_old) * (max_new - min_new) + min_new


def normalize_lab(lab_image):
    L, A, B = lab_image[:, 0:1, :, :], lab_image[:, 1:2, :, :], lab_image[:, 2:3, :, :]

    L_normalized = L / 100.0
    A_normalized = scale_to_range(A, -128, 127, 0, 1)
    B_normalized = scale_to_range(B, -128, 127, 0, 1)    

    lab_normalized = torch.cat([L_normalized, A_normalized, B_normalized], dim=1)

    return lab_normalized

def denormalize_lab(lab_normalized):
    L_normalized, A_normalized, B_normalized = torch.split(lab_normalized, 1, dim=1)

    L = L_normalized * 100.0
    A = scale_to_range(A_normalized, 0, 1, -128, 127)
    B = scale_to_range(B_normalized, 0, 1, -128, 127)

    lab_image = torch.cat([L, A, B], dim=1)
    return lab_image


def rgb_to_lab(image):
    return kornia.color.rgb_to_lab(image)

def lab_to_rgb(image):
    return kornia.color.lab_to_rgb(image)

# cv2_layer() and ImageMedianBlur adapted from: https://github.com/Nourepide/ComfyUI-Allor/
    
def cv2_layer(tensor, function):
    """
    This function applies a given function to each channel of an input tensor and returns the result as a PyTorch tensor.

    :param tensor: A PyTorch tensor of shape (H, W, C) or (N, H, W, C), where C is the number of channels, H is the height, and W is the width of the image.
    :param function: A function that takes a numpy array of shape (H, W, C) as input and returns a numpy array of the same shape.
    :return: A PyTorch tensor of the same shape as the input tensor, where the given function has been applied to each channel of each image in the tensor.
    """
    shape_size = tensor.shape.__len__()

    def produce(image):
        channels = image[0, 0, :].shape[0]

        rgb = image[:, :, 0:3].numpy()
        result_rgb = function(rgb)

        if channels <= 3:
            return torch.from_numpy(result_rgb)
        elif channels == 4:
            alpha = image[:, :, 3:4].numpy()
            result_alpha = function(alpha)[..., np.newaxis]
            result_rgba = np.concatenate((result_rgb, result_alpha), axis=2)

            return torch.from_numpy(result_rgba)

    if shape_size == 3:
        return torch.from_numpy(produce(tensor))
    elif shape_size == 4:
        return torch.stack([
            produce(tensor[i]) for i in range(len(tensor))
        ])
    else:
        raise ValueError("Incompatible tensor dimension.")
    

# adapted from https://github.com/cubiq/ComfyUI_essentials
def image_resize(image,
                width,
                height,
                method          = "stretch",
                interpolation   = "nearest",
                condition       = "always",
                multiple_of     = 0,
                keep_proportion = False):
    
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



class ImageRepeatTileToSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "width":  ("INT",     {"default": 1024, "min": 1, "max": 1048576, "step": 1,}),
                "height": ("INT",     {"default": 1024, "min": 1, "max": 1048576, "step": 1,}),
                "crop":   ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, image, width, height, crop,            
            method          = "stretch",
            interpolation   = "lanczos",
            condition       = "always",
            multiple_of     = 0,
            keep_proportion = False,
        ):

        img = image.clone().detach()
        
        b, h, w, c = img.shape
        
        h_tgt = int(torch.ceil(torch.div(height, h)))
        w_tgt = int(torch.ceil(torch.div(width,  w)))
        
        img_tiled = torch.tile(img, (h_tgt, w_tgt, 1))
        
        if crop:
            img_tiled = img_tiled[:,:height, :width, :]
        else:
            img_tiled  = image_resize(img_tiled, width, height, method, interpolation, condition, multiple_of, keep_proportion)

        return (img_tiled,)




# Rewrite of the WAS Film Grain node, much improved speed and efficiency (https://github.com/WASasquatch/was-node-suite-comfyui)

class Film_Grain: 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":              ("IMAGE",),
                "density":            ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "intensity":          ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "highlights":         ("FLOAT", {"default": 1.0, "min": 0.01, "max": 255.0, "step": 0.01}),
                "supersample_factor": ("INT",   {"default": 4, "min": 1, "max": 8, "step": 1}),
                "repeats":            ("INT",   {"default": 1, "min": 1, "max": 1000, "step": 1})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/images"

    def main(self, image, density, intensity, highlights, supersample_factor, repeats=1):
        image = image.repeat(repeats, 1, 1, 1)
        return (self.apply_film_grain(image, density, intensity, highlights, supersample_factor), )

    def apply_film_grain(self, img, density=0.1, intensity=1.0, highlights=1.0, supersample_factor=4):

        img_batch = img.clone()
        img_list = []
        for i in range(img_batch.shape[0]):
            img = img_batch[i].unsqueeze(0)
            img = tensor2pil(img)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            # apply grayscale noise with specified density/intensity/highlights to PIL image
            img_gray = img.convert('L')
            original_size = img.size
            img_gray = img_gray.resize(
                ((img.size[0] * supersample_factor), (img.size[1] * supersample_factor)), Image.Resampling(2))
            num_pixels = int(density * img_gray.size[0] * img_gray.size[1])

            img_gray_tensor = torch.from_numpy(np.array(img_gray).astype(np.float32) / 255.0).to(device)
            img_gray_flat = img_gray_tensor.view(-1)
            num_pixels = int(density * img_gray_flat.numel())
            indices = torch.randint(0, img_gray_flat.numel(), (num_pixels,), device=img_gray_flat.device)
            values = torch.randint(0, 256, (num_pixels,), device=img_gray_flat.device, dtype=torch.float32) / 255.0
            
            img_gray_flat[indices] = values
            img_gray = img_gray_flat.view(img_gray_tensor.shape)
            
            img_gray_np = (img_gray.cpu().numpy() * 255).astype(np.uint8)
            img_gray = Image.fromarray(img_gray_np)

            img_noise = img_gray.convert('RGB')
            img_noise = img_noise.filter(ImageFilter.GaussianBlur(radius=0.125))
            img_noise = img_noise.resize(original_size, Image.Resampling(1))
            img_noise = img_noise.filter(ImageFilter.EDGE_ENHANCE_MORE)
            img_final = Image.blend(img, img_noise, intensity)
            enhancer = ImageEnhance.Brightness(img_final)
            img_highlights = enhancer.enhance(highlights)
            
            img_list.append(pil2tensor(img_highlights).squeeze(dim=0))
            
        img_highlights = torch.stack(img_list, dim=0)
        return img_highlights



class Image_Grain_Add: 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                #"density": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                #"intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                #"highlights": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 255.0, "step": 0.01}),
                #"supersample_factor": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                #"repeats": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    CATEGORY = "RES4LYF/images"

    def main(self, image, weight=0.5, density=1.0, intensity=1.0, highlights=1.0, supersample_factor=1.0, repeats=1):
        image = image.repeat(repeats, 1, 1, 1)
        image_grain = self.apply_film_grain(image, density, intensity, highlights, supersample_factor)
        
        return (image + weight * (hard_light_blend(image_grain, image) - image), )


    def apply_film_grain(self, img, density=0.1, intensity=1.0, highlights=1.0, supersample_factor=4):

        img_batch = img.clone()
        img_list = []
        for i in range(img_batch.shape[0]):
            img = img_batch[i].unsqueeze(0)
            img = tensor2pil(img)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            # apply grayscale noise with specified density/intensity/highlights to PIL image
            img_gray = img.convert('L')
            original_size = img.size
            img_gray = img_gray.resize(
                ((img.size[0] * supersample_factor), (img.size[1] * supersample_factor)), Image.Resampling(2))
            num_pixels = int(density * img_gray.size[0] * img_gray.size[1])

            img_gray_tensor = torch.from_numpy(np.array(img_gray).astype(np.float32) / 255.0).to(device)
            img_gray_flat = img_gray_tensor.view(-1)
            num_pixels = int(density * img_gray_flat.numel())
            indices = torch.randint(0, img_gray_flat.numel(), (num_pixels,), device=img_gray_flat.device)
            values = torch.randint(0, 256, (num_pixels,), device=img_gray_flat.device, dtype=torch.float32) / 255.0
            
            img_gray_flat[indices] = values
            img_gray = img_gray_flat.view(img_gray_tensor.shape)
            
            img_gray_np = (img_gray.cpu().numpy() * 255).astype(np.uint8)
            img_gray = Image.fromarray(img_gray_np)

            img_noise = img_gray.convert('RGB')
            img_noise = img_noise.filter(ImageFilter.GaussianBlur(radius=0.125))
            img_noise = img_noise.resize(original_size, Image.Resampling(1))
            img_noise = img_noise.filter(ImageFilter.EDGE_ENHANCE_MORE)
            img_final = Image.blend(img, img_noise, intensity)
            enhancer = ImageEnhance.Brightness(img_final)
            img_highlights = enhancer.enhance(highlights)
            
            img_list.append(pil2tensor(img_highlights).squeeze(dim=0))
            
        img_highlights = torch.stack(img_list, dim=0)
        return img_highlights




class Frequency_Separation_Hard_Light:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original":  ("IMAGE",),
                "low_pass":  ("IMAGE",),
            },
            "required": {
            },
        }
        
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, high_pass=None, original=None, low_pass=None):

        if high_pass is None:
            high_pass = hard_light_freq_sep(original.to(torch.float64).to('cuda'), low_pass.to(torch.float64).to('cuda'))
        
        if original is None:
            original = hard_light_blend(low_pass.to(torch.float64).to('cuda'), high_pass.to(torch.float64).to('cuda'))

        return (high_pass, original, low_pass,)


class Frequency_Separation_Hard_Light_LAB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original":  ("IMAGE",),
                "low_pass":  ("IMAGE",),
            },
            "required": {
            },
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, high_pass=None, original=None, low_pass=None):

        if original is not None:
            lab_original = rgb_to_lab(original.to(torch.float64).permute(0, 3, 1, 2))
            lab_original_normalized = normalize_lab(lab_original)
        
        if low_pass is not None:
            lab_low_pass = rgb_to_lab(low_pass.to(torch.float64).permute(0, 3, 1, 2))
            lab_low_pass_normalized = normalize_lab(lab_low_pass)

        if high_pass is not None:
            lab_high_pass = rgb_to_lab(high_pass.to(torch.float64).permute(0, 3, 1, 2))
            lab_high_pass_normalized = normalize_lab(lab_high_pass)

        #original_l = lab_original_normalized[:, :1, :, :]  
        #low_pass_l = lab_low_pass_normalized[:, :1, :, :]  

        if high_pass is None:
            lab_high_pass_normalized = hard_light_freq_sep(lab_original_normalized.permute(0, 2, 3, 1), lab_low_pass_normalized.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            lab_high_pass = denormalize_lab(lab_high_pass_normalized)
            high_pass = lab_to_rgb(lab_high_pass).permute(0, 2, 3, 1)
        if original is None:
            lab_original_normalized = hard_light_blend(lab_low_pass_normalized.permute(0, 2, 3, 1), lab_high_pass_normalized.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            lab_original = denormalize_lab(lab_original_normalized)
            original = lab_to_rgb(lab_original).permute(0, 2, 3, 1)

        return (high_pass, original, low_pass)
    
    
class Frame_Select:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("IMAGE",),
                "select": ("INT",  {"default": 0, "min": 0, "max": 10000}),
                
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, frames=None, select=0):
        frame = frames[select].unsqueeze(0).clone()
        return (frame,)
    
    
class Frames_Slice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames": ("IMAGE",),
                "start":  ("INT",  {"default": 0, "min": 0, "max": 10000}),
                "stop":   ("INT",  {"default": 1, "min": 1, "max": 10000}),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, frames=None, start=0, stop=1):
        frames_slice = frames[start:stop].clone()
        return (frames_slice,)


class Frames_Concat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {

            "required": {
                "frames_0": ("IMAGE",),
                "frames_1": ("IMAGE",),
            },
            "optional": {
            },
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, frames_0, frames_1):
        frames_concat = torch.cat((frames_0, frames_1), dim=0).squeeze(0).clone()
        return (frames_concat,)
    
    
    
class Image_Channels_LAB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "RGB": ("IMAGE",),
                "L": ("IMAGE",),
                "A": ("IMAGE",),
                "B": ("IMAGE",),
            },
            "required": {
            },
        }
        
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("RGB","L","A","B",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, RGB=None, L=None, A=None, B=None):

        if RGB is not None:
            LAB = rgb_to_lab(RGB.to(torch.float64).permute(0, 3, 1, 2))
            L, A, B = LAB[:, 0:1, :, :], LAB[:, 1:2, :, :], LAB[:, 2:3, :, :]
        else:
            LAB = torch.cat([L,A,B], dim=1)
            RGB = lab_to_rgb(LAB.to(torch.float64)).permute(0,2,3,1)

        return (RGB, L, A, B,)
    
    

class Frequency_Separation_Vivid_Light:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original":  ("IMAGE",),
                "low_pass":  ("IMAGE",),
            },
            "required": {
            },
        }
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, high_pass=None, original=None, low_pass=None):

        if high_pass is None:
            high_pass = hard_light_freq_sep(low_pass.to(torch.float64), original.to(torch.float64))
        
        if original is None:
            original = hard_light_blend(high_pass.to(torch.float64), low_pass.to(torch.float64))

        return (high_pass, original, low_pass,)


class Frequency_Separation_Linear_Light:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original":  ("IMAGE",),
                "low_pass":  ("IMAGE",),
            },
            "required": {
            },
        }
        
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, high_pass=None, original=None, low_pass=None):

        if high_pass is None:
            high_pass = linear_light_freq_sep(original.to(torch.float64).to('cuda'), low_pass.to(torch.float64).to('cuda'))
        
        if original is None:
            original = linear_light_blend(low_pass.to(torch.float64).to('cuda'), high_pass.to(torch.float64).to('cuda'))

        return (high_pass, original, low_pass,)


class Frequency_Separation_FFT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original":  ("IMAGE",),
                "low_pass":  ("IMAGE",),
            },
            "required": {
                "cutoff":    ("FLOAT", {"default": 5.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "sigma":     ("FLOAT", {"default": 5.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            },
        }
        
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, high_pass=None, original=None, low_pass=None, cutoff=5.0, sigma=5.0):

        if high_pass is None:
            low_pass, high_pass = freq_sep_fft(original.to(torch.float64), cutoff=cutoff, sigma=sigma)
        
        if original is None:
            original = low_pass + high_pass

        return (high_pass, original, low_pass,)
    
    


class ImageSharpenFS:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":    ("IMAGE",),
                #"method":    (["hard", "linear", "vivid"], {"default": "hard"}),
                "method":    (["hard", "linear"], {"default": "hard"}),
                "type":      (["median", "gaussian"],      {"default": "median"}),
                "intensity": ("INT",                       {"default": 6, "min": 1, "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, images, method, type, intensity):
        match type:
            case "median":
                IB = ImageMedianBlur()
            case "gaussian":
                IB = ImageGaussianBlur()
            
        match method:
            case "hard":
                FS = Frequency_Separation_Hard_Light()
            case "linear":
                FS = Frequency_Separation_Linear_Light()
                
        img_lp = IB.main(images, intensity)
        
        fs_hp, fs_orig, fs_lp = FS.main(None, images, *img_lp)
        
        _, img_sharpened, _ = FS.main(high_pass=fs_hp, original=None, low_pass=images)
        
        return (img_sharpened,)


    
    

class ImageMedianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "size":   ("INT", {"default": 6, "min": 1, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, images, size):
        size -= 1

        img = images.clone().detach()
        img = (img * 255).to(torch.uint8)

        return ((cv2_layer(img, lambda x: cv2.medianBlur(x, size)) / 255),)



class ImageGaussianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "size":   ("INT", {"default": 6, "min": 1, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, images, size):
        size -= 1 

        img = images.clone().detach()
        img = (img * 255).to(torch.uint8)

        return ((cv2_layer(img, lambda x: cv2.GaussianBlur(x, (size, size), 0)) / 255),)



def fast_smudge_blur_comfyui(img, kernel_size=51):
    img = img.to('cuda').float()

    # (b, h, w, c) to (b, c, h, w)
    img = img.permute(0, 3, 1, 2)

    num_channels = img.shape[1]

    box_kernel_1d = torch.ones(num_channels, 1, kernel_size, device=img.device, dtype=img.dtype) / kernel_size

    # apply box blur separately in horizontal and vertical directions
    blurred_img = F.conv2d(        img, box_kernel_1d.unsqueeze(2), padding=kernel_size // 2, groups=num_channels)
    blurred_img = F.conv2d(blurred_img, box_kernel_1d.unsqueeze(3), padding=kernel_size // 2, groups=num_channels)

    # (b, c, h, w) to (b, h, w, c)
    blurred_img = blurred_img.permute(0, 2, 3, 1)

    return blurred_img



class FastSmudgeBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":      ("IMAGE",), 
                "kernel_size": ("INT", {"default": 51, "min": 1, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, images, kernel_size):
        img = images.clone().detach().to('cuda').float()
        
        # (b, h, w, c) to (b, c, h, w)
        img = img.permute(0, 3, 1, 2)

        num_channels = img.shape[1]

        # box blur kernel (separable convolution)
        box_kernel_1d = torch.ones(num_channels, 1, kernel_size, device=img.device, dtype=img.dtype) / kernel_size

        padding_size = kernel_size // 2

        # apply box blur in horizontal/vertical dim separately
        blurred_img = F.conv2d(
            img, box_kernel_1d.unsqueeze(2), padding=(padding_size, 0), groups=num_channels
        )
        blurred_img = F.conv2d(
            blurred_img, box_kernel_1d.unsqueeze(3), padding=(0, padding_size), groups=num_channels
        )

        # (b, c, h, w) to (b, h, w, c)
        blurred_img = blurred_img.permute(0, 2, 3, 1)

        return (blurred_img,)



class Image_Pair_Split:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "img_pair": ("IMAGE",),
                }
            }
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("img_0","img_1",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, img_pair):
        img_0, img_1 = img_pair.chunk(2, dim=0)

        return (img_0, img_1,)



class Image_Crop_Location_Exact:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "x":      ("INT", {"default": 0,   "max": 10000000, "min": 0, "step": 1}),
                "y":      ("INT", {"default": 0,   "max": 10000000, "min": 0, "step": 1}),
                "width":  ("INT", {"default": 256, "max": 10000000, "min": 1, "step": 1}),
                "height": ("INT", {"default": 256, "max": 10000000, "min": 1, "step": 1}),
                "edge":   (["original", "short", "long"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA",)
    RETURN_NAMES = ("image", "crop_data",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"

    def main(self, image, x=0, y=0, width=256, height=256, edge="original"):
        if image.dim() != 4:
            raise ValueError("Expected a 4D tensor (batch, channels, height, width).")
        
        if edge == "short":
            side = width if width < height else height
            width, height = side, side
        if edge == "long":
            side = width if width > height else height
            width, height = side, side

        batch_size, img_height, img_width, channels = image.size()

        crop_left   = max(x, 0)
        crop_top    = max(y, 0)
        crop_right  = min(x + width, img_width)
        crop_bottom = min(y + height, img_height)

        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError("Invalid crop dimensions. Please check the values for x, y, width, and height.")

        cropped_image = image[:, crop_top:crop_bottom, crop_left:crop_right, :]

        crop_data = ((crop_width, crop_height), (crop_left, crop_top, crop_right, crop_bottom))

        return cropped_image, crop_data
    



class Masks_Unpack4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "masks": ("MASK",),
                }
            }
    RETURN_TYPES = ("MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("masks","masks","masks","masks",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    DESCRIPTION  = "Unpack a list of masks into separate outputs."

    def main(self, masks,):
        return (*masks,)

class Masks_Unpack8:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "masks": ("MASK",),
                }
            }
    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("masks","masks","masks","masks","masks","masks","masks","masks",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    DESCRIPTION  = "Unpack a list of masks into separate outputs."

    def main(self, masks,):
        return (*masks,)

class Masks_Unpack16:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "masks": ("MASK",),
                }
            }
    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("masks","masks","masks","masks","masks","masks","masks","masks","masks","masks","masks","masks","masks","masks","masks","masks",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    DESCRIPTION  = "Unpack a list of masks into separate outputs."

    def main(self, masks,):
        return (*masks,)






class Image_Get_Color_Swatches:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image_color_swatches": ("IMAGE",),
                }
            }
    RETURN_TYPES = ("COLOR_SWATCHES",)
    RETURN_NAMES = ("color_swatches",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"
    DESCRIPTION  = "Get color swatches, in the order they appear, from top to bottom, in an input image. For use with color masks."

    def main(self, image_color_swatches):
        rgb = (image_color_swatches * 255).round().clamp(0, 255).to(torch.uint8)
        color_swatches = read_swatch_colors(rgb.squeeze().numpy(), min_fraction=0.01)
        #color_swatches = read_swatch_colors(rgb.squeeze().numpy(), ignore=(255,255,255), min_fraction=0.01)

        return (color_swatches,)

class Masks_From_Color_Swatches:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image_color_mask": ("IMAGE",),
                "color_swatches":   ("COLOR_SWATCHES",),
                }
            }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"
    DESCRIPTION  = "Create masks from a multicolor image using color swatches to identify regions. Returns them as a list."

    def main(self, image_color_mask, color_swatches):
        rgb = (image_color_mask * 255).round().clamp(0, 255).to(torch.uint8)
        masks = build_masks_from_swatch(rgb.squeeze().numpy(), color_swatches, tol=8)
        masks = cleanup_and_fill_masks(masks)
        masks = torch.stack(masks, dim=0).unsqueeze(1)
        return (masks,)



class Masks_From_Colors:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image_color_swatches": ("IMAGE",),
                "image_color_mask":     ("IMAGE",),
                }
            }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/images"
    DESCRIPTION  = "Create masks from a multicolor image using color swatches to identify regions. Returns them as a list."

    def main(self, image_color_swatches, image_color_mask, ):
        rgb = (image_color_swatches * 255).round().clamp(0, 255).to(torch.uint8)
        color_swatches = read_swatch_colors(rgb.squeeze().numpy(), min_fraction=0.01)
        #color_swatches = read_swatch_colors(rgb.squeeze().numpy(), ignore=(255,255,255), min_fraction=0.01)
        
        rgb = (image_color_mask * 255).round().clamp(0, 255).to(torch.uint8)
        masks = build_masks_from_swatch(rgb.squeeze().numpy(), color_swatches, tol=8)
        masks = cleanup_and_fill_masks(masks)
        
        original_len = len(masks)
        masks = [m for m in masks if m.sum() != 0]
        
        removed = original_len - len(masks)
        print(f"Removed {removed} empty masks.")
        masks = torch.stack(masks, dim=0).unsqueeze(1)
        return (masks,)







from PIL import Image
import numpy as np

def read_swatch_colors(
    img,
    ignore: Tuple[int,int,int] = (-1,-1,-1),
    min_fraction: float = 0.2
) -> List[Tuple[int,int,int]]:
    """
    1. Load swatch, RGB.
    2. Count every unique color (except `ignore`).
    3. Discard any color whose count < (min_fraction * largest_count).
    4. Sort the remaining by their first y-position (top→bottom).
    """
    H, W, _ = img.shape
    flat = img.reshape(-1,3)
    
    # count all colors
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    # build list of (color, count), skipping white
    cc = [
        (tuple(c.tolist()), cnt)
        for c, cnt in zip(colors, counts)
        if tuple(c.tolist()) != ignore
    ]
    if not cc:
        return []
    
    # find largest band size
    max_cnt = max(cnt for _,cnt in cc)
    # filter by relative size
    kept = [c for c,cnt in cc if cnt >= max_cnt * min_fraction]
    
    # find first‐y for each kept color
    first_y = {}
    for color in kept:
        # mask of where that color lives
        mask = np.all(img == color, axis=-1)
        ys, xs = np.nonzero(mask)
        first_y[color] = int(np.min(ys))
    
    # sort top→bottom
    kept.sort(key=lambda c: first_y[c])
    return kept



import numpy as np
import torch
from typing import List, Tuple
from PIL import Image


def build_masks_from_swatch(
    mask_img: np.ndarray,
    swatch_colors: List[Tuple[int,int,int]],
    tol: int = 8
) -> List[torch.Tensor]:
    """
    1. Normalize mask_img → uint8 H×W×3 (handles float [0,1] or [0,255], channel-first too).
    2. Bin every pixel into buckets of size `tol`.
    3. Detect user-painted region (non-black).
    4. In swatch order, claim all exact matches (first-wins).
    5. Fill in any *painted but unclaimed* pixel by nearest‐swatch in RGB distance.
    Returns a list of BoolTensors [H,W], one per swatch color.
    """
    # --- 1) ensure H×W×3 uint8 ---
    img = mask_img
    # channel-first → channel-last
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1,2,0))
    # float → uint8
    if np.issubdtype(img.dtype, np.floating):
        m = img.max()
        if m <= 1.01:
            img = (img * 255.0).round()
        else:
            img = img.round()
    img = img.clip(0,255).astype(np.uint8)

    H, W, _ = img.shape

    # --- 2) bin into tol-sized buckets ---
    binned = (img // tol) * tol  # still uint8

    # --- 3) painted region mask (non-black) ---
    painted = np.any(img != 0, axis=2)  # H×W bool

    # --- snap swatch colors into same buckets ---
    snapped = np.array([
        ((np.array(c)//tol)*tol).astype(np.uint8)
        for c in swatch_colors
    ])  # C×3

    claimed = np.zeros((H, W), dtype=bool)
    masks   = []

    # --- 4) first-pass exact matches ---
    for s in snapped:
        m = (
            (binned[:,:,0] == s[0]) &
            (binned[:,:,1] == s[1]) &
            (binned[:,:,2] == s[2])
        )
        m &= ~claimed
        masks.append(torch.from_numpy(m))
        claimed |= m

    # --- 5) fill-in only within painted & unclaimed pixels ---
    miss = painted & (~claimed)
    if miss.any():
        flat       = binned.reshape(-1,3).astype(int)  # (H*W)×3
        flat_miss  = miss.reshape(-1)                 # (H*W,)
        # squared RGB distances to each swatch: → (H*W)×C
        d2         = np.sum((flat[:,None,:] - snapped[None,:,:])**2, axis=2)
        nearest    = np.argmin(d2, axis=1)            # (H*W,)

        for i in range(len(masks)):
            assign = (flat_miss & (nearest == i)).reshape(H, W)
            masks[i] = masks[i] | torch.from_numpy(assign)

    return masks




import numpy as np
import torch
from typing import List
from collections import deque

def _remove_small_components(
    mask: np.ndarray,
    rel_thresh: float = 0.01
) -> np.ndarray:
    """
    Remove connected components smaller than rel_thresh * max_component_size.
    4-connectivity.
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, bool)
    comps = []  # list of (size, pixels_list)

    # 1) find all components
    for y in range(H):
        for x in range(W):
            if mask[y,x] and not visited[y,x]:
                q = deque([(y,x)])
                visited[y,x] = True
                pix = [(y,x)]
                while q:
                    cy,cx = q.popleft()
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx = cy+dy, cx+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx] = True
                            q.append((ny,nx))
                            pix.append((ny,nx))
                comps.append(pix)

    if not comps:
        return np.zeros_like(mask)

    # 2) compute threshold
    sizes = [len(c) for c in comps]
    max_size = max(sizes)
    min_size = max_size * rel_thresh

    # 3) build a new mask keeping only large comps
    out = np.zeros_like(mask)
    for pix in comps:
        if len(pix) >= min_size:
            for (y,x) in pix:
                out[y,x] = True

    return out

def cleanup_and_fill_masks(
    masks: List[torch.Tensor],
    rel_thresh: float = 0.01
) -> List[torch.Tensor]:
    """
    1) Remove any component < rel_thresh * (largest component) per mask
    2) Then re-assign any freed pixels to nearest-swatches by neighbor-count
    """
    # stack into C×H×W
    np_masks = np.stack([m.cpu().numpy() for m in masks], axis=0)
    C, H, W = np_masks.shape

    # 1) component pruning
    for c in range(C):
        np_masks[c] = _remove_small_components(np_masks[c], rel_thresh)

    # 2) figure out what’s still unclaimed
    claimed = np_masks.any(axis=0)  # H×W

    # 3) build neighbor‐counts to know who's closest
    #    (reuse the same 8-neighbor idea to bias to the largest local region)
    shifts = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    neighbor_counts = np.zeros_like(np_masks, int)
    for dy,dx in shifts:
        neighbor_counts += np.roll(np.roll(np_masks, dy, axis=1), dx, axis=2)

    # 4) for every pixel still unclaimed, pick the mask with the highest neighbor count
    miss = ~claimed
    if miss.any():
        # which mask “wins” that pixel?
        winner = np.argmax(neighbor_counts, axis=0)  # H×W
        for c in range(C):
            assign = (miss & (winner == c))
            np_masks[c][assign] = True

    # back to torch
    cleaned = [torch.from_numpy(np_masks[c]) for c in range(C)]
    return cleaned

import os
import folder_paths


class MaskSketch:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    
    def load_image(self, image):
        width, height = 512, 512  # or whatever size you prefer

        # White image: RGB values all set to 1.0
        white_image = torch.ones((1, height, width, 3), dtype=torch.float32)

        # White mask: all ones (or zeros if you're using inverse alpha)
        white_mask = torch.zeros((1, height, width), dtype=torch.float32)

        return (white_image, white_mask)
        
    def load_image_orig(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True



# based on https://github.com/cubiq/ComfyUI_essentials/blob/main/mask.py
import math
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import numpy as np
from scipy.ndimage import distance_transform_edt

class MaskBoundingBoxAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "padding":      ("INT",   { "default": 0, "min": 0,   "max": 4096, "step": 1 }),
                "blur":         ("INT",   { "default": 0, "min": 0,   "max": 256,  "step": 1 }),
                "aspect_ratio": ("FLOAT", { "default": 1.0, "min": 0.01,"max": 10.0, "step": 0.01 }),
                "transpose":    ("BOOLEAN",{"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask":  ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK","MASK","INT","INT","INT","INT")
    RETURN_NAMES = ("image","mask","mask_blurred","x","y","width","height")
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, padding, blur, aspect_ratio, transpose, image=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        B, H, W = mask.shape
        hard     = mask.clone()

        # build outward-only “blurred” mask via distance transform
        if blur > 0:
            m_bool = hard[0].cpu().numpy().astype(bool)
            d_out  = distance_transform_edt(~m_bool)
            d_in   = distance_transform_edt( m_bool)
            alpha  = np.zeros_like(d_out, np.float32)
            alpha[d_in>0] = 1.0
            ramp = np.clip(1.0 - (d_out / blur), 0.0, 1.0)
            alpha[d_out>0] = ramp[d_out>0]
            mask_blur_full = torch.from_numpy(alpha)[None,...].to(hard.device)
        else:
            mask_blur_full = hard.clone()

        # calc tight bbox + padding on the "hard" mask
        ys, xs = torch.where(hard[0] > 0)
        x1 = max(0, int(xs.min()) - padding)
        x2 = min(W, int(xs.max()) + 1 + padding)
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(H, int(ys.max()) + 1 + padding)
        w0 = x2 - x1
        h0 = y2 - y1

        if image is None:
            img_full = hard.unsqueeze(-1).repeat(1,1,1,3).to(torch.float32)
        else:
            img_full = image
            
        if img_full.shape[1:3] != (H, W):
            img_full = comfy.utils.common_upscale(
                img_full.permute(0,3,1,2),
                W, H, upscale_method="bicubic", crop="center"
            ).permute(0,2,3,1)

        ar = aspect_ratio
        req_w = math.ceil(h0 * ar)   # how wide we'd need to be to hit AR at h0
        req_h = math.floor(w0 / ar)  # how tall we'd need to be to hit AR at w0

        new_x1, new_x2 = x1, x2
        new_y1, new_y2 = y1, y2

        flush_left  = (x1 == 0)
        flush_right = (x2 == W)
        flush_top   = (y1 == 0)
        flush_bot   = (y2 == H)

        if not transpose:
            if req_w > w0: # widen?
                target_w = min(W, req_w)
                delta    = target_w - w0
                if flush_right:
                    new_x1, new_x2 = W - target_w, W
                elif flush_left:
                    new_x1, new_x2 = 0, target_w
                else:
                    off = delta // 2
                    new_x1 = max(0, x1 - off)
                    new_x2 = new_x1 + target_w
                    if new_x2 > W:
                        new_x2 = W
                        new_x1 = W - target_w

            elif req_h > h0: # vertical bloater?
                target_h = min(H, req_h)
                delta    = target_h - h0
                if flush_bot:
                    new_y1, new_y2 = H - target_h, H
                elif flush_top:
                    new_y1, new_y2 = 0, target_h
                else:
                    off = delta // 2
                    new_y1 = max(0, y1 - off)
                    new_y2 = new_y1 + target_h
                    if new_y2 > H:
                        new_y2 = H
                        new_y1 = H - target_h

        else:
            if req_h > h0:
                target_h = min(H, req_h)
                delta    = target_h - h0
                if flush_bot:
                    new_y1, new_y2 = H - target_h, H
                elif flush_top:
                    new_y1, new_y2 = 0, target_h
                else:
                    off = delta // 2
                    new_y1 = max(0, y1 - off)
                    new_y2 = new_y1 + target_h
                    if new_y2 > H:
                        new_y2 = H
                        new_y1 = H - target_h

            elif req_w > w0:
                target_w = min(W, req_w)
                delta    = target_w - w0
                if flush_right:
                    new_x1, new_x2 = W - target_w, W
                elif flush_left:
                    new_x1, new_x2 = 0, target_w
                else:
                    off = delta // 2
                    new_x1 = max(0, x1 - off)
                    new_x2 = new_x1 + target_w
                    if new_x2 > W:
                        new_x2 = W
                        new_x1 = W - target_w

        final_w = new_x2 - new_x1
        final_h = new_y2 - new_y1

        # done... crop image & masks
        img_crop      = img_full[:,    new_y1:new_y2, new_x1:new_x2, :]
        mask_crop     = hard[:,       new_y1:new_y2, new_x1:new_x2   ]
        mask_blurred  = mask_blur_full[:, new_y1:new_y2, new_x1:new_x2]

        return (
            img_crop,
            mask_crop,
            mask_blurred,
            new_x1,
            new_y1,
            final_w,
            final_h,
        )


