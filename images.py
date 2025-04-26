import torch
import torch.nn.functional as F

from torchvision import transforms

from torch  import Tensor
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar, List

import numpy as np
import kornia
import cv2

from PIL import Image, ImageFilter, ImageEnhance

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
    



# Rewrite of the WAS Film Grain node, much improved speed and efficiency (https://github.com/WASasquatch/was-node-suite-comfyui)

class Film_Grain: 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "density": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 255.0, "step": 0.01}),
                "supersample_factor": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "repeats": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1})
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
                "method":    (["hard", "linear", "vivid"], {"default": "hard"}),
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
    1. Snap every pixel in mask_img into bins of size tol
    2. Snap each swatch color into the same bins
    3. In swatch order, claim pixels exactly matching each swatch bin (first-wins)
    4. For any unclaimed pixel, compute its nearest swatch‐bin by RGB distance
       and assign it to that mask.
    Returns: list of H×W bool‐tensors, one per swatch color.
    """
    H, W, _ = mask_img.shape

    # 1) Bin the mask
    binned = (mask_img // tol) * tol  # H×W×3

    # 2) Bin the swatch colors
    snapped_swatches = np.array([(np.array(c)//tol)*tol for c in swatch_colors])  # C×3

    claimed = np.zeros((H, W), dtype=bool)
    masks   = []

    # 3) First‐pass exact matches
    for c in snapped_swatches:
        match = (
            (binned[:,:,0] == c[0]) &
            (binned[:,:,1] == c[1]) &
            (binned[:,:,2] == c[2])
        )
        match &= ~claimed
        masks.append(torch.from_numpy(match))
        claimed |= match

    # 4) Fill‐in pass for any unclaimed pixels
    miss = ~claimed
    if miss.any():
        # flatten for vectorized nearest‐neighbor
        flat = binned.reshape(-1,3)           # (H*W)×3
        flat_miss = miss.reshape(-1)          # (H*W,)
        # compute squared‐distance to each swatch color
        # (H*W)×1×3  minus 1×C×3 → (H*W)×C×3 → sum over rgb → (H*W)×C
        dists = np.sum((flat[:,None,:] - snapped_swatches[None,:,:])**2, axis=2)
        nearest = np.argmin(dists, axis=1)    # (H*W,)

        # assign each missed pixel to its nearest swatch mask
        for i in range(len(masks)):
            assign = (flat_miss & (nearest == i)).reshape(H, W)
            # OR into the existing mask
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



