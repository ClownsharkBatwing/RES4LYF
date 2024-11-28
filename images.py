import torch
import numpy as np
import kornia
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import torch.nn.functional as F

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
    low_pass_img = torch.fft.ifft2(torch.fft.ifftshift(low_pass_fft), dim=(-2, -1)).real
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
    return torch.where(blend <= 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))

def hard_light_freq_sep(original, low_pass):
    high_pass = (color_burn_blend(original, (1 - low_pass)) + divide_blend(original, low_pass)) / 2
    return high_pass


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
    




class Film_Grain: # Rewrite of the WAS Film Grain node, much improved speed and efficiency (https://github.com/WASasquatch/was-node-suite-comfyui)
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
                "supersample_factor": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"

    CATEGORY = "image/filter"

    def main(self, image, density, intensity, highlights, supersample_factor):
        return (pil2tensor(self.apply_film_grain(tensor2pil(image), density, intensity, highlights, supersample_factor)), )

    def apply_film_grain(self, img, density=0.1, intensity=1.0, highlights=1.0, supersample_factor=4):

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

        return img_highlights
    

class Frequency_Separation_Hard_Light:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original": ("IMAGE",),
                "low_pass": ("IMAGE",),
            },
            "required": {
            },
        }
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION = "main"

    CATEGORY = "image/channels"

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
                "original": ("IMAGE",),
                "low_pass": ("IMAGE",),
            },
            "required": {
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION = "main"

    CATEGORY = "image/channels"

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
    FUNCTION = "main"

    CATEGORY = "image/channels"

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
                "original": ("IMAGE",),
                "low_pass": ("IMAGE",),
            },
            "required": {
            },
        }
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION = "main"

    CATEGORY = "image/channels"

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
                "original": ("IMAGE",),
                "low_pass": ("IMAGE",),
            },
            "required": {
            },
        }
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION = "main"

    CATEGORY = "image/channels"

    def main(self, high_pass=None, original=None, low_pass=None):

        if high_pass is None:
            high_pass = hard_light_freq_sep(original, low_pass)
        
        if original is None:
            original = hard_light_blend(low_pass, high_pass)

        return (high_pass, original, low_pass,)

class Frequency_Separation_FFT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "high_pass": ("IMAGE",),
                "original": ("IMAGE",),
                "low_pass": ("IMAGE",),
            },
            "required": {
                "cutoff": ("FLOAT", {"default": 5.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 5.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            },
        }
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("high_pass", "original", "low_pass",)
    FUNCTION = "main"

    CATEGORY = "image/channels"

    def main(self, high_pass=None, original=None, low_pass=None, cutoff=5.0, sigma=5.0):

        if high_pass is None:
            low_pass, high_pass = freq_sep_fft(original.to(torch.float64), cutoff=cutoff, sigma=sigma)
        
        if original is None:
            original = low_pass + high_pass

        return (high_pass, original, low_pass,)
    

class ImageMedianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "size": ("INT", {
                    "default": 6,
                    "min": 1,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "image/filter"

    def main(self, images, size):
        size -= 1

        img = images.clone().detach()
        img = (img * 255).to(torch.uint8)

        return ((cv2_layer(img, lambda x: cv2.medianBlur(x, size)) / 255),)

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
                "images": ("IMAGE",), 
                "kernel_size": ("INT", {
                    "default": 51,
                    "min": 1,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "image/filter"

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
        return {"required": { "img_pair": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("img_0","img_1",)

    FUNCTION = "main"

    CATEGORY = "image/batch"

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
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "y": ("INT", {"default": 0, "max": 10000000, "min": 0, "step": 1}),
                "width": ("INT", {"default": 256, "max": 10000000, "min": 1, "step": 1}),
                "height": ("INT", {"default": 256, "max": 10000000, "min": 1, "step": 1}),
                "edge": (["original", "short", "long"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    FUNCTION = "main"

    CATEGORY = "image/transform"

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
    


"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

class VGG19StyleTransfer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "style_weight": ("FLOAT", {
                    "default": 1e6,
                    "min": 1e3,
                    "max": 1e7,
                    "step": 1e3,
                }),
                "content_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                }),
                "num_steps": ("INT", {
                    "default": 500,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "image/stylization"
    def main(self, content_image, style_image, style_weight, content_weight, num_steps):
        # Explicitly enable gradients
        with torch.inference_mode(False):
            with torch.enable_grad():
                # Preprocess and reattach images
                content_image = content_image.clone().detach().float().requires_grad_(False)
                style_image = style_image.clone().detach().float().requires_grad_(False)

                content_image = self._preprocess(content_image)
                style_image = self._preprocess(style_image)

                # Load VGG19 model
                vgg = self._load_vgg19()

                # Extract content and style features
                content_layers = [21]  # relu4_2
                style_layers = [1, 6, 11, 20, 29]  # relu1_1, relu2_1, ..., relu5_1
                content_features = self._extract_features(content_image, vgg, content_layers)
                style_features = self._extract_features(style_image, vgg, style_layers, gram=True)

                # Initialize generated image and attach to computation graph
                generated_image = content_image.clone().detach().requires_grad_(True)
                generated_image = torch.nn.Parameter(generated_image)

                # Use LBFGS optimizer
                optimizer = optim.LBFGS([generated_image])

                # Optimization loop
                def closure():
                    optimizer.zero_grad()

                    # Extract features from the generated image
                    gen_content_features = self._extract_features(generated_image, vgg, content_layers)
                    gen_style_features = self._extract_features(generated_image, vgg, style_layers, gram=True)

                    # Compute losses
                    content_loss = self._compute_content_loss(gen_content_features, content_features, content_weight)
                    style_loss = self._compute_style_loss(gen_style_features, style_features, style_weight)

                    # Combine losses
                    total_loss = content_loss + style_loss

                    # Debugging to ensure proper gradient flow
                    print("Content Loss:", content_loss)
                    print("Style Loss:", style_loss)
                    print("Total Loss:", total_loss)
                    print("Generated Image grad_fn:", generated_image.grad_fn)

                    # Backpropagate gradients
                    total_loss.backward()

                    return total_loss  # Return the total loss (tensor) for LBFGS

                # Run optimization
                for step in range(num_steps):
                    optimizer.step(closure)
                    if step % 50 == 0:
                        print(f"Step {step}/{num_steps}")

                # Postprocess and return the result
                generated_image = self._postprocess(generated_image.detach())
            return (generated_image,)


    def _load_vgg19(self):
        # Load pretrained VGG19 model and freeze weights
        vgg = models.vgg19(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg.to("cuda").eval()

    def _preprocess(self, img):
        # Normalize the input image to [0, 1] and apply VGG-specific normalization
        img = img / 255.0  # Normalize to [0, 1]
        img = img.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std

    def _postprocess(self, img):
        # Reverse the normalization and convert back to [0, 255]
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = img * std + mean  # Denormalize
        img = torch.clamp(img, 0, 1)  # Clamp to valid range
        img = (img * 255).to(torch.uint8)  # Convert to [0, 255]
        return img.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)

    def _extract_features(self, img, model, layers, gram=False):
        # Extract features from specified layers
        features = {}
        x = img
        for idx, layer in enumerate(model.children()):
            x = layer(x)
            if idx in layers:
                features[idx] = self._gram_matrix(x) if gram else x
        return features

    def _gram_matrix(self, tensor):
        # Compute Gram matrix for style representation
        _, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram / (c * h * w)

    def _compute_content_loss(self, gen_features, content_features, weight):
        # Compute content loss (MSE)
        loss = 0
        for layer in gen_features.keys():
            loss += torch.mean((gen_features[layer] - content_features[layer]) ** 2)
        return weight * loss

    def _compute_style_loss(self, gen_features, style_features, weight):
        # Compute style loss (MSE on Gram matrices)
        loss = 0
        for layer in gen_features.keys():
            loss += torch.mean((gen_features[layer] - style_features[layer]) ** 2)
        return weight * loss
"""