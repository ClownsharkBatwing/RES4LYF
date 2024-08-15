import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Film_Grain: #Rewrite of the WAS Film Grain node, much improved speed and efficiency
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

    CATEGORY = "WAS Suite/Image/Filter"

    def main(self, image, density, intensity, highlights, supersample_factor):
        return (pil2tensor(self.apply_film_grain(tensor2pil(image), density, intensity, highlights, supersample_factor)), )

    def apply_film_grain(self, img, density=0.1, intensity=1.0, highlights=1.0, supersample_factor=4):
        #Apply grayscale noise with specified density, intensity, and highlights to a PIL image.
        img_gray = img.convert('L')
        original_size = img.size
        img_gray = img_gray.resize(
            ((img.size[0] * supersample_factor), (img.size[1] * supersample_factor)), Image.Resampling(2))
        num_pixels = int(density * img_gray.size[0] * img_gray.size[1])

        img_gray_tensor = torch.from_numpy(np.array(img_gray).astype(np.float32) / 255.0).to("cuda")
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
