import torch
from torch import nn, Tensor, Generator, lerp
from torch.nn.functional import unfold
from typing import Callable, Tuple
from math import pi
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from torch.distributions import StudentT, Laplace
import numpy as np
import pywt
import functools

"""def cast_fp64(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Cast all tensor arguments to float64
        new_args = [arg.to(torch.float64) if torch.is_tensor(arg) else arg for arg in args]
        new_kwargs = {k: v.to(torch.float64) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    return wrapper"""

def cast_fp64(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find the first tensor argument to determine the target device
        target_device = None
        for arg in args:
            if torch.is_tensor(arg):
                target_device = arg.device
                break
        if target_device is None:
            for v in kwargs.values():
                if torch.is_tensor(v):
                    target_device = v.device
                    break
        
        # Cast all tensor arguments to float64 and move them to the target device
        new_args = [arg.to(torch.float64).to(target_device) if torch.is_tensor(arg) else arg for arg in args]
        new_kwargs = {k: v.to(torch.float64).to(target_device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        
        return func(*new_args, **new_kwargs)
    return wrapper

def noise_generator_factory(cls, **fixed_params):
    def create_instance(**kwargs):
        # Combine fixed_params with kwargs, giving priority to kwargs
        params = {**fixed_params, **kwargs}
        return cls(**params)
    return create_instance

def like(x):
    return {'size': x.shape, 'dtype': x.dtype, 'layout': x.layout, 'device': x.device}

def scale_to_range(x, scaled_min = -1.73, scaled_max = 1.73):
    return scaled_min + (x - x.min()) * (scaled_max - scaled_min) / (x.max() - x.min())

def normalize(x):
     return (x - x.mean())/ x.std()

class NoiseGenerator:
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None):
        self.seed = seed

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if x is not None:
            self.x      = x
            self.size   = x.shape
            self.dtype  = x.dtype
            self.layout = x.layout
            self.device = x.device
        else:   
            self.x      = torch.zeros(size, dtype, layout, device)

        # allow overriding parameters imported from latent 'x' if specified
        if size is not None:
            self.size   = size
        if dtype is not None:
            self.dtype  = dtype
        if layout is not None:
            self.layout = layout
        if device is not None:
            self.device = device

        if generator is None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.generator = generator

    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")
    
    def update(self, **kwargs):
        updated_values = []
        for attribute_name, value in kwargs.items():
            if value is not None:
                setattr(self, attribute_name, value)
            updated_values.append(getattr(self, attribute_name))
        return tuple(updated_values)

class BrownianNoiseGenerator(NoiseGenerator):
    def __call__(self, *, sigma=None, sigma_next=None):
        return BrownianTreeNoiseSampler(self.x, self.sigma_min, self.sigma_max, seed=self.seed, cpu = self.device.type=='cpu')(sigma, sigma_next)

class FractalNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 alpha=0.0, k=1.0, scale=0.1): 
        self.update(alpha=alpha, k=k, scale=scale)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, alpha=None, k=None, scale=None, **kwargs):
        self.update(alpha=alpha, k=k, scale=scale)
        #pdb.set_trace()

        b, c, h, w = self.size
        
        noise = torch.normal(mean=0.0, std=1.0, size=self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)
        
        y_freq = torch.fft.fftfreq(h, 1/h, device=self.device)
        x_freq = torch.fft.fftfreq(w, 1/w, device=self.device)
        freq = torch.sqrt(y_freq[:, None]**2 + x_freq[None, :]**2).clamp(min=1e-10)
        
        spectral_density = self.k / torch.pow(freq, self.alpha * self.scale)
        spectral_density[0, 0] = 0

        noise_fft = torch.fft.fft2(noise)
        modified_fft = noise_fft * spectral_density
        noise = torch.fft.ifft2(modified_fft).real

        return noise / torch.std(noise)

class PyramidNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 discount=0.8, mode='nearest-exact'):
        self.update(discount=discount, mode=mode)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, discount=None, mode=None, **kwargs):
        self.update(discount=discount, mode=mode)

        x = torch.zeros(self.size, dtype=self.dtype, layout=self.layout, device=self.device)
        b, c, h, w = self.size
        orig_h, orig_w = h, w

        r = 1
        for i in range(5):
            r *= 2
            x += torch.nn.functional.interpolate(
                torch.normal(mean=0, std=0.5 ** i, size=(b, c, h * r, w * r), dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator),
                size=(orig_h, orig_w), mode=self.mode
            ) * self.discount ** i
        return x / x.std()

class HiresPyramidNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 discount=0.7, mode='nearest-exact'):
        self.update(discount=discount, mode=mode)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, discount=None, mode=None, **kwargs):
        self.update(discount=discount, mode=mode)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        u = nn.Upsample(size=(orig_h, orig_w), mode=self.mode).to(self.device)

        noise = ((torch.rand(size=self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator) - 0.5) * 2 * 1.73)

        for i in range(4):
            r = torch.rand(1, device=self.device, generator=self.generator).item() * 2 + 2
            h, w = min(orig_h * 15, int(h * (r ** i))), min(orig_w * 15, int(w * (r ** i)))

            new_noise = torch.randn((b, c, h, w), dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)
            upsampled_noise = u(new_noise)
            noise += upsampled_noise * self.discount ** i
            
            if h >= orig_h * 15 or w >= orig_w * 15:
                break  # if resolution is too high
        
        return noise / noise.std()

class UniformNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 mean=0.0, scale=1.73):
        self.update(mean=mean, scale=scale)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, mean=None, scale=None, **kwargs):
        self.update(mean=mean, scale=scale)

        noise = torch.rand(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)

        return self.scale * 2 * (noise - 0.5) + self.mean

class GaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 mean=0.0, std=1.0):
        self.update(mean=mean, std=std)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, mean=None, std=None, **kwargs):
        self.update(mean=mean, std=std)

        noise = torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)

        return noise * self.std + self.mean

class LaplacianNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 loc=0, scale=1.0):
        self.update(loc=loc, scale=scale)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, loc=None, scale=None, **kwargs):
        self.update(loc=loc, scale=scale)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        noise = torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator) / 4.0

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.generator.initial_seed())
        laplacian_noise = Laplace(loc=self.loc, scale=self.scale).rsample(self.size).to(self.device)
        self.generator.manual_seed(self.generator.initial_seed() + 1)
        torch.random.set_rng_state(rng_state)

        noise += laplacian_noise
        return noise / noise.std()

class StudentTNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 loc=0, scale=0.2, df=1):
        self.update(loc=loc, scale=scale, df=df)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, loc=None, scale=None, df=None, **kwargs):
        self.update(loc=loc, scale=scale, df=df)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.generator.initial_seed())
        noise = StudentT(loc=0, scale=0.2, df=1).rsample(self.size)
        self.generator.manual_seed(self.generator.initial_seed() + 1)
        torch.random.set_rng_state(rng_state)

        s = torch.quantile(noise.flatten(start_dim=1).abs(), 0.75, dim=-1)
        s = s.reshape(*s.shape, 1, 1, 1)
        noise = noise.clamp(-s, s)
        return torch.copysign(torch.pow(torch.abs(noise), 0.5), noise)

class WaveletNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 wavelet='haar'):
        self.update(wavelet=wavelet)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, wavelet=None, **kwargs):
        self.update(wavelet=wavelet)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        # noise for spatial dimensions only
        coeffs = pywt.wavedecn(torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator), wavelet=self.wavelet, mode='periodization')
        noise = pywt.waverecn(coeffs, wavelet=self.wavelet, mode='periodization')
        noise_tensor = torch.tensor(noise, dtype=self.dtype, device=self.device)

        noise_tensor = (noise_tensor - noise_tensor.mean()) / noise_tensor.std()
        return noise_tensor

class PerlinNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 detail=0.0):
        self.update(detail=detail)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    @staticmethod
    def get_positions(block_shape: Tuple[int, int]) -> Tensor:
        bh, bw = block_shape
        positions = torch.stack(
            torch.meshgrid(
                [(torch.arange(b) + 0.5) / b for b in (bw, bh)],
                indexing="xy",
            ),
            -1,
        ).view(1, bh, bw, 1, 1, 2)
        return positions

    @staticmethod
    def unfold_grid(vectors: Tensor) -> Tensor:
        batch_size, _, gpy, gpx = vectors.shape
        return (
            unfold(vectors, (2, 2))
            .view(batch_size, 2, 4, -1)
            .permute(0, 2, 3, 1)
            .view(batch_size, 4, gpy - 1, gpx - 1, 2)
        )

    @staticmethod
    def smooth_step(t: Tensor) -> Tensor:
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def perlin_noise_tensor(
        self,
        vectors: Tensor, positions: Tensor, step: Callable = None
    ) -> Tensor:
        if step is None:
            step = self.smooth_step

        batch_size = vectors.shape[0]
        # grid height, grid width
        gh, gw = vectors.shape[2:4]
        # block height, block width
        bh, bw = positions.shape[1:3]

        for i in range(2):
            if positions.shape[i + 3] not in (1, vectors.shape[i + 2]):
                raise Exception(
                    f"Blocks shapes do not match: vectors ({vectors.shape[1]}, {vectors.shape[2]}), positions {gh}, {gw})"
                )

        if positions.shape[0] not in (1, batch_size):
            raise Exception(
                f"Batch sizes do not match: vectors ({vectors.shape[0]}), positions ({positions.shape[0]})"
            )

        vectors = vectors.view(batch_size, 4, 1, gh * gw, 2)
        positions = positions.view(positions.shape[0], bh * bw, -1, 2)

        step_x = step(positions[..., 0])
        step_y = step(positions[..., 1])

        row0 = lerp(
            (vectors[:, 0] * positions).sum(dim=-1),
            (vectors[:, 1] * (positions - positions.new_tensor((1, 0)))).sum(dim=-1),
            step_x,
        )
        row1 = lerp(
            (vectors[:, 2] * (positions - positions.new_tensor((0, 1)))).sum(dim=-1),
            (vectors[:, 3] * (positions - positions.new_tensor((1, 1)))).sum(dim=-1),
            step_x,
        )
        noise = lerp(row0, row1, step_y)
        return (
            noise.view(
                batch_size,
                bh,
                bw,
                gh,
                gw,
            )
            .permute(0, 3, 1, 4, 2)
            .reshape(batch_size, gh * bh, gw * bw)
        )

    def perlin_noise(
        self,
        grid_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        batch_size: int = 1,
        generator: Generator = None,
        *args,
        **kwargs,
    ) -> Tensor:
        # grid height and width
        gh, gw = grid_shape
        # output height and width
        oh, ow = out_shape
        # block height and width
        bh, bw = oh // gh, ow // gw

        if oh != bh * gh:
            raise Exception(f"Output height {oh} must be divisible by grid height {gh}")
        if ow != bw * gw != 0:
            raise Exception(f"Output width {ow} must be divisible by grid width {gw}")

        angle = torch.empty(
            [batch_size] + [s + 1 for s in grid_shape], device=self.device, *args, **kwargs
        ).uniform_(to=2.0 * pi, generator=self.generator)
        # random vectors on grid points
        vectors = self.unfold_grid(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
        # positions inside grid cells [0, 1)
        positions = self.get_positions((bh, bw)).to(vectors)
        return self.perlin_noise_tensor(self, vectors, positions).squeeze(0)

    def __call__(self, *, detail=None, **kwargs):
        self.update(detail=detail) #currently unused

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        noise = torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator) / 2.0
        noise_size_H = noise.size(dim=2)
        noise_size_W = noise.size(dim=3)
        perlin = None
        for i in range(2):
            noise += self.perlin_noise((noise_size_H, noise_size_W), (noise_size_H, noise_size_W), batch_size=self.x.shape[1], generator=self.generator).to(self.device)
        return noise / noise.std()

NOISE_GENERATOR_CLASSES = {
    "fractal": FractalNoiseGenerator,
    "pyramid": PyramidNoiseGenerator,
    "gaussian": GaussianNoiseGenerator,
    "uniform": UniformNoiseGenerator,
    "pyramid-bilinear": noise_generator_factory(PyramidNoiseGenerator, mode='bilinear'),
    "pyramid-bicubic": noise_generator_factory(PyramidNoiseGenerator, mode='bicubic'),   
    "pyramid-nearest": noise_generator_factory(PyramidNoiseGenerator, mode='nearest'),  
    "hires-pyramid-bilinear": noise_generator_factory(HiresPyramidNoiseGenerator, mode='bilinear'),
    "hires-pyramid-bicubic": noise_generator_factory(HiresPyramidNoiseGenerator, mode='bicubic'),   
    "hires-pyramid-nearest": noise_generator_factory(HiresPyramidNoiseGenerator, mode='nearest'),  
    "brownian": BrownianNoiseGenerator,
    "laplacian": LaplacianNoiseGenerator,
    "studentt": StudentTNoiseGenerator,
    "wavelet": WaveletNoiseGenerator,
    "wavelet": WaveletNoiseGenerator,
    "perlin": PerlinNoiseGenerator,
}

NOISE_GENERATOR_NAMES = tuple(NOISE_GENERATOR_CLASSES.keys())

@cast_fp64
def prepare_noise(latent_image, seed, noise_type, noise_inds=None, alpha=1.0, k=1.0): # From `sample.py`
    #optional arg skip can be used to skip and discard x number of noise generations for a given seed
    noise_func = NOISE_GENERATOR_CLASSES.get(noise_type)(x=latent_image, seed=seed)

    if noise_type == "fractal":
        noise_func.alpha = alpha
        noise_func.k = k

    # from here until return is very similar to comfy/sample.py 
    if noise_inds is None:
        return noise_func(x=latent_image)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = noise_func(size = [1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, device=latent_image.device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

