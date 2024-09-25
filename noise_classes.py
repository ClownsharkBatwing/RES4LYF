import torch
from torch import nn, Tensor, Generator, lerp
from torch.nn.functional import unfold
import torch.nn.functional as F
from typing import Callable, Tuple
from math import pi
from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler
from torch.distributions import StudentT, Laplace
import numpy as np
import pywt
import functools



class PrecisionTool:
    def __init__(self, cast_type='fp64'):
        self.cast_type = cast_type

    def cast_tensor(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.cast_type not in ['fp64', 'fp32', 'fp16']:
                return func(*args, **kwargs)
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
            
        # Recursive function to cast tensors in nested dictionaries
            def cast_and_move_to_device(data):
                if torch.is_tensor(data):
                    if self.cast_type == 'fp64':
                        return data.to(torch.float64).to(target_device)
                    elif self.cast_type == 'fp32':
                        return data.to(torch.float32).to(target_device)
                    elif self.cast_type == 'fp16':
                        return data.to(torch.float16).to(target_device)
                elif isinstance(data, dict):
                    return {k: cast_and_move_to_device(v) for k, v in data.items()}
                return data

            # Cast all tensor arguments and move them to the target device
            new_args = [cast_and_move_to_device(arg) for arg in args]
            new_kwargs = {k: cast_and_move_to_device(v) for k, v in kwargs.items()}
            
            return func(*new_args, **new_kwargs)
        return wrapper

    def set_cast_type(self, new_value):
        if new_value in ['fp64', 'fp32', 'fp16']:
            self.cast_type = new_value
        else:
            self.cast_type = 'fp64'

precision_tool = PrecisionTool(cast_type='fp64')


def noise_generator_factory(cls, **fixed_params):
    def create_instance(**kwargs):
        # Combine fixed_params with kwargs, giving priority to kwargs
        params = {**fixed_params, **kwargs}
        return cls(**params)
    return create_instance

def like(x):
    return {'size': x.shape, 'dtype': x.dtype, 'layout': x.layout, 'device': x.device}

def scale_to_range(x, scaled_min = -1.73, scaled_max = 1.73): #1.73 is roughly the square root of 3
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
    #def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
    #             sigma=14.614642, sigma_next=0.0291675): 
    #    self.update(sigma=sigma, sigma_next=sigma_next)
    #    super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, sigma=None, sigma_next=None, **kwargs):
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
    
from opensimplex import OpenSimplex

class SimplexNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 scale=0.01):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)
        self.noise = OpenSimplex(seed=seed)
        self.scale = scale
        
    def __call__(self, *, scale=None, **kwargs):
        self.update(scale=scale)
        
        b, c, h, w = self.size

        noise_array = self.noise.noise3array(np.arange(w),np.arange(h),np.arange(c))
        self.noise = OpenSimplex(seed=self.noise.get_seed()+1)
        
        noise_tensor = torch.from_numpy(noise_array).to(self.device)
        noise_tensor = torch.unsqueeze(noise_tensor, dim=0)
        
        return noise_tensor / noise_tensor.std()
        #return normalize(scale_to_range(noise_tensor))



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


class InterpolatedPyramidNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 discount=0.7, mode='nearest-exact'):
        self.update(discount=discount, mode=mode)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, discount=None, mode=None, **kwargs):
        self.update(discount=discount, mode=mode)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        #u = nn.Upsample(size=(orig_h, orig_w), mode=self.mode).to(self.device)
        #u = nn.functional.interpolate(size=(orig_h, orig_w), mode=self.mode)

        noise = ((torch.rand(size=self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator) - 0.5) * 2 * 1.73)
        multipliers = [1]

        for i in range(4):
            r = torch.rand(1, device=self.device, generator=self.generator).item() * 2 + 2
            h, w = min(orig_h * 15, int(h * (r ** i))), min(orig_w * 15, int(w * (r ** i)))

            new_noise = torch.randn((b, c, h, w), dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)
            #upsampled_noise = u(new_noise)
            upsampled_noise = nn.functional.interpolate(new_noise, size=(orig_h, orig_w), mode=self.mode)

            noise += upsampled_noise * self.discount ** i
            multipliers.append(        self.discount ** i)
            
            #if h <= 1 or w <= 1:
            if h >= orig_h * 15 or w >= orig_w * 15:
                break  # if resolution is too high
        
        noise = noise / sum([m ** 2 for m in multipliers]) ** 0.5 
        return noise #/ noise.std()

class CascadeBPyramidNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 levels=10, mode='nearest', size_range=[1,16]):
        self.update(epsilon=x, levels=levels, mode=mode, size_range=size_range)

        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, levels=10, mode='nearest', size_range=[1,16], **kwargs):
        self.update(levels=levels, mode=mode)

        b, c, h, w = self.size
        orig_h, orig_w = h, w

        epsilon = torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)
        multipliers = [1]
        for i in range(1, levels):
            m = 0.75 ** i

            h, w = int(epsilon.size(-2) // (2 ** i)), int(epsilon.size(-2) // (2 ** i))
            if size_range is None or (size_range[0] <= h <= size_range[1] or size_range[0] <= w <= size_range[1]):
                offset = torch.randn(epsilon.size(0), epsilon.size(1), h, w, device=self.device, generator=self.generator)
                epsilon = epsilon + torch.nn.functional.interpolate(offset, size=epsilon.shape[-2:], mode=self.mode) * m
                multipliers.append(m)

            if h <= 1 or w <= 1:
                break
        epsilon = epsilon / sum([m ** 2 for m in multipliers]) ** 0.5 #divides the epsilon tensor by the square root of the sum of the squared multipliers.

        return epsilon
    


    def _pyramid_noise(self, epsilon, size_range=None, levels=10, scale_mode='nearest'):
        epsilon = epsilon.clone()
        multipliers = [1]
        for i in range(1, levels):
            m = 0.75 ** i
            h, w = epsilon.size(-2) // (2 ** i), epsilon.size(-2) // (2 ** i)
            if size_range is None or (size_range[0] <= h <= size_range[1] or size_range[0] <= w <= size_range[1]):
                offset = torch.randn(epsilon.size(0), epsilon.size(1), h, w, device=self.device)
                epsilon = epsilon + torch.nn.functional.interpolate(offset, size=epsilon.shape[-2:],
                                                                    mode=scale_mode) * m
                multipliers.append(m)
            if h <= 1 or w <= 1:
                break
        epsilon = epsilon / sum([m ** 2 for m in multipliers]) ** 0.5
        # epsilon = epsilon / epsilon.std()
        return epsilon


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

        noise = StudentT(loc=self.loc, scale=self.scale, df=self.df).rsample(self.size)

        s = torch.quantile(noise.flatten(start_dim=1).abs(), 0.75, dim=-1)
        s = s.reshape(*s.shape, 1, 1, 1)
        noise = noise.clamp(-s, s)

        noise_latent = torch.copysign(torch.pow(torch.abs(noise), 0.5), noise).to(self.device)

        self.generator.manual_seed(self.generator.initial_seed() + 1)
        torch.random.set_rng_state(rng_state)
        return (noise_latent - noise_latent.mean()) / noise_latent.std()

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
        coeffs = pywt.wavedecn(torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator).to('cpu'), wavelet=self.wavelet, mode='periodization')
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
    
class GreenNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None):
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, **kwargs):
        b, c, h, w = self.size

        # Generate random noise
        noise = torch.randn(size=self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)
        
        # Generate frequency grid
        y_freq = torch.fft.fftfreq(h, 1/h, device=self.device)
        x_freq = torch.fft.fftfreq(w, 1/w, device=self.device)
        freq = torch.sqrt(y_freq[:, None]**2 + x_freq[None, :]**2).clamp(min=1e-10)

        # Create a mid-frequency emphasis for green noise (centered on mid frequencies)
        power = torch.sqrt(freq)
        power[0, 0] = 1  # Avoid division by zero

        # Apply the frequency emphasis
        noise_fft = torch.fft.fft2(noise)
        modified_fft = noise_fft / torch.sqrt(power)
        noise = torch.fft.ifft2(modified_fft).real

        # Normalize and return the latent noise
        #return noise / noise.std()
        return normalize(noise)


class VelvetNoiseGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, num_impulses=100):
        self.update(num_impulses=num_impulses)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, num_impulses=None, **kwargs):
        self.update(num_impulses=num_impulses)
        
        b, c, h, w = self.size
        noise = torch.zeros(size=self.size, dtype=self.dtype, layout=self.layout, device=self.device)
        
        # Randomly place impulses across the image
        for _ in range(self.num_impulses):
            # Generate random positions for the impulses
            i = torch.randint(0, h, (1,), generator=self.generator, device=self.device).item()
            j = torch.randint(0, w, (1,), generator=self.generator, device=self.device).item()
            # Generate random values for the impulses
            noise[..., i, j] = torch.randn((1,), generator=self.generator, device=self.device)

        #return noise / noise.std()
        return normalize(noise)

class LatentFilmGrainGenerator(NoiseGenerator):
    def __init__(self, x=None, size=None, dtype=None, layout=None, device=None, seed=42, generator=None, sigma_min=None, sigma_max=None, 
                 density=1.0, intensity=1.0, highlights=1.0, supersample_factor=4):
        # Initialize the generator with noise parameters
        self.update(density=density, intensity=intensity, highlights=highlights, supersample_factor=supersample_factor)
        super().__init__(x, size, dtype, layout, device, seed, generator, sigma_min, sigma_max)

    def __call__(self, *, density=None, intensity=None, highlights=None, supersample_factor=None, **kwargs):
        # Update parameters
        self.update(density=density, intensity=intensity, highlights=highlights, supersample_factor=supersample_factor)

        # Generate film grain in latent space
        latent_noise = self.apply_film_grain_to_latent()
        return latent_noise

    def apply_film_grain_to_latent(self):
        # Get latent space size
        b, c, h, w = self.size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate initial latent tensor
        latent_tensor = torch.randn(self.size, dtype=self.dtype, layout=self.layout, device=self.device, generator=self.generator)

        # Supersample latent space for higher resolution film grain
        supersampled_size = (h * self.supersample_factor, w * self.supersample_factor)
        latent_tensor = F.interpolate(latent_tensor, size=supersampled_size, mode="bilinear", align_corners=False)

        # Flatten the latent tensor to add noise to a subset of pixels
        latent_tensor_flat = latent_tensor.view(-1)  # Flatten into a 1D tensor
        num_pixels = int(self.density * latent_tensor_flat.size(0))

        # Select random indices for applying noise
        indices = torch.randint(0, latent_tensor_flat.size(0), (num_pixels,), dtype=torch.int64, device=device)
        noise_values = torch.randn(num_pixels, dtype=self.dtype, device=self.device, generator=self.generator) * self.intensity

        # Add noise to latent space
        latent_tensor_flat.index_add_(0, indices, noise_values)  # Index into the 1D tensor
        latent_tensor = latent_tensor_flat.view(b, c, *supersampled_size)  # Reshape back to original size

        # Downsample back to original latent space resolution
        latent_tensor = F.interpolate(latent_tensor, size=(h, w), mode="bilinear", align_corners=False)

        # Apply highlights (contrast adjustment in latent space)
        latent_tensor = latent_tensor * self.highlights

        return latent_tensor / latent_tensor.std()





NOISE_GENERATOR_CLASSES = {
    "fractal": FractalNoiseGenerator,
    #"pyramid": PyramidNoiseGenerator,
    "gaussian": GaussianNoiseGenerator,
    "uniform": UniformNoiseGenerator,
    "pyramid-cascade_B": CascadeBPyramidNoiseGenerator,
    "pyramid-interpolated": InterpolatedPyramidNoiseGenerator,
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
    "simplex": SimplexNoiseGenerator,
    "perlin": PerlinNoiseGenerator,
    #"green": GreenNoiseGenerator,
    #"velvet": VelvetNoiseGenerator,
    #"film_grain": LatentFilmGrainGenerator,
}

NOISE_GENERATOR_NAMES = tuple(NOISE_GENERATOR_CLASSES.keys())

@precision_tool.cast_tensor
def prepare_noise(latent_image, seed, noise_type, noise_inds=None, alpha=1.0, k=1.0): # From `sample.py`
    #optional arg skip can be used to skip and discard x number of noise generations for a given seed
    noise_func = NOISE_GENERATOR_CLASSES.get(noise_type)(x=latent_image, seed=seed, sigma_min=0.0291675, sigma_max=14.614642)

    if noise_type == "fractal":
        noise_func.alpha = alpha
        noise_func.k = k

    # from here until return is very similar to comfy/sample.py 
    if noise_inds is None:
        return noise_func(sigma=14.614642, sigma_next=0.0291675)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = noise_func(size = [1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, device=latent_image.device)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

