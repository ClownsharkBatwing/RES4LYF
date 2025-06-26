import torch
import numpy as np
from math import *
import builtins
from scipy.interpolate import CubicSpline
from scipy import special, stats
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math


from comfy.k_diffusion.sampling import get_sigmas_polyexponential, get_sigmas_karras
import comfy.samplers

from torch import Tensor, nn
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

from .res4lyf import RESplain
from .helper  import get_res4lyf_scheduler_list


def rescale_linear(input, input_min, input_max, output_min, output_max):
    output = ((input - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min;
    return output

class set_precision_sigmas:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "sigmas": ("SIGMAS", ),   
                    "precision": (["16", "32", "64"], ),
                    "set_default": ("BOOLEAN", {"default": False})
                     },
                }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("passthrough",)
    CATEGORY = "RES4LYF/precision"

    FUNCTION = "main"

    def main(self, precision="32", sigmas=None, set_default=False):
        match precision:
            case "16":
                if set_default is True:
                    torch.set_default_dtype(torch.float16)
                sigmas = sigmas.to(torch.float16)
            case "32":
                if set_default is True:
                    torch.set_default_dtype(torch.float32)
                sigmas = sigmas.to(torch.float32)
            case "64":
                if set_default is True:
                    torch.set_default_dtype(torch.float64)
                sigmas = sigmas.to(torch.float64)
        return (sigmas, )


class SimpleInterpolator(nn.Module):
    def __init__(self):
        super(SimpleInterpolator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_interpolator(model, sigma_schedule, steps, epochs=5000, lr=0.01):
    with torch.inference_mode(False):
        model = SimpleInterpolator()
        sigma_schedule = sigma_schedule.clone()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        x_train = torch.linspace(0, 1, steps=steps).unsqueeze(1)
        y_train = sigma_schedule.unsqueeze(1)

        # disable inference mode for training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # fwd pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    return model

def interpolate_sigma_schedule_model(sigma_schedule, target_steps):
    model = SimpleInterpolator()
    sigma_schedule = sigma_schedule.float().detach()

    # train on original sigma schedule
    trained_model = train_interpolator(model, sigma_schedule, len(sigma_schedule))

    # generate target steps for interpolation
    x_interpolated = torch.linspace(0, 1, target_steps).unsqueeze(1)

    # inference w/o gradients
    trained_model.eval()
    with torch.no_grad():
        interpolated_sigma = trained_model(x_interpolated).squeeze()

    return interpolated_sigma




class sigmas_interpolate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_0": ("SIGMAS", {"forceInput": True}),
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "mode": (["linear", "nearest", "polynomial", "exponential", "power", "model"],),
                "order": ("INT", {"default": 8, "min": 1,"max": 64,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS",)
    RETURN_NAMES = ("sigmas_0", "sigmas_1")
    CATEGORY = "RES4LYF/sigmas"
    



    def interpolate_sigma_schedule_poly(self, sigma_schedule, target_steps):
        order = self.order
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # orig steps (assuming even spacing)
        original_steps = np.linspace(0, 1, len(sigma_schedule_np))

        # fit polynomial of the given order
        coefficients = np.polyfit(original_steps, sigma_schedule_np, deg=order)

        # generate new steps where we want to interpolate the data
        target_steps_np = np.linspace(0, 1, target_steps)

        # eval polynomial at new steps
        interpolated_sigma_np = np.polyval(coefficients, target_steps_np)

        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)
        return interpolated_sigma

    def interpolate_sigma_schedule_constrained(self, sigma_schedule, target_steps):
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # orig steps
        original_steps = np.linspace(0, 1, len(sigma_schedule_np))

        # target steps for interpolation
        target_steps_np = np.linspace(0, 1, target_steps)

        # fit cubic spline with fixed start and end values
        cs = CubicSpline(original_steps, sigma_schedule_np, bc_type=((1, 0.0), (1, 0.0)))

        # eval spline at the target steps
        interpolated_sigma_np = cs(target_steps_np)

        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma
    
    def interpolate_sigma_schedule_exp(self, sigma_schedule, target_steps):
        # transform to log space
        log_sigma_schedule = torch.log(sigma_schedule)

        # define the original and target step ranges
        original_steps = torch.linspace(0, 1, steps=len(sigma_schedule))
        target_steps = torch.linspace(0, 1, steps=target_steps)

        # interpolate in log space
        interpolated_log_sigma = F.interpolate(
            log_sigma_schedule.unsqueeze(0).unsqueeze(0),  # Add fake batch and channel dimensions
            size=target_steps.shape[0],
            mode='linear',
            align_corners=True
        ).squeeze()

        # transform back to exponential space
        interpolated_sigma_schedule = torch.exp(interpolated_log_sigma)

        return interpolated_sigma_schedule
    
    def interpolate_sigma_schedule_power(self, sigma_schedule, target_steps):
        sigma_schedule_np = sigma_schedule.cpu().numpy()
        original_steps = np.linspace(1, len(sigma_schedule_np), len(sigma_schedule_np))

        # power regression using a log-log transformation
        log_x = np.log(original_steps)
        log_y = np.log(sigma_schedule_np)

        # linear regression on log-log data
        coefficients = np.polyfit(log_x, log_y, deg=1)  # degree 1 for linear fit in log-log space
        a = np.exp(coefficients[1])  # a = "b" = intercept (exp because of the log transform)
        b = coefficients[0]  # b = "m" = slope

        target_steps_np = np.linspace(1, len(sigma_schedule_np), target_steps)

        # power law prediction: y = a * x^b
        interpolated_sigma_np = a * (target_steps_np ** b)

        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma
            
    def interpolate_sigma_schedule_linear(self, sigma_schedule, target_steps):
        return F.interpolate(sigma_schedule.unsqueeze(0).unsqueeze(0), target_steps, mode='linear').squeeze(0).squeeze(0)

    def interpolate_sigma_schedule_nearest(self, sigma_schedule, target_steps):
        return F.interpolate(sigma_schedule.unsqueeze(0).unsqueeze(0), target_steps, mode='nearest').squeeze(0).squeeze(0)    
    
    def interpolate_nearest_neighbor(self, sigma_schedule, target_steps):
        original_steps = torch.linspace(0, 1, steps=len(sigma_schedule))
        target_steps = torch.linspace(0, 1, steps=target_steps)

        # interpolate original -> target steps using nearest neighbor
        indices = torch.searchsorted(original_steps, target_steps)
        indices = torch.clamp(indices, 0, len(sigma_schedule) - 1)  # clamp indices to valid range

        # set nearest neighbor via indices
        interpolated_sigma = sigma_schedule[indices]

        return interpolated_sigma


    def main(self, sigmas_0, sigmas_1, mode, order):

        self.order = order

        if   mode == "linear": 
            interpolate = self.interpolate_sigma_schedule_linear
        if   mode == "nearest": 
            interpolate = self.interpolate_nearest_neighbor
        elif mode == "polynomial":
            interpolate = self.interpolate_sigma_schedule_poly
        elif mode == "exponential":
            interpolate = self.interpolate_sigma_schedule_exp
        elif mode == "power":
            interpolate = self.interpolate_sigma_schedule_power
        elif mode == "model":
            with torch.inference_mode(False):
                interpolate = interpolate_sigma_schedule_model
        
        sigmas_0 = interpolate(sigmas_0, len(sigmas_1))
        return (sigmas_0, sigmas_1,)
    
class sigmas_noise_inversion:
    # flip sigmas for unsampling, and pad both fwd/rev directions with null bytes to disable noise scaling, etc from the model.
    # will cause model to return epsilon prediction instead of calculated denoised latent image.
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS",)
    RETURN_NAMES = ("sigmas_fwd","sigmas_rev",)
    CATEGORY = "RES4LYF/sigmas"
    DESCRIPTION = "For use with unsampling. Connect sigmas_fwd to the unsampling (first) node, and sigmas_rev to the sampling (second) node."
    
    def main(self, sigmas):
        sigmas = sigmas.clone().to(torch.float64)
        
        null = torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)
        sigmas_fwd = torch.flip(sigmas, dims=[0])
        sigmas_fwd = torch.cat([sigmas_fwd, null])
        
        sigmas_rev = torch.cat([null, sigmas])
        sigmas_rev = torch.cat([sigmas_rev, null])
        
        return (sigmas_fwd, sigmas_rev,)


def compute_sigma_next_variance_floor(sigma):
    return (-1 + torch.sqrt(1 + 4 * sigma)) / 2

class sigmas_variance_floor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    DESCRIPTION = ("Process a sigma schedule so that any steps that are too large for variance-locked SDE sampling are replaced with the maximum permissible value."
        "Will be very difficult to approach sigma = 0 due to the nature of the math, as steps become very small much below approximately sigma = 0.15 to 0.2.")
    
    def main(self, sigmas):
        dtype = sigmas.dtype
        sigmas = sigmas.clone().to(torch.float64)
        for i in range(len(sigmas) - 1):
            sigma_next = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
            
            if sigmas[i+1] < sigma_next and sigmas[i+1] > 0.0:
                print("swapped i+1 with sigma_next+0.001: ", sigmas[i+1], sigma_next + 0.001)
                sigmas[i+1] = sigma_next + 0.001
        return (sigmas.to(dtype),)


class sigmas_from_text:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, text):
        text_list = [float(val) for val in text.replace(",", " ").split()]
        #text_list = [float(val.strip()) for val in text.split(",")]

        sigmas = torch.tensor(text_list) #.to('cuda').to(torch.float64)
        
        return (sigmas,)



class sigmas_concatenate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (torch.cat((sigmas_1, sigmas_2.to(sigmas_1))),)

class sigmas_truncate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmas_until": ("INT", {"default": 10, "min": 0,"max": 1000,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, sigmas_until):
        sigmas = sigmas.clone()
        return (sigmas[:sigmas_until],)

class sigmas_start:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmas_until": ("INT", {"default": 10, "min": 0,"max": 1000,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, sigmas_until):
        sigmas = sigmas.clone()
        return (sigmas[sigmas_until:],)
        
class sigmas_split:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmas_start": ("INT", {"default": 0, "min": 0,"max": 1000,"step": 1}),
                "sigmas_end": ("INT", {"default": 1000, "min": 0,"max": 1000,"step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, sigmas_start, sigmas_end):
        sigmas = sigmas.clone()
        return (sigmas[sigmas_start:sigmas_end],)

        sigmas_stop_step = sigmas_end - sigmas_start
        return (sigmas[sigmas_start:][:sigmas_stop_step],)
    
class sigmas_pad:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "value": ("FLOAT", {"default": 0.0, "min": -10000,"max": 10000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, value):
        sigmas = sigmas.clone()
        return (torch.cat((sigmas, torch.tensor([value], dtype=sigmas.dtype))),)
    
class sigmas_unpad:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas):
        sigmas = sigmas.clone()
        return (sigmas[:-1],)

class sigmas_set_floor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "floor": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "new_floor": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "set_floor"

    CATEGORY = "RES4LYF/sigmas"

    def set_floor(self, sigmas, floor, new_floor):
        sigmas = sigmas.clone()
        sigmas[sigmas <= floor] = new_floor
        return (sigmas,)    
    
class sigmas_delete_below_floor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "floor": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "delete_below_floor"

    CATEGORY = "RES4LYF/sigmas"

    def delete_below_floor(self, sigmas, floor):
        sigmas = sigmas.clone()
        return (sigmas[sigmas >= floor],)    

class sigmas_delete_value:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "value": ("FLOAT", {"default": 0.0, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "delete_value"

    CATEGORY = "RES4LYF/sigmas"

    def delete_value(self, sigmas, value):
        return (sigmas[sigmas != value],) 

class sigmas_delete_consecutive_duplicates:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "delete_consecutive_duplicates"

    CATEGORY = "RES4LYF/sigmas"

    def delete_consecutive_duplicates(self, sigmas_1):
        mask = sigmas_1[:-1] != sigmas_1[1:]
        mask = torch.cat((mask, torch.tensor([True])))
        return (sigmas_1[mask],) 

class sigmas_cleanup:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "sigmin": ("FLOAT", {"default": 0.0291675, "min": 0,"max": 1000,"step": 0.01})
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "cleanup"

    CATEGORY = "RES4LYF/sigmas"

    def cleanup(self, sigmas, sigmin):
        sigmas_culled = sigmas[sigmas >= sigmin]
    
        mask = sigmas_culled[:-1] != sigmas_culled[1:]
        mask = torch.cat((mask, torch.tensor([True])))
        filtered_sigmas = sigmas_culled[mask]
        return (torch.cat((filtered_sigmas,torch.tensor([0]))),)

class sigmas_mult:
    def __init__(self):
        pass   

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "multiplier": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01})
            },
            "optional": {
                "sigmas2": ("SIGMAS", {"forceInput": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, multiplier, sigmas2=None):
        if sigmas2 is not None:
            return (sigmas * sigmas2 * multiplier,)
        else:
            return (sigmas * multiplier,)    

class sigmas_modulus:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "divisor": ("FLOAT", {"default": 1, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, divisor):
        return (sigmas % divisor,)
        
class sigmas_quotient:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "divisor": ("FLOAT", {"default": 1, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, divisor):
        return (sigmas // divisor,)

class sigmas_add:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "addend": ("FLOAT", {"default": 1, "min": -1000,"max": 1000,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, addend):
        return (sigmas + addend,)

class sigmas_power:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "power": ("FLOAT", {"default": 1, "min": -100,"max": 100,"step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, power):
        return (sigmas ** power,)

class sigmas_abs:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas):
        return (abs(sigmas),)

class sigmas2_mult:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (sigmas_1 * sigmas_2,)

class sigmas2_add:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_1": ("SIGMAS", {"forceInput": True}),
                "sigmas_2": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (sigmas_1 + sigmas_2,)

class sigmas_rescale:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 1.0, "min": -10000,"max": 10000,"step": 0.01}),
                "end": ("FLOAT", {"default": 0.0, "min": -10000,"max": 10000,"step": 0.01}),
                "sigmas": ("SIGMAS", ),
            },
            "optional": {
            }
        }
    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas_rescaled",)
    CATEGORY = "RES4LYF/sigmas"
    DESCRIPTION = ("Can be used to set denoise. Results are generally better than with the approach used by KSampler and most nodes with denoise values "
                   "(which slice the sigmas schedule according to step count, not the noise level). Will also flip the sigma schedule if the start and end values are reversed." 
                   )
      
    def main(self, start=0, end=-1, sigmas=None):

        s_out_1 = ((sigmas - sigmas.min()) * (start - end)) / (sigmas.max() - sigmas.min()) + end     
        
        return (s_out_1,)


class sigmas_math1:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "stop": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "trim": ("INT", {"default": 0, "min": -10000,"max": 0,"step": 1}),
                "x": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "y": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "z": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "f1": ("STRING", {"default": "s", "multiline": True}),
                "rescale" : ("BOOLEAN", {"default": False}),
                "max1": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min1": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            },
            "optional": {
                "a": ("SIGMAS", {"forceInput": False}),
                "b": ("SIGMAS", {"forceInput": False}),               
                "c": ("SIGMAS", {"forceInput": False}),
            }
        }
    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    def main(self, start=0, stop=0, trim=0, a=None, b=None, c=None, x=1.0, y=1.0, z=1.0, f1="s", rescale=False, min1=1.0, max1=1.0):
        if stop == 0:
            t_lens = [len(tensor) for tensor in [a, b, c] if tensor is not None]
            t_len = stop = min(t_lens) if t_lens else 0
        else:
            stop = stop + 1
            t_len = stop - start 
            
        stop = stop + trim
        t_len = t_len + trim
        
        t_a = t_b = t_c = None
        if a is not None:
            t_a = a[start:stop]
        if b is not None:
            t_b = b[start:stop]
        if c is not None:
            t_c = c[start:stop]               
            
        t_s = torch.arange(0.0, t_len)
    
        t_x = torch.full((t_len,), x)
        t_y = torch.full((t_len,), y)
        t_z = torch.full((t_len,), z)
        eval_namespace = {"__builtins__": None, "round": builtins.round, "np": np, "a": t_a, "b": t_b, "c": t_c, "x": t_x, "y": t_y, "z": t_z, "s": t_s, "torch": torch}
        eval_namespace.update(np.__dict__)
        
        s_out_1 = eval(f1, eval_namespace)
        
        if rescale == True:
            s_out_1 = ((s_out_1 - min(s_out_1)) * (max1 - min1)) / (max(s_out_1) - min(s_out_1)) + min1     
        
        return (s_out_1,)

class sigmas_math3:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "stop": ("INT", {"default": 0, "min": 0,"max": 10000,"step": 1}),
                "trim": ("INT", {"default": 0, "min": -10000,"max": 0,"step": 1}),
            },
            "optional": {
                "a": ("SIGMAS", {"forceInput": False}),
                "b": ("SIGMAS", {"forceInput": False}),               
                "c": ("SIGMAS", {"forceInput": False}),
                "x": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "y": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "z": ("FLOAT", {"default": 1, "min": -10000,"max": 10000,"step": 0.01}),
                "f1": ("STRING", {"default": "s", "multiline": True}),
                "rescale1" : ("BOOLEAN", {"default": False}),
                "max1": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min1": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "f2": ("STRING", {"default": "s", "multiline": True}),
                "rescale2" : ("BOOLEAN", {"default": False}),
                "max2": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min2": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "f3": ("STRING", {"default": "s", "multiline": True}),
                "rescale3" : ("BOOLEAN", {"default": False}),
                "max3": ("FLOAT", {"default": 14.614642, "min": -10000,"max": 10000,"step": 0.01}),
                "min3": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            }
        }
    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS","SIGMAS")
    CATEGORY = "RES4LYF/sigmas"
    def main(self, start=0, stop=0, trim=0, a=None, b=None, c=None, x=1.0, y=1.0, z=1.0, f1="s", f2="s", f3="s", rescale1=False, rescale2=False, rescale3=False, min1=1.0, max1=1.0, min2=1.0, max2=1.0, min3=1.0, max3=1.0):
        if stop == 0:
            t_lens = [len(tensor) for tensor in [a, b, c] if tensor is not None]
            t_len = stop = min(t_lens) if t_lens else 0
        else:
            stop = stop + 1
            t_len = stop - start 
            
        stop = stop + trim
        t_len = t_len + trim
        
        t_a = t_b = t_c = None
        if a is not None:
            t_a = a[start:stop]
        if b is not None:
            t_b = b[start:stop]
        if c is not None:
            t_c = c[start:stop]               
            
        t_s = torch.arange(0.0, t_len)
    
        t_x = torch.full((t_len,), x)
        t_y = torch.full((t_len,), y)
        t_z = torch.full((t_len,), z)
        eval_namespace = {"__builtins__": None, "np": np, "a": t_a, "b": t_b, "c": t_c, "x": t_x, "y": t_y, "z": t_z, "s": t_s, "torch": torch}
        eval_namespace.update(np.__dict__)
        
        s_out_1 = eval(f1, eval_namespace)
        s_out_2 = eval(f2, eval_namespace)
        s_out_3 = eval(f3, eval_namespace)
        
        if rescale1 == True:
            s_out_1 = ((s_out_1 - min(s_out_1)) * (max1 - min1)) / (max(s_out_1) - min(s_out_1)) + min1
        if rescale2 == True:
            s_out_2 = ((s_out_2 - min(s_out_2)) * (max2 - min2)) / (max(s_out_2) - min(s_out_2)) + min2
        if rescale3 == True:
            s_out_3 = ((s_out_3 - min(s_out_3)) * (max3 - min3)) / (max(s_out_3) - min(s_out_3)) + min3        
        
        return s_out_1, s_out_2, s_out_3

class sigmas_iteration_karras:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps_up": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "steps_down": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "rho_up": ("FLOAT", {"default": 3, "min": -10000,"max": 10000,"step": 0.01}),
                "rho_down": ("FLOAT", {"default": 4, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_start": ("FLOAT", {"default":0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "s_max": ("FLOAT", {"default": 2, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_end": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            },
            "optional": {
                "momentums": ("SIGMAS", {"forceInput": False}),
                "sigmas": ("SIGMAS", {"forceInput": False}),             
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("momentums","sigmas")
    CATEGORY = "RES4LYF/schedulers"
    
    def main(self, steps_up, steps_down, rho_up, rho_down, s_min_start, s_max, s_min_end, sigmas=None, momentums=None):
        s_up = get_sigmas_karras(steps_up, s_min_start, s_max, rho_up)
        s_down = get_sigmas_karras(steps_down, s_min_end, s_max, rho_down) 
        s_up = s_up[:-1]
        s_down = s_down[:-1]  
        s_up = torch.flip(s_up, dims=[0])
        sigmas_new = torch.cat((s_up, s_down), dim=0)
        momentums_new = torch.cat((s_up, -1*s_down), dim=0)
        
        if sigmas is not None:
            sigmas = torch.cat([sigmas, sigmas_new])
        else:
            sigmas = sigmas_new
            
        if momentums is not None:
            momentums = torch.cat([momentums, momentums_new])
        else:
            momentums = momentums_new
        
        return (momentums,sigmas) 
 
class sigmas_iteration_polyexp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps_up": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "steps_down": ("INT", {"default": 30, "min": 0,"max": 10000,"step": 1}),
                "rho_up": ("FLOAT", {"default": 0.6, "min": -10000,"max": 10000,"step": 0.01}),
                "rho_down": ("FLOAT", {"default": 0.8, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_start": ("FLOAT", {"default":0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
                "s_max": ("FLOAT", {"default": 2, "min": -10000,"max": 10000,"step": 0.01}),
                "s_min_end": ("FLOAT", {"default": 0.0291675, "min": -10000,"max": 10000,"step": 0.01}),
            },
            "optional": {
                "momentums": ("SIGMAS", {"forceInput": False}),
                "sigmas": ("SIGMAS", {"forceInput": False}),             
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("momentums","sigmas")
    CATEGORY = "RES4LYF/schedulers"
    
    def main(self, steps_up, steps_down, rho_up, rho_down, s_min_start, s_max, s_min_end, sigmas=None, momentums=None):
        s_up = get_sigmas_polyexponential(steps_up, s_min_start, s_max, rho_up)
        s_down = get_sigmas_polyexponential(steps_down, s_min_end, s_max, rho_down) 
        s_up = s_up[:-1]
        s_down = s_down[:-1]
        s_up = torch.flip(s_up, dims=[0])
        sigmas_new = torch.cat((s_up, s_down), dim=0)
        momentums_new = torch.cat((s_up, -1*s_down), dim=0)

        if sigmas is not None:
            sigmas = torch.cat([sigmas, sigmas_new])
        else:
            sigmas = sigmas_new

        if momentums is not None:
            momentums = torch.cat([momentums, momentums_new])
        else:
            momentums = momentums_new

        return (momentums,sigmas) 

class tan_scheduler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 0,"max": 100000,"step": 1}),
                "offset": ("FLOAT", {"default": 20, "min": 0,"max": 100000,"step": 0.1}),
                "slope": ("FLOAT", {"default": 20, "min": -100000,"max": 100000,"step": 0.1}),
                "start": ("FLOAT", {"default": 20, "min": -100000,"max": 100000,"step": 0.1}),
                "end": ("FLOAT", {"default": 20, "min": -100000,"max": 100000,"step": 0.1}),
                "sgm" : ("BOOLEAN", {"default": False}),
                "pad" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/schedulers"
    
    def main(self, steps, slope, offset, start, end, sgm, pad):
        smax = ((2/pi)*atan(-slope*(0-offset))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-offset))+1)/2

        srange = smax-smin
        sscale = start - end
        
        if sgm:
            steps+=1

        sigmas = [  ( (((2/pi)*atan(-slope*(x-offset))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        if sgm:
            sigmas = sigmas[:-1]
        if pad:
            sigmas = torch.tensor(sigmas+[0])
        else:
            sigmas = torch.tensor(sigmas)
        return (sigmas,)

class tan_scheduler_2stage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "midpoint": ("INT", {"default": 20, "min": 0,"max": 100000,"step": 1}),
                "pivot_1": ("INT", {"default": 10, "min": 0,"max": 100000,"step": 1}),
                "pivot_2": ("INT", {"default": 30, "min": 0,"max": 100000,"step": 1}),
                "slope_1": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.1}),
                "slope_2": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.1}),
                "start": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.1}),
                "middle": ("FLOAT", {"default": 0.5, "min": -100000,"max": 100000,"step": 0.1}),
                "end": ("FLOAT", {"default": 0.0, "min": -100000,"max": 100000,"step": 0.1}),
                "pad" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def get_tan_sigmas(self, steps, slope, pivot, start, end):
        smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

        srange = smax-smin
        sscale = start - end

        sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        return sigmas

    def main(self, steps, midpoint, start, middle, end, pivot_1, pivot_2, slope_1, slope_2, pad):
        steps += 2
        stage_2_len = steps - midpoint
        stage_1_len = steps - stage_2_len

        tan_sigmas_1 = self.get_tan_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
        tan_sigmas_2 = self.get_tan_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
        
        tan_sigmas_1 = tan_sigmas_1[:-1]
        if pad:
            tan_sigmas_2 = tan_sigmas_2+[0]

        tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

        return (tan_sigmas,)

class tan_scheduler_2stage_simple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "pivot_1": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "pivot_2": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "slope_1": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "slope_2": ("FLOAT", {"default": 1, "min": -100000,"max": 100000,"step": 0.01}),
                "start": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.01}),
                "middle": ("FLOAT", {"default": 0.5, "min": -100000,"max": 100000,"step": 0.01}),
                "end": ("FLOAT", {"default": 0.0, "min": -100000,"max": 100000,"step": 0.01}),
                "pad" : ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def get_tan_sigmas(self, steps, slope, pivot, start, end):
        smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

        srange = smax-smin
        sscale = start - end

        sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        return sigmas

    def main(self, steps, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False, model_sampling=None):
        steps += 2

        midpoint = int( (steps*pivot_1 + steps*pivot_2) / 2 )
        pivot_1 = int(steps * pivot_1)
        pivot_2 = int(steps * pivot_2)

        slope_1 = slope_1 / (steps/40)
        slope_2 = slope_2 / (steps/40)

        stage_2_len = steps - midpoint
        stage_1_len = steps - stage_2_len

        tan_sigmas_1 = self.get_tan_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
        tan_sigmas_2 = self.get_tan_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
        
        tan_sigmas_1 = tan_sigmas_1[:-1]
        if pad:
            tan_sigmas_2 = tan_sigmas_2+[0]

        tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

        return (tan_sigmas,)
    
class linear_quadratic_advanced:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.01}),
                "inflection_percent": ("FLOAT", {"default": 0.5, "min": 0,"max": 1,"step": 0.01}),
            },
            # "optional": {
            # }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def main(self, steps, denoise, inflection_percent, model=None):
        sigmas = get_sigmas(model, "linear_quadratic", steps, denoise, inflection_percent)

        return (sigmas, )


class constant_scheduler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 0,"max": 100000,"step": 1}),
                "value_start": ("FLOAT", {"default": 1.0, "min": -100000,"max": 100000,"step": 0.01}),
                "value_end": ("FLOAT", {"default": 0.0, "min": -100000,"max": 100000,"step": 0.01}),
                "cutoff_percent": ("FLOAT", {"default": 1.0, "min": 0,"max": 1,"step": 0.01}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "RES4LYF/schedulers"

    def main(self, steps, value_start, value_end, cutoff_percent):
        sigmas = torch.ones(steps + 1) * value_start
        cutoff_step = int(round(steps * cutoff_percent)) + 1
        sigmas = torch.concat((sigmas[:cutoff_step], torch.ones(steps + 1 - cutoff_step) * value_end), dim=0)

        return (sigmas,)
    
    
    



class ClownScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "pad_start_value":      ("FLOAT",                                     {"default": 0.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "start_value":          ("FLOAT",                                     {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "end_value":            ("FLOAT",                                     {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "pad_end_value":        ("FLOAT",                                     {"default": 0.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "scheduler":            (["constant"] + get_res4lyf_scheduler_list(), {"default": "beta57"},),
                "scheduler_start_step": ("INT",                                       {"default": 0,   "min":  0,        "max": 10000}),
                "scheduler_end_step":   ("INT",                                       {"default": 30,  "min": -1,        "max": 10000}),
                "total_steps":          ("INT",                                       {"default": 100, "min": -1,        "max": 10000}),
                "flip_schedule":        ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "model":                ("MODEL", ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/schedulers"

    def create_callback(self, **kwargs):
        def callback(model):
            kwargs["model"] = model  
            schedule, = self.prepare_schedule(**kwargs)
            return schedule
        return callback

    def main(self,
            model                        = None,
            pad_start_value      : float = 1.0,
            start_value          : float = 0.0,
            end_value            : float = 1.0,
            pad_end_value                = None,
            denoise              : int   = 1.0,
            scheduler                    = None,
            scheduler_start_step : int   = 0,
            scheduler_end_step   : int   = 30,
            total_steps          : int   = 60,
            flip_schedule                = False,
            ) -> Tuple[Tensor]:
        
        if model is None:
            callback = self.create_callback(pad_start_value = pad_start_value,
                                            start_value     = start_value,
                                            end_value       = end_value,
                                            pad_end_value   = pad_end_value,
                                            
                                            scheduler       = scheduler,
                                            start_step      = scheduler_start_step,
                                            end_step        = scheduler_end_step,
                                            flip_schedule   = flip_schedule,
                                            )
        else:
            default_dtype  = torch.float64
            default_device = torch.device("cuda") 
            
            if scheduler_end_step == -1:
                scheduler_total_steps = total_steps - scheduler_start_step
            else:
                scheduler_total_steps = scheduler_end_step - scheduler_start_step
            
            if total_steps == -1:
                total_steps = scheduler_start_step + scheduler_end_step
            
            end_pad_steps = total_steps - scheduler_end_step
            
            if scheduler != "constant":
                values     = get_sigmas(model, scheduler, scheduler_total_steps, denoise).to(dtype=default_dtype, device=default_device) 
                values     = ((values - values.min()) * (start_value - end_value))   /   (values.max() - values.min())   +   end_value
            else:
                values = torch.linspace(start_value, end_value, scheduler_total_steps, dtype=default_dtype, device=default_device)
            
            if flip_schedule:
                values = torch.flip(values, dims=[0])
            
            prepend    = torch.full((scheduler_start_step,),  pad_start_value, dtype=default_dtype, device=default_device)
            postpend   = torch.full((end_pad_steps,),         pad_end_value,   dtype=default_dtype, device=default_device)
            
            values     = torch.cat((prepend, values, postpend), dim=0)

        #ositive[0][1]['callback_regional'] = callback
        
        return (values,)



    def prepare_schedule(self,
                                model                    = None,
                                pad_start_value  : float = 1.0,
                                start_value      : float = 0.0,
                                end_value        : float = 1.0,
                                pad_end_value            = None,
                                weight_scheduler         = None,
                                start_step       : int   = 0,
                                end_step         : int   = 30,
                                flip_schedule            = False,
                                ) -> Tuple[Tensor]:

        default_dtype  = torch.float64
        default_device = torch.device("cuda") 
        
        return (None,)




def get_sigmas_simple_exponential(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    sigs = torch.FloatTensor(sigs)
    exp = torch.exp(torch.log(torch.linspace(1, 0, steps + 1)))
    return sigs * exp

extra_schedulers = {
    "simple_exponential": get_sigmas_simple_exponential
}



def get_sigmas(model, scheduler, steps, denoise, shift=0.0, lq_inflection_percent=0.5): #adapted from comfyui
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return (torch.FloatTensor([]),)
        total_steps = int(steps/denoise)

    try:
        model_sampling = model.get_model_object("model_sampling")
    except:
        if hasattr(model, "model"):
            model_sampling = model.model.model_sampling
        elif hasattr(model, "inner_model"):
            model_sampling = model.inner_model.inner_model.model_sampling
        else:
            raise Exception("get_sigmas: Could not get model_sampling")

    if shift > 1e-6:
        import copy
        model_sampling = copy.deepcopy(model_sampling)
        model_sampling.set_parameters(shift=shift)
        RESplain("model_sampling shift manually set to " + str(shift), debug=True)
    
    if scheduler == "beta57":
        sigmas = comfy.samplers.beta_scheduler(model_sampling, total_steps, alpha=0.5, beta=0.7).cpu()
    elif scheduler == "linear_quadratic":
        linear_steps = int(total_steps * lq_inflection_percent)
        sigmas = comfy.samplers.linear_quadratic_schedule(model_sampling, total_steps, threshold_noise=0.025, linear_steps=linear_steps).cpu()
    else:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps).cpu()
    
    sigmas = sigmas[-(steps + 1):]
    return sigmas

#/// Adam Kormendi /// Inspired from Unreal Engine Maths ///


# Sigmoid Function
class sigmas_sigmoid:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "variant": (["logistic", "tanh", "softsign", "hardswish", "mish", "swish"], {"default": "logistic"}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, variant, gain, offset, normalize_output):
        # Apply gain and offset
        x = gain * (sigmas + offset)
        
        if variant == "logistic":
            result = 1.0 / (1.0 + torch.exp(-x))
        elif variant == "tanh":
            result = torch.tanh(x)
        elif variant == "softsign":
            result = x / (1.0 + torch.abs(x))
        elif variant == "hardswish":
            result = x * torch.minimum(torch.maximum(x + 3, torch.zeros_like(x)), torch.tensor(6.0)) / 6.0
        elif variant == "mish":
            result = x * torch.tanh(torch.log(1.0 + torch.exp(x)))
        elif variant == "swish":
            result = x * torch.sigmoid(x)
        
        if normalize_output:
            # Normalize to [min(sigmas), max(sigmas)]
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Easing Function -----
class sigmas_easing:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "easing_type": (["sine", "quad", "cubic", "quart", "quint", "expo", "circ", 
                                 "back", "elastic", "bounce"], {"default": "cubic"}),
                "easing_mode": (["in", "out", "in_out"], {"default": "in_out"}),
                "normalize_input": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, easing_type, easing_mode, normalize_input, normalize_output, strength):
        # Normalize input to [0, 1] if requested
        if normalize_input:
            t = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        else:
            t = torch.clamp(sigmas, 0.0, 1.0)
        
        # Apply strength
        t_orig = t.clone()
        t = t ** strength
            
        # Apply easing function based on type and mode
        if easing_mode == "in":
            result = self._ease_in(t, easing_type)
        elif easing_mode == "out":
            result = self._ease_out(t, easing_type)
        else:  # in_out
            result = self._ease_in_out(t, easing_type)
            
        # Normalize output if requested
        if normalize_output:
            if normalize_input:
                result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            else:
                result = ((result - result.min()) / (result.max() - result.min()))
                
        return (result,)
    
    def _ease_in(self, t, easing_type):
        if easing_type == "sine":
            return 1 - torch.cos((t * math.pi) / 2)
        elif easing_type == "quad":
            return t * t
        elif easing_type == "cubic":
            return t * t * t
        elif easing_type == "quart":
            return t * t * t * t
        elif easing_type == "quint":
            return t * t * t * t * t
        elif easing_type == "expo":
            return torch.where(t == 0, torch.zeros_like(t), torch.pow(2, 10 * t - 10))
        elif easing_type == "circ":
            return 1 - torch.sqrt(1 - torch.pow(t, 2))
        elif easing_type == "back":
            c1 = 1.70158
            c3 = c1 + 1
            return c3 * t * t * t - c1 * t * t
        elif easing_type == "elastic":
            c4 = (2 * math.pi) / 3
            return torch.where(
                t == 0, 
                torch.zeros_like(t),
                torch.where(
                    t == 1,
                    torch.ones_like(t),
                    -torch.pow(2, 10 * t - 10) * torch.sin((t * 10 - 10.75) * c4)
                )
            )
        elif easing_type == "bounce":
            return 1 - self._ease_out_bounce(1 - t)
    
    def _ease_out(self, t, easing_type):
        if easing_type == "sine":
            return torch.sin((t * math.pi) / 2)
        elif easing_type == "quad":
            return 1 - (1 - t) * (1 - t)
        elif easing_type == "cubic":
            return 1 - torch.pow(1 - t, 3)
        elif easing_type == "quart":
            return 1 - torch.pow(1 - t, 4)
        elif easing_type == "quint":
            return 1 - torch.pow(1 - t, 5)
        elif easing_type == "expo":
            return torch.where(t == 1, torch.ones_like(t), 1 - torch.pow(2, -10 * t))
        elif easing_type == "circ":
            return torch.sqrt(1 - torch.pow(t - 1, 2))
        elif easing_type == "back":
            c1 = 1.70158
            c3 = c1 + 1
            return 1 + c3 * torch.pow(t - 1, 3) + c1 * torch.pow(t - 1, 2)
        elif easing_type == "elastic":
            c4 = (2 * math.pi) / 3
            return torch.where(
                t == 0, 
                torch.zeros_like(t),
                torch.where(
                    t == 1,
                    torch.ones_like(t),
                    torch.pow(2, -10 * t) * torch.sin((t * 10 - 0.75) * c4) + 1
                )
            )
        elif easing_type == "bounce":
            return self._ease_out_bounce(t)
    
    def _ease_in_out(self, t, easing_type):
        if easing_type == "sine":
            return -(torch.cos(math.pi * t) - 1) / 2
        elif easing_type == "quad":
            return torch.where(t < 0.5, 2 * t * t, 1 - torch.pow(-2 * t + 2, 2) / 2)
        elif easing_type == "cubic":
            return torch.where(t < 0.5, 4 * t * t * t, 1 - torch.pow(-2 * t + 2, 3) / 2)
        elif easing_type == "quart":
            return torch.where(t < 0.5, 8 * t * t * t * t, 1 - torch.pow(-2 * t + 2, 4) / 2)
        elif easing_type == "quint":
            return torch.where(t < 0.5, 16 * t * t * t * t * t, 1 - torch.pow(-2 * t + 2, 5) / 2)
        elif easing_type == "expo":
            return torch.where(
                t < 0.5, 
                torch.pow(2, 20 * t - 10) / 2,
                (2 - torch.pow(2, -20 * t + 10)) / 2
            )
        elif easing_type == "circ":
            return torch.where(
                t < 0.5,
                (1 - torch.sqrt(1 - torch.pow(2 * t, 2))) / 2,
                (torch.sqrt(1 - torch.pow(-2 * t + 2, 2)) + 1) / 2
            )
        elif easing_type == "back":
            c1 = 1.70158
            c2 = c1 * 1.525
            return torch.where(
                t < 0.5,
                (torch.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2,
                (torch.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
            )
        elif easing_type == "elastic":
            c5 = (2 * math.pi) / 4.5
            return torch.where(
                t < 0.5,
                -(torch.pow(2, 20 * t - 10) * torch.sin((20 * t - 11.125) * c5)) / 2,
                (torch.pow(2, -20 * t + 10) * torch.sin((20 * t - 11.125) * c5)) / 2 + 1
            )
        elif easing_type == "bounce":
            return torch.where(
                t < 0.5,
                (1 - self._ease_out_bounce(1 - 2 * t)) / 2,
                (1 + self._ease_out_bounce(2 * t - 1)) / 2
            )
    
    def _ease_out_bounce(self, t):
        n1 = 7.5625
        d1 = 2.75
        
        mask1 = t < 1 / d1
        mask2 = t < 2 / d1
        mask3 = t < 2.5 / d1
        
        result = torch.zeros_like(t)
        result = torch.where(mask1, n1 * t * t, result)
        result = torch.where(mask2 & ~mask1, n1 * (t - 1.5 / d1) * (t - 1.5 / d1) + 0.75, result)
        result = torch.where(mask3 & ~mask2, n1 * (t - 2.25 / d1) * (t - 2.25 / d1) + 0.9375, result)
        result = torch.where(~mask3, n1 * (t - 2.625 / d1) * (t - 2.625 / d1) + 0.984375, result)
        
        return result

# -----  Hyperbolic Function -----
class sigmas_hyperbolic:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "function": (["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"], {"default": "tanh"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, function, scale, normalize_output):
        # Apply scaling
        x = sigmas * scale
        
        if function == "sinh":
            result = torch.sinh(x)
        elif function == "cosh":
            result = torch.cosh(x)
        elif function == "tanh":
            result = torch.tanh(x)
        elif function == "asinh":
            result = torch.asinh(x)
        elif function == "acosh":
            # Domain of acosh is [1, inf)
            result = torch.acosh(torch.clamp(x, min=1.0))
        elif function == "atanh":
            # Domain of atanh is (-1, 1)
            result = torch.atanh(torch.clamp(x, min=-0.99, max=0.99))
        
        if normalize_output:
            # Normalize to [min(sigmas), max(sigmas)]
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Gaussian Distribution Function -----
class sigmas_gaussian:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "mean": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "std": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "operation": (["pdf", "cdf", "inverse_cdf", "transform", "modulate"], {"default": "transform"}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, mean, std, operation, normalize_output):
        # Standardize values (z-score)
        z = (sigmas - sigmas.mean()) / sigmas.std()
        
        if operation == "pdf":
            # Probability density function
            result = (1 / (std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((sigmas - mean) / std) ** 2)
        elif operation == "cdf":
            # Cumulative distribution function
            result = 0.5 * (1 + torch.erf((sigmas - mean) / (std * math.sqrt(2))))
        elif operation == "inverse_cdf":
            # Inverse CDF (quantile function)
            # First normalize to [0.01, 0.99] to avoid numerical issues
            normalized = ((sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())) * 0.98 + 0.01
            result = mean + std * torch.sqrt(2) * torch.erfinv(2 * normalized - 1)
        elif operation == "transform":
            # Transform to Gaussian distribution with specified mean and std
            result = z * std + mean
        elif operation == "modulate":
            # Modulate with a Gaussian curve centered at mean
            result = sigmas * torch.exp(-0.5 * ((sigmas - mean) / std) ** 2)
        
        if normalize_output:
            # Normalize to [min(sigmas), max(sigmas)]
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Percentile Function -----
class sigmas_percentile:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "percentile_min": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 49.0, "step": 0.1}),
                "percentile_max": ("FLOAT", {"default": 95.0, "min": 51.0, "max": 100.0, "step": 0.1}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "clip_outliers": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, percentile_min, percentile_max, target_min, target_max, clip_outliers):
        # Convert to numpy for percentile computation
        sigmas_np = sigmas.cpu().numpy()
        
        # Compute percentiles
        p_min = np.percentile(sigmas_np, percentile_min)
        p_max = np.percentile(sigmas_np, percentile_max)
        
        # Convert back to tensor
        p_min = torch.tensor(p_min, device=sigmas.device, dtype=sigmas.dtype)
        p_max = torch.tensor(p_max, device=sigmas.device, dtype=sigmas.dtype)
        
        # Map values from [p_min, p_max] to [target_min, target_max]
        if clip_outliers:
            sigmas_clipped = torch.clamp(sigmas, p_min, p_max)
            result = ((sigmas_clipped - p_min) / (p_max - p_min)) * (target_max - target_min) + target_min
        else:
            result = ((sigmas - p_min) / (p_max - p_min)) * (target_max - target_min) + target_min
            
        return (result,)

# ----- Kernel Smooth Function -----
class sigmas_kernel_smooth:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "kernel": (["gaussian", "box", "triangle", "epanechnikov", "cosine"], {"default": "gaussian"}),
                "kernel_size": ("INT", {"default": 5, "min": 3, "max": 51, "step": 2}),  # Must be odd
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, kernel, kernel_size, sigma):
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Define kernel weights
        if kernel == "gaussian":
            # Gaussian kernel
            kernel_1d = self._gaussian_kernel(kernel_size, sigma)
        elif kernel == "box":
            # Box (uniform) kernel
            kernel_1d = torch.ones(kernel_size, device=sigmas.device, dtype=sigmas.dtype) / kernel_size
        elif kernel == "triangle":
            # Triangle kernel
            x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=sigmas.device, dtype=sigmas.dtype)
            kernel_1d = (1.0 - torch.abs(x) / (kernel_size//2))
            kernel_1d = kernel_1d / kernel_1d.sum()
        elif kernel == "epanechnikov":
            # Epanechnikov kernel
            x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=sigmas.device, dtype=sigmas.dtype)
            x = x / (kernel_size//2)  # Scale to [-1, 1]
            kernel_1d = 0.75 * (1 - x**2)
            kernel_1d = kernel_1d / kernel_1d.sum()
        elif kernel == "cosine":
            # Cosine kernel
            x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=sigmas.device, dtype=sigmas.dtype)
            x = x / (kernel_size//2) * (math.pi/2)  # Scale to [-π/2, π/2]
            kernel_1d = torch.cos(x)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
        # Pad input to handle boundary conditions
        pad_size = kernel_size // 2
        padded = F.pad(sigmas.unsqueeze(0).unsqueeze(0), (pad_size, pad_size), mode='reflect')
        
        # Apply convolution
        smoothed = F.conv1d(padded, kernel_1d.unsqueeze(0).unsqueeze(0))
        
        return (smoothed.squeeze(),)
    
    def _gaussian_kernel(self, kernel_size, sigma):
        # Generate 1D Gaussian kernel
        x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size)
        kernel = torch.exp(-x**2 / (2*sigma**2))
        return kernel / kernel.sum()

# ----- Quantile Normalization -----
class sigmas_quantile_norm:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "target_distribution": (["uniform", "normal", "exponential", "logistic", "custom"], {"default": "uniform"}),
                "num_quantiles": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
            },
            "optional": {
                "reference_sigmas": ("SIGMAS", {"forceInput": False}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, target_distribution, num_quantiles, reference_sigmas=None):
        # Convert to numpy for processing
        sigmas_np = sigmas.cpu().numpy()
        
        # Sort values
        sorted_values = np.sort(sigmas_np)
        
        # Create rank for each value (fractional rank)
        ranks = np.zeros_like(sigmas_np)
        for i, val in enumerate(sigmas_np):
            ranks[i] = np.searchsorted(sorted_values, val, side='right') / len(sorted_values)
        
        # Generate target distribution
        if target_distribution == "uniform":
            # Uniform distribution between min and max of sigmas
            target_values = np.linspace(sigmas_np.min(), sigmas_np.max(), num_quantiles)
        elif target_distribution == "normal":
            # Normal distribution with same mean and std as sigmas
            target_values = np.random.normal(sigmas_np.mean(), sigmas_np.std(), num_quantiles)
            target_values.sort()
        elif target_distribution == "exponential":
            # Exponential distribution with lambda=1/mean
            target_values = np.random.exponential(1/max(1e-6, sigmas_np.mean()), num_quantiles)
            target_values.sort()
        elif target_distribution == "logistic":
            # Logistic distribution
            target_values = np.random.logistic(0, 1, num_quantiles)
            target_values.sort()
            # Rescale to match sigmas range
            target_values = (target_values - target_values.min()) / (target_values.max() - target_values.min())
            target_values = target_values * (sigmas_np.max() - sigmas_np.min()) + sigmas_np.min()
        elif target_distribution == "custom" and reference_sigmas is not None:
            # Use provided reference distribution
            reference_np = reference_sigmas.cpu().numpy()
            target_values = np.sort(reference_np)
            if len(target_values) < num_quantiles:
                # Interpolate if reference is smaller
                old_indices = np.linspace(0, len(target_values)-1, len(target_values))
                new_indices = np.linspace(0, len(target_values)-1, num_quantiles)
                target_values = np.interp(new_indices, old_indices, target_values)
            else:
                # Subsample if reference is larger
                indices = np.linspace(0, len(target_values)-1, num_quantiles, dtype=int)
                target_values = target_values[indices]
        else:
            # Default to uniform
            target_values = np.linspace(sigmas_np.min(), sigmas_np.max(), num_quantiles)
        
        # Map each value to its corresponding quantile in the target distribution
        result_np = np.interp(ranks, np.linspace(0, 1, len(target_values)), target_values)
        
        # Convert back to tensor
        result = torch.tensor(result_np, device=sigmas.device, dtype=sigmas.dtype)
        
        return (result,)

# ----- Adaptive Step Function -----
class sigmas_adaptive_step:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "adaptation_type": (["gradient", "curvature", "importance", "density"], {"default": "gradient"}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "min_step": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "max_step": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "target_steps": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, adaptation_type, sensitivity, min_step, max_step, target_steps):
        if len(sigmas) <= 1:
            return (sigmas,)
            
        # Compute step sizes based on chosen adaptation type
        if adaptation_type == "gradient":
            # Compute gradient (first difference)
            grads = torch.abs(sigmas[1:] - sigmas[:-1])
            # Normalize gradients
            if grads.max() > grads.min():
                norm_grads = (grads - grads.min()) / (grads.max() - grads.min())
            else:
                norm_grads = torch.ones_like(grads)
            
            # Convert to step sizes: smaller steps where gradient is large
            step_sizes = 1.0 / (1.0 + norm_grads * sensitivity)
            
        elif adaptation_type == "curvature":
            # Compute second derivative approximation
            if len(sigmas) >= 3:
                # Second difference
                second_diff = sigmas[2:] - 2*sigmas[1:-1] + sigmas[:-2]
                # Pad to match length
                second_diff = F.pad(second_diff, (0, 1), mode='replicate')
            else:
                second_diff = torch.zeros_like(sigmas[:-1])
                
            # Normalize curvature
            abs_curve = torch.abs(second_diff)
            if abs_curve.max() > abs_curve.min():
                norm_curve = (abs_curve - abs_curve.min()) / (abs_curve.max() - abs_curve.min())
            else:
                norm_curve = torch.ones_like(abs_curve)
                
            # Convert to step sizes: smaller steps where curvature is high
            step_sizes = 1.0 / (1.0 + norm_curve * sensitivity)
            
        elif adaptation_type == "importance":
            # Importance based on values: focus more on extremes
            centered = torch.abs(sigmas - sigmas.mean())
            if centered.max() > centered.min():
                importance = (centered - centered.min()) / (centered.max() - centered.min())
            else:
                importance = torch.ones_like(centered)
                
            # Steps are smaller for important regions
            step_sizes = 1.0 / (1.0 + importance[:-1] * sensitivity)
            
        elif adaptation_type == "density":
            # Density-based adaptation using kernel density estimation
            # Use a simple histogram approximation
            sigma_min, sigma_max = sigmas.min(), sigmas.max()
            bins = 20
            hist = torch.histc(sigmas, bins=bins, min=sigma_min, max=sigma_max)
            hist = hist / hist.sum()  # Normalize
            
            # Map each sigma to its bin density
            bin_indices = torch.floor((sigmas - sigma_min) / (sigma_max - sigma_min) * (bins-1)).long()
            bin_indices = torch.clamp(bin_indices, 0, bins-1)
            densities = hist[bin_indices]
            
            # Compute step sizes: smaller steps in high density regions
            step_sizes = 1.0 / (1.0 + densities[:-1] * sensitivity)
        
        # Scale step sizes to [min_step, max_step]
        if step_sizes.max() > step_sizes.min():
            step_sizes = (step_sizes - step_sizes.min()) / (step_sizes.max() - step_sizes.min())
            step_sizes = step_sizes * (max_step - min_step) + min_step
        else:
            step_sizes = torch.ones_like(step_sizes) * min_step
            
        # Cumulative sum to get positions
        positions = torch.cat([torch.tensor([0.0], device=step_sizes.device), torch.cumsum(step_sizes, dim=0)])
        
        # Normalize positions to match original range
        positions = positions / positions[-1] * (sigmas[-1] - sigmas[0]) + sigmas[0]
        
        # Resample if target_steps is specified
        if target_steps > 0:
            new_positions = torch.linspace(sigmas[0], sigmas[-1], target_steps, device=sigmas.device)
            # Interpolate to get new sigma values
            new_sigmas = torch.zeros_like(new_positions)
            
            # Simple linear interpolation
            for i, pos in enumerate(new_positions):
                # Find enclosing original positions
                idx = torch.searchsorted(positions, pos)
                idx = torch.clamp(idx, 1, len(positions)-1)
                
                # Linear interpolation
                t = (pos - positions[idx-1]) / (positions[idx] - positions[idx-1])
                new_sigmas[i] = sigmas[idx-1] * (1-t) + sigmas[idx-1] * t
                
            result = new_sigmas
        else:
            result = positions
            
        return (result,)

# ----- Chaos Function -----
class sigmas_chaos:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "system": (["logistic", "henon", "tent", "sine", "cubic"], {"default": "logistic"}),
                "parameter": ("FLOAT", {"default": 3.9, "min": 0.1, "max": 5.0, "step": 0.01}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "use_as_seed": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, system, parameter, iterations, normalize_output, use_as_seed):
        # Normalize input to [0,1] for chaotic maps
        if use_as_seed:
            # Use input as initial seed
            x = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        else:
            # Use single initial value and apply iterations
            x = torch.zeros_like(sigmas)
            for i in range(len(sigmas)):
                # Use i/len as initial value for variety
                x[i] = i / len(sigmas)
        
        # Apply chaos map iterations
        for _ in range(iterations):
            if system == "logistic":
                # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
                x = parameter * x * (1 - x)
                
            elif system == "henon":
                # Simplified 1D version of Henon map
                x = 1 - parameter * x**2
                
            elif system == "tent":
                # Tent map
                x = torch.where(x < 0.5, parameter * x, parameter * (1 - x))
                
            elif system == "sine":
                # Sine map: x_{n+1} = r * sin(pi * x_n)
                x = parameter * torch.sin(math.pi * x)
                
            elif system == "cubic":
                # Cubic map: x_{n+1} = r * x_n * (1 - x_n^2)
                x = parameter * x * (1 - x**2)
                
        # Normalize output if requested
        if normalize_output:
            result = ((x - x.min()) / (x.max() - x.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
        else:
            result = x
            
        return (result,)

# ----- Reaction Diffusion Function -----
class sigmas_reaction_diffusion:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "system": (["gray_scott", "fitzhugh_nagumo", "brusselator"], {"default": "gray_scott"}),
                "iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "dt": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "param_a": ("FLOAT", {"default": 0.04, "min": 0.01, "max": 0.1, "step": 0.001}),
                "param_b": ("FLOAT", {"default": 0.06, "min": 0.01, "max": 0.1, "step": 0.001}),
                "diffusion_a": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "diffusion_b": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, system, iterations, dt, param_a, param_b, diffusion_a, diffusion_b, normalize_output):
        # Initialize a and b based on sigmas
        a = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        b = 1.0 - a
        
        # Pad for diffusion calculation (periodic boundary)
        a_pad = F.pad(a.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
        b_pad = F.pad(b.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
        
        # Simple 1D reaction-diffusion
        for _ in range(iterations):
            # Compute Laplacian (diffusion term) as second derivative
            laplacian_a = a_pad[:-2] + a_pad[2:] - 2 * a
            laplacian_b = b_pad[:-2] + b_pad[2:] - 2 * b
            
            if system == "gray_scott":
                # Gray-Scott model for pattern formation
                # a is "U" (activator), b is "V" (inhibitor)
                feed = 0.055  # feed rate
                kill = 0.062  # kill rate
                
                # Update equations
                a_new = a + dt * (diffusion_a * laplacian_a - a * b**2 + feed * (1 - a))
                b_new = b + dt * (diffusion_b * laplacian_b + a * b**2 - (feed + kill) * b)
                
            elif system == "fitzhugh_nagumo":
                # FitzHugh-Nagumo model (simplified)
                # a is the membrane potential, b is the recovery variable
                
                # Update equations
                a_new = a + dt * (diffusion_a * laplacian_a + a - a**3 - b + param_a)
                b_new = b + dt * (diffusion_b * laplacian_b + param_b * (a - b))
                
            elif system == "brusselator":
                # Brusselator model
                # a is U, b is V
                
                # Update equations
                a_new = a + dt * (diffusion_a * laplacian_a + 1 - (param_b + 1) * a + param_a * a**2 * b)
                b_new = b + dt * (diffusion_b * laplacian_b + param_b * a - param_a * a**2 * b)
            
            # Update and repad
            a, b = a_new, b_new
            a_pad = F.pad(a.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
            b_pad = F.pad(b.unsqueeze(0).unsqueeze(0), (1, 1), mode='circular').squeeze()
            
        # Use the activator component as the result
        result = a
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Attractor Function -----
class sigmas_attractor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "attractor": (["lorenz", "rossler", "aizawa", "chen", "thomas"], {"default": "lorenz"}),
                "iterations": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "dt": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
                "component": (["x", "y", "z", "magnitude"], {"default": "x"}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, attractor, iterations, dt, component, normalize_output):
        # Initialize 3D state from sigmas
        n = len(sigmas)
        
        # Normalize sigmas to a reasonable range for the attractor
        norm_sigmas = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min()) * 2.0 - 1.0
        
        # Create initial state
        x = norm_sigmas
        y = torch.roll(norm_sigmas, 1)  # Shifted version for variety
        z = torch.roll(norm_sigmas, 2)  # Another shifted version
        
        # Parameters for the attractors
        if attractor == "lorenz":
            sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        elif attractor == "rossler":
            a, b, c = 0.2, 0.2, 5.7
        elif attractor == "aizawa":
            a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
        elif attractor == "chen":
            a, b, c = 5.0, -10.0, -0.38
        elif attractor == "thomas":
            b = 0.208186
            
        # Run the attractor dynamics
        for _ in range(iterations):
            if attractor == "lorenz":
                # Lorenz attractor
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                
            elif attractor == "rossler":
                # Rössler attractor
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)
                
            elif attractor == "aizawa":
                # Aizawa attractor
                dx = (z - b) * x - d * y
                dy = d * x + (z - b) * y
                dz = c + a * z - z**3/3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
                
            elif attractor == "chen":
                # Chen attractor
                dx = a * (y - x)
                dy = (c - a) * x - x * z + c * y
                dz = x * y - b * z
                
            elif attractor == "thomas":
                # Thomas attractor
                dx = -b * x + torch.sin(y)
                dy = -b * y + torch.sin(z)
                dz = -b * z + torch.sin(x)
                
            # Update state
            x = x + dt * dx
            y = y + dt * dy
            z = z + dt * dz
            
        # Select component
        if component == "x":
            result = x
        elif component == "y":
            result = y
        elif component == "z":
            result = z
        elif component == "magnitude":
            result = torch.sqrt(x**2 + y**2 + z**2)
            
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Catmull-Rom Spline -----
class sigmas_catmull_rom:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "points": ("INT", {"default": 100, "min": 5, "max": 1000, "step": 5}),
                "boundary_condition": (["repeat", "clamp", "mirror"], {"default": "clamp"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, tension, points, boundary_condition):
        n = len(sigmas)
        
        # Need at least 4 points for Catmull-Rom interpolation
        if n < 4:
            # If we have fewer, just use linear interpolation
            t = torch.linspace(0, 1, points, device=sigmas.device)
            result = torch.zeros(points, device=sigmas.device, dtype=sigmas.dtype)
            
            for i in range(points):
                idx = min(int(i * (n - 1) / (points - 1)), n - 2)
                alpha = (i * (n - 1) / (points - 1)) - idx
                result[i] = (1 - alpha) * sigmas[idx] + alpha * sigmas[idx + 1]
                
            return (result,)
        
        # Handle boundary conditions for control points
        if boundary_condition == "repeat":
            # Repeat endpoints
            p0 = sigmas[0]
            p3 = sigmas[-1]
        elif boundary_condition == "clamp":
            # Extrapolate
            p0 = 2 * sigmas[0] - sigmas[1]
            p3 = 2 * sigmas[-1] - sigmas[-2]
        elif boundary_condition == "mirror":
            # Mirror
            p0 = sigmas[1]
            p3 = sigmas[-2]
            
        # Create extended control points
        control_points = torch.cat([torch.tensor([p0], device=sigmas.device), sigmas, torch.tensor([p3], device=sigmas.device)])
        
        # Compute spline
        result = torch.zeros(points, device=sigmas.device, dtype=sigmas.dtype)
        
        # Parameter to adjust curve tension (0 = Catmull-Rom, 1 = Linear)
        alpha = 1.0 - tension
        
        for i in range(points):
            # Determine which segment we're in
            t = i / (points - 1) * (n - 1)
            idx = min(int(t), n - 2)
            
            # Normalized parameter within the segment [0, 1]
            t_local = t - idx
            
            # Get control points for this segment
            p0 = control_points[idx]
            p1 = control_points[idx + 1]
            p2 = control_points[idx + 2]
            p3 = control_points[idx + 3]
            
            # Catmull-Rom basis functions
            t2 = t_local * t_local
            t3 = t2 * t_local
            
            # Compute spline point
            result[i] = (
                (-alpha * t3 + 2 * alpha * t2 - alpha * t_local) * p0 +
                ((2 - alpha) * t3 + (alpha - 3) * t2 + 1) * p1 +
                ((alpha - 2) * t3 + (3 - 2 * alpha) * t2 + alpha * t_local) * p2 +
                (alpha * t3 - alpha * t2) * p3
            ) * 0.5
            
        return (result,)

# ----- Lambert W-Function -----
class sigmas_lambert_w:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "branch": (["principal", "secondary"], {"default": "principal"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "max_iterations": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, branch, scale, normalize_output, max_iterations):
        # Apply scaling
        x = sigmas * scale
        
        # Lambert W function (numerically approximated)
        result = torch.zeros_like(x)
        
        # Process each value separately (since Lambert W is non-vectorized)
        for i in range(len(x)):
            xi = x[i].item()
            
            # Initial guess varies by branch
            if branch == "principal":
                # Valid for x >= -1/e
                if xi < -1/math.e:
                    xi = -1/math.e  # Clamp to domain
                
                # Initial guess for W₀(x)
                if xi < 0:
                    w = 0.0
                elif xi < 1:
                    w = xi * (1 - xi * (1 - 0.5 * xi))
                else:
                    w = math.log(xi)
                    
            else:  # secondary branch
                # Valid for -1/e <= x < 0
                if xi < -1/math.e:
                    xi = -1/math.e  # Clamp to lower bound
                elif xi >= 0:
                    xi = -0.01  # Clamp to upper bound
                
                # Initial guess for W₋₁(x)
                w = math.log(-xi)
                
            # Halley's method for numerical approximation
            for _ in range(max_iterations):
                ew = math.exp(w)
                wew = w * ew
                
                # If we've converged, break
                if abs(wew - xi) < 1e-10:
                    break
                
                # Halley's update
                wpe = w + 1  # w plus 1
                div = ew * wpe - (ew * w - xi) * wpe / (2 * wpe * ew)
                w_next = w - (wew - xi) / div
                
                # Check for convergence
                if abs(w_next - w) < 1e-10:
                    w = w_next
                    break
                    
                w = w_next
                
            result[i] = w
            
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Zeta & Eta Functions -----
class sigmas_zeta_eta:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "function": (["riemann_zeta", "dirichlet_eta", "lerch_phi"], {"default": "riemann_zeta"}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "approx_terms": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, function, offset, scale, normalize_output, approx_terms):
        # Apply offset and scaling
        s = sigmas * scale + offset
        
        # Process based on function type
        if function == "riemann_zeta":
            # Riemann zeta function
            # For Re(s) > 1, ζ(s) = sum(1/n^s, n=1 to infinity)
            # For performance reasons, we'll use scipy's implementation for CPU
            # and a truncated series approximation for GPU
            
            # Move to CPU for scipy
            s_cpu = s.cpu().numpy()
            
            # Apply zeta function
            result_np = np.zeros_like(s_cpu)
            
            for i, si in enumerate(s_cpu):
                # Handle special values
                if si == 1.0:
                    # ζ(1) is the harmonic series, which diverges to infinity
                    result_np[i] = float('inf')
                elif si < 0 and si == int(si) and int(si) % 2 == 0:
                    # ζ(-2n) = 0 for n > 0
                    result_np[i] = 0.0
                else:
                    try:
                        # Use scipy for computation
                        result_np[i] = float(special.zeta(si))
                    except (ValueError, OverflowError):
                        # Fall back to approximation for problematic values
                        if si > 1:
                            # Truncated series for Re(s) > 1
                            result_np[i] = sum(1.0 / np.power(n, si) for n in range(1, approx_terms))
                        else:
                            # Use functional equation for Re(s) < 0
                            if si < 0:
                                # ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
                                # Gamma function blows up at negative integers, so use the fact that
                                # ζ(-n) = -B_{n+1}/(n+1) for n > 0, where B is a Bernoulli number
                                # However, as this gets complex, we'll use a simpler approximation
                                result_np[i] = 0.0  # Default for problematic values
            
            # Convert back to tensor
            result = torch.tensor(result_np, device=sigmas.device, dtype=sigmas.dtype)
            
        elif function == "dirichlet_eta":
            # Dirichlet eta function (alternating zeta function)
            # η(s) = sum((-1)^(n+1)/n^s, n=1 to infinity)
            
            # For GPU efficiency, compute directly using alternating series
            result = torch.zeros_like(s)
            
            # Use a fixed number of terms for approximation
            for i in range(1, approx_terms + 1):
                term = torch.pow(i, -s) * (1 if i % 2 == 1 else -1)
                result += term
                
        elif function == "lerch_phi":
            # Lerch transcendent with fixed parameters
            # Φ(z, s, a) = sum(z^n / (n+a)^s, n=0 to infinity)
            # We'll use z=0.5, a=1 for simplicity
            z, a = 0.5, 1.0
            
            result = torch.zeros_like(s)
            for i in range(approx_terms):
                term = torch.pow(z, i) / torch.pow(i + a, s)
                result += term
            
        # Replace infinities and NaNs with large or small values
        result = torch.where(torch.isfinite(result), result, torch.sign(result) * 1e10)
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Gamma & Beta Functions -----
class sigmas_gamma_beta:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "function": (["gamma", "beta", "incomplete_gamma", "incomplete_beta", "log_gamma"], {"default": "gamma"}),
                "offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01}),
                "parameter_a": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "parameter_b": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, function, offset, scale, parameter_a, parameter_b, normalize_output):
        # Apply offset and scaling
        x = sigmas * scale + offset
        
        # Convert to numpy for special functions
        x_np = x.cpu().numpy()
        
        # Apply function
        if function == "gamma":
            # Gamma function Γ(x)
            # For performance and stability, use scipy
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                # Handle special cases
                if xi <= 0 and xi == int(xi):
                    # Gamma has poles at non-positive integers
                    result_np[i] = float('inf')
                else:
                    try:
                        result_np[i] = float(special.gamma(xi))
                    except (ValueError, OverflowError):
                        # Use approximation for large values
                        result_np[i] = float('inf')
                        
        elif function == "log_gamma":
            # Log Gamma function log(Γ(x))
            # More numerically stable for large values
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                # Handle special cases
                if xi <= 0 and xi == int(xi):
                    # log(Γ(x)) is undefined for non-positive integers
                    result_np[i] = float('inf')
                else:
                    try:
                        result_np[i] = float(special.gammaln(xi))
                    except (ValueError, OverflowError):
                        # Use approximation for large values
                        result_np[i] = float('inf')
                    
        elif function == "beta":
            # Beta function B(a, x)
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                try:
                    result_np[i] = float(special.beta(parameter_a, xi))
                except (ValueError, OverflowError):
                    # Handle cases where beta is undefined
                    result_np[i] = float('inf')
                    
        elif function == "incomplete_gamma":
            # Regularized incomplete gamma function P(a, x)
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                if xi < 0:
                    # Undefined for negative x
                    result_np[i] = 0.0
                else:
                    try:
                        result_np[i] = float(special.gammainc(parameter_a, xi))
                    except (ValueError, OverflowError):
                        result_np[i] = 1.0  # Approach 1 for large x
                    
        elif function == "incomplete_beta":
            # Regularized incomplete beta function I(x; a, b)
            result_np = np.zeros_like(x_np)
            
            for i, xi in enumerate(x_np):
                # Clamp to [0,1] for domain of incomplete beta
                xi_clamped = min(max(xi, 0), 1)
                
                try:
                    result_np[i] = float(special.betainc(parameter_a, parameter_b, xi_clamped))
                except (ValueError, OverflowError):
                    result_np[i] = 0.5  # Default for errors
                    
        # Convert back to tensor
        result = torch.tensor(result_np, device=sigmas.device, dtype=sigmas.dtype)
        
        # Replace infinities and NaNs
        result = torch.where(torch.isfinite(result), result, torch.sign(result) * 1e10)
        
        # Normalize output if requested
        if normalize_output:
            # Handle cases where result has infinities
            if torch.isinf(result).any() or torch.isnan(result).any():
                # Replace inf/nan with max/min finite values
                max_val = torch.max(result[torch.isfinite(result)]) if torch.any(torch.isfinite(result)) else 1e10
                min_val = torch.min(result[torch.isfinite(result)]) if torch.any(torch.isfinite(result)) else -1e10
                
                result = torch.where(torch.isinf(result) & (result > 0), max_val, result)
                result = torch.where(torch.isinf(result) & (result < 0), min_val, result)
                result = torch.where(torch.isnan(result), (max_val + min_val) / 2, result)
            
            # Now normalize
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

# ----- Sigma Lerp -----
class sigmas_lerp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"forceInput": True}),
                "sigmas_b": ("SIGMAS", {"forceInput": True}),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ensure_length": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_a, sigmas_b, t, ensure_length):
        if ensure_length and len(sigmas_a) != len(sigmas_b):
            # Resize the smaller one to match the larger one
            if len(sigmas_a) < len(sigmas_b):
                sigmas_a = torch.nn.functional.interpolate(
                    sigmas_a.unsqueeze(0).unsqueeze(0), 
                    size=len(sigmas_b), 
                    mode='linear'
                ).squeeze(0).squeeze(0)
            else:
                sigmas_b = torch.nn.functional.interpolate(
                    sigmas_b.unsqueeze(0).unsqueeze(0), 
                    size=len(sigmas_a), 
                    mode='linear'
                ).squeeze(0).squeeze(0)
        
        return ((1 - t) * sigmas_a + t * sigmas_b,)

# ----- Sigma InvLerp -----
class sigmas_invlerp:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, min_value, max_value):
        # Clamp values to avoid division by zero
        if min_value == max_value:
            max_value = min_value + 1e-5
            
        normalized = (sigmas - min_value) / (max_value - min_value)
        # Clamp the values to be in [0, 1]
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return (normalized,)

# ----- Sigma ArcSine -----
class sigmas_arcsine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "normalize_input": ("BOOLEAN", {"default": True}),
                "scale_output": ("BOOLEAN", {"default": True}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, normalize_input, scale_output, out_min, out_max):
        if normalize_input:
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
        else:
            # Ensure values are in valid arcsin domain
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
            
        result = torch.asin(sigmas)
        
        if scale_output:
            # ArcSine output is in range [-π/2, π/2]
            # Normalize to [0, 1] and then scale to [out_min, out_max]
            result = (result + math.pi/2) / math.pi
            result = result * (out_max - out_min) + out_min
            
        return (result,)

# ----- Sigma LinearSine -----
class sigmas_linearsine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "phase": ("FLOAT", {"default": 0.0, "min": -6.28, "max": 6.28, "step": 0.01}), # -2π to 2π
                "linear_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, amplitude, frequency, phase, linear_weight):
        # Create indices for the sine function
        indices = torch.linspace(0, 1, len(sigmas), device=sigmas.device)
        
        # Calculate sine component
        sine_component = amplitude * torch.sin(2 * math.pi * frequency * indices + phase)
        
        # Blend linear and sine components
        step_indices = torch.linspace(0, 1, len(sigmas), device=sigmas.device)
        result = linear_weight * sigmas + (1 - linear_weight) * (step_indices.unsqueeze(0) * sine_component)
        
        return (result.squeeze(0),)

# ----- Sigmas Append -----
class sigmas_append:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1})
            },
            "optional": {
                "additional_sigmas": ("SIGMAS", {"forceInput": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, value, count, additional_sigmas=None):
        # Create tensor of the value to append
        append_values = torch.full((count,), value, device=sigmas.device, dtype=sigmas.dtype)
        
        # Append the values
        result = torch.cat([sigmas, append_values], dim=0)
        
        # If additional sigmas provided, append those as well
        if additional_sigmas is not None:
            result = torch.cat([result, additional_sigmas], dim=0)
            
        return (result,)

# ----- Sigma Arccosine -----
class sigmas_arccosine:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "normalize_input": ("BOOLEAN", {"default": True}),
                "scale_output": ("BOOLEAN", {"default": True}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, normalize_input, scale_output, out_min, out_max):
        if normalize_input:
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
        else:
            # Ensure values are in valid arccos domain
            sigmas = torch.clamp(sigmas, -1.0, 1.0)
            
        result = torch.acos(sigmas)
        
        if scale_output:
            # ArcCosine output is in range [0, π]
            # Normalize to [0, 1] and then scale to [out_min, out_max]
            result = result / math.pi
            result = result * (out_max - out_min) + out_min
            
        return (result,)

# ----- Sigma Arctangent -----
class sigmas_arctangent:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "scale_output": ("BOOLEAN", {"default": True}),
                "out_min": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "out_max": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, scale_output, out_min, out_max):
        result = torch.atan(sigmas)
        
        if scale_output:
            # ArcTangent output is in range [-π/2, π/2]
            # Normalize to [0, 1] and then scale to [out_min, out_max]
            result = (result + math.pi/2) / math.pi
            result = result * (out_max - out_min) + out_min
            
        return (result,)

# ----- Sigma CrossProduct -----
class sigmas_crossproduct:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"forceInput": True}),
                "sigmas_b": ("SIGMAS", {"forceInput": True}),
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_a, sigmas_b):
        # Ensure we have at least 3 elements in each tensor
        # If not, pad with zeros or truncate
        if len(sigmas_a) < 3:
            sigmas_a = torch.nn.functional.pad(sigmas_a, (0, 3 - len(sigmas_a)))
        if len(sigmas_b) < 3:
            sigmas_b = torch.nn.functional.pad(sigmas_b, (0, 3 - len(sigmas_b)))
        
        # Take the first 3 elements of each tensor
        a = sigmas_a[:3]
        b = sigmas_b[:3]
        
        # Compute cross product
        c = torch.zeros(3, device=sigmas_a.device, dtype=sigmas_a.dtype)
        c[0] = a[1] * b[2] - a[2] * b[1]
        c[1] = a[2] * b[0] - a[0] * b[2]
        c[2] = a[0] * b[1] - a[1] * b[0]
        
        return (c,)

# ----- Sigma DotProduct -----
class sigmas_dotproduct:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_a": ("SIGMAS", {"forceInput": True}),
                "sigmas_b": ("SIGMAS", {"forceInput": True}),
                "normalize": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas_a, sigmas_b, normalize):
        # Ensure equal lengths by taking the minimum
        min_length = min(len(sigmas_a), len(sigmas_b))
        a = sigmas_a[:min_length]
        b = sigmas_b[:min_length]
        
        if normalize:
            a_norm = torch.norm(a)
            b_norm = torch.norm(b)
            # Avoid division by zero
            if a_norm > 0 and b_norm > 0:
                a = a / a_norm
                b = b / b_norm
        
        # Compute dot product
        result = torch.sum(a * b)
        
        # Return as a single-element tensor
        return (torch.tensor([result], device=sigmas_a.device, dtype=sigmas_a.dtype),)

# ----- Sigma Fmod -----
class sigmas_fmod:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "divisor": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 10000.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, divisor):
        # Ensure divisor is not zero
        if divisor == 0:
            divisor = 0.0001
            
        result = torch.fmod(sigmas, divisor)
        return (result,)

# ----- Sigma Frac -----
class sigmas_frac:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas):
        # Get the fractional part (x - floor(x))
        result = sigmas - torch.floor(sigmas)
        return (result,)

# ----- Sigma If -----
class sigmas_if:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition_sigmas": ("SIGMAS", {"forceInput": True}),
                "true_sigmas": ("SIGMAS", {"forceInput": True}),
                "false_sigmas": ("SIGMAS", {"forceInput": True}),
                "threshold": ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "comp_type": (["greater", "less", "equal", "not_equal"], {"default": "greater"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, condition_sigmas, true_sigmas, false_sigmas, threshold, comp_type):
        # Make sure we have values to compare
        max_length = max(len(condition_sigmas), len(true_sigmas), len(false_sigmas))
        
        # Extend all tensors to the maximum length using interpolation
        if len(condition_sigmas) != max_length:
            condition_sigmas = torch.nn.functional.interpolate(
                condition_sigmas.unsqueeze(0).unsqueeze(0), 
                size=max_length, 
                mode='linear'
            ).squeeze(0).squeeze(0)
            
        if len(true_sigmas) != max_length:
            true_sigmas = torch.nn.functional.interpolate(
                true_sigmas.unsqueeze(0).unsqueeze(0), 
                size=max_length, 
                mode='linear'
            ).squeeze(0).squeeze(0)
            
        if len(false_sigmas) != max_length:
            false_sigmas = torch.nn.functional.interpolate(
                false_sigmas.unsqueeze(0).unsqueeze(0), 
                size=max_length, 
                mode='linear'
            ).squeeze(0).squeeze(0)
            
        # Create mask based on comparison type
        if comp_type == "greater":
            mask = condition_sigmas > threshold
        elif comp_type == "less":
            mask = condition_sigmas < threshold
        elif comp_type == "equal":
            mask = torch.isclose(condition_sigmas, torch.tensor(threshold, device=condition_sigmas.device))
        elif comp_type == "not_equal":
            mask = ~torch.isclose(condition_sigmas, torch.tensor(threshold, device=condition_sigmas.device))
        
        # Apply the mask to select values
        result = torch.where(mask, true_sigmas, false_sigmas)
        
        return (result,)

# ----- Sigma Logarithm2 -----
class sigmas_logarithm2:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "handle_negative": ("BOOLEAN", {"default": True}),
                "epsilon": ("FLOAT", {"default": 1e-10, "min": 1e-15, "max": 0.1, "step": 1e-10})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, handle_negative, epsilon):
        if handle_negative:
            # For negative values, compute -log2(-x) and negate the result
            mask_negative = sigmas < 0
            mask_positive = ~mask_negative
            
            # Prepare positive and negative parts
            pos_part = torch.log2(torch.clamp(sigmas[mask_positive], min=epsilon))
            neg_part = -torch.log2(torch.clamp(-sigmas[mask_negative], min=epsilon))
            
            # Create result tensor
            result = torch.zeros_like(sigmas)
            result[mask_positive] = pos_part
            result[mask_negative] = neg_part
        else:
            # Simply compute log2, clamping values to avoid log(0)
            result = torch.log2(torch.clamp(sigmas, min=epsilon))
            
        return (result,)

# ----- Sigma SmoothStep -----
class sigmas_smoothstep:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "edge0": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "edge1": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "mode": (["smoothstep", "smootherstep"], {"default": "smoothstep"})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, edge0, edge1, mode):
        # Normalize the values to the range [0, 1]
        t = torch.clamp((sigmas - edge0) / (edge1 - edge0), 0.0, 1.0)
        
        if mode == "smoothstep":
            # Smooth step: 3t^2 - 2t^3
            result = t * t * (3.0 - 2.0 * t)
        else:  # smootherstep
            # Smoother step: 6t^5 - 15t^4 + 10t^3
            result = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
            
        # Scale back to the original range
        result = result * (edge1 - edge0) + edge0
        
        return (result,)

# ----- Sigma SquareRoot -----
class sigmas_squareroot:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "handle_negative": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, handle_negative):
        if handle_negative:
            # For negative values, compute sqrt(-x) and negate the result
            mask_negative = sigmas < 0
            mask_positive = ~mask_negative
            
            # Prepare positive and negative parts
            pos_part = torch.sqrt(sigmas[mask_positive])
            neg_part = -torch.sqrt(-sigmas[mask_negative])
            
            # Create result tensor
            result = torch.zeros_like(sigmas)
            result[mask_positive] = pos_part
            result[mask_negative] = neg_part
        else:
            # Only compute square root for non-negative values
            # Negative values will be set to 0
            result = torch.sqrt(torch.clamp(sigmas, min=0))
            
        return (result,)

# ----- Sigma TimeStep -----
class sigmas_timestep:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "dt": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 10.0, "step": 0.01}),
                "scaling": (["linear", "quadratic", "sqrt", "log"], {"default": "linear"}),
                "decay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, dt, scaling, decay):
        # Create time steps
        timesteps = torch.arange(len(sigmas), device=sigmas.device, dtype=sigmas.dtype) * dt
        
        # Apply scaling
        if scaling == "quadratic":
            timesteps = timesteps ** 2
        elif scaling == "sqrt":
            timesteps = torch.sqrt(timesteps)
        elif scaling == "log":
            # Add small epsilon to avoid log(0)
            timesteps = torch.log(timesteps + 1e-10)
            
        # Apply decay
        if decay > 0:
            decay_factor = torch.exp(-decay * timesteps)
            timesteps = timesteps * decay_factor
            
        # Normalize to match the range of sigmas
        timesteps = ((timesteps - timesteps.min()) / 
                     (timesteps.max() - timesteps.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (timesteps,)

class sigmas_gaussian_cdf:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "mu": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, mu, sigma, normalize_output):
        # Apply Gaussian CDF transformation
        result = 0.5 * (1 + torch.erf((sigmas - mu) / (sigma * math.sqrt(2))))
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

class sigmas_stepwise_multirate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
                "rates": ("STRING", {"default": "1.0,0.5,0.25", "multiline": False}),
                "boundaries": ("STRING", {"default": "0.3,0.7", "multiline": False}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, "step": 0.01}),
                "pad_end": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, rates, boundaries, start_value, end_value, pad_end):
        # Parse rates and boundaries
        rates_list = [float(r) for r in rates.split(',')]
        if len(rates_list) < 1:
            rates_list = [1.0]
            
        boundaries_list = [float(b) for b in boundaries.split(',')]
        if len(boundaries_list) != len(rates_list) - 1:
            # Create equal size segments if boundaries don't match rates
            boundaries_list = [i / len(rates_list) for i in range(1, len(rates_list))]
        
        # Convert boundaries to step indices
        boundary_indices = [int(b * steps) for b in boundaries_list]
        
        # Create steps array
        result = torch.zeros(steps)
        
        # Fill segments with different rates
        current_idx = 0
        for i, rate in enumerate(rates_list):
            next_idx = boundary_indices[i] if i < len(boundary_indices) else steps
            segment_length = next_idx - current_idx
            if segment_length <= 0:
                continue
                
            segment_start = start_value if i == 0 else result[current_idx-1]
            segment_end = end_value if i == len(rates_list) - 1 else start_value * (1 - boundaries_list[i])
            
            # Apply rate to the segment
            t = torch.linspace(0, 1, segment_length)
            segment = segment_start + (segment_end - segment_start) * (t ** rate)
            
            result[current_idx:next_idx] = segment
            current_idx = next_idx
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0])])
            
        return (result,)

class sigmas_harmonic_decay:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, "step": 0.01}),
                "harmonic_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "decay_rate": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pad_end": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, harmonic_offset, decay_rate, pad_end):
        # Create harmonic series: 1/(n+offset)^rate
        n = torch.arange(1, steps + 1, dtype=torch.float32)
        harmonic_values = 1.0 / torch.pow(n + harmonic_offset, decay_rate)
        
        # Normalize to [0, 1]
        normalized = (harmonic_values - harmonic_values.min()) / (harmonic_values.max() - harmonic_values.min())
        
        # Scale to [end_value, start_value] and reverse (higher values first)
        result = start_value - (start_value - end_value) * normalized
        result = torch.flip(result, [0])
        
        # Add padding zero at the end if requested
        if pad_end:
            result = torch.cat([result, torch.tensor([0.0])])
            
        return (result,)

class sigmas_adaptive_noise_floor:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "min_noise_level": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "adaptation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "window_size": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, min_noise_level, adaptation_factor, window_size):
        # Initialize result with original sigmas
        result = sigmas.clone()
        
        # Apply adaptive noise floor
        for i in range(window_size, len(sigmas)):
            # Calculate local statistics in the window
            window = sigmas[i-window_size:i]
            local_mean = torch.mean(window)
            local_var = torch.var(window)
            
            # Adapt the noise floor based on local statistics
            adaptive_floor = min_noise_level + adaptation_factor * local_var / (local_mean + 1e-6)
            
            # Apply the floor if needed
            if result[i] < adaptive_floor:
                result[i] = adaptive_floor
        
        return (result,)

class sigmas_collatz_iteration:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "iterations": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "scaling_factor": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 10.0, "step": 0.01}),
                "normalize_output": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, iterations, scaling_factor, normalize_output):
        # Scale input to reasonable range for Collatz
        scaled_input = sigmas * scaling_factor
        
        # Apply Collatz iterations
        result = scaled_input.clone()
        
        for _ in range(iterations):
            # Create masks for even and odd values
            even_mask = (result % 2 == 0)
            odd_mask = ~even_mask
            
            # Apply Collatz function: n/2 for even, 3n+1 for odd
            result[even_mask] = result[even_mask] / 2
            result[odd_mask] = 3 * result[odd_mask] + 1
        
        # Normalize output if requested
        if normalize_output:
            result = ((result - result.min()) / (result.max() - result.min())) * (sigmas.max() - sigmas.min()) + sigmas.min()
            
        return (result,)

class sigmas_conway_sequence:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 50, "step": 1}),
                "sequence_type": (["look_and_say", "audioactive", "paperfolding", "thue_morse"], {"default": "look_and_say"}),
                "normalize_range": ("BOOLEAN", {"default": True}),
                "min_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, sequence_type, normalize_range, min_value, max_value):
        if sequence_type == "look_and_say":
            # Start with "1"
            s = "1"
            lengths = [1]  # Length of first term is 1
            
            # Generate look-and-say sequence
            for _ in range(min(steps - 1, 25)):  # Limit to prevent excessive computation
                next_s = ""
                i = 0
                while i < len(s):
                    count = 1
                    while i + 1 < len(s) and s[i] == s[i + 1]:
                        i += 1
                        count += 1
                    next_s += str(count) + s[i]
                    i += 1
                s = next_s
                lengths.append(len(s))
            
            # Convert to tensor
            result = torch.tensor(lengths, dtype=torch.float32)
            
        elif sequence_type == "audioactive":
            # Audioactive sequence (similar to look-and-say but counts digits)
            a = [1]
            for _ in range(min(steps - 1, 30)):
                b = []
                digit_count = {}
                for digit in a:
                    digit_count[digit] = digit_count.get(digit, 0) + 1
                
                for digit in sorted(digit_count.keys()):
                    b.append(digit_count[digit])
                    b.append(digit)
                a = b
            
            result = torch.tensor(a, dtype=torch.float32)
            if len(result) > steps:
                result = result[:steps]
            
        elif sequence_type == "paperfolding":
            # Paper folding sequence (dragon curve)
            sequence = []
            for i in range(min(steps, 30)):
                sequence.append(1 if (i & (i + 1)) % 2 == 0 else 0)
            
            result = torch.tensor(sequence, dtype=torch.float32)
            
        elif sequence_type == "thue_morse":
            # Thue-Morse sequence
            sequence = [0]
            while len(sequence) < steps:
                sequence.extend([1 - x for x in sequence])
            
            result = torch.tensor(sequence, dtype=torch.float32)[:steps]
        
        # Normalize to desired range
        if normalize_range:
            if result.max() > result.min():
                result = (result - result.min()) / (result.max() - result.min())
                result = result * (max_value - min_value) + min_value
            else:
                result = torch.ones_like(result) * min_value
        
        return (result,)

class sigmas_gilbreath_sequence:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 10, "max": 100, "step": 1}),
                "levels": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "normalize_range": ("BOOLEAN", {"default": True}),
                "min_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, levels, normalize_range, min_value, max_value):
        # Generate first few prime numbers
        def sieve_of_eratosthenes(limit):
            sieve = [True] * (limit + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, limit + 1, i):
                        sieve[j] = False
            return [i for i in range(limit + 1) if sieve[i]]
        
        # Get primes
        primes = sieve_of_eratosthenes(steps * 6)  # Get enough primes
        primes = primes[:steps]
        
        # Generate Gilbreath sequence levels
        sequences = [primes]
        for level in range(1, levels):
            prev_seq = sequences[level-1]
            new_seq = [abs(prev_seq[i] - prev_seq[i+1]) for i in range(len(prev_seq)-1)]
            sequences.append(new_seq)
        
        # Select the requested level
        selected_level = min(levels-1, len(sequences)-1)
        result_list = sequences[selected_level]
        
        # Ensure we have enough values
        while len(result_list) < steps:
            result_list.append(1)  # Gilbreath conjecture: eventually all 1s
        
        # Convert to tensor
        result = torch.tensor(result_list[:steps], dtype=torch.float32)
        
        # Normalize to desired range
        if normalize_range:
            if result.max() > result.min():
                result = (result - result.min()) / (result.max() - result.min())
                result = result * (max_value - min_value) + min_value
            else:
                result = torch.ones_like(result) * min_value
        
        return (result,)

class sigmas_cnf_inverse:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "time_steps": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "flow_type": (["linear", "quadratic", "sigmoid", "exponential"], {"default": "sigmoid"}),
                "reverse": ("BOOLEAN", {"default": True})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, sigmas, time_steps, flow_type, reverse):
        # Create normalized time steps
        t = torch.linspace(0, 1, time_steps)
        
        # Apply CNF flow transformation
        if flow_type == "linear":
            flow = t
        elif flow_type == "quadratic":
            flow = t**2
        elif flow_type == "sigmoid":
            flow = 1 / (1 + torch.exp(-10 * (t - 0.5)))
        elif flow_type == "exponential":
            flow = torch.exp(3 * t) - 1
            flow = flow / flow.max()  # Normalize to [0,1]
        
        # Reverse flow if requested
        if reverse:
            flow = 1 - flow
        
        # Interpolate sigmas according to flow
        # First normalize sigmas to [0,1] for interpolation
        normalized_sigmas = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
        
        # Create indices for interpolation
        indices = flow * (len(sigmas) - 1)
        
        # Linear interpolation
        result = torch.zeros(time_steps, device=sigmas.device, dtype=sigmas.dtype)
        for i in range(time_steps):
            idx_low = int(indices[i])
            idx_high = min(idx_low + 1, len(sigmas) - 1)
            frac = indices[i] - idx_low
            
            result[i] = (1 - frac) * normalized_sigmas[idx_low] + frac * normalized_sigmas[idx_high]
        
        # Scale back to original sigma range
        result = result * (sigmas.max() - sigmas.min()) + sigmas.min()
        
        return (result,)

class sigmas_riemannian_flow:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "metric_type": (["euclidean", "hyperbolic", "spherical", "lorentzian"], {"default": "hyperbolic"}),
                "curvature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, metric_type, curvature, start_value, end_value):
        # Create parameter t in [0, 1]
        t = torch.linspace(0, 1, steps)
        
        # Apply different Riemannian metrics
        if metric_type == "euclidean":
            # Simple linear interpolation in Euclidean space
            result = start_value * (1 - t) + end_value * t
            
        elif metric_type == "hyperbolic":
            # Hyperbolic space geodesic
            K = -curvature  # Negative curvature for hyperbolic space
            
            # Convert to hyperbolic coordinates (using Poincaré disk model)
            x_start = torch.tanh(start_value / 2)
            x_end = torch.tanh(end_value / 2)
            
            # Distance in hyperbolic space
            d = torch.acosh(1 + 2 * ((x_start - x_end)**2) / ((1 - x_start**2) * (1 - x_end**2)))
            
            # Geodesic interpolation
            lambda_t = torch.sinh(t * d) / torch.sinh(d)
            result = 2 * torch.atanh((1 - lambda_t) * x_start + lambda_t * x_end)
            
        elif metric_type == "spherical":
            # Spherical space geodesic (great circle)
            K = curvature  # Positive curvature for spherical space
            
            # Convert to angular coordinates
            theta_start = start_value * torch.sqrt(K)
            theta_end = end_value * torch.sqrt(K)
            
            # Geodesic interpolation along great circle
            result = torch.sin((1 - t) * theta_start + t * theta_end) / torch.sqrt(K)
            
        elif metric_type == "lorentzian":
            # Lorentzian spacetime-inspired metric (time dilation effect)
            gamma = 1 / torch.sqrt(1 - curvature * t**2)  # Lorentz factor
            result = start_value * (1 - t) + end_value * t
            result = result * gamma  # Apply time dilation
        
        # Ensure the values are in the desired range
        result = torch.clamp(result, min=min(start_value, end_value), max=max(start_value, end_value))
        
        # Ensure result is decreasing if start_value > end_value
        if start_value > end_value and result[0] < result[-1]:
            result = torch.flip(result, [0])
            
        return (result,)

class sigmas_langevin_dynamics:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 10.0, "step": 0.01}),
                "friction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, temperature, friction, seed):
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Potential function (quadratic well centered at end_value)
        def U(x):
            return 0.5 * (x - end_value)**2
        
        # Gradient of the potential
        def grad_U(x):
            return x - end_value
        
        # Initialize state
        x = torch.tensor([start_value], dtype=torch.float32)
        v = torch.zeros(1)  # Initial velocity
        
        # Discretization parameters
        dt = 1.0 / steps
        sqrt_2dt = math.sqrt(2 * dt)
        
        # Storage for trajectory
        trajectory = [start_value]
        
        # Langevin dynamics integration (velocity Verlet with Langevin thermostat)
        for _ in range(steps - 1):
            # Half step in velocity
            v = v - dt * friction * v - dt * grad_U(x) / 2
            
            # Full step in position
            x = x + dt * v
            
            # Random force (thermal noise)
            noise = torch.randn(1) * sqrt_2dt * temperature
            
            # Another half step in velocity with noise
            v = v - dt * friction * v - dt * grad_U(x) / 2 + noise
            
            # Store current position
            trajectory.append(x.item())
        
        # Convert to tensor
        result = torch.tensor(trajectory, dtype=torch.float32)
        
        # Ensure we reach the end value
        result[-1] = end_value
        
        return (result,)

class sigmas_persistent_homology:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "persistence_type": (["linear", "exponential", "logarithmic", "sigmoidal"], {"default": "exponential"}),
                "birth_density": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "death_density": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, persistence_type, birth_density, death_density):
        # Basic filtration function (linear by default)
        t = torch.linspace(0, 1, steps)
        
        # Persistence diagram simulation
        # Create birth and death times
        birth_points = int(steps * birth_density)
        death_points = int(steps * death_density)
        
        # Filtration function based on selected type
        if persistence_type == "linear":
            filtration = t
        elif persistence_type == "exponential":
            filtration = 1 - torch.exp(-5 * t)
        elif persistence_type == "logarithmic":
            filtration = torch.log(1 + 9 * t) / torch.log(torch.tensor([10.0]))
        elif persistence_type == "sigmoidal":
            filtration = 1 / (1 + torch.exp(-10 * (t - 0.5)))
        
        # Generate birth-death pairs
        birth_indices = torch.linspace(0, steps // 2, birth_points).long()
        death_indices = torch.linspace(steps // 2, steps - 1, death_points).long()
        
        # Create persistence barcode
        barcode = torch.zeros(steps)
        for b_idx in birth_indices:
            for d_idx in death_indices:
                if b_idx < d_idx:
                    # Add a persistence feature from birth to death
                    barcode[b_idx:d_idx] += 1
        
        # Normalize and weight the barcode
        if barcode.max() > 0:
            barcode = barcode / barcode.max()
        
        # Modulate the filtration function with the persistence barcode
        result = filtration * (0.7 + 0.3 * barcode)
        
        # Scale to desired range
        result = start_value + (end_value - start_value) * result
        
        return (result,)

class sigmas_normalizing_flows:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 30, "min": 5, "max": 100, "step": 1}),
                "start_value": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 50.0, "step": 0.1}),
                "end_value": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
                "flow_type": (["affine", "planar", "radial", "realnvp"], {"default": "realnvp"}),
                "num_transforms": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 99999, "step": 1})
            }
        }

    FUNCTION = "main"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "RES4LYF/sigmas"
    
    def main(self, steps, start_value, end_value, flow_type, num_transforms, seed):
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Create base linear schedule from start_value to end_value
        base_schedule = torch.linspace(start_value, end_value, steps)
        
        # Apply different normalizing flow transformations
        if flow_type == "affine":
            # Affine transformation: f(x) = a*x + b
            result = base_schedule.clone()
            for _ in range(num_transforms):
                a = torch.rand(1) * 0.5 + 0.75  # Scale in [0.75, 1.25]
                b = (torch.rand(1) - 0.5) * 0.2  # Shift in [-0.1, 0.1]
                result = a * result + b
                
        elif flow_type == "planar":
            # Planar flow: f(x) = x + u * tanh(w * x + b)
            result = base_schedule.clone()
            for _ in range(num_transforms):
                u = torch.rand(1) * 0.4 - 0.2  # in [-0.2, 0.2]
                w = torch.rand(1) * 2 - 1  # in [-1, 1]
                b = torch.rand(1) * 0.2 - 0.1  # in [-0.1, 0.1]
                result = result + u * torch.tanh(w * result + b)
                
        elif flow_type == "radial":
            # Radial flow: f(x) = x + beta * (x - x0) / (alpha + |x - x0|)
            result = base_schedule.clone()
            for _ in range(num_transforms):
                # Pick a random reference point within the range
                idx = torch.randint(0, steps, (1,))
                x0 = result[idx]
                
                alpha = torch.rand(1) * 0.5 + 0.5  # in [0.5, 1.0]
                beta = torch.rand(1) * 0.4 - 0.2  # in [-0.2, 0.2]
                
                # Apply radial flow
                diff = result - x0
                r = torch.abs(diff)
                result = result + beta * diff / (alpha + r)
                
        elif flow_type == "realnvp":
            # Simplified RealNVP-inspired flow with masking
            result = base_schedule.clone()
            
            for _ in range(num_transforms):
                # Create alternating mask
                mask = torch.zeros(steps)
                mask[::2] = 1  # Mask even indices
                
                # Generate scale and shift parameters
                log_scale = torch.rand(steps) * 0.2 - 0.1  # in [-0.1, 0.1]
                shift = torch.rand(steps) * 0.2 - 0.1  # in [-0.1, 0.1]
                
                # Apply affine coupling transformation
                scale = torch.exp(log_scale * mask)
                masked_shift = shift * mask
                
                # Transform
                result = result * scale + masked_shift
        
        # Rescale to ensure we maintain start_value and end_value
        if result[0] != start_value or result[-1] != end_value:
            result = (result - result[0]) / (result[-1] - result[0]) * (end_value - start_value) + start_value
        
        return (result,)









def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
    smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

    srange = smax-smin
    sscale = start - end

    sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
    
    return sigmas

def bong_tangent_scheduler(model_sampling, steps, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    steps += 2

    midpoint = int( (steps*pivot_1 + steps*pivot_2) / 2 )
    pivot_1 = int(steps * pivot_1)
    pivot_2 = int(steps * pivot_2)

    slope_1 = slope_1 / (steps/40)
    slope_2 = slope_2 / (steps/40)

    stage_2_len = steps - midpoint
    stage_1_len = steps - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
    
    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2+[0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas
