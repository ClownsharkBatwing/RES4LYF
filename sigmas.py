import torch

import numpy as np
from math import *
import builtins
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from comfy.k_diffusion.sampling import get_sigmas_polyexponential, get_sigmas_karras

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
    CATEGORY = "sampling/custom_sampling/"

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

        # Disable inference mode explicitly for training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    return model

def interpolate_sigma_schedule_model(sigma_schedule, target_steps):
    model = SimpleInterpolator()
    sigma_schedule = sigma_schedule.float().detach()

    # Train the model on the original sigma schedule
    trained_model = train_interpolator(model, sigma_schedule, len(sigma_schedule))

    # Generate the target steps for interpolation
    x_interpolated = torch.linspace(0, 1, target_steps).unsqueeze(1)

    # Perform inference with gradients disabled
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    



    def interpolate_sigma_schedule_poly(self, sigma_schedule, target_steps):
        order = self.order
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # Original steps (assuming they are uniformly spaced)
        original_steps = np.linspace(0, 1, len(sigma_schedule_np))

        # Fit a polynomial of the given order to the data using numpy
        coefficients = np.polyfit(original_steps, sigma_schedule_np, deg=order)

        # Generate the new steps where we want to interpolate the data
        target_steps_np = np.linspace(0, 1, target_steps)

        # Evaluate the polynomial at the new steps
        interpolated_sigma_np = np.polyval(coefficients, target_steps_np)

        # Convert the result back to a PyTorch tensor
        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma

    def interpolate_sigma_schedule_constrained(self, sigma_schedule, target_steps):
        # Convert to numpy for interpolation
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # Original steps
        original_steps = np.linspace(0, 1, len(sigma_schedule_np))

        # Target steps for interpolation
        target_steps_np = np.linspace(0, 1, target_steps)

        # Fit cubic spline with fixed start and end values
        cs = CubicSpline(original_steps, sigma_schedule_np, bc_type=((1, 0.0), (1, 0.0)))

        # Evaluate the spline at the target steps
        interpolated_sigma_np = cs(target_steps_np)

        # Convert back to a PyTorch tensor
        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma
    
    def interpolate_sigma_schedule_exp(self, sigma_schedule, target_steps):
        # Step 1: Transform to log space
        log_sigma_schedule = torch.log(sigma_schedule)

        # Step 2: Define the original and target step ranges
        original_steps = torch.linspace(0, 1, steps=len(sigma_schedule))
        target_steps = torch.linspace(0, 1, steps=target_steps)

        # Step 3: Interpolate in log space
        interpolated_log_sigma = F.interpolate(
            log_sigma_schedule.unsqueeze(0).unsqueeze(0),  # Add fake batch and channel dimensions
            size=target_steps.shape[0],
            mode='linear',
            align_corners=True
        ).squeeze()

        # Step 4: Transform back to exponential space
        interpolated_sigma_schedule = torch.exp(interpolated_log_sigma)

        return interpolated_sigma_schedule
    
    def interpolate_sigma_schedule_power(self, sigma_schedule, target_steps):
        # Convert to numpy for easier manipulation
        sigma_schedule_np = sigma_schedule.cpu().numpy()

        # Original steps (assuming uniformly spaced)
        original_steps = np.linspace(1, len(sigma_schedule_np), len(sigma_schedule_np))

        # Perform power regression using a log-log transformation
        log_x = np.log(original_steps)
        log_y = np.log(sigma_schedule_np)

        # Perform linear regression on log-log data
        coefficients = np.polyfit(log_x, log_y, deg=1)  # degree 1 for linear fit in log-log space
        a = np.exp(coefficients[1])  # Intercept (exp because of the log transform)
        b = coefficients[0]  # Slope

        # Generate the target steps for interpolation
        target_steps_np = np.linspace(1, len(sigma_schedule_np), target_steps)

        # Predict using the power law: y = a * x^b
        interpolated_sigma_np = a * (target_steps_np ** b)

        # Convert back to a PyTorch tensor
        interpolated_sigma = torch.tensor(interpolated_sigma_np, device=sigma_schedule.device, dtype=sigma_schedule.dtype)

        return interpolated_sigma
            
    def interpolate_sigma_schedule_linear(self, sigma_schedule, target_steps):
        return F.interpolate(sigma_schedule.unsqueeze(0).unsqueeze(0), target_steps, mode='linear').squeeze(0).squeeze(0)

    def interpolate_sigma_schedule_nearest(self, sigma_schedule, target_steps):
        return F.interpolate(sigma_schedule.unsqueeze(0).unsqueeze(0), target_steps, mode='nearest').squeeze(0).squeeze(0)    
    
    def interpolate_nearest_neighbor(self, sigma_schedule, target_steps):
        # Original steps (e.g., 0 to 1 normalized)
        original_steps = torch.linspace(0, 1, steps=len(sigma_schedule))

        # Target steps (e.g., reducing down to 10 steps)
        target_steps = torch.linspace(0, 1, steps=target_steps)

        # Nearest neighbor interpolation
        indices = torch.searchsorted(original_steps, target_steps)
        indices = torch.clamp(indices, 0, len(sigma_schedule) - 1)  # Ensure indices are within valid range

        # Select the closest values based on the indices
        interpolated_sigma = sigma_schedule[indices]

        return interpolated_sigma


    def main(self, sigmas_0, sigmas_1, mode, order):

        self.order = order
        #with torch.set_grad_enabled(True):
        #    original_sigma = torch.tensor([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2])
        #    sigmas_0 = interpolate_sigma_schedule_model(original_sigma, 100)
        
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
        
        #if  (len(sigmas_0) < len(sigmas_1)):
        #    sigmas_0 = interpolate(sigmas_0, len(sigmas_1))
        #elif(len(sigmas_0) > len(sigmas_1)):
        #    sigmas_1 = interpolate(sigmas_1, len(sigmas_0))

        sigmas_0 = interpolate(sigmas_0, len(sigmas_1))
        return (sigmas_0, sigmas_1,)
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas):
        dtype = sigmas.dtype
        sigmas = sigmas.clone().to(torch.float64)
        for i in range(len(sigmas) - 1):
            sigma_next = (-1 + torch.sqrt(1 + 4 * sigmas[i])) / 2
            
            if sigmas[i+1] < sigma_next and sigmas[i+1] > 0.0:
                print("swapped i+1 with sigma_next+0.001: ", sigmas[i+1], sigma_next + 0.001)
                sigmas[i+1] = sigma_next + 0.001
        return (sigmas.to(dtype),)


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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (torch.cat((sigmas_1, sigmas_2)),)

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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas, sigmas_until):
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas, sigmas_until):
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas, sigmas_start, sigmas_end):
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas, value):
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas):
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

    CATEGORY = "sampling/custom_sampling/sigmas"

    def set_floor(self, sigmas, floor, new_floor):
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

    CATEGORY = "sampling/custom_sampling/sigmas"

    def delete_below_floor(self, sigmas, floor):
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

    CATEGORY = "sampling/custom_sampling/sigmas"

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

    CATEGORY = "sampling/custom_sampling/sigmas"

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

    CATEGORY = "sampling/custom_sampling/sigmas"

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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
    def main(self, sigmas_1, sigmas_2):
        return (sigmas_1 + sigmas_2,)

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
    CATEGORY = "sampling/custom_sampling/sigmas"
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
    CATEGORY = "sampling/custom_sampling/sigmas"
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/sigmas"
    
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
    CATEGORY = "sampling/custom_sampling/schedulers"
    
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
    CATEGORY = "sampling/custom_sampling/schedulers"

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
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_tan_sigmas(self, steps, slope, pivot, start, end):
        smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
        smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

        srange = smax-smin
        sscale = start - end

        sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]
        
        return sigmas

    def main(self, steps, start, middle, end, pivot_1, pivot_2, slope_1, slope_2, pad):
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

