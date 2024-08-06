import torch

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import node_helpers

import functools
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
        
        # Recursive function to cast tensors in nested dictionaries
        def cast_and_move_to_device(data):
            if torch.is_tensor(data):
                return data.to(torch.float64).to(target_device)
            elif isinstance(data, dict):
                return {k: cast_and_move_to_device(v) for k, v in data.items()}
            return data

        # Cast all tensor arguments to float64 and move them to the target device
        new_args = [cast_and_move_to_device(arg) for arg in args]
        new_kwargs = {k: cast_and_move_to_device(v) for k, v in kwargs.items()}
        
        return func(*new_args, **new_kwargs)
    return wrapper


def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def multiply_nested_tensors(structure, scalar):
    if isinstance(structure, torch.Tensor):
        return structure * scalar
    elif isinstance(structure, list):
        return [multiply_nested_tensors(item, scalar) for item in structure]
    elif isinstance(structure, dict):
        return {key: multiply_nested_tensors(value, scalar) for key, value in structure.items()}
    else:
        return structure


class ConditioningMultiply :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                              "multiplier": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "main"

    CATEGORY = "conditioning"

    def main(self, conditioning, multiplier):
        c = multiply_nested_tensors(conditioning, multiplier)
        return (c,)



class ConditioningCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_1": ("CONDITIONING", ), "conditioning_2": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"

    CATEGORY = "conditioning"

    def combine(self, conditioning_1, conditioning_2):
        import pdb; pdb.set_trace()
        return (conditioning_1 + conditioning_2, )

class ConditioningAverage :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ), "conditioning_from": ("CONDITIONING", ),
                              "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "addWeighted"

    CATEGORY = "conditioning"

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
        import pdb; pdb.set_trace()
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return (out, )
    
class ConditioningSetTimestepRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_range"

    CATEGORY = "advanced/conditioning"

    def set_range(self, conditioning, start, end):
        import pdb; pdb.set_trace()
        c = node_helpers.conditioning_set_values(conditioning, {"start_percent": start,
                                                                "end_percent": end})
        return (c, )

class ConditioningAverageScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "conditioning_0": ("CONDITIONING", ), 
                    "conditioning_1": ("CONDITIONING", ),
                    "ratio": ("SIGMAS", ),
                    }
            }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "main"

    CATEGORY = "conditioning"

    @staticmethod
    def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength): #this function borrowed from comfyui
        #import pdb; pdb.set_trace()
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)
        return out

    @staticmethod
    def create_percent_array(steps):
        step_size = 1.0 / steps
        return [{"start_percent": i * step_size, "end_percent": (i + 1) * step_size} for i in range(steps)]

    def main(self, conditioning_0, conditioning_1, ratio):
        #import pdb; pdb.set_trace()
        steps = len(ratio)

        percents = self.create_percent_array(steps)

        cond = []
        for i in range(steps):
            average = self.addWeighted(conditioning_0, conditioning_1, ratio[i].item())
            cond += node_helpers.conditioning_set_values(average, {"start_percent": percents[i]["start_percent"], "end_percent": percents[i]["end_percent"]})

        return (cond,)


class StableCascade_StageB_Conditioning64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "conditioning": ("CONDITIONING",),
                              "stage_c": ("LATENT",),
                             }}
    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "set_prior"

    CATEGORY = "conditioning/stable_cascade"

    @cast_fp64
    def set_prior(self, conditioning, stage_c):
        c = []
        for t in conditioning:
            d = t[1].copy()
            d['stable_cascade_prior'] = stage_c['samples']
            n = [t[0], d]
            c.append(n)
        return (c, )


