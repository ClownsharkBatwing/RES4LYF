import torch
import base64
import pickle # used strictly for serializing conditioning in the ConditioningToBase64 and Base64ToConditioning nodes for API use. (Offloading T5 processing to another machine to avoid model shuffling.)

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import node_helpers

import functools
from .noise_classes import precision_tool
from copy import deepcopy

from .helper import initialize_or_scale
import torch.nn.functional as F



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


class CLIPTextEncodeFluxUnguided:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l, t5xxl):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]], )


class StyleModelApplyAdvanced: 
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "main"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Use with Flux Redux."

    def main(self, clip_vision_output, style_model, conditioning, strength=1.0):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        cond = strength * cond
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], cond), dim=1), t[1].copy()]
            c.append(n)
        return (c, )

class ConditioningZeroAndTruncate: 
    # needs updating to ensure dims are correct for arbitrary models without hardcoding. 
    # vanilla ConditioningZeroOut node doesn't truncate and SD3.5M degrades badly with large embeddings, even if zeroed out, as the negative conditioning
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"

    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Use for negative conditioning with SD3.5. ConditioningZeroOut does not truncate the embedding, \
                   which results in severe degradation of image quality with SD3.5 when the token limit is exceeded."

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros((1,2048), dtype=t[0].dtype, device=t[0].device)
                n = [torch.zeros((1,154,4096), dtype=t[0].dtype, device=t[0].device), d]
            c.append(n)
        return (c, )

class ConditioningZeroAndTruncate2: 
    # needs updating to ensure dims are correct for arbitrary models without hardcoding. 
    # vanilla ConditioningZeroOut node doesn't truncate and SD3.5M degrades badly with large embeddings, even if zeroed out, as the negative conditioning
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"

    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Use for negative conditioning with SD3.5. ConditioningZeroOut does not truncate the embedding, \
                   which results in severe degradation of image quality with SD3.5 when the token limit is exceeded."

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros((1,2048), dtype=t[0].dtype, device=t[0].device)
                n = [torch.zeros((1,154,4096), dtype=t[0].dtype, device=t[0].device), d]
            c.append(n)
        return (c, )

class ConditioningTruncate: 
    # needs updating to ensure dims are correct for arbitrary models without hardcoding. 
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"

    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Use for positive conditioning with SD3.5. Tokens beyond 77 result in degradation of image quality."

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = d["pooled_output"][:, :2048]
                n = [t[0][:, :154, :4096], d]
            c.append(n)
        return (c, )


class ConditioningMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                              "multiplier": ("FLOAT", {"default": 1.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01})
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

class ConditioningAverageScheduler: # don't think this is implemented correctly. needs to be reworked
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

    @precision_tool.cast_tensor
    def set_prior(self, conditioning, stage_c):
        c = []
        for t in conditioning:
            d = t[1].copy()
            d['stable_cascade_prior'] = stage_c['samples']
            n = [t[0], d]
            c.append(n)
        return (c, )









class Conditioning_Recast64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "cond_0": ("CONDITIONING",),
                             },
                "optional": { "cond_1": ("CONDITIONING",),}
                }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("cond_0_recast","cond_1_recast",)

    FUNCTION = "main"

    CATEGORY = "conditioning/"

    @precision_tool.cast_tensor
    def main(self, cond_0, cond_1 = None):
        cond_0[0][0] = cond_0[0][0].to(torch.float64)
        cond_0[0][1]["pooled_output"] = cond_0[0][1]["pooled_output"].to(torch.float64)
        
        if cond_1 is not None:
            cond_1[0][0] = cond_1[0][0].to(torch.float64)
            cond_1[0][1]["pooled_output"] = cond_1[0][1]["pooled_output"].to(torch.float64)

        return (cond_0, cond_1,)


class ConditioningToBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "conditioning"

    def notify(self, unique_id=None, extra_pnginfo=None, conditioning=None):
        conditioning_pickle = pickle.dumps(conditioning)
        conditioning_base64 = base64.b64encode(conditioning_pickle).decode('utf-8')
        text = [conditioning_base64]
        
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}

class Base64ToConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"

    CATEGORY = "conditioning"

    def main(self, data):
        conditioning_pickle = base64.b64decode(data)
        conditioning = pickle.loads(conditioning_pickle)
        return (conditioning,)








class RegionalMask(torch.nn.Module):
    def __init__(self, mask: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.start_percent = start_percent
        self.end_percent   = end_percent

    def __call__(self, transformer_options, *args, **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.mask
        return None
    
    

class RegionalConditioning(torch.nn.Module):
    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('region_cond', region_cond)
        self.start_percent = start_percent
        self.end_percent   = end_percent

    def __call__(self, transformer_options, *args,  **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.region_cond
        return None



class FluxRegionalPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "cond": ("CONDITIONING",),
        }, "optional": {
            "cond_regional": ("CONDITIONING_REGIONAL",),
            "mask": ("MASK",),
        }}

    RETURN_TYPES = ("CONDITIONING_REGIONAL","MASK",)
    RETURN_NAMES = ("cond_regional","mask_inv")
    FUNCTION = "main"

    CATEGORY = "conditioning"

    def main(self, cond, mask, cond_regional=[]):
        cond_regional = [*cond_regional]
        cond_regional.append({'mask': mask, 'cond': cond[0][0]})
        mask_inv = 1-mask
        return (cond_regional,mask_inv,)



class RegionalGenerateConditioningsAndMasks:
    def __init__(self, conditioning_regional, weight, start_percent, end_percent):
        self.conditioning_regional = conditioning_regional
        self.weight = weight
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, latent):
        b, c, h, w = latent.shape
        h //= 2  # 16x16 PE
        w //= 2
        img_len = h * w

        cond_r = torch.cat([cond_reg['cond'] for cond_reg in self.conditioning_regional], dim=1)
        text_len = 256 + cond_r.shape[1]  # 256 = main prompt tokens... half of t5, comfy issue

        regional_mask = torch.zeros((text_len + img_len, text_len + img_len), dtype=torch.bool)
        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool)
        union_masks = torch.zeros((img_len, img_len), dtype=torch.bool)

        conditioning_regional = [
            {
                'mask': torch.ones((1, h, w), dtype=torch.float64),
                'cond': torch.ones((1, 256, 4096), dtype=torch.float64),
            },
            *self.conditioning_regional,
        ]

        cond_len = 0
        for cond_reg_dict in conditioning_regional:
            cond_reg = cond_reg_dict['cond']
            region_mask = 1 - cond_reg_dict['mask'][0]
            region_mask = torch.nn.functional.interpolate(
                region_mask[None, None, :, :], (h, w), mode='nearest-exact'
            ).flatten().unsqueeze(1).repeat(1, cond_reg.size(1))

            cond_len_next = cond_len + cond_reg.shape[1]
            regional_mask[cond_len:cond_len_next, cond_len:cond_len_next] = True
            regional_mask[cond_len:cond_len_next, text_len:] = region_mask.transpose(-1, -2)
            regional_mask[text_len:, cond_len:cond_len_next] = region_mask

            img_size_masks = region_mask[:, :1].repeat(1, img_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(
                self_attend_masks, torch.logical_and(img_size_masks, img_size_masks_transpose)
            )
            union_masks = torch.logical_or(
                union_masks, torch.logical_or(img_size_masks, img_size_masks_transpose)
            )
            cond_len = cond_len_next

        background_masks = torch.logical_not(union_masks)
        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)
        regional_mask[text_len:, text_len:] = background_and_self_attend_masks

        regional_mask = RegionalMask(regional_mask, self.start_percent, self.end_percent)
        regional_conditioning = RegionalConditioning(cond_r, self.start_percent, self.end_percent)

        return regional_conditioning, regional_mask



class FluxRegionalConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "conditioning": ("CONDITIONING",),
            "conditioning_regional": ("CONDITIONING_REGIONAL",),
            "weight": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0,   "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent":   ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }, 
            "optional": {
                "weights": ("SIGMAS", ),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"

    CATEGORY = "conditioning"

    def main(self, conditioning, conditioning_regional, weight,start_percent, end_percent, start_step=0, end_step=30,  weights=None,  latent=None):
        default_dtype = torch.float64
        max_steps = 10000
        weights = initialize_or_scale(weights, weight, max_steps).to(default_dtype)
        weights = F.pad(weights, (0, max_steps), value=0.0)

        regional_generate_conditionings_and_masks_fn = RegionalGenerateConditioningsAndMasks(
            conditioning_regional=conditioning_regional,
            weight=weight,
            start_percent=start_percent,
            end_percent=end_percent,
        )

        conditioning[0][1]['regional_generate_conditionings_and_masks_fn'] = regional_generate_conditionings_and_masks_fn
        conditioning[0][1]['regional_conditioning_weights'] = weights
        return (conditioning,)
