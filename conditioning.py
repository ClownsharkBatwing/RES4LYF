import torch
import torch.nn.functional as F

from torch  import Tensor
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

import copy
import base64
import pickle # used strictly for serializing conditioning in the ConditioningToBase64 and Base64ToConditioning nodes for API use. (Offloading T5 processing to another machine to avoid model shuffling.)

import comfy.supported_models
import node_helpers
import gc


from .sigmas  import get_sigmas

from .helper  import initialize_or_scale, precision_tool, get_res4lyf_scheduler_list
from .latents import get_orthogonal, get_collinear
from .res4lyf import RESplain
from .beta.constants import MAX_STEPS


def multiply_nested_tensors(structure, scalar):
    if isinstance(structure, torch.Tensor):
        return structure * scalar
    elif isinstance(structure, list):
        return [multiply_nested_tensors(item, scalar) for item in structure]
    elif isinstance(structure, dict):
        return {key: multiply_nested_tensors(value, scalar) for key, value in structure.items()}
    else:
        return structure




class ConditioningOrthoCollin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_0": ("CONDITIONING", ), 
            "conditioning_1": ("CONDITIONING", ),
            "t5_strength":    ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
            "clip_strength":  ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "RES4LYF/conditioning"

    def combine(self, conditioning_0, conditioning_1, t5_strength, clip_strength):

        t5_0_1_collin         = get_collinear (conditioning_0[0][0], conditioning_1[0][0])
        t5_1_0_ortho          = get_orthogonal(conditioning_1[0][0], conditioning_0[0][0])

        t5_combined           = t5_0_1_collin + t5_1_0_ortho
        
        t5_1_0_collin         = get_collinear (conditioning_1[0][0], conditioning_0[0][0])
        t5_0_1_ortho          = get_orthogonal(conditioning_0[0][0], conditioning_1[0][0])

        t5_B_combined         = t5_1_0_collin + t5_0_1_ortho

        pooled_0_1_collin     = get_collinear (conditioning_0[0][1]['pooled_output'].unsqueeze(0), conditioning_1[0][1]['pooled_output'].unsqueeze(0)).squeeze(0)
        pooled_1_0_ortho      = get_orthogonal(conditioning_1[0][1]['pooled_output'].unsqueeze(0), conditioning_0[0][1]['pooled_output'].unsqueeze(0)).squeeze(0)

        pooled_combined       = pooled_0_1_collin + pooled_1_0_ortho
        
        #conditioning_0[0][0] = conditioning_0[0][0] + t5_strength * (t5_combined - conditioning_0[0][0])
        
        #conditioning_0[0][0] = t5_strength * t5_combined + (1-t5_strength) * t5_B_combined
        
        conditioning_0[0][0]  = t5_strength * t5_0_1_collin + (1-t5_strength) * t5_1_0_collin
        
        conditioning_0[0][1]['pooled_output'] = conditioning_0[0][1]['pooled_output'] + clip_strength * (pooled_combined - conditioning_0[0][1]['pooled_output'])

        return (conditioning_0, )



class CLIPTextEncodeFluxUnguided:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING","INT","INT",)
    RETURN_NAMES = ("conditioning", "clip_l_end", "t5xxl_end",)
    FUNCTION = "encode"
    CATEGORY = "RES4LYF/conditioning"

    def encode(self, clip, clip_l, t5xxl):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        clip_l_end=0
        for i in range(len(tokens['l'][0])):
            if tokens['l'][0][i][0] == 49407:
                clip_l_end=i
                break
        t5xxl_end=0
        for i in range(len(tokens['l'][0])):   # bug? should this be t5xxl?
            if tokens['t5xxl'][0][i][0] == 1:
                t5xxl_end=i
                break

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning = [[cond, output]]
        conditioning[0][1]['clip_l_end'] = clip_l_end
        conditioning[0][1]['t5xxl_end'] = t5xxl_end
        return (conditioning, clip_l_end, t5xxl_end,)


class StyleModelApplyAdvanced: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning":       ("CONDITIONING", ),
                "style_model":        ("STYLE_MODEL", ),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "strength":           ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
    }
        
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"
    DESCRIPTION  = "Use with Flux Redux."

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
    def INPUT_TYPES(cls):
        return { "required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "zero_out"
    CATEGORY     = "RES4LYF/conditioning"
    DESCRIPTION  = "Use for negative conditioning with SD3.5. ConditioningZeroOut does not truncate the embedding, \
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
    def INPUT_TYPES(cls):
        return { "required": {"conditioning": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "zero_out"
    CATEGORY     = "RES4LYF/conditioning"
    DESCRIPTION  = "Use for positive conditioning with SD3.5. Tokens beyond 77 result in degradation of image quality."

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
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING", ), 
                            "multiplier": ("FLOAT", {"default": 1.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01})
                            }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def main(self, conditioning, multiplier):
        c = multiply_nested_tensors(conditioning, multiplier)
        return (c,)



class ConditioningAdd:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning_1": ("CONDITIONING", ), 
                            "conditioning_2": ("CONDITIONING", ), 
                            "multiplier": ("FLOAT", {"default": 1.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01})
                            }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def main(self, conditioning_1, conditioning_2, multiplier):
        
        conditioning_1[0][0]                  += multiplier * conditioning_2[0][0]
        conditioning_1[0][1]['pooled_output'] += multiplier * conditioning_2[0][1]['pooled_output'] 
        
        return (conditioning_1,)




class ConditioningCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning_1": ("CONDITIONING", ), "conditioning_2": ("CONDITIONING", )}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "combine"
    CATEGORY     = "RES4LYF/conditioning"

    def combine(self, conditioning_1, conditioning_2):
        return (conditioning_1 + conditioning_2, )



class ConditioningAverage :
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to":          ("CONDITIONING", ), 
                "conditioning_from":        ("CONDITIONING", ),
                "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                }
            }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    CATEGORY     = "RES4LYF/conditioning"
    FUNCTION     = "addWeighted"

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
        out = []

        if len(conditioning_from) > 1:
            RESplain("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

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
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING", ),
                            "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                            "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                            }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "set_range"
    CATEGORY     = "RES4LYF/conditioning"

    def set_range(self, conditioning, start, end):
        c = node_helpers.conditioning_set_values(conditioning, {"start_percent": start,
                                                                "end_percent": end})
        return (c, )

class ConditioningAverageScheduler: # don't think this is implemented correctly. needs to be reworked
    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "conditioning_0": ("CONDITIONING", ), 
                    "conditioning_1": ("CONDITIONING", ),
                    "ratio": ("SIGMAS", ),
                    }
            }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    @staticmethod
    def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength): #this function borrowed from comfyui
        out = []

        if len(conditioning_from) > 1:
            RESplain("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

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
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "conditioning": ("CONDITIONING",),
                "stage_c":      ("LATENT",),
                }
            }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "set_prior"
    CATEGORY     = "RES4LYF/conditioning"

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
    def INPUT_TYPES(cls):
        return {"required": { "cond_0": ("CONDITIONING",),
                            },
                "optional": { "cond_1": ("CONDITIONING",),}
                }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("cond_0_recast","cond_1_recast",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/precision"

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES   = ("STRING",)
    RETURN_NAMES   = ("string",)
    FUNCTION       = "notify"
    OUTPUT_NODE    = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY       = "RES4LYF/utilities"

    def notify(self, unique_id=None, extra_pnginfo=None, conditioning=None):
        
        conditioning_pickle = pickle.dumps(conditioning)
        conditioning_base64 = base64.b64encode(conditioning_pickle).decode('utf-8')
        text = [conditioning_base64]
        
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                RESplain("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                RESplain("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/utilities"

    def main(self, data):
        conditioning_pickle = base64.b64decode(data)
        conditioning = pickle.loads(conditioning_pickle)
        return (conditioning,)






class ConditioningBatch4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "conditioning_0": ("CONDITIONING",),
                },
            "optional": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "conditioning_3": ("CONDITIONING",),
            }
            }
        
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def main(self, conditioning_0, conditioning_1=None, conditioning_2=None, conditioning_3=None, ):
        c = []
        c.append(conditioning_0[0])
        
        if conditioning_1 is not None:
            c.append(conditioning_1[0])
            
        if conditioning_2 is not None:
            c.append(conditioning_2[0])
            
        if conditioning_3 is not None:
            c.append(conditioning_3[0])
            
        return (c, )




class ConditioningBatch8:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "conditioning_0": ("CONDITIONING",),
                },
            "optional": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "conditioning_5": ("CONDITIONING",),
                "conditioning_6": ("CONDITIONING",),
                "conditioning_7": ("CONDITIONING",),
            }
            }
        
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def main(self, conditioning_0, conditioning_1=None, conditioning_2=None, conditioning_3=None, conditioning_4=None, conditioning_5=None, conditioning_6=None, conditioning_7=None, ):
        c = []
        c.append(conditioning_0[0])
        
        if conditioning_1 is not None:
            c.append(conditioning_1[0])
            
        if conditioning_2 is not None:
            c.append(conditioning_2[0])
            
        if conditioning_3 is not None:
            c.append(conditioning_3[0])
            
        if conditioning_4 is not None:
            c.append(conditioning_4[0])
            
        if conditioning_5 is not None:
            c.append(conditioning_5[0])
            
        if conditioning_6 is not None:
            c.append(conditioning_6[0])
            
        if conditioning_7 is not None:
            c.append(conditioning_7[0])
            
        return (c, )




class RegionalMask(torch.nn.Module):
    def __init__(self,
                mask                  : torch.Tensor,
                conditioning          : torch.Tensor,
                conditioning_regional : torch.Tensor,
                latent                : torch.Tensor,
                start_percent         : float,
                end_percent           : float,
                mask_type             : str,
                img_len               : int,
                text_len              : int,
                dtype                 : torch.dtype = torch.float16) -> None:
        
        super().__init__()
        self.mask                  = mask.clone().to('cuda')
        self.conditioning          = copy.deepcopy(conditioning)
        self.conditioning_regional = copy.deepcopy(conditioning_regional)
        self.latent                = latent.clone()
        self.start_percent         = start_percent
        self.end_percent           = end_percent
        self.mask_type             = mask_type
        self.img_len               = img_len
        self.text_len              = text_len
        self.dtype                 = dtype

    def __call__(self, transformer_options, weight=0, dtype=torch.float16, *args, **kwargs):
        sigma = transformer_options['sigmas'][0]
        #if self.start_percent <= 1 - sigma < self.end_percent:        # could be an issue, 1 - sigma? 
        if self.mask_type.startswith("gradient"):
            return self.mask.clone().to(sigma.device) * weight
        elif self.mask_type.startswith("boolean"):
            return self.mask.clone().to(sigma.device) > 0

    
class RegionalConditioning(torch.nn.Module):
    def __init__(self, conditioning: torch.Tensor, region_cond: torch.Tensor, region_pooled: torch.Tensor, start_percent: float, end_percent: float, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.conditioning  = conditioning
        self.region_cond   = region_cond.clone().to('cuda')
        self.region_pooled = region_pooled #.clone().to('cuda')
        self.start_percent = start_percent
        self.end_percent   = end_percent
        self.dtype         = dtype

    def __call__(self, transformer_options, dtype=torch.float16, *args,  **kwargs):
        sigma = transformer_options['sigmas'][0]
        #if self.start_percent <= 1 - sigma < self.end_percent:
        return self.region_cond.clone().to(sigma.device).to(dtype)
        return None
    
    def concat_cond(self, context, transformer_options, dtype=torch.float16, *args,  **kwargs):
        sigma = transformer_options['sigmas'][0]
        #if self.start_percent <= 1 - sigma < self.end_percent:
        region_cond = self.region_cond.clone().to(sigma.device).to(dtype)
        if self.conditioning is None:
            return self.region_cond.clone().to(sigma.device).to(dtype)
        else:
            return torch.cat([context, region_cond.clone().to(dtype)], dim=1)
        return None

    def concat_pooled(self, vec, transformer_options, dtype=torch.float16, *args,  **kwargs):
        sigma = transformer_options['sigmas'][0]
        #if self.start_percent <= 1 - sigma < self.end_percent:
        region_pooled = self.region_pooled.clone().to(sigma.device).to(dtype)
        if self.conditioning is None:
            return self.region_pooled.clone().to(sigma.device).to(dtype)
        else:
            return torch.cat([vec, region_pooled.clone().to(dtype)], dim=1)
        return None

class RectifiedFlow_RegionalPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "cond": ("CONDITIONING",),
            }, 
            "optional": {
                "cond_regional": ("CONDITIONING_REGIONAL",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING_REGIONAL","MASK",)
    RETURN_NAMES = ("cond_regional","mask_inv")
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def main(self, cond, mask, cond_regional=[]):
        cond_regional = [*cond_regional]
        if 'cond_pooled' not in cond[0][1]:
            cond[0][1]['cond_pooled'] = None
        cond_regional.append({'mask': mask, 'cond': cond[0][0], 'cond_pooled': cond[0][1]['pooled_output']})
        mask_inv      = 1-mask
        return (cond_regional,mask_inv,)

def fp_not(tensor):
    return 1 - tensor

def fp_or(tensor1, tensor2):
    return torch.maximum(tensor1, tensor2)

def fp_and(tensor1, tensor2):
    return torch.minimum(tensor1, tensor2)

def fp_and2(tensor1, tensor2):
    triu = torch.triu(torch.ones_like(tensor1))
    tril = torch.tril(torch.ones_like(tensor2))
    triu.diagonal().fill_(0.0)
    tril.diagonal().fill_(0.0)
    new_tensor = tensor1 * triu + tensor2 * tril
    new_tensor.diagonal().fill_(1.0)
    
    return new_tensor


class RegionalGenerateConditioningsAndMasks:
    def __init__(self, conditioning, conditioning_regional, weight, mask_type, model_config):
        self.conditioning          = conditioning
        self.conditioning_regional = conditioning_regional
        self.weight                = weight
        self.mask_type             = mask_type
        self.model_config          = model_config
    def __call__(self, latent, dtype=torch.float16):
        t = 1
        if latent.ndim == 4:
            b, c, h, w = latent.shape
        elif latent.ndim == 5:
            b, c, t, h, w = latent.shape
        if not isinstance(self.model_config, comfy.supported_models.Stable_Cascade_C):
            h //= 2  # 16x16 PE      patch_size = 2  1024x1024 rgb -> 128x128 16ch latent -> 64x64 img
            w //= 2
        self.img_len = img_len = h * w
        self.h = h
        self.w = w
        CROSS_ATTN_ONLY = False
        text_register_tokens = 0
        if   isinstance(self.model_config, comfy.supported_models.SD3):
            text_len_base = 154
            num_channels  = 4096
        elif isinstance(self.model_config, comfy.supported_models.Flux) or isinstance(self.model_config, comfy.supported_models.FluxSchnell):
            text_len_base = 256
            num_channels  = 4096
        elif isinstance(self.model_config, comfy.supported_models.AuraFlow):
            text_len_base = 256
            num_channels  = 2048
            #text_register_tokens = 8
        elif isinstance(self.model_config, comfy.supported_models.Stable_Cascade_C):
            text_len_base = 77
            num_channels  = 1280
            text_register_tokens = 8
        elif isinstance(self.model_config, comfy.supported_models.WAN21_T2V) or isinstance(self.model_config, comfy.supported_models.WAN21_I2V):
            text_len_base = 512
            num_channels  = 5120
            CROSS_ATTN_ONLY = True
        else:
            text_len_base = 154    #UGLY
            num_channels  = 4096
        
        if 'cond_pooled' in self.conditioning_regional[0] and self.conditioning_regional[0]['cond_pooled'] is not None:
            cond_pooled = torch.cat([cond_reg['cond_pooled'] for cond_reg in self.conditioning_regional], dim=1)
        else:
            cond_pooled = None

        cond_r = torch.cat([cond_reg['cond'] for cond_reg in self.conditioning_regional], dim=1)           #1,256,2048 aura cond     
        
        if self.conditioning is not None:
            self.text_len = text_len = text_len_base + cond_r.shape[1]  # 256 = main prompt tokens... half of t5, comfy issue
            conditioning_regional = [
                {
                    'mask': torch.ones((1,             h,    w)).to(dtype),
                    'cond': torch.ones((1, text_len_base, num_channels)).to(dtype),
                },
                *self.conditioning_regional,
            ]
        else:
            self.text_len = text_len              = cond_r.shape[1] + text_register_tokens # 256 = main prompt tokens... half of t5, comfy issue        # gets set to 308 with sd35m. 154 * 2 = 308 (THIS IS WITH CFG)
            conditioning_regional = self.conditioning_regional
        
        if isinstance(self.model_config, comfy.supported_models.Stable_Cascade_C):
            self.text_off = text_off = 0
        else:
            self.text_off = text_off = text_len
            
        if CROSS_ATTN_ONLY:
            cross_attn_mask    = torch.zeros((t*img_len, text_len), dtype=dtype)

            self_attn_mask     = torch.zeros((        t * img_len,        t * img_len), dtype=dtype)
            self_attn_mask_bkg = torch.zeros((        t * img_len,        t * img_len), dtype=dtype)
            
            prev_len = 0
            for cond_reg_dict in conditioning_regional:
                cond_reg    = cond_reg_dict['cond']
                region_mask = cond_reg_dict['mask'][0].to(dtype)
                
                if region_mask.ndim == 3:
                    t_region_mask = region_mask.shape[0]
                else:
                    t_region_mask = 1
                    region_mask.unsqueeze_(0)
                img2txt_mask    = torch.nn.functional.interpolate(region_mask[None, None, :, :].to(torch.float16), (t_region_mask, h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1)
                
                if t_region_mask == 1:
                    img2txt_mask = img2txt_mask.repeat(1, cond_reg.shape[1])
                
                #img2txt_mask    = torch.nn.functional.interpolate(region_mask[None, None, :, :], (h, w), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, cond_reg.shape[1])  #cond_reg.shape(1) = 256   4096/256 = 16
                txt2img_mask    = img2txt_mask.transpose(-1, -2)
                
                #img2txt_mask_sq = torch.nn.functional.interpolate(region_mask[None, None, :, :], (h, w), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, img_len)
                #txt2img_mask_sq = img2txt_mask_sq.transpose(-1, -2)
                
                curr_len = prev_len + cond_reg.shape[1]
                
                if t_region_mask == 1:
                    cross_attn_mask[:, prev_len:curr_len] = img2txt_mask.repeat(t,1)
                else:
                    cross_attn_mask[:, prev_len:curr_len] = img2txt_mask
                #self_attn_mask = fp_or(self_attn_mask, fp_and(img2txt_mask_sq.repeat(t,t), txt2img_mask_sq.repeat(t,t)))
                
                region_mask_flat    = torch.nn.functional.interpolate(region_mask[None, None, :, :].to(torch.float16), (t_region_mask, h, w), mode='nearest-exact').to(dtype).flatten()
                
                if t_region_mask > 1:
                    self_attn_mask = fp_or(self_attn_mask, region_mask_flat.unsqueeze(0) * region_mask_flat.unsqueeze(1))
                else:
                    self_attn_mask = fp_or(self_attn_mask, region_mask_flat.repeat(t).unsqueeze(0) * region_mask_flat.repeat(t).unsqueeze(1))
                
                prev_len = curr_len
                
            all_attn_mask = torch.cat((cross_attn_mask,self_attn_mask), dim=1)
            
            
            
        else:
            all_attn_mask      = torch.zeros((text_off+t*img_len, text_len+t*img_len), dtype=dtype)
            
            self_attn_mask     = torch.zeros((       t * img_len,        t * img_len), dtype=dtype)
            self_attn_mask_bkg = torch.zeros((       t * img_len,        t * img_len), dtype=dtype)
            
            prev_len = 0
            for cond_reg_dict in conditioning_regional:
                cond_reg    = cond_reg_dict['cond']
                region_mask = cond_reg_dict['mask'][0].to(dtype)

                img2txt_mask    = torch.nn.functional.interpolate(region_mask[None, None, :, :].to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, cond_reg.shape[1])  #cond_reg.shape(1) = 256   4096/256 = 16
                txt2img_mask    = img2txt_mask   .transpose(-1, -2)
                
                img2txt_mask_sq = torch.nn.functional.interpolate(region_mask[None, None, :, :].to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                txt2img_mask_sq = img2txt_mask_sq.transpose(-1, -2)

                curr_len = prev_len + cond_reg.shape[1]
                
                all_attn_mask[prev_len:curr_len, prev_len:curr_len] = 1.0           # self             TXT 2 TXT
                all_attn_mask[prev_len:curr_len, text_len:        ] = txt2img_mask.repeat(1,t)  # cross            TXT 2 regional IMG
                all_attn_mask[text_off:        , prev_len:curr_len] = img2txt_mask.repeat(t,1)  # cross   regional IMG 2 TXT

                self_attn_mask = fp_or(self_attn_mask, fp_and(img2txt_mask_sq.repeat(t,t), txt2img_mask_sq.repeat(t,t)))
                
                prev_len = curr_len

            all_attn_mask[text_off:, text_len:] = fp_or(self_attn_mask, self_attn_mask_bkg) #combine foreground/background self-attn

        all_attn_mask         = RegionalMask(all_attn_mask, self.conditioning, self.conditioning_regional, latent, self.start_percent, self.end_percent, self.mask_type, img_len, text_len)
        regional_conditioning = RegionalConditioning(self.conditioning, cond_r, cond_pooled, self.start_percent, self.end_percent)

        if self.mask_type.endswith("_masked"):
            trimask = torch.tril(torch.ones(img_len, img_len)).to(all_attn_mask.mask.dtype).to(all_attn_mask.mask.device)
            trimask.diagonal().fill_(0.0)
            all_attn_mask.mask[text_off:,text_len:] = fp_or(trimask, all_attn_mask.mask[text_off:,text_len:])
            
        if self.mask_type.endswith("_unmasked"):
            trimask = 1-torch.tril(torch.ones(img_len, img_len)).to(all_attn_mask.mask.dtype).to(all_attn_mask.mask.device)
            trimask.diagonal().fill_(0.0)
            all_attn_mask.mask[text_off:,text_len:] = fp_or(trimask, all_attn_mask.mask[text_off:,text_len:])

        if self.mask_type.startswith("boolean"):
            all_attn_mask.mask = all_attn_mask.mask > 0
            
        torch.cuda.empty_cache() 
        gc.collect()

        return regional_conditioning, all_attn_mask


class RectifiedFlow_RegionalConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "mask_weight":           ("FLOAT",      {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "self_attn_floor":       ("FLOAT",      {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "start_percent":         ("FLOAT",      {"default": 0,   "min": 0.0,      "max": 1.0,     "step": 0.01}),
                "end_percent":           ("FLOAT",      {"default": 1.0, "min": 0.0,      "max": 1.0,     "step": 0.01}),
                "mask_type":             (["gradient", "boolean"], {"default": "gradient"}),
            }, 
            "optional": {
                "conditioning":          ("CONDITIONING",),
                "conditioning_regional": ("CONDITIONING_REGIONAL",),
                "mask_weights":          ("SIGMAS", ),
                "self_attn_floors":      ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def main(self,
            conditioning_regional,
            mask_weight      = 1.0,
            start_percent    = 0.0,
            end_percent      = 1.0,
            start_step       = 0,
            end_step         = -1,
            conditioning     = None,
            mask_weights     = None,
            self_attn_floors = None,
            self_attn_floor  = 0.0,
            mask_type        = "gradient",
            model_config     = None,
            ):
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        weight, weights = mask_weight, mask_weights
        floor, floors = self_attn_floor, self_attn_floors
        default_dtype = torch.float64
        
        weights = initialize_or_scale(weights, weight, MAX_STEPS).to(default_dtype)
        weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        floors = initialize_or_scale(floors, floor, MAX_STEPS).to(default_dtype)
        floors = F.pad(floors, (0, MAX_STEPS), value=0.0)

        regional_generate_conditionings_and_masks_fn = RegionalGenerateConditioningsAndMasks(conditioning, conditioning_regional, weight, start_percent, end_percent, mask_type, model_config)
        
        pooled_len = 0
        if conditioning is not None:
            text_len_base = conditioning[0][0].shape[1]
            pooled_len = conditioning[0][1]['pooled_output'].shape[1] if 'pooled_output' in conditioning[0][1] else 0

        else:
            if   isinstance(model_config, comfy.supported_models.SD3):
                text_len_base = 154
                pooled_len    = 2048
            elif isinstance(model_config, comfy.supported_models.Flux) or isinstance(model_config, comfy.supported_models.FluxSchnell) or isinstance(model_config, comfy.supported_models.AuraFlow):
                text_len_base = 256
                pooled_len    = 768

            elif isinstance(model_config, comfy.supported_models.Stable_Cascade_C):
                text_len_base = 85
                pooled_len    = 1280

        if conditioning is None:
            conditioning = [
                                [
                                    torch.zeros_like(conditioning_regional[0]['cond']),
                                    {'pooled_output':
                                        torch.zeros((1,pooled_len), dtype=conditioning_regional[0]['cond'].dtype, device=conditioning_regional[0]['cond'].device),
                                    }
                                ],
            ]

        conditioning[0][1]['regional_generate_conditionings_and_masks_fn'] = regional_generate_conditionings_and_masks_fn
        conditioning[0][1]['regional_conditioning_weights']                = weights
        conditioning[0][1]['regional_conditioning_floors']                 = floors
        return (copy.deepcopy(conditioning),)












class ClownRegionalConditioningAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "weight":            ("FLOAT",                 {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed":      ("FLOAT",                 {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "mask_type":         (["gradient", "boolean"], {"default": "boolean"}),
                "invert_mask":       ("BOOLEAN",               {"default": False}),
            }, 
            "optional": {
                "positive_masked":   ("CONDITIONING", ),
                "positive_unmasked": ("CONDITIONING", ),
                "mask":              ("MASK", ),
                "weights":           ("SIGMAS", ),
                "region_bleeds":     ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def create_callback(self, **kwargs):
        def callback(model):
            kwargs["model"] = model  
            pos_cond, = self.prepare_regional_cond(**kwargs)
            return pos_cond
        return callback

    def main(self,
            weight                   : float  = 1.0,
            start_percent            : float  = 0.0,
            end_percent              : float  = 1.0,
            weight_scheduler                  = None,
            start_step               : int    = 0,
            end_step                 : int    = 10000,
            positive_masked                   = None,
            positive_unmasked                 = None,
            weights                  : Tensor = None,
            region_bleeds            : Tensor = None,
            region_bleed             : float  = 1.0,
            mask_type                : str    = "gradient",
            mask                              = None,
            invert_mask              : bool   = False
            ) -> Tuple[Tensor]:
        
        default_dtype  = torch.float64
        default_device = torch.device("cuda") 
        
        if weights is None:
            weights       = torch.full((MAX_STEPS,), weight,       dtype=default_dtype, device=default_device)
            
        if region_bleeds is None:
            region_bleeds = torch.full((MAX_STEPS,), region_bleed, dtype=default_dtype, device=default_device)
        
        positive, = ClownRegionalConditioning().main(
                                                weight            = weight,
                                                start_percent     = start_percent,
                                                end_percent       = end_percent,
                                                weight_scheduler  = weight_scheduler,
                                                start_step        = start_step,
                                                end_step          = end_step,
                                                positive_masked   = positive_masked,
                                                positive_unmasked = positive_unmasked,
                                                weights           = weights,
                                                region_bleeds     = region_bleeds,
                                                region_bleed      = region_bleed,
                                                mask_type         = mask_type,
                                                mask              = mask,
                                                invert_mask       = invert_mask,
                                                )
        return (positive,)






class ClownRegionalConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "weight":                  ("FLOAT",                                     {"default": 1.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed":            ("FLOAT",                                     {"default": 0.0, "min":  -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed_start_step": ("INT",                                       {"default": 0,   "min":  0,        "max": 10000}),
                "weight_scheduler":        (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                "start_step":              ("INT",                                       {"default": 0,   "min":  0,        "max": 10000}),
                "end_step":                ("INT",                                       {"default": -1,  "min": -1,        "max": 10000}),
                "mask_type":               (["gradient", "gradient_masked", "gradient_unmasked", "boolean", "boolean_masked", "boolean_unmasked"],                     {"default": "gradient"}),
                "narcissism_area":         (["masked", "unmasked", "off"],               {"default": "masked"}),
                "narcissism_start_step":   ("INT",                                       {"default": 0,   "min": -1,        "max": 10000}),
                "narcissism_end_step":     ("INT",                                       {"default": 5,   "min": -1,        "max": 10000}),
                "invert_mask":             ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "positive_masked":         ("CONDITIONING", ),
                "positive_unmasked":       ("CONDITIONING", ),
                "mask":                    ("MASK", ),
                "weights":                 ("SIGMAS", ),
                "region_bleeds":           ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def create_callback(self, **kwargs):
        def callback(model):
            kwargs["model"] = model  
            pos_cond, = self.prepare_regional_cond(**kwargs)
            return pos_cond
        return callback

    def main(self,
            weight                   : float  = 1.0,
            start_percent            : float  = 0.0,
            end_percent              : float  = 1.0,
            weight_scheduler                  = None,
            start_step               : int    = 0,
            end_step                 : int    = -1,
            positive_masked                   = None,
            positive_unmasked                 = None,
            weights                  : Tensor = None,
            region_bleeds            : Tensor = None,
            region_bleed             : float  = 0.0,
            region_bleed_start_step  : int    = 0,
            mask_type                : str    = "boolean",
            mask                              = None,
            narcissism_area          : str    = "masked",
            narcissism_start_step    : int    = 0,
            narcissism_end_step      : int    = 5,
            invert_mask              : bool   = False
            ) -> Tuple[Tensor]:
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if narcissism_end_step == -1:
            narcissism_end_step = MAX_STEPS
        
        callback = self.create_callback(weight                   = weight,
                                        start_percent            = start_percent,
                                        end_percent              = end_percent,
                                        weight_scheduler         = weight_scheduler,
                                        start_step               = start_step,
                                        end_step                 = end_step,
                                        weights                  = weights,
                                        region_bleeds            = region_bleeds,
                                        region_bleed             = region_bleed,
                                        region_bleed_start_step  = region_bleed_start_step,
                                        mask_type                = mask_type,
                                        mask                     = mask,
                                        invert_mask              = invert_mask,
                                        positive_masked          = positive_masked,
                                        positive_unmasked        = positive_unmasked,
                                        narcissism_area          = narcissism_area,
                                        narcissism_start_step    = narcissism_start_step,
                                        narcissism_end_step      = narcissism_end_step,
                                        )
        pooled_len = 768
        if positive_masked is not None:
            pooled     = positive_masked[0][1].get('pooled_output')
            pooled_len = pooled.shape[-1] if pooled is not None else pooled_len
            positive = [[
                torch.zeros_like(positive_masked[0][0]),
                {"pooled_output": torch.zeros( (1,pooled_len), dtype=positive_masked[0][0].dtype, device=positive_masked[0][0].device  )},
            ]]
        elif positive_unmasked is not None:
            pooled     = positive_unmasked[0][1].get('pooled_output')
            pooled_len = pooled.shape[-1] if pooled is not None else pooled_len
            positive = [[
                torch.zeros_like(positive_unmasked[0][0]),
                {"pooled_output": torch.zeros( (1,pooled_len), dtype=positive_unmasked[0][0].dtype, device=positive_unmasked[0][0].device  )},
            ]]

        
        positive[0][1]['callback_regional'] = callback
        
        return (positive,)



    def prepare_regional_cond(self,
                                model,
                                weight                   : float  = 1.0,
                                start_percent            : float  = 0.0,
                                end_percent              : float  = 1.0,
                                weight_scheduler                  = None,
                                start_step               : int    = 0,
                                end_step                 : int    = -1,
                                positive_masked                   = None,
                                positive_unmasked                 = None,
                                weights                  : Tensor = None,
                                region_bleeds            : Tensor = None,
                                region_bleed             : float  = 0.0,
                                region_bleed_start_step  : int    = 0,
                                mask_type                : str    = "gradient",
                                mask                              = None,
                                invert_mask              : bool   = False,
                                narcissism_area       : str       = "on",
                                narcissism_start_step : int       = 0,
                                narcissism_end_step   : int       = 5,
                                ) -> Tuple[Tensor]:

        default_dtype  = torch.float64
        default_device = torch.device("cuda") 
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if weights is None and weight_scheduler != "constant":
            total_steps = end_step - start_step
            weights     = get_sigmas(model, weight_scheduler, total_steps, 1.0).to(dtype=default_dtype, device=default_device) #/ model.inner_model.inner_model.model_sampling.sigma_max  #scaling doesn't matter as this is a flux-only node
            prepend     = torch.zeros(start_step,                                  dtype=default_dtype, device=default_device)
            weights     = torch.cat((prepend, weights), dim=0)
        
        if invert_mask and mask is not None:
            mask = 1-mask

        #weight, weights = mask_weight, mask_weights
        floor, floors = region_bleed, region_bleeds
        
        weights = initialize_or_scale(weights, weight, end_step).to(default_dtype).to(default_device)
        weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        prepend    = torch.full((region_bleed_start_step,),  0.0, dtype=default_dtype, device=default_device)
        floors  = initialize_or_scale(floors,  floor,  end_step).to(default_dtype).to(default_device)
        floors  = F.pad(floors,  (0, MAX_STEPS), value=0.0)
        floors  = torch.cat((prepend, floors), dim=0)

        if (positive_masked is None) and (positive_unmasked is None):
            positive = None

        elif mask is not None:
            if   isinstance(model.model.model_config, comfy.supported_models.SD3):
                text_len_base = 154
                pooled_len    = 2048
                text_channels = 4096
            elif isinstance(model.model.model_config, comfy.supported_models.Flux) \
                or isinstance(model.model.model_config, comfy.supported_models.FluxSchnell):
                text_len_base = 256
                pooled_len    = 768
                text_channels = 4096
            elif isinstance(model.model.model_config, comfy.supported_models.AuraFlow):
                text_len_base = 256
                pooled_len    = 768
                text_channels = 2048
            elif isinstance(model.model.model_config, comfy.supported_models.Stable_Cascade_C):
                text_len_base = 77
                pooled_len    = 1280
                text_channels = 1280
            
            #elif isinstance(model.model.model_config, comfy.supported_models.Cascade_StageC):
            if positive_masked is None:    
                if positive_masked is None:
                    positive_masked = [[
                        torch.zeros((1, text_len_base, text_channels)),
                        {'pooled_output': torch.zeros((1, pooled_len))}
                        ]]
            if positive_unmasked is None:    
                if positive_unmasked is None:
                    positive_unmasked = [[
                        torch.zeros((1, text_len_base, text_channels)),
                        {'pooled_output': torch.zeros((1, pooled_len))}
                        ]]
            cond_regional, mask_inv     = RectifiedFlow_RegionalPrompt().main(cond=positive_masked,                                    mask=mask)
            cond_regional, mask_inv_inv = RectifiedFlow_RegionalPrompt().main(cond=positive_unmasked    , cond_regional=cond_regional, mask=mask_inv)
            
            positive, = RectifiedFlow_RegionalConditioning().main( 
                                                        conditioning_regional = cond_regional,
                                                        self_attn_floor       = floor,
                                                        self_attn_floors      = floors,
                                                        mask_weight           = weight,
                                                        mask_weights          = weights,
                                                        start_percent         = start_percent,
                                                        end_percent           = end_percent,
                                                        mask_type             = mask_type,
                                                        model_config          = model.model.model_config,
                                                        )
            positive_masked_tokens   = positive_masked[0][0]  .shape[1]
            positive_unmasked_tokens = positive_unmasked[0][0].shape[1]
            
            positive_min_tokens = min(positive_masked_tokens, positive_unmasked_tokens)
            
            positive[0][0] = (positive_masked[0][0][:,:positive_min_tokens,:] + positive_unmasked[0][0][:,:positive_min_tokens,:]) / 2
            
            if   positive_masked_tokens < positive_unmasked_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_unmasked[0][0][:,positive_min_tokens:,:]), dim=1)
            elif positive_masked_tokens > positive_unmasked_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_masked  [0][0][:,positive_min_tokens:,:]), dim=1)
                
            if 'pooled_output' in positive[0][1] and positive_masked[0][1]['pooled_output'] is not None:
                positive_masked_pooled_tokens   = positive_masked[0][1]['pooled_output'].shape[1]
                positive_unmasked_pooled_tokens = positive_unmasked[0][1]['pooled_output'].shape[1]
                
                positive_min_pooled_tokens = min(positive_masked_pooled_tokens, positive_unmasked_pooled_tokens)
                
                positive[0][1]['pooled_output'] = (positive_masked[0][1]['pooled_output'][:,:positive_min_pooled_tokens] + positive_unmasked[0][1]['pooled_output'][:,:positive_min_pooled_tokens]) / 2
                
                if   positive_masked_pooled_tokens < positive_unmasked_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_unmasked[0][1]['pooled_output'][:,positive_min_pooled_tokens:]), dim=1)
                elif positive_masked_pooled_tokens > positive_unmasked_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_masked  [0][1]['pooled_output'][:,positive_min_pooled_tokens:]), dim=1)
                
        else:
            positive = positive_masked
            
        if   mask is not None and narcissism_area == "masked":
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask.clone()
            if 'pooled_output' in positive[0][1]:
                positive[0][1]['pooled_output'] = positive_unmasked[0][1]['pooled_output']
            
        elif mask is not None and narcissism_area == "unmasked":
            positive[0][1]['regional_conditioning_mask_orig'] = mask.clone()
            if 'pooled_output' in positive[0][1]:
                positive[0][1]['pooled_output'] = positive_masked[0][1]['pooled_output']
            
        else:
            positive[0][1]['regional_conditioning_mask_orig'] = None
        
        positive[0][1]['narcissism_start_step'] = narcissism_start_step
        positive[0][1]['narcissism_end_step']   = narcissism_end_step
        
        return (positive,)















class ClownRegionalConditioning3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "weight":            ("FLOAT",                                     {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed":      ("FLOAT",                                     {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed_start_step": ("INT",                                       {"default": 0,   "min":  0,        "max": 10000}),
                "weight_scheduler":  (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                "start_step":        ("INT",                                       {"default": 0,    "min":  0,        "max": 10000}),
                "end_step":          ("INT",                                       {"default": 100,  "min": -1,        "max": 10000}),
                "mask_type":         (["gradient", "gradient_masked", "gradient_unmasked", "boolean", "boolean_masked", "boolean_unmasked"],                     {"default": "gradient"}),
                "narcissism_area":      (["A", "B", "AB", "unmasked", "off"],         {"default": "masked"}),
                "narcissism_start_step":("INT",                                 {"default": 0,   "min": -1,        "max": 10000}),
                "narcissism_end_step":  ("INT",                                 {"default": 5,   "min": -1,        "max": 10000}),
                "invert_mask":       ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "positive_A":        ("CONDITIONING", ),
                "positive_B":        ("CONDITIONING", ),
                "positive_unmasked": ("CONDITIONING", ),
                "mask_A":            ("MASK", ),
                "mask_B":            ("MASK", ),
                "weights":           ("SIGMAS", ),
                "region_bleeds":     ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/conditioning"

    def create_callback(self, **kwargs):
        def callback(model):
            kwargs["model"] = model  
            pos_cond, = self.prepare_regional_cond(**kwargs)
            return pos_cond
        return callback

    def main(self,
            weight                   : float  = 1.0,
            start_percent            : float  = 0.0,
            end_percent              : float  = 1.0,
            weight_scheduler                  = None,
            start_step               : int    = 0,
            end_step                 : int    = -1,
            positive_A                   = None,
            positive_B                        = None,
            positive_unmasked                 = None,
            weights                  : Tensor = None,
            region_bleeds            : Tensor = None,
            region_bleed             : float  = 0.0,
            region_bleed_start_step  : int    = 0,

            mask_type                : str    = "boolean",
            mask_A                              = None,
            mask_B                              = None,
            narcissism_area       : str    = "AB",
            narcissism_start_step : int    = 0,
            narcissism_end_step   : int    = 5,
            invert_mask              : bool   = False
            ) -> Tuple[Tensor]:
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        callback = self.create_callback(weight                   = weight,
                                        start_percent            = start_percent,
                                        end_percent              = end_percent,
                                        weight_scheduler         = weight_scheduler,
                                        start_step               = start_step,
                                        end_step                 = end_step,
                                        weights                  = weights,
                                        region_bleeds            = region_bleeds,
                                        region_bleed             = region_bleed,
                                        region_bleed_start_step  = region_bleed_start_step,
                                        mask_type                = mask_type,
                                        mask_A                   = mask_A,
                                        mask_B                   = mask_B,
                                        invert_mask              = invert_mask,
                                        positive_A               = positive_A,
                                        positive_B               = positive_B,
                                        positive_unmasked        = positive_unmasked,
                                        narcissism_area       = narcissism_area,
                                        narcissism_start_step = narcissism_start_step,
                                        narcissism_end_step   = narcissism_end_step,
                                        )
        pooled_len = 768
        if positive_A is not None:
            pooled     = positive_A[0][1].get('pooled_output')
            pooled_len = pooled.shape[-1] if pooled is not None else pooled_len
            positive = [[
                torch.zeros_like(positive_A[0][0]),
                {"pooled_output": torch.zeros( (1,pooled_len), dtype=positive_A[0][0].dtype, device=positive_A[0][0].device  )},
                #{}
                #{"pooled_output": torch.zeros_like(positive_masked[0][1]['pooled_output'])}
            ]]
            
        if positive_B is not None:
            pooled     = positive_B[0][1].get('pooled_output')
            pooled_len = pooled.shape[-1] if pooled is not None else pooled_len
            positive = [[
                torch.zeros_like(positive_B[0][0]),
                {"pooled_output": torch.zeros( (1,pooled_len), dtype=positive_B[0][0].dtype, device=positive_B[0][0].device  )},
                #{}
                #{"pooled_output": torch.zeros_like(positive_masked[0][1]['pooled_output'])}
            ]]
            
        elif positive_unmasked is not None:
            pooled     = positive_unmasked[0][1].get('pooled_output')
            pooled_len = pooled.shape[-1] if pooled is not None else pooled_len
            positive = [[
                torch.zeros_like(positive_unmasked[0][0]),
                {"pooled_output": torch.zeros( (1,pooled_len), dtype=positive_unmasked[0][0].dtype, device=positive_unmasked[0][0].device  )},
                #{"pooled_output": torch.zeros_like(positive_unmasked[0][1]['pooled_output'])}
            ]]
        """positive = [[
            torch.zeros((1, 256, 4096)),
            {'pooled_output': torch.zeros((1, 768))}
            ]]"""
        
        positive[0][1]['callback_regional'] = callback
        
        return (positive,)



    def prepare_regional_cond(self,
                                model,
                                weight                   : float  = 1.0,
                                start_percent            : float  = 0.0,
                                end_percent              : float  = 1.0,
                                weight_scheduler                  = None,
                                start_step               : int    =  0,
                                end_step                 : int    = -1,
                                positive_A                        = None,
                                positive_B                        = None,

                                positive_unmasked                 = None,
                                weights                  : Tensor = None,
                                region_bleeds            : Tensor = None,
                                region_bleed             : float  = 0.0,
                                region_bleed_start_step  : int    = 0,

                                mask_type                : str    = "boolean",
                                mask_A                            = None,
                                mask_B                            = None,
                                invert_mask              : bool   = False,
                                narcissism_area       : str    = "AB",
                                narcissism_start_step : int    = 0,
                                narcissism_end_step   : int    = 5,
                                ) -> Tuple[Tensor]:

        default_dtype  = torch.float64
        default_device = torch.device("cuda") 
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if weights is None and weight_scheduler != "constant":
            total_steps = end_step - start_step
            weights     = get_sigmas(model, weight_scheduler, total_steps, 1.0).to(dtype=default_dtype, device=default_device) #/ model.inner_model.inner_model.model_sampling.sigma_max  #scaling doesn't matter as this is a flux-only node
            prepend     = torch.zeros(start_step,                                  dtype=default_dtype, device=default_device)
            weights     = torch.cat((prepend, weights), dim=0)
        
        if invert_mask and mask_A is not None:
            mask_A = 1-mask_A
            
        if invert_mask and mask_B is not None:
            mask_B = 1-mask_B
        
        mask_AB_inv = torch.ones_like(mask_A) - mask_A - mask_B

        #weight, weights = mask_weight, mask_weights
        floor, floors = region_bleed, region_bleeds
        
        weights = initialize_or_scale(weights, weight, end_step).to(default_dtype)
        weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        prepend    = torch.full((region_bleed_start_step,),  0.0, dtype=default_dtype, device=default_device)
        floors  = initialize_or_scale(floors,  floor,  end_step).to(default_dtype).to(default_device)
        floors  = F.pad(floors,  (0, MAX_STEPS), value=0.0)
        floors  = torch.cat((prepend, floors), dim=0)

        if (positive_A is None) and (positive_B is None) and (positive_unmasked is None):
            positive = None

        elif mask_A is not None:
            if   isinstance(model.model.model_config, comfy.supported_models.SD3):
                text_len_base = 154
                pooled_len    = 2048
                text_channels = 4096
            elif isinstance(model.model.model_config, comfy.supported_models.Flux) \
                or isinstance(model.model.model_config, comfy.supported_models.FluxSchnell):
                text_len_base = 256
                pooled_len    = 768
                text_channels = 4096
            elif isinstance(model.model.model_config, comfy.supported_models.AuraFlow):
                text_len_base = 256
                pooled_len    = 768
                text_channels = 2048
            elif isinstance(model.model.model_config, comfy.supported_models.Stable_Cascade_C):
                text_len_base = 77
                pooled_len    = 1280
                text_channels = 1280
            
            #elif isinstance(model.model.model_config, comfy.supported_models.Cascade_StageC):
            if positive_A is None:    
                if positive_A is None:
                    positive_A = [[
                        torch.zeros((1, text_len_base, text_channels)),
                        {'pooled_output': torch.zeros((1, pooled_len))}
                        ]]
            if positive_B is None:    
                if positive_B is None:
                    positive_B = [[
                        torch.zeros((1, text_len_base, text_channels)),
                        {'pooled_output': torch.zeros((1, pooled_len))}
                        ]]
            if positive_unmasked is None:    
                if positive_unmasked is None:
                    positive_unmasked = [[
                        torch.zeros((1, text_len_base, text_channels)),
                        {'pooled_output': torch.zeros((1, pooled_len))}
                        ]]
            cond_regional, mask_inv     = RectifiedFlow_RegionalPrompt().main(cond=positive_A,                                     mask=mask_A)
            cond_regional, mask_inv     = RectifiedFlow_RegionalPrompt().main(cond=positive_B,        cond_regional=cond_regional, mask=mask_B)
            cond_regional, mask_inv_inv = RectifiedFlow_RegionalPrompt().main(cond=positive_unmasked, cond_regional=cond_regional, mask=mask_AB_inv)
            
            positive, = RectifiedFlow_RegionalConditioning().main( 
                                                        conditioning_regional = cond_regional,
                                                        self_attn_floor       = floor,
                                                        self_attn_floors      = floors,
                                                        mask_weight           = weight,
                                                        mask_weights          = weights,
                                                        start_percent         = start_percent,
                                                        end_percent           = end_percent,
                                                        mask_type             = mask_type,
                                                        model_config          = model.model.model_config,
                                                        )
            positive_A_tokens        = positive_A[0][0].shape[1]
            positive_B_tokens        = positive_B[0][0].shape[1]
            positive_unmasked_tokens = positive_unmasked[0][0].shape[1]
            
            values = sorted([positive_A_tokens, positive_B_tokens, positive_unmasked_tokens])

            positive_min_tokens  = values[0]
            positive_mid_tokens  = values[1]
            positive_max_tokens  = values[2]
            
            positive[0][0] = (positive_A[0][0][:,:positive_min_tokens,:] + positive_B[0][0][:,:positive_min_tokens,:] + positive_unmasked[0][0][:,:positive_min_tokens,:]) / 3
            
            if   positive_A_tokens        != positive_min_tokens and positive_A_tokens != positive_max_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_A       [0][0][:,positive_min_tokens:,:]), dim=1)
                
            elif positive_B_tokens        != positive_min_tokens and positive_B_tokens != positive_max_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_B       [0][0][:,positive_min_tokens:,:]), dim=1)
                
            elif positive_unmasked_tokens != positive_min_tokens and positive_unmasked_tokens != positive_max_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_unmasked[0][0][:,positive_min_tokens:,:]), dim=1)
                
                
                
            if   positive_A_tokens        == positive_mid_tokens and positive_mid_tokens != positive_max_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_A       [0][0][:,positive_mid_tokens:,:]), dim=1)
                
            elif positive_B_tokens        == positive_mid_tokens and positive_mid_tokens != positive_max_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_B       [0][0][:,positive_mid_tokens:,:]), dim=1)
                
            elif positive_unmasked_tokens == positive_mid_tokens and positive_mid_tokens != positive_max_tokens:
                positive[0][0] = torch.cat((positive[0][0], positive_unmasked[0][0][:,positive_mid_tokens:,:]), dim=1)
            
            
            
            if 'pooled_output' in positive[0][1] and positive_A[0][1]['pooled_output'] is not None:
                positive_A_pooled_tokens = positive_A[0][1]['pooled_output'].shape[1]
                positive_B_pooled_tokens = positive_B[0][1]['pooled_output'].shape[1]
                positive_unmasked_pooled_tokens = positive_unmasked[0][1]['pooled_output'].shape[1]
                
                values = sorted([positive_A_pooled_tokens, positive_B_pooled_tokens, positive_unmasked_pooled_tokens])

                positive_min_pooled_tokens  = values[0]
                positive_mid_pooled_tokens  = values[1]
                positive_max_pooled_tokens  = values[2]
                
                positive[0][1]['pooled_output'] = (positive_A[0][1]['pooled_output'][:,:positive_min_pooled_tokens] + positive_B[0][1]['pooled_output'][:,:positive_min_pooled_tokens] + positive_unmasked[0][1]['pooled_output'][:,:positive_min_pooled_tokens]) / 3
                
                if   positive_A_pooled_tokens        != positive_min_pooled_tokens and positive_A_pooled_tokens != positive_max_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_A       [0][1]['pooled_output'][:,positive_min_pooled_tokens:,:]), dim=1)
                    
                elif positive_B_pooled_tokens        != positive_min_pooled_tokens and positive_B_pooled_tokens != positive_max_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_B       [0][1]['pooled_output'][:,positive_min_pooled_tokens:,:]), dim=1)
                    
                elif positive_unmasked_pooled_tokens != positive_min_pooled_tokens and positive_unmasked_pooled_tokens != positive_max_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_unmasked[0][1]['pooled_output'][:,positive_min_pooled_tokens:,:]), dim=1)
                    
                    
                    
                if   positive_A_pooled_tokens        == positive_mid_pooled_tokens and positive_mid_pooled_tokens != positive_max_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_A       [0][1]['pooled_output'][:,positive_mid_pooled_tokens:,:]), dim=1)
                    
                elif positive_B_pooled_tokens        == positive_mid_pooled_tokens and positive_mid_pooled_tokens != positive_max_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_B       [0][1]['pooled_output'][:,positive_mid_pooled_tokens:,:]), dim=1)
                    
                elif positive_unmasked_pooled_tokens == positive_mid_pooled_tokens and positive_mid_pooled_tokens != positive_max_pooled_tokens:
                    positive[0][1]['pooled_output'] = torch.cat((positive[0][1]['pooled_output'], positive_unmasked[0][1]['pooled_output'][:,positive_mid_pooled_tokens:,:]), dim=1)
                
        else:
            positive = positive_A
        
        if   mask_A is not None and narcissism_area == "A":
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask_A.clone()
            
        elif mask_B is not None and narcissism_area == "B":
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask_B.clone()
            
        elif mask_A is not None and mask_B is not None and narcissism_area == "AB":
            positive[0][1]['regional_conditioning_mask_orig'] = torch.clamp(1 - mask_A.clone() - mask_B.clone(), min=0.0)
            
        elif mask_A is not None and mask_B is not None and narcissism_area == "unmasked":
            positive[0][1]['regional_conditioning_mask_orig'] = 1-torch.clamp(1 - mask_A.clone() - mask_B.clone(), min=0.0)
            
        else:
            positive[0][1]['regional_conditioning_mask_orig'] = None
        
        
        positive[0][1]['narcissism_start_step'] = narcissism_start_step
        positive[0][1]['narcissism_end_step']   = narcissism_end_step
        
        return (positive,)















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





