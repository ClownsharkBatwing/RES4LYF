import torch
import torch.nn.functional as F

from torch  import Tensor
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar, List
from dataclasses import dataclass, field

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
            "t5_strength"   : ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
            "clip_strength" : ("FLOAT", {"default": 1.0, "min": -10000, "max": 10000, "step":0.01}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION     = "combine"
    CATEGORY     = "RES4LYF/conditioning"
    EXPERIMENTAL = True

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
            "clip"  : ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl" : ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING","INT","INT",)
    RETURN_NAMES = ("conditioning", "clip_l_end", "t5xxl_end",)
    FUNCTION     = "encode"
    CATEGORY     = "RES4LYF/conditioning"
    EXPERIMENTAL = True


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
    EXPERIMENTAL = True

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
        c.append(conditioning_0)
        
        if conditioning_1 is not None:
            c.append(conditioning_1)
            
        if conditioning_2 is not None:
            c.append(conditioning_2)
            
        if conditioning_3 is not None:
            c.append(conditioning_3)

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
        c.append(conditioning_0)
        
        if conditioning_1 is not None:
            c.append(conditioning_1)
            
        if conditioning_2 is not None:
            c.append(conditioning_2)
            
        if conditioning_3 is not None:
            c.append(conditioning_3)
            
        if conditioning_4 is not None:
            c.append(conditioning_4)
            
        if conditioning_5 is not None:
            c.append(conditioning_5)
            
        if conditioning_6 is not None:
            c.append(conditioning_6)
            
        if conditioning_7 is not None:
            c.append(conditioning_7)
            
        return (c, )



class EmptyConditioningGenerator:
    def __init__(self, model=None, conditioning=None, device=None, dtype=None):
        """ device, dtype currently unused """
        if model is not None:
                    
            self.device = device
            self.dtype  = dtype
        
            import comfy.supported_models
            self.model_config = model.model.model_config

            if isinstance(self.model_config, comfy.supported_models.SD3):
                self.text_len_base = 154
                self.text_channels = 4096
                self.pooled_len    = 2048
            elif isinstance(self.model_config, (comfy.supported_models.Flux, comfy.supported_models.FluxSchnell)):
                self.text_len_base = 256
                self.text_channels = 4096
                self.pooled_len    = 768
            elif isinstance(self.model_config, comfy.supported_models.AuraFlow):
                self.text_len_base = 256
                self.text_channels = 2048
                self.pooled_len    = 1
            elif isinstance(self.model_config, comfy.supported_models.Stable_Cascade_C):
                self.text_len_base = 77
                self.text_channels = 1280
                self.pooled_len    = 1280
            elif isinstance(self.model_config, comfy.supported_models.WAN21_T2V) or isinstance(self.model_config, comfy.supported_models.WAN21_I2V):
                self.text_len_base = 512
                self.text_channels = 5120 # sometimes needs to be 4096, like when initializing in samplers_py in shark?
                self.pooled_len    = 1
            else:
                raise ValueError(f"Unknown model config: {type(config)}")
        elif conditioning is not None:
            self.device        = conditioning[0][0].device
            self.dtype         = conditioning[0][0].dtype
            self.text_len_base = conditioning[0][0].shape[-2]
            self.pooled_len    = conditioning[0][1]['pooled_output'].shape[-1]
            self.text_channels = conditioning[0][0].shape[-1]
            
    def get_empty_conditioning(self):
        return [[
            torch.zeros((1, self.text_len_base, self.text_channels)),
            {'pooled_output': torch.zeros((1, self.pooled_len))}
        ]]

    def get_empty_conditionings(self, count):
        return [self.get_empty_conditioning() for _ in range(count)]
    
    def zero_none_conditionings_(self, *conds):
        if len(conds) == 1 and isinstance(conds[0], (list, tuple)):
            conds = conds[0]
        for i, cond in enumerate(conds):
            conds[i] = self.get_empty_conditioning() if cond is None else cond
        return conds

def zero_conditioning_from_list(conds):
    for cond in conds:
        if cond is not None:
            for i in range(len(cond)):
                pooled     = cond[i][1].get('pooled_output')
                pooled_len = pooled.shape[-1] if pooled is not None else 1    # 1 default pooled_output len for those without it
                
                cond_zero  = [[
                    torch.zeros_like(cond[i][0]),
                    {"pooled_output": torch.zeros((1,pooled_len), dtype=cond[i][0].dtype, device=cond[i][0].device)},
                ]]
                
            return cond_zero


class TemporalMaskGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "switch_frame": ("INT", {"default": 33, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "frames":       ("INT", {"default": 65, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "invert_mask":             ("BOOLEAN",                                  {"default": False}),

                    },
                "optional": 
                    {
                    }
                }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",) 
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    EXPERIMENTAL = True
    
    def main(self,
            switch_frame = 33,
            frames = 65,
            invert_mask = False,
            ):
        
        switch_frame = switch_frame // 4
        frames = frames // 4 + 1
        
        temporal_mask = torch.ones((frames, 2, 2))
        
        temporal_mask[switch_frame:,...] = 0.0
        
        if invert_mask:
            temporal_mask = 1 - temporal_mask
        
        return (temporal_mask,)




class TemporalSplitAttnMask_Midframe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "self_attn_midframe":  ("INT", {"default": 33, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "cross_attn_midframe": ("INT", {"default": 33, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "self_attn_invert":    ("BOOLEAN",                                  {"default": False}),
                    "cross_attn_invert":   ("BOOLEAN",                                  {"default": False}),
                    "frames":             ("INT", {"default": 65, "min": 1, "step": 4, "max": 0xffffffffffffffff}),

                    },
                "optional": 
                    {
                    }
                }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",) 
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    EXPERIMENTAL = True
    
    def main(self,
            self_attn_midframe = 33,
            cross_attn_midframe = 33,
            self_attn_invert = False,
            cross_attn_invert = False,
            frames = 65,
            ):

        frames = frames // 4 + 1
        
        temporal_self_mask  = torch.ones((frames, 2, 2))
        temporal_cross_mask = torch.ones((frames, 2, 2))

        
        self_attn_midframe  = self_attn_midframe  // 4
        cross_attn_midframe = cross_attn_midframe // 4
        
        temporal_self_mask[self_attn_midframe  :,...] = 0.0
        temporal_cross_mask[cross_attn_midframe:,...] = 0.0
        
        if self_attn_invert:
            temporal_self_mask  = 1 - temporal_self_mask
            
        if cross_attn_invert:
            temporal_cross_mask = 1 - temporal_cross_mask
        
        temporal_attn_masks = torch.stack([temporal_cross_mask, temporal_self_mask])
        
        return (temporal_attn_masks,)




class TemporalSplitAttnMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "self_attn_start":  ("INT", {"default": 1,  "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "self_attn_stop":   ("INT", {"default": 33, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "cross_attn_start": ("INT", {"default": 1,  "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "cross_attn_stop":  ("INT", {"default": 33, "min": 1, "step": 4, "max": 0xffffffffffffffff}),

                    #"frames":           ("INT", {"default": 65, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    },
                "optional": 
                    {
                    }
                }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",) 
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    
    def main(self,
            self_attn_start  = 0,
            self_attn_stop   = 33,
            cross_attn_start = 0,
            cross_attn_stop  = 33,
            #frames           = 65,
            ):

        #frames = frames // 4 + 1
        
        self_attn_start  = self_attn_start  // 4 #+ 1
        self_attn_stop   = self_attn_stop   // 4 + 1
        cross_attn_start = cross_attn_start // 4 #+ 1
        cross_attn_stop  = cross_attn_stop  // 4 + 1
        
        max_stop = max(self_attn_stop, cross_attn_stop)
        
        temporal_self_mask  = torch.zeros((max_stop, 1, 1))
        temporal_cross_mask = torch.zeros((max_stop, 1, 1))

        temporal_self_mask [ self_attn_start: self_attn_stop,...] = 1.0
        temporal_cross_mask[cross_attn_start:cross_attn_stop,...] = 1.0
        
        temporal_attn_masks = torch.stack([temporal_cross_mask, temporal_self_mask])
        
        return (temporal_attn_masks,)




class TemporalCrossAttnMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                    "cross_attn_start": ("INT", {"default": 1,  "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    "cross_attn_stop":  ("INT", {"default": 33, "min": 1, "step": 4, "max": 0xffffffffffffffff}),
                    },
                "optional": 
                    {
                    }
                }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("temporal_mask",) 
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/masks"
    
    def main(self,
            cross_attn_start = 0,
            cross_attn_stop  = 33,
            ):
        
        cross_attn_start = cross_attn_start // 4 #+ 1
        cross_attn_stop  = cross_attn_stop  // 4 + 1
        
        temporal_self_mask  = torch.zeros((cross_attn_stop, 1, 1))  # dummy to satisfy stack
        temporal_cross_mask = torch.zeros((cross_attn_stop, 1, 1))

        temporal_cross_mask[cross_attn_start:cross_attn_stop,...] = 1.0
        
        temporal_attn_masks = torch.stack([temporal_cross_mask, temporal_self_mask])
        
        return (temporal_attn_masks,)




@dataclass
class RegionalParameters:
    weights    : List[float] = field(default_factory=list)
    floors     : List[float] = field(default_factory=list)
    narc_idx   :      int    = -1
    narc_start :      int    =  0
    narc_stop  :      int    = -1



REG_MASK_TYPE_2 = [
    "gradient",
    "gradient_masked",
    "gradient_unmasked",
    "boolean",
    "boolean_masked",
    "boolean_unmasked",
]

REG_MASK_TYPE_3 = [
    "gradient",
    "gradient_A",
    "gradient_B",
    "gradient_unmasked",
    "gradient_AB",
    "gradient_A,unmasked",
    "gradient_B,unmasked",

    "boolean",
    "boolean_A",
    "boolean_B",
    "boolean_unmasked",
    "boolean_AB",
    "boolean_A,unmasked",
    "boolean_B,unmasked",
]

REG_MASK_TYPE_AB = [
    "gradient",
    "gradient_A",
    "gradient_B",
    "boolean",
    "boolean_A",
    "boolean_B",
]

REG_MASK_TYPE_ABC = [
    "gradient",
    "gradient_A",
    "gradient_B",
    "gradient_C",
    "gradient_AB",
    "gradient_AC",
    "gradient_BC",

    "boolean",
    "boolean_A",
    "boolean_B",
    "boolean_C",
    "boolean_AB",
    "boolean_AC",
    "boolean_BC",

]



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
                "mask_type":               (REG_MASK_TYPE_2,                             {"default": "boolean"}),
                "edge_width":              ("INT",                                       {"default": 0,  "min": 0,          "max": 10000}),
                #"narcissism_area":         (["masked", "unmasked", "off"],               {"default": "off"}),
                #"narcissism_start_step":   ("INT",                                       {"default": 0,   "min": -1,        "max": 10000}),
                #"narcissism_end_step":     ("INT",                                       {"default": 5,   "min": -1,        "max": 10000}),
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
            start_sigma              : float  = 0.0,
            end_sigma                : float  = 1.0,
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
            edge_width               : int    = 0,
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
                                        start_sigma              = start_sigma,
                                        end_sigma                = end_sigma,
                                        weight_scheduler         = weight_scheduler,
                                        start_step               = start_step,
                                        end_step                 = end_step,
                                        weights                  = weights,
                                        region_bleeds            = region_bleeds,
                                        region_bleed             = region_bleed,
                                        region_bleed_start_step  = region_bleed_start_step,
                                        mask_type                = mask_type,
                                        edge_width               = edge_width,
                                        mask                     = mask,
                                        invert_mask              = invert_mask,
                                        positive_masked          = positive_masked,
                                        positive_unmasked        = positive_unmasked,
                                        narcissism_area          = narcissism_area,
                                        narcissism_start_step    = narcissism_start_step,
                                        narcissism_end_step      = narcissism_end_step,
                                        )

        positive = zero_conditioning_from_list([positive_masked, positive_unmasked])
        
        positive[0][1]['callback_regional'] = callback
        
        return (positive,)



    def prepare_regional_cond(self,
                                model,
                                weight                   : float  = 1.0,
                                start_sigma              : float  = 0.0,
                                end_sigma                : float  = 1.0,
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
                                edge_width               : int    = 0,
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
        
        prepend = torch.full((region_bleed_start_step,),  0.0, dtype=default_dtype, device=default_device)
        floors  = initialize_or_scale(floors,  floor,  end_step).to(default_dtype).to(default_device)
        floors  = F.pad(floors,  (0, MAX_STEPS), value=0.0)
        floors  = torch.cat((prepend, floors), dim=0)

        if (positive_masked is None) and (positive_unmasked is None):
            positive = None

        elif mask is not None:
            EmptyCondGen = EmptyConditioningGenerator(model)
            positive_masked, positive_unmasked = EmptyCondGen.zero_none_conditionings_([positive_masked, positive_unmasked])
            
            positive_masked_tokens   = positive_masked[0][0]  .shape[1]
            positive_unmasked_tokens = positive_unmasked[0][0].shape[1]
            
            positive_min_tokens = min(positive_masked_tokens, positive_unmasked_tokens)
            
            positive = copy.deepcopy(positive_masked)
            positive[0][0] = (positive_masked[0][0][:,:positive_min_tokens,:] + positive_unmasked[0][0][:,:positive_min_tokens,:]) / 2
            
            from .attention_masks import FullAttentionMask, CrossAttentionMask, SplitAttentionMask, RegionalContext
            if isinstance(model.model.model_config, comfy.supported_models.WAN21_T2V) or isinstance(model.model.model_config, comfy.supported_models.WAN21_I2V):
                if model.model.diffusion_model.blocks[0].self_attn.winderz_type != "false":
                    AttnMask = CrossAttentionMask(mask_type, edge_width)
                else:
                    AttnMask = SplitAttentionMask(mask_type, edge_width)
            else:
                AttnMask = FullAttentionMask(mask_type, edge_width)
            AttnMask.add_region(positive_masked  [0][0],   mask)
            AttnMask.add_region(positive_unmasked[0][0], 1-mask)
            
            RegContext = RegionalContext()
            RegContext.add_region(positive_masked  [0][0])
            RegContext.add_region(positive_unmasked[0][0])
            
            positive[0][1]['AttnMask'] = AttnMask
            positive[0][1]['RegContext'] = RegContext
            
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
            narc_idx = 0
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask.clone()
            if 'pooled_output' in positive[0][1]:
                positive[0][1]['pooled_output'] = positive_unmasked[0][1]['pooled_output']
        
        elif mask is not None and narcissism_area == "unmasked":
            narc_idx = 1
            positive[0][1]['regional_conditioning_mask_orig'] = mask.clone()
            if 'pooled_output' in positive[0][1]:
                positive[0][1]['pooled_output'] = positive_masked[0][1]['pooled_output']
            
        else:
            narc_idx = -1
            positive[0][1]['regional_conditioning_mask_orig'] = None
        
        positive[0][1]['RegParam'] = RegionalParameters(weights, floors, narc_idx, narcissism_start_step, narcissism_end_step)
        
        return (positive,)








class ClownRegionalConditioning_AB:
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
                "mask_type":               (REG_MASK_TYPE_AB,                            {"default": "boolean"}),
                "edge_width":              ("INT",                                       {"default": 0,  "min": 0,          "max": 10000}),
                #"narcissism_area":         (["masked", "unmasked", "off"],               {"default": "off"}),
                #"narcissism_start_step":   ("INT",                                       {"default": 0,   "min": -1,        "max": 10000}),
                #"narcissism_end_step":     ("INT",                                       {"default": 5,   "min": -1,        "max": 10000}),
                "invert_mask":             ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "positive_A":              ("CONDITIONING", ),
                "positive_B":              ("CONDITIONING", ),
                "mask_A":                  ("MASK", ),
                "mask_B":                  ("MASK", ),
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
            start_sigma              : float  = 0.0,
            end_sigma                : float  = 1.0,
            weight_scheduler                  = None,
            start_step               : int    = 0,
            end_step                 : int    = -1,
            positive_A                        = None,
            positive_B                        = None,
            weights                  : Tensor = None,
            region_bleeds            : Tensor = None,
            region_bleed             : float  = 0.0,
            region_bleed_start_step  : int    = 0,
            mask_type                : str    = "boolean",
            edge_width               : int    = 0,
            mask_A                            = None,
            mask_B                            = None,
            narcissism_area          : str    = "masked",
            narcissism_start_step    : int    = 0,
            narcissism_end_step      : int    = 5,
            invert_mask              : bool   = False
            ) -> Tuple[Tensor]:
        
        positive_masked   = positive_A 
        positive_unmasked = positive_B
        
        mask   = mask_A
        unmask = mask_B
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        if narcissism_end_step == -1:
            narcissism_end_step = MAX_STEPS
        
        callback = self.create_callback(weight                   = weight,
                                        start_sigma              = start_sigma,
                                        end_sigma                = end_sigma,
                                        weight_scheduler         = weight_scheduler,
                                        start_step               = start_step,
                                        end_step                 = end_step,
                                        weights                  = weights,
                                        region_bleeds            = region_bleeds,
                                        region_bleed             = region_bleed,
                                        region_bleed_start_step  = region_bleed_start_step,
                                        mask_type                = mask_type,
                                        edge_width               = edge_width,
                                        mask                     = mask,
                                        unmask                   = unmask,
                                        invert_mask              = invert_mask,
                                        positive_masked          = positive_masked,
                                        positive_unmasked        = positive_unmasked,
                                        narcissism_area          = narcissism_area,
                                        narcissism_start_step    = narcissism_start_step,
                                        narcissism_end_step      = narcissism_end_step,
                                        )

        positive = zero_conditioning_from_list([positive_masked, positive_unmasked])
        
        positive[0][1]['callback_regional'] = callback
        
        return (positive,)



    def prepare_regional_cond(self,
                                model,
                                weight                   : float  = 1.0,
                                start_sigma              : float  = 0.0,
                                end_sigma                : float  = 1.0,
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
                                edge_width               : int    = 0,
                                mask                              = None,
                                unmask                            = None,
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
            mask   = 1-mask
            unmask = 1-unmask

        #weight, weights = mask_weight, mask_weights
        floor, floors = region_bleed, region_bleeds
        
        weights = initialize_or_scale(weights, weight, end_step).to(default_dtype).to(default_device)
        weights = F.pad(weights, (0, MAX_STEPS), value=0.0)
        
        prepend = torch.full((region_bleed_start_step,),  0.0, dtype=default_dtype, device=default_device)
        floors  = initialize_or_scale(floors,  floor,  end_step).to(default_dtype).to(default_device)
        floors  = F.pad(floors,  (0, MAX_STEPS), value=0.0)
        floors  = torch.cat((prepend, floors), dim=0)

        if (positive_masked is None) and (positive_unmasked is None):
            positive = None

        elif mask is not None:
            EmptyCondGen = EmptyConditioningGenerator(model)
            positive_masked, positive_unmasked = EmptyCondGen.zero_none_conditionings_([positive_masked, positive_unmasked])
            
            positive_masked_tokens   = positive_masked[0][0]  .shape[1]
            positive_unmasked_tokens = positive_unmasked[0][0].shape[1]
            
            positive_min_tokens = min(positive_masked_tokens, positive_unmasked_tokens)
            
            positive = copy.deepcopy(positive_masked)
            positive[0][0] = (positive_masked[0][0][:,:positive_min_tokens,:] + positive_unmasked[0][0][:,:positive_min_tokens,:]) / 2
            
            from .attention_masks import FullAttentionMask, CrossAttentionMask, SplitAttentionMask, RegionalContext
            if isinstance(model.model.model_config, comfy.supported_models.WAN21_T2V) or isinstance(model.model.model_config, comfy.supported_models.WAN21_I2V):
                if model.model.diffusion_model.blocks[0].self_attn.winderz_type != "false":
                    AttnMask = CrossAttentionMask(mask_type, edge_width)
                else:
                    AttnMask = SplitAttentionMask(mask_type, edge_width)
            else:
                AttnMask = FullAttentionMask(mask_type, edge_width)
            AttnMask.add_region(positive_masked  [0][0],   mask)
            AttnMask.add_region(positive_unmasked[0][0], unmask)
            
            RegContext = RegionalContext()
            RegContext.add_region(positive_masked  [0][0])
            RegContext.add_region(positive_unmasked[0][0])
            
            positive[0][1]['AttnMask'] = AttnMask
            positive[0][1]['RegContext'] = RegContext
            
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
            narc_idx = 0
            positive[0][1]['regional_conditioning_mask_orig'] = unmask.clone()
            if 'pooled_output' in positive[0][1]:
                positive[0][1]['pooled_output'] = positive_unmasked[0][1]['pooled_output']
        
        elif mask is not None and narcissism_area == "unmasked":
            narc_idx = 1
            positive[0][1]['regional_conditioning_mask_orig'] = mask.clone()
            if 'pooled_output' in positive[0][1]:
                positive[0][1]['pooled_output'] = positive_masked[0][1]['pooled_output']
            
        else:
            narc_idx = -1
            positive[0][1]['regional_conditioning_mask_orig'] = None
        
        positive[0][1]['RegParam'] = RegionalParameters(weights, floors, narc_idx, narcissism_start_step, narcissism_end_step)
        
        return (positive,)



class ClownRegionalConditioning3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "weight":                  ("FLOAT",                                     {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed":            ("FLOAT",                                     {"default": 0.0,  "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed_start_step": ("INT",                                       {"default": 0,   "min":  0,        "max": 10000}),
                "weight_scheduler":        (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                "start_step":              ("INT",                                       {"default": 0,    "min":  0,        "max": 10000}),
                "end_step":                ("INT",                                       {"default": 100,  "min": -1,        "max": 10000}),
                "mask_type":               (REG_MASK_TYPE_3,                             {"default": "boolean"}),
                "edge_width":              ("INT",                                       {"default": 0,  "min": 0,          "max": 10000}),
                #"narcissism_area":         (["A", "B", "AB", "unmasked", "off"],         {"default": "off"}),
                #"narcissism_start_step":   ("INT",                                       {"default": 0,   "min": -1,        "max": 10000}),
                #"narcissism_end_step":     ("INT",                                       {"default": 5,   "min": -1,        "max": 10000}),
                "invert_mask":             ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "positive_A":              ("CONDITIONING", ),
                "positive_B":              ("CONDITIONING", ),
                "positive_unmasked":       ("CONDITIONING", ),
                "mask_A":                  ("MASK", ),
                "mask_B":                  ("MASK", ),
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
            start_sigma              : float  = 0.0,
            end_sigma                : float  = 1.0,
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
            edge_width               : int    = 0,
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
                                        start_sigma              = start_sigma,
                                        end_sigma                = end_sigma,
                                        weight_scheduler         = weight_scheduler,
                                        start_step               = start_step,
                                        end_step                 = end_step,
                                        weights                  = weights,
                                        region_bleeds            = region_bleeds,
                                        region_bleed             = region_bleed,
                                        region_bleed_start_step  = region_bleed_start_step,
                                        mask_type                = mask_type,
                                        edge_width               = edge_width,
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

        positive = zero_conditioning_from_list([positive_A, positive_B, positive_unmasked])
        
        positive[0][1]['callback_regional'] = callback
        
        return (positive,)



    def prepare_regional_cond(self,
                                model,
                                weight                   : float  = 1.0,
                                start_sigma              : float  = 0.0,
                                end_sigma                : float  = 1.0,
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
                                edge_width               : int    = 0,
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
            
            EmptyCondGen = EmptyConditioningGenerator(model)
            positive_A, positive_B, positive_unmasked = EmptyCondGen.zero_none_conditionings_([positive_A, positive_B, positive_unmasked])

            positive_A_tokens        = positive_A[0][0].shape[1]
            positive_B_tokens        = positive_B[0][0].shape[1]
            positive_unmasked_tokens = positive_unmasked[0][0].shape[1]
            
            values = sorted([positive_A_tokens, positive_B_tokens, positive_unmasked_tokens])

            positive_min_tokens  = values[0]
            positive_mid_tokens  = values[1]
            positive_max_tokens  = values[2]
            
            positive = copy.deepcopy(positive_A)
            positive[0][0] = (positive_A[0][0][:,:positive_min_tokens,:] + positive_B[0][0][:,:positive_min_tokens,:] + positive_unmasked[0][0][:,:positive_min_tokens,:]) / 3
            
            from .attention_masks import FullAttentionMask, CrossAttentionMask, SplitAttentionMask, RegionalContext
            if isinstance(model.model.model_config, comfy.supported_models.WAN21_T2V) or isinstance(model.model.model_config, comfy.supported_models.WAN21_I2V):
                if model.model.diffusion_model.blocks[0].self_attn.winderz_type != "false":
                    AttnMask = CrossAttentionMask(mask_type, edge_width)
                else:
                    AttnMask = SplitAttentionMask(mask_type, edge_width)
            else:
                AttnMask = FullAttentionMask(mask_type, edge_width)
            AttnMask.add_region(positive_A       [0][0], mask_A)
            AttnMask.add_region(positive_B       [0][0], mask_B)
            AttnMask.add_region(positive_unmasked[0][0], mask_AB_inv)
            
            RegContext = RegionalContext()
            RegContext.add_region(positive_A  [0][0])
            RegContext.add_region(positive_B[0][0])
            RegContext.add_region(positive_unmasked[0][0])

            
            positive[0][1]['AttnMask']   = AttnMask
            positive[0][1]['RegContext'] = RegContext
            
            
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
            narc_idx = 0
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask_A.clone()
            
        elif mask_B is not None and narcissism_area == "B":
            narc_idx = 1
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask_B.clone()
            
        elif mask_A is not None and mask_B is not None and narcissism_area == "AB":
            narc_idx = 2
            positive[0][1]['regional_conditioning_mask_orig'] = torch.clamp(1 - mask_A.clone() - mask_B.clone(), min=0.0)
            
        elif mask_A is not None and mask_B is not None and narcissism_area == "unmasked":
            narc_idx = 25436747  # need to deal with this properly!
            positive[0][1]['regional_conditioning_mask_orig'] = 1-torch.clamp(1 - mask_A.clone() - mask_B.clone(), min=0.0)
            
        else:
            narc_idx = -1
            positive[0][1]['regional_conditioning_mask_orig'] = None
        

        positive[0][1]['RegParam'] = RegionalParameters(weights, floors, narc_idx, narcissism_start_step, narcissism_end_step)
        
        return (positive,)























class ClownRegionalConditioning_ABC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "weight":                  ("FLOAT",                                     {"default": 1.0,  "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed":            ("FLOAT",                                     {"default": 0.0,  "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "region_bleed_start_step": ("INT",                                       {"default": 0,    "min":  0,       "max": 10000}),
                "weight_scheduler":        (["constant"] + get_res4lyf_scheduler_list(), {"default": "constant"},),
                "start_step":              ("INT",                                       {"default": 0,    "min":  0,       "max": 10000}),
                "end_step":                ("INT",                                       {"default": 100,  "min": -1,       "max": 10000}),
                "mask_type":               (REG_MASK_TYPE_ABC,                           {"default": "boolean"}),
                "edge_width":              ("INT",                                       {"default": 0,    "min": 0,        "max": 10000}),
                #"narcissism_area":         (["A", "B", "AB", "unmasked", "off"],         {"default": "off"}),
                #"narcissism_start_step":   ("INT",                                       {"default": 0,   "min": -1,        "max": 10000}),
                #"narcissism_end_step":     ("INT",                                       {"default": 5,   "min": -1,        "max": 10000}),
                "invert_mask":             ("BOOLEAN",                                   {"default": False}),
            }, 
            "optional": {
                "positive_A":              ("CONDITIONING", ),
                "positive_B":              ("CONDITIONING", ),
                "positive_C":              ("CONDITIONING", ),
                "mask_A":                  ("MASK", ),
                "mask_B":                  ("MASK", ),
                "mask_C":                  ("MASK", ),
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
            start_sigma              : float  = 0.0,
            end_sigma                : float  = 1.0,
            weight_scheduler                  = None,
            start_step               : int    = 0,
            end_step                 : int    = -1,
            positive_A                        = None,
            positive_B                        = None,
            positive_C                        = None,
            weights                  : Tensor = None,
            region_bleeds            : Tensor = None,
            region_bleed             : float  = 0.0,
            region_bleed_start_step  : int    = 0,

            mask_type                : str    = "boolean",
            edge_width               : int    = 0,
            mask_A                              = None,
            mask_B                              = None,
            mask_C                              = None,
            narcissism_area       : str    = "AB",
            narcissism_start_step : int    = 0,
            narcissism_end_step   : int    = 5,
            invert_mask              : bool   = False
            ) -> Tuple[Tensor]:
        
        if end_step == -1:
            end_step = MAX_STEPS
        
        callback = self.create_callback(weight                   = weight,
                                        start_sigma              = start_sigma,
                                        end_sigma                = end_sigma,
                                        weight_scheduler         = weight_scheduler,
                                        start_step               = start_step,
                                        end_step                 = end_step,
                                        weights                  = weights,
                                        region_bleeds            = region_bleeds,
                                        region_bleed             = region_bleed,
                                        region_bleed_start_step  = region_bleed_start_step,
                                        mask_type                = mask_type,
                                        edge_width               = edge_width,
                                        mask_A                   = mask_A,
                                        mask_B                   = mask_B,
                                        mask_C                   = mask_C,
                                        invert_mask              = invert_mask,
                                        positive_A               = positive_A,
                                        positive_B               = positive_B,
                                        positive_C               = positive_C,
                                        narcissism_area       = narcissism_area,
                                        narcissism_start_step = narcissism_start_step,
                                        narcissism_end_step   = narcissism_end_step,
                                        )

        positive = zero_conditioning_from_list([positive_A, positive_B, positive_C])
        
        positive[0][1]['callback_regional'] = callback
        
        return (positive,)



    def prepare_regional_cond(self,
                                model,
                                weight                   : float  = 1.0,
                                start_sigma              : float  = 0.0,
                                end_sigma                : float  = 1.0,
                                weight_scheduler                  = None,
                                start_step               : int    =  0,
                                end_step                 : int    = -1,
                                positive_A                        = None,
                                positive_B                        = None,

                                positive_C                        = None,
                                weights                  : Tensor = None,
                                region_bleeds            : Tensor = None,
                                region_bleed             : float  = 0.0,
                                region_bleed_start_step  : int    = 0,

                                mask_type                : str    = "boolean",
                                edge_width               : int    = 0,
                                mask_A                            = None,
                                mask_B                            = None,
                                mask_C                            = None,
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
        
        mask_AB_inv = mask_C
        if invert_mask and mask_AB_inv is not None:
            mask_AB_inv = 1-mask_AB_inv
            
        positive_unmasked = positive_C
        
        #mask_AB_inv = torch.ones_like(mask_A) - mask_A - mask_B

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
            
            EmptyCondGen = EmptyConditioningGenerator(model)
            positive_A, positive_B, positive_unmasked = EmptyCondGen.zero_none_conditionings_([positive_A, positive_B, positive_unmasked])

            positive_A_tokens        = positive_A[0][0].shape[1]
            positive_B_tokens        = positive_B[0][0].shape[1]
            positive_unmasked_tokens = positive_unmasked[0][0].shape[1]
            
            values = sorted([positive_A_tokens, positive_B_tokens, positive_unmasked_tokens])

            positive_min_tokens  = values[0]
            positive_mid_tokens  = values[1]
            positive_max_tokens  = values[2]
            
            positive = copy.deepcopy(positive_A)
            positive[0][0] = (positive_A[0][0][:,:positive_min_tokens,:] + positive_B[0][0][:,:positive_min_tokens,:] + positive_unmasked[0][0][:,:positive_min_tokens,:]) / 3
            
            from .attention_masks import FullAttentionMask, CrossAttentionMask, SplitAttentionMask, RegionalContext
            if isinstance(model.model.model_config, comfy.supported_models.WAN21_T2V) or isinstance(model.model.model_config, comfy.supported_models.WAN21_I2V):
                if model.model.diffusion_model.blocks[0].self_attn.winderz_type != "false":
                    AttnMask = CrossAttentionMask(mask_type, edge_width)
                else:
                    AttnMask = SplitAttentionMask(mask_type, edge_width)
            else:
                AttnMask = FullAttentionMask(mask_type, edge_width)
            AttnMask.add_region(positive_A       [0][0], mask_A)
            AttnMask.add_region(positive_B       [0][0], mask_B)
            AttnMask.add_region(positive_unmasked[0][0], mask_AB_inv)
            
            RegContext = RegionalContext()
            RegContext.add_region(positive_A  [0][0])
            RegContext.add_region(positive_B[0][0])
            RegContext.add_region(positive_unmasked[0][0])

            
            positive[0][1]['AttnMask']   = AttnMask
            positive[0][1]['RegContext'] = RegContext
            
            
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
            narc_idx = 0
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask_A.clone()
            
        elif mask_B is not None and narcissism_area == "B":
            narc_idx = 1
            positive[0][1]['regional_conditioning_mask_orig'] = 1-mask_B.clone()
            
        elif mask_A is not None and mask_B is not None and narcissism_area == "AB":
            narc_idx = 2
            positive[0][1]['regional_conditioning_mask_orig'] = torch.clamp(1 - mask_A.clone() - mask_B.clone(), min=0.0)
            
        elif mask_A is not None and mask_B is not None and narcissism_area == "unmasked":
            narc_idx = 25436747  # need to deal with this properly!
            positive[0][1]['regional_conditioning_mask_orig'] = 1-torch.clamp(1 - mask_A.clone() - mask_B.clone(), min=0.0)
            
        else:
            narc_idx = -1
            positive[0][1]['regional_conditioning_mask_orig'] = None
        

        positive[0][1]['RegParam'] = RegionalParameters(weights, floors, narc_idx, narcissism_start_step, narcissism_end_step)
        
        return (positive,)





