import torch
from .helper import precision_tool


class set_precision:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent_image": ("LATENT", ),      
                    "precision":    (["16", "32", "64"], ),
                    "set_default":  ("BOOLEAN", {"default": False})
                     },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/precision"


    def main(self,
            precision    = "32",
            latent_image = None,
            set_default  = False
            ):
        
        match precision:
            case "16":
                if set_default is True:
                    torch.set_default_dtype(torch.float16)
                x = latent_image["samples"].to(torch.float16)
            case "32":
                if set_default is True:
                    torch.set_default_dtype(torch.float32)
                x = latent_image["samples"].to(torch.float32)
            case "64":
                if set_default is True:
                    torch.set_default_dtype(torch.float64)
                x = latent_image["samples"].to(torch.float64)
        return ({"samples": x}, )



class set_precision_universal:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "precision":    (["bf16", "fp16", "fp32", "fp64", "passthrough"], {"default": "fp32"}),
                    "set_default":  ("BOOLEAN",                                       {"default": False})
                    },
            "optional": {
                    "cond_pos":     ("CONDITIONING",),
                    "cond_neg":     ("CONDITIONING",),
                    "sigmas":       ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                    },
                }

    RETURN_TYPES = ("CONDITIONING", 
                    "CONDITIONING", 
                    "SIGMAS", 
                    "LATENT",)
    
    RETURN_NAMES = ("cond_pos",
                    "cond_neg",
                    "sigmas",
                    "latent_image",)

    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/precision"


    def main(self,
            precision    = "fp32",
            cond_pos     = None,
            cond_neg     = None,
            sigmas       = None,
            latent_image = None,
            set_default  = False
            ):
        
        dtype = None
        match precision:
            case "bf16":
                dtype = torch.bfloat16
            case "fp16":
                dtype = torch.float16
            case "fp32":
                dtype = torch.float32
            case "fp64":
                dtype = torch.float64
            case "passthrough":
                return (cond_pos, cond_neg, sigmas, latent_image, )
        
        if cond_pos is not None:
            cond_pos[0][0] = cond_pos[0][0].clone().to(dtype)
            cond_pos[0][1]["pooled_output"] = cond_pos[0][1]["pooled_output"].clone().to(dtype)
        
        if cond_neg is not None:
            cond_neg[0][0] = cond_neg[0][0].clone().to(dtype)
            cond_neg[0][1]["pooled_output"] = cond_neg[0][1]["pooled_output"].clone().to(dtype)
            
        if sigmas is not None:
            sigmas = sigmas.clone().to(dtype)
        
        if latent_image is not None:
            x = latent_image["samples"].clone().to(dtype)    
            latent_image = {"samples": x}

        if set_default is True:
            torch.set_default_dtype(dtype)
        
        return (cond_pos, cond_neg, sigmas, latent_image, )



class set_precision_advanced:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "latent_image":     ("LATENT", ),      
                    "global_precision": (["64", "32", "16"], ),
                    "shark_precision":  (["64", "32", "16"], ),
                     },
                }

    RETURN_TYPES = ("LATENT","LATENT","LATENT","LATENT","LATENT",)
    
    RETURN_NAMES = ("passthrough",
                    "latent_cast_to_global",
                    "latent_16",
                    "latent_32",
                    "latent_64",
                    )
    
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/precision"


    def main(self,
            global_precision = "32",
            shark_precision  = "64",
            latent_image     = None
            ):
        
        dtype_map = {
            "16": torch.float16,
            "32": torch.float32,
            "64": torch.float64
        }
        precision_map = {
            "16": 'fp16',
            "32": 'fp32',
            "64": 'fp64'
        }

        torch.set_default_dtype(dtype_map[global_precision])
        precision_tool.set_cast_type(precision_map[shark_precision])

        latent_passthrough = latent_image["samples"]

        latent_out16 = latent_image["samples"].to(torch.float16)
        latent_out32 = latent_image["samples"].to(torch.float32)
        latent_out64 = latent_image["samples"].to(torch.float64)

        target_dtype = dtype_map[global_precision]
        if latent_image["samples"].dtype != target_dtype:
            latent_image["samples"] = latent_image["samples"].to(target_dtype)

        latent_cast_to_global = latent_image["samples"]

        return ({"samples": latent_passthrough},
                {"samples": latent_cast_to_global},
                {"samples": latent_out16},
                {"samples": latent_out32},
                {"samples": latent_out64}
                )


