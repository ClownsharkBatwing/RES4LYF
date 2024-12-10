import folder_paths

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd
import comfy.supported_models

import torch


class FluxLoader:
    # adapted from https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py
    

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd
    
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                        "model_name": ([f for f in folder_paths.get_filename_list("checkpoints") + folder_paths.get_filename_list("diffusion_models") if f.endswith((".ckpt", ".safetensors", ".sft", ".pt"))], ),
                        "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                        "clip_name1": ([".use_ckpt_clip"] + folder_paths.get_filename_list("text_encoders"), ),
                        "clip_name2_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"), ),
                        "vae_name": ([".use_ckpt_vae"] + s.vae_list(), ),
                        "clip_vision_name": ([".none"] + folder_paths.get_filename_list("clip_vision"), ),
                        "style_model_name": ([".none"] + folder_paths.get_filename_list("style_models"), ),
                         }
               }
    RETURN_TYPES = ("MODEL","CLIP","VAE","CLIP_VISION","STYLE_MODEL",)
    RETURN_NAMES = ("model","clip","vae","clip_vision","style_model",)
    FUNCTION = "main"

    CATEGORY = "advanced/loaders"

    def main(self, model_name, weight_dtype, clip_name1, clip_name2_opt, vae_name, clip_vision_name, style_model_name):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        # Self-documenting code
        try:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", model_name)
        except FileNotFoundError:
            ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        if clip_name1 == ".use_ckpt_clip" and clip_name2_opt != ".none":
            raise ValueError("Cannot specify both \".use_ckpt_clip\" and another clip")

        output_vae = True if vae_name == ".use_ckpt_vae" else False
        output_clip = True if clip_name1 == ".use_ckpt_clip" else False
        ckpt_out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options)

        if clip_name1 == ".use_ckpt_clip":
            if (ckpt_out[1] is None):
                raise ValueError("Model does not have a clip")
            clip = ckpt_out[1]
        else:    
            clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", clip_name1)]
            for clip_name in [clip_name2_opt]:
                if clip_name != ".none":
                    clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name))

            clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.FLUX)

        if clip_vision_name == ".none":
            clip_vision = None
        else:
            clip_vision_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name)
            clip_vision = comfy.clip_vision.load(clip_vision_path)

        if style_model_name == ".none":
            style_model = None
        else:
            style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model_name)
            style_model = comfy.sd.load_style_model(style_model_path)

        if vae_name == ".use_ckpt_vae":
            if ckpt_out[2] is None:
                raise ValueError("Model does not have a VAE")
            vae = ckpt_out[2]
        elif vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
            vae = comfy.sd.VAE(sd=sd)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)

        model = ckpt_out[0]
        return (model, clip, vae, clip_vision, style_model,)



class SD35Loader:
    # adapted from https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {"required":{ 
            "model_name": ([f for f in folder_paths.get_filename_list("checkpoints") + folder_paths.get_filename_list("diffusion_models") if f.endswith((".ckpt", ".safetensors", ".sft", ".pt"))], ),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            "clip_name1": ([".use_ckpt_clip"] + folder_paths.get_filename_list("text_encoders"), ),
            "clip_name2_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"), ),
            "clip_name3_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"), ),
            "vae_name": ([".use_ckpt_vae"] + folder_paths.get_filename_list("vae") + ["taesd", "taesdxl", "taesd3", "taef1"], ),
            }
            }   
    RETURN_TYPES = ("MODEL","CLIP","VAE",)
    RETURN_NAMES = ("model","clip","vae",)
    FUNCTION = "main"

    CATEGORY = "advanced/loaders"

    def main(self, model_name, weight_dtype, clip_name1, clip_name2_opt, clip_name3_opt, vae_name):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        # Self-documenting code
        try:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", model_name)
        except FileNotFoundError:
            ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        if clip_name1 == ".use_ckpt_clip" and (clip_name2_opt != ".none" or clip_name3_opt != ".none"):
            raise ValueError("Cannot specify both \".use_ckpt_clip\" and another clip")

        output_vae = True if vae_name == ".use_ckpt_vae" else False
        output_clip = True if clip_name1 == ".use_ckpt_clip" else False
        ckpt_out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options=model_options)
        
        if clip_name1 == ".use_ckpt_clip":
            if (ckpt_out[1] is None):
                raise ValueError("Model does not have a clip")
            clip = ckpt_out[1]
        else:
            clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", clip_name1)]
            for clip_name in [clip_name2_opt, clip_name3_opt]:
                if clip_name != ".none":
                    clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name))
        
            clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.SD3)

        if vae_name == ".use_ckpt_vae":
            if ckpt_out[2] is None:
                raise ValueError("Model does not have a VAE")
            vae = ckpt_out[2]
        elif vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
            vae = comfy.sd.VAE(sd=sd)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
        
        model = ckpt_out[0]
        return (model, clip, vae,)