import folder_paths
import torch
import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.model_sampling
import comfy.latent_formats
import comfy.sd
import comfy.clip_vision
import comfy.supported_models
from comfy.utils import load_torch_file

# Documentation: Self-documenting code
# Instructions for use: Obvious
# Expected results: Fork desync
# adapted from https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py

clip_types = ["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "hunyuan_dit", "flux", "mochi", "ltxv", "hunyuan_video", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"]

class BaseModelLoader:
    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]

        # VAE scale and shift mapping
        vae_params = {
            "taesd": (0.18215, 0.0),
            "taesdxl": (0.13025, 0.0),
            "taesd3": (1.5305, 0.0609),
            "taef1": (0.3611, 0.1159)
        }
        
        if name in vae_params:
            scale, shift = vae_params[name]
            sd["vae_scale"] = torch.tensor(scale)
            sd["vae_shift"] = torch.tensor(shift)
            
        return sd
    
    @staticmethod
    def guess_clip_type(model):
        import comfy.model_base as mb
        
        type_map = [
            (mb.SDXLRefiner, "sdxl"),
            (mb.SDXL, "sdxl"),
            (mb.SD15_instructpix2pix, "stable_diffusion"),
            (mb.SDXL_instructpix2pix, "sdxl"),
            (mb.StableCascade_C, "stable_cascade"),
            (mb.StableCascade_B, "stable_cascade"),
            (mb.Flux, "flux"),
            (mb.LTXV, "ltxv"),
            (mb.HunyuanDiT, "hunyuan_dit"),
            (mb.HunyuanVideo, "hunyuan_video"),
            (mb.HunyuanVideoI2V, "hunyuan_video"),
            (mb.HunyuanVideoSkyreelsI2V, "hunyuan_video"),
            (mb.PixArt, "pixart"),
            (mb.CosmosVideo, "cosmos"),
            (mb.Lumina2, "lumina2"),
            (mb.WAN21, "wan"),
            (mb.WAN21_Vace, "wan"),
            (mb.WAN21_Camera, "wan"),
            (mb.HiDream, "hidream"),
            (mb.Chroma, "chroma"),
            (mb.ACEStep, "ace"),
            (mb.SD3, "sd3"),
            (mb.GenmoMochi, "mochi"),
        ]

        for cls, clip_type in type_map:
            if isinstance(model, cls):
                return clip_type.upper()

        # fallback
        known_types = {
            "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "hunyuan_dit", "flux", "mochi", "ltxv",
            "hunyuan_video", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"
        }

        class_name = model.__class__.__name__.lower()
        for t in known_types:
            if t in class_name:
                return t.upper()

        default_clip_type = "stable_diffusion"
        return default_clip_type.upper()

    @staticmethod
    def get_model_files():
        return [f for f in folder_paths.get_filename_list("checkpoints") + 
                folder_paths.get_filename_list("diffusion_models") 
                if f.endswith((".ckpt", ".safetensors", ".sft", ".pt"))]

    @staticmethod
    def get_weight_options():
        return ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]

    @staticmethod
    def get_clip_options():
        return [".use_ckpt_clip"] + folder_paths.get_filename_list("text_encoders")

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
                f1_taesd_enc = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_dec = True

        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    def process_weight_dtype(self, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        return model_options

    def load_checkpoint(self, model_name, output_vae, output_clip, model_options):
        try:
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", model_name)
            
            out = None
            try:
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path, 
                    output_vae=output_vae,
                    output_clip=output_clip,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options
                )
            except RuntimeError as e:
                if "ERROR: Could not detect model type of:" in str(e):
                    error_msg = ""
                    if output_vae is True:
                        error_msg += "Model/Checkpoint file does not contain a VAE\n"
                    if output_clip is True:
                        error_msg += "Model/Checkpoint file does not contain a CLIP\n"
                    if error_msg != "":
                        raise ValueError(error_msg)
                    else:               
                        out = (comfy.sd.load_diffusion_model(ckpt_path, model_options),)
                else:
                    raise e
            
            return out
            
        except FileNotFoundError:
            ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

            model = comfy.sd.load_diffusion_model(ckpt_path, model_options=model_options)
            return (model, )



    def load_vae(self, vae_name, ckpt_out):
        if vae_name == ".use_ckpt_vae":
            if ckpt_out[2] is None:
                raise ValueError("Model does not have a VAE")
            return ckpt_out[2]
        elif vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
            return comfy.sd.VAE(sd=sd)
        elif vae_name == ".none":
            return None
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            return comfy.sd.VAE(sd=sd)


def load_clipvision(ckpt_path):
    sd = load_torch_file(ckpt_path)
    clip_vision = comfy.clip_vision.load(ckpt_path)
    return clip_vision
    
class FluxLoader(BaseModelLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name":       (s.get_model_files(),),
            "weight_dtype":     (s.get_weight_options(),),
            "clip_name1":       (s.get_clip_options(),),
            "clip_name2_opt":   ([".none"]         + folder_paths.get_filename_list("text_encoders"),),
            "vae_name":         ([".use_ckpt_vae"] + s.vae_list(),),
            "clip_vision_name": ([".none"]         + folder_paths.get_filename_list("clip_vision"),),
            "style_model_name": ([".none"]         + folder_paths.get_filename_list("style_models"),),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION", "STYLE_MODEL")
    RETURN_NAMES = ("model", "clip", "vae", "clip_vision", "style_model")
    FUNCTION = "main"
    CATEGORY = "RES4LYF/loaders"

    def main(self, model_name, weight_dtype, clip_name1, clip_name2_opt, vae_name, clip_vision_name, style_model_name):
        model_options = self.process_weight_dtype(weight_dtype)
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if clip_name1 == ".use_ckpt_clip" and clip_name2_opt != ".none":
            raise ValueError("Cannot specify both \".use_ckpt_clip\" and another clip")
        
        output_vae = vae_name == ".use_ckpt_vae"
        output_clip = clip_name1 == ".use_ckpt_clip"
        ckpt_out = self.load_checkpoint(model_name, output_vae, output_clip, model_options)

        if clip_name1 == ".use_ckpt_clip":
            if ckpt_out[1] is None:
                raise ValueError("Model does not have a clip")
            clip = ckpt_out[1]
        else:
            clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", clip_name1)]
            if clip_name2_opt != ".none":
                clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name2_opt))
            clip = comfy.sd.load_clip(clip_paths, 
                                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                    clip_type=comfy.sd.CLIPType.FLUX)

        clip_vision = None if clip_vision_name == ".none" else \
            load_clipvision(folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name))

        style_model = None if style_model_name == ".none" else \
            comfy.sd.load_style_model(folder_paths.get_full_path_or_raise("style_models", style_model_name))
            
        vae = self.load_vae(vae_name, ckpt_out)
        
        return (ckpt_out[0], clip, vae, clip_vision, style_model)

class SD35Loader(BaseModelLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (s.get_model_files(),),
            "weight_dtype": (s.get_weight_options(),),
            "clip_name1": (s.get_clip_options(),),
            "clip_name2_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"),),
            "clip_name3_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"),),
            "vae_name": ([".use_ckpt_vae"] + folder_paths.get_filename_list("vae") + ["taesd", "taesdxl", "taesd3", "taef1"],),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "main"
    CATEGORY = "RES4LYF/loaders"
    
    def main(self, model_name, weight_dtype, clip_name1, clip_name2_opt, clip_name3_opt, vae_name):
        model_options = self.process_weight_dtype(weight_dtype)
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        if clip_name1 == ".use_ckpt_clip" and (clip_name2_opt != ".none" or clip_name3_opt != ".none"):
            raise ValueError("Cannot specify both \".use_ckpt_clip\" and another clip")

        output_vae = vae_name == ".use_ckpt_vae"
        output_clip = clip_name1 == ".use_ckpt_clip"
        ckpt_out = self.load_checkpoint(model_name, output_vae, output_clip, model_options)

        if clip_name1 == ".use_ckpt_clip":
            if ckpt_out[1] is None:
                raise ValueError("Model does not have a clip")
            clip = ckpt_out[1]
        else:
            clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", clip_name1)]
            for clip_name in [clip_name2_opt, clip_name3_opt]:
                if clip_name != ".none":
                    clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name))
            clip = comfy.sd.load_clip(clip_paths,
                                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                    clip_type=comfy.sd.CLIPType.SD3)

        vae = self.load_vae(vae_name, ckpt_out)

        return (ckpt_out[0], clip, vae)
    
class RES4LYFModelLoader(BaseModelLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name":     (s.get_model_files(),),
            "weight_dtype":   (s.get_weight_options(),),
            "clip_name1_opt": ([".none"] + s.get_clip_options(),),
            "clip_name2_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"),),
            "clip_name3_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"),),
            "clip_name4_opt": ([".none"] + folder_paths.get_filename_list("text_encoders"),),
            "clip_type":      ([".auto"] + clip_types,),
            "vae_name":       ([".none", ".use_ckpt_vae"] + folder_paths.get_filename_list("vae") + ["taesd", "taesdxl", "taesd3", "taef1"],),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "main"
    CATEGORY = "RES4LYF/loaders"
    
    def main(self, model_name, weight_dtype, clip_name1_opt, clip_name2_opt, clip_name3_opt, clip_name4_opt, clip_type, vae_name):
        model_options = self.process_weight_dtype(weight_dtype)
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if clip_name1_opt == ".use_ckpt_clip" and (clip_name2_opt != ".none" or clip_name3_opt != ".none" or clip_name4_opt != ".none"):
            raise ValueError("Cannot specify both \".use_ckpt_clip\" and another clip")

        output_vae = vae_name == ".use_ckpt_vae"
        output_clip = clip_name1_opt == ".use_ckpt_clip"
        ckpt_out = self.load_checkpoint(model_name, output_vae, output_clip, model_options)

        if clip_name1_opt == ".use_ckpt_clip":
            if ckpt_out[1] is None:
                raise ValueError("Model does not have a clip")
            clip = ckpt_out[1]
        elif clip_name1_opt == ".none":
            clip = None
        else:
            clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", clip_name1_opt)]
            for clip_name in [clip_name2_opt, clip_name3_opt, clip_name4_opt]:
                if clip_name != ".none":
                    clip_paths.append(folder_paths.get_full_path_or_raise("text_encoders", clip_name))
            if "auto" in clip_type and ckpt_out[0].model is not None:
                sdCLIPType = getattr(comfy.sd.CLIPType, self.guess_clip_type(ckpt_out[0].model), comfy.sd.CLIPType.STABLE_DIFFUSION)
            else:
                sdCLIPType = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip = comfy.sd.load_clip(clip_paths,
                                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                    clip_type=sdCLIPType)

        vae = self.load_vae(vae_name, ckpt_out)

        return (ckpt_out[0], clip, vae)

from .style_transfer import Retrojector
import torch.nn as nn

class LayerPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model":       ("MODEL",),
            "embedder":    (s.get_model_patches(),),
            "gates":       (s.get_model_patches(),),
            "last_layer":  (s.get_model_patches(),),
            "dtype":       (["bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
            #"retrojector": (s.get_model_patches(),),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main"
    CATEGORY = "RES4LYF/patchers"
    
    @staticmethod
    def get_model_patches():
        return [f for f in folder_paths.get_filename_list("diffusion_models") if f.endswith((".safetensors", ".sft"))]
    
    def main(self, model, embedder, gates, last_layer, retrojector=None, dtype="float64"):
        
        dtype = getattr(torch, dtype)
        
        embedder    = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("diffusion_models", embedder))
        last_layer  = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("diffusion_models", last_layer))
        #retrojector = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("diffusion_models", retrojector))
        
        gates  = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("diffusion_models", gates))
        m = model.model.diffusion_model
        
        if embedder:
            m.x_embedder.proj = nn.Linear(
                m.x_embedder.proj.in_features, 
                m.x_embedder.proj.out_features, 
                bias=True,
                device=m.x_embedder.proj.weight.data.device,
                dtype=dtype
                )
            m.x_embedder.proj.weight.data = embedder['x_embedder.proj.weight'].to(dtype)
            m.x_embedder.proj.bias.data   = embedder['x_embedder.proj.bias'].to(dtype)
        
        if gates:
            for key, tensor in gates.items():
                #print(f"Patching {key} with shape {tensor.shape}")
                set_nested_attr(model=m, key=key, value=tensor, dtype=dtype)
        
        if last_layer:
            m.final_layer.linear.weight.data = last_layer['final_layer.linear.weight'].to(dtype)
            m.final_layer.linear.bias.data   = last_layer['final_layer.linear.bias'].to(dtype)
            m.final_layer.adaLN_modulation[1].weight.data = last_layer['final_layer.adaLN_modulation.1.weight'].to(dtype)
            m.final_layer.adaLN_modulation[1].bias.data = last_layer['final_layer.adaLN_modulation.1.bias'].to(dtype)

        #if retrojector:
        #    m.Retrojector = Retrojector(model.model.diffusion_model.img_in, pinv_dtype=style_dtype, dtype=style_dtype)
        #    m.final_layer.linear.weight.data = last_layer['final_layer.linear.weight']
        #    m.final_layer.linear.bias.data   = last_layer['final_layer.linear.bias']
        #    m.final_layer.adaLN_modulation[1].weight.data = last_layer['final_layer.adaLN_modulation.1.weight']
        #    m.final_layer.adaLN_modulation[1].bias.data = last_layer['final_layer.adaLN_modulation.1.bias']

        return (model,)



def set_nested_attr(model, key, value, dtype):
    parts = key.split(".")
    attr = model
    for p in parts[:-1]:
        if p.isdigit():
            attr = attr[int(p)]
        else:
            attr = getattr(attr, p)
    getattr(attr, parts[-1]).data.copy_(value.to(getattr(attr, parts[-1]).device, dtype=dtype))



