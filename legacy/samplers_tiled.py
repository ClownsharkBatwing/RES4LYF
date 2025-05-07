# tiled sampler code adapted from https://github.com/BlenderNeko/ComfyUI_TiledKSampler 
# and heavily modified for use with https://github.com/ClownsharkBatwing/UltraCascade

import sys
import os
import copy
from functools import partial

from tqdm.auto import tqdm

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd
import comfy.controlnet
import comfy.model_management
import comfy.sample
import comfy.sampler_helpers
import latent_preview

from nodes import MAX_RESOLUTION
#MAX_RESOLUTION=8192

import comfy.clip_vision
import folder_paths

from . import tiling
from .noise_classes import *

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor

def cv_cond(cv_out, conditioning, strength, noise_augmentation): 

    c = []
    for t in conditioning:
        o = t[1].copy()
        x = {"clip_vision_output": cv_out, "strength": strength, "noise_augmentation": noise_augmentation}
        if "unclip_conditioning" in o:
            o["unclip_conditioning"] = o["unclip_conditioning"][:] + [x]
        else:
            o["unclip_conditioning"] = [x]
        n = [t[0], o]
        c.append(n)
    
    return c


def recursion_to_list(obj, attr):
    current = obj
    yield current
    while True:
        current = getattr(current, attr, None)
        if current is not None:
            yield current
        else:
            return

def copy_cond(cond):
    return [[c1,c2.copy()] for c1,c2 in cond]

def slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, cond, area):
    tile_h_end = tile_h + tile_h_len
    tile_w_end = tile_w + tile_w_len
    coords = area[0] #h_len, w_len, h, w,
    mask = area[1]
    if coords is not None:
        h_len, w_len, h, w = coords
        h_end = h + h_len
        w_end = w + w_len
        if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
            new_h = max(0, h - tile_h)
            new_w = max(0, w - tile_w)
            new_h_end = min(tile_h_end, h_end - tile_h)
            new_w_end = min(tile_w_end, w_end - tile_w)
            cond[1]['area'] = (new_h_end - new_h, new_w_end - new_w, new_h, new_w)
        else:
            return (cond, True)
    if mask is not None:
        new_mask = tiling.get_slice(mask, tile_h,tile_h_len,tile_w,tile_w_len)
        if new_mask.sum().cpu() == 0.0 and 'mask' in cond[1]:
            return (cond, True)
        else:
            cond[1]['mask'] = new_mask
    return (cond, False)

def slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen):
    tile_h_end = tile_h + tile_h_len
    tile_w_end = tile_w + tile_w_len
    if gligen is None:
        return
    gligen_type = gligen[0]
    gligen_model = gligen[1]
    gligen_areas = gligen[2]
    
    gligen_areas_new = []
    for emb, h_len, w_len, h, w in gligen_areas:
        h_end = h + h_len
        w_end = w + w_len
        if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
            new_h = max(0, h - tile_h)
            new_w = max(0, w - tile_w)
            new_h_end = min(tile_h_end, h_end - tile_h)
            new_w_end = min(tile_w_end, w_end - tile_w)
            gligen_areas_new.append((emb, new_h_end - new_h, new_w_end - new_w, new_h, new_w))

    if len(gligen_areas_new) == 0:
        del cond['gligen']
    else:
        cond['gligen'] = (gligen_type, gligen_model, gligen_areas_new)

def slice_cnet(h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
    if img is None:
        img = model.cond_hint_original
    hint = tiling.get_slice(img, h*8, h_len*8, w*8, w_len*8)
    if isinstance(model, comfy.controlnet.ControlLora):
        model.cond_hint = hint.float().to(model.device)
    else:
        model.cond_hint = hint.to(model.control_model.dtype).to(model.device)

def slices_T2I(h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
    model.control_input = None
    if img is None:
        img = model.cond_hint_original
    model.cond_hint = tiling.get_slice(img, h*8, h_len*8, w*8, w_len*8).float().to(model.device)

# TODO: refactor some of the mess


def cnets_and_cnet_imgs(positive, negative, shape): 
    # cnets
    cnets =  [c['control'] for (_, c) in positive + negative if 'control' in c]
    # unroll recursion
    cnets = list(set([x for m in cnets for x in recursion_to_list(m, "previous_controlnet")]))
    # filter down to only cnets
    cnets = [x for x in cnets if isinstance(x, comfy.controlnet.ControlNet)]
    cnet_imgs = [
        torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
        if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
        for m in cnets]
    return cnets, cnet_imgs

def T2Is_and_T2I_imgs(positive, negative, shape): 
    # T2I
    T2Is =  [c['control'] for (_, c) in positive + negative if 'control' in c]
    # unroll recursion
    T2Is = [x for m in T2Is for x in recursion_to_list(m, "previous_controlnet")]
    # filter down to only T2I
    T2Is = [x for x in T2Is if isinstance(x, comfy.controlnet.T2IAdapter)]
    T2I_imgs = [
        torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
        if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 or (m.channels_in == 1 and m.cond_hint_original.shape[1] != 1) else None
        for m in T2Is
    ]
    T2I_imgs = [
        torch.mean(img, 1, keepdim=True) if img is not None and m.channels_in == 1 and m.cond_hint_original.shape[1] else img
        for m, img in zip(T2Is, T2I_imgs)
    ]
    return T2Is, T2I_imgs

def spatial_conds_posneg(positive, negative, shape, device): #cond area and mask
    spatial_conds_pos = [
        (c[1]['area'] if 'area' in c[1] else None, 
            comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
        for c in positive
    ]
    spatial_conds_neg = [
        (c[1]['area'] if 'area' in c[1] else None, 
            comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
        for c in negative
    ]
    return spatial_conds_pos, spatial_conds_neg 

def gligen_posneg(positive, negative):
    #gligen
    gligen_pos = [
        c[1]['gligen'] if 'gligen' in c[1] else None
        for c in positive
    ]
    gligen_neg = [
        c[1]['gligen'] if 'gligen' in c[1] else None
        for c in negative
    ]
    return gligen_pos, gligen_neg


def cascade_tiles(x, input_x, tile_h, tile_w, tile_h_len, tile_w_len):
    h_cascade = input_x.shape[-2]
    w_cascade = input_x.shape[-1]
    
    h_samples = x.shape[-2]
    w_samples = x.shape[-1]
    
    tile_h_cascade = (h_cascade * tile_h) // h_samples
    tile_w_cascade = (w_cascade * tile_w) // w_samples
    
    tile_h_len_cascade = (h_cascade * tile_h_len) // h_samples
    tile_w_len_cascade = (w_cascade * tile_w_len) // w_samples
    
    return tile_h_cascade, tile_w_cascade, tile_h_len_cascade, tile_w_len_cascade



def sample_common(model, x, noise, noise_mask, noise_seed, tile_width, tile_height, tiling_strategy, cfg, positive, negative,
                  preview=False, sampler=None, sigmas=None,
                  clip_name=None, strength=1.0, noise_augment=1.0, image_cv=None, max_tile_batch_size=3,
                  guide=None, guide_type='residual', guide_weight=1.0, guide_weights=None,
                  ):

    device = comfy.model_management.get_torch_device()
    steps = len(sigmas)-1
    
    conds0 = \
        {"positive": comfy.sampler_helpers.convert_cond(positive),
         "negative": comfy.sampler_helpers.convert_cond(negative)}

    conds = {}
    for k in conds0:
        conds[k] = list(map(lambda a: a.copy(), conds0[k]))

    modelPatches, inference_memory = comfy.sampler_helpers.get_additional_models(conds, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + modelPatches, model.memory_required(noise.shape) + inference_memory)
    

    if model.model.model_config.unet_config['stable_cascade_stage'] == 'up':
        compression = 1
        guide_weight = 1.0 if guide_weight is None else guide_weight
        guide_type = 'residual' if guide_type is None else guide_type
        guide = guide['samples'] if guide is not None else None
        guide_weights = initialize_or_scale(guide_weights, guide_weight, 10000)

        patch = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("ultracascade", {}).get("main")  #CHANGED HERE
        if patch is not None:
            patch.update(x_lr=guide, guide_weights=guide_weights, guide_type=guide_type)
        else:
            model = model.clone()
            model.model.diffusion_model.set_sigmas_prev(sigmas_prev=sigmas[:1])
            model.model.diffusion_model.set_guide_weights(guide_weights=guide_weights)
            model.model.diffusion_model.set_guide_type(guide_type=guide_type)
        
    elif model.model.model_config.unet_config['stable_cascade_stage'] == 'c':
        compression = 1
        
    elif model.model.model_config.unet_config['stable_cascade_stage'] == 'b':
        compression = 4
        
        c_pos, c_neg = [], []
        for t in positive:
            d_pos = t[1].copy()
            d_neg = t[1].copy()
            
            d_pos['stable_cascade_prior'] = guide['samples']

            pooled_output = d_neg.get("pooled_output", None)
            if pooled_output is not None:
                d_neg["pooled_output"] = torch.zeros_like(pooled_output)
            
            c_pos.append([t[0], d_pos])            
            c_neg.append([torch.zeros_like(t[0]), d_neg])
        positive = c_pos
        negative = c_neg
        effnet_samples = positive[0][1]['stable_cascade_prior'].clone()
        effnet_interpolated = nn.functional.interpolate(effnet_samples.clone().to(torch.float16).to(device), size=torch.Size((x.shape[-2] // 2, x.shape[-1] // 2,)), mode='bilinear', align_corners=True)
        effnet_full_map = model.model.diffusion_model.effnet_mapper(effnet_interpolated)
    else:
        compression = 8 #sd1.5, sdxl, sd3, flux, etc
        
    
    if image_cv is not None: #CLIP VISION LOAD
        clip_path = folder_paths.get_full_path("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        



    cnets,             cnet_imgs         = cnets_and_cnet_imgs (positive, negative, x.shape)
    T2Is,              T2I_imgs          = T2Is_and_T2I_imgs   (positive, negative, x.shape)
    spatial_conds_pos, spatial_conds_neg = spatial_conds_posneg(positive, negative, x.shape, device)
    gligen_pos,        gligen_neg        = gligen_posneg       (positive, negative)
    
    
    
    tile_width  = min(x.shape[-1] * compression, tile_width) 
    tile_height = min(x.shape[2]  * compression, tile_height)
    
    if tiling_strategy != 'padded':
        if noise_mask is not None:
            x += sigmas[0] * noise_mask * model.model.process_latent_out(noise)
        else:
            x += sigmas[0] * model.model.process_latent_out(noise)
    


    if tiling_strategy == 'random' or tiling_strategy == 'random strict':
        tiles = tiling.get_tiles_and_masks_rgrid(steps, x.shape, tile_height, tile_width, torch.manual_seed(noise_seed), compression=compression)
    elif tiling_strategy == 'padded':
        tiles = tiling.get_tiles_and_masks_padded(steps, x.shape, tile_height, tile_width, compression=compression)
    else:
        tiles = tiling.get_tiles_and_masks_simple(steps, x.shape, tile_height, tile_width, compression=compression)



    total_steps = sum([num_steps for img_pass in tiles for steps_list in img_pass for _,_,_,_,num_steps,_ in steps_list])
    current_step = [0]
    with tqdm(total=total_steps) as pbar_tqdm:
        pbar = comfy.utils.ProgressBar(total_steps)
        def callback(step, x0, x, total_steps, step_inc=1):
            current_step[0] += step_inc
            preview_bytes = None
            if preview == True:
                previewer = latent_preview.get_previewer(device, model.model.latent_format)
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
            pbar.update_absolute(current_step[0], preview=preview_bytes)
            pbar_tqdm.update(step_inc)
            
            
            
        if tiling_strategy == "random strict":
            x_next = x.clone()
            
        for img_pass in tiles: # img_pass is a set of non-intersecting tiles
            effnet_slices, effnet_map_slices, tiled_noise_list, tiled_latent_list, tiled_mask_list, tile_h_list, tile_w_list, tile_h_len_list, tile_w_len_list = [],[],[],[],[],[],[],[],[]

            for i in range(len(img_pass)):     
                for iteration, (tile_h, tile_h_len, tile_w, tile_w_len, tile_steps, tile_mask) in enumerate(img_pass[i]):
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = tiling.get_slice(noise_mask, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                    if tile_mask is not None:
                        if tiled_mask is not None:
                            tiled_mask *= tile_mask.to(device)
                        else:
                            tiled_mask  = tile_mask.to(device)
                    
                    if tiling_strategy == 'padded' or tiling_strategy == 'random strict':
                        tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask = tiling.mask_at_boundary(   tile_h, tile_h_len, tile_w, tile_w_len, 
                                                                                                        tile_height, tile_width, x.shape[-2], x.shape[-1],
                                                                                                        tiled_mask, device, compression=compression)
                        
                    if tiled_mask is not None and tiled_mask.sum().cpu() == 0.0:
                            continue
                            
                    tiled_latent = tiling.get_slice(x, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                    
                    if tiling_strategy == 'padded':
                        tiled_noise = tiling.get_slice(noise, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                    else:
                        if tiled_mask is None or noise_mask is None:
                            tiled_noise = torch.zeros_like(tiled_latent)
                        else:
                            tiled_noise = tiling.get_slice(noise, tile_h, tile_h_len, tile_w, tile_w_len).to(device) * (1 - tiled_mask)
                    
                    #TODO: all other condition based stuff like area sets and GLIGEN should also happen here

                    #cnets
                    for m, img in zip(cnets, cnet_imgs):
                        slice_cnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
                    
                    #T2I
                    for m, img in zip(T2Is, T2I_imgs):
                        slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)

                    pos = copy.deepcopy(positive)
                    neg = copy.deepcopy(negative)

                    #cond areas
                    pos = [slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(pos, spatial_conds_pos)]
                    pos = [c for c, ignore in pos if not ignore]
                    neg = [slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(neg, spatial_conds_neg)]
                    neg = [c for c, ignore in neg if not ignore]

                    #gligen
                    for cond, gligen in zip(pos, gligen_pos):
                        slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
                    for cond, gligen in zip(neg, gligen_neg):
                        slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
                    
                    start_step = i * tile_steps
                    last_step  = i * tile_steps + tile_steps

                    if last_step is not None and last_step < (len(sigmas) - 1):
                        sigmas = sigmas[:last_step + 1]

                    if start_step is not None:
                        if start_step < (len(sigmas) - 1):
                            sigmas = sigmas[start_step:]
                        else:
                            if tiled_latent is not None:
                                return tiled_latent
                            else:
                                return torch.zeros_like(noise)
                
                    # SLICE, DICE, AND DENOISE
                    if image_cv is not None: #slice and dice ClipVision for tiling
                        image_cv    = image_cv.   permute(0,3,1,2)
                        tile_h_cascade, tile_w_cascade, tile_h_len_cascade, tile_w_len_cascade = cascade_tiles(x, image_cv, tile_h, tile_w, tile_h_len, tile_w_len)

                        image_slice = copy.deepcopy(image_cv)
                        image_slice = tiling.get_slice(image_slice, tile_h_cascade, tile_h_len_cascade, tile_w_cascade, tile_w_len_cascade).to(device)
                        image_slice = image_slice.permute(0,2,3,1)
                        image_cv    = image_cv.   permute(0,2,3,1)
                        
                        cv_out_slice = clip_vision.encode_image(image_slice)
                        pos = cv_cond(cv_out_slice, pos, strength, noise_augment) 
                    
                    if model.model.model_config.unet_config['stable_cascade_stage'] == 'up': #slice and dice stage UP guide
                        tile_h_cascade, tile_w_cascade, tile_h_len_cascade, tile_w_len_cascade = cascade_tiles(x, guide, tile_h, tile_w, tile_h_len, tile_w_len)

                        guide_slice = copy.deepcopy(guide)
                        guide_slice = tiling.get_slice(guide_slice.clone(), tile_h_cascade, tile_h_len_cascade, tile_w_cascade, tile_w_len_cascade).to(device)
                        model.model.diffusion_model.set_x_lr(x_lr=guide_slice)
                        
                        tile_result = comfy.sample.sample_custom(model, tiled_noise, cfg, sampler, sigmas, pos, neg, tiled_latent, noise_mask=tiled_mask, callback=callback, disable_pbar=True, seed=noise_seed)  

                    elif model.model.model_config.unet_config['stable_cascade_stage'] == 'b':  #slice and dice stage B conditioning
                        tile_h_cascade, tile_w_cascade, tile_h_len_cascade, tile_w_len_cascade = cascade_tiles(x, effnet_samples.clone(), tile_h, tile_w, tile_h_len, tile_w_len)
                        effnet_slice = tiling.get_slice(effnet_samples.clone(), tile_h_cascade, tile_h_len_cascade, tile_w_cascade, tile_w_len_cascade).to(device)
                        effnet_slices.append(effnet_slice)
                        
                        tile_h_cascade, tile_w_cascade, tile_h_len_cascade, tile_w_len_cascade = cascade_tiles(x, effnet_full_map.clone(), tile_h, tile_w, tile_h_len, tile_w_len)
                        effnet_map_slice = tiling.get_slice(effnet_full_map.clone(), tile_h_cascade, tile_h_len_cascade, tile_w_cascade, tile_w_len_cascade).to(device)
                        effnet_map_slices.append(effnet_map_slice)

                    else: # not stage UP or stage B, default
                        tile_result = comfy.sample.sample_custom(model, tiled_noise, cfg, sampler, sigmas, pos, neg, tiled_latent, noise_mask=tiled_mask, callback=callback, disable_pbar=True, seed=noise_seed)  

                    if model.model.model_config.unet_config['stable_cascade_stage'] != 'b':
                        tile_result = tile_result.cpu()
                        if tiled_mask is not None:
                            tiled_mask = tiled_mask.cpu()
                        if tiling_strategy == "random strict":
                            tiling.set_slice(x_next, tile_result, tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask)
                        else:
                            tiling.set_slice(x, tile_result, tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask)
                        

                    tiled_noise_list .append(tiled_noise)
                    tiled_latent_list.append(tiled_latent)
                    tiled_mask_list  .append(tiled_mask)
                    tile_h_list      .append(tile_h)
                    tile_w_list      .append(tile_w)
                    tile_h_len_list  .append(tile_h_len)
                    tile_w_len_list  .append(tile_w_len)
                    
                    #END OF NON-INTERSECTING SET OF TILES
                    
                if tiling_strategy == "random strict":   # IS THIS ONE LEVEL OVER??
                    x = x_next.clone()
            
            if model.model.model_config.unet_config['stable_cascade_stage'] == 'b':

                for start_idx in range(0, len(tiled_latent_list), max_tile_batch_size):
                    
                    end_idx = start_idx + max_tile_batch_size
                    
                    #print("Tiled batch size: ", min(max_tile_batch_size, len(tiled_latent_list))) #end_idx - start_idx)
                    
                    tiled_noise_batch  = torch.cat(tiled_noise_list [start_idx:end_idx])
                    tiled_latent_batch = torch.cat(tiled_latent_list[start_idx:end_idx])
                    tiled_mask_batch   = torch.cat(tiled_mask_list  [start_idx:end_idx])
                    
                    print("Tiled batch size: ", tiled_latent_batch.shape[0])

                    pos[0][1]['stable_cascade_prior'] = torch.cat(effnet_slices[start_idx:end_idx])
                    neg[0][1]['stable_cascade_prior'] = torch.cat(effnet_slices[start_idx:end_idx])
                    
                    tile_result = comfy.sample.sample_custom(model, tiled_noise_batch, cfg, sampler, sigmas, pos, neg, tiled_latent_batch, noise_mask=tiled_mask_batch, callback=partial(callback, step_inc=tiled_latent_batch.shape[0]), disable_pbar=True, seed=noise_seed)
                    
                    for i in range(tile_result.shape[0]):
                        idx = start_idx + i
                        
                        single_tile = tile_result[i].unsqueeze(dim=0)
                        single_mask = tiled_mask_batch[i].unsqueeze(dim=0)
                        
                        tiling.set_slice(x, single_tile, tile_h_list[idx], tile_h_len_list[idx], tile_w_list[idx], tile_w_len_list[idx], single_mask.cpu())

                x = x.to('cpu') 

    comfy.sampler_helpers.cleanup_additional_models(modelPatches)

    return x.cpu()



class UltraSharkSampler_Tiled: #this is for use with https://github.com/ClownsharkBatwing/UltraCascade
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_is_latent": ("BOOLEAN", {"default": False}),
                    "noise_type": (NOISE_GENERATOR_NAMES, ),
                    "alpha": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":0.1, "round": 0.01}),
                    "k": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step":2.0, "round": 0.01}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "guide_type": (['residual', 'weighted'], ),
                    "guide_weight": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.01, "round": 0.01}),
                    
                    "tile_width": ("INT", {"default": 1024, "min": 2, "max": MAX_RESOLUTION, "step": 1}),
                    "tile_height": ("INT", {"default": 1024, "min": 2, "max": MAX_RESOLUTION, "step": 1}),
                    "tiling_strategy": (["padded", "random", "random strict",  'simple'], ),
                    "max_tile_batch_size": ("INT", {"default": 64, "min": 1, "max": 256, "step": 1}),

                    "model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER",),
                    "sigmas": ("SIGMAS",),
                    "latent_image": ("LATENT", ),
                    
                    "clip_name":            (folder_paths.get_filename_list("clip_vision"), {'default': "clip-vit-large-patch14.safetensors"}),
                    "strength":           ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "noise_augment":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                    },
                    "optional": {
                        "latent_noise": ("LATENT", ),
                        "guide": ("LATENT", ),
                        "guide_weights": ("SIGMAS",),
                        "image_cv": ("IMAGE",),

                    },
                    
                    }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "RES4LYF/legacy/samplers/ultracascade"
    DESCRIPTION = "For use with UltraCascade."
    DEPRECATED = True

    def sample(self, model, noise_seed, add_noise, noise_is_latent, noise_type, alpha, k, tile_width, tile_height, tiling_strategy, cfg, positive, negative, latent_image, latent_noise=None, sampler=None, sigmas=None, guide=None,
               clip_name=None, strength=1.0, noise_augment=1.0, image_cv=None, max_tile_batch_size=3,
               guide_type='residual', guide_weight=1.0, guide_weights=None,
               ):
        
        x = latent_image["samples"].clone()

        torch.manual_seed(noise_seed)

        if not add_noise:
            noise = torch.zeros(x.size(), dtype=x.dtype, layout=x.layout, device="cpu")
        elif latent_noise is None:
            skip = latent_image["batch_index"] if "batch_index" in latent_image else None
            noise = prepare_noise(x, noise_seed, noise_type, skip, alpha, k)
        else:
            noise = latent_noise["samples"]

        if noise_is_latent: #add noise and latent together and normalize --> noise
            noise += x.cpu()
            noise.sub_(noise.mean()).div_(noise.std())

        noise_mask = latent_image["noise_mask"].clone() if "noise_mask" in latent_image else None

        latent_out = latent_image.copy()
        latent_out['samples'] = sample_common(model, x=x, noise=noise, noise_mask=noise_mask, noise_seed=noise_seed, tile_width=tile_width, tile_height=tile_height, tiling_strategy=tiling_strategy, cfg=cfg, positive=positive, negative=negative, 
                             preview=True, sampler=sampler, sigmas=sigmas,
                             clip_name=clip_name, strength=strength, noise_augment=noise_augment, image_cv=image_cv, max_tile_batch_size=max_tile_batch_size,
                             guide=guide, guide_type=guide_type, guide_weight=guide_weight, guide_weights=guide_weights,
                             )
        return (latent_out,)


