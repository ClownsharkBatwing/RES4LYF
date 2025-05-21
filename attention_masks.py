import torch
import torch.nn.functional as F

from torch  import Tensor
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

from einops import rearrange

import copy
import base64

import comfy.supported_models
import node_helpers
import gc


from .sigmas  import get_sigmas

from .helper  import initialize_or_scale, precision_tool, get_res4lyf_scheduler_list
from .latents import get_orthogonal, get_collinear, get_edge_mask, checkerboard_variable
from .res4lyf import RESplain
from .beta.constants import MAX_STEPS



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



class CoreAttnMask:
    def __init__(self, mask, mask_type=None, start_sigma=None, end_sigma=None, start_block=0, end_block=-1, idle_device='cpu', work_device='cuda'):
        self.mask        = mask.to(idle_device)
        self.start_sigma = start_sigma
        self.end_sigma   = end_sigma
        self.start_block = start_block
        self.end_block   = end_block
        self.work_device = work_device
        self.idle_device = idle_device
        self.mask_type   = mask_type
    
    def set_sigma_range(self, start_sigma, end_sigma):
        self.start_sigma = start_sigma
        self.end_sigma   = end_sigma
        
    def set_block_range(self, start_block, end_block):
        self.start_block = start_block
        self.end_block   = end_block

    def __call__(self, weight=1.0, mask_type=None, transformer_options=None, block_idx=0):
        """ 
        Return mask if block_idx is in range, sigma passed via transformer_options is in range, else return None. If no range is specified, return mask.
        """
        if block_idx < self.start_block:
            return None
        if block_idx > self.end_block and self.end_block > 0:
            return None
        
        mask_type = self.mask_type if mask_type is None else mask_type
        
        if transformer_options is None:
            return self.mask.to(self.work_device) * weight if mask_type.startswith("gradient") else self.mask.to(self.work_device) > 0

        sigma = transformer_options['sigmas'][0].to(self.start_sigma.device)
        
        if self.start_sigma is not None and self.end_sigma is not None:
            if self.start_sigma >= sigma > self.end_sigma:
                return self.mask.to(self.work_device) * weight if mask_type.startswith("gradient") else self.mask.to(self.work_device) > 0
        else:
            return self.mask.to(self.work_device) * weight if mask_type.startswith("gradient") else self.mask.to(self.work_device) > 0
        
        return None



class BaseAttentionMask:
    def __init__(self, mask_type="gradient", edge_width=0, edge_width_list=None, use_self_attn_mask_list=None, dtype=torch.float16):
        self.t                    = 1
        self.img_len              = 0
        self.text_len             = 0
        self.text_off             = 0

        self.h                    = 0
        self.w                    = 0
    
        self.text_register_tokens = 0
        
        self.context_lens         = []
        self.context_lens_list    = []
        self.masks                = []
        
        self.num_regions          = 0
        
        self.attn_mask            = None
        self.mask_type            = mask_type
        self.edge_width           = edge_width
        
        self.edge_width_list      = edge_width_list
        self.use_self_attn_mask_list = use_self_attn_mask_list
        
        if mask_type == "gradient":
            self.dtype            = dtype
        else:
            self.dtype            = torch.bool


    def set_latent(self, latent):
        if latent.ndim == 4:
            self.b, self.c, self.h, self.w = latent.shape
            
        elif latent.ndim == 5:
            self.b, self.c, self.t, self.h, self.w = latent.shape
            
        #if not isinstance(self.model_config, comfy.supported_models.Stable_Cascade_C):
        self.h //= 2  # 16x16 PE      patch_size = 2  1024x1024 rgb -> 128x128 16ch latent -> 64x64 img
        self.w //= 2
            
        self.img_len = self.h * self.w        

    def add_region(self, context, mask):
        self.context_lens.append(context.shape[-2])
        self.masks       .append(mask)
        
        self.text_len = sum(self.context_lens)
        self.text_off = self.text_len
        
        self.num_regions += 1
        
    def add_region_sizes(self, context_size_list, mask):
        
        self.context_lens     .append(sum(context_size_list))
        self.context_lens_list.append(    context_size_list)
        self.masks            .append(mask)
        
        self.text_len = sum(sum(sublist) for sublist in self.context_lens_list)
        self.text_off = self.text_len

        self.num_regions += 1
        
    def add_regions(self, contexts, masks):
        for context, mask in zip(contexts, masks):
            self.add_region(context, mask)
    
    def clear_regions(self):
        self.context_lens  = []
        self.masks         = []
        self.text_len      = 0
        self.text_off      = 0
        self.num_regions   = 0
        
    def generate(self):
        print("Initializing ergosphere.")
        
    def get(self, **kwargs):
        return self.attn_mask(**kwargs)
    
    def attn_mask_recast(self, dtype):
        if self.attn_mask.mask.dtype != dtype:
            self.attn_mask.mask = self.attn_mask.mask.to(dtype)




class FullAttentionMask(BaseAttentionMask):
    def generate(self, mask_type=None, dtype=None):
        mask_type = self.mask_type if mask_type is None else mask_type
        dtype     = self.dtype     if dtype     is None else dtype
        text_off  = self.text_off
        text_len  = self.text_len
        img_len   = self.img_len
        t         = self.t
        h         = self.h
        w         = self.w
        
        if self.edge_width_list is None:
            self.edge_width_list = [self.edge_width] * self.num_regions
        
        attn_mask = torch.zeros((text_off+t*img_len, text_len+t*img_len), dtype=dtype)
        
        prev_len = 0
        for context_len, mask in zip(self.context_lens, self.masks):
            
            img2txt_mask    = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, context_len)
            img2txt_mask_sq = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)

            curr_len = prev_len + context_len
            
            attn_mask[prev_len:curr_len, prev_len:curr_len] = 1.0                                         # self             TXT 2 TXT
            attn_mask[prev_len:curr_len, text_len:        ] = img2txt_mask.transpose(-1, -2).repeat(1,t)  # cross            TXT 2 regional IMG    # txt2img_mask
            attn_mask[text_off:        , prev_len:curr_len] = img2txt_mask.repeat(t,1)                    # cross   regional IMG 2 TXT

            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], fp_and(img2txt_mask_sq.repeat(t,t), img2txt_mask_sq.transpose(-1, -2).repeat(t,t))) # img2txt_mask_sq, txt2img_mask_sq
            
            prev_len = curr_len
            
        if self.mask_type.endswith("_masked") or self.mask_type.endswith("_A") or self.mask_type.endswith("_AB") or self.mask_type.endswith("_AC") or self.mask_type.endswith("_A,unmasked"):
            img2txt_mask_sq = F.interpolate(self.masks[0].unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
        
        if self.mask_type.endswith("_unmasked") or self.mask_type.endswith("_C") or self.mask_type.endswith("_BC") or self.mask_type.endswith("_AC") or self.mask_type.endswith("_B,unmasked") or self.mask_type.endswith("_A,unmasked"):
            img2txt_mask_sq = F.interpolate(self.masks[-1].unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        if self.mask_type.endswith("_B") or self.mask_type.endswith("_AB") or self.mask_type.endswith("_BC") or self.mask_type.endswith("_B,unmasked"):
            img2txt_mask_sq = F.interpolate(self.masks[1].unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        if self.edge_width > 0:
            edge_mask = torch.zeros_like(self.masks[0])
            for mask in self.masks:
                edge_mask = fp_or(edge_mask, get_edge_mask(mask, dilation=self.edge_width))
                
            img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        elif self.edge_width_list is not None:
            edge_mask = torch.zeros_like(self.masks[0])
            
            for mask, edge_width in zip(self.masks, self.edge_width_list):
                if edge_width != 0:
                    edge_mask_new = get_edge_mask(mask, dilation=abs(edge_width))
                    edge_mask     = fp_or(edge_mask, fp_and(edge_mask_new, mask)) #fp_and here is to ensure edge_mask only grows into the region for current mask
                    
                    img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                    attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        if self.use_self_attn_mask_list is not None:
            for mask, use_self_attn_mask in zip(self.masks, self.use_self_attn_mask_list):
                if not use_self_attn_mask:
                    img2txt_mask_sq = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                    attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
        
        
        
        self.attn_mask = CoreAttnMask(attn_mask, mask_type=mask_type)









class FullAttentionMaskHiDream(BaseAttentionMask):
    def generate(self, mask_type=None, dtype=None):
        mask_type = self.mask_type if mask_type is None else mask_type
        dtype     = self.dtype     if dtype     is None else dtype
        text_off  = self.text_off
        text_len  = self.text_len
        img_len   = self.img_len
        t         = self.t
        h         = self.h
        w         = self.w
        
        if self.edge_width_list is None:
            self.edge_width_list = [self.edge_width] * self.num_regions
        
        attn_mask = torch.zeros((text_off+t*img_len, text_len+t*img_len), dtype=dtype)
        reg_num  = 0
        prev_len = 0
        for context_len, mask in zip(self.context_lens, self.masks):

            img2txt_mask_sq = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)

            curr_len = prev_len + context_len
            
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], fp_and(img2txt_mask_sq.repeat(t,t), img2txt_mask_sq.transpose(-1,-2).repeat(t,t))) # img2txt_mask_sq, txt2img_mask_sq
            
            prev_len = curr_len
            reg_num += 1
        
        self.self_attn_mask = attn_mask[text_off:, text_len:].clone()
        
        if self.mask_type.endswith("_masked") or self.mask_type.endswith("_A") or self.mask_type.endswith("_AB") or self.mask_type.endswith("_AC") or self.mask_type.endswith("_A,unmasked"):
            img2txt_mask_sq = F.interpolate(self.masks[0].unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
        
        if self.mask_type.endswith("_unmasked") or self.mask_type.endswith("_C") or self.mask_type.endswith("_BC") or self.mask_type.endswith("_AC") or self.mask_type.endswith("_B,unmasked") or self.mask_type.endswith("_A,unmasked"):
            img2txt_mask_sq = F.interpolate(self.masks[-1].unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        if self.mask_type.endswith("_B") or self.mask_type.endswith("_AB") or self.mask_type.endswith("_BC") or self.mask_type.endswith("_B,unmasked"):
            img2txt_mask_sq = F.interpolate(self.masks[1].unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
        
        if   self.edge_width > 0:
            edge_mask = torch.zeros_like(self.masks[0])
            for mask in self.masks:
                edge_mask_new = get_edge_mask(mask, dilation=abs(self.edge_width))
                edge_mask = fp_or(edge_mask, edge_mask_new)
                #edge_mask = fp_or(edge_mask, get_edge_mask(mask, dilation=self.edge_width))
                
            img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        elif self.edge_width < 0: # edge masks using cross-attn too
            edge_mask = torch.zeros_like(self.masks[0])
            for mask in self.masks:
                edge_mask = fp_or(edge_mask, get_edge_mask(mask, dilation=abs(self.edge_width)))
                
            img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
            attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
        
        elif self.edge_width_list is not None:
            edge_mask = torch.zeros_like(self.masks[0])
            
            for mask, edge_width in zip(self.masks, self.edge_width_list):
                if edge_width != 0:
                    edge_mask_new = get_edge_mask(mask, dilation=abs(edge_width))
                    edge_mask     = fp_or(edge_mask, fp_and(edge_mask_new, mask)) #fp_and here is to ensure edge_mask only grows into the region for current mask
                    
                    img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                    attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)
            
        if self.use_self_attn_mask_list is not None:
            for mask, use_self_attn_mask in zip(self.masks, self.use_self_attn_mask_list):
                if not use_self_attn_mask:
                    img2txt_mask_sq = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                    attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)

        text_len_t5     = sum(sublist[0] for sublist in self.context_lens_list)
        img2txt_mask_t5 = torch.empty((img_len, text_len_t5)).to(attn_mask)
        offset_t5_start = 0
        reg_num_slice   = 0
        for context_len, mask_slice, edge_width in zip(self.context_lens, self.masks, self.edge_width_list):
            if self.edge_width < 0: # edge masks using cross-attn too
                mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(self.edge_width)))
            if edge_width < 0: # edge masks using cross-attn too
                mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(edge_width)))
            
            slice_len     = self.context_lens_list[reg_num_slice][0]
            offset_t5_end = offset_t5_start + slice_len
            
            img2txt_mask_slice = F.interpolate(mask_slice.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, slice_len)
            
            img2txt_mask_t5[:, offset_t5_start:offset_t5_end] = img2txt_mask_slice
            
            offset_t5_start = offset_t5_end
            reg_num_slice += 1
        
        text_len_llama     = sum(sublist[1] for sublist in self.context_lens_list)
        img2txt_mask_llama = torch.empty((img_len, text_len_llama)).to(attn_mask)
        offset_llama_start = 0
        reg_num_slice      = 0
        for context_len, mask_slice, edge_width in zip(self.context_lens, self.masks, self.edge_width_list):
            if self.edge_width < 0: # edge masks using cross-attn too
                mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(self.edge_width)))
            if edge_width < 0: # edge masks using cross-attn too
                mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(edge_width)))
                
            slice_len        = self.context_lens_list[reg_num_slice][1]
            offset_llama_end = offset_llama_start + slice_len
            
            img2txt_mask_slice = F.interpolate(mask_slice.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, slice_len)
            
            img2txt_mask_llama[:, offset_llama_start:offset_llama_end] = img2txt_mask_slice
            
            offset_llama_start = offset_llama_end
            reg_num_slice += 1
        
        img2txt_mask = torch.cat([img2txt_mask_t5, img2txt_mask_llama.repeat(1,2)], dim=-1)
        
        attn_mask[:-text_off , :-text_len ] = attn_mask[text_off:, text_len:].clone()
        attn_mask[:-text_off ,  -text_len:] = img2txt_mask
        attn_mask[ -text_off:, :-text_len ] = img2txt_mask.transpose(-2,-1)

        attn_mask[img_len:,img_len:] = 1.0   # txt -> txt "self-cross" attn is critical with hidream in most cases. checkerboard strategies are generally poo
        
        # mask cross attention between text embeds
        flat = [v for group in zip(*self.context_lens_list) for v in group]
        checkvar = checkerboard_variable(flat)
        attn_mask[img_len:, img_len:] = checkvar
        
        self.attn_mask = CoreAttnMask(attn_mask, mask_type=mask_type)


        #flat = [v for group in zip(*self.context_lens_list) for v in group]

    def gen_edge_mask(self, block_idx):
        mask_type = self.mask_type
        dtype     = self.dtype     
        text_off  = self.text_off
        text_len  = self.text_len
        img_len   = self.img_len
        t         = self.t
        h         = self.h
        w         = self.w
        
        if self.edge_width_list is None:
            return self.attn_mask.mask
        else:
            #attn_mask = self.attn_mask.mask.clone()
            attn_mask = torch.zeros_like(self.attn_mask.mask)
            attn_mask[text_off:, text_len:] = self.self_attn_mask.clone()
            edge_mask = torch.zeros_like(self.masks[0])
            
            for mask, edge_width in zip(self.masks, self.edge_width_list):
                #edge_width *= (block_idx/48)
                edge_width *= torch.rand(1).item()
                edge_width = int(edge_width)
                if edge_width != 0:
                    #edge_width *= (block_idx/48)
                    #edge_width = int(edge_width)
                    edge_mask_new = get_edge_mask(mask, dilation=abs(edge_width))
                    edge_mask     = fp_or(edge_mask, fp_and(edge_mask_new, mask)) #fp_and here is to ensure edge_mask only grows into the region for current mask
                    
                    img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                    
                    attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)


            if self.use_self_attn_mask_list is not None:
                for mask, use_self_attn_mask in zip(self.masks, self.use_self_attn_mask_list):
                    if not use_self_attn_mask:
                        img2txt_mask_sq = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, img_len)
                        attn_mask[text_off:, text_len:] = fp_or(attn_mask[text_off:, text_len:], img2txt_mask_sq)

            text_len_t5     = sum(sublist[0] for sublist in self.context_lens_list)
            img2txt_mask_t5 = torch.empty((img_len, text_len_t5)).to(attn_mask)
            offset_t5_start = 0
            reg_num_slice   = 0
            for context_len, mask_slice, edge_width in zip(self.context_lens, self.masks, self.edge_width_list):
                if self.edge_width < 0: # edge masks using cross-attn too
                    mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(self.edge_width)))
                if edge_width < 0: # edge masks using cross-attn too
                    mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(edge_width)))
                
                slice_len     = self.context_lens_list[reg_num_slice][0]
                offset_t5_end = offset_t5_start + slice_len
                
                img2txt_mask_slice = F.interpolate(mask_slice.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, slice_len)
                
                img2txt_mask_t5[:, offset_t5_start:offset_t5_end] = img2txt_mask_slice
                
                offset_t5_start = offset_t5_end
                reg_num_slice += 1
            
            text_len_llama     = sum(sublist[1] for sublist in self.context_lens_list)
            img2txt_mask_llama = torch.empty((img_len, text_len_llama)).to(attn_mask)
            offset_llama_start = 0
            reg_num_slice      = 0
            for context_len, mask_slice, edge_width in zip(self.context_lens, self.masks, self.edge_width_list):
                if self.edge_width < 0: # edge masks using cross-attn too
                    mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(self.edge_width)))
                if edge_width < 0: # edge masks using cross-attn too
                    mask_slice = fp_or(mask_slice, get_edge_mask(mask_slice, dilation=abs(edge_width)))
                    
                slice_len        = self.context_lens_list[reg_num_slice][1]
                offset_llama_end = offset_llama_start + slice_len
                
                img2txt_mask_slice = F.interpolate(mask_slice.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, slice_len)
                
                img2txt_mask_llama[:, offset_llama_start:offset_llama_end] = img2txt_mask_slice
                
                offset_llama_start = offset_llama_end
                reg_num_slice += 1
            
            img2txt_mask = torch.cat([img2txt_mask_t5, img2txt_mask_llama.repeat(1,2)], dim=-1)
            
            attn_mask[:-text_off , :-text_len ] = attn_mask[text_off:, text_len:].clone()
            attn_mask[:-text_off ,  -text_len:] = img2txt_mask
            attn_mask[ -text_off:, :-text_len ] = img2txt_mask.transpose(-2,-1)

            attn_mask[img_len:,img_len:] = 1.0   # txt -> txt "self-cross" attn is critical with hidream in most cases. checkerboard strategies are generally poo
            
            # mask cross attention between text embeds
            flat = [v for group in zip(*self.context_lens_list) for v in group]
            checkvar = checkerboard_variable(flat)
            attn_mask[img_len:, img_len:] = checkvar
            
            return attn_mask.to('cuda')
        
        
class RegionalContext:
    def __init__(self, idle_device='cpu', work_device='cuda'):
        self.context  = None
        self.clip_fea = None
        self.llama3   = None
        self.llama3_list = []
        self.t5_list     = []
        self.pooled_output = None
        self.idle_device = idle_device
        self.work_device = work_device
    
    def add_region(self, context, clip_fea=None):
        if self.context is not None:
            self.context = torch.cat([self.context, context], dim=1)
        else:
            self.context = context
            
        if clip_fea is not None:
            if self.clip_fea is not None:
                self.clip_fea = torch.cat([self.clip_fea, clip_fea], dim=1)
            else:
                self.clip_fea = clip_fea

    def add_region_clip_fea(self, clip_fea):
        if self.clip_fea is not None:
            self.clip_fea = torch.cat([self.clip_fea, clip_fea], dim=1)
        else:
            self.clip_fea = clip_fea

    def add_region_llama3(self, llama3):
        if self.llama3 is not None:
            self.llama3 = torch.cat([self.llama3, llama3], dim=-2)   # base shape 1,32,128,4096
        else:
            self.llama3 = llama3
            
    def add_region_hidream(self, t5, llama3):
        self.t5_list    .append(t5)
        self.llama3_list.append(llama3)

    def clear_regions(self):
        if self.context is not None:
            del self.context
            self.context = None
        if self.clip_fea is not None:
            del self.clip_fea
            self.clip_fea = None
        if self.llama3 is not None:
            del self.llama3
            self.llama3 = None
            
        del self.t5_list
        del self.llama3_list
        self.t5_list     = []
        self.llama3_list = []

    def get(self):
        return self.context.to(self.work_device)

    def get_clip_fea(self):
        if self.clip_fea is not None:
            return self.clip_fea.to(self.work_device)
        else:
            return None

    def get_llama3(self):
        if self.llama3 is not None:
            return self.llama3.to(self.work_device)
        else:
            return None


class CrossAttentionMask(BaseAttentionMask):
    def generate(self, mask_type=None, dtype=None):
        mask_type = self.mask_type if mask_type is None else mask_type
        dtype     = self.dtype     if dtype     is None else dtype
        text_off  = self.text_off
        text_len  = self.text_len
        img_len   = self.img_len
        t         = self.t
        h         = self.h
        w         = self.w
        
        cross_attn_mask = torch.zeros((t * img_len,    text_len), dtype=dtype)
    
        prev_len = 0
        for context_len, mask in zip(self.context_lens, self.masks):

            cross_mask, self_mask = None, None
            if mask.ndim == 6:
                mask.squeeze_(0)
            if mask.ndim == 3:
                t_mask = mask.shape[0]
            elif mask.ndim == 4:
                if mask.shape[0] > 1:

                    cross_mask = mask[0]
                    if cross_mask.shape[-3] > self.t:
                        cross_mask = cross_mask[:self.t,...]
                    elif cross_mask.shape[-3] < self.t:
                        cross_mask = F.pad(cross_mask.permute(1,2,0), [0,self.t-cross_mask.shape[-3]], value=0).permute(2,0,1)

                    t_mask = self.t
                else:
                    t_mask = mask.shape[-3]
                    mask.squeeze_(0)
            elif mask.ndim == 5:
                t_mask = mask.shape[-3]
            else:
                t_mask = 1
                mask.unsqueeze_(0)
                
            if cross_mask is not None:
                img2txt_mask    = F.interpolate(cross_mask.unsqueeze(0).unsqueeze(0).to(torch.float16), (t_mask, h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1)
            else:
                img2txt_mask    = F.interpolate(      mask.unsqueeze(0).unsqueeze(0).to(torch.float16), (t_mask, h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1)
            
            if t_mask == 1: # ...why only if == 1?
                img2txt_mask = img2txt_mask.repeat(1, context_len)   

            curr_len = prev_len + context_len
            
            if t_mask == 1:
                cross_attn_mask[:, prev_len:curr_len] = img2txt_mask.repeat(t,1)
            else:
                cross_attn_mask[:, prev_len:curr_len] = img2txt_mask
            
            prev_len = curr_len
                            
        self.attn_mask = CoreAttnMask(cross_attn_mask, mask_type=mask_type)




class SplitAttentionMask(BaseAttentionMask):
    def generate(self, mask_type=None, dtype=None):
        mask_type = self.mask_type if mask_type is None else mask_type
        dtype     = self.dtype     if dtype     is None else dtype
        text_off  = self.text_off
        text_len  = self.text_len
        img_len   = self.img_len
        t         = self.t
        h         = self.h
        w         = self.w
        
        if self.edge_width_list is None:
            self.edge_width_list = [self.edge_width] * self.num_regions
        
        cross_attn_mask = torch.zeros((t * img_len,    text_len), dtype=dtype)
        self_attn_mask  = torch.zeros((t * img_len, t * img_len), dtype=dtype)
    
        prev_len = 0
        self_masks = []
        for context_len, mask in zip(self.context_lens, self.masks):

            cross_mask, self_mask = None, None
            if mask.ndim == 6:
                mask.squeeze_(0)
            if mask.ndim == 3:
                t_mask = mask.shape[0]
            elif mask.ndim == 4:

                if mask.shape[0] > 1:
                    cross_mask = mask[0]
                    if cross_mask.shape[-3] > self.t:
                        cross_mask = cross_mask[:self.t,...]
                    elif cross_mask.shape[-3] < self.t:
                        cross_mask = F.pad(cross_mask.permute(1,2,0), [0,self.t-cross_mask.shape[-3]], value=0).permute(2,0,1)

                    self_mask = mask[1]
                    if self_mask.shape[-3] > self.t:
                        self_mask = self_mask[:self.t,...]
                    elif self_mask.shape[-3] < self.t:
                        self_mask = F.pad(self_mask.permute(1,2,0), [0,self.t-self_mask.shape[-3]], value=0).permute(2,0,1)

                    t_mask = self.t
                else:
                    t_mask = mask.shape[-3]
                    mask.squeeze_(0)
            elif mask.ndim == 5:
                t_mask = mask.shape[-3]
            else:
                t_mask = 1
                mask.unsqueeze_(0)
                
            if cross_mask is not None:
                img2txt_mask    = F.interpolate(cross_mask.unsqueeze(0).unsqueeze(0).to(torch.float16), (t_mask, h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1)
            else:
                img2txt_mask    = F.interpolate(      mask.unsqueeze(0).unsqueeze(0).to(torch.float16), (t_mask, h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1)
            
            if t_mask == 1: # ...why only if == 1?
                img2txt_mask = img2txt_mask.repeat(1, context_len)   

            curr_len = prev_len + context_len
            
            if t_mask == 1:
                cross_attn_mask[:, prev_len:curr_len] = img2txt_mask.repeat(t,1)
            else:
                cross_attn_mask[:, prev_len:curr_len] = img2txt_mask
            
            if self_mask is not None:
                img2txt_mask_sq = F.interpolate(self_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, t_mask * img_len)
            else:
                img2txt_mask_sq = F.interpolate(     mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, t_mask * img_len)
            self_masks.append(img2txt_mask_sq)
            
            if t_mask > 1:
                self_attn_mask = fp_or(self_attn_mask, fp_and(img2txt_mask_sq, img2txt_mask_sq.transpose(-1,-2)))
            else:
                self_attn_mask = fp_or(self_attn_mask, fp_and(img2txt_mask_sq.repeat(t,t), img2txt_mask_sq.transpose(-1,-2)).repeat(t,t))
            
            prev_len = curr_len

        if self.mask_type.endswith("_masked") or self.mask_type.endswith("_A") or self.mask_type.endswith("_AB") or self.mask_type.endswith("_AC") or self.mask_type.endswith("_A,unmasked"):
            self_attn_mask = fp_or(self_attn_mask, self_masks[0])
        
        if self.mask_type.endswith("_unmasked") or self.mask_type.endswith("_C") or self.mask_type.endswith("_BC") or self.mask_type.endswith("_AC") or self.mask_type.endswith("_B,unmasked") or self.mask_type.endswith("_A,unmasked"):
            self_attn_mask = fp_or(self_attn_mask, self_masks[-1])
            
        if self.mask_type.endswith("_B") or self.mask_type.endswith("_AB") or self.mask_type.endswith("_BC") or self.mask_type.endswith("_B,unmasked"):
            self_attn_mask = fp_or(self_attn_mask, self_masks[1])
            
        if   self.edge_width > 0:
            edge_mask = torch.zeros_like(self.masks[0])
            for mask in self.masks:
                edge_mask_new = get_edge_mask(mask, dilation=abs(self.edge_width))
                edge_mask = fp_or(edge_mask, edge_mask_new)
                #edge_mask = fp_or(edge_mask, get_edge_mask(mask, dilation=self.edge_width))
            
            img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, t_mask * img_len)
            self_attn_mask = fp_or(self_attn_mask, img2txt_mask_sq)
            
        elif self.edge_width_list is not None:
            edge_mask = torch.zeros_like(self.masks[0])
            
            for mask, edge_width in zip(self.masks, self.edge_width_list):
                if edge_width != 0:
                    edge_mask_new = get_edge_mask(mask, dilation=abs(edge_width))
                    edge_mask     = fp_or(edge_mask, fp_and(edge_mask_new, mask)) #fp_and here is to ensure edge_mask only grows into the region for current mask
                    
                    img2txt_mask_sq = F.interpolate(edge_mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, t_mask * img_len)
                    self_attn_mask = fp_or(self_attn_mask, img2txt_mask_sq)
            
        if self.use_self_attn_mask_list is not None:
            for mask, use_self_attn_mask in zip(self.masks, self.use_self_attn_mask_list):
                if not use_self_attn_mask:
                    img2txt_mask_sq = F.interpolate(mask.unsqueeze(0).to(torch.float16), (h, w), mode='nearest-exact').to(dtype).flatten().unsqueeze(1).repeat(1, t_mask * img_len)
                    self_attn_mask = fp_or(self_attn_mask, img2txt_mask_sq)
        
        
        attn_mask = torch.cat([cross_attn_mask, self_attn_mask], dim=1)
        
        self.attn_mask = CoreAttnMask(attn_mask, mask_type=mask_type)

