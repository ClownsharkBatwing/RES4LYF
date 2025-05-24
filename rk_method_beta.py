import torch
from torch import Tensor
from typing import Optional, Callable, Tuple, List, Dict, Any, Union

import comfy.model_patcher
import comfy.supported_models

import itertools 

from .phi_functions        import Phi
from .rk_coefficients_beta import get_implicit_sampler_name_list, get_rk_methods_beta
from ..helper              import ExtraOptions
from ..latents             import get_orthogonal, get_collinear, get_cosine_similarity, tile_latent, untile_latent

from ..res4lyf             import RESplain

MAX_STEPS = 10000


def get_data_from_step   (x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x:Tensor, x_next:Tensor, sigma:Tensor, sigma_next:Tensor) -> Tensor:
    h = sigma_next - sigma
    return (x - x_next) / h



class RK_Method_Beta:
    def __init__(self,
                model,
                rk_type               : str,
                noise_anchor          : float,
                noise_boost_normalize : bool        = True,
                model_device          : str         = 'cuda',
                work_device           : str         = 'cpu',
                dtype                 : torch.dtype = torch.float64,
                extra_options         : str         = ""
                ):
        
        self.work_device                 = work_device
        self.model_device                = model_device
        self.dtype                       : torch.dtype = dtype

        self.model                       = model

        if hasattr(model, "model"):
            model_sampling = model.model.model_sampling
        elif hasattr(model, "inner_model"):
            model_sampling = model.inner_model.inner_model.model_sampling
        
        self.sigma_min                   : Tensor                   = model_sampling.sigma_min.to(dtype=dtype, device=work_device)
        self.sigma_max                   : Tensor                   = model_sampling.sigma_max.to(dtype=dtype, device=work_device)

        self.rk_type                     : str                      = rk_type

        self.IMPLICIT                    : str                      = rk_type in get_implicit_sampler_name_list(nameOnly=True)
        self.EXPONENTIAL                 : bool                     = RK_Method_Beta.is_exponential(rk_type)

        self.SYNC_SUBSTEP_MEAN_CW        : bool                     = noise_boost_normalize

        self.A                           : Optional[Tensor]         = None
        self.B                           : Optional[Tensor]         = None
        self.U                           : Optional[Tensor]         = None
        self.V                           : Optional[Tensor]         = None

        self.rows                        : int                      = 0
        self.cols                        : int                      = 0

        self.denoised                    : Optional[Tensor]         = None
        self.uncond                      : Optional[Tensor]         = None

        self.y0                          : Optional[Tensor]         = None
        self.y0_inv                      : Optional[Tensor]         = None

        self.multistep_stages            : int                      = 0
        self.row_offset                  : Optional[int]            = None

        self.cfg_cw                      : float                    = 1.0
        self.extra_args                  : Optional[Dict[str, Any]] = None

        self.extra_options               : str                      = extra_options
        self.EO                          : ExtraOptions             = ExtraOptions(extra_options)

        self.reorder_tableau_indices     : list[int]                = self.EO("reorder_tableau_indices", [-1])

        self.LINEAR_ANCHOR_X_0           : float                    = noise_anchor
        
        self.tile_sizes                  : Optional[List[Tuple[int,int]]] = None
        self.tile_cnt                    : int                      = 0
        self.latent_compression_ratio    : int                      = 8

    @staticmethod
    def is_exponential(rk_type:str) -> bool:
        if rk_type.startswith(( "res", 
                                "dpmpp", 
                                "ddim", 
                                "pec", 
                                "etdrk", 
                                "lawson", 
                                "abnorsett",
                                )): 
            return True
        else:
            return False

    @staticmethod
    def create(model,
            rk_type       : str,
            noise_anchor  : float       = 1.0,
            noise_boost_normalize  : bool = True,
            model_device  : str         = 'cuda',
            work_device   : str         = 'cpu',
            dtype         : torch.dtype = torch.float64,
            extra_options : str         = ""
            ) -> "Union[RK_Method_Exponential, RK_Method_Linear]":
        
        if RK_Method_Beta.is_exponential(rk_type):
            return RK_Method_Exponential(model, rk_type, noise_anchor, noise_boost_normalize, model_device, work_device, dtype, extra_options)
        else:
            return RK_Method_Linear     (model, rk_type, noise_anchor, noise_boost_normalize, model_device, work_device, dtype, extra_options)
                
    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")
    
    def model_epsilon(self, x:Tensor, sigma:Tensor, **extra_args) -> Tuple[Tensor, Tensor]:
        s_in     = x.new_ones([x.shape[0]])
        denoised = self.model(x, sigma * s_in, **extra_args)
        denoised = self.calc_cfg_channelwise(denoised)
        eps      = (x - denoised) / (sigma * s_in).view(x.shape[0], 1, 1, 1)       #return x0 ###################################THIS WORKS ONLY WITH THE MODEL SAMPLING PATCH
        return eps, denoised
    
    def model_denoised(self, x:Tensor, sigma:Tensor, **extra_args) -> Tensor:
        s_in     = x.new_ones([x.shape[0]])
        control_tiles = None
        y0_style_pos = self.extra_args['model_options']['transformer_options'].get("y0_style_pos")
        y0_style_neg = self.extra_args['model_options']['transformer_options'].get("y0_style_neg")
        y0_style_pos_tile, sy0_style_neg_tiles = None, None
        
        if self.EO("tile_model_calls"):
            tile_h = self.EO("tile_h", 128)
            tile_w = self.EO("tile_w", 128)
            
            denoised_tiles = []
            
            tiles, orig_shape, grid, strides = tile_latent(x, tile_size=(tile_h,tile_w))
            
            for i in range(tiles.shape[0]):
                tile = tiles[i].unsqueeze(0)
                
                denoised_tile = self.model(tile, sigma * s_in, **extra_args)
                
                denoised_tiles.append(denoised_tile)
                
            denoised_tiles = torch.cat(denoised_tiles, dim=0)
            
            denoised = untile_latent(denoised_tiles, orig_shape, grid, strides)
            
        elif self.tile_sizes is not None:
            tile_h_full = self.tile_sizes[self.tile_cnt % len(self.tile_sizes)][0]
            tile_w_full = self.tile_sizes[self.tile_cnt % len(self.tile_sizes)][1] 
            
            if tile_h_full == -1:
                tile_h      = x.shape[-2]
                tile_h_full = tile_h * self.latent_compression_ratio
            else:
                tile_h = tile_h_full // self.latent_compression_ratio
                
            if tile_w_full == -1:
                tile_w      = x.shape[-1]
                tile_w_full = tile_w * self.latent_compression_ratio
            else:
                tile_w = tile_w_full // self.latent_compression_ratio
            
            #tile_h = tile_h_full // self.latent_compression_ratio
            #tile_w = tile_w_full // self.latent_compression_ratio
            
            self.tile_cnt += 1
            
            #if len(self.tile_sizes) == 1 and self.tile_cnt % 2 == 1:
            #    tile_h, tile_w = tile_w, tile_h
            #    tile_h_full, tile_w_full = tile_w_full, tile_h_full
            
            if (self.tile_cnt // len(self.tile_sizes)) % 2 == 1 and self.EO("tiles_autorotate"):
                tile_h, tile_w = tile_w, tile_h
                tile_h_full, tile_w_full = tile_w_full, tile_h_full
            
            xt_negative = self.model.inner_model.conds.get('xt_negative', self.model.inner_model.conds.get('negative'))
            negative_control = xt_negative[0].get('control')
            
            if negative_control is not None and hasattr(negative_control, 'cond_hint_original'):
                negative_cond_hint_init = negative_control.cond_hint.clone() if negative_control.cond_hint is not None else None
            
            xt_positive = self.model.inner_model.conds.get('xt_positive', self.model.inner_model.conds.get('positive'))
            positive_control = xt_positive[0].get('control')
            
            if positive_control is not None and hasattr(positive_control, 'cond_hint_original'):
                positive_cond_hint_init = positive_control.cond_hint.clone() if positive_control.cond_hint is not None else None
                if positive_control.cond_hint_original.shape[-1] != x.shape[-2] * self.latent_compression_ratio or positive_control.cond_hint_original.shape[-2] != x.shape[-1] * self.latent_compression_ratio:
                    positive_control_pretile = comfy.utils.bislerp(positive_control.cond_hint_original.clone().to(torch.float16).to('cuda'), x.shape[-1] * self.latent_compression_ratio, x.shape[-2] * self.latent_compression_ratio)
                    positive_control.cond_hint_original = positive_control_pretile.to(positive_control.cond_hint_original)
                positive_control_pretile = positive_control.cond_hint_original.clone().to(torch.float16).to('cuda')
                control_tiles, control_orig_shape, control_grid, control_strides = tile_latent(positive_control_pretile, tile_size=(tile_h_full,tile_w_full))
                control_tiles = control_tiles
            
            denoised_tiles = []
            
            tiles, orig_shape, grid, strides = tile_latent(x, tile_size=(tile_h,tile_w))
            
            if y0_style_pos is not None:
                y0_style_pos_tiles, _, _, _ = tile_latent(y0_style_pos, tile_size=(tile_h,tile_w))
            if y0_style_neg is not None:
                y0_style_neg_tiles, _, _, _ = tile_latent(y0_style_neg, tile_size=(tile_h,tile_w))
            
            for i in range(tiles.shape[0]):
                tile = tiles[i].unsqueeze(0)
                
                if control_tiles is not None:
                    positive_control.cond_hint = control_tiles[i].unsqueeze(0).to(positive_control.cond_hint)
                    if negative_control is not None:
                        negative_control.cond_hint = control_tiles[i].unsqueeze(0).to(positive_control.cond_hint)
                
                if y0_style_pos is not None:
                    self.extra_args['model_options']['transformer_options']['y0_style_pos'] = y0_style_pos_tiles[i].unsqueeze(0)
                if y0_style_neg is not None:
                    self.extra_args['model_options']['transformer_options']['y0_style_neg'] = y0_style_neg_tiles[i].unsqueeze(0)
                
                denoised_tile = self.model(tile, sigma * s_in, **extra_args)
                
                denoised_tiles.append(denoised_tile)
                
            denoised_tiles = torch.cat(denoised_tiles, dim=0)
            
            denoised = untile_latent(denoised_tiles, orig_shape, grid, strides)
            
        else:
            denoised = self.model(x, sigma * s_in, **extra_args)
        
        if control_tiles is not None:
            positive_control.cond_hint = positive_cond_hint_init
            if negative_control is not None:
                negative_control.cond_hint = negative_cond_hint_init
                
        if y0_style_pos is not None:
            self.extra_args['model_options']['transformer_options']['y0_style_pos'] = y0_style_pos
        if y0_style_neg is not None:
            self.extra_args['model_options']['transformer_options']['y0_style_neg'] = y0_style_neg
        
        denoised = self.calc_cfg_channelwise(denoised)
        return denoised

    def update_transformer_options(self,
                transformer_options : Optional[dict] = None,
                ):

        self.extra_args.setdefault("model_options", {}).setdefault("transformer_options", {}).update(transformer_options)
        return

    def set_coeff(self,
                rk_type    : str,
                h          : Tensor,
                c1         : float  = 0.0,
                c2         : float  = 0.5,
                c3         : float  = 1.0,
                step       : int    = 0,
                sigmas     : Optional[Tensor] = None,
                sigma_down : Optional[Tensor] = None,
                ) -> None:

        self.rk_type     = rk_type
        self.IMPLICIT    = rk_type in get_implicit_sampler_name_list(nameOnly=True)
        self.EXPONENTIAL = RK_Method_Beta.is_exponential(rk_type) 

        sigma            = sigmas[step]
        sigma_next       = sigmas[step+1]
        
        h_prev = []
        a, b, u, v, ci, multistep_stages, hybrid_stages, FSAL = get_rk_methods_beta(rk_type,
                                                                                    h,
                                                                                    c1,
                                                                                    c2,
                                                                                    c3,
                                                                                    h_prev,
                                                                                    step,
                                                                                    sigmas,
                                                                                    sigma,
                                                                                    sigma_next,
                                                                                    sigma_down,
                                                                                    self.extra_options,
                                                                                    )
        
        self.multistep_stages = multistep_stages
        self.hybrid_stages    = hybrid_stages
        
        self.A = torch.tensor(a,  dtype=h.dtype, device=h.device)
        self.B = torch.tensor(b,  dtype=h.dtype, device=h.device)
        self.C = torch.tensor(ci, dtype=h.dtype, device=h.device)

        self.U = torch.tensor(u,  dtype=h.dtype, device=h.device) if u is not None else None
        self.V = torch.tensor(v,  dtype=h.dtype, device=h.device) if v is not None else None
        
        self.rows = self.A.shape[0]
        self.cols = self.A.shape[1]
        
        self.row_offset = 1 if not self.IMPLICIT and self.A[0].sum() == 0 else 0  
        
        if self.IMPLICIT and self.reorder_tableau_indices[0] != -1:
            self.reorder_tableau(self.reorder_tableau_indices)



    def reorder_tableau(self, indices:list[int]) -> None:
        #if indices[0]:
        self.A    = self.A   [indices]
        self.B[0] = self.B[0][indices]
        self.C    = self.C   [indices]
        self.C = torch.cat((self.C, self.C[-1:])) 
        return



    def update_substep(self,
                        x_0        : Tensor,
                        x_         : Tensor,
                        eps_       : Tensor,
                        eps_prev_  : Tensor,
                        row        : int,
                        row_offset : int,
                        h_new      : Tensor,
                        h_new_orig : Tensor,
                        lying_eps_row_factor : float = 1.0,
                        ) -> Tensor:
        
        if row < self.rows - row_offset   and   self.multistep_stages == 0:
            row_tmp_offset = row + row_offset

        else:
            row_tmp_offset = row + 1
                
        zr_base   = self.zum(row+row_offset+self.multistep_stages, eps_, eps_prev_)
        
        if self.SYNC_SUBSTEP_MEAN_CW and lying_eps_row_factor != 1.0:
            zr_orig = self.zum(row+row_offset+self.multistep_stages, eps_, eps_prev_)
            x_orig_row = x_0 + h_new * zr_orig
        
        #eps_row      = eps_     [row].clone()
        #eps_prev_row = eps_prev_[row].clone()
        
        eps_     [row] *= lying_eps_row_factor
        eps_prev_[row] *= lying_eps_row_factor
        zr = self.zum(row+row_offset+self.multistep_stages, eps_, eps_prev_)
        
        x_[row_tmp_offset] = x_0 + h_new * zr
        
        if self.SYNC_SUBSTEP_MEAN_CW and lying_eps_row_factor != 1.0:
            x_[row_tmp_offset] = x_[row_tmp_offset] - x_[row_tmp_offset].mean(dim=(-2,-1), keepdim=True) + x_orig_row.mean(dim=(-2,-1), keepdim=True)
        
        #eps_     [row] = eps_row
        #eps_prev_[row] = eps_prev_row
        
        if (self.SYNC_SUBSTEP_MEAN_CW and h_new != h_new_orig) or self.EO("sync_mean_noise"):
            if not self.EO("disable_sync_mean_noise"):
                x_row_down = x_0 + h_new_orig * zr
                x_[row_tmp_offset] = x_[row_tmp_offset] - x_[row_tmp_offset].mean(dim=(-2,-1), keepdim=True) + x_row_down.mean(dim=(-2,-1), keepdim=True)
        
        return x_


    
    def a_k_einsum(self, row:int, k     :Tensor) -> Tensor:
        return torch.einsum('i, i... -> ...', self.A[row], k[:self.cols])
    
    def b_k_einsum(self, row:int, k     :Tensor) -> Tensor:
        return torch.einsum('i, i... -> ...', self.B[row], k[:self.cols])
    
    def u_k_einsum(self, row:int, k_prev:Tensor) -> Tensor:
        return torch.einsum('i, i... -> ...', self.U[row], k_prev[:self.cols]) if (self.U is not None and k_prev is not None) else 0
    
    def v_k_einsum(self, row:int, k_prev:Tensor) -> Tensor:
        return torch.einsum('i, i... -> ...', self.V[row], k_prev[:self.cols]) if (self.V is not None and k_prev is not None) else 0
    
    
    
    def zum(self, row:int, k:Tensor, k_prev:Tensor=None,) -> Tensor:
        if row < self.rows:
            return self.a_k_einsum(row, k) + self.u_k_einsum(row, k_prev)
        else:
            row = row - self.rows
            return self.b_k_einsum(row, k) + self.v_k_einsum(row, k_prev)
        
    def zum_tableau(self,  k:Tensor, k_prev:Tensor=None,) -> Tensor:
        a_k_sum = torch.einsum('ij, j... -> i...', self.A, k[:self.cols])
        u_k_sum = torch.einsum('ij, j... -> i...', self.U, k_prev[:self.cols]) if (self.U is not None and k_prev is not None) else 0
        return a_k_sum + u_k_sum
        


    def init_cfg_channelwise(self, x:Tensor, cfg_cw:float=1.0, **extra_args) -> Dict[str, Any]:
        self.uncond = [torch.full_like(x, 0.0)]
        self.cfg_cw = cfg_cw
        if cfg_cw != 1.0:
            def post_cfg_function(args):
                self.uncond[0] = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
        return extra_args
            
            
    def calc_cfg_channelwise(self, denoised:Tensor) -> Tensor:
        if self.cfg_cw != 1.0:            
            avg = 0
            for b, c in itertools.product(range(denoised.shape[0]), range(denoised.shape[1])):
                avg     += torch.norm(denoised[b][c] - self.uncond[0][b][c])
            avg  /= denoised.shape[1]
            
            for b, c in itertools.product(range(denoised.shape[0]), range(denoised.shape[1])):
                ratio     = torch.nan_to_num(torch.norm(denoised[b][c] - self.uncond[0][b][c])   /   avg,     0)
                denoised_new = self.uncond[0] + ratio * self.cfg_cw * (denoised - self.uncond[0])
            return denoised_new
        else:
            return denoised
        

    @staticmethod
    def calculate_res_2m_step(
                            x_0        : Tensor,
                            denoised_  : Tensor,
                            sigma_down : Tensor,
                            sigmas     : Tensor,
                            step       : int,
                            ) -> Tuple[Tensor, Tensor]:
        
        if denoised_[2].sum() == 0:
            return None, None
        
        sigma      = sigmas[step]
        sigma_prev = sigmas[step-1]
        
        h_prev = -torch.log(sigma/sigma_prev)
        h      = -torch.log(sigma_down/sigma)

        c1 = 0
        c2 = (-h_prev / h).item()

        ci = [c1,c2]
        φ = Phi(h, ci, analytic_solution=True)

        b2 = φ(2)/c2
        b1 = φ(1) - b2
        
        eps_2 = denoised_[1] - x_0
        eps_1 = denoised_[0] - x_0

        h_a_k_sum = h * (b1 * eps_1 + b2 * eps_2)
        
        x = torch.exp(-h) * x_0 + h_a_k_sum
        
        denoised = x_0 + (sigma / (sigma - sigma_down)) * h_a_k_sum

        return x, denoised


    @staticmethod
    def calculate_res_3m_step(
                            x_0        : Tensor,
                            denoised_  : Tensor,
                            sigma_down : Tensor,
                            sigmas     : Tensor,
                            step       : int,
                            ) -> Tuple[Tensor, Tensor]:
        
        if denoised_[3].sum() == 0:
            return None, None
        
        sigma       = sigmas[step]
        sigma_prev  = sigmas[step-1]
        sigma_prev2 = sigmas[step-2]

        h       = -torch.log(sigma_down/sigma)
        h_prev  = -torch.log(sigma/sigma_prev)
        h_prev2 = -torch.log(sigma/sigma_prev2)

        c1 = 0
        c2 = (-h_prev  / h).item()
        c3 = (-h_prev2 / h).item()

        ci = [c1,c2,c3]
        φ = Phi(h, ci, analytic_solution=True)
        
        gamma = (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))

        b3 = (1 / (gamma * c2 + c3)) * φ(2, -h)      
        b2 = gamma * b3 
        b1 = φ(1, -h) - b2 - b3    
        
        eps_3 = denoised_[2] - x_0
        eps_2 = denoised_[1] - x_0
        eps_1 = denoised_[0] - x_0

        h_a_k_sum = h * (b1 * eps_1 + b2 * eps_2 + b3 * eps_3)
        
        x = torch.exp(-h) * x_0 + h_a_k_sum
        
        denoised = x_0 + (sigma / (sigma - sigma_down)) * h_a_k_sum

        return x, denoised

    def swap_rk_type_at_step_or_threshold(self,
                                            x_0               : Tensor,
                                            data_prev_        : Tensor,
                                            NS,
                                            sigmas            : Tensor,
                                            step              : Tensor,
                                            rk_swap_step      : int,
                                            rk_swap_threshold : float,
                                            rk_swap_type      : str,
                                            rk_swap_print     : bool,
                                            ) -> str:
        if rk_swap_type == "":
            if self.EXPONENTIAL:
                rk_swap_type = "res_3m" 
            else:
                rk_swap_type = "deis_3m"
            
        if step > rk_swap_step and self.rk_type != rk_swap_type:
            RESplain("Switching rk_type to:", rk_swap_type)
            self.rk_type = rk_swap_type
            
            if RK_Method_Beta.is_exponential(rk_swap_type):
                self.__class__ = RK_Method_Exponential
            else:
                self.__class__ = RK_Method_Linear
                
            if rk_swap_type in get_implicit_sampler_name_list(nameOnly=True):
                self.IMPLICIT   = True
                self.row_offset = 0
                NS.row_offset   = 0
            else:
                self.IMPLICIT   = False
                self.row_offset = 1
                NS.row_offset   = 1
            NS.h_fn     = self.h_fn
            NS.t_fn     = self.t_fn
            NS.sigma_fn = self.sigma_fn
            
            
            
        if step > 2 and sigmas[step+1] > 0 and self.rk_type != rk_swap_type and rk_swap_threshold > 0:
            x_res_2m, denoised_res_2m = self.calculate_res_2m_step(x_0, data_prev_, NS.sigma_down, sigmas, step)
            x_res_3m, denoised_res_3m = self.calculate_res_3m_step(x_0, data_prev_, NS.sigma_down, sigmas, step)
            if denoised_res_2m is not None:
                if rk_swap_print:
                    RESplain("res_3m - res_2m:", torch.norm(denoised_res_3m - denoised_res_2m).item())
                if rk_swap_threshold > torch.norm(denoised_res_2m - denoised_res_3m):
                    RESplain("Switching rk_type to:", rk_swap_type, "at step:", step)
                    self.rk_type = rk_swap_type
            
                    if RK_Method_Beta.is_exponential(rk_swap_type):
                        self.__class__ = RK_Method_Exponential
                    else:
                        self.__class__ = RK_Method_Linear
                
                    if rk_swap_type in get_implicit_sampler_name_list(nameOnly=True):
                        self.IMPLICIT   = True
                        self.row_offset = 0
                        NS.row_offset   = 0
                    else:
                        self.IMPLICIT   = False
                        self.row_offset = 1
                        NS.row_offset   = 1
                    NS.h_fn     = self.h_fn
                    NS.t_fn     = self.t_fn
                    NS.sigma_fn = self.sigma_fn
            
        return self.rk_type


    def bong_iter(self,
                    x_0       : Tensor,
                    x_        : Tensor,
                    eps_      : Tensor,
                    eps_prev_ : Tensor,
                    data_     : Tensor,
                    sigma     : Tensor,
                    s_        : Tensor,
                    row       : int,
                    row_offset: int,
                    h         : Tensor,
                    step      : int,
                    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        if x_0.ndim == 4:
            norm_dim = (-2,-1)
        elif x_0.ndim == 5:
            norm_dim = (-4,-2,-1)
        
        if self.EO("bong_start_step", 0) > step or step > self.EO("bong_stop_step", 10000):
            return x_0, x_, eps_
        
        bong_iter_max_row = self.rows - row_offset
        if self.EO("bong_iter_max_row_full"):
            bong_iter_max_row = self.rows
            
        if self.EO("bong_iter_lock_x_0_ch_means"):
            x_0_ch_means = x_0.mean(dim=norm_dim, keepdim=True)
            
        if self.EO("bong_iter_lock_x_row_ch_means"):
            x_row_means = []
            for rr in range(row+row_offset):
                x_row_mean = x_[rr].mean(dim=norm_dim, keepdim=True)
                x_row_means.append(x_row_mean)
        
        if row < bong_iter_max_row   and   self.multistep_stages == 0:
            bong_strength = self.EO("bong_strength", 1.0)
            
            if bong_strength != 1.0:
                x_0_tmp  = x_0.clone()
                x_tmp_   = x_.clone()
                eps_tmp_ = eps_.clone()
            
            for i in range(100):
                x_0 = x_[row+row_offset] - h * self.zum(row+row_offset, eps_, eps_prev_)
                
                if self.EO("bong_iter_lock_x_0_ch_means"):
                    x_0 = x_0 - x_0.mean(dim=norm_dim, keepdim=True) + x_0_ch_means
                    
                for rr in range(row+row_offset):
                    x_[rr] = x_0 + h * self.zum(rr, eps_, eps_prev_)
                    
                if self.EO("bong_iter_lock_x_row_ch_means"):
                    for rr in range(row+row_offset):
                        x_[rr] = x_[rr] - x_[rr].mean(dim=norm_dim, keepdim=True) + x_row_means[rr]
                    
                for rr in range(row+row_offset):
                    if self.EO("zonkytar"):
                        #eps_[rr] = self.get_unsample_epsilon(x_[rr], x_0, data_[rr], sigma, s_[rr])
                        eps_[rr] = self.get_epsilon(x_[rr], x_0, data_[rr], sigma, s_[rr])
                    else:
                        eps_[rr] = self.get_epsilon(x_0, x_[rr], data_[rr], sigma, s_[rr])
                    
            if bong_strength != 1.0:
                x_0  = x_0_tmp  + bong_strength * (x_0  - x_0_tmp)
                x_   = x_tmp_   + bong_strength * (x_   - x_tmp_)
                eps_ = eps_tmp_ + bong_strength * (eps_ - eps_tmp_)
        
        return x_0, x_, eps_


    def newton_iter(self,
                    x_0        : Tensor,
                    x_         : Tensor,
                    eps_       : Tensor,
                    eps_prev_  : Tensor,
                    data_      : Tensor,
                    s_         : Tensor,
                    row        : int,
                    h          : Tensor,
                    sigmas     : Tensor,
                    step       : int,
                    newton_name: str,
                    ) -> Tuple[Tensor, Tensor]:
        
        newton_iter_name = "newton_iter_" + newton_name
        
        default_anchor_x_all = False
        if newton_name == "lying":
            default_anchor_x_all = True
        
        newton_iter                 = self.EO(newton_iter_name,                      100)
        newton_iter_skip_last_steps = self.EO(newton_iter_name + "_skip_last_steps",   0)
        newton_iter_mixing_rate     = self.EO(newton_iter_name + "_mixing_rate",     1.0)
        
        newton_iter_anchor          = self.EO(newton_iter_name + "_anchor",            0)
        newton_iter_anchor_x_all    = self.EO(newton_iter_name + "_anchor_x_all",    default_anchor_x_all)
        newton_iter_type            = self.EO(newton_iter_name + "_type",           "from_epsilon")
        newton_iter_sequence        = self.EO(newton_iter_name + "_sequence",       "double")
        
        row_b_offset = 0
        if self.EO(newton_iter_name + "_include_row_b"):
            row_b_offset = 1
        
        if step >= len(sigmas)-1-newton_iter_skip_last_steps   or   sigmas[step+1] == 0   or   not self.IMPLICIT:
            return x_, eps_
        
        sigma = sigmas[step]
        
        start, stop = 0, self.rows+row_b_offset
        if newton_name   == "pre":
            start = row
        elif newton_name == "post":
            start = row + 1
            
        if newton_iter_anchor >= 0:
            eps_anchor = eps_[newton_iter_anchor].clone()
            
        if newton_iter_anchor_x_all:
            x_orig_ = x_.clone()
            
        for n_iter in range(newton_iter):
            for r in range(start, stop):
                if newton_iter_anchor >= 0:
                    eps_[newton_iter_anchor] = eps_anchor.clone()
                if newton_iter_anchor_x_all:
                    x_ = x_orig_.clone()
                x_tmp, eps_tmp = x_[r].clone(), eps_[r].clone()
                
                seq_start, seq_stop = r, r+1
                
                if newton_iter_sequence == "double":
                    seq_start, seq_stop = start, stop
                    
                for r_ in range(seq_start, seq_stop):
                    x_[r_] = x_0 + h * self.zum(r_, eps_, eps_prev_)

                for r_ in range(seq_start, seq_stop):
                    if newton_iter_type == "from_data":
                        data_[r_] = get_data_from_step(x_0, x_[r_], sigma, s_[r_])  
                        eps_ [r_] = self.get_epsilon(x_0, x_[r_], data_[r_], sigma, s_[r_])
                    elif newton_iter_type == "from_step":
                        eps_ [r_] = get_epsilon_from_step(x_0, x_[r_], sigma, s_[r_])
                    elif newton_iter_type == "from_alt":
                        eps_ [r_] = x_0/sigma - x_[r_]/s_[r_]
                    elif newton_iter_type == "from_epsilon":
                        eps_ [r_] = self.get_epsilon(x_0, x_[r_], data_[r_], sigma, s_[r_])
                    
                    if self.EO(newton_iter_name + "_opt"):
                        opt_timing, opt_type, opt_subtype = self.EO(newton_iter_name+"_opt", [str])
                        
                        opt_start, opt_stop = 0, self.rows+row_b_offset
                        if    opt_timing == "early":
                            opt_stop  = row + 1
                        elif  opt_timing == "late":
                            opt_start = row + 1

                        for r2 in range(opt_start, opt_stop): 
                            if r_ != r2:
                                if   opt_subtype == "a":
                                    eps_a = eps_[r2]
                                    eps_b = eps_[r_]
                                elif opt_subtype == "b":
                                    eps_a = eps_[r_]
                                    eps_b = eps_[r2]
                                
                                if   opt_type == "ortho":
                                    eps_ [r_] = get_orthogonal(eps_a, eps_b)
                                elif opt_type == "collin":
                                    eps_ [r_] = get_collinear (eps_a, eps_b)
                                elif opt_type == "proj":
                                    eps_ [r_] = get_collinear (eps_a, eps_b) + get_orthogonal(eps_b, eps_a)
                                    
                    x_  [r_] =   x_tmp + newton_iter_mixing_rate * (x_  [r_] -   x_tmp)
                    eps_[r_] = eps_tmp + newton_iter_mixing_rate * (eps_[r_] - eps_tmp)
                    
                if newton_iter_sequence == "double":
                    break
        
        return x_, eps_




class RK_Method_Exponential(RK_Method_Beta):
    def __init__(self,
                model,
                rk_type       : str,
                noise_anchor  : float,
                noise_boost_normalize  : bool,

                model_device  : str         = 'cuda',
                work_device   : str         = 'cpu',
                dtype         : torch.dtype = torch.float64,
                extra_options : str         = "",
                ):
        
        super().__init__(model,
                        rk_type,
                        noise_anchor,
                        noise_boost_normalize,
                        model_device  = model_device,
                        work_device   = work_device,
                        dtype         = dtype,
                        extra_options = extra_options,
                        ) 
        
    @staticmethod
    def alpha_fn(neg_h:Tensor) -> Tensor:
        return torch.exp(neg_h)

    @staticmethod
    def sigma_fn(t:Tensor) -> Tensor:
        return t.neg().exp()

    @staticmethod
    def t_fn(sigma:Tensor) -> Tensor:
        return sigma.log().neg()
    
    @staticmethod
    def h_fn(sigma_down:Tensor, sigma:Tensor) -> Tensor:
        return -torch.log(sigma_down/sigma)

    def __call__(self,
                x         : Tensor,
                sub_sigma : Tensor,
                x_0       : Optional[Tensor] = None,
                sigma     : Optional[Tensor] = None,
                transformer_options : Optional[dict] = None,
                ) -> Tuple[Tensor, Tensor]:
        
        x_0   = x         if x_0   is None else x_0
        sigma = sub_sigma if sigma is None else sigma
        
        if transformer_options is not None:
            self.extra_args.setdefault("model_options", {}).setdefault("transformer_options", {}).update(transformer_options)

        denoised = self.model_denoised(x.to(self.model_device), sub_sigma.to(self.model_device), **self.extra_args).to(sigma.device)
        
        eps_anchored = (x_0 - denoised) / sigma
        eps_unmoored = (x   - denoised) / sub_sigma
        
        eps      = eps_unmoored + self.LINEAR_ANCHOR_X_0 * (eps_anchored - eps_unmoored)
        
        denoised = x_0 - sigma * eps
        
        epsilon  = denoised - x_0
        
        return epsilon, denoised
    
    
    
    def get_epsilon(self,
                    x_0       : Tensor,
                    x         : Tensor,
                    denoised  : Tensor,
                    sigma     : Tensor,
                    sub_sigma : Tensor,
                    ) -> Tensor:
        
        eps_anchored = (x_0 - denoised) / sigma
        eps_unmoored = (x   - denoised) / sub_sigma
        
        eps      = eps_unmoored + self.LINEAR_ANCHOR_X_0 * (eps_anchored - eps_unmoored)
        
        denoised = x_0 - sigma * eps
        
        return denoised - x_0
    
    
    
    def get_epsilon_anchored(self, x_0:Tensor, denoised:Tensor, sigma:Tensor) -> Tensor:
        return denoised - x_0
    
    
    
    def get_guide_epsilon(self,
                            x_0           : Tensor,
                            x             : Tensor,
                            y             : Tensor,
                            sigma         : Tensor,
                            sigma_cur     : Tensor,
                            sigma_down    : Optional[Tensor] = None,
                            epsilon_scale : Optional[Tensor] = None,
                            ) -> Tensor:

        sigma_cur = epsilon_scale if epsilon_scale is not None else sigma_cur

        if sigma_down > sigma:
            eps_unmoored = (sigma_cur/(self.sigma_max - sigma_cur)) * (x   - y)
        else:
            eps_unmoored = y - x 
        
        if self.EO("manually_anchor_unsampler"):
            if sigma_down > sigma:
                eps_anchored = (sigma    /(self.sigma_max - sigma)) * (x_0 - y)
            else:
                eps_anchored = y - x_0
            eps_guide = eps_unmoored + self.LINEAR_ANCHOR_X_0 * (eps_anchored - eps_unmoored)
        else:
            eps_guide = eps_unmoored
        
        return eps_guide




class RK_Method_Linear(RK_Method_Beta):
    def __init__(self,
                model,
                rk_type       : str,
                noise_anchor  : float,
                noise_boost_normalize  : bool,
                model_device  : str         = 'cuda',
                work_device   : str         = 'cpu',
                dtype         : torch.dtype = torch.float64,
                extra_options : str         = "",
                ):
        
        super().__init__(model,
                        rk_type,
                        noise_anchor,
                        noise_boost_normalize,
                        model_device  = model_device,
                        work_device   = work_device,
                        dtype         = dtype,
                        extra_options = extra_options,
                        ) 
        
    @staticmethod
    def alpha_fn(neg_h:Tensor) -> Tensor:
        return torch.ones_like(neg_h)

    @staticmethod
    def sigma_fn(t:Tensor) -> Tensor:
        return t

    @staticmethod
    def t_fn(sigma:Tensor) -> Tensor:
        return sigma
    
    @staticmethod
    def h_fn(sigma_down:Tensor, sigma:Tensor) -> Tensor:
        return sigma_down - sigma
    
    def __call__(self,
                x         : Tensor,
                sub_sigma : Tensor,
                x_0       : Optional[Tensor] = None,
                sigma     : Optional[Tensor] = None,
                transformer_options : Optional[dict] = None,
                ) -> Tuple[Tensor, Tensor]:
        
        x_0   = x         if x_0   is None else x_0
        sigma = sub_sigma if sigma is None else sigma
        
        if transformer_options is not None:
            self.extra_args.setdefault("model_options", {}).setdefault("transformer_options", {}).update(transformer_options)     
        
        denoised = self.model_denoised(x.to(self.model_device), sub_sigma.to(self.model_device), **self.extra_args).to(sigma.device)

        epsilon_anchor   = (x_0 - denoised) / sigma
        epsilon_unmoored =   (x - denoised) / sub_sigma
        
        epsilon = epsilon_unmoored + self.LINEAR_ANCHOR_X_0 * (epsilon_anchor - epsilon_unmoored)

        return epsilon, denoised



    def get_epsilon(self,
                    x_0       : Tensor,
                    x         : Tensor,
                    denoised  : Tensor,
                    sigma     : Tensor,
                    sub_sigma : Tensor,
                    ) -> Tensor:
        
        eps_anchor   = (x_0 - denoised) / sigma
        eps_unmoored =   (x - denoised) / sub_sigma
        
        return eps_unmoored + self.LINEAR_ANCHOR_X_0 * (eps_anchor - eps_unmoored)
    
    
    
    def get_epsilon_anchored(self, x_0:Tensor, denoised:Tensor, sigma:Tensor) -> Tensor:
        return (x_0 - denoised) / sigma
    
    
    
    def get_guide_epsilon(self, 
                            x_0           : Tensor, 
                            x             : Tensor, 
                            y             : Tensor, 
                            sigma         : Tensor, 
                            sigma_cur     : Tensor, 
                            sigma_down    : Optional[Tensor] = None, 
                            epsilon_scale : Optional[Tensor] = None, 
                            ) -> Tensor:

        if sigma_down > sigma:
            sigma_ratio = self.sigma_max - sigma_cur.clone()
        else:
            sigma_ratio = sigma_cur.clone()
        sigma_ratio = epsilon_scale if epsilon_scale is not None else sigma_ratio

        if sigma_down is None:
            return (x - y) / sigma_ratio
        else:
            if sigma_down > sigma:
                return (y - x) / sigma_ratio
            else:
                return (x - y) / sigma_ratio


