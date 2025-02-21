import torch
from torch import FloatTensor

import torch.nn.functional as F
import torchvision.transforms as T

import comfy.model_patcher
import comfy.supported_models

import itertools 

from .noise_classes import *
from .rk_coefficients_beta import *
from .phi_functions import *
from .helper import get_orthogonal, get_collinear, get_extra_options_list, has_nested_attr

MAX_STEPS = 10000


def get_data_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (sigma_next * x - sigma * x_next) / h

def get_epsilon_from_step(x, x_next, sigma, sigma_next):
    h = sigma_next - sigma
    return (x - x_next) / h



class RK_Method_Beta:
    def __init__(self, model, rk_type, device='cuda', dtype=torch.float64, extra_options=""):
        self.device = device
        self.dtype  = dtype
                
        self.model = model
        self.model_sampling = model.inner_model.inner_model.model_sampling
        self.sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(dtype)
        self.sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(dtype)
        
        self.rk_type = rk_type
             
        self.IMPLICIT = rk_type in IRK_SAMPLER_NAMES_BETA
        self.EXPONENTIAL = RK_Method_Beta.is_exponential(rk_type)
        self.LINEAR_ANCHOR_X_0 = False
        self.SYNC_SUBSTEP_MEAN_CW = True
        
        self.A = None
        self.B = None
        self.U = None
        self.V = None
        
        self.rows = 0
        self.cols = 0
        
        self.denoised = None
        self.uncond   = None
        
        self.y0     = None
        self.y0_inv = None
        
        self.multistep_stages = 0
        self.row_offset = None
        
        self.cfg_cw = 1.0
        self.extra_args = None
        
        self.extra_options = extra_options
        
        self.reorder_tableau_indices = get_extra_options_list("reorder_tableau_indices", "", extra_options).split(",")
        if self.reorder_tableau_indices[0]:
            self.reorder_tableau_indices = [int(self.reorder_tableau_indices[_]) for _ in range(len(self.reorder_tableau_indices))]
            
        if extra_options_flag("linear_anchor_x_0", extra_options):
            self.LINEAR_ANCHOR_X_0 = True
        else:
            self.LINEAR_ANCHOR_X_0 = False

    @staticmethod
    def is_exponential(rk_type):
        if rk_type.startswith(("res", "dpmpp", "ddim", "pec", "etdrk", "lawson", "abnorsett"   )): 
            return True
        else:
            return False

    @staticmethod
    def create(model, rk_type, device='cuda', dtype=torch.float64, extra_options=""):
        if RK_Method_Beta.is_exponential(rk_type):
            return RK_Method_Exponential(model, rk_type, device, dtype)
        else:
            return RK_Method_Linear(model, rk_type, device, dtype)
                
    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")
    
    def model_epsilon(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        denoised = self.model(x, sigma * s_in, **extra_args)
        denoised = self.calc_cfg_channelwise(denoised)

        #return x0 ###################################THIS WORKS ONLY WITH THE MODEL SAMPLING PATCH
        eps = (x - denoised) / (sigma * s_in).view(x.shape[0], 1, 1, 1)
        return eps, denoised
    
    def model_denoised(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        denoised = self.model(x, sigma * s_in, **extra_args)
        denoised = self.calc_cfg_channelwise(denoised)
        return denoised


    def set_coeff(self, rk_type, h, c1=0.0, c2=0.5, c3=1.0, step=0, sigmas=None, sigma_down=None):

        self.rk_type     = rk_type
        self.IMPLICIT    = rk_type in IRK_SAMPLER_NAMES_BETA
        self.EXPONENTIAL = RK_Method_Beta.is_exponential(rk_type) 

        sigma = sigmas[step]
        sigma_next = sigmas[step+1]
        
        h_prev = []
        a, b, u, v, ci, multistep_stages, hybrid_stages, FSAL = get_rk_methods_beta(rk_type, h, c1, c2, c3, h_prev, step, sigmas, sigma, sigma_next, sigma_down, self.extra_options)
        
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
        
        if self.IMPLICIT:
            self.reorder_tableau(self.reorder_tableau_indices)


    def reorder_tableau(self, indices):
        if indices[0]:
            self.A    = self.A   [indices]
            self.B[0] = self.B[0][indices]
            self.C    = self.C   [indices]
            self.C = torch.cat((self.C, self.C[-1:])) 
        return



    def update_substep(self, x_0, x_, eps_, eps_prev_, row, row_offset, h_new, h_new_orig, extra_options):
            
        if row < self.rows - row_offset   and   self.multistep_stages == 0:
            row_tmp_offset = row + row_offset

        else:
            row_tmp_offset = row + 1
                
        zr = self.zum(row+row_offset+self.multistep_stages, eps_, eps_prev_)
        
        x_[row_tmp_offset] = x_0 + h_new * zr
        
        if (self.SYNC_SUBSTEP_MEAN_CW and h_new != h_new_orig) or extra_options_flag("sync_mean_noise", extra_options):
            if not extra_options_flag("disable_sync_mean_noise", extra_options):
                x_row_down = x_0 + h_new_orig * zr
                x_[row_tmp_offset] = x_[row_tmp_offset] - x_[row_tmp_offset].mean(dim=(-2,-1), keepdim=True) + x_row_down.mean(dim=(-2,-1), keepdim=True)
        
        return x_


    
    def a_k_einsum(self, row, k):
        return torch.einsum('i, i... -> ...', self.A[row], k[:self.cols])
    
    def b_k_einsum(self, row, k):
        return torch.einsum('i, i... -> ...', self.B[row], k[:self.cols])
    
    def u_k_einsum(self, row, k_prev):
        return torch.einsum('i, i... -> ...', self.U[row], k_prev[:self.cols]) if (self.U is not None and k_prev is not None) else 0
    
    def v_k_einsum(self, row, k_prev):
        return torch.einsum('i, i... -> ...', self.V[row], k_prev[:self.cols]) if (self.V is not None and k_prev is not None) else 0
    
    def zum(self, row, k, k_prev=None,):
        if row < self.rows:
            return self.a_k_einsum(row, k) + self.u_k_einsum(row, k_prev)
        else:
            row = row - self.rows
            return self.b_k_einsum(row, k) + self.v_k_einsum(row, k_prev)
        
    def zum_tableau(self, k, k_prev=None,):
        a_k_sum = torch.einsum('ij, j... -> i...', self.A, k[:self.cols])
        u_k_sum = torch.einsum('ij, j... -> i...', self.U, k_prev[:self.cols]) if (self.U is not None and k_prev is not None) else 0
        return a_k_sum + u_k_sum
        

    def init_cfg_channelwise(self, x, cfg_cw=1.0, **extra_args):
        self.uncond = [torch.full_like(x, 0.0)]
        self.cfg_cw = cfg_cw
        if cfg_cw != 1.0:
            def post_cfg_function(args):
                self.uncond[0] = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
        return extra_args
            
            
    def calc_cfg_channelwise(self, denoised):
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
    def calculate_res_2m_step(x_0, denoised_, sigma_down, sigmas, step,):
        if denoised_[2].sum() == 0:
            return None
        
        sigma = sigmas[step]
        sigma_prev = sigmas[step-1]
        
        h_prev = -torch.log(sigma/sigma_prev)
        h = -torch.log(sigma_down/sigma)

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
    def calculate_res_3m_step(x_0, denoised_, sigma_down, sigmas, step,):
        if denoised_[3].sum() == 0:
            return None
        
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

    def swap_rk_type_at_step_or_threshold(self, x_0, data_prev_, sigma_down, sigmas, step, RK, rk_swap_step, rk_swap_threshold, rk_swap_type, rk_swap_print):
        if rk_swap_type == "":
            if self.EXPONENTIAL:
                rk_swap_type = "res_3m" 
            else:
                rk_swap_type = "deis_3m"
            
        if step > rk_swap_step:
            print("Switching rk_type to:", rk_swap_type)
            self.rk_type = rk_swap_type
        if step > 2 and sigmas[step+1] > 0 and self.rk_type != rk_swap_type and rk_swap_threshold > 0:
            x_res_2m, denoised_res_2m = RK.calculate_res_2m_step(x_0, data_prev_, sigma_down, sigmas, step)
            x_res_3m, denoised_res_3m = RK.calculate_res_3m_step(x_0, data_prev_, sigma_down, sigmas, step)
            if rk_swap_print:
                print("res_3m - res_2m:", torch.norm(denoised_res_3m - denoised_res_2m).item())
            if rk_swap_threshold > torch.norm(denoised_res_2m - denoised_res_3m):
                print("Switching rk_type to:", rk_swap_type, "at step:", step)
                self.rk_type = rk_swap_type
                
        return self.rk_type


    def bong_iter(self, x_0, x_, eps_, eps_prev_, data_, sigma, s_, row, row_offset, h, extra_options):
        bong_iter_max_row = self.rows - row_offset
        if extra_options_flag("bong_iter_max_row_full", extra_options):
            bong_iter_max_row = self.rows
        
        if row < bong_iter_max_row   and   self.multistep_stages == 0:
            bong_strength = float(get_extra_options_kv("use_bong", "1.0", extra_options))
            
            if bong_strength != 1.0:
                x_0_tmp = x_0.clone()
                x_tmp_ = x_.clone()
                eps_tmp_ = eps_.clone()
                
            for i in range(100):
                x_0 = x_[row+row_offset] - h * self.zum(row+row_offset, eps_, eps_prev_)
                for rr in range(row+row_offset):
                    x_[rr] = x_0 + h * self.zum(rr, eps_, eps_prev_)
                for rr in range(row+row_offset):
                    if extra_options_flag("zonkytar", extra_options):
                        #eps_[rr] = self.get_unsample_epsilon(x_[rr], x_0, data_[rr], sigma, s_[rr])
                        eps_[rr] = self.get_epsilon(x_[rr], x_0, data_[rr], sigma, s_[rr])
                    else:
                        eps_[rr] = self.get_epsilon(x_0, x_[rr], data_[rr], sigma, s_[rr])
                    
            if bong_strength != 1.0:
                x_0  = x_0_tmp  + bong_strength * (x_0  - x_0_tmp)
                x_   = x_tmp_   + bong_strength * (x_   - x_tmp_)
                eps_ = eps_tmp_ + bong_strength * (eps_ - eps_tmp_)
        
        return x_0, x_, eps_


    def newton_iter(self, x_0, x_, eps_, eps_prev_, data_, s_, row, h, sigmas, step, newton_name, extra_options):
        
        newton_iter_name = "newton_iter_" + newton_name
        
        default_anchor_x_all = False
        if newton_name == "lying":
            default_anchor_x_all = True
        
        newton_iter                 =   int(get_extra_options_kv(newton_iter_name,                    str("100"),   extra_options))
        newton_iter_skip_last_steps =   int(get_extra_options_kv(newton_iter_name + "_skip_last_steps", str("0"),   extra_options))
        newton_iter_mixing_rate     = float(get_extra_options_kv(newton_iter_name + "_mixing_rate",    str("1.0"),  extra_options))
        
        newton_iter_anchor          =   int(get_extra_options_kv(newton_iter_name + "_anchor",          str("0"),   extra_options))
        newton_iter_anchor_x_all    =  bool(get_extra_options_kv(newton_iter_name + "_anchor_x_all",   str(default_anchor_x_all),       extra_options))
        newton_iter_type            =       get_extra_options_kv(newton_iter_name + "_type",      "from_epsilon",   extra_options)
        newton_iter_sequence        =       get_extra_options_kv(newton_iter_name + "_sequence",        "double",   extra_options)
        
        row_b_offset = 0
        if extra_options_flag(newton_iter_name + "_include_row_b", extra_options):
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
                        eps_[r_] = get_epsilon_from_step(x_0, x_[r_], sigma, s_[r_])
                    elif newton_iter_type == "from_alt":
                        eps_[r_] = x_0/sigma - x_[r_]/s_[r_]
                    elif newton_iter_type == "from_epsilon":
                        eps_ [r_] = self.get_epsilon(x_0, x_[r_], data_[r_], sigma, s_[r_])
                    
                    if extra_options_flag(newton_iter_name + "_opt", extra_options):
                        opt_timing, opt_type, opt_subtype = get_extra_options_list(newton_iter_name+"_opt", "", extra_options).split(",")
                        
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
                                    eps_ [r_] = get_collinear(eps_a, eps_b)
                                elif opt_type == "proj":
                                    eps_ [r_] = get_collinear(eps_a, eps_b) + get_orthogonal(eps_b, eps_a)
                                    
                    x_  [r_] =   x_tmp + newton_iter_mixing_rate * (x_  [r_] -   x_tmp)
                    eps_[r_] = eps_tmp + newton_iter_mixing_rate * (eps_[r_] - eps_tmp)
                    
                if newton_iter_sequence == "double":
                    break
        
        return x_, eps_




class RK_Method_Exponential(RK_Method_Beta):
    def __init__(self, model, rk_type, device='cuda', dtype=torch.float64):
        super().__init__(model, rk_type, device, dtype) 
        
    @staticmethod
    def alpha_fn(neg_h):
        return torch.exp(neg_h)

    @staticmethod
    def sigma_fn(t):
        return t.neg().exp()

    @staticmethod
    def t_fn(sigma):
        return sigma.log().neg()
    
    @staticmethod
    def h_fn(sigma_down, sigma):
        return -torch.log(sigma_down/sigma)

    def __call__(self, x, sub_sigma, x_0, sigma): 
        denoised = self.model_denoised(x, sub_sigma, **self.extra_args)
        epsilon = denoised - x_0
        #print("MODEL SUB_SIGMA: ", round(float(sub_sigma),3), round(float(sigma),3))

        return epsilon, denoised
    
    def get_epsilon(self, x_0, x, denoised, sigma, sub_sigma):
        return denoised - x_0
        
    def get_epsilon_anchored(self, x_0, denoised, sigma):
        return denoised - x_0
    
    def get_unsample_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, unsample_resample_scale=None, extra_options=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = unsample_resample_scale if unsample_resample_scale is not None else sigma_cur

        if (extra_options_flag("power_unsample", extra_options)   or   extra_options_flag("power_resample", extra_options))   and not extra_options_flag("disable_power_", extra_options):
            if sigma_down is None:
                return y - x_0
            else:
                if sigma_down > sigma:
                    return (x_0 - y) * sigma_cur
                else:
                    return (y - x_0) * sigma_cur
        else:
            if sigma_down is None:
                return (y - x_0) #/ sigma_cur
            else:
                if sigma_down > sigma:
                    return (x_0 - y) #/ sigma_cur
                else:
                    return (y - x_0) #/ sigma_cur



class RK_Method_Linear(RK_Method_Beta):
    def __init__(self, model, rk_type, device='cuda', dtype=torch.float64):
        super().__init__(model, rk_type, device, dtype) 
        
    @staticmethod
    def alpha_fn(neg_h):
        return torch.ones_like(neg_h)

    @staticmethod
    def sigma_fn(t):
        return t

    @staticmethod
    def t_fn(sigma):
        return sigma
    
    @staticmethod
    def h_fn(sigma_down, sigma):
        return sigma_down - sigma
    
    def __call__(self, x, sub_sigma, x_0, sigma): 
        denoised = self.model_denoised(x, sub_sigma, **self.extra_args)
        
        if self.LINEAR_ANCHOR_X_0:
            epsilon = (x_0 - denoised) / sigma
        else:
            epsilon = (x - denoised) / sub_sigma
        #print("MODEL SUB_SIGMA: ", round(float(sub_sigma),3), round(float(sigma),3))

        return epsilon, denoised

    def get_epsilon(self, x_0, x, denoised, sigma, sub_sigma):
        if self.LINEAR_ANCHOR_X_0:
            eps = (x_0 - denoised) / sigma
        else:
            eps = (x - denoised) / sub_sigma
        return eps
    
    def get_epsilon_anchored(self, x_0, denoised, sigma):
        eps = (x_0 - denoised) / sigma
        return eps
    
    def get_unsample_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, unsample_resample_scale=None, extra_options=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = unsample_resample_scale if unsample_resample_scale is not None else sigma_cur

        if sigma_down is None:
            return (x - y) / sigma_cur
        else:
            if sigma_down > sigma:
                return (y - x) / sigma_cur
            else:
                return (x - y) / sigma_cur



class RK_NoiseSampler:
    def __init__(self, RK, model, step=0, device='cuda', dtype=torch.float64, extra_options=""):
        self.device = device
        self.dtype = dtype
        
        self.model = model
                
        self.sigma_fn = RK.sigma_fn
        self.t_fn = RK.t_fn
        self.h_fn = RK.h_fn

        self.row_offset = 1 if not RK.IMPLICIT else 0
        
        self.step   = step
        
        self.sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(self.dtype).to(self.device)
        self.sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(self.dtype).to(self.device)
        
        self.noise_sampler  = None
        self.noise_sampler2 = None
        
        self.noise_mode_sde         = None
        self.noise_mode_sde_substep = None
        
        self.LOCK_H_SCALE = True
        
        if has_nested_attr(model, "inner_model.inner_model.model_sampling"):
            model_sampling = model.inner_model.inner_model.model_sampling
        elif has_nested_attr(model, "model.model_sampling"):
            model_sampling = model.model.model_sampling
        
        self.CONST = isinstance(model_sampling, comfy.model_sampling.CONST)
        self.VARIANCE_PRESERVING = isinstance(model_sampling, comfy.model_sampling.CONST)
        
        self.extra_options = extra_options
        
        self.DOWN_SUBSTEP = extra_options_flag("down_substep", extra_options)  
        self.DOWN_STEP    = extra_options_flag("down_step",    extra_options)  



    def init_noise_samplers(self, x, noise_seed, noise_sampler_type, noise_sampler_type2, noise_mode_sde, noise_mode_sde_substep, overshoot_mode, overshoot_mode_substep, noise_boost_step, noise_boost_substep, alpha, alpha2, k=1., k2=1., scale=0.1, scale2=0.1):
        self.noise_sampler_type     = noise_sampler_type
        self.noise_sampler_type2    = noise_sampler_type2
        self.noise_mode_sde         = noise_mode_sde
        self.noise_mode_sde_substep = noise_mode_sde_substep
        self.overshoot_mode         = overshoot_mode
        self.overshoot_mode_substep = overshoot_mode_substep
        self.noise_boost_step       = noise_boost_step
        self.noise_boost_substep    = noise_boost_substep
        self.s_in                   = x.new_ones([1])
        
        if noise_seed < 0:
            seed = torch.initial_seed()+1 
            print("SDE noise seed: ", seed, " (set via torch.initial_seed()+1)")
        else:
            seed = noise_seed
            print("SDE noise seed: ", seed)
            
        seed2 = seed + MAX_STEPS #for substep noise generation. offset needed to ensure seeds are not reused
            
        if noise_sampler_type == "fractal":
            self.noise_sampler        = NOISE_GENERATOR_CLASSES.get(noise_sampler_type )(x=x, seed=seed,  sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler.alpha  = alpha
            self.noise_sampler.k      = k
            self.noise_sampler.scale  = scale
        if noise_sampler_type2 == "fractal":
            self.noise_sampler2       = NOISE_GENERATOR_CLASSES.get(noise_sampler_type2)(x=x, seed=seed2, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler2.alpha = alpha2
            self.noise_sampler2.k     = k2
            self.noise_sampler2.scale = scale2
        else:
            self.noise_sampler  = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type )(x=x, seed=seed,  sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler2 = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type2)(x=x, seed=seed2, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            
            
            
    def set_substep_list(self, RK):
        self.multistep_stages = RK.multistep_stages
        self.rows = RK.rows
        self.C    = RK.C
        #self.s_ = [(self.sigma_fn(self.t_fn(self.sigma) + self.h*c_)) * self.s_in for c_ in self.C]
        self.s_ = self.sigma_fn(self.t_fn(self.sigma) + self.h * self.C)
    
    
    
    def get_sde_coeff(self, sigma_next, sigma_down=None, sigma_up=None, eta=0.0):
        if self.VARIANCE_PRESERVING:
            if sigma_down is not None:
                alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
                sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5 
                
            elif sigma_up is not None:
                if sigma_up >= sigma_next:
                    print("Maximum VPSDE noise level exceeded: falling back to hard noise mode.")
                    if eta >= 1:
                        sigma_up = sigma_next * 0.9999 #avoid sqrt(neg_num) later 
                    else:
                        sigma_up = sigma_next * eta 
                    
                sigma_signal   = self.sigma_max - sigma_next
                sigma_residual = (sigma_next ** 2 - sigma_up ** 2) ** .5
                alpha_ratio    = sigma_signal + sigma_residual
                sigma_down     = sigma_residual / alpha_ratio     
                
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            
            if sigma_down is not None:
                sigma_up   = (sigma_next ** 2 - sigma_down ** 2) ** .5   # not sure this is correct
            elif sigma_up is not None:
                sigma_down = (sigma_next ** 2 - sigma_up   ** 2) ** .5    
        
        return alpha_ratio, sigma_down, sigma_up



    def set_sde_step(self, sigma, sigma_next, eta, overshoot, s_noise):
        self.sigma_0    = sigma
        self.sigma_next = sigma_next
        
        self.s_noise    = s_noise
        self.eta        = eta
        self.overshoot  = overshoot
        
        self.sigma_up_eta, self.sigma_eta, self.sigma_down_eta, self.alpha_ratio_eta \
            = self.get_sde_step(sigma, sigma_next, eta, self.noise_mode_sde, self.DOWN_STEP, SUBSTEP=False)
        self.sigma_up, self.sigma, self.sigma_down, self.alpha_ratio \
            = self.get_sde_step(sigma, sigma_next, overshoot, self.overshoot_mode, self.DOWN_STEP, SUBSTEP=False)
            
        self.h         = self.h_fn(self.sigma_down, self.sigma)
        self.h_no_eta  = self.h_fn(self.sigma_next, self.sigma)
        self.h = self.h + self.noise_boost_step * (self.h_no_eta - self.h)
        

        
        
        
    def set_sde_substep(self, row, multistep_stages, eta_substep, overshoot_substep, s_noise_substep, full_iter=0, diag_iter=0, implicit_steps_full=0, implicit_steps_diag=0):    
        self.sub_sigma_up, self.sub_sigma, self.sub_sigma_next, self.sub_sigma_down, self.sub_alpha_ratio \
            = 0., self.s_[row], self.s_[row+self.row_offset+multistep_stages], self.s_[row+self.row_offset+multistep_stages], 1.
            
        self.sub_sigma_up_eta, self.sub_sigma_eta, self.sub_sigma_down_eta, self.sub_alpha_ratio_eta = self.sub_sigma_up, self.sub_sigma, self.sub_sigma_down, self.sub_alpha_ratio
        
        self.s_noise_substep   = s_noise_substep
        self.eta_substep       = eta_substep
        self.overshoot_substep = overshoot_substep

        if row < self.rows   and   self.s_[row+self.row_offset+multistep_stages] > 0:
            if   diag_iter > 0 and diag_iter == implicit_steps_diag and extra_options_flag("implicit_substep_skip_final_eta",  self.extra_options):
                pass
            elif diag_iter > 0 and                                      extra_options_flag("implicit_substep_only_first_eta",  self.extra_options):
                pass
            elif full_iter > 0 and full_iter == implicit_steps_full and extra_options_flag("implicit_step_skip_final_eta",     self.extra_options):
                pass
            elif full_iter > 0 and                                      extra_options_flag("implicit_step_only_first_eta",     self.extra_options):
                pass
            elif (full_iter > 0 or diag_iter > 0)                   and self.noise_sampler_type2 == "brownian":
                pass # brownian noise does not increment its seed when generated, deactivate on implicit repeats to avoid burn
            elif full_iter > 0 and                                      extra_options_flag("implicit_step_only_first_all_eta", self.extra_options):
                self.sigma_down = self.sigma_next
                self.sigma_up *= 0
                self.alpha_ratio /= self.alpha_ratio
                self.h_new = self.h = self.h_no_eta
            elif (row < self.rows-self.row_offset-multistep_stages   or   diag_iter < implicit_steps_diag)   or   extra_options_flag("substep_eta_use_final", self.extra_options):
                self.sub_sigma_up,     self.sub_sigma,     self.sub_sigma_down,     self.sub_alpha_ratio     = self.get_sde_substep(self.s_[row], self.s_[row+self.row_offset+multistep_stages], overshoot_substep, noise_mode_override=self.overshoot_mode_substep, DOWN=self.DOWN_SUBSTEP)
                self.sub_sigma_up_eta, self.sub_sigma_eta, self.sub_sigma_down_eta, self.sub_alpha_ratio_eta = self.get_sde_substep(self.s_[row], self.s_[row+self.row_offset+multistep_stages], eta_substep,       noise_mode_override=self.noise_mode_sde_substep, DOWN=self.DOWN_SUBSTEP)

        self.h_new      = self.h * self.h_fn(self.sub_sigma_down,     self.sigma) / self.h_fn(self.sub_sigma_next, self.sigma) 
        self.h_eta      = self.h * self.h_fn(self.sub_sigma_down_eta, self.sigma) / self.h_fn(self.sub_sigma_next, self.sigma) 
        self.h_new_orig = self.h_new.clone()
        self.h_new      = self.h_new + self.noise_boost_substep * (self.h - self.h_eta)
        
        
        

    def get_sde_substep(self, sigma, sigma_next, eta=0.0, noise_mode_override=None, DOWN=False,):
        return self.get_sde_step(sigma, sigma_next, eta, noise_mode_override=noise_mode_override, DOWN=DOWN, SUBSTEP=True,)

    def get_sde_step(self, sigma, sigma_next, eta=0.0, noise_mode_override=None, DOWN=False, SUBSTEP=False, ):
            
        if noise_mode_override is not None:
            noise_mode = noise_mode_override
        elif SUBSTEP:
            noise_mode = self.noise_mode_sde_substep
        else:
            noise_mode = self.noise_mode_sde
        
        if DOWN:
            eta_fn = lambda eta_scale: 1-eta_scale
            sud_fn = lambda sd: (sd, None)
        else:
            eta_fn = lambda eta_scale:   eta_scale
            sud_fn = lambda su: (None, su)
        
        su, sd, sud = None, None, None
        eta_ratio   = None
        sigma_base  = sigma_next
        
        match noise_mode:
            case "hard":
                eta_ratio = eta
            case "exp": 
                h = -(sigma_next/sigma).log()
                eta_ratio = (1 - (-2*eta*h).exp())**.5
            case "soft":
                eta_ratio = 1-(1 - eta) + eta * ((sigma_next) / sigma)
            case "softer":
                eta_ratio = 1-torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
            case "soft-linear":
                eta_ratio = 1-eta * (sigma_next - sigma)
            case "sinusoidal":
                eta_ratio = torch.sin(torch.pi * sigma_next) ** 2
            case "eps":
                eta_ratio = eta * torch.sqrt((sigma_next/sigma) ** 2 * (sigma ** 2 - sigma_next ** 2) ) 
                
            case "lorentzian":
                eta_ratio  = eta
                alpha      = 1 / ((sigma_next.to(torch.float64))**2 + 1)
                sigma_base = ((1 - alpha) ** 0.5).to(sigma.dtype)
                
            case "hard_var":
                sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
                if sigma_next > sigma_var:
                    eta_ratio  = 0
                    sigma_base = sigma_next
                else:
                    eta_ratio  = eta
                    sigma_base = torch.sqrt((sigma - sigma_next).abs() + 1e-10)
                    
            case "hard_sq":
                sigma_hat = sigma * (1 + eta)
                su        = (sigma_hat ** 2 - sigma ** 2) ** .5    #su
                
                if self.VARIANCE_PRESERVING:
                    alpha_ratio, sd, su = self.get_sde_coeff(sigma_next, None, su, eta)
                else:
                    sd          = sigma_next
                    sigma       = sigma_hat
                    alpha_ratio = torch.ones_like(sigma)
                    
            case "vpsde":
                alpha_ratio, sd, su = self.get_vpsde_step_RF(sigma, sigma_next, eta)
                
        if eta_ratio is not None:
            sud = sigma_base * eta_fn(eta_ratio)
            alpha_ratio, sd, su = self.get_sde_coeff(sigma_next, *sud_fn(sud), eta)
        
        su          = torch.nan_to_num(su,          0.0)
        sd          = torch.nan_to_num(sd,    float(sigma_next))
        alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

        return su, sigma, sd, alpha_ratio
    
    def get_vpsde_step_RF(self, sigma, sigma_next, eta):
        dt = sigma - sigma_next
        sigma_up = eta * sigma * dt**0.5
        alpha_ratio = 1 - dt * (eta**2/4) * (1 + sigma)
        sigma_down = sigma_next - (eta/4)*sigma*(1-sigma)*(sigma - sigma_next)
        return sigma_up, sigma_down, alpha_ratio
        

    def swap_noise_step(self, x_0, x_next, brownian_sigma=None, brownian_sigma_next=None, ):
        if self.sigma_up_eta == 0:
            return x_next
        
        if brownian_sigma is None:
            brownian_sigma = self.sigma.clone()
        if brownian_sigma_next is None:
            brownian_sigma_next = self.sigma_next.clone()
        if self.sigma_next == 0:
            return x_next
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
        eps_next = (x_0 - x_next) / (self.sigma - self.sigma_next)
        denoised_next = x_0 - self.sigma * eps_next
        
        if brownian_sigma_next > brownian_sigma:
            s_tmp               = brownian_sigma
            brownian_sigma      = brownian_sigma_next
            brownian_sigma_next = s_tmp
        
        noise = self.noise_sampler(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

        x = self.alpha_ratio_eta * (denoised_next + self.sigma_down_eta * eps_next) + self.sigma_up_eta * noise * self.s_noise
        return x


    def swap_noise_substep(self, x_0, x_next, brownian_sigma=None, brownian_sigma_next=None, ):
        if self.sub_sigma_up_eta == 0:
            return x_next
        
        if brownian_sigma is None:
            brownian_sigma = self.sub_sigma.clone()
        if brownian_sigma_next is None:
            brownian_sigma_next = self.sub_sigma_next.clone()
        if self.sigma_next == 0:
            return x_next
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
        eps_next = (x_0 - x_next) / (self.sigma - self.sub_sigma_next)
        denoised_next = x_0 - self.sigma * eps_next
        
        if brownian_sigma_next > brownian_sigma:
            s_tmp               = brownian_sigma
            brownian_sigma      = brownian_sigma_next
            brownian_sigma_next = s_tmp
        
        noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

        x = self.sub_alpha_ratio_eta * (denoised_next + self.sub_sigma_down_eta * eps_next) + self.sub_sigma_up_eta * noise * self.s_noise_substep
        return x


    def swap_noise(self, x_0, x_next, sigma_0, sigma, sigma_next, sigma_down, sigma_up, alpha_ratio, s_noise, SUBSTEP=False, brownian_sigma=None, brownian_sigma_next=None, ):
        if sigma_up == 0:
            return x_next
        
        if brownian_sigma is None:
            brownian_sigma = sigma.clone()
        if brownian_sigma_next is None:
            brownian_sigma_next = sigma_next.clone()
        if sigma_next == 0:
            return x_next
        if brownian_sigma == brownian_sigma_next:
            brownian_sigma_next *= 0.999
        eps_next = (x_0 - x_next) / (sigma_0 - sigma_next)
        denoised_next = x_0 - sigma_0 * eps_next
        
        if brownian_sigma_next > brownian_sigma:
            s_tmp = brownian_sigma
            brownian_sigma = brownian_sigma_next
            brownian_sigma_next = s_tmp
        
        if not SUBSTEP:
            noise = self.noise_sampler(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
        else:
            noise = self.noise_sampler2(sigma=brownian_sigma, sigma_next=brownian_sigma_next)
            
        noise = (noise - noise.mean(dim=(-2, -1), keepdim=True)) / noise.std(dim=(-2, -1), keepdim=True)

        x = alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise * s_noise
        return x


    def add_noise_pre(self, x_0, x, sigma_up, sigma_0, sigma, sigma_next, real_sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None, SUBSTEP=False):
        if not self.CONST and noise_mode == "hard_sq": 
            if self.LOCK_H_SCALE:
                x = self.swap_noise(x_0, x, sigma, sigma_0, sigma_next, real_sigma_down, sigma_up, alpha_ratio, s_noise, SUBSTEP, )
            else:
                x = self.add_noise(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP, )
        return x
        
    def add_noise_post(self, x_0, x, sigma_up, sigma_0, sigma, sigma_next, real_sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None, SUBSTEP=False):
        if self.CONST   or   (not self.CONST and noise_mode != "hard_sq"):
            if self.LOCK_H_SCALE:
                x = self.swap_noise(x_0, x, sigma_0, sigma, sigma_next, real_sigma_down, sigma_up, alpha_ratio, s_noise, SUBSTEP, )
            else:
                x = self.add_noise(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP, )
        return x

    def add_noise(self, x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t, SUBSTEP, ):

        if sigma_next > 0.0 and sigma_up > 0.0:
            if sigma_next > sigma:
                s_tmp = sigma
                sigma = sigma_next
                sigma_next = s_tmp
            
            if sigma == sigma_next:
                sigma_next = sigma * 0.9999
            if not SUBSTEP:
                noise = self.noise_sampler (sigma=sigma, sigma_next=sigma_next)
            else:
                noise = self.noise_sampler2(sigma=sigma, sigma_next=sigma_next)

            #noise_ortho = get_orthogonal(noise, x)
            #noise_ortho = noise_ortho / noise_ortho.std()model,

            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            x_next = alpha_ratio * x + noise * sigma_up * s_noise
            
            return x_next
        
        else:
            return x
    
    def sigma_from_to(self, x_0, x_down, sigma, sigma_down, sigma_next):   #sigma, sigma_from, sigma_to
        eps = (x_0 - x_down) / (sigma - sigma_down)
        denoised = x_0 - sigma * eps
        x_next = denoised + sigma_next * eps
        return x_next

    def rebound_overshoot_step(self, x_0, x):
        eps = (x_0 - x) / (self.sigma - self.sigma_down)
        denoised = x_0 - self.sigma * eps
        x = denoised + self.sigma_next * eps
        return x
    
    def rebound_overshoot_substep(self, x_0, x):
        sub_eps = (x_0 - x) / (self.sigma - self.sub_sigma_down)
        sub_denoised = x_0 - self.sigma * sub_eps
        x = sub_denoised + self.sub_sigma_next * sub_eps
        return x

    def prepare_sigmas(self, sigmas, sigmas_override, d_noise, sampler_mode):
        if sigmas_override is not None:
            sigmas = sigmas_override.clone()
        sigmas = sigmas.clone() * d_noise
        
        if sigmas[0] == 0.0:      #remove padding used to prevent comfy from adding noise to the latent (for unsampling, etc.)
            UNSAMPLE = True
            sigmas = sigmas[1:-1]
        else:
            UNSAMPLE = False
        
        if hasattr(self.model, "sigmas"):
            self.model.sigmas = sigmas
            
        if sampler_mode == "standard":
            UNSAMPLE = False
        
        consecutive_duplicate_mask = torch.cat((torch.tensor([True], device=sigmas.device), torch.diff(sigmas) != 0))
        sigmas = sigmas[consecutive_duplicate_mask]
                
        if sigmas[-1] == 0:
            if sigmas[-2] < self.sigma_min:
                sigmas[-2] = self.sigma_min
            elif (sigmas[-2] - self.sigma_min).abs() > 1e-4:
                sigmas = torch.cat((sigmas[:-1], self.sigma_min.unsqueeze(0), sigmas[-1:]))
        
        self.sigmas       = sigmas
        self.UNSAMPLE     = UNSAMPLE
        self.d_noise      = d_noise
        self.sampler_mode = sampler_mode
        
        return sigmas, UNSAMPLE
    