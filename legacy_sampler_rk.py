import torch
import torch.nn.functional as F
import torchvision.transforms as T
import re


from tqdm.auto import trange
import math
import copy
import gc

import comfy.model_patcher

from .noise_classes import *
from .extra_samplers_helpers import get_deis_coeff_list
from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3
from .latents import hard_light_blend


def get_epsilon(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    x0 = model(x, sigma * s_in, **extra_args)
    eps = (x - x0) / (sigma * s_in) 
    return eps

def get_denoised(model, x, sigma, **extra_args):
    s_in = x.new_ones([x.shape[0]])
    x0 = model(x, sigma * s_in, **extra_args)
    return x0


# Remainder solution
def __phi(j, neg_h):
  remainder = torch.zeros_like(neg_h)
  
  for k in range(j): 
    remainder += (neg_h)**k / math.factorial(k)
  phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
  
  return phi_j_h
  
  
def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))




from typing import Optional


def _gamma(n: int,) -> int:
  """
  https://en.wikipedia.org/wiki/Gamma_function
  for every positive integer n,
  Γ(n) = (n-1)!
  """
  return math.factorial(n-1)

def _incomplete_gamma(s: int, x: float, gamma_s: Optional[int] = None) -> float:
  """
  https://en.wikipedia.org/wiki/Incomplete_gamma_function#Special_values
  if s is a positive integer,
  Γ(s, x) = (s-1)!*∑{k=0..s-1}(x^k/k!)
  """
  if gamma_s is None:
    gamma_s = _gamma(s)

  sum_: float = 0
  # {k=0..s-1} inclusive
  for k in range(s):
    numerator: float = x**k
    denom: int = math.factorial(k)
    quotient: float = numerator/denom
    sum_ += quotient
  incomplete_gamma_: float = sum_ * math.exp(-x) * gamma_s
  return incomplete_gamma_


# Exact analytic solution originally calculated by Clybius. https://github.com/Clybius/ComfyUI-Extra-Samplers/tree/main
def phi(j: int, neg_h: float, ):
  """
  For j={1,2,3}: you could alternatively use Kat's phi_1, phi_2, phi_3 which perform fewer steps

  Lemma 1
  https://arxiv.org/abs/2308.02157
  ϕj(-h) = 1/h^j*∫{0..h}(e^(τ-h)*(τ^(j-1))/((j-1)!)dτ)

  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84
  = 1/h^j*[(e^(-h)*(-τ)^(-j)*τ(j))/((j-1)!)]{0..h}
  https://www.wolframalpha.com/input?i=integrate+e%5E%28%CF%84-h%29*%28%CF%84%5E%28j-1%29%2F%28j-1%29%21%29d%CF%84+between+0+and+h
  = 1/h^j*((e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h)))/(j-1)!)
  = (e^(-h)*(-h)^(-j)*h^j*(Γ(j)-Γ(j,-h))/((j-1)!*h^j)
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/(j-1)!
  = (e^(-h)*(-h)^(-j)*(Γ(j)-Γ(j,-h))/Γ(j)
  = (e^(-h)*(-h)^(-j)*(1-Γ(j,-h)/Γ(j))

  requires j>0
  """
  assert j > 0
  gamma_: float = _gamma(j)
  incomp_gamma_: float = _incomplete_gamma(j, neg_h, gamma_s=gamma_)
  phi_: float = math.exp(neg_h) * neg_h**-j * (1-incomp_gamma_/gamma_)
  return phi_


rk_coeff = {
    "gauss-legendre_5s": (
    [
        [4563950663 / 32115191526, 
         (310937500000000 / 2597974476091533 + 45156250000 * (739**0.5) / 8747388808389), 
         (310937500000000 / 2597974476091533 - 45156250000 * (739**0.5) / 8747388808389),
         (5236016175 / 88357462711 + 709703235 * (739**0.5) / 353429850844),
         (5236016175 / 88357462711 - 709703235 * (739**0.5) / 353429850844)],
         
        [(4563950663 / 32115191526 - 38339103 * (739**0.5) / 6250000000),
         (310937500000000 / 2597974476091533 + 9557056475401 * (739**0.5) / 3498955523355600000),
         (310937500000000 / 2597974476091533 - 14074198220719489 * (739**0.5) / 3498955523355600000),
         (5236016175 / 88357462711 + 5601362553163918341 * (739**0.5) / 2208936567775000000000),
         (5236016175 / 88357462711 - 5040458465159165409 * (739**0.5) / 2208936567775000000000)],
         
        [(4563950663 / 32115191526 + 38339103 * (739**0.5) / 6250000000),
         (310937500000000 / 2597974476091533 + 14074198220719489 * (739**0.5) / 3498955523355600000),
         (310937500000000 / 2597974476091533 - 9557056475401 * (739**0.5) / 3498955523355600000),
         (5236016175 / 88357462711 + 5040458465159165409 * (739**0.5) / 2208936567775000000000),
         (5236016175 / 88357462711 - 5601362553163918341 * (739**0.5) / 2208936567775000000000)],
         
        [(4563950663 / 32115191526 - 38209 * (739**0.5) / 7938810),
         (310937500000000 / 2597974476091533 - 359369071093750 * (739**0.5) / 70145310854471391),
         (310937500000000 / 2597974476091533 - 323282178906250 * (739**0.5) / 70145310854471391),
         (5236016175 / 88357462711 - 470139 * (739**0.5) / 1413719403376),
         (5236016175 / 88357462711 - 44986764863 * (739**0.5) / 21205791050640)],
         
        [(4563950663 / 32115191526 + 38209 * (739**0.5) / 7938810),
         (310937500000000 / 2597974476091533 + 359369071093750 * (739**0.5) / 70145310854471391),
         (310937500000000 / 2597974476091533 + 323282178906250 * (739**0.5) / 70145310854471391),
         (5236016175 / 88357462711 + 44986764863 * (739**0.5) / 21205791050640),
         (5236016175 / 88357462711 + 470139 * (739**0.5) / 1413719403376)],
        
        [4563950663 / 16057595763,
         621875000000000 / 2597974476091533,
         621875000000000 / 2597974476091533,
         10472032350 / 88357462711,
         10472032350 / 88357462711]
    ],
    [
        1 / 2,
        1 / 2 - 99 * (739**0.5) / 10000,
        1 / 2 + 99 * (739**0.5) / 10000,
        1 / 2 - (739**0.5) / 60,
        1 / 2 + (739**0.5) / 60
    ]
    ),
    "gauss-legendre_4s": (
        [
            [1/4, 1/4 - 15**0.5 / 6, 1/4 + 15**0.5 / 6, 1/4],            
            [1/4 + 15**0.5 / 6, 1/4, 1/4 - 15**0.5 / 6, 1/4],          
            [1/4, 1/4 + 15**0.5 / 6, 1/4, 1/4 - 15**0.5 / 6],            
            [1/4 - 15**0.5 / 6, 1/4, 1/4 + 15**0.5 / 6, 1/4],           
            [1/8, 3/8, 3/8, 1/8]                                        
        ],
        [
            1/2 - 15**0.5 / 10,                                     
            1/2 + 15**0.5 / 10,                                         
            1/2 + 15**0.5 / 10,                                        
            1/2 - 15**0.5 / 10                                         
        ]
    ),
    "gauss-legendre_3s": (
        [
            [5/36, 2/9 - 15**0.5 / 15, 5/36 - 15**0.5 / 30],
            [5/36 + 15**0.5 / 24, 2/9, 5/36 - 15**0.5 / 24],
            [5/36 + 15**0.5 / 30, 2/9 + 15**0.5 / 15, 5/36],
            [5/18, 4/9, 5/18]
        ],
        [1/2 - 15**0.5 / 10, 1/2, 1/2 + 15**0.5 / 10]
    ),
    "gauss-legendre_2s": (
        [
            [1/4, 1/4 - 3**0.5 / 6],
            [1/4 + 3**0.5 / 6, 1/4],
            [1/2, 1/2],
        ],
        [1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6]
    ),
    "radau_iia_3s": (
        [    
            [11/45 - 7*6**0.5 / 360, 37/225 - 169*6**0.5 / 1800, -2/225 + 6**0.5 / 75],
            [37/225 + 169*6**0.5 / 1800, 11/45 + 7*6**0.5 / 360, -2/225 - 6**0.5 / 75],
            [4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9],
            [4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9],
        ],
        [2/5 - 6**0.5 / 10, 2/5 + 6**0.5 / 10, 1.]
    ),
    "radau_iia_2s": (
        [    
            [5/12, -1/12],
            [3/4, 1/4],
            [3/4, 1/4],
        ],
        [1/3, 1]
    ),
    "lobatto_iiic_3s": (
        [    
            [1/6, -1/3, 1/6],
            [1/6, 5/12, -1/12],
            [1/6, 2/3, 1/6],
            [1/6, 2/3, 1/6],
        ],
        [0, 1/2, 1]
    ),
    "lobatto_iiic_2s": (
        [    
            [1/2, -1/2],
            [1/2, 1/2],
            [1/2, 1/2],
        ],
        [0, 1]
    ),
    "dormand-prince_13s": (
        [
            [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0],
            [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0],
            [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0, 0, 0, 0],
            [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0],
            [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0],
            [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0],
            [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0],
            [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0],
            [14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]
        ],
        [0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248 / 9719169821, 13/20, 1201146811 / 1299019798, 1, 1],
    ),
    "dormand-prince_6s": (
        [
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ],
        [0, 1/5, 3/10, 4/5, 8/9, 1],
    ),
    "dormand-prince_6s_alt": (
        [
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ],
        [0, 1/5, 3/10, 4/5, 8/9, 1],
    ),
    "dormand-prince_7s": (
        [
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ],
        [0, 1/5, 3/10, 4/5, 8/9, 1],
    ),
    "bogacki-shampine_7s": ( #5th order
        [
            [1/6, 0, 0, 0, 0, 0, 0],
            [2/27, 4/27, 0, 0, 0, 0, 0],
            [183/1372, -162/343, 1053/1372, 0, 0, 0, 0],
            [68/297, -4/11, 42/143, 1960/3861, 0, 0, 0],
            [597/22528, 81/352, 63099/585728, 58653/366080, 4617/20480, 0, 0],
            [174197/959244, -30942/79937, 8152137/19744439, 666106/1039181, -29421/29068, 482048/414219, 0],
            [587/8064, 0, 4440339/15491840, 24353/124800, 387/44800, 2152/5985, 7267/94080]
        ],
        [0, 1/6, 2/9, 3/7, 2/3, 3/4, 1] 
    ),
    "rk4_4s": (
        [
            [1/2, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 1, 0],
            [1/6, 1/3, 1/3, 1/6]
        ],
        [0, 1/2, 1/2, 1],
    ),
    "rk38_4s": (
        [
            [1/3, 0, 0, 0],
            [-1/3, 1, 0, 0],
            [1, -1, 1, 0],
            [1/8, 3/8, 3/8, 1/8]
        ],
        [0, 1/3, 2/3, 1],
    ),
    "ralston_4s": (
        [
            [2/5, 0, 0, 0],
            [(-2889+1428 * 5**0.5)/1024,   (3785-1620 * 5**0.5)/1024,  0, 0],
            [(-3365+2094 * 5**0.5)/6040,   (-975-3046 * 5**0.5)/2552,  (467040+203968*5**0.5)/240845, 0],
            [(263+24*5**0.5)/1812, (125-1000*5**0.5)/3828, (3426304+1661952*5**0.5)/5924787, (30-4*5**0.5)/123]
        ],
        [0, 2/5, (14-3 * 5**0.5)/16, 1],
    ),
    "heun_3s": (
        [
            [1/3, 0, 0],
            [0, 2/3, 0],
            [1/4, 0, 3/4]
        ],
        [0, 1/3, 2/3],
    ),
    "kutta_3s": (
        [
            [1/2, 0, 0],
            [-1, 2, 0],
            [1/6, 2/3, 1/6]
        ],
        [0, 1/2, 1],
    ),
    "ralston_3s": (
        [
            [1/2, 0, 0],
            [0, 3/4, 0],
            [2/9, 1/3, 4/9]
        ],
        [0, 1/2, 3/4],
    ),
    "houwen-wray_3s": (
        [
            [8/15, 0, 0],
            [1/4, 5/12, 0],
            [1/4, 0, 3/4]
        ],
        [0, 8/15, 2/3],
    ),
    "ssprk3_3s": (
        [
            [1, 0, 0],
            [1/4, 1/4, 0],
            [1/6, 1/6, 2/3]
        ],
        [0, 1, 1/2],
    ),
    "midpoint_2s": (
        [
            [1/2, 0],
            [0, 1]
        ],
        [0, 1/2],
    ),
    "heun_2s": (
        [
            [1, 0],
            [1/2, 1/2]
        ],
        [0, 1],
    ),
    "ralston_2s": (
        [
            [2/3, 0],
            [1/4, 3/4]
        ],
        [0, 2/3],
    ),
    "euler": (
        [
            [1],
        ],
        [0],
    ),
}


def get_rk_methods(rk_type, h, c1=0.0, c2=0.5, c3=1.0, h_prev=None, h_prev2=None, stepcount=0, sigmas=None):
    FSAL = False
    multistep_stages = 0
    
    if rk_type[:4] == "deis": 
        order = int(rk_type[-2])
        if stepcount < order:
            if order == 4:
                rk_type = "res_3s"
                order = 3
            elif order == 3:
                rk_type = "res_3s"
            elif order == 2:
                rk_type = "res_2s"
        else:
            rk_type = "deis"
            multistep_stages = order-1

    
    if rk_type[-2:] == "2m": #multistep method
        if h_prev is not None: 
            multistep_stages = 1
            c2 = -h_prev / h
            rk_type = rk_type[:-2] + "2s"
        else:
            rk_type = rk_type[:-2] + "2s"
            
    if rk_type[-2:] == "3m": #multistep method
        if h_prev2 is not None: 
            multistep_stages = 2
            c2 = -h_prev2 / h_prev
            c3 = -h_prev / h
            rk_type = rk_type[:-2] + "3s"
        else:
            rk_type = rk_type[:-2] + "3s"
    
    if rk_type in rk_coeff:
        ab, ci = copy.deepcopy(rk_coeff[rk_type])
        ci = ci[:]
        ci.append(1)
        
        alpha_fn = lambda h: 1
        t_fn = lambda sigma: sigma
        sigma_fn = lambda t: t
        h_fn = lambda sigma_down, sigma: sigma_down - sigma
        model_call = get_denoised
        EPS_PRED = False

    else:
        alpha_fn = lambda neg_h: torch.exp(neg_h)
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
        model_call = get_denoised
        EPS_PRED = False
    
    match rk_type:
        case "deis": 
            alpha_fn = lambda neg_h: torch.exp(neg_h)
            t_fn = lambda sigma: sigma.log().neg()
            sigma_fn = lambda t: t.neg().exp()
            h_fn = lambda sigma_down, sigma: -torch.log(sigma_down/sigma)
            model_call = get_epsilon
            EPS_PRED = True

            coeff_list = get_deis_coeff_list(sigmas, multistep_stages+1, deis_mode="rhoab")
            coeff_list = [[elem / h for elem in inner_list] for inner_list in coeff_list]
            if multistep_stages == 1:
                b1, b2 = coeff_list[stepcount]
                ab = [
                        [0, 0],
                        [b1, b2],
                ]
                ci = [0, 0, 1]
            if multistep_stages == 2:
                b1, b2, b3 = coeff_list[stepcount]
                ab = [
                        [0, 0, 0],
                        [0, 0, 0],
                        [b1, b2, b3],
                ]
                ci = [0, 0, 0, 1]
            if multistep_stages == 3:
                b1, b2, b3, b4 = coeff_list[stepcount]
                ab = [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [b1, b2, b3, b4],
                ]
                ci = [0, 0, 0, 0, 1]

        case "dormand-prince_6s":
            FSAL = True

        case "ddim":
            b1 = phi(1, -h)
            ab = [
                    [b1],
            ]
            ci = [0, 1]

        case "res_2s":
            a2_1 = c2 * phi(1, -h*c2)
            b1 =        phi(1, -h) - phi(2, -h)/c2
            b2 =        phi(2, -h)/c2
            
            a2_1 /= (1 - torch.exp(-h*c2)) / h
            b1 /= phi(1, -h)
            b2 /= phi(1, -h)

            ab = [
                    [a2_1, 0],
                    [b1, b2],
            ]
            ci = [0, c2, 1]

        case "res_3s":
            gamma = calculate_gamma(c2, c3)
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = gamma * c2 * phi(2, -h*c2) + (c3 ** 2 / c2) * phi(2, -h*c3) #phi_2_c3_h  # a32 from k2 to k3
            a3_1 = c3 * phi(1, -h*c3) - a3_2 # a31 from k1 to k3
            b3 = (1 / (gamma * c2 + c3)) * phi(2, -h)      
            b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
            b1 = phi(1, -h) - b2 - b3     
            0
            a3_2 /= (1 - torch.exp(-h*c3)) / h
            a3_1 /= (1 - torch.exp(-h*c3)) / h
            b1 /= phi(1, -h)
            b2 /= phi(1, -h)
            b3 /= phi(1, -h)
            
            ab = [
                    [a2_1, 0, 0],
                    [a3_1, a3_2, 0],
                    [b1, b2, b3],
            ]
            ci = [c1, c2, c3, 1]
            #ci = [0, c2, c3, 1]

        case "dpmpp_2s":
            #c2 = 0.5
            a2_1 =         c2   * phi(1, -h*c2)
            b1 = (1 - 1/(2*c2)) * phi(1, -h)
            b2 =     (1/(2*c2)) * phi(1, -h)
            
            a2_1 /= (1 - torch.exp(-h*c2)) / h
            b1 /= phi(1, -h)
            b2 /= phi(1, -h)
            
            ab = [
                    [a2_1, 0],
                    [b1, b2],
            ]
            ci = [0, c2, 1]
            
        case "dpmpp_sde_2s":
            c2 = 1.0 #hardcoded to 1.0 to more closely emulate the configuration for k-diffusion's implementation
            a2_1 =         c2   * phi(1, -h*c2)
            b1 = (1 - 1/(2*c2)) * phi(1, -h)
            b2 =     (1/(2*c2)) * phi(1, -h)
            
            a2_1 /= (1 - torch.exp(-h*c2)) / h
            b1 /= phi(1, -h)
            b2 /= phi(1, -h)
            
            ab = [
                    [a2_1, 0],
                    [b1, b2],
            ]
            ci = [0, c2, 1]

        case "dpmpp_3s":
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = (c3**2 / c2) * phi(2, -h*c3)
            a3_1 = c3 * phi(1, -h*c3) - a3_2
            b2 = 0
            b3 = (1/c3) * phi(2, -h)
            b1 = phi(1, -h) - b2 - b3
            
            a2_1 /= (1 - torch.exp(-h*c2)) / h
            a3_2 /= (1 - torch.exp(-h*c3)) / h
            a3_1 /= (1 - torch.exp(-h*c3)) / h
            b1 /= phi(1, -h)
            b2 /= phi(1, -h)
            b3 /= phi(1, -h)
            
            ab = [
                    [a2_1, 0, 0],
                    [a3_1, a3_2, 0],
                    [b1, b2, b3],
            ]
            ci = [0, c2, c3, 1]
            
        case "rk_exp_5s":
                
            c1, c2, c3, c4, c5 = 0., 0.5, 0.5, 1., 0.5
            
            a2_1 = 0.5 * phi(1, -h * c2)
            
            a3_1 = 0.5 * phi(1, -h * c3) - phi(2, -h * c3)
            a3_2 = phi(2, -h * c3)
            
            a4_1 = phi(1, -h * c4) - 2 * phi(2, -h * c4)
            a4_2 = a4_3 = phi(2, -h * c4)
            
            a5_2 = a5_3 = 0.5 * phi(2, -h * c5) - phi(3, -h * c4) + 0.25 * phi(2, -h * c4) - 0.5 * phi(3, -h * c5)
            a5_4 = 0.25 * phi(2, -h * c5) - a5_2
            a5_1 = 0.5 * phi(1, -h * c5) - 2 * a5_2 - a5_4
                    
            b1 = phi(1, -h) - 3 * phi(2, -h) + 4 * phi(3, -h)
            b2 = b3 = 0
            b4 = -phi(2, -h) + 4*phi(3, -h)
            b5 = 4 * phi(2, -h) - 8 * phi(3, -h)
            
            a2_1 /= (1 - torch.exp(-h*c2)) / h
            
            a3_1 /= (1 - torch.exp(-h*c3)) / h
            a3_2 /= (1 - torch.exp(-h*c3)) / h
            
            a4_1 /= (1 - torch.exp(-h*c4)) / h
            a4_2 /= (1 - torch.exp(-h*c4)) / h
            a4_3 /= (1 - torch.exp(-h*c4)) / h
            
            a5_1 /= (1 - torch.exp(-h*c5)) / h
            a5_2 /= (1 - torch.exp(-h*c5)) / h
            a5_3 /= (1 - torch.exp(-h*c5)) / h
            a5_4 /= (1 - torch.exp(-h*c5)) / h
            
            b1 /= phi(1, -h)
            b2 /= phi(1, -h)
            b3 /= phi(1, -h)
            b4 /= phi(1, -h)
            b5 /= phi(1, -h)
            
            ab = [
                    [a2_1, 0, 0, 0, 0],
                    [a3_1, a3_2, 0, 0, 0],
                    [a4_1, a4_2, a4_3, 0, 0],
                    [a5_1, a5_2, a5_3, a5_4, 0],
                    [b1, b2, b3, b4, b5],
            ]
            ci = [0., 0.5, 0.5, 1., 0.5, 1]


    return ab, ci, multistep_stages, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED

def get_rk_methods_order(rk_type):
    ab, ci, multistep_stages, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, torch.tensor(1.0).to('cuda').to(torch.float64), c1=0.0, c2=0.5, c3=1.0)
    return len(ci)-1

def get_rk_methods_order_and_fn(rk_type, h=None, c1=None, c2=None, c3=None, h_prev=None, h_prev2=None, stepcount=0, sigmas=None):
    if h == None:
        ab, ci, multistep_stages, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, torch.tensor(1.0).to('cuda').to(torch.float64), c1=0.0, c2=0.5, c3=1.0)
    else:
        ab, ci, multistep_stages, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, h, c1, c2, c3, h_prev, h_prev2, stepcount, sigmas)
    return len(ci)-1, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED

def get_rk_methods_coeff(rk_type, h, c1, c2, c3, h_prev=None, h_prev2=None, stepcount=0, sigmas=None):
    ab, ci, multistep_stages, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED = get_rk_methods(rk_type, h, c1, c2, c3, h_prev, h_prev2, stepcount, sigmas)
    return ab, ci, multistep_stages, EPS_PRED










@torch.no_grad()
def legacy_sample_rk(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, noise_sampler_type="brownian", noise_mode="hard", noise_seed=-1, rk_type="res_2m", implicit_sampler_name="default",
              sigma_fn_formula="", t_fn_formula="",
                  eta=0.0, eta_var=0.0, s_noise=1., d_noise=1., alpha=-1.0, k=1.0, scale=0.1, c1=0.0, c2=0.5, c3=1.0, MULTISTEP=False, cfgpp=0.0, implicit_steps=0, reverse_weight=0.0, exp_mode=False,
                  latent_guide=None, latent_guide_inv=None, latent_guide_weight=0.0, latent_guide_weights=None, guide_mode="blend",
                  GARBAGE_COLLECT=False, mask=None, LGW_MASK_RESCALE_MIN=True, sigmas_override=None, t_is=None,
                  ):
    extra_args = {} if extra_args is None else extra_args
    
    if sigmas_override is not None:
        sigmas = sigmas_override.clone()
    sigmas = sigmas.clone() * d_noise
    sigmin = model.inner_model.inner_model.model_sampling.sigma_min 
    sigmax = model.inner_model.inner_model.model_sampling.sigma_max 
    
    UNSAMPLE = False
    if sigmas[0] == 0.0:      #remove padding used to avoid need for model patch with noise inversion
        UNSAMPLE = True
        sigmas = sigmas[1:-1]
    
    if mask is None:
        mask = torch.ones_like(x)
        LGW_MASK_RESCALE_MIN = False
    else:
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, x.shape[1], 1, 1) 
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        mask = mask.to(x.dtype).to(x.device)
        
    y0, y0_inv = torch.zeros_like(x), torch.zeros_like(x)
    if latent_guide is not None:
        if sigmas[0] > sigmas[1]:
            y0 = latent_guide = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
        else:
            x = model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)

    if latent_guide_inv is not None:
        if sigmas[0] > sigmas[1]:
            y0_inv = latent_guide_inv = model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)
        elif UNSAMPLE and mask is not None:
            x = mask * x + (1-mask) * model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)

    uncond = [torch.full_like(x, 0.0)]
    if cfgpp != 0.0:
        def post_cfg_function(args):
            uncond[0] = args["uncond_denoised"]
            return args["denoised"]
        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    if noise_seed == -1:
        seed = torch.initial_seed() + 1
    else:
        seed = noise_seed

    if noise_sampler_type == "fractal":
        noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigmin, sigma_max=sigmax)
        noise_sampler.alpha = alpha
        noise_sampler.k = k
        noise_sampler.scale = scale
    else:
        noise_sampler = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigmin, sigma_max=sigmax)

    if UNSAMPLE and sigmas[0] < sigmas[1]: #sigma_next > sigma:
        y0 = noise_sampler(sigma=sigmax, sigma_next=sigmin)
        y0 = (y0 - y0.mean()) / y0.std()
        y0_inv = noise_sampler(sigma=sigmax, sigma_next=sigmin)
        y0_inv = (y0_inv - y0_inv.mean()) / y0_inv.std()
        
    order, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED = get_rk_methods_order_and_fn(rk_type)

    if exp_mode:
        model_call = get_denoised
        alpha_fn = lambda neg_h: torch.exp(neg_h)
        t_fn     = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp() 
    
    xi, ki, ki_u = [torch.zeros_like(x)]*(order+2), [torch.zeros_like(x)]*(order+1), [torch.zeros_like(x)]*(order+1)
    h, h_prev, h_prev2 = None, None, None
        
    xi[0] = x

    for _ in trange(len(sigmas)-1, disable=disable):
        sigma, sigma_next = sigmas[_], sigmas[_+1]
        
        if sigma_next == 0.0:
            rk_type = "euler"
            eta, eta_var = 0, 0

        order, model_call, alpha_fn, t_fn, sigma_fn, h_fn, FSAL, EPS_PRED = get_rk_methods_order_and_fn(rk_type)
        
        sigma_up, sigma, sigma_down, alpha_ratio = get_res4lyf_step_with_model(model, sigma, sigma_next, eta, eta_var, noise_mode, h_fn(sigma_next,sigma) )
        t_down, t = t_fn(sigma_down), t_fn(sigma)
        h = h_fn(sigma_down, sigma)
        
        c2, c3 = get_res4lyf_half_step3(sigma, sigma_down, c2, c3, t_fn=t_fn, sigma_fn=sigma_fn, t_fn_formula=t_fn_formula, sigma_fn_formula=sigma_fn_formula)
        ab, ci, multistep_stages, EPS_PRED = get_rk_methods_coeff(rk_type, h, c1, c2, c3, h_prev, h_prev2, _, sigmas)
        order = len(ci)-1
        
        if exp_mode:
            for i in range(order):
                for j in range(order):
                    ab[i][j] = ab[i][j] * phi(1, -h * ci[i+1])
        
        if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard" and sigma_next > 0.0:
            noise = noise_sampler(sigma=sigmas[_], sigma_next=sigmas[_+1])
            noise = torch.nan_to_num((noise - noise.mean()) / noise.std(), 0.0)
            xi[0] = alpha_ratio * xi[0] + noise * s_noise * sigma_up

        xi_0 = xi[0] # needed for implicit sampling

        if (MULTISTEP == False and FSAL == False) or _ == 0:
            ki[0]   = model_call(model, xi_0, sigma, **extra_args)
            if EPS_PRED and rk_type.startswith("deis"):
                ki[0] = (xi_0 - ki[0]) / sigma
                ki[0] = ki[0] * (sigma_down-sigma)/(sigma_next-sigma)
            ki_u[0] = uncond[0]

        if cfgpp != 0.0:
            ki[0] = uncond[0] + cfgpp * (ki[0] - uncond[0])
        ki_u[0] = uncond[0]

        for iteration in range(implicit_steps+1):
            for i in range(multistep_stages, order):
                if implicit_steps > 0 and iteration > 0 and implicit_sampler_name != "default":
                    ab, ci, multistep_stages, EPS_PRED = get_rk_methods_coeff(implicit_sampler_name, h, c1, c2, c3, h_prev, h_prev2, _, sigmas)
                    order = len(ci)-1
                    if len(ki) < order + 1:
                        last_value_ki = ki[-1]
                        last_value_ki_u = ki_u[-1]
                        ki.extend(  [last_value_ki]   * ((order + 1) - len(ki)))
                        ki_u.extend([last_value_ki_u] * ((order + 1) - len(ki_u)))
                    if len(xi) < order + 2:
                        xi.extend([torch.zeros_like(xi[0])] * ((order + 2) - len(xi)))
                    
                    ki[0]   = model_call(model, xi_0, sigma, **extra_args)
                    ki_u[0] = uncond[0]
                
                sigma_mid = sigma_fn(t + h*ci[i+1])
                alpha_t_1 = alpha_t_1_inv = torch.exp(torch.log(sigma_down/sigma) * ci[i+1] )
                if sigma_next > sigma:
                    alpha_t_1_inv = torch.nan_to_num(   torch.exp(torch.log((sigmax - sigma_down)/(sigmax - sigma)) * ci[i+1]),    1.)
                
                if LGW_MASK_RESCALE_MIN: 
                    lgw_mask = mask * (1 - latent_guide_weights[_]) + latent_guide_weights[_]
                    lgw_mask_inv = (1-mask) * (1 - latent_guide_weights[_]) + latent_guide_weights[_]
                else:
                    lgw_mask = mask * latent_guide_weights[_]    
                    lgw_mask_inv = (1-mask) * latent_guide_weights[_]   
                    
                ks, ks_u, ys, ys_inv = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
                for j in range(order):
                    ks     += ab[i][j] * ki[j]
                    ks_u   += ab[i][j] * ki_u[j]
                    ys     += ab[i][j] * y0
                    ys_inv += ab[i][j] * y0_inv
                    
                if EPS_PRED and rk_type.startswith("deis"):
                    epsilon = (h * ks) / (sigma_down - sigma)       #xi[(i+1)%order]  = xi_0 + h*ks
                    ks = xi_0 - epsilon * sigma        # denoised
                else:
                    if implicit_sampler_name.startswith("lobatto") == False:
                        ks /= sum(ab[i])
                    elif iteration == 0:
                        ks /= sum(ab[i])
                
                if UNSAMPLE == False and latent_guide is not None and latent_guide_weights[_] > 0.0:
                    if guide_mode == "hard_light":
                        lg = latent_guide * sum(ab[i])
                        if EPS_PRED:
                            lg = (alpha_fn(-h*ci[i+1]) * xi[0] - latent_guide) / (sigma_fn(t + h*ci[i]) + 1e-8)
                        hard_light_blend_1 = hard_light_blend(lg, ks)
                        ks = (1 - lgw_mask) * ks   +   lgw_mask * hard_light_blend_1
                        
                    elif guide_mode == "mean_std":
                        ks2 = torch.zeros_like(x)
                        for n in range(latent_guide.shape[1]):
                            ks2[0][n] = (ks[0][n] - ks[0][n].mean()) / ks[0][n].std()
                            ks2[0][n] = (ks2[0][n] * latent_guide[0][n].std()) + latent_guide[0][n].mean()
                        ks = (1 - lgw_mask) * ks   +   lgw_mask * ks2
                        
                    elif guide_mode == "mean":
                        ks2 = torch.zeros_like(x)
                        for n in range(latent_guide.shape[1]):
                            ks2[0][n] = (ks[0][n] - ks[0][n].mean())
                            ks2[0][n] = (ks2[0][n]) + latent_guide[0][n].mean()
                        ks3 = torch.zeros_like(x)
                        
                        for n in range(latent_guide.shape[1]):
                            ks3[0][n] = (ks[0][n] - ks[0][n].mean())
                            ks3[0][n] = (ks3[0][n]) + latent_guide_inv[0][n].mean()
                        ks = (1 - lgw_mask) * ks   +   lgw_mask * ks2
                        ks = (1 - lgw_mask_inv) * ks   +   lgw_mask_inv * ks3
                        
                    elif guide_mode == "std":
                        ks2 = torch.zeros_like(x)
                        for n in range(latent_guide.shape[1]):
                            ks2[0][n] = (ks[0][n]) / ks[0][n].std()
                            ks2[0][n] = (ks2[0][n] * latent_guide[0][n].std())
                        ks = (1 - lgw_mask) * ks   +   lgw_mask * ks2
                        
                    elif guide_mode == "blend": 
                        ks = (1 - lgw_mask)     * ks   +   lgw_mask     * ys   #+   (1-lgw_mask) * latent_guide_inv
                        ks = (1 - lgw_mask_inv) * ks   +   lgw_mask_inv * ys_inv
                        
                    elif guide_mode == "inversion": 
                        UNSAMPLE = True

                cfgpp_term = cfgpp*h*(ks - ks_u)
                xi[(i+1)%order]  = (1-UNSAMPLE * lgw_mask) * (alpha_t_1     * (xi_0 + cfgpp_term)    +    (1 - alpha_t_1)     * ks )     \
                                    + UNSAMPLE * lgw_mask  * (alpha_t_1_inv * (xi_0 + cfgpp_term)    +    (1 - alpha_t_1_inv) * ys )
                if UNSAMPLE:
                    xi[(i+1)%order]  = (1-lgw_mask_inv) * xi[(i+1)%order]   + UNSAMPLE * lgw_mask_inv  * (alpha_t_1_inv * (xi_0 + cfgpp_term)    +      (1 - alpha_t_1_inv) * ys_inv )

                if (i+1)%order > 0 and (i+1)%order > multistep_stages-1:
                    if GARBAGE_COLLECT: gc.collect(); torch.cuda.empty_cache()
                    ki[i+1]   = model_call(model, xi[i+1], sigma_fn(t + h*ci[i+1]), **extra_args)
                    if EPS_PRED and rk_type.startswith("deis"):
                        ki[i+1] = (xi[i+1] - ki[i+1]) / sigma_fn(t + h*ci[i+1])
                        ki[i+1] = ki[i+1] * (sigma_down-sigma)/(sigma_next-sigma)
                    ki_u[i+1] = uncond[0]

            if FSAL and _ > 0:
                ki  [0] = ki[order-1]
                ki_u[0] = ki_u[order-1]
            if MULTISTEP and _ > 0:
                ki  [0] = denoised
                ki_u[0] = ki_u[order-1]
            for ms in range(multistep_stages):
                ki  [multistep_stages - ms] = ki  [multistep_stages - ms - 1]
                ki_u[multistep_stages - ms] = ki_u[multistep_stages - ms - 1]
            if iteration < implicit_steps and implicit_sampler_name == "default":
                ki  [0] = model_call(model, xi[0], sigma_down, **extra_args)
                ki_u[0] = uncond[0]
            elif iteration == implicit_steps and implicit_sampler_name != "default" and implicit_steps > 0:
                ks, ks_u, ys, ys_inv = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
                for j in range(order):
                    ks     += ab[i+1][j] * ki[j]
                    ks_u   += ab[i+1][j] * ki_u[j]
                    ys     += ab[i+1][j] * y0
                    ys_inv += ab[i+1][j] * y0_inv
                ks /= sum(ab[i+1])
                
                cfgpp_term = cfgpp*h*(ks - ks_u)  #GUIDES NOT FULLY IMPLEMENTED HERE WITH IMPLICIT FINAL STEP
                xi[(i+1)%order]  = (1-UNSAMPLE * lgw_mask) * (alpha_t_1     * (xi_0 + cfgpp_term)    +    (1 - alpha_t_1)     * ks )     \
                                    + UNSAMPLE * lgw_mask  * (alpha_t_1_inv * (xi_0 + cfgpp_term)    +    (1 - alpha_t_1_inv) * ys )
                if UNSAMPLE:
                    xi[(i+1)%order]  = (1-lgw_mask_inv) * xi[(i+1)%order]   + UNSAMPLE * lgw_mask_inv  * (alpha_t_1_inv * (xi_0 + cfgpp_term)    +      (1 - alpha_t_1_inv) * ys_inv )
                

            if EPS_PRED == True and exp_mode == False and not rk_type.startswith("deis"):
                denoised = alpha_fn(-h*ci[i+1]) * xi[0] - sigma * ks
            elif EPS_PRED == True and rk_type.startswith("deis"):
                epsilon = (h * ks) / (sigma_down - sigma)
                denoised = xi_0 - epsilon * sigma        # denoised
            elif iteration == implicit_steps and implicit_sampler_name != "default" and implicit_steps > 0:
                denoised = ks
            else:
                denoised = ks / sum(ab[i])
            
            """if iteration < implicit_steps and implicit_sampler_name != "default":
                for idx in range(len(ki)):
                        ki[idx] = denoised"""

        if callback is not None:
            callback({'x': xi[0], 'i': _, 'sigma': sigma, 'sigma_next': sigma_next, 'denoised': denoised})
            
        if (isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST) or noise_mode != "hard") and sigma_next > 0.0:
            noise = noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = (noise - noise.mean()) / noise.std()
            
            if guide_mode == "noise_mean":
                noise2 = torch.zeros_like(x)
                for n in range(latent_guide.shape[1]):
                    noise2[0][n] = (noise[0][n] - noise[0][n].mean())
                    noise2[0][n] = (noise2[0][n]) + latent_guide[0][n].mean()
                noise = (1 - lgw_mask) * noise   +   lgw_mask * noise2
            
            xi[0] = alpha_ratio * xi[0] + noise * s_noise * sigma_up
            
        h_prev2 = h_prev
        h_prev = h
        
    return xi[0]

