import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi
import gc
import math
import copy


import torch.nn.functional as F
import torchvision.transforms as T

import functools

from .noise_classes import *

import comfy.model_patcher
import comfy.supported_models
from .extra_samplers_helpers import get_deis_coeff_list


from .noise_sigmas_timesteps_scaling import get_res4lyf_step_with_model, get_res4lyf_half_step3


def _phi(j, neg_h):
  remainder = torch.zeros_like(neg_h)
  
  for k in range(j): 
    remainder += (neg_h)**k / math.factorial(k)
  phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
  
  return phi_j_h
  
  
def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))


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
    ],
    [
        
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
        ],
        [    
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
        ],
        [
            [5/18, 4/9, 5/18]
        ],
        [1/2 - 15**0.5 / 10, 1/2, 1/2 + 15**0.5 / 10]
    ),
    "gauss-legendre_2s": (
        [
            [1/4, 1/4 - 3**0.5 / 6],
            [1/4 + 3**0.5 / 6, 1/4],
        ],
        [
            [1/2, 1/2],
        ],
        [1/2 - 3**0.5 / 6, 1/2 + 3**0.5 / 6]
    ),
    "radau_iia_3s": (
        [    
            [11/45 - 7*6**0.5 / 360, 37/225 - 169*6**0.5 / 1800, -2/225 + 6**0.5 / 75],
            [37/225 + 169*6**0.5 / 1800, 11/45 + 7*6**0.5 / 360, -2/225 - 6**0.5 / 75],
            [4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9],
        ],
        [
            [4/9 - 6**0.5 / 36, 4/9 + 6**0.5 / 36, 1/9],
        ],
        [2/5 - 6**0.5 / 10, 2/5 + 6**0.5 / 10, 1.]
    ),
    "radau_iia_2s": (
        [    
            [5/12, -1/12],
            [3/4, 1/4],
        ],
        [
            [3/4, 1/4],
        ],
        [1/3, 1]
    ),
    "lobatto_iiic_3s": (
        [    
            [1/6, -1/3, 1/6],
            [1/6, 5/12, -1/12],
            [1/6, 2/3, 1/6],
        ],
        [
            [1/6, 2/3, 1/6],
        ],
        [0, 1/2, 1]
    ),
    "lobatto_iiic_2s": (
        [    
            [1/2, -1/2],
            [1/2, 1/2],
        ],
        [
            [1/2, 1/2],
        ],
        [0, 1]
    ),
    
    "crouzeix_2s": (
        [
            [1/2 + 3**0.5 / 6, 0],
            [-(3**0.5 / 3), 1/2 + 3**0.5 / 6]
        ],
        [
            [1/2, 1/2],
        ],
        [1/2 + 3**0.5 / 6, 1/2 - 3**0.5 / 6],
    ),
    
    
    "dormand-prince_13s": (
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        ],
        [
            [14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4],
        ],
        [0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248 / 9719169821, 13/20, 1201146811 / 1299019798, 1, 1],
    ),
    "dormand-prince_6s": (
        [
            [0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        ],
        [
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
        ],
        [0, 1/5, 3/10, 4/5, 8/9, 1],
    ),
    "bogacki-shampine_7s": ( #5th order
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1/6, 0, 0, 0, 0, 0, 0],
            [2/27, 4/27, 0, 0, 0, 0, 0],
            [183/1372, -162/343, 1053/1372, 0, 0, 0, 0],
            [68/297, -4/11, 42/143, 1960/3861, 0, 0, 0],
            [597/22528, 81/352, 63099/585728, 58653/366080, 4617/20480, 0, 0],
            [174197/959244, -30942/79937, 8152137/19744439, 666106/1039181, -29421/29068, 482048/414219, 0],
        ],
        [
            [587/8064, 0, 4440339/15491840, 24353/124800, 387/44800, 2152/5985, 7267/94080],
        ],
        [0, 1/6, 2/9, 3/7, 2/3, 3/4, 1] 
    ),
    "rk4_4s": (
        [
            [0, 0, 0, 0],
            [1/2, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 1, 0],
        ],
        [
            [1/6, 1/3, 1/3, 1/6],
        ],
        [0, 1/2, 1/2, 1],
    ),
    "rk38_4s": (
        [
            [0, 0, 0, 0],
            [1/3, 0, 0, 0],
            [-1/3, 1, 0, 0],
            [1, -1, 1, 0],
        ],
        [
            [1/8, 3/8, 3/8, 1/8],
        ],
        [0, 1/3, 2/3, 1],
    ),
    "ralston_4s": (
        [
            [0, 0, 0, 0],
            [2/5, 0, 0, 0],
            [(-2889+1428 * 5**0.5)/1024,   (3785-1620 * 5**0.5)/1024,  0, 0],
            [(-3365+2094 * 5**0.5)/6040,   (-975-3046 * 5**0.5)/2552,  (467040+203968*5**0.5)/240845, 0],
        ],
        [
            [(263+24*5**0.5)/1812, (125-1000*5**0.5)/3828, (3426304+1661952*5**0.5)/5924787, (30-4*5**0.5)/123],
        ],
        [0, 2/5, (14-3 * 5**0.5)/16, 1],
    ),
    "heun_3s": (
        [
            [0, 0, 0],
            [1/3, 0, 0],
            [0, 2/3, 0],
        ],
        [
            [1/4, 0, 3/4],
        ],
        [0, 1/3, 2/3],
    ),
    "kutta_3s": (
        [
            [0, 0, 0],
            [1/2, 0, 0],
            [-1, 2, 0],
        ],
        [
            [1/6, 2/3, 1/6],
        ],
        [0, 1/2, 1],
    ),
    "ralston_3s": (
        [
            [0, 0, 0],
            [1/2, 0, 0],
            [0, 3/4, 0],
        ],
        [
            [2/9, 1/3, 4/9],
        ],
        [0, 1/2, 3/4],
    ),
    "houwen-wray_3s": (
        [
            [0, 0, 0],
            [8/15, 0, 0],
            [1/4, 5/12, 0],
        ],
        [
            [1/4, 0, 3/4],
        ],
        [0, 8/15, 2/3],
    ),
    "ssprk3_3s": (
        [
            [0, 0, 0],
            [1, 0, 0],
            [1/4, 1/4, 0],
        ],
        [
            [1/6, 1/6, 2/3],
        ],
        [0, 1, 1/2],
    ),
    "midpoint_2s": (
        [
            [0, 0],
            [1/2, 0],
        ],
        [
            [0, 1],
        ],
        [0, 1/2],
    ),
    "heun_2s": (
        [
            [0, 0],
            [1, 0],
        ],
        [
            [1/2, 1/2],
        ],
        [0, 1],
    ),
    "ralston_2s": (
        [
            [0, 0],
            [2/3, 0],
        ],
        [
            [1/4, 3/4],
        ],
        [0, 2/3],
    ),
    "euler": (
        [
            [0],
        ],
        [
            [1],
        ],
        [0],
    ),
}


def get_rk_methods(rk_type, h, c1=0.0, c2=0.5, c3=1.0, h_prev=None, h_prev2=None, stepcount=0, sigmas=None, sigma=None, sigma_next=None, sigma_down=None):
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
        a, b, ci = copy.deepcopy(rk_coeff[rk_type])

    match rk_type:
        case "deis": 
            coeff_list = get_deis_coeff_list(sigmas, multistep_stages+1, deis_mode="rhoab")
            coeff_list = [[elem / h for elem in inner_list] for inner_list in coeff_list]
            if multistep_stages == 1:
                b1, b2 = coeff_list[stepcount]
                a = [
                        [0, 0],
                        [0, 0],
                ]
                b = [
                        [b1, b2],
                ]
                ci = [0, 0]
            if multistep_stages == 2:
                b1, b2, b3 = coeff_list[stepcount]
                a = [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                ]
                b = [
                        [b1, b2, b3],
                ]
                ci = [0, 0, 0]
            if multistep_stages == 3:
                b1, b2, b3, b4 = coeff_list[stepcount]
                a = [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                ]
                b = [
                    [b1, b2, b3, b4],
                ]
                ci = [0, 0, 0, 0]
            if multistep_stages > 0:
                for i in range(len(b[0])): 
                    b[0][i] *= ((sigma_down - sigma) / (sigma_next - sigma))

        case "dormand-prince_6s":
            FSAL = True

        case "ddim":
            b1 = phi(1, -h)
            a = [
                    [0],
            ]
            b = [
                    [b1],
            ]
            ci = [0]

        case "res_2s":
            a2_1 = c2 * phi(1, -h*c2)
            b1 =        phi(1, -h) - phi(2, -h)/c2
            b2 =        phi(2, -h)/c2

            a = [
                    [0,0],
                    [a2_1, 0],
            ]
            b = [
                    [b1, b2],
            ]
            ci = [0, c2]

        case "res_3s":
            gamma = calculate_gamma(c2, c3)
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = gamma * c2 * phi(2, -h*c2) + (c3 ** 2 / c2) * phi(2, -h*c3) #phi_2_c3_h  # a32 from k2 to k3
            a3_1 = c3 * phi(1, -h*c3) - a3_2 # a31 from k1 to k3
            b3 = (1 / (gamma * c2 + c3)) * phi(2, -h)      
            b2 = gamma * b3  #simplified version of: b2 = (gamma / (gamma * c2 + c3)) * phi_2_h  
            b1 = phi(1, -h) - b2 - b3     
            
            a = [
                    [0, 0, 0],
                    [a2_1, 0, 0],
                    [a3_1, a3_2, 0],
            ]
            b = [
                    [b1, b2, b3],
            ]
            ci = [c1, c2, c3]

        case "dpmpp_2s":
            a2_1 =         c2   * phi(1, -h*c2)
            b1 = (1 - 1/(2*c2)) * phi(1, -h)
            b2 =     (1/(2*c2)) * phi(1, -h)

            a = [
                    [0, 0],
                    [a2_1, 0],
            ]
            b = [
                    [b1, b2],
            ]
            ci = [0, c2]
            
        case "dpmpp_sde_2s":
            c2 = 1.0 #hardcoded to 1.0 to more closely emulate the configuration for k-diffusion's implementation
            a2_1 =         c2   * phi(1, -h*c2)
            b1 = (1 - 1/(2*c2)) * phi(1, -h)
            b2 =     (1/(2*c2)) * phi(1, -h)

            a = [
                    [0, 0],
                    [a2_1, 0],
            ]
            b = [
                    [b1, b2],
            ]
            ci = [0, c2]

        case "dpmpp_3s":
            a2_1 = c2 * phi(1, -h*c2)
            a3_2 = (c3**2 / c2) * phi(2, -h*c3)
            a3_1 = c3 * phi(1, -h*c3) - a3_2
            b2 = 0
            b3 = (1/c3) * phi(2, -h)
            b1 = phi(1, -h) - b2 - b3

            a = [
                    [0, 0, 0],
                    [a2_1, 0, 0],
                    [a3_1, a3_2, 0],  
            ]
            b = [
                    [b1, b2, b3],
            ]
            ci = [0, c2, c3]
            
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

            a = [
                    [0, 0, 0, 0, 0],
                    [a2_1, 0, 0, 0, 0],
                    [a3_1, a3_2, 0, 0, 0],
                    [a4_1, a4_2, a4_3, 0, 0],
                    [a5_1, a5_2, a5_3, a5_4, 0],
            ]
            b = [
                    [b1, b2, b3, b4, b5],
            ]
            ci = [0., 0.5, 0.5, 1., 0.5]

    ci = ci[:]
    if rk_type.startswith("lob") == False:
        ci.append(1)
    return a, b, ci, multistep_stages, FSAL

from typing import Protocol, Optional, Dict, Any, TypedDict, NamedTuple


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




class RK_Method:
    def __init__(self, model, name="", method="explicit", dynamic_method=False, device='cuda', dtype=torch.float64):
        self.model = model
        self.model_sampling = model.inner_model.inner_model.model_sampling
        self.device = device
        self.dtype = dtype
        
        self.method = method
        self.dynamic_method = dynamic_method
        
        self.stages = 0
        self.name = name
        self.ab = None
        self.a = None
        self.b = None
        self.c = None
        self.denoised = None
        self.uncond = None
        
        self.rows = 0
        self.cols = 0
        
        self.y0 = None
        self.y0_inv = None
        
        self.sigma_min = model.inner_model.inner_model.model_sampling.sigma_min.to(dtype)
        self.sigma_max = model.inner_model.inner_model.model_sampling.sigma_max.to(dtype)
        
        self.noise_sampler = None
        
        self.h_prev = None
        self.h_prev2 = None
        self.multistep_stages = 0
        
        #self.UNSAMPLE = False
        
    def __call__(self):
        raise NotImplementedError("This method got clownsharked!")
    
    def model_epsilon(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        x0 = self.model(x, sigma * s_in, **extra_args)
        #return x0 ###################################THIS WORKS ONLY WITH THE MODEL SAMPLING PATCH
        eps = (x - x0) / (sigma * s_in) 
        return eps, x0
    
    def model_denoised(self, x, sigma, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        x0 = self.model(x, sigma * s_in, **extra_args)
        return x0
    
    @staticmethod
    def phi(j, neg_h):
        remainder = torch.zeros_like(neg_h)
        for k in range(j): 
            remainder += (neg_h)**k / math.factorial(k)
        phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
        return phi_j_h
    
    @staticmethod
    def calculate_gamma(c2, c3):
        return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))
    
    def init_noise_sampler(self, x, noise_seed, noise_sampler_type, alpha, k=1., scale=0.1):
        seed = torch.initial_seed()+1 if noise_seed == -1 else noise_seed
        if noise_sampler_type == "fractal":
            self.noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            self.noise_sampler.alpha = alpha
            self.noise_sampler.k = k
            self.noise_sampler.scale = scale
        else:
            self.noise_sampler = NOISE_GENERATOR_CLASSES_SIMPLE.get(noise_sampler_type)(x=x, seed=seed, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            
    def add_noise_pre(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None):
        if isinstance(self.model_sampling, comfy.model_sampling.CONST) == False and noise_mode == "hard":
            return self.add_noise(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
        
    def add_noise_post(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL=False, sde_noise_t=None):
        if isinstance(self.model_sampling, comfy.model_sampling.CONST) == True  or (isinstance(self.model_sampling, comfy.model_sampling.CONST) == False and noise_mode != "hard"):
            return self.add_noise(x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t)
        else:
            return x
    
    """   def add_noise(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t):
        noise = self.noise_sampler(sigma=sigma, sigma_next=sigma_next)
        noise = torch.nan_to_num((noise - noise.mean()) / noise.std(), 0.0)
        
        return alpha_ratio * x + noise * s_noise * sigma_up"""

    
    
    
    #def add_noise_orig(self, x, y0, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t):
    
    def add_noise(self, x, y0, lgw, sigma_up, sigma, sigma_next, sigma_down, alpha_ratio, s_noise, SDE_NOISE_EXTERNAL, sde_noise_t):
        
        
        if sigma_next > 0.0:

            noise = self.noise_sampler(sigma=sigma, sigma_next=sigma_next)
            noise = torch.nan_to_num((noise - noise.mean()) / noise.std(), 0.0)
            
            #cvf = self.get_epsilon(x, x, y0, sigma, sigma_up, sigma_down, None)
            #for i in range(cvf.shape[1]):
            #    cvf[0][i] = (cvf[0][i] - cvf[0][i].mean()) / cvf[0][i].std()
            #noise = noise + lgw * (cvf - noise)
            
            #x = (1 - s_noise) * x + s_noise * (y0 + sigma_down * noise)
            if SDE_NOISE_EXTERNAL:
                noise = (1-s_noise) * noise + s_noise * sde_noise_t
            
            return alpha_ratio * x + noise * sigma_up  #removed s_noise for hack use
        
            #return alpha_ratio * x + noise * s_noise * sigma_up
        else:
            return x
    
    def ab_sum(self, ab, row, columns, ki, ki_u, y0, y0_inv):
        ks, ks_u, ys, ys_inv = torch.zeros_like(ki[0]), torch.zeros_like(ki[0]), torch.zeros_like(ki[0]), torch.zeros_like(ki[0])
        for col in range(columns):
            ks     += ab[row][col] * ki  [col]
            ks_u   += ab[row][col] * ki_u[col]
            ys     += ab[row][col] * y0
            ys_inv += ab[row][col] * y0_inv
        return ks, ks_u, ys, ys_inv
    
    def prepare_sigmas(self, sigmas):
        if sigmas[0] == 0.0:      #remove padding used to avoid need for model patch with noise inversion
            UNSAMPLE = True
            sigmas = sigmas[1:-1]
        else: 
            UNSAMPLE = False
            
        if hasattr(self.model, "sigmas"):
            self.model.sigmas = sigmas
            
        return sigmas, UNSAMPLE
    
    def prepare_mask(self, x, mask, LGW_MASK_RESCALE_MIN):
        if mask is None:
            mask = torch.ones_like(x)
            LGW_MASK_RESCALE_MIN = False
        else:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, x.shape[1], 1, 1) 
            mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            mask = mask.to(x.dtype).to(x.device)
        return mask, LGW_MASK_RESCALE_MIN
        
    def prepare_weighted_mask(self, mask, weight, LGW_MASK_RESCALE_MIN):
        if LGW_MASK_RESCALE_MIN: 
            lgw_mask = mask * (1 - weight) + weight
            lgw_mask_inv = (1-mask) * (1 - weight) + weight
        else:
            lgw_mask = mask * weight
            lgw_mask_inv = (1-mask) * weight
        return lgw_mask, lgw_mask_inv
    
    def set_coeff(self, rk_type, h, c1=0.0, c2=0.5, c3=1.0, stepcount=0, sigmas=None, sigma=None, sigma_down=None):
        if rk_type == "default": 
            return
        #if self.a is None or self.dynamic_method == True:
        sigma = sigmas[stepcount]
        sigma_next = sigmas[stepcount+1]
        
        a, b, ci, multistep_stages, FSAL = get_rk_methods(rk_type, h, c1, c2, c3, self.h_prev, self.h_prev2, stepcount, sigmas, sigma, sigma_next, sigma_down)
        
        self.multistep_stages = multistep_stages
        
        self.a = torch.tensor(a, dtype=h.dtype, device=h.device)
        self.a = self.a.view(*self.a.shape, 1, 1, 1, 1)
        
        self.b = torch.tensor(b, dtype=h.dtype, device=h.device)
        self.b = self.b.view(*self.b.shape, 1, 1, 1, 1)
        
        self.c = torch.tensor(ci, dtype=h.dtype, device=h.device)
        self.rows = self.a.shape[0]
        self.cols = self.a.shape[1]
            
    def a_k_sum(self, k, row):
        if len(k.shape) == 4:
            ks = k * self.a[row].sum(dim=0)
        ks = (k[0:self.cols] * self.a[row]).sum(dim=0)
        return ks
    
    def b_k_sum(self, k, row):
        if len(k.shape) == 4:
            ks = k * self.b[row].sum(dim=0)
        ks = (k[0:self.cols] * self.b[row]).sum(dim=0)
        return ks



    def init_guides(self, x, latent_guide, latent_guide_inv, mask, sigmas, UNSAMPLE):
        y0, y0_inv = torch.zeros_like(x), torch.zeros_like(x)
        
        if latent_guide is not None:
            if sigmas[0] > sigmas[1]:
                y0 = latent_guide = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
            elif UNSAMPLE and mask is not None:
                x = (1-mask) * x + mask * self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)
            else:
                x = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(x.device)

        if latent_guide_inv is not None:
            if sigmas[0] > sigmas[1]:
                y0_inv = latent_guide_inv = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)
            elif UNSAMPLE and mask is not None:
                x = mask * x + (1-mask) * self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)
            else:
                x = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_INV AFTER SETTING TO LG above!
                
        if UNSAMPLE and sigmas[0] < sigmas[1]: #sigma_next > sigma:
            y0 = self.noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min)
            y0 = (y0 - y0.mean()) / y0.std()
            y0_inv = self.noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min)
            y0_inv = (y0_inv - y0_inv.mean()) / y0_inv.std()
            #x = (x - x.mean()) / x.std()
            
        return x, y0, y0_inv
    
    
    
    def init_cfgpp(self, model, x, cfgpp, extra_args):
        self.uncond = [torch.full_like(x, 0.0)]
        if cfgpp != 0.0:
            def post_cfg_function(args):
                self.uncond[0] = args["uncond_denoised"]
                return args["denoised"]
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)
    
    
    
    def process_guides(self, guide_mode, latent_guide, latent_guide_inv, lgw_mask, lgw_mask_inv, weight, sigma_mid, EPS_PRED, UNSAMPLE):
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
        


class RK_Method_Exponential(RK_Method):
    def __init__(self, model, name="", method="explicit", device='cuda', dtype=torch.float64):
        super().__init__(model, name, method, device, dtype) 
        self.exponential = True
        self.eps_pred = True
        
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

    def __call__(self, x_0, x, sigma, h, **extra_args):

        denoised = self.model_denoised(x, sigma, **extra_args)
        epsilon = denoised - x_0
        
        if self.uncond == None:
            self.uncond = torch.zeros_like(x)
        denoised_u = self.uncond.clone()
        if torch.all(denoised_u == 0):
            epsilon_u = torch.zeros_like(x_0)
        else:
            epsilon_u = denoised_u - x_0
            
        self.h_prev2 = self.h_prev
        self.h_prev = h
        return epsilon, denoised
    
    def data_to_vel(self, x, data, sigma):
        return data - x
    
    def get_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, t_i=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = t_i if t_i is not None else sigma_cur

        if sigma_down is None:
            return y - x_0
        else:
            if sigma_down > sigma:
                return (x_0 - y) * sigma_cur
            else:
                return (y - x_0) * sigma_cur
            
    
class RK_Method_Linear(RK_Method):
    def __init__(self, model, name="", method="explicit", device='cuda', dtype=torch.float64):
        super().__init__(model, name, method, device, dtype) 
        self.expanential = False
        self.eps_pred = True
        
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
    
    def __call__(self, x_0, x, sigma, h, **extra_args):
        s_in = x.new_ones([x.shape[0]])
        
        epsilon, denoised = self.model_epsilon(x, sigma, **extra_args)
        #denoised = x - sigma * epsilon
        
        if self.uncond == None:
            self.uncond = torch.zeros_like(x)
        denoised_u = self.uncond.clone()
        if torch.all(denoised_u == 0):
            epsilon_u = torch.zeros_like(x_0)
        else:
            epsilon_u  = (x_0 - denoised_u) / (sigma * s_in)
            
        self.h_prev2 = self.h_prev
        self.h_prev = h
        return epsilon, denoised

    def data_to_vel(self, x, data, sigma):
        return (data - x) / sigma
    
    def get_epsilon(self, x_0, x, y, sigma, sigma_cur, sigma_down=None, t_i=None):
        if sigma_down > sigma:
            sigma_cur = self.sigma_max - sigma_cur.clone()
        sigma_cur = t_i if t_i is not None else sigma_cur

        if sigma_down is None:
            return (x - y) / sigma_cur
        else:
            if sigma_down > sigma:
                return (y - x) / sigma_cur
            else:
                return (x - y) / sigma_cur
    
    



