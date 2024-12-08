import torch
import math
from typing import Optional


# Remainder solution
def _phi(j, neg_h):
    remainder = torch.zeros_like(neg_h)
    
    for k in range(j): 
        remainder += (neg_h)**k / math.factorial(k)
    phi_j_h = ((neg_h).exp() - remainder) / (neg_h)**j
    
    return phi_j_h
  
def calculate_gamma(c2, c3):
    return (3*(c3**3) - 2*c3) / (c2*(2 - 3*c2))

# Exact analytic solution originally calculated by Clybius. https://github.com/Clybius/ComfyUI-Extra-Samplers/tree/main
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


class Phi:
    def __init__(self, h, c, analytic_solution=True): 
        self.h = h
        self.c = c
        if analytic_solution:
            self.phi_f = phi
        else:
            self.phi_f = _phi #remainder method
    
    def __call__(self, j, i=-1):
        if i < 0:
            c = 1
        else:
            c = self.c[i-1]
            if c == 0:
                return 0
        return self.phi_f(j, -self.h * c)
        
