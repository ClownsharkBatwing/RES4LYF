import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi

from comfy.k_diffusion.sampling import get_ancestral_step, to_d


import functools

from .noise_classes import *
#from comfy_extras.nodes_advanced_samplers import sample_euler_cfgpp_alt

@cast_fp64
@torch.no_grad()
def sample_dpmpp_sde_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    eta=1., s_noise=1., noise_sampler=None, r=1/2, k=1.0, scale=0.1, noise_sampler_type="brownian", alpha: FloatTensor = torch.zeros((1,))
):
    #DPM-Solver++ (stochastic with eta parameter).
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        x_hat = x
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d

            d = to_d(x_hat, sigmas[i], x)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt

            x = x + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * s_noise * su

    return x

@cast_fp64
@torch.no_grad()
def sample_dpmpp_sde_cfgpp_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eulers_mom=None,
    eta=1., s_noise=1., noise_sampler=None, r=1/2, k=1.0, scale=0.1, noise_sampler_type="brownian", cfgpp: FloatTensor = torch.zeros((1,)), alpha: FloatTensor = torch.zeros((1,))
):
    #DPM-Solver++ (stochastic with eta parameter).
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args

    #import pdb; pdb.set_trace()
    if cfgpp.sum().item() != 0.0:
        temp = [0]
        def post_cfg_function(args):
            temp[0] = args["uncond_denoised"]
            return args["denoised"]

        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        x_hat = x
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

        if sigmas[i + 1] == 0:
            # Euler method
            if cfgpp.sum().item() == 0.0:
                d = to_d(x, sigmas[i], denoised)
            else:
                d = to_d(x - cfgpp[i]*denoised + cfgpp[i]*temp[0], sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            if cfgpp.sum().item() == 0.0:
                x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            else:
                x_2 = (sigma_fn(s_) / sigma_fn(t)) * (x + cfgpp[i]*(denoised - temp[0])) - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            if cfgpp.sum().item() == 0.0:
                x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            else:
                x = (sigma_fn(t_next_) / sigma_fn(t)) * (x + cfgpp[i]*(denoised - temp[0])) - (t - t_next_).expm1() * denoised_d

            if eulers_mom is not None:
                d = to_d(x_hat, sigmas[i], x)
                dt = sigmas[i + 1] - sigmas[i]
                x = x + eulers_mom[i].item() * d * dt

            x = x + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * s_noise * su
    return x

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

@cast_fp64
def sample_dpmpp_dualsdemomentum_advanced(model, x, sigmas, seed=42, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, r=1/2, momentum=0.0, momentums=None, etas=None, s_noises=None,rs=None,scheduled_r=False):
    return sample_dpmpp_dualsde_momentum_advanced(model, x, sigmas, seed=seed, extra_args=extra_args, callback=callback, disable=disable, eta=etas, s_noise=s_noises, noise_sampler_type=noise_sampler_type, noise_sampler=noise_sampler, r=rs, momentum=momentums, scheduled_r=False)

@cast_fp64
@torch.no_grad()
def sample_dpmpp_dualsde_momentum_advanced (
    model, 
    x, 
    sigmas, 
    seed=42,
    extra_args=None, 
    callback=None, 
    disable=None,
    noise_sampler=None, 
    noise_sampler_type=None,
    momentum: FloatTensor = torch.zeros((1,)),
    eta: FloatTensor = torch.zeros((1,)),
    s_noise: FloatTensor = torch.zeros((1,)),
    r: FloatTensor = torch.zeros((1,)),
    scheduled_r=False
):
    """DPM-Solver++ (Stochastic with Momentum). Personal modified sampler by Clybius"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    denoisedsde_1, denoisedsde_2, denoisedsde_3 = None, None, None
    h_1, h_2, h_3 = None, None, None

    def momentum_func(diff, velocity, timescale=1.0, current_momentum=0): # Diff is current diff, vel is previous diff
        offset=-current_momentum / 2.0
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = current_momentum * (timescale + offset) * velocity + (1 - current_momentum * (timescale + offset)) * diff
        return momentum_vel

    vel = None
    vel_2 = None
    vel_sde = None

    current_r = r[0]

    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max

        current_momentum = momentum[i]
        current_eta = eta[i]
        current_s_noise = s_noise[i]
        if scheduled_r == True:
            current_r = r[i]

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            h_eta = h * (current_eta + 1)
            s = t + h * current_r
            fac = 1 / (2 * current_r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), current_eta)
            s_ = t_fn(sd)
            diff_2 = momentum_func((t - s_).expm1() * denoised, vel_2, time, current_momentum)
            vel_2 = diff_2
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - diff_2
            x_2 = x_2 + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(s)) * current_s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), current_eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            diff = momentum_func((t - t_next_).expm1() * denoised_d, vel, time, current_momentum)
            vel = diff
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - diff

            if h_3 is not None:
                r0 = h_3 / h_2
                r1 = h_2 / h
                r2 = h / h_1
                d1_0 = (denoised_d - denoisedsde_1) / r2
                d1_1 = (denoisedsde_1 - denoisedsde_2) / r1
                d1_2 = (denoisedsde_2 - denoisedsde_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time, current_momentum)
                vel_sde = diff
                x = x + diff
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised_d - denoisedsde_1) / r0
                d1_1 = (denoisedsde_1 - denoisedsde_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time, current_momentum)
                vel_sde = diff
                x = x + diff
            elif h_1 is not None:
                current_r = h_1 / h
                d = (denoised_d - denoisedsde_1) / current_r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = diff
                x = x + diff

            if current_eta:
                x = x + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * current_s_noise * su
            #if 'denoised_d' in locals():
            denoisedsde_1, denoisedsde_2, denoisedsde_3 = denoised_d, denoisedsde_1, denoisedsde_2 
            #if 'h' in locals():
            h_1, h_2, h_3 = h, h_1, h_2
    return x

# Many thanks to Kat + Birch-San for this wonderful sampler implementation! https://github.com/Birch-san/sdxl-play/commits/res/
from .refined_exp_solver import sample_refined_exp_s_advanced

@cast_fp64
def sample_res_solver_advanced(model, 
                               x, 
                               sigmas, etas, c2s, momentums, eulers_moms, offsets, branch_mode, branch_depth, branch_width,
                               guides_1, guides_2, latent_guide_1, latent_guide_2, guide_mode_1, guide_mode_2, guide_1_channels,
                               k, clownseed=0, cfgpps=0.0, alphas=None, latent_noise=None, latent_self_guide_1=False,latent_shift_guide_1=False,
                               extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_sampler=None, denoise_to_zero=True, simple_phi_calc=False, c2=0.5, momentum=0.0, eulers_mom=0.0, offset=0.0):
    return sample_refined_exp_s_advanced(
        model=model, 
        x=x, 
        clownseed=clownseed,
        sigmas=sigmas, 
        branch_mode=branch_mode,
        branch_depth=branch_depth,
        branch_width=branch_width,
        latent_guide_1=latent_guide_1,
        latent_guide_2=latent_guide_2,
        guide_1=guides_1,
        guide_2=guides_2,
        guide_mode_1=guide_mode_1,
        guide_mode_2=guide_mode_2,
        guide_1_channels=guide_1_channels,
        extra_args=extra_args, 
        callback=callback, 
        disable=disable, 
        noise_sampler=noise_sampler,
        denoise_to_zero=denoise_to_zero, 
        simple_phi_calc=simple_phi_calc, 
        cfgpp=cfgpps,
        c2=c2s, 
        eta=etas,
        momentum=momentums,
        eulers_mom=eulers_moms,
        offset=offsets,
        alpha=alphas,
        noise_sampler_type=noise_sampler_type,
        k=k,
        latent_noise=latent_noise,
        latent_self_guide_1=latent_self_guide_1,
        latent_shift_guide_1=latent_shift_guide_1
    )


from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x - denoised + temp[0], sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic) with post-configuration."""
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x - denoised + temp[0], sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * (x + (denoised - temp[0])) - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * (x + (denoised - temp[0])) - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M) with post-configuration."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - temp[0])) - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE with post-configuration."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * (x + (denoised - temp[0])) + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x
@torch.no_grad()
def sample_dpmpp_3m_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE with post-configuration."""

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * (x + (denoised - temp[0])) + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

@torch.no_grad()
def sample_euler_ancestral_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="pyramid-cascade_B", ):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        #if noise_sampler_type == "fractal":
        #    noise_sampler.alpha = alpha[i]
        #    noise_sampler.k = k
        #    noise_sampler.scale = scale

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="pyramid-cascade_B", ):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        #if noise_sampler_type == "fractal":
        #    noise_sampler.alpha = alpha[i]
        #    noise_sampler.k = k
        #    noise_sampler.scale = scale

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint', noise_sampler_type="pyramid-cascade_B", ):
    """DPM-Solver++(2M) SDE."""
    if len(sigmas) <= 1:
        return x

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None # step size

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    #       C2
                    x = x + 1.0 * ( (-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 *   (-h - eta_h).expm1().neg()                     * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="pyramid-cascade_B", ):
    """DPM-Solver++(3M) SDE."""

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x

def sample_rk4(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements the fourth-order Runge-Kutta method."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # Step size
        dt = sigma_next - sigma
        
        # Runge-Kutta 4th order calculations
        k1 = to_d(x, sigma, model(x, sigma * s_in, **extra_args))
        k2 = to_d(x + 0.5 * dt * k1, sigma + 0.5 * dt, model(x + 0.5 * dt * k1, (sigma + 0.5 * dt) * s_in, **extra_args))
        k3 = to_d(x + 0.5 * dt * k2, sigma + 0.5 * dt, model(x + 0.5 * dt * k2, (sigma + 0.5 * dt) * s_in, **extra_args))
        k4 = to_d(x + dt * k3, sigma_next, model(x + dt * k3, sigma_next * s_in, **extra_args))
        
        # Combine steps
        dx = (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Update x
        x = x + dx * dt
        
        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_next': sigma_next, 'dx': dx})
    
    return x

#sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
#sigmas = sigmas[-(steps + 1):]

@torch.no_grad()
#def sample_euler_ancestral_recursive(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
def sample_euler_ancestral_recursive(model, x, sigmas, i=0, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if noise_sampler is None:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed", None)
        noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    #for i in trange(len(sigmas) - 1, disable=disable):
    denoised = model(x, sigmas[i] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    d = to_d(x, sigmas[i], denoised)
    # Euler method
    dt = sigma_down - sigmas[i]
    x = x + d * dt
    if sigmas[i + 1] > 0:
        x = x + noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

    i += 1

    if i == len(sigmas) - 1:
        return x
    else:
        x = sample_euler_ancestral_recursive(model, x, sigmas, i=i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler)

    return x

#def sample_euler_ancestral_recursive_call(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", ):

@torch.no_grad()
#def sample_euler_ancestral_recursive(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
def sample_euler_ancestral_recursive_get2(model, x, sigmas, i=0, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler1=None, noise_sampler2=None, noise_sampler_type="gaussian", ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if noise_sampler1 is None:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed", None)
        noise_sampler1 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
        noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed+1000, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    #for i in trange(len(sigmas) - 1, disable=disable):

    #sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
    #sigma_hat = sigmas[i] * (1 + sigma_up)
    sigma_hat = sigmas[i] * 1.25

    if sigmas[i + 1] > 0:
        eps1 = noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1])
        eps2 = noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1])
        x1 = x + eps1 * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        x2 = x + eps2 * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
    else:
        return x

    denoised1 = model(x1, sigma_hat * s_in, **extra_args)
    denoised2 = model(x2, sigma_hat * s_in, **extra_args)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1})

    dt = sigmas[i+1] - sigmas[i]
    d1 = to_d(x, sigmas[i], denoised1)
    d2 = to_d(x, sigmas[i], denoised2)
    x1 = x + d1 * dt
    x2 = x + d2 * dt

    i += 1

    if i == len(sigmas) - 1:
        return x
    else:
        if torch.norm(x - x1) > torch.norm(x - x2):
            x_next = x1
        else:
            x_next = x2
        x = sample_euler_ancestral_recursive_get2(model, x_next, sigmas, i=i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler1=noise_sampler1, noise_sampler2=noise_sampler2)

    #depth -= 1
    return x


@torch.no_grad()
#def sample_euler_ancestral_recursive(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
def sample_euler_ancestral_recursive_depth_get2_call(model, x, sigmas, x_root=None, depth=2, i=0, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler1=None, noise_sampler2=None, noise_sampler_type="gaussian", ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if x_root is None:
        x_root = x

    if noise_sampler1 is None:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed", None)
        noise_sampler1 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
        noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed+1000, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    #for i in trange(len(sigmas) - 1, disable=disable):

    #sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
    #sigma_hat = sigmas[i] * (1 + sigma_up)
    sigma_hat = sigmas[i] * 1.25

    x = model(x, sigmas[0] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=1)
    sigmas = sigmas[1:]

    for i in trange(len(sigmas) - 1, disable=disable):
        if 2*i > len(sigmas) - 1:
            return x
        x = sample_euler_ancestral_recursive_depth_get2(model, x, sigmas, sigma_down, sigma_up, x_root=x, depth=2, i=2*i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler1=noise_sampler1, noise_sampler2=noise_sampler2)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=1)
    return x




@torch.no_grad()
#def sample_euler_ancestral_recursive(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
def sample_euler_ancestral_recursive_depth_get2(model, x, sigmas, sigma_down, sigma_up, x_root=None, depth=2, i=0, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler1=None, noise_sampler2=None, noise_sampler_type="gaussian", ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if x_root is None:
        x_root = x

    if noise_sampler1 is None:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed", None)
        noise_sampler1 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
        noise_sampler2 = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed+1000, sigma_min=sigma_min, sigma_max=sigma_max)
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    #for i in trange(len(sigmas) - 1, disable=disable):

    #sigma_hat = sigmas[i] * (1 + sigma_up)
    sigma_hat = sigmas[i] * 1.25

    if sigmas[i + 1] > 0:
        eps1 = noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1])
        eps2 = noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1])
        #x1 = x + eps1 * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        #x2 = x + eps2 * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        x1 = x + eps1 * sigma_up
        x2 = x + eps2 * sigma_up
    else:
        return x

    denoised1 = model(x1, sigmas[i] * s_in, **extra_args)
    denoised2 = model(x2, sigmas[i] * s_in, **extra_args)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised1})

    #dt = sigmas[i+1] - sigmas[i]
    dt = sigma_down - sigmas[i]
    d1 = to_d(x, sigmas[i], denoised1)
    d2 = to_d(x, sigmas[i], denoised2)
    x1 = x + d1 * dt
    x2 = x + d2 * dt

    i += 1
    depth -= 1

    if i < len(sigmas) - 1 and depth > 0:
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=1)
        x1 = sample_euler_ancestral_recursive_depth_get2(model, x1, sigmas, sigma_down, sigma_up, x_root=x_root, depth=depth, i=i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler1=noise_sampler1, noise_sampler2=noise_sampler2)
        x2 = sample_euler_ancestral_recursive_depth_get2(model, x2, sigmas, sigma_down, sigma_up, x_root=x_root, depth=depth, i=i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler1=noise_sampler1, noise_sampler2=noise_sampler2)

    if torch.norm(x_root - x1) > torch.norm(x_root - x2):
        return x1
    else:
        return x2
    
    if i == len(sigmas) - 1:
        return x
    else:
        if torch.norm(x - x1) > torch.norm(x - x2):
            x_next = x1
        else:
            x_next = x2
        x = sample_euler_ancestral_recursive_get2(model, x_next, sigmas, i=i, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler1=noise_sampler1, noise_sampler2=noise_sampler2)

    #depth -= 1
    return x


@torch.no_grad()
def sample_euler_ancestral2(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler1 = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed=seed+1000, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])

    i=0
    denoised = model(x, sigmas[i] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    d = to_d(x, sigmas[i], denoised)
    dt = sigma_down - sigmas[i]
    x = x + d * dt

    if sigmas[i + 1] > 0:
        x1 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        x2 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

    sigmas = sigmas[1:]

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised1 = model(x1, sigmas[i] * s_in, **extra_args)
        denoised2 = model(x2, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d1 = to_d(x1, sigmas[i], denoised1)
        dt = sigma_down - sigmas[i]
        x1 = x1 + d1 * dt

        d2 = to_d(x2, sigmas[i], denoised2)
        dt = sigma_down - sigmas[i]
        x2 = x2 + d2 * dt

        if torch.norm(x - x1) > torch.norm(x - x2):
            x = x1
        else:
            x = x2

        if sigmas[i + 1] > 0:
            x1 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
            x2 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

    return x


@torch.no_grad()
def sample_euler_ancestral6(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler1 = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed=seed+1000, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])

    i=0
    denoised = model(x, sigmas[i] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    d = to_d(x, sigmas[i], denoised)
    dt = sigma_down - sigmas[i]
    x = x + d * dt

    if sigmas[i + 1] > 0:
        x1 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        x2 = x + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

    sigmas = sigmas[1:]

    #for i in trange(int((len(sigmas) - 1)/2), disable=disable):
    while i < len(sigmas) - 1:
        print("i is = ", i)
        denoised1 = model(x1, sigmas[i] * s_in, **extra_args)
        denoised2 = model(x2, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d1 = to_d(x1, sigmas[i], denoised1)
        dt = sigma_down - sigmas[i]
        x1 = x1 + d1 * dt

        d2 = to_d(x2, sigmas[i], denoised2)
        dt = sigma_down - sigmas[i]
        x2 = x2 + d2 * dt

        x1_hat_1 = x1 + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        x1_hat_2 = x1 + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        x2_hat_1 = x2 + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        x2_hat_2 = x2 + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up

        ### next segment
        i += 1
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        ### SEGMENT 1

        denoised1_1 = model(x1_hat_1, sigmas[i] * s_in, **extra_args)
        denoised1_2 = model(x1_hat_2, sigmas[i] * s_in, **extra_args)
        denoised2_1 = model(x2_hat_1, sigmas[i] * s_in, **extra_args)
        denoised2_2 = model(x2_hat_2, sigmas[i] * s_in, **extra_args)

        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d1_1 = to_d(x1_hat_1, sigmas[i], denoised1_1)
        dt = sigma_down - sigmas[i]
        x1_1 = x1_hat_1 + d1_1 * dt

        d1_2 = to_d(x1_hat_2, sigmas[i], denoised1_2)
        dt = sigma_down - sigmas[i]
        x1_2 = x1_hat_2 + d1_2 * dt

        d2_1 = to_d(x2_hat_1, sigmas[i], denoised2_1)
        dt = sigma_down - sigmas[i]
        x2_1 = x2_hat_1 + d2_1 * dt

        d2_2 = to_d(x2_hat_2, sigmas[i], denoised2_2)
        dt = sigma_down - sigmas[i]
        x2_2 = x2_hat_2 + d2_2 * dt

        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        ### END SEGMENT 1 & 2

        norm1_1 = torch.norm(x - x1_1)
        norm1_2 = torch.norm(x - x1_2)
        norm2_1 = torch.norm(x - x2_1)
        norm2_2 = torch.norm(x - x2_2)

        if norm1_1 > norm1_2 and norm1_1 > norm2_1 and norm1_1 > norm2_2:
            x = x1_1
        if norm1_2 > norm1_1 and norm1_2 > norm2_1 and norm1_2 > norm2_2:
            x = x1_2
        if norm2_1 > norm1_1 and norm2_1 > norm1_2 and norm2_1 > norm2_2:
            x = x2_1
        if norm2_2 > norm1_1 and norm2_2 > norm1_2 and norm2_2 > norm2_1:
            x = x2_2

        if sigmas[i + 1] > 0:
            x1 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
            x2 = x + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * sigma_up
        
        i += 1

    return x




@torch.no_grad()
def sample_euler_stochastic6(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.25, s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    #noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None)
    noise_sampler1 = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    noise_sampler2 = NOISE_GENERATOR_CLASSES.get('gaussian')(x=x, seed=seed+1000, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])

    i=0
    denoised = model(x, sigmas[i] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

    if callback is not None:
        callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    d = to_d(x, sigmas[i], denoised)
    dt = sigma_down - sigmas[i]
    x = x + d * dt

    sigma_hat = sigmas[i] * eta
    if sigmas[i + 1] > 0:
        x1 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        x2 = x + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()

    sigmas = sigmas[1:]

    #if sigmas[i + 1] > 0:
    #    eps1 = noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1])
    #    eps2 = noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1])
        #x1 = x + eps1 * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()

    #for i in trange(int((len(sigmas) - 1)/2), disable=disable):
    while i < len(sigmas) - 1:
        print("i is = ", i)
        denoised1 = model(x1, sigma_hat * s_in, **extra_args)
        denoised2 = model(x2, sigma_hat * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d1 = to_d(x1, sigma_hat, denoised1)
        dt = sigmas[i + 1] - sigma_hat
        x1 = x1 + d1 * dt

        d2 = to_d(x2, sigma_hat, denoised2)
        dt = sigmas[i + 1] - sigma_hat
        x2 = x2 + d2 * dt

        i += 1
        sigma_hat = sigmas[i] * eta
        x1_hat_1 = x1 + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        x1_hat_2 = x1 + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()

        x2_hat_1 = x2 + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        x2_hat_2 = x2 + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()

        ### next segment
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        ### SEGMENT 1

        denoised1_1 = model(x1_hat_1, sigma_hat * s_in, **extra_args)
        denoised1_2 = model(x1_hat_2, sigma_hat * s_in, **extra_args)
        denoised2_1 = model(x2_hat_1, sigma_hat * s_in, **extra_args)
        denoised2_2 = model(x2_hat_2, sigma_hat * s_in, **extra_args)

        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d1_1 = to_d(x1_hat_1, sigma_hat, denoised1_1)
        dt = sigmas[i + 1] - sigma_hat
        x1_1 = x1_hat_1 + d1_1 * dt

        d1_2 = to_d(x1_hat_2, sigma_hat, denoised1_2)
        dt = sigmas[i + 1] - sigma_hat
        x1_2 = x1_hat_2 + d1_2 * dt

        d2_1 = to_d(x2_hat_1, sigma_hat, denoised2_1)
        dt = sigmas[i + 1] - sigma_hat
        x2_1 = x2_hat_1 + d2_1 * dt

        d2_2 = to_d(x2_hat_2, sigma_hat, denoised2_2)
        dt = sigmas[i + 1] - sigma_hat
        x2_2 = x2_hat_2 + d2_2 * dt

        #if callback is not None:
        #    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        ### END SEGMENT 1 & 2

        norm1_1 = torch.norm(x - x1_1)
        norm1_2 = torch.norm(x - x1_2)
        norm2_1 = torch.norm(x - x2_1)
        norm2_2 = torch.norm(x - x2_2)

        if norm1_1 > norm1_2 and norm1_1 > norm2_1 and norm1_1 > norm2_2:
            x = x1_1
        if norm1_2 > norm1_1 and norm1_2 > norm2_1 and norm1_2 > norm2_2:
            x = x1_2
        if norm2_1 > norm1_1 and norm2_1 > norm1_2 and norm2_1 > norm2_2:
            x = x2_1
        if norm2_2 > norm1_1 and norm2_2 > norm1_2 and norm2_2 > norm2_1:
            x = x2_2

        if sigmas[i + 1] > 0:
            sigma_hat = sigmas[i] * eta
            x1 = x + noise_sampler1(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
            x2 = x + noise_sampler2(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * (sigma_hat ** 2 - sigmas[i] ** 2).sqrt()
        
        i += 1

    return x



# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
    added = 0
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

# The following function adds the samplers during initialization, in __init__.py
def add_schedulers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    added = 0
    for scheduler in extra_schedulers: #getattr(self, "sample_{}".format(extra_samplers))
        if scheduler not in KSampler.SCHEDULERS:
            try:
                idx = KSampler.SCHEDULERS.index("ddim_uniform") # Last item in the samplers list
                KSampler.SCHEDULERS.insert(idx+1, scheduler) # Add our custom samplers
                setattr(k_diffusion_sampling, "get_sigmas_{}".format(scheduler), extra_schedulers[scheduler])
                added += 1
            except ValueError as err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

# Add any extra samplers to the following dictionary
#from .refined_exp_solver import sample_refined_exp_s, sample_refined_exp_s_advanced
extra_samplers = {
    "euler_ancestral_advanced": sample_euler_ancestral_advanced,
    "euler_recursive": sample_euler_ancestral_recursive,
    "sample_euler_ancestral2": sample_euler_ancestral2,
    "sample_euler_ancestral6": sample_euler_ancestral6,
    "sample_euler_stochastic6": sample_euler_stochastic6,
    "sample_euler_ancestral_recursive_get2": sample_euler_ancestral_recursive_get2,
    "sample_euler_ancestral_recursive_depth_get2": sample_euler_ancestral_recursive_depth_get2,
    "sample_euler_ancestral_recursive_depth_get2_call": sample_euler_ancestral_recursive_depth_get2_call,
    "dpmpp_2s_ancestral_advanced": sample_dpmpp_2s_ancestral_advanced,
    "res_momentumized_advanced": sample_res_solver_advanced,
    "dpmpp_dualsde_momentumized_advanced": sample_dpmpp_dualsdemomentum_advanced,
    "dpmpp_sde_advanced": sample_dpmpp_sde_advanced,
    "dpmpp_2m_sde_advanced": sample_dpmpp_2m_sde_advanced,
    "dpmpp_3m_sde_advanced": sample_dpmpp_3m_sde_advanced,
    "dpmpp_sde_cfgpp_advanced": sample_dpmpp_sde_cfgpp_advanced,

    "dpmpp_2s_a_cfg++": sample_dpmpp_2s_ancestral_cfgpp,
    "dpmpp_2m_cfg++": sample_dpmpp_2m_cfgpp,
    "dpmpp_sde_cfg++": sample_dpmpp_sde_cfgpp,
    "dpmpp_2m_sde_cfg++": sample_dpmpp_2m_sde_cfgpp,
    "dpmpp_3m_sde_cfg++": sample_dpmpp_3m_sde_cfgpp,

    "rk4":  sample_rk4,
}

discard_penultimate_sigma_samplers = set((
    "dpmpp_dualsde_momentumized",
    "clyb_4m_sde_momentumized"
))

def get_sigmas_simple_exponential(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    sigs = torch.FloatTensor(sigs)
    exp = torch.exp(torch.log(torch.linspace(1, 0, steps + 1)))
    return sigs * exp

extra_schedulers = {
    "simple_exponential": get_sigmas_simple_exponential
}
