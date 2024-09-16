import torch
from torch import FloatTensor
from tqdm.auto import trange
from math import pi

from comfy.k_diffusion.sampling import get_ancestral_step, to_d


import functools

from .noise_classes import *
#from comfy_extras.nodes_advanced_samplers import sample_euler_cfgpp_alt

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def get_RF_step(sigma, sigma_next, eta):
    """Calculates the noise level (sigma_down) to step down to and the amount of noise to add (sigma_up) when doing a rectified flow sampling step, 
    and a mixing ratio (alpha_ratio) for scaling the latent during noise addition."""
    downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
    sigma_down = sigma_next * downstep_ratio
    alpha_ip1  = 1 - sigma_next
    alpha_down = 1 - sigma_down + 1.0e-8
    alpha_ratio = alpha_ip1 / alpha_down
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5 
    #sigma_up = (   (sigma_next ** 2 - sigma_down ** 2)   *   ((1 - sigma_next) ** 2 / (1 - sigma_down)** 2)   )   ** 0.5
    return (sigma_down, sigma_up, alpha_ratio, )  # sigma_up = renoise_coeff

@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_sde_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    momentum=1.0, eta=1., s_noise=1., noise_sampler=None, r=1/2, noise_sampler_type="brownian", k=1.0, scale=0.1, alpha=None,
):
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_dpmpp_sde_advanced_RF(model, x, sigmas, extra_args, callback, disable, momentum, eta, s_noise, noise_sampler, r, noise_sampler_type, k, scale, alpha)
    #DPM-Solver++ (stochastic with eta parameter).
    if len(sigmas) <= 1:
        return x
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1

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

@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_sde_advanced_RF(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    momentum=1.0, eta=1., s_noise=1., noise_sampler=None, r=1/2, noise_sampler_type="brownian", k=1.0, scale=0.1, alpha=None,
):
    """DPM-Solver++ (stochastic with eta parameter) adapted for Rectified Flow."""
    diff, diff_2, vel, vel_2 = None, None, None, None
    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel
    
    if len(sigmas) <= 1:
        return x
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1

    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
            
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        diff_2 = momentum_func(denoised, vel_2, sigmas[i])
        vel_2 = diff_2
        denoised = diff_2
        
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
            s = t + h * r    #timestep scaled by r between full step and partial
            fac = 1 / (2 * r)
            
            sigma_s = sigma_fn(s)    #NEW...
            if sigmas[i] == 1.0:
                sigma_s = 0.9999

            # Step 1
            sd, su, alpha_ratio = get_RF_step(sigma_fn(t), sigma_s, eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = alpha_ratio * x_2 + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_s) * s_noise * su         #SIGMA_S subs for SIGMA_FN(S)
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)
            
            diff = momentum_func(denoised_2, vel, sigmas[i])
            vel = diff
            denoised_2 = diff

            # Step 2
            sd, su, alpha_ratio = get_RF_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = alpha_ratio * x + noise_sampler(sigma=sigma_fn(t), sigma_next=sigma_fn(t_next)) * s_noise * su
            
            print("alpha__: ", alpha_ratio, sd, su)
            del denoised, denoised_d, denoised_2, x_2
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    return x


@precision_tool.cast_tensor
@torch.no_grad()
def sample_dpmpp_sde_cfgpp_advanced(
    model, x, sigmas, extra_args=None, callback=None, disable=None, eulers_mom=None,
    eta=1., s_noise=1., noise_sampler=None, r=1/2, k=1.0, scale=0.1, noise_sampler_type="brownian", cfgpp: FloatTensor = torch.zeros((1,)), alpha: FloatTensor = torch.zeros((1,))
):
    #DPM-Solver++ (stochastic with eta parameter).
    if len(sigmas) <= 1:
        return x

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1

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

@precision_tool.cast_tensor
def sample_dpmpp_dualsdemomentum_advanced(model, x, sigmas, seed=42, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler_type="gaussian", noise_sampler=None, r=1/2, momentum=0.0, momentums=None, etas=None, s_noises=None,rs=None,scheduled_r=False):
    return sample_dpmpp_dualsde_momentum_advanced(model, x, sigmas, seed=seed, extra_args=extra_args, callback=callback, disable=disable, eta=etas, s_noise=s_noises, noise_sampler_type=noise_sampler_type, noise_sampler=noise_sampler, r=rs, momentum=momentums, scheduled_r=False)

@precision_tool.cast_tensor
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
from .refined_exp_solver import sample_refined_exp_s_advanced, sample_refined_exp_s_advanced_RF

@precision_tool.cast_tensor
def sample_res_solver_advanced(model, 
                               x, 
                               sigmas, etas, c2s, momentums, eulers_moms, offsets, branch_mode, branch_depth, branch_width,
                               guides_1, guides_2, latent_guide_1, latent_guide_2, guide_mode_1, guide_mode_2, guide_1_channels,
                               k, clownseed=0, cfgpps=0.0, alphas=None, latent_noise=None, latent_self_guide_1=False,latent_shift_guide_1=False,
                               extra_args=None, callback=None, disable=None, noise_sampler_type="gaussian", noise_sampler=None, denoise_to_zero=True, simple_phi_calc=False, c2=0.5, momentum=0.0, eulers_mom=0.0, offset=0.0):
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_refined_exp_s_advanced_RF(
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
    else:
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
    seed = extra_args.get("seed", None) + 1
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

    seed = extra_args.get("seed", None) + 1
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

    seed = extra_args.get("seed", None) + 1
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
def sample_euler_ancestral_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    """Ancestral sampling with Euler method steps."""
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        return sample_euler_ancestral_advanced_RF(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, noise_sampler_type, k, scale, alpha)
    
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

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
def sample_euler_ancestral_advanced_RF(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None) + 1
    
    s_in = x.new_ones([x.shape[0]])
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down =     sigmas[i + 1] * downstep_ratio
        alpha_ip1  = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0 and eta > 0:
            x = (alpha_ip1 / alpha_down) * x   +   noise_sampler(sigma=sigmas[i], sigma_next=sigmas[i + 1]) * s_noise * renoise_coeff
            
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", k=1.0, scale=0.1, alpha=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    alpha = torch.zeros_like(sigmas) if alpha is None else alpha
    extra_args = {} if extra_args is None else extra_args

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    seed = extra_args.get("seed", None) + 1
    noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        if noise_sampler_type == "fractal":
            noise_sampler.alpha = alpha[i]
            noise_sampler.k = k
            noise_sampler.scale = scale

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

    seed = extra_args.get("seed", None) + 1
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

    seed = extra_args.get("seed", None) + 1
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

@torch.no_grad()
def sample_euler_ancestral_recursive(model, x, sigmas, i=0, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_sampler_type="gaussian", ):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    if noise_sampler is None:
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        seed = extra_args.get("seed", None) + 1
        noise_sampler = NOISE_GENERATOR_CLASSES.get(noise_sampler_type)(x=x, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)

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
