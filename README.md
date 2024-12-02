# RES4LYF

At the heart of this repository is the "ClownsharKSampler", which was specifically designed to support both rectified flow and probability flow models. It features 40 different selectible samplers (30 explicit, 7 fully implicit, 3 diagonally implicit) all available in both ODE or SDE modes with 20 noise types, 7 noise scaling modes, and options for implicit Runge-Kutta sampling refinement steps. Several new explicit samplers are implemented, most notably RES_2M, RES_3S, and RES_5S. Additionally, img2img capabilities include both latent image guidance and unsampling/resampling (via new forms of rectified noise inversion). 

A particular emphasis of this project has been to facilitate modulating parameters vs. time, which can facilitate large gains in image quality from the sampling process. To this end, a wide variety of sigma, latent, and noise manipulation nodes are included. 

Much of this work remains experimental and is subject to further changes.

# ClownsharKSampler
![image](https://github.com/user-attachments/assets/430fb77a-7353-4b40-acb6-cbd33392f7fc)

This is an all-in-one sampling node designed for convenience without compromising on control or quality. 

There are several key sections to the parameters which will be explained below.

## INPUTS
![image](https://github.com/user-attachments/assets/e8fe825d-2fb1-4e93-874c-89fb73ba68f7)

The only two mandatory inputs here are "model" and "latent_image". 

**POSITIVE and NEGATIVE:** If you connect nothing to either of these inputs, the node will automatically generate null conditioning. If you are unsampling, you actually don't need to hook up any conditioning at all (and will set CFG = 1.0). In most cases, merely using the positive conditioning will suffice, unless you really need to use a specific negative prompt.

**SIGMAS:** If a sigmas scheduler node is connected to this input, it will override the scheduler and steps settings chosen within the node.

## NOISE SETTINGS:
![image](https://github.com/user-attachments/assets/caaa41a4-5afa-4c3c-8fb2-003b9a6b2578)

**NOISE_TYPE_INIT:** This sets the initial noise type applied to the latent image. 

**NOISE_TYPE_SDE:** This sets the noise type used during SDE sampling. Note that SDE sampling is identical to ODE sampling in most ways - the difference is that noise is added after each step. It's like a form of carefully controlled continuous noise injection.

**NOISE_MODE_SDE:** This determines what method is used for scaling the amount of noise to be added based on the "eta" setting below. They are listed in order of strength of the effect. 

**ETA:** This controls how much noise is added after each step. Note that for most of the noise modes, anything equal to or greater than 1.0 will trigger internal scaling to prevent NaN errors. The exception is the noise mode "exp" which allows for settings far above 1.0. 

**NOISE_SEED:** Largely identical to the setting in KSampler. Set to -1 to have it increment the most recently used seed (by the workflow) by 1.

**CONTROL_AFTER_GENERATE:** Self-explanatory. I recommend setting to "fixed" or "increment" (as you don't have to reload the workflow to regenerate something, you can just decement it by one).

## SAMPLER SETTINGS:
![image](https://github.com/user-attachments/assets/d5ef0bef-7388-44f0-a119-220beec9883d)

**SAMPLER_MODE:** In virtually all situations, use "standard". However, if you are unsampling, set to "unsample", and if you are resampling (the stage after unsampling), set to "resample". Both of these modes will disable noise addition within ComfyUI, which is essential for these methods to work properly. 

**SAMPLER_NAME:** This is used similarly to the KSampler setting. This selects the explicit sampler type. Note the use of numbers and letters at the end of each sampler name: "2m, 3m, 2s, 3s, 5s, etc." 

Samplers that end in "s" use substeps between each step. One ending with "2s" has two stages per step, therefore costs two model calls per step (Euler costs one - model calls are what determine inference time). "3s" would take three model calls per step, and therefore take three times as long to run as Euler. However, the increase in accuracy can be very dramatic, especially when using noise (SDE sampling). The "res" family of samplers are particularly notable (they are effectively refinements of the dpmpp family, with new, higher order, much more accurate versions implemented here).

Samplers that end in "m" are "multistep" samplers, which instead of issuing new model calls for substeps, recycle previous steps as estimations for these substeps. They're less accurate, but all run at Euler speed (one model call per step). Sometimes this can be an advantage, as multistep samplers tend to converge more linearly toward a target image. This can be useful for img2img transformations, unsampling, or when using latent image guides.

**IMPLICIT_SAMPLER_NAME:** This is very useful with SD3.5 Medium for improving coherence, reducing artifacts and mutations, etc. It may be difficult to use with a model like Flux unless you plan on setting up a queue of generations and walking away. It will use the explicit step type as a predictor for each of the implicit substeps, so if you choose a slow explicit sampler, you will be waiting a long time. Euler, res_2m, deis_2m, etc. will often suffice as a predictor for implicit sampling, though any sampler may be used. Try "res_5s" as your explicit sampler type, and "gauss-legendre_5s", if you wish to demonstrate your commitment to climate change (and image quality).

Setting this to "none" has the same effect as setting implicit_steps = 0.

## SCHEDULER AND DENOISE SETTINGS:
![image](https://github.com/user-attachments/assets/b89d3956-1734-4368-8bb4-429b9989cd4d)

These are identical in most ways to the settings by the same name in KSampler. 

**SCHEDULER:** There is one extra sigma scheduler offered by default: "beta57" which is the beta schedule with modified parameters (alpha = 0.5, beta = 0.7).

**IMPLICIT_STEPS:** This controls the number of implicit steps to run. Note that it will double, triple, etc. the runtime as you increase the stepcount. Typically, gains diminish quickly after 2-3 implicit steps.

**DENOISE:** This is identical to the KSampler setting. Controls the amount of noise removed from the image. Note that with this method, the effect will change significantly depending on your choice of scheduler.

**DENOISE_ALT:** Instead of splitting the sigma schedule like "denoise", this multiplies them. The results are different, but track more closely from one scheduler to another when using the same value. This can be particularly useful for img2img workflows.

**CFG:** This is identical to the KSampler setting. Typically, you'll set this to 1.0 (to disable it) when using Flux, if you're using Flux guidance. However, the effect is quite nice when using dedistilled models if you use "CLIP Text Encode" without any Flux guidance, and set CFG to 3.0. 

If you've never quite understood CFG, you can think of it this way. Imagine you're walking down the street and see what looks like an enticing music festival in the distance (your positive conditioning). You're on the fence about attending, but then, suddenly, a horde of pickleshark cannibals come storming out of a nearby bar (your negative conditioning). Together, the two team up to drive you toward the music festival. That's CFG.

## SHIFT SETTINGS:
![image](https://github.com/user-attachments/assets/e9a2e2d7-be5c-4b63-8647-275409600b56)

These are present for convenience as they are used in virtually every workflow.

**SHIFT:** This is the same as "shift" for the ModelSampling nodes for SD3.5, AuraFlow, etc., and is equivalent to "max_shift" for Flux. Set this value to -1 to disable setting shift (or max_shift) within the node.

**BASE_SHIFT:** This is only used by Flux. Set this value to -1 to disable setting base_shift within the node.

**SHIFT_SCALING:** This changes how the shift values are calculated. "exponential" is the default used by Flux, whereas "linear" is the default used by SD3.5 and AuraFlow. In most cases, "exponential" leads to better results, though "linear" has some niche uses. 

# Sampler and noise mode list

## Explicit samplers
Bolded samplers are added as options to the sampler dropdown in ComyfUI (an ODE and SDE version for each).
**res_2m**

**res_3m**

**res_2s**

**res_3s**
**res_5s**

**deis_2m
deis_3m
deis_4m**

ralston_2s
ralston_3s
ralston_4s

dpmpp_2m
dpmpp_3m
dpmpp_2s
dpmpp_sde_2s
dpmpp_3s

midpoint_2s
heun_2s
heun_3s

houwen-wray_3s
kutta_3s
ssprk3_3s

rk38_4s
rk4_4s

dormand-prince_6s
dormand-prince_13s
bogacki-shampine_7s

ddim
euler

# CURRENT FLUX WORKFLOWS

## TXT2IMG Workflow:
This uses my amateur cell phone lora, which is freely available (https://huggingface.co/ClownsharkBatwing/CSBW_Style/blob/main/amateurphotos_1_amateurcellphonephoto_recapt2.safetensors). It significantly reduces the plastic, blurred look of Flux Dev.
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20flux.png)
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20WF%20flux.png)

## INPAINTING Workflow:

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/inpainting%20flux.png)
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/inpainting%20WF%20flux.png)

## DUAL GUIDES WITH MASKS Workflow:

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20masked%20flux.png)
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20masked%20WF%20flux.png)

# PREVIOUS WORKFLOWS
**THE FOLLOWING WORKFLOWS ARE FOR A PREVIOUS VERSION OF THE NODE.** 
These will still work! You will, however, need to manually delete and recreate the sampler and guide nodes and input the settings as they appear in the screenshots. The layout of the nodes has been changed slightly. To replicate their behavior precisely, add to the new extra_options box in ClownsharKSampler: truncate_conditioning=true (if that setting was used in the screenshot for the node).

![image](https://github.com/user-attachments/assets/a55ec484-1339-45a2-bcc4-76934f4648d4)

**TXT2IMG Workflow:** 

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20SD35M.png)

**TXT2IMG Workflow (Latent Image Guides):**
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20guided%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20guided%20SD35M.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20guided%20SD35M%20input.png

**TXT2IMG Workflow (Dual Guides with Masking):**
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M.png)

Input images and mask:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20input1.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20input2.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20mask.png

**IMG2IMG Workflow (Unsampling):** 

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SD35L%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SD35L.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SD35L%20input.png

**IMG2IMG Workflow (Unsampling with SDXL):**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SDXL%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SDXL.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SDXL%20input.png

**IMG2IMG Workflow (Unsampling with latent image guide):**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20guided%20unsampling%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20guided%20unsampling%20SD35M.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20guided%20unsampling%20SD35M%20input.png

**IMG2IMG Workflow (Unsampling with dual latent image guides and masking):**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M.png)

Input images and mask:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20input1.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20input2.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20mask.png

**INSTALLATION:** 

If you are using a venv, you will need to first run from within your ComfyUI folder (that contains your "venv" folder):

_Linux:_

source venv/bin/activate

_Windows:_

venv\Scripts\activate

_Then, "cd" into your "custom_nodes" folder and run the following commands:_

git clone https://github.com/ClownsharkBatwing/RES4LYF/

cd RES4LYF

_If you are using a venv, run these commands:_

pip install -r requirements.txt

_Alternatively, if you are using the portable version of ComfyUI you will need to replace "pip" with the path to your embedded pip executable. For example, on Windows:_

X:\path\to\your\comfy_portable_folder\python_embedded\Scripts\pip.exe install -r requirements.txt

