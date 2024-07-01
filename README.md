# RES4LYF

These nodes are for advanced manipulation of sigmas, latents, and various advanced sampling methods, including scheduling all parameters for the RES sampler (the code for which was adapted from https://github.com/Clybius/ComfyUI-Extra-Samplers) via a node called "ClownSampler". A variety of new noise types have also been added, which are available as noise sampler options and in "SharkSampler" (which is effectively an advanced version of SamplerCustomNoise with a number of added noise types).

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

pip install opensimplex --no-deps

_Alternatively, if you are using the portable version of ComfyUI you will need to replace "pip" with the path to your embedded pip executable. For example, on Windows:_

X:\path\to\your\comfy_portable_folder\python_embedded\Scripts\pip.exe install -r requirements.txt

X:\path\to\your\comfy_portable_folder\python_embedded\Scripts\pip.exe install opensimplex --no-deps

**NOISE TYPES:**

fractal (this is also known as 1/f or power noise; positive alpha values contain more low frequency noise, negative contain more high frequency noise)

brown (fractal with alpha = 2.0)

pink (fractal with alpha = 1.0)

white (fractal with alpha = 0.0)

blue (fractal with alpha = -1.0)

violet (fractal with alpha = -2.0)

gaussian

uniform

pyramid-cascade_B

pyramid-interpolated

pyramid-bilinear

pyramid-bicubic

pyramid-nearest

hires-pyramid-bilinear

hires-pyramid-bicubic

hires-pyramid-nearest

brownian

laplacian

studentt

wavelet

simplex

perlin

**GENERAL UTILITY NODES:**

Set Precision

**SAMPLER NODES:**

ClownSampler

SharkSampler

SamplerDPMPP_DualSDE_Advanced

SamplerDPMPP_SDE_Advanced

SamplerDPMPP_SDE_CFG++_Advanced

SamplerEulerAncestral_Advanced

SamplerDPMPP_2S_Ancestral_Advanced

SamplerDPMPP_2M_SDE_Advanced

SamplerDPMPP_3M_SDE_Advanced

**SIGMAS MANIPULATION NODES:**

Sigmas Truncate

Sigmas Start

Sigmas Split

Sigams Concat

Sigmas Pad

Sigmas Unpad

Sigmas SetFloor

Sigmas DeleteBelowFloor

Sigmas DeleteDuplicates

Sigmas Cleanup

Sigmas Mult

Sigmas Modulus

Sigmas Quotient

Sigmas Add

Sigmas Power

Sigmas Abs

Sigmas2 Mult

Sigmas2 Add

Sigmas Math1

Sigmas Math3

Sigmas Iteration Karras

Sigmas Iteration Polyexp

Tan Scheduler

Tan Scheduler 2

Tan Scheduler 2 Simple

**CONDITIONING MANIPULATION NODES:**

ConditioningAverageScheduler

**LATENT MANIPULATION NODES:**

LatentNoiseList

LatentBatch_channels

LatentBatch_channels_16

LatentNoiseBatch_fractal

LatentNoiseBatch_gaussian

LatentNoiseBatch_gaussian_channels

Latent to Cuda

Latent Batcher

Latent Normalize Channels

LatentNoiseBatch_perlin

EmptyLatentImage64

EmptyLatentImageCustom

StableCascade_StageC_VAEEncode_Exact

LatentPhaseMagnitude

LatentPhaseMagnitudeMultiply

LatentPhaseMagnitudeOffset

LatentPhaseMagnitudePower

