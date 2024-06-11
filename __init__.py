from . import extra_samplers
from . import samplers
from . import sigmas
from . import latents

import torch

extra_samplers.add_samplers()

#torch.set_default_dtype(torch.float64)

NODE_CLASS_MAPPINGS = {
    "LatentNoiseList": latents.LatentNoiseList,
    "LatentNoiseBatch_1f": latents.LatentNoiseBatch_fractal,
    "LatentNoiseBatch_gauss": latents.LatentNoiseBatch_gauss,
    "LatentNoiseBatch_gaussian_channels": latents.LatentNoiseBatch_gaussian_channels,
    "Latent to Cuda": latents.latent_to_cuda,
    "Latent Batcher": latents.latent_batch,
    "Set Precision": latents.set_precision,
    "LatentNoiseBatch_perlin": latents.LatentNoiseBatch_perlin,
    "EmptyLatentImage64": latents.EmptyLatentImage64,

    "LatentPhaseMagnitude": latents.LatentPhaseMagnitude,
    "LatentPhaseMagnitudeMultiply": latents.LatentPhaseMagnitudeMultiply,
    "LatentPhaseMagnitudeOffset": latents.LatentPhaseMagnitudeOffset,
    "LatentPhaseMagnitudePower": latents.LatentPhaseMagnitudePower,

    "ClownSampler": samplers.ClownSampler,
    "SharkSampler": samplers.SharkSampler,
    "SamplerDPMPP_DualSDE_Advanced": samplers.SamplerDPMPP_DUALSDE_MOMENTUMIZED_ADVANCED,
    "SamplerDPMPP_SDE_Advanced": samplers.SamplerDPMPP_SDE_ADVANCED,

    "Sigmas Truncate": sigmas.sigmas_truncate,
    "Sigmas Start": sigmas.sigmas_start,
    "Sigmas Split": sigmas.sigmas_split,
    "Sigmas Concat": sigmas.sigmas_concatenate,
    "Sigmas Pad": sigmas.sigmas_pad,
    "Sigmas Unpad": sigmas.sigmas_unpad,
    
    "Sigmas SetFloor": sigmas.sigmas_set_floor,
    "Sigmas DeleteBelowFloor": sigmas.sigmas_delete_below_floor,
    "Sigmas DeleteDuplicates": sigmas.sigmas_delete_consecutive_duplicates,
    "Sigmas Cleanup": sigmas.sigmas_cleanup,
    
    "Sigmas Mult": sigmas.sigmas_mult,
    "Sigmas Modulus": sigmas.sigmas_modulus,
    "Sigmas Quotient": sigmas.sigmas_quotient,
    "Sigmas Add": sigmas.sigmas_add,
    "Sigmas Power": sigmas.sigmas_power,
    "Sigmas Abs": sigmas.sigmas_abs,
    
    "Sigmas2 Mult": sigmas.sigmas2_mult,
    "Sigmas2 Add": sigmas.sigmas2_add,

    "Sigmas Math1": sigmas.sigmas_math1,
    "Sigmas Math3": sigmas.sigmas_math3,

    "Sigmas Iteration Karras": sigmas.sigmas_iteration_karras,
    "Sigmas Iteration Polyexp": sigmas.sigmas_iteration_polyexp,

    "Tan Scheduler": sigmas.tan_scheduler,
    "Tan Scheduler 2": sigmas.tan_scheduler_2stage,
    "Tan Scheduler 2 Simple": sigmas.tan_scheduler_2stage_simple,

}
__all__ = ['NODE_CLASS_MAPPINGS']
