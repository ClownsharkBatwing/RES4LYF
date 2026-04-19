import torch
import torch.nn.functional as F
from torch import Tensor

import itertools
import copy

from typing          import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .noise_classes import NoiseGenerator
    NoiseGeneratorSubclass = TypeVar("NoiseGeneratorSubclass", bound="NoiseGenerator") 

from einops          import rearrange

from ..sigmas        import get_sigmas
from ..helper        import ExtraOptions, FrameWeightsManager, initialize_or_scale, is_video_model
from ..latents       import normalize_zscore, get_collinear, get_orthogonal, get_cosine_similarity, get_pearson_similarity, \
                            get_slerp_weight_for_cossim, normalize_latent, hard_light_blend, slerp_tensor, get_orthogonal_noise_from_channelwise, \
                            get_edge_mask, is_packed_latent

from .rk_method_beta import RK_Method_Beta
from .constants      import MAX_STEPS
from ..res4lyf       import RESplain, is_debug_logging_enabled

import comfy.utils


def flatten_to_match(guide: Tensor, target: Tensor) -> Tensor:
    """Flatten guide tensor to match target's shape when target is flat [1,1,N].

    If target is not flat, returns guide unchanged.
    If guide is NestedTensor, packs it to flat format.
    If guide is regular tensor, reshapes to [1,1,N].
    """
    if target.ndim != 3:
        return guide

    # Target is flat [1,1,N], flatten the guide to match
    if hasattr(guide, 'is_nested') and guide.is_nested:
        # NestedTensor - pack to flat
        flat_guide, _ = comfy.utils.pack_latents(guide.unbind())
        return flat_guide
    else:
        # Regular tensor - reshape to flat
        return guide.reshape(1, 1, -1)


class LatentGuide:
    OFFLOADABLE_ATTRS = [
        # Guide targets (full latent size)
        'y0', 'y0_inv', 'y0_mean', 'y0_adain', 'y0_attninj', 'y0_style_pos', 'y0_style_neg',
        # Spatial masks (full latent size)
        'mask', 'mask_inv', 'mask_sync', 'mask_drift_x', 'mask_drift_y',
        'mask_lure_x', 'mask_lure_y', 'mask_mean', 'mask_adain', 'mask_attninj',
        'mask_style_pos', 'mask_style_neg',
        # Self-refine state (full latent size)
        'self_refine_epsilon_ref', '_self_refine_iter_prediction',
        '_self_refine_certain_mask_accum', '_debug_certainty_mask',
        # Frame weights (if populated)
        'frame_weights', 'frame_weights_inv',
    ]

    def offload(self, device):
        for attr in self.OFFLOADABLE_ATTRS:
            val = getattr(self, attr, None)
            if isinstance(val, Tensor):
                setattr(self, attr, val.to(device))
        # Handle list-of-tensor attributes
        for list_attr in ('x_lying_', 's_lying_'):
            val = getattr(self, list_attr, None)
            if isinstance(val, list):
                setattr(self, list_attr, [t.to(device) if isinstance(t, Tensor) else t for t in val])

    def restore(self):
        self.offload(self.device)

    def __init__(self,
                model,
                sigmas               : Tensor,
                UNSAMPLE             : bool,
                VE_MODEL             : bool,
                LGW_MASK_RESCALE_MIN : bool,
                extra_options        : str,
                device               : str = 'cpu',
                dtype                : torch.dtype = torch.float64,
                frame_weights_mgr    : FrameWeightsManager = None,
                latent_shapes        : list = None,
                ):

        self.dtype                    = dtype
        self.device                   = device
        self.model                    = model
        self.latent_shapes            = latent_shapes

        if hasattr(model, "model"):
            model_sampling = model.model.model_sampling
        elif hasattr(model, "inner_model"):
            model_sampling = model.inner_model.inner_model.model_sampling
        
        self.sigma_min                 = model_sampling.sigma_min.to(dtype=dtype, device=device)
        self.sigma_max                 = model_sampling.sigma_max.to(dtype=dtype, device=device)
        self.sigmas                    = sigmas                  .to(dtype=dtype, device=device)
        self.UNSAMPLE                  = UNSAMPLE
        self.VE_MODEL                  = VE_MODEL
        self.VIDEO                     = is_video_model(model)
        self.SAMPLE                    = (sigmas[0] > sigmas[1])    # type torch.bool
        self.y0                        = None
        self.y0_inv                    = None
        self.y0_mean                   = None
        self.y0_adain                  = None
        self.y0_attninj                = None
        self.y0_style_pos              = None
        self.y0_style_neg              = None

        # Original shape for pack/unpack (pack-first experiment)
        self.y0_original_shape         = None

        self.guide_mode                = ""
        self.max_steps                 = MAX_STEPS
        self.mask                      = None
        self.mask_inv                  = None
        self.invert_mask               = False
        self.mask_sync                 = None
        self.mask_drift_x              = None
        self.mask_drift_y              = None
        self.mask_lure_x               = None
        self.mask_lure_y               = None
        self.mask_mean                 = None
        self.mask_adain                = None
        self.mask_attninj              = None
        self.mask_style_pos            = None
        self.mask_style_neg            = None
        self.x_lying_                  = None
        self.s_lying_                  = None
        
        self.LGW_MASK_RESCALE_MIN      = LGW_MASK_RESCALE_MIN
        self.HAS_LATENT_GUIDE          = False
        self.HAS_LATENT_GUIDE_INV      = False
        self.HAS_LATENT_GUIDE_MEAN     = False
        self.HAS_LATENT_GUIDE_ADAIN    = False
        self.HAS_LATENT_GUIDE_ATTNINJ  = False
        self.HAS_LATENT_GUIDE_STYLE_POS= False
        self.HAS_LATENT_GUIDE_STYLE_NEG= False
        self.USE_DENOISED_AS_GUIDE     = False
        self.SELF_REFINE_EPSILON_MODE  = False
        self.self_refine_epsilon_ref   = None
        self.self_refine_epsilon_last_step = -1
        self.self_refine_epsilon_last_row = -1
        self.self_refine_epsilon_call_count = 0  # Track calls per (step, row)
        self.self_refine_threshold     = 0.25
        self.self_refine_cutoff        = 0.99
        self.self_refine_metric        = "l1"
        self._self_refine_converged    = False
        
        self.lgw                       = torch.full_like(sigmas, 0., dtype=dtype) 
        self.lgw_inv                   = torch.full_like(sigmas, 0., dtype=dtype)
        self.lgw_mean                  = torch.full_like(sigmas, 0., dtype=dtype)
        self.lgw_adain                 = torch.full_like(sigmas, 0., dtype=dtype)
        self.lgw_attninj               = torch.full_like(sigmas, 0., dtype=dtype)
        self.lgw_style_pos             = torch.full_like(sigmas, 0., dtype=dtype)
        self.lgw_style_neg             = torch.full_like(sigmas, 0., dtype=dtype)
        
        self.cossim_tgt                = torch.full_like(sigmas, 0., dtype=dtype) 
        self.cossim_tgt_inv            = torch.full_like(sigmas, 0., dtype=dtype) 
        
        self.guide_cossim_cutoff_          = 1.0
        self.guide_bkg_cossim_cutoff_      = 1.0
        self.guide_mean_cossim_cutoff_     = 1.0
        self.guide_adain_cossim_cutoff_    = 1.0
        self.guide_attninj_cossim_cutoff_  = 1.0
        self.guide_style_pos_cossim_cutoff_= 1.0
        self.guide_style_neg_cossim_cutoff_= 1.0

        self.frame_weights_mgr        = frame_weights_mgr
        self.frame_weights            = None
        self.frame_weights_inv        = None
        
        #self.freqsep_lowpass_method   = "none"
        #self.freqsep_sigma            = 0.
        #self.freqsep_kernel_size      = 0 
        
        self.extra_options            = extra_options
        self.EO                       = ExtraOptions(extra_options)


    def init_guides(self,
            x             : Tensor,
            RK_IMPLICIT   : bool,
            guides        : Optional[Tensor]                   = None,
            noise_sampler : Optional["NoiseGeneratorSubclass"] = None,
            batch_num     : int                                = 0,
            sigma_init                                         = None,
            guide_inversion_y0                                 = None,
            guide_inversion_y0_inv                             = None,
        ) -> Tensor:
        
        latent_guide_weight              = 0.0
        latent_guide_weight_inv          = 0.0
        latent_guide_weight_sync         = 0.0
        latent_guide_weight_sync_inv     = 0.0
        latent_guide_weight_drift_x      = 0.0
        latent_guide_weight_drift_x_inv  = 0.0
        latent_guide_weight_drift_y      = 0.0
        latent_guide_weight_drift_y_inv  = 0.0
        latent_guide_weight_lure_x       = 0.0
        latent_guide_weight_lure_x_inv   = 0.0
        latent_guide_weight_lure_y       = 0.0
        latent_guide_weight_lure_y_inv   = 0.0
        
        latent_guide_weight_mean         = 0.0
        latent_guide_weight_adain        = 0.0
        latent_guide_weight_attninj      = 0.0
        latent_guide_weight_style_pos    = 0.0
        latent_guide_weight_style_neg    = 0.0

        latent_guide_weights             = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_inv         = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_sync        = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_sync_inv    = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_drift_x     = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_drift_x_inv = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_drift_y     = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_drift_y_inv = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_lure_x      = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_lure_x_inv  = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_lure_y      = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_lure_y_inv  = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_mean        = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_adain       = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_attninj     = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_style_pos   = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_style_neg   = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)

        latent_guide           = None
        latent_guide_inv       = None
        latent_guide_mean      = None
        latent_guide_adain     = None
        latent_guide_attninj   = None
        latent_guide_style_pos = None
        latent_guide_style_neg = None
        
        self.drift_x_data  = 0.0
        self.drift_x_sync  = 0.0
        self.drift_y_data  = 0.0
        self.drift_y_sync  = 0.0
        self.drift_y_guide = 0.0
        
        if guides is not None:
            self.guide_mode                 = guides.get("guide_mode", "none")
            
            if self.guide_mode.startswith("inversion"):
                self.guide_mode = self.guide_mode.replace("inversion", "epsilon", 1)
            else:
                self.SAMPLE   = True
                self.UNSAMPLE = False
    
            self.self_refine_threshold       = guides.get("self_refine_threshold", self.EO("self_refine_threshold", 0.25))
            self.self_refine_cutoff          = guides.get("self_refine_cutoff",    self.EO("self_refine_cutoff", 0.99))
            self.self_refine_metric          = guides.get("self_refine_metric",    self.EO("self_refine_metric", "l1")).lower()

            latent_guide_weight              = guides.get("weight_masked",           0.)
            latent_guide_weight_inv          = guides.get("weight_unmasked",         0.)
            latent_guide_weight_sync         = guides.get("weight_masked_sync",      0.)
            latent_guide_weight_sync_inv     = guides.get("weight_unmasked_sync",    0.)
            latent_guide_weight_drift_x      = guides.get("weight_masked_drift_x",   0.)
            latent_guide_weight_drift_x_inv  = guides.get("weight_unmasked_drift_x", 0.)
            latent_guide_weight_drift_y      = guides.get("weight_masked_drift_y",   0.)
            latent_guide_weight_drift_y_inv  = guides.get("weight_unmasked_drift_y", 0.)
            latent_guide_weight_lure_x       = guides.get("weight_masked_lure_x",    0.)
            latent_guide_weight_lure_x_inv   = guides.get("weight_unmasked_lure_x",  0.)
            latent_guide_weight_lure_y       = guides.get("weight_masked_lure_y",    0.)
            latent_guide_weight_lure_y_inv   = guides.get("weight_unmasked_lure_y",  0.)
            latent_guide_weight_mean         = guides.get("weight_mean",             0.)
            latent_guide_weight_adain        = guides.get("weight_adain",            0.)
            latent_guide_weight_attninj      = guides.get("weight_attninj",          0.)
            latent_guide_weight_style_pos    = guides.get("weight_style_pos",        0.)
            latent_guide_weight_style_neg    = guides.get("weight_style_neg",        0.)
            #latent_guide_synweight_style_pos = guides.get("synweight_style_pos", 0.)
            #latent_guide_synweight_style_neg = guides.get("synweight_style_neg", 0.)
            
            self.drift_x_data                = guides.get("drift_x_data", 0.)
            self.drift_x_sync                = guides.get("drift_x_sync", 0.)
            self.drift_y_data                = guides.get("drift_y_data", 0.)
            self.drift_y_sync                = guides.get("drift_y_sync", 0.)
            self.drift_y_guide               = guides.get("drift_y_guide", 0.)

            latent_guide_weights             = guides.get("weights_masked")
            latent_guide_weights_inv         = guides.get("weights_unmasked")
            latent_guide_weights_sync        = guides.get("weights_masked_sync")
            latent_guide_weights_sync_inv    = guides.get("weights_unmasked_sync")
            latent_guide_weights_drift_x     = guides.get("weights_masked_drift_x")
            latent_guide_weights_drift_x_inv = guides.get("weights_unmasked_drift_x")
            latent_guide_weights_drift_y     = guides.get("weights_masked_drift_y")
            latent_guide_weights_drift_y_inv = guides.get("weights_unmasked_drift_y")
            latent_guide_weights_lure_x      = guides.get("weights_masked_lure_x")
            latent_guide_weights_lure_x_inv  = guides.get("weights_unmasked_lure_x")
            latent_guide_weights_lure_y      = guides.get("weights_masked_lure_y")
            latent_guide_weights_lure_y_inv  = guides.get("weights_unmasked_lure_y")
            latent_guide_weights_mean        = guides.get("weights_mean")
            latent_guide_weights_adain       = guides.get("weights_adain")
            latent_guide_weights_attninj     = guides.get("weights_attninj")
            latent_guide_weights_style_pos   = guides.get("weights_style_pos")
            latent_guide_weights_style_neg   = guides.get("weights_style_neg")
            #latent_guide_synweights_style_p os = guides.get("synweights_style_pos")
            #latent_guide_synweights_style_neg = guides.get("synweights_style_neg")

            latent_guide                     = guides.get("guide_masked")
            latent_guide_inv                 = guides.get("guide_unmasked")
            latent_guide_mean                = guides.get("guide_mean")
            latent_guide_adain               = guides.get("guide_adain")
            latent_guide_attninj             = guides.get("guide_attninj")
            latent_guide_style_pos           = guides.get("guide_style_pos")
            latent_guide_style_neg           = guides.get("guide_style_neg")

            self.mask                        = guides.get("mask")
            self.mask_inv                    = guides.get("unmask")
            self.invert_mask                 = guides.get("invert_mask", False)
            self.mask_sync                   = guides.get("mask_sync")
            self.mask_drift_x                = guides.get("mask_drift_x")
            self.mask_drift_y                = guides.get("mask_drift_y")
            self.mask_lure_x                 = guides.get("mask_lure_x")
            self.mask_lure_y                 = guides.get("mask_lure_y")
            self.mask_mean                   = guides.get("mask_mean")
            self.mask_adain                  = guides.get("mask_adain")
            self.mask_attninj                = guides.get("mask_attninj")
            self.mask_style_pos              = guides.get("mask_style_pos")
            self.mask_style_neg              = guides.get("mask_style_neg")

            scheduler_                       = guides.get("weight_scheduler_masked")
            scheduler_inv_                   = guides.get("weight_scheduler_unmasked")
            scheduler_sync_                  = guides.get("weight_scheduler_masked_sync")
            scheduler_sync_inv_              = guides.get("weight_scheduler_unmasked_sync")
            scheduler_drift_x_               = guides.get("weight_scheduler_masked_drift_x")
            scheduler_drift_x_inv_           = guides.get("weight_scheduler_unmasked_drift_x")
            scheduler_drift_y_               = guides.get("weight_scheduler_masked_drift_y")
            scheduler_drift_y_inv_           = guides.get("weight_scheduler_unmasked_drift_y")
            scheduler_lure_x_                = guides.get("weight_scheduler_masked_lure_x")
            scheduler_lure_x_inv_            = guides.get("weight_scheduler_unmasked_lure_x")
            scheduler_lure_y_                = guides.get("weight_scheduler_masked_lure_y")
            scheduler_lure_y_inv_            = guides.get("weight_scheduler_unmasked_lure_y")
            scheduler_mean_                  = guides.get("weight_scheduler_mean")
            scheduler_adain_                 = guides.get("weight_scheduler_adain")
            scheduler_attninj_               = guides.get("weight_scheduler_attninj")
            scheduler_style_pos_             = guides.get("weight_scheduler_style_pos")
            scheduler_style_neg_             = guides.get("weight_scheduler_style_neg")

            start_steps_                     = guides.get("start_step_masked",   0)
            start_steps_inv_                 = guides.get("start_step_unmasked", 0)
            start_steps_sync_                = guides.get("start_step_masked_sync",   0)
            start_steps_sync_inv_            = guides.get("start_step_unmasked_sync", 0)
            start_steps_drift_x_             = guides.get("start_step_masked_drift_x",   0)
            start_steps_drift_x_inv_         = guides.get("start_step_unmasked_drift_x", 0)
            start_steps_drift_y_             = guides.get("start_step_masked_drift_y",   0)
            start_steps_drift_y_inv_         = guides.get("start_step_unmasked_drift_y", 0)
            start_steps_lure_x_              = guides.get("start_step_masked_lure_x",   0)
            start_steps_lure_x_inv_          = guides.get("start_step_unmasked_lure_x", 0)
            start_steps_lure_y_              = guides.get("start_step_masked_lure_y",   0)
            start_steps_lure_y_inv_          = guides.get("start_step_unmasked_lure_y", 0)
            start_steps_mean_                = guides.get("start_step_mean",     0)
            start_steps_adain_               = guides.get("start_step_adain",    0)
            start_steps_attninj_             = guides.get("start_step_attninj",  0)
            start_steps_style_pos_           = guides.get("start_step_style_pos", 0)
            start_steps_style_neg_           = guides.get("start_step_style_neg", 0)

            steps_                           = guides.get("end_step_masked",     1)
            steps_inv_                       = guides.get("end_step_unmasked",   1)
            steps_sync_                      = guides.get("end_step_masked_sync",     1)
            steps_sync_inv_                  = guides.get("end_step_unmasked_sync",   1)
            steps_drift_x_                   = guides.get("end_step_masked_drift_x",     1)
            steps_drift_x_inv_               = guides.get("end_step_unmasked_drift_x",   1)
            steps_drift_y_                   = guides.get("end_step_masked_drift_y",     1)
            steps_drift_y_inv_               = guides.get("end_step_unmasked_drift_y",   1)
            steps_lure_x_                    = guides.get("end_step_masked_lure_x",     1)
            steps_lure_x_inv_                = guides.get("end_step_unmasked_lure_x",   1)
            steps_lure_y_                    = guides.get("end_step_masked_lure_y",     1)
            steps_lure_y_inv_                = guides.get("end_step_unmasked_lure_y",   1)
            
            steps_mean_                      = guides.get("end_step_mean",       1)
            steps_adain_                     = guides.get("end_step_adain",      1)
            steps_attninj_                   = guides.get("end_step_attninj",    1)
            steps_style_pos_                 = guides.get("end_step_style_pos",  1)
            steps_style_neg_                 = guides.get("end_step_style_neg",  1)

            self.guide_cossim_cutoff_           = guides.get("cutoff_masked",       1.)
            self.guide_bkg_cossim_cutoff_       = guides.get("cutoff_unmasked",     1.)
            self.guide_mean_cossim_cutoff_      = guides.get("cutoff_mean",         1.)
            self.guide_adain_cossim_cutoff_     = guides.get("cutoff_adain",        1.)
            self.guide_attninj_cossim_cutoff_   = guides.get("cutoff_attninj",      1.)
            self.guide_style_pos_cossim_cutoff_ = guides.get("cutoff_style_pos",  1.)
            self.guide_style_neg_cossim_cutoff_ = guides.get("cutoff_style_neg",  1.)
            
            self.sync_lure_iter                 = guides.get("sync_lure_iter",  0)
            self.sync_lure_sequence             = guides.get("sync_lure_sequence")
            
            #self.SYNC_SEPARATE = False
            #if scheduler_sync_ is not None:
            #    self.SYNC_SEPARATE = True
            self.SYNC_SEPARATE = True
            if scheduler_sync_ is None and scheduler_ is not None:

                latent_guide_weight_sync      = latent_guide_weight
                latent_guide_weight_sync_inv  = latent_guide_weight_inv
                latent_guide_weights_sync     = latent_guide_weights
                latent_guide_weights_sync_inv = latent_guide_weights_inv
                
                scheduler_sync_               = scheduler_
                scheduler_sync_inv_           = scheduler_inv_
                
                start_steps_sync_             = start_steps_
                start_steps_sync_inv_         = start_steps_inv_
                
                steps_sync_                   = steps_
                steps_sync_inv_               = steps_inv_
            
            self.SYNC_drift_X = True
            if scheduler_drift_x_ is None and scheduler_ is not None:
                self.SYNC_drift_X = False

                latent_guide_weight_drift_x      = latent_guide_weight
                latent_guide_weight_drift_x_inv  = latent_guide_weight_inv
                latent_guide_weights_drift_x     = latent_guide_weights
                latent_guide_weights_drift_x_inv = latent_guide_weights_inv
                
                scheduler_drift_x_               = scheduler_
                scheduler_drift_x_inv_           = scheduler_inv_
                
                start_steps_drift_x_             = start_steps_
                start_steps_drift_x_inv_         = start_steps_inv_
                
                steps_drift_x_                   = steps_
                steps_drift_x_inv_               = steps_inv_
                
            self.SYNC_drift_Y = True
            if scheduler_drift_y_ is None and scheduler_ is not None:
                self.SYNC_drift_Y = False

                latent_guide_weight_drift_y      = latent_guide_weight
                latent_guide_weight_drift_y_inv  = latent_guide_weight_inv
                latent_guide_weights_drift_y     = latent_guide_weights
                latent_guide_weights_drift_y_inv = latent_guide_weights_inv
                
                scheduler_drift_y_               = scheduler_
                scheduler_drift_y_inv_           = scheduler_inv_
                
                start_steps_drift_y_             = start_steps_
                start_steps_drift_y_inv_         = start_steps_inv_
                
                steps_drift_y_                   = steps_
                steps_drift_y_inv_               = steps_inv_
            
            self.SYNC_LURE_X = True
            if scheduler_lure_x_ is None and scheduler_ is not None:
                self.SYNC_LURE_X = False

                latent_guide_weight_lure_x      = latent_guide_weight
                latent_guide_weight_lure_x_inv  = latent_guide_weight_inv
                latent_guide_weights_lure_x     = latent_guide_weights
                latent_guide_weights_lure_x_inv = latent_guide_weights_inv
                
                scheduler_lure_x_               = scheduler_
                scheduler_lure_x_inv_           = scheduler_inv_
                
                start_steps_lure_x_             = start_steps_
                start_steps_lure_x_inv_         = start_steps_inv_
                
                steps_lure_x_                   = steps_
                steps_lure_x_inv_               = steps_inv_
                
            self.SYNC_LURE_Y = True
            if scheduler_lure_y_ is None and scheduler_ is not None:
                self.SYNC_LURE_Y = False

                latent_guide_weight_lure_y      = latent_guide_weight
                latent_guide_weight_lure_y_inv  = latent_guide_weight_inv
                latent_guide_weights_lure_y     = latent_guide_weights
                latent_guide_weights_lure_y_inv = latent_guide_weights_inv
                
                scheduler_lure_y_               = scheduler_
                scheduler_lure_y_inv_           = scheduler_inv_
                
                start_steps_lure_y_             = start_steps_
                start_steps_lure_y_inv_         = start_steps_inv_
                
                steps_lure_y_                   = steps_
                steps_lure_y_inv_               = steps_inv_

            if self.guide_mode.startswith("fully_") and not RK_IMPLICIT:
                raise ValueError("fully_pseudoimplicit is only supported for implicit RK samplers.")
                #self.guide_mode = self.guide_mode[6:]   # fully_pseudoimplicit is only supported for implicit samplers, default back to pseudoimplicit

            guide_sigma_shift = self.EO("guide_sigma_shift", 0.0)                                                                         # effectively hardcoding shift to 0 !!!!!!
            
            if latent_guide_weights is None and scheduler_ is not None:
                total_steps                     = steps_ - start_steps_
                latent_guide_weights            = get_sigmas(self.model, scheduler_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights            = torch.cat((prepend, latent_guide_weights.to(self.device)), dim=0)
                
            if latent_guide_weights_inv is None and scheduler_inv_ is not None:
                total_steps                     = steps_inv_ - start_steps_inv_
                latent_guide_weights_inv        = get_sigmas(self.model, scheduler_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_inv        = torch.cat((prepend, latent_guide_weights_inv.to(self.device)), dim=0)

            if latent_guide_weights_sync is None and scheduler_sync_ is not None:
                total_steps                     = steps_sync_ - start_steps_sync_
                latent_guide_weights_sync       = get_sigmas(self.model, scheduler_sync_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_sync_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights_sync       = torch.cat((prepend, latent_guide_weights_sync.to(self.device)), dim=0)
                
            if latent_guide_weights_sync_inv is None and scheduler_sync_inv_ is not None:
                total_steps                     = steps_sync_inv_ - start_steps_sync_inv_
                latent_guide_weights_sync_inv   = get_sigmas(self.model, scheduler_sync_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_sync_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_sync_inv   = torch.cat((prepend, latent_guide_weights_sync_inv.to(self.device)), dim=0)
                
            if latent_guide_weights_drift_x is None and scheduler_drift_x_ is not None:
                total_steps                     = steps_drift_x_ - start_steps_drift_x_
                latent_guide_weights_drift_x    = get_sigmas(self.model, scheduler_drift_x_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_drift_x_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights_drift_x    = torch.cat((prepend, latent_guide_weights_drift_x.to(self.device)), dim=0)
                
            if latent_guide_weights_drift_x_inv is None and scheduler_drift_x_inv_ is not None:
                total_steps                      = steps_drift_x_inv_ - start_steps_drift_x_inv_
                latent_guide_weights_drift_x_inv = get_sigmas(self.model, scheduler_drift_x_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                          = torch.zeros(start_steps_drift_x_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_drift_x_inv = torch.cat((prepend, latent_guide_weights_drift_x_inv.to(self.device)), dim=0)
                
            if latent_guide_weights_drift_y is None and scheduler_drift_y_ is not None:
                total_steps                     = steps_drift_y_ - start_steps_drift_y_
                latent_guide_weights_drift_y    = get_sigmas(self.model, scheduler_drift_y_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_drift_y_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights_drift_y    = torch.cat((prepend, latent_guide_weights_drift_y.to(self.device)), dim=0)
                
            if latent_guide_weights_drift_y_inv is None and scheduler_drift_y_inv_ is not None:
                total_steps                      = steps_drift_y_inv_ - start_steps_drift_y_inv_
                latent_guide_weights_drift_y_inv = get_sigmas(self.model, scheduler_drift_y_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                          = torch.zeros(start_steps_drift_y_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_drift_y_inv = torch.cat((prepend, latent_guide_weights_drift_y_inv.to(self.device)), dim=0)
                
            if latent_guide_weights_lure_x is None and scheduler_lure_x_ is not None:
                total_steps                     = steps_lure_x_ - start_steps_lure_x_
                latent_guide_weights_lure_x     = get_sigmas(self.model, scheduler_lure_x_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_lure_x_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights_lure_x     = torch.cat((prepend, latent_guide_weights_lure_x.to(self.device)), dim=0)
                
            if latent_guide_weights_lure_x_inv is None and scheduler_lure_x_inv_ is not None:
                total_steps                     = steps_lure_x_inv_ - start_steps_lure_x_inv_
                latent_guide_weights_lure_x_inv = get_sigmas(self.model, scheduler_lure_x_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_lure_x_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_lure_x_inv = torch.cat((prepend, latent_guide_weights_lure_x_inv.to(self.device)), dim=0)
                
            if latent_guide_weights_lure_y is None and scheduler_lure_y_ is not None:
                total_steps                     = steps_lure_y_ - start_steps_lure_y_
                latent_guide_weights_lure_y     = get_sigmas(self.model, scheduler_lure_y_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_lure_y_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights_lure_y     = torch.cat((prepend, latent_guide_weights_lure_y.to(self.device)), dim=0)
                
            if latent_guide_weights_lure_y_inv is None and scheduler_lure_y_inv_ is not None:
                total_steps                     = steps_lure_y_inv_ - start_steps_lure_y_inv_
                latent_guide_weights_lure_y_inv = get_sigmas(self.model, scheduler_lure_y_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_lure_y_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_lure_y_inv = torch.cat((prepend, latent_guide_weights_lure_y_inv.to(self.device)), dim=0)
                

            if latent_guide_weights_mean is None and scheduler_mean_ is not None:
                total_steps                     = steps_mean_ - start_steps_mean_
                latent_guide_weights_mean       = get_sigmas(self.model, scheduler_mean_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_mean_,                                                        dtype=self.dtype, device=self.device) 
                latent_guide_weights_mean       = torch.cat((prepend, latent_guide_weights_mean.to(self.device)), dim=0)
            
            if latent_guide_weights_adain is None and scheduler_adain_ is not None:
                total_steps                     = steps_adain_ - start_steps_adain_
                latent_guide_weights_adain      = get_sigmas(self.model, scheduler_adain_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_adain_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_adain      = torch.cat((prepend, latent_guide_weights_adain.to(self.device)), dim=0)
            
            if latent_guide_weights_attninj is None and scheduler_attninj_ is not None:
                total_steps                     = steps_attninj_ - start_steps_attninj_
                latent_guide_weights_attninj    = get_sigmas(self.model, scheduler_attninj_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_attninj_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_attninj    = torch.cat((prepend, latent_guide_weights_attninj.to(self.device)), dim=0)
            
            if latent_guide_weights_style_pos is None and scheduler_style_pos_ is not None:
                total_steps                     = steps_style_pos_ - start_steps_style_pos_
                latent_guide_weights_style_pos  = get_sigmas(self.model, scheduler_style_pos_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_style_pos_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_style_pos  = torch.cat((prepend, latent_guide_weights_style_pos.to(self.device)), dim=0)
            
            if latent_guide_weights_style_neg is None and scheduler_style_neg_ is not None:
                total_steps                     = steps_style_neg_ - start_steps_style_neg_
                latent_guide_weights_style_neg  = get_sigmas(self.model, scheduler_style_neg_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                         = torch.zeros(start_steps_style_neg_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_style_neg  = torch.cat((prepend, latent_guide_weights_style_neg.to(self.device)), dim=0)
            
            if scheduler_ != "constant":
                latent_guide_weights            = initialize_or_scale(latent_guide_weights,      latent_guide_weight,      self.max_steps)
            if scheduler_inv_ != "constant":
                latent_guide_weights_inv        = initialize_or_scale(latent_guide_weights_inv,  latent_guide_weight_inv,  self.max_steps)
            if scheduler_sync_ != "constant":
                latent_guide_weights_sync       = initialize_or_scale(latent_guide_weights_sync,      latent_guide_weight_sync,      self.max_steps)
            if scheduler_sync_inv_ != "constant": 
                latent_guide_weights_sync_inv   = initialize_or_scale(latent_guide_weights_sync_inv,  latent_guide_weight_sync_inv,  self.max_steps)
                
            latent_guide_weights_sync     = 1 - latent_guide_weights_sync     if latent_guide_weights_sync     is not None else latent_guide_weights
            latent_guide_weights_sync_inv = 1 - latent_guide_weights_sync_inv if latent_guide_weights_sync_inv is not None else latent_guide_weights_inv
            latent_guide_weight_sync      = 1 - latent_guide_weight_sync
            latent_guide_weight_sync_inv  = 1 - latent_guide_weight_sync_inv# these are more intuitive to use if these are reversed... so that sync weight = 1.0 means "maximum guide strength"
            
            
            if scheduler_drift_x_ != "constant": 
                latent_guide_weights_drift_x     = initialize_or_scale(latent_guide_weights_drift_x,      latent_guide_weight_drift_x,      self.max_steps)
            if scheduler_drift_x_inv_ != "constant": 
                latent_guide_weights_drift_x_inv = initialize_or_scale(latent_guide_weights_drift_x_inv,  latent_guide_weight_drift_x_inv,  self.max_steps)
            if scheduler_drift_y_ != "constant": 
                latent_guide_weights_drift_y     = initialize_or_scale(latent_guide_weights_drift_y,      latent_guide_weight_drift_y,      self.max_steps)
            if scheduler_drift_y_inv_ != "constant": 
                latent_guide_weights_drift_y_inv = initialize_or_scale(latent_guide_weights_drift_y_inv,  latent_guide_weight_drift_y_inv,  self.max_steps)
            if scheduler_lure_x_ != "constant": 
                latent_guide_weights_lure_x      = initialize_or_scale(latent_guide_weights_lure_x,      latent_guide_weight_lure_x,      self.max_steps)
            if scheduler_lure_x_inv_ != "constant": 
                latent_guide_weights_lure_x_inv  = initialize_or_scale(latent_guide_weights_lure_x_inv,  latent_guide_weight_lure_x_inv,  self.max_steps)
            if scheduler_lure_y_ != "constant": 
                latent_guide_weights_lure_y      = initialize_or_scale(latent_guide_weights_lure_y,      latent_guide_weight_lure_y,      self.max_steps)
            if scheduler_lure_y_inv_ != "constant": 
                latent_guide_weights_lure_y_inv  = initialize_or_scale(latent_guide_weights_lure_y_inv,  latent_guide_weight_lure_y_inv,  self.max_steps)
            if scheduler_mean_ != "constant": 
                latent_guide_weights_mean        = initialize_or_scale(latent_guide_weights_mean, latent_guide_weight_mean, self.max_steps)
            if scheduler_adain_ != "constant": 
                latent_guide_weights_adain       = initialize_or_scale(latent_guide_weights_adain, latent_guide_weight_adain, self.max_steps)
            if scheduler_attninj_ != "constant": 
                latent_guide_weights_attninj     = initialize_or_scale(latent_guide_weights_attninj, latent_guide_weight_attninj, self.max_steps)
            if scheduler_style_pos_ != "constant": 
                latent_guide_weights_style_pos   = initialize_or_scale(latent_guide_weights_style_pos, latent_guide_weight_style_pos, self.max_steps)
            if scheduler_style_neg_ != "constant": 
                latent_guide_weights_style_neg   = initialize_or_scale(latent_guide_weights_style_neg, latent_guide_weight_style_neg, self.max_steps)

            latent_guide_weights            [steps_            :] = 0
            latent_guide_weights_inv        [steps_inv_        :] = 0
            latent_guide_weights_sync       [steps_sync_       :] = 1 #one
            latent_guide_weights_sync_inv   [steps_sync_inv_   :] = 1 #one
            latent_guide_weights_drift_x    [steps_drift_x_    :] = 0
            latent_guide_weights_drift_x_inv[steps_drift_x_inv_:] = 0
            latent_guide_weights_drift_y    [steps_drift_y_    :] = 0
            latent_guide_weights_drift_y_inv[steps_drift_y_inv_:] = 0
            latent_guide_weights_lure_x     [steps_lure_x_     :] = 0
            latent_guide_weights_lure_x_inv [steps_lure_x_inv_ :] = 0
            latent_guide_weights_lure_y     [steps_lure_y_     :] = 0
            latent_guide_weights_lure_y_inv [steps_lure_y_inv_ :] = 0
            latent_guide_weights_mean       [steps_mean_       :] = 0
            latent_guide_weights_adain      [steps_adain_      :] = 0
            latent_guide_weights_attninj    [steps_attninj_    :] = 0
            latent_guide_weights_style_pos  [steps_style_pos_  :] = 0
            latent_guide_weights_style_neg  [steps_style_neg_  :] = 0
        
        self.lgw             = F.pad(latent_guide_weights,             (0, self.max_steps), value=0.0)
        self.lgw_inv         = F.pad(latent_guide_weights_inv,         (0, self.max_steps), value=0.0)
        self.lgw_sync        = F.pad(latent_guide_weights_sync,        (0, self.max_steps), value=1.0) #one
        self.lgw_sync_inv    = F.pad(latent_guide_weights_sync_inv,    (0, self.max_steps), value=1.0) #one
        self.lgw_drift_x     = F.pad(latent_guide_weights_drift_x,     (0, self.max_steps), value=0.0)
        self.lgw_drift_x_inv = F.pad(latent_guide_weights_drift_x_inv, (0, self.max_steps), value=0.0)
        self.lgw_drift_y     = F.pad(latent_guide_weights_drift_y,     (0, self.max_steps), value=0.0)
        self.lgw_drift_y_inv = F.pad(latent_guide_weights_drift_y_inv, (0, self.max_steps), value=0.0)
        self.lgw_lure_x      = F.pad(latent_guide_weights_lure_x,      (0, self.max_steps), value=0.0)
        self.lgw_lure_x_inv  = F.pad(latent_guide_weights_lure_x_inv,  (0, self.max_steps), value=0.0)
        self.lgw_lure_y      = F.pad(latent_guide_weights_lure_y,      (0, self.max_steps), value=0.0)
        self.lgw_lure_y_inv  = F.pad(latent_guide_weights_lure_y_inv,  (0, self.max_steps), value=0.0)
        self.lgw_mean        = F.pad(latent_guide_weights_mean,        (0, self.max_steps), value=0.0)
        self.lgw_adain       = F.pad(latent_guide_weights_adain,       (0, self.max_steps), value=0.0)
        self.lgw_attninj     = F.pad(latent_guide_weights_attninj,     (0, self.max_steps), value=0.0)
        self.lgw_style_pos   = F.pad(latent_guide_weights_style_pos,   (0, self.max_steps), value=0.0)
        self.lgw_style_neg   = F.pad(latent_guide_weights_style_neg,   (0, self.max_steps), value=0.0)
        
        mask, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask, self.LGW_MASK_RESCALE_MIN)
        self.mask = mask.to(dtype=self.dtype, device=self.device)

        if self.mask_inv is not None:
            mask_inv, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_inv, self.LGW_MASK_RESCALE_MIN)
            self.mask_inv = mask_inv.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_inv = (1-self.mask)
            
        if self.mask_sync is not None:
            mask_sync, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_sync, self.LGW_MASK_RESCALE_MIN)
            self.mask_sync = mask_sync.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_sync = self.mask
            
        if self.mask_drift_x is not None:
            mask_drift_x, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_drift_x, self.LGW_MASK_RESCALE_MIN)
            self.mask_drift_x = mask_drift_x.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_drift_x = self.mask
            
        if self.mask_drift_y is not None:
            mask_drift_y, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_drift_y, self.LGW_MASK_RESCALE_MIN)
            self.mask_drift_y = mask_drift_y.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_drift_y = self.mask
            
        if self.mask_lure_x is not None:
            mask_lure_x, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_lure_x, self.LGW_MASK_RESCALE_MIN)
            self.mask_lure_x = mask_lure_x.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_lure_x = self.mask
            
        if self.mask_lure_y is not None:
            mask_lure_y, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_lure_y, self.LGW_MASK_RESCALE_MIN)
            self.mask_lure_y = mask_lure_y.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_lure_y = self.mask
            
        mask_style_pos, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_style_pos, self.LGW_MASK_RESCALE_MIN)
        self.mask_style_pos = mask_style_pos.to(dtype=self.dtype, device=self.device)

        
        mask_style_neg, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_style_neg, self.LGW_MASK_RESCALE_MIN)
        self.mask_style_neg = mask_style_neg.to(dtype=self.dtype, device=self.device)
    
        if latent_guide is not None:
            self.HAS_LATENT_GUIDE = True
            if type(latent_guide) is dict:
                latent_guide_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide) is torch.Tensor:
                latent_guide_samples = latent_guide.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide)}")

            latent_guide_samples = flatten_to_match(latent_guide_samples, x).clone()

            if self.SAMPLE:
                self.y0 = latent_guide_samples
            elif sigma_init != 0.0:
                pass
            elif self.UNSAMPLE: # and self.mask is not None:
                mask = self.mask.to(x.device)
                x = (1-mask) * x + mask * latent_guide_samples.to(x.device)
            else:
                x = latent_guide_samples.to(x.device)
        else:
            self.y0 = torch.zeros_like(x, dtype=self.dtype, device=self.device)

        # Initialize self_refine_epsilon mode (including projection variant via _projection suffix)
        self.SELF_REFINE_EPSILON_MODE = self.guide_mode.startswith("self_refine_epsilon")
        if self.SELF_REFINE_EPSILON_MODE:
            self.HAS_LATENT_GUIDE = True  # Enable guide processing
            self.y0 = torch.zeros_like(x, dtype=self.dtype, device=self.device)  # y0 will be set dynamically to denoised_prev
            self.self_refine_epsilon_ref = None  # Reference for within-step refinement
            self.self_refine_epsilon_last_step = -1  # Track which step we're on
            self.self_refine_epsilon_last_row = -1
            self.self_refine_epsilon_call_count = 0
        # Per-iteration tracking (for self_refine_per_iteration mode)
        self._self_refine_last_iter = -1
        self._self_refine_iter_prediction = None
        self._self_refine_certain_mask_accum = None
        self._debug_certainty_mask = None

        if latent_guide_inv is not None:
            self.HAS_LATENT_GUIDE_INV = True
            if type(latent_guide_inv) is dict:
                latent_guide_inv_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_inv) is torch.Tensor:
                latent_guide_inv_samples = latent_guide_inv.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_inv)}")

            latent_guide_inv_samples = flatten_to_match(latent_guide_inv_samples, x).clone()

            if self.SAMPLE:
                self.y0_inv = latent_guide_inv_samples
            elif sigma_init != 0.0:
                pass
            elif self.UNSAMPLE: # and self.mask is not None:
                mask_inv = self.mask_inv.to(x.device)
                x = (1-mask_inv) * x + mask_inv * latent_guide_inv_samples.to(x.device) #fixed old approach, which was mask, (1-mask)
            else:
                x = latent_guide_inv_samples.to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_INV AFTER SETTING TO LG above!
        else:
            self.y0_inv = torch.zeros_like(x, dtype=self.dtype, device=self.device)

        if latent_guide_mean is not None:
            self.HAS_LATENT_GUIDE_MEAN = True
            if type(latent_guide_mean) is dict:
                latent_guide_mean_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_mean['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_mean) is torch.Tensor:
                latent_guide_mean_samples = latent_guide_mean.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_mean)}")

            latent_guide_mean_samples = flatten_to_match(latent_guide_mean_samples, x).clone()
            self.y0_mean = latent_guide_mean_samples
            """if self.SAMPLE:
                self.y0_mean = latent_guide_mean_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                mask_mean = self.mask_mean.to(x.device)
                x = (1-mask_mean) * x + mask_mean * latent_guide_mean_samples.to(x.device) #fixed old approach, which was mask, (1-mask)     # NECESSARY?
            else:
                x = latent_guide_mean_samples.to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_MEAN AFTER SETTING TO LG above!"""
        else:
            self.y0_mean = torch.zeros_like(x, dtype=self.dtype, device=self.device)

        if latent_guide_adain is not None:
            self.HAS_LATENT_GUIDE_ADAIN = True
            if type(latent_guide_adain) is dict:
                latent_guide_adain_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_adain['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_adain) is torch.Tensor:
                latent_guide_adain_samples = latent_guide_adain.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_adain)}")

            latent_guide_adain_samples = flatten_to_match(latent_guide_adain_samples, x).clone()
            self.y0_adain = latent_guide_adain_samples
            """if self.SAMPLE:
                self.y0_adain = latent_guide_adain_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                if self.mask_adain is not None:
                    mask_adain = self.mask_adain.to(x.device)
                    x = (1-mask_adain) * x + mask_adain * latent_guide_adain_samples.to(x.device) #fixed old approach, which was mask, (1-mask)     # NECESSARY?
                else:
                    x = latent_guide_adain_samples.to(x.device)
            else:
                x = latent_guide_adain_samples.to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_ADAIN AFTER SETTING TO LG above!"""
        else:
            self.y0_adain = torch.zeros_like(x, dtype=self.dtype, device=self.device)

        if latent_guide_attninj is not None:
            self.HAS_LATENT_GUIDE_ATTNINJ = True
            if type(latent_guide_attninj) is dict:
                latent_guide_attninj_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_attninj['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_attninj) is torch.Tensor:
                latent_guide_attninj_samples = latent_guide_attninj.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_attninj)}")

            latent_guide_attninj_samples = flatten_to_match(latent_guide_attninj_samples, x).clone()
            self.y0_attninj = latent_guide_attninj_samples
            """if self.SAMPLE:
                self.y0_attninj = latent_guide_attninj_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                if self.mask_attninj is not None:
                    mask_attninj = self.mask_attninj.to(x.device)
                    x = (1-mask_attninj) * x + mask_attninj * latent_guide_attninj_samples.to(x.device) #fixed old approach, which was mask, (1-mask)     # NECESSARY?
                else:
                    x = latent_guide_attninj_samples.to(x.device)  
            else:
                x = latent_guide_attninj_samples.to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_ADAIN AFTER SETTING TO LG above!"""
        else:
            self.y0_attninj = torch.zeros_like(x, dtype=self.dtype, device=self.device)


        if latent_guide_style_pos is not None:
            self.HAS_LATENT_GUIDE_STYLE_POS = True
            if type(latent_guide_style_pos) is dict:
                latent_guide_style_pos_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_style_pos['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_style_pos) is torch.Tensor:
                latent_guide_style_pos_samples = latent_guide_style_pos.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_style_pos)}")

            latent_guide_style_pos_samples = flatten_to_match(latent_guide_style_pos_samples, x).clone()
            self.y0_style_pos = latent_guide_style_pos_samples
            """if self.SAMPLE:
                self.y0_style_pos = latent_guide_style_pos_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                if self.mask_style_pos is not None:
                    mask_style_pos = self.mask_style_pos.to(x.device)
                    x = (1-mask_style_pos) * x + mask_style_pos * latent_guide_style_pos_samples.to(x.device) #fixed old approach, which was mask, (1-mask)     # NECESSARY?
                else:
                    x = latent_guide_style_pos_samples.to(x.device)  
            else:
                x = latent_guide_style_pos_samples.to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_ADAIN AFTER SETTING TO LG above!"""
        else:
            self.y0_style_pos = torch.zeros_like(x, dtype=self.dtype, device=self.device)


        if latent_guide_style_neg is not None:
            self.HAS_LATENT_GUIDE_STYLE_NEG = True
            if type(latent_guide_style_neg) is dict:
                latent_guide_style_neg_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_style_neg['samples']).to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_style_neg) is torch.Tensor:
                latent_guide_style_neg_samples = latent_guide_style_neg.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_style_neg)}")

            latent_guide_style_neg_samples = flatten_to_match(latent_guide_style_neg_samples, x).clone()
            self.y0_style_neg = latent_guide_style_neg_samples
            """if self.SAMPLE:
                self.y0_style_neg = latent_guide_style_neg_samples
            elif self.UNSAMPLE: # and self.mask is not None:
                if self.mask_style_neg is not None:
                    mask_style_neg = self.mask_style_neg.to(x.device)
                    x = (1-mask_style_neg) * x + mask_style_neg * latent_guide_style_neg_samples.to(x.device) #fixed old approach, which was mask, (1-mask)     # NECESSARY?
                else:
                    x = latent_guide_style_neg_samples.to(x.device)  
            else:
                x = latent_guide_style_neg_samples.to(x.device)   #THIS COULD LEAD TO WEIRD BEHAVIOR! OVERWRITING X WITH LG_ADAIN AFTER SETTING TO LG above!"""
        else:
            self.y0_style_neg = torch.zeros_like(x, dtype=self.dtype, device=self.device)

        if self.UNSAMPLE and not self.SAMPLE: #sigma_next > sigma:   # TODO: VERIFY APPROACH FOR INVERSION
            if guide_inversion_y0 is not None:
                self.y0 = guide_inversion_y0.clone()
            else:
                self.y0     = noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min).to(dtype=self.dtype, device=self.device)
                self.y0     = normalize_zscore(self.y0,     channelwise=True, inplace=True)
                self.y0    *= self.sigma_max

            if guide_inversion_y0_inv is not None:
                self.y0_inv = guide_inversion_y0_inv.clone()
            else:
                self.y0_inv = noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min).to(dtype=self.dtype, device=self.device)
                self.y0_inv = normalize_zscore(self.y0_inv, channelwise=True, inplace=True)
                self.y0_inv*= self.sigma_max

            
        if self.frame_weights_mgr is not None and x.ndim == 5:
            num_frames = x.shape[2]
            self.frame_weights     = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights', num_frames)
            self.frame_weights_inv = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights_inv', num_frames)
            
        x, self.y0, self.y0_inv = self.normalize_inputs(x, self.y0, self.y0_inv)       # ???

        return x

    def prepare_weighted_masks(self, step:int, lgw_type="default") -> Tuple[Tensor, Tensor]:
        if lgw_type == "sync":
            lgw_     = self.lgw_sync    [step]
            lgw_inv_ = self.lgw_sync_inv[step]
            mask     = torch.ones_like (self.y0) if self.mask_sync is None else   self.mask_sync
            mask_inv = torch.zeros_like(self.y0) if self.mask_sync is None else 1-self.mask_sync
        elif lgw_type == "drift_x":
            lgw_     = self.lgw_drift_x    [step]
            lgw_inv_ = self.lgw_drift_x_inv[step]
            mask     = torch.ones_like (self.y0) if self.mask_drift_x is None else   self.mask_drift_x
            mask_inv = torch.zeros_like(self.y0) if self.mask_drift_x is None else 1-self.mask_drift_x
        elif lgw_type == "drift_y":
            lgw_     = self.lgw_drift_y    [step]
            lgw_inv_ = self.lgw_drift_y_inv[step]
            mask     = torch.ones_like (self.y0) if self.mask_drift_y is None else   self.mask_drift_y
            mask_inv = torch.zeros_like(self.y0) if self.mask_drift_y is None else 1-self.mask_drift_y
        elif lgw_type == "lure_x":
            lgw_     = self.lgw_lure_x    [step]
            lgw_inv_ = self.lgw_lure_x_inv[step]
            mask     = torch.ones_like (self.y0) if self.mask_lure_x is None else   self.mask_lure_x
            mask_inv = torch.zeros_like(self.y0) if self.mask_lure_x is None else 1-self.mask_lure_x
        elif lgw_type == "lure_y":
            lgw_     = self.lgw_lure_y    [step]
            lgw_inv_ = self.lgw_lure_y_inv[step]
            mask     = torch.ones_like (self.y0) if self.mask_lure_y is None else   self.mask_lure_y
            mask_inv = torch.zeros_like(self.y0) if self.mask_lure_y is None else 1-self.mask_lure_y
        else:
            lgw_     = self.lgw    [step]
            lgw_inv_ = self.lgw_inv[step]
            mask     = torch.ones_like (self.y0) if self.mask     is None else self.mask
            mask_inv = torch.zeros_like(self.y0) if self.mask_inv is None else self.mask_inv

        if self.LGW_MASK_RESCALE_MIN:
            lgw_mask     =    mask  * (1-lgw_)     + lgw_
            lgw_mask_inv = (1-mask) * (1-lgw_inv_) + lgw_inv_
        else:
            if self.HAS_LATENT_GUIDE:
                lgw_mask = mask * lgw_
            else:
                lgw_mask = torch.zeros_like(mask)
            
            if self.HAS_LATENT_GUIDE_INV:
                if mask_inv is not None:
                    lgw_mask_inv = torch.minimum(mask_inv, (1-mask) * lgw_inv_)
                    #lgw_mask_inv = torch.minimum(1-mask_inv, (1-mask) * lgw_inv_)
                else:
                    lgw_mask_inv = (1-mask) * lgw_inv_
            else:
                lgw_mask_inv = torch.zeros_like(mask)

        return lgw_mask, lgw_mask_inv


    def get_masks_for_step(self, step:int, lgw_type="default") -> Tuple[Tensor, Tensor]:
        lgw_mask, lgw_mask_inv = self.prepare_weighted_masks(step, lgw_type=lgw_type)
        normalize_frame_weights_per_step = self.EO("normalize_frame_weights_per_step")
        normalize_frame_weights_per_step_inv = self.EO("normalize_frame_weights_per_step_inv")

        if self.VIDEO and self.frame_weights_mgr and lgw_mask.ndim >= 5:
            num_frames = lgw_mask.shape[2]
            if self.HAS_LATENT_GUIDE:
                frame_weights = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights', num_frames, step)
                apply_frame_weights(lgw_mask, frame_weights, normalize_frame_weights_per_step)
            if self.HAS_LATENT_GUIDE_INV:
                frame_weights_inv = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights_inv', num_frames, step)
                apply_frame_weights(lgw_mask_inv, frame_weights_inv, normalize_frame_weights_per_step_inv)

        return lgw_mask.to(self.device), lgw_mask_inv.to(self.device)



    def get_cossim_adjusted_lgw_masks(self, data:Tensor, step:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # PACK-FIRST: Require flat [1,1,N] input - callers must pack first
        data_for_cossim = data

        if self.HAS_LATENT_GUIDE:
            y0     = self.y0.clone()
        else:
            y0     = torch.zeros_like(data_for_cossim)

        if self.HAS_LATENT_GUIDE_INV:
            y0_inv = self.y0_inv.clone()
        else:
            y0_inv = torch.zeros_like(data_for_cossim)

        if y0.shape[0] > 1:                                    # this is for changing the guide on a per-step basis
            y0 = y0[min(step, y0.shape[0]-1)].unsqueeze(0)

        lgw_mask, lgw_mask_inv = self.get_masks_for_step(step)

        y0_cossim, y0_cossim_inv  = 1.0, 1.0
        if self.HAS_LATENT_GUIDE:
            y0_cossim     = get_pearson_similarity(data_for_cossim, y0,     mask=lgw_mask)
        if self.HAS_LATENT_GUIDE_INV:
            y0_cossim_inv = get_pearson_similarity(data_for_cossim, y0_inv, mask=lgw_mask_inv)

        #if y0_cossim < self.guide_cossim_cutoff_ or y0_cossim_inv < self.guide_bkg_cossim_cutoff_:
        if y0_cossim     >= self.guide_cossim_cutoff_:
            lgw_mask     *= 0
        if y0_cossim_inv >= self.guide_bkg_cossim_cutoff_:
            lgw_mask_inv *= 0

        return y0, y0_inv, lgw_mask, lgw_mask_inv


    def get_self_refine_epsilon_mask(self, current: Tensor, previous: Tensor, step_sched: int) -> Tensor:
        """
        Compute per-pixel certainty mask for self_refine_epsilon mode.
        Returns mask where 1 = certain (guide), 0 = uncertain (no guide).

        When invert_mask=False (default): guide CERTAIN (low-diff/stable) regions
        When invert_mask=True: guide UNCERTAIN (high-diff/changing) regions
        """
        threshold = self.self_refine_threshold
        metric = self.self_refine_metric

        if metric == "l2":
            # Normalized L2 (Euclidean distance per pixel, normalized by channel count)
            diff = current - previous
            if diff.ndim >= 4 and diff.shape[1] > 1:
                diff = torch.sqrt(torch.sum(diff ** 2, dim=1, keepdim=True)) / diff.shape[1]
            else:
                diff = torch.abs(diff)
        else:
            # L1 (absolute difference, averaged over channels)
            diff = torch.abs(current - previous)
            if diff.ndim >= 4 and diff.shape[1] > 1:
                diff = diff.mean(dim=1, keepdim=True)

        # Certain = low diff (BELOW threshold)
        # Uncertain = high diff (ABOVE threshold)
        certain_mask = (diff < threshold).float()

        # Apply invert_mask: if True, guide uncertain regions instead of certain
        if self.invert_mask:
            certain_mask = 1.0 - certain_mask

        # Apply spatial mask from ClownGuides if provided (intersection)
        if self.mask is not None:
            spatial_mask = self.mask
            if spatial_mask.shape != certain_mask.shape:
                if spatial_mask.ndim == certain_mask.ndim:
                    if spatial_mask.shape[1] == 1 and certain_mask.shape[1] > 1:
                        spatial_mask = spatial_mask.expand_as(certain_mask)
                    elif certain_mask.shape[1] == 1 and spatial_mask.shape[1] > 1:
                        spatial_mask = spatial_mask.mean(dim=1, keepdim=True)
            certain_mask = certain_mask * spatial_mask

        # Apply guide weight schedule
        lgw = self.lgw[step_sched] if step_sched < len(self.lgw) else 0.0

        if self.EO("self_refine_debug"):
            coverage = certain_mask.mean().item()
            mode = "UNCERTAIN (inverted)" if self.invert_mask else "CERTAIN"
            spatial_info = " (with spatial mask)" if self.mask is not None else ""
            RESplain(f"self_refine_epsilon step {step_sched}: guiding {mode} regions{spatial_info}, coverage={coverage:.2%}, metric={metric}, threshold={threshold}, lgw={lgw:.4f}")

        # Store raw mask for visualization (before lgw scaling)
        self._debug_certainty_mask = certain_mask.clone()

        return certain_mask * lgw










    @torch.no_grad
    def process_pseudoimplicit_guides_substep(self,
                                            x_0                         : Tensor,
                                            x_                          : Tensor,
                                            eps_                        : Tensor,
                                            eps_prev_                   : Tensor,
                                            data_                       : Tensor,
                                            denoised_prev               : Tensor,
                                            row                         : int,
                                            step                        : int,
                                            step_sched                  : int,
                                            sigmas                      : Tensor,
                                            NS                                  ,
                                            RK                                  ,
                                            pseudoimplicit_row_weights  : Tensor,
                                            pseudoimplicit_step_weights : Tensor,
                                            full_iter                   : int,
                                            BONGMATH                    : bool,
                                            ):

        # Check if this is a pseudoimplicit mode (including self_refine_pseudoimplicit variants)
        is_pseudoimplicit_mode = "pseudoimplicit" in self.guide_mode or self.guide_mode.startswith("self_refine_pseudoimplicit")
        if not is_pseudoimplicit_mode or (self.lgw[step_sched] == 0 and self.lgw_inv[step_sched] == 0):
            return x_0, x_, eps_, None, None

        if x_0.ndim == 3:  # packed NestedTensor
            BLOCKED_PSEUDOIMPLICIT_MODES = {"pseudoimplicit_cw", "pseudoimplicit_projection_cw",
                                            "fully_pseudoimplicit_cw", "fully_pseudoimplicit_projection_cw"}
            if self.guide_mode in BLOCKED_PSEUDOIMPLICIT_MODES:
                raise NotImplementedError(f"Mode '{self.guide_mode}' requires channel structure, incompatible with packed latents")

        sigma = sigmas[step]

        # Handle self_refine_pseudoimplicit modes
        if self.guide_mode.startswith("self_refine_pseudoimplicit"):
            # Per-iteration mode: track changes across implicit iterations
            per_iteration_mode = not self.EO("self_refine_by_step")

            # Skip conditions
            if per_iteration_mode:
                if step == 0 and full_iter == 0:
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_pseudoimplicit step {step}, iter {full_iter}, row {row}: SKIPPED - no valid reference")
                    return x_0, x_, eps_, None, None
            else:
                if step == 0 or denoised_prev.abs().max() == 0:
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_pseudoimplicit step {step}, row {row}: SKIPPED - no valid denoised_prev")
                    return x_0, x_, eps_, None, None

            is_new_step = (step != self.self_refine_epsilon_last_step)
            is_new_iter = (full_iter != self._self_refine_last_iter) if per_iteration_mode else False

            # Reference management
            if per_iteration_mode:
                if is_new_step:
                    # New step: reset everything
                    self.self_refine_epsilon_ref = denoised_prev.clone()
                    self.self_refine_epsilon_last_step = step
                    self._self_refine_last_iter = full_iter
                    self._self_refine_certain_mask_accum = None
                    self._self_refine_iter_prediction = None
                    self._self_refine_converged = False
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_pseudoimplicit step {step}: NEW STEP - initialized reference from denoised_prev")

                elif is_new_iter:
                    # New iteration: update reference to previous iteration's prediction
                    if self._self_refine_iter_prediction is not None:
                        self.self_refine_epsilon_ref = self._self_refine_iter_prediction.clone()
                        if self.EO("self_refine_debug"):
                            RESplain(f"self_refine_pseudoimplicit step {step}, iter {full_iter}: updated reference from iter {full_iter-1}")
                    self._self_refine_last_iter = full_iter
                    if self.EO("self_refine_dont_accumulate_certainty"):
                        self._self_refine_certain_mask_accum = None

                if row == 0:
                    self._self_refine_iter_prediction = data_[row].clone()
            else:
                # Non-iterative mode: update reference at each step
                if is_new_step:
                    self.self_refine_epsilon_ref = denoised_prev.clone()
                    self.self_refine_epsilon_last_step = step
                    self._self_refine_converged = False
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_pseudoimplicit step {step}: initialized reference from denoised_prev")

            # Use reference as guide target
            y0 = self.self_refine_epsilon_ref

            # Compute certainty mask
            lgw_mask = self.get_self_refine_epsilon_mask(data_[row], y0, step_sched)

            # Accumulate certainty across iterations
            if per_iteration_mode and not self.EO("self_refine_dont_accumulate_certainty"):
                if self._self_refine_certain_mask_accum is not None:
                    binary_mask = (lgw_mask > 0).float()
                    binary_accum = (self._self_refine_certain_mask_accum > 0).float()
                    combined = torch.maximum(binary_mask, binary_accum)
                    lgw = self.lgw[step_sched] if step_sched < len(self.lgw) else 0.0
                    lgw_mask = combined * lgw
                self._self_refine_certain_mask_accum = lgw_mask.clone()

            if lgw_mask.max() == 0:
                if self.EO("self_refine_debug"):
                    RESplain(f"self_refine_pseudoimplicit step {step}, row {row}: SKIPPED - no certain regions")
                return x_0, x_, eps_, None, None

            # Check coverage against cutoff
            coverage = (lgw_mask > 0).float().mean().item()
            if coverage >= self.self_refine_cutoff:
                self._self_refine_converged = True
                if self.EO("self_refine_debug"):
                    iter_info = f", iter {full_iter}" if per_iteration_mode else ""
                    RESplain(f"self_refine_pseudoimplicit step {step}{iter_info}, row {row}: CONVERGED - coverage={coverage:.2%} >= cutoff={self.self_refine_cutoff:.2%}")

            # Compute guide epsilon
            eps_substep_guide = RK.get_guide_epsilon(x_0, x_[row], y0, sigma, NS.s_[row], NS.sigma_down, None)

            # Pseudoimplicit sigma adjustment
            maxmin_ratio = (NS.sub_sigma - RK.sigma_min) / NS.sub_sigma
            sub_sigma_2 = NS.sub_sigma - maxmin_ratio * (NS.sub_sigma * pseudoimplicit_row_weights[row] * pseudoimplicit_step_weights[full_iter] * self.lgw[step_sched])

            eps_tmp_ = eps_.clone()
            eps_row = eps_[row]

            # Blend with certainty mask
            if "_projection" in self.guide_mode:
                # Projection variant: preserve magnitude, steer direction
                eps_row_lerp = eps_row + lgw_mask * (eps_substep_guide - eps_row)
                eps_collinear = get_collinear(eps_row, eps_row_lerp)
                eps_ortho = get_orthogonal(eps_row_lerp, eps_row)
                eps_sum = eps_collinear + eps_ortho
                eps_row = eps_row + lgw_mask * (eps_sum - eps_row)
            else:
                # Standard lerp blending
                eps_row = eps_row + lgw_mask * (eps_substep_guide - eps_row)
            eps_[row] = eps_row

            # Compute pseudoimplicit x
            x_row_pseudoimplicit = x_[row] + RK.h_fn(sub_sigma_2, NS.sub_sigma) * eps_[row]
            sub_sigma_pseudoimplicit = sub_sigma_2

            eps_ = eps_tmp_

            if self.EO("self_refine_debug"):
                coverage = (lgw_mask > 0).float().mean().item()
                iter_info = f", iter {full_iter}" if per_iteration_mode else ""
                RESplain(f"self_refine_pseudoimplicit step {step}{iter_info}, row {row}: APPLIED - certain_coverage={coverage:.2%}")

            # Apply bongmath if enabled
            if RK.IMPLICIT and BONGMATH and step < sigmas.shape[0]-1 and not self.EO("disable_pseudobongmath"):
                x_[row] = NS.sigma_from_to(x_0, x_row_pseudoimplicit, sigma, sub_sigma_pseudoimplicit, NS.s_[row])
                x_0, x_, eps_ = RK.bong_iter(x_0, x_, eps_, eps_prev_, data_, sigma, NS.s_, row, RK.row_offset, NS.h, step, step_sched)

            return x_0, x_, eps_, x_row_pseudoimplicit, sub_sigma_pseudoimplicit

        if self.s_lying_ is not None:
            if row >= len(self.s_lying_):
                return x_0, x_, eps_, None, None

        if self.guide_mode.startswith("fully_"):
            data_cossim_test = denoised_prev
        else:
            data_cossim_test = data_[row]

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_cossim_test, step_sched)

        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return x_0, x_, eps_, None, None


        if "fully_pseudoimplicit" in self.guide_mode:
            if self.x_lying_ is None:
                return x_0, x_, eps_, None, None
            else:
                x_row_pseudoimplicit     = self.x_lying_[row]
                sub_sigma_pseudoimplicit = self.s_lying_[row]



        if RK.IMPLICIT:
            x_ = RK.update_substep(x_0,
                                    x_,
                                    eps_,
                                    eps_prev_,
                                    row,
                                    RK.row_offset,
                                    NS.h_new,
                                    NS.h_new_orig,
                                    )

            x_[row] = NS.rebound_overshoot_substep(x_0, x_[row])

            if row > 0:
                x_[row] = NS.swap_noise_substep(x_0, x_[row])
                if BONGMATH and step < sigmas.shape[0]-1 and not self.EO("disable_pseudoimplicit_bongmath"):
                    x_0, x_, eps_ = RK.bong_iter(x_0,
                                                x_,
                                                eps_,
                                                eps_prev_,
                                                data_,
                                                sigma,
                                                NS.s_,
                                                row,
                                                RK.row_offset,
                                                NS.h,
                                                step,
                                                step_sched,
                                                )
        else:
            eps_[row] = RK.get_epsilon(x_0, x_[row], denoised_prev, sigma, NS.s_[row])

        if self.EO("pseudoimplicit_denoised_prev"):
            eps_[row] = RK.get_epsilon(x_0, x_[row], denoised_prev, sigma, NS.s_[row])

        eps_substep_guide     = torch.zeros_like(x_0)
        eps_substep_guide_inv = torch.zeros_like(x_0)

        if self.HAS_LATENT_GUIDE:
            eps_substep_guide     = RK.get_guide_epsilon(x_0, x_[row], y0,     sigma, NS.s_[row], NS.sigma_down, None)
        if self.HAS_LATENT_GUIDE_INV:
            eps_substep_guide_inv = RK.get_guide_epsilon(x_0, x_[row], y0_inv, sigma, NS.s_[row], NS.sigma_down, None)

        if self.guide_mode in {"pseudoimplicit", "pseudoimplicit_cw", "pseudoimplicit_projection", "pseudoimplicit_projection_cw"}:
            maxmin_ratio = (NS.sub_sigma - RK.sigma_min) / NS.sub_sigma

            if   self.EO("guide_pseudoimplicit_power_substep_flip_maxmin_scaling"):
                maxmin_ratio *= (RK.rows-row) / RK.rows
            elif self.EO("guide_pseudoimplicit_power_substep_maxmin_scaling"):
                maxmin_ratio *= row / RK.rows

            sub_sigma_2 = NS.sub_sigma - maxmin_ratio * (NS.sub_sigma * pseudoimplicit_row_weights[row] * pseudoimplicit_step_weights[full_iter] * self.lgw[step_sched])

            eps_tmp_ = eps_.clone()

            eps_ = self.process_channelwise(x_0,
                                            eps_,
                                            data_,
                                            row,
                                            eps_substep_guide,
                                            eps_substep_guide_inv,
                                            y0,
                                            y0_inv,
                                            lgw_mask,
                                            lgw_mask_inv,
                                            use_projection = self.guide_mode in {"pseudoimplicit_projection", "pseudoimplicit_projection_cw"},
                                            channelwise    = self.guide_mode in {"pseudoimplicit_cw",         "pseudoimplicit_projection_cw"},
                                            )

            if self.EO("debug_pseudoimplicit"):
                RESplain(
                    f"Step {step}, Row {row}: eps_[row] post-blend mean/std="
                    f"{eps_[row].mean().item():.6f}/{eps_[row].std().item():.6f}"
                )

            x_row_tmp = x_[row] + RK.h_fn(sub_sigma_2, NS.sub_sigma) * eps_[row]

            eps_                     = eps_tmp_
            x_row_pseudoimplicit     = x_row_tmp
            sub_sigma_pseudoimplicit = sub_sigma_2


        if RK.IMPLICIT and BONGMATH and step < sigmas.shape[0]-1 and not self.EO("disable_pseudobongmath"):
            x_[row] = NS.sigma_from_to(x_0, x_row_pseudoimplicit, sigma, sub_sigma_pseudoimplicit, NS.s_[row])

            x_0, x_, eps_ = RK.bong_iter(x_0,
                                        x_,
                                        eps_,
                                        eps_prev_,
                                        data_,
                                        sigma,
                                        NS.s_,
                                        row,
                                        RK.row_offset,
                                        NS.h,
                                        step,
                                        step_sched,
                                        )

        return x_0, x_, eps_, x_row_pseudoimplicit, sub_sigma_pseudoimplicit



    @torch.no_grad
    def prepare_fully_pseudoimplicit_guides_substep(self,
                                                    x_0,
                                                    x_,
                                                    eps_,
                                                    eps_prev_,
                                                    data_,
                                                    denoised_prev,
                                                    row,
                                                    step,
                                                    step_sched,
                                                    sigmas,
                                                    eta_substep,
                                                    overshoot_substep,
                                                    s_noise_substep,
                                                    NS,
                                                    RK,
                                                    pseudoimplicit_row_weights,
                                                    pseudoimplicit_step_weights,
                                                    full_iter,
                                                    BONGMATH,
                                                    ):

        if "fully_pseudoimplicit" not in self.guide_mode or (self.lgw[step_sched] == 0 and self.lgw_inv[step_sched] == 0):
            if self.EO("debug_pseudoimplicit"):
                RESplain(f"prepare_fully: SKIPPED - mode={self.guide_mode}, lgw={self.lgw[step_sched]:.4f}, lgw_inv={self.lgw_inv[step_sched]:.4f}")
            return x_0, x_, eps_

        # PACK-FIRST EXPERIMENT: Block channelwise fully_pseudoimplicit modes
        if x_0.ndim == 3:  # packed NestedTensor
            BLOCKED_FULLY_MODES = {"fully_pseudoimplicit_cw", "fully_pseudoimplicit_projection_cw"}
            if self.guide_mode in BLOCKED_FULLY_MODES:
                raise NotImplementedError(f"Mode '{self.guide_mode}' requires channel structure, incompatible with packed latents")

        sigma = sigmas[step]

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(denoised_prev, step_sched)

        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return x_0, x_, eps_


        # PREPARE FULLY PSEUDOIMPLICIT GUIDES
        if self.guide_mode in {"fully_pseudoimplicit", "fully_pseudoimplicit_cw", "fully_pseudoimplicit_projection", "fully_pseudoimplicit_projection_cw"} and (self.lgw[step_sched] > 0 or self.lgw_inv[step_sched] > 0):
            x_lying_   = x_.clone()
            eps_lying_ = eps_.clone()
            s_lying_   = []

            for r in range(RK.rows):

                NS.set_sde_substep(r, RK.multistep_stages, eta_substep, overshoot_substep, s_noise_substep)

                maxmin_ratio      = (NS.sub_sigma - RK.sigma_min) / NS.sub_sigma
                fully_sub_sigma_2 =  NS.sub_sigma - maxmin_ratio * (NS.sub_sigma * pseudoimplicit_row_weights[r] * pseudoimplicit_step_weights[full_iter] * self.lgw[step_sched])

                s_lying_.append(fully_sub_sigma_2)

                if RK.IMPLICIT:
                    x_ = RK.update_substep(x_0,
                                            x_,
                                            eps_,
                                            eps_prev_,
                                            r,
                                            RK.row_offset,
                                            NS.h_new,
                                            NS.h_new_orig,
                                            )

                    x_[r] = NS.rebound_overshoot_substep(x_0, x_[r])

                    if r > 0:
                        x_[r] = NS.swap_noise_substep(x_0, x_[r])
                        if BONGMATH and step < sigmas.shape[0]-1 and not self.EO("disable_fully_pseudoimplicit_bongmath"):
                            x_0, x_, eps_ = RK.bong_iter(x_0,
                                                        x_,
                                                        eps_,
                                                        eps_prev_,
                                                        data_,
                                                        sigma,
                                                        NS.s_,
                                                        r,
                                                        RK.row_offset,
                                                        NS.h,
                                                        step,
                                                        step_sched,
                                                        )

                if self.EO("fully_pseudoimplicit_denoised_prev"):
                    eps_[r] = RK.get_epsilon(x_0, x_[r], denoised_prev, sigma, NS.s_[r])

                eps_substep_guide     = torch.zeros_like(x_0)
                eps_substep_guide_inv = torch.zeros_like(x_0)

                if self.HAS_LATENT_GUIDE:
                    eps_substep_guide     = RK.get_guide_epsilon(x_0, x_[r], y0,     sigma, NS.s_[r], NS.sigma_down, None)
                if self.HAS_LATENT_GUIDE_INV:
                    eps_substep_guide_inv = RK.get_guide_epsilon(x_0, x_[r], y0_inv, sigma, NS.s_[r], NS.sigma_down, None)

                eps_ = self.process_channelwise(x_0,
                                                eps_,
                                                data_,
                                                r,
                                                eps_substep_guide,
                                                eps_substep_guide_inv,
                                                y0,
                                                y0_inv,
                                                lgw_mask,
                                                lgw_mask_inv,
                                                use_projection = self.guide_mode in {"fully_pseudoimplicit_projection", "fully_pseudoimplicit_projection_cw"},
                                                channelwise    = self.guide_mode in {"fully_pseudoimplicit_cw",         "fully_pseudoimplicit_projection_cw"},
                                                )

                x_lying_[r]   = x_[r] + RK.h_fn(fully_sub_sigma_2, NS.sub_sigma) * eps_[r]
                data_lying    = x_[r] + RK.h_fn(0,                 NS.s_[r])     * eps_[r]

                eps_lying_[r] = RK.get_epsilon(x_0, x_[r], data_lying, sigma, NS.s_[r])

            if not self.EO("pseudoimplicit_disable_eps_lying"):
                eps_ = eps_lying_

            if not self.EO("pseudoimplicit_disable_newton_iter"):
                x_, eps_ = RK.newton_iter(x_0,
                                        x_,
                                        eps_,
                                        eps_prev_,
                                        data_,
                                        NS.s_,
                                        0,
                                        NS.h,
                                        sigmas,
                                        step,
                                        "lying",
                                        False,  # SYNC_GUIDE_ACTIVE
                                        )

            self.x_lying_ = x_lying_
            self.s_lying_ = s_lying_

        return x_0, x_, eps_ 



    @torch.no_grad
    def process_guides_data_substep(self,
                                x_row         : Tensor,
                                data_row      : Tensor,
                                step          : int,
                                sigma_row     : Tensor,
                                ):
        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return x_row

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_row, step)

        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):
            return x_row

        if self.guide_mode in {"data", "data_projection", "lure", "lure_projection"}:
            x_row = self.get_data_substep(x_row, data_row, y0, y0_inv, lgw_mask, lgw_mask_inv, step, sigma_row)

        return x_row




    @torch.no_grad
    def get_data_substep(self,
                        x_row         : Tensor,
                        data_row      : Tensor,
                        y0            : Tensor,
                        y0_inv        : Tensor,
                        lgw_mask      : Tensor,
                        lgw_mask_inv  : Tensor,
                        step          : int,
                        sigma_row     : Tensor,
                        frame_target  : float = 1.0,
                        ):

        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return x_row

        if self.guide_mode in {"data", "data_projection", "lure", "lure_projection"}:
            data_targets = self.EO("data_targets", [1.0])
            step_target = step if len(data_targets) > step else len(data_targets)-1
            
            cossim_target = frame_target * data_targets[step_target]
            
            if self.HAS_LATENT_GUIDE:
                if self.guide_mode.endswith("projection"):
                    d_collinear_d_lerp = get_collinear(data_row, y0)  
                    d_lerp_ortho_d     = get_orthogonal(y0, data_row)  
                    y0                 = d_collinear_d_lerp + d_lerp_ortho_d
                    
                if   cossim_target == 1.0:
                    d_slerped = y0
                elif cossim_target == 0.0:
                    d_slerped = data_row
                else:
                    y0_pearsim    = get_pearson_similarity(data_row, y0,     mask=self.mask)
                    slerp_weight  = get_slerp_weight_for_cossim(y0_pearsim.item(), cossim_target)
                    d_slerped     = slerp_tensor(slerp_weight, data_row, y0) # lgw_mask * slerp_weight same as using mask below
                    
                """if self.guide_mode == "data_projection":
                    d_collinear_d_lerp = get_collinear(data_row, d_slerped)  
                    d_lerp_ortho_d     = get_orthogonal(d_slerped, data_row)  
                    d_slerped          = d_collinear_d_lerp + d_lerp_ortho_d"""
                    
                if self.VE_MODEL:
                    x_row = x_row + lgw_mask * (d_slerped - data_row) 
                else:
                    x_row = x_row + lgw_mask * (self.sigma_max - sigma_row) * (d_slerped - data_row) 

                
            if self.HAS_LATENT_GUIDE_INV:
                if self.guide_mode.endswith("projection"):
                    d_collinear_d_lerp = get_collinear(data_row, y0_inv)  
                    d_lerp_ortho_d     = get_orthogonal(y0_inv, data_row)  
                    y0_inv             = d_collinear_d_lerp + d_lerp_ortho_d
                
                if   cossim_target == 1.0:
                    d_slerped_inv = y0_inv
                elif cossim_target == 0.0:
                    d_slerped_inv = data_row
                else:
                    y0_pearsim    = get_pearson_similarity(data_row, y0_inv, mask=self.mask_inv)
                    slerp_weight  = get_slerp_weight_for_cossim(y0_pearsim.item(), cossim_target)
                    d_slerped_inv = slerp_tensor(slerp_weight, data_row, y0_inv)
                    
                """if self.guide_mode == "data_projection":
                    d_collinear_d_lerp = get_collinear(data_row, d_slerped_inv)  
                    d_lerp_ortho_d     = get_orthogonal(d_slerped_inv, data_row)  
                    d_slerped_inv      = d_collinear_d_lerp + d_lerp_ortho_d"""
                    
                if self.VE_MODEL:
                    x_row = x_row + lgw_mask_inv * (d_slerped_inv - data_row) 
                else:
                    x_row = x_row + lgw_mask_inv * (self.sigma_max - sigma_row) * (d_slerped_inv - data_row) 

                    
        return x_row

    @torch.no_grad
    def swap_data(self,
        x     : Tensor,
        data  : Tensor,
        y     : Tensor,
        sigma : Tensor,
        mask  : Optional[Tensor] = None,
    ):
        mask = 1.0 if mask is None else mask
        if self.VE_MODEL:
            return x + mask * (y - data)
        else:
            return x + mask * (self.sigma_max - sigma) * (y - data)

    @torch.no_grad
    def process_guides_eps_substep(self,
                                x_0           : Tensor,
                                x_row         : Tensor,
                                data_row      : Tensor,
                                eps_row       : Tensor,
                                step          : int,
                                sigma         : Tensor,
                                sigma_down    : Tensor,
                                sigma_row     : Tensor,
                                RK=None,
                                ):
        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return eps_row

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_row, step)

        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):
            return eps_row

        eps_y0     = torch.zeros_like(x_0)
        eps_y0_inv = torch.zeros_like(x_0)

        if self.HAS_LATENT_GUIDE:
            eps_y0     = RK.get_guide_epsilon(x_0, x_row, y0, sigma, sigma_row, sigma_down, None)

        if self.HAS_LATENT_GUIDE_INV:
            eps_y0_inv = RK.get_guide_epsilon(x_0, x_row, y0_inv, sigma, sigma_row, sigma_down, None)

        if self.guide_mode in {"epsilon", "epsilon_projection"}:
            eps_row = self.get_eps_substep(eps_row, eps_y0, eps_y0_inv, lgw_mask, lgw_mask_inv, step, sigma_row)

        return eps_row



    @torch.no_grad
    def get_eps_substep(self,
                        eps_row       : Tensor,
                        eps_y0        : Tensor,
                        eps_y0_inv    : Tensor,
                        lgw_mask      : Tensor,
                        lgw_mask_inv  : Tensor,
                        step          : int,
                        sigma_row     : Tensor,
                        frame_target  : float = 1.0,
                        ):
        
        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return eps_row

        if self.guide_mode in {"epsilon", "epsilon_projection"}:
            eps_targets = self.EO("eps_targets", [1.0])
            step_target = step if len(eps_targets) > step else len(eps_targets)-1
            
            cossim_target = frame_target * eps_targets[step_target]
            
            if self.HAS_LATENT_GUIDE:
                if self.guide_mode == "epsilon_projection":
                    d_collinear_d_lerp = get_collinear(eps_row, eps_y0)  
                    d_lerp_ortho_d     = get_orthogonal(eps_y0, eps_row)  
                    eps_y0             = d_collinear_d_lerp + d_lerp_ortho_d
                    
                if   cossim_target == 1.0:
                    d_slerped = eps_y0
                elif cossim_target == 0.0:
                    d_slerped = eps_row
                else:
                    y0_pearsim    = get_pearson_similarity(eps_row, eps_y0,     mask=self.mask)
                    slerp_weight  = get_slerp_weight_for_cossim(y0_pearsim.item(), cossim_target)
                    d_slerped     = slerp_tensor(slerp_weight, eps_row, eps_y0) # lgw_mask * slerp_weight same as using mask below
                    
                """if self.guide_mode == "data_projection":
                    d_collinear_d_lerp = get_collinear(data_row, d_slerped)  
                    d_lerp_ortho_d     = get_orthogonal(d_slerped, data_row)  
                    d_slerped          = d_collinear_d_lerp + d_lerp_ortho_d"""
                    
                eps_row = eps_row + lgw_mask * (d_slerped - eps_row) 

                
            if self.HAS_LATENT_GUIDE_INV:
                if self.guide_mode == "epsilon_projection":
                    d_collinear_d_lerp = get_collinear(eps_row, eps_y0_inv)  
                    d_lerp_ortho_d     = get_orthogonal(eps_y0_inv, eps_row)  
                    eps_y0_inv             = d_collinear_d_lerp + d_lerp_ortho_d
                
                if   cossim_target == 1.0:
                    d_slerped_inv = eps_y0_inv
                elif cossim_target == 0.0:
                    d_slerped_inv = eps_row
                else:
                    y0_pearsim    = get_pearson_similarity(eps_row, eps_y0_inv, mask=self.mask_inv)
                    slerp_weight  = get_slerp_weight_for_cossim(y0_pearsim.item(), cossim_target)
                    d_slerped_inv = slerp_tensor(slerp_weight, eps_row, eps_y0_inv)
                    
                """if self.guide_mode == "data_projection":
                    d_collinear_d_lerp = get_collinear(data_row, d_slerped_inv)  
                    d_lerp_ortho_d     = get_orthogonal(d_slerped_inv, data_row)  
                    d_slerped_inv      = d_collinear_d_lerp + d_lerp_ortho_d"""
                    
                eps_row = eps_row + lgw_mask_inv * (d_slerped_inv - eps_row) 

        return eps_row





    @torch.no_grad
    def process_guides_substep(self,
                                x_0           : Tensor,
                                x_            : Tensor,
                                eps_          : Tensor,
                                data_         : Tensor,
                                denoised_prev : Tensor,
                                row           :  int,
                                step          :  int,
                                step_sched    :  int,
                                sigma         : Tensor,
                                sigma_next    : Tensor,
                                sigma_down    : Tensor,
                                s_            : Tensor,
                                epsilon_scale :  float,
                                RK,
                                full_iter     :  int = 0,
                                ):

        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return eps_, x_

        is_flat = x_0.ndim == 3  # packed NestedTensor: [1,1,N]

        if is_flat:
            BLOCKED_MODES = {"epsilon_cw", "epsilon_projection_cw"}
            if self.guide_mode in BLOCKED_MODES:
                raise NotImplementedError(f"Mode '{self.guide_mode}' requires spatial structure, incompatible with packed latents")

            if self.frame_weights_mgr is not None:
                raise NotImplementedError("Frame weights require temporal structure, incompatible with pack-first experiment")

        # Handle self_refine_epsilon mode - uses denoised_prev as guide target
        if self.SELF_REFINE_EPSILON_MODE:
            lgw = self.lgw[step_sched] if step_sched < len(self.lgw) else 0.0
            if lgw == 0:
                return eps_, x_

            data_row = data_[row]
            eps_row = eps_[row]
            x_row = x_[row]
            sigma_row = s_[row]

            # Per-iteration mode: track changes across implicit iterations
            per_iteration_mode = not self.EO("self_refine_by_step")

            # Skip conditions
            if per_iteration_mode:
                # In per-iteration mode, skip step 0 iter 0 only
                if step == 0 and full_iter == 0:
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_epsilon step {step}, iter {full_iter}, row {row}: SKIPPED - no valid reference")
                    return eps_, x_
            else:
                # Non-iterative mode: skip entire step 0
                if step == 0 or denoised_prev.abs().max() == 0:
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_epsilon step {step}, row {row}: SKIPPED - no valid denoised_prev")
                    return eps_, x_

            # Track state changes
            is_new_step = (step != self.self_refine_epsilon_last_step)
            is_new_iter = (full_iter != self._self_refine_last_iter) if per_iteration_mode else False
            is_new_row = (row != self.self_refine_epsilon_last_row)

            # Track calls per (step, row) - function is called twice per row (for eps_ and eps_prev_)
            if is_new_step or is_new_iter or is_new_row:
                # First call for this (step, iter, row)
                self.self_refine_epsilon_call_count = 1
                self.self_refine_epsilon_last_row = row
            else:
                # Second call for same (step, iter, row) - this is eps_prev_
                self.self_refine_epsilon_call_count += 1

                if self.EO("self_refine_dont_guide_eps_prev"):
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_epsilon step {step}, iter {full_iter}, row {row}: SKIPPED eps_prev_")
                    return eps_, x_

            # Reference management
            if per_iteration_mode:
                # Per-iteration mode: update reference based on iteration changes
                if is_new_step:
                    # New step: reset everything, use denoised_prev as initial reference
                    self.self_refine_epsilon_ref = denoised_prev.clone()
                    self.self_refine_epsilon_last_step = step
                    self._self_refine_last_iter = full_iter
                    self._self_refine_certain_mask_accum = None
                    self._self_refine_iter_prediction = None
                    self._self_refine_converged = False
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_epsilon step {step}: NEW STEP - initialized reference from denoised_prev")

                elif is_new_iter:
                    # New iteration within same step: update reference to previous iteration's prediction
                    if self._self_refine_iter_prediction is not None:
                        self.self_refine_epsilon_ref = self._self_refine_iter_prediction.clone()
                        if self.EO("self_refine_debug"):
                            RESplain(f"self_refine_epsilon step {step}, iter {full_iter}: updated reference from iter {full_iter-1}")
                    self._self_refine_last_iter = full_iter
                    # Reset mask accumulator for new iteration if not using accumulation
                    if self.EO("self_refine_dont_accumulate_certainty"):
                        self._self_refine_certain_mask_accum = None

                # Capture current prediction for next iteration's reference (at row 0, first call only)
                if row == 0 and self.self_refine_epsilon_call_count == 1:
                    self._self_refine_iter_prediction = data_row.clone()
            else:
                # Non-iterative mode: update reference only on new steps
                if is_new_step:
                    self.self_refine_epsilon_ref = denoised_prev.clone()
                    self.self_refine_epsilon_last_step = step
                    self._self_refine_converged = False
                    if self.EO("self_refine_debug"):
                        RESplain(f"self_refine_epsilon step {step}: initialized reference from denoised_prev")

            # Compare current prediction against reference
            y0 = self.self_refine_epsilon_ref

            # Compute certainty mask (guide certain regions, let uncertain evolve)
            lgw_mask = self.get_self_refine_epsilon_mask(data_row, y0, step_sched)

            # Accumulate certainty across iterations
            if per_iteration_mode and not self.EO("self_refine_dont_accumulate_certainty"):
                if self._self_refine_certain_mask_accum is not None:
                    # Union with previous certain regions
                    binary_mask = (lgw_mask > 0).float()
                    binary_accum = (self._self_refine_certain_mask_accum > 0).float()
                    combined = torch.maximum(binary_mask, binary_accum)
                    lgw = self.lgw[step_sched] if step_sched < len(self.lgw) else 0.0
                    lgw_mask = combined * lgw
                self._self_refine_certain_mask_accum = lgw_mask.clone()

            if lgw_mask.max() == 0:
                if self.EO("self_refine_debug"):
                    RESplain(f"self_refine_epsilon step {step}, iter {full_iter}, row {row}: SKIPPED - no certain regions")
                return eps_, x_

            # Check coverage against cutoff
            coverage = (lgw_mask > 0).float().mean().item()
            if coverage >= self.self_refine_cutoff:
                self._self_refine_converged = True
                if self.EO("self_refine_debug"):
                    iter_info = f", iter {full_iter}" if per_iteration_mode else ""
                    RESplain(f"self_refine_epsilon step {step}{iter_info}, row {row}: CONVERGED - coverage={coverage:.2%} >= cutoff={self.self_refine_cutoff:.2%}")

            # Compute guide epsilon (direction toward reference)
            eps_y0 = RK.get_guide_epsilon(x_0, x_row, y0, sigma, sigma_row, sigma_down, None)

            # Blend: anchor certain regions toward reference
            if "_projection" in self.guide_mode:
                # Projection variant: preserve magnitude, steer direction
                eps_row_lerp = eps_row + lgw_mask * (eps_y0 - eps_row)
                eps_collinear = get_collinear(eps_row, eps_row_lerp)
                eps_ortho = get_orthogonal(eps_row_lerp, eps_row)
                eps_sum = eps_collinear + eps_ortho
                eps_[row] = eps_row + lgw_mask * (eps_sum - eps_row)
            else:
                # Standard lerp blending
                eps_[row] = eps_row + lgw_mask * (eps_y0 - eps_row)

            if self.EO("self_refine_debug"):
                call_type = "eps_" if self.self_refine_epsilon_call_count == 1 else "eps_prev_"
                coverage = (lgw_mask > 0).float().mean().item()
                iter_info = f", iter {full_iter}" if per_iteration_mode else ""
                RESplain(f"self_refine_epsilon step {step}{iter_info}, row {row} ({call_type}): APPLIED - certain_coverage={coverage:.2%}, eps mean/std={eps_[row].mean():.4f}/{eps_[row].std():.4f}")

            return eps_, x_

        # Local references for the row tensors we'll modify
        eps_row = eps_[row]
        data_row = data_[row]
        x_row = x_[row]
        x_row1 = x_[row+1]

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_row, step_sched)
        
        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return eps_, x_

        if self.EO(["substep_eps_ch_mean_std", "substep_eps_ch_mean", "substep_eps_ch_std", "substep_eps_mean_std", "substep_eps_mean", "substep_eps_std"]):
            eps_row_orig = eps_row.clone()

        if self.EO("dynamic_guides_mean_std"):
            y_shift, y_inv_shift = normalize_latent([y0, y0_inv], [data_row, data_row])
            y0 = y_shift
            if self.EO("dynamic_guides_inv"):
                y0_inv = y_inv_shift

        if self.EO("dynamic_guides_mean"):
            y_shift, y_inv_shift = normalize_latent([y0, y0_inv], [data_row, data_row], std=False)
            y0 = y_shift
            if self.EO("dynamic_guides_inv"):
                y0_inv = y_inv_shift



        if "data_old" == self.guide_mode:
            y0_tmp = y0.clone()
            if self.HAS_LATENT_GUIDE:
                y0_tmp = (1-lgw_mask) * data_row + lgw_mask * y0
                y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
            x_row1 = y0_tmp + eps_row

        if self.guide_mode == "data_old_projection":

            d_lerp             = data_row   +   lgw_mask * (y0-data_row)   +   lgw_mask_inv * (y0_inv-data_row)

            d_collinear_d_lerp = get_collinear(data_row, d_lerp)
            d_lerp_ortho_d     = get_orthogonal(d_lerp, data_row)

            data_row           = d_collinear_d_lerp + d_lerp_ortho_d

            x_row1             = data_row + eps_row * sigma
            


            #elif (self.UNSAMPLE or self.guide_mode in {"epsilon", "epsilon_cw", "epsilon_projection", "epsilon_projection_cw"}) and (self.lgw[step] > 0 or self.lgw_inv[step] > 0):
        elif self.guide_mode in {"epsilon", "epsilon_cw", "epsilon_projection", "epsilon_projection_cw"} and (self.lgw[step_sched] > 0 or self.lgw_inv[step_sched] > 0):
            if sigma_down < sigma   or   s_[row] < RK.sigma_max:

                eps_substep_guide     = torch.zeros_like(x_0)
                eps_substep_guide_inv = torch.zeros_like(x_0)

                if self.HAS_LATENT_GUIDE:
                    eps_substep_guide     = RK.get_guide_epsilon(x_0, x_row, y0,     sigma, s_[row], sigma_down, epsilon_scale)

                if self.HAS_LATENT_GUIDE_INV:
                    eps_substep_guide_inv = RK.get_guide_epsilon(x_0, x_row, y0_inv, sigma, s_[row], sigma_down, epsilon_scale)

                tol_value = self.EO("tol", -1.0)
                if tol_value >= 0:
                    if is_flat:
                        raise NotImplementedError("Tolerance mode requires batch/channel structure, incompatible with packed latents")
                    for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                        current_diff       = torch.norm(data_[row][b][c] - y0    [b][c])
                        current_diff_inv   = torch.norm(data_[row][b][c] - y0_inv[b][c])

                        lgw_scaled         = torch.nan_to_num(1-(tol_value/current_diff),     0)
                        lgw_scaled_inv     = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)

                        lgw_tmp            = min(self.lgw[step_sched]    , lgw_scaled)
                        lgw_tmp_inv        = min(self.lgw_inv[step_sched], lgw_scaled_inv)

                        lgw_mask_clamp     = torch.clamp(lgw_mask,     max=lgw_tmp)
                        lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)

                        eps_[row][b][c]    = eps_[row][b][c] + lgw_mask_clamp[b][0] * (eps_substep_guide[b][c] - eps_[row][b][c]) + lgw_mask_clamp_inv[b][0] * (eps_substep_guide_inv[b][c] - eps_[row][b][c])

                elif self.guide_mode in {"epsilon"}:
                    if self.EO("slerp_epsilon_guide"):
                        if eps_substep_guide.sum() != 0:
                            eps_row = slerp_tensor(lgw_mask, eps_row, eps_substep_guide)
                        if eps_substep_guide_inv.sum() != 0:
                            eps_row = slerp_tensor(lgw_mask_inv, eps_row, eps_substep_guide_inv)
                    else:
                        eps_row = eps_row + lgw_mask * (eps_substep_guide - eps_row) + lgw_mask_inv * (eps_substep_guide_inv - eps_row)

                elif self.guide_mode in {"epsilon_projection"}:
                    if self.EO("slerp_epsilon_guide"):
                        if eps_substep_guide.sum() != 0:
                            eps_row_slerp = slerp_tensor(self.mask, eps_row, eps_substep_guide)
                        if eps_substep_guide_inv.sum() != 0:
                            eps_row_slerp = slerp_tensor((1-self.mask), eps_row_slerp, eps_substep_guide_inv)

                        eps_collinear_eps_slerp = get_collinear(eps_row, eps_row_slerp)
                        eps_slerp_ortho_eps     = get_orthogonal(eps_row_slerp, eps_row)

                        eps_sum                = eps_collinear_eps_slerp + eps_slerp_ortho_eps

                        eps_row = slerp_tensor(lgw_mask, eps_row, eps_sum)
                        eps_row = slerp_tensor(lgw_mask_inv, eps_row, eps_sum)
                    else:
                        eps_row_lerp           = eps_row   +   self.mask * (eps_substep_guide-eps_row)   +   (1-self.mask) * (eps_substep_guide_inv-eps_row)

                        eps_collinear_eps_lerp = get_collinear(eps_row, eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_row)

                        eps_sum                = eps_collinear_eps_lerp + eps_lerp_ortho_eps

                        eps_row                = eps_row + lgw_mask * (eps_sum - eps_row) + lgw_mask_inv * (eps_sum - eps_row)

                elif self.guide_mode in {"epsilon_cw", "epsilon_projection_cw"}:
                    eps_ = self.process_channelwise(x_0,
                                                    eps_,
                                                    data_,
                                                    row,
                                                    eps_substep_guide,
                                                    eps_substep_guide_inv,
                                                    y0,
                                                    y0_inv,
                                                    lgw_mask,
                                                    lgw_mask_inv,
                                                    use_projection = self.guide_mode == "epsilon_projection_cw",
                                                    channelwise    = True
                                                    )

        temporal_smoothing = self.EO("temporal_smoothing", 0.0)
        if temporal_smoothing > 0:
            if is_flat:
                raise NotImplementedError("Temporal smoothing requires temporal structure, incompatible with packed latents")
            eps_row = apply_temporal_smoothing(eps_row, temporal_smoothing)

        if self.EO("substep_eps_ch_mean_std"):
            eps_row = normalize_latent(eps_row, eps_row_orig)
        if self.EO("substep_eps_ch_mean"):
            eps_row = normalize_latent(eps_row, eps_row_orig, std=False)
        if self.EO("substep_eps_ch_std"):
            eps_row = normalize_latent(eps_row, eps_row_orig, mean=False)
        if self.EO("substep_eps_mean_std"):
            eps_row = normalize_latent(eps_row, eps_row_orig, channelwise=False)
        if self.EO("substep_eps_mean"):
            eps_row = normalize_latent(eps_row, eps_row_orig, std=False, channelwise=False)
        if self.EO("substep_eps_std"):
            eps_row = normalize_latent(eps_row, eps_row_orig, mean=False, channelwise=False)

        # Write results back to tensors
        eps_[row] = eps_row
        if self.guide_mode in {"data_old", "data_old_projection"}:
            x_[row+1] = x_row1
        if self.guide_mode == "data_old_projection":
            data_[row] = data_row

        return eps_, x_
    

    def process_channelwise(self,
                            x_0                   : Tensor,
                            eps_                  : Tensor,
                            data_                 : Tensor,
                            row                   : int,
                            eps_substep_guide     : Tensor,
                            eps_substep_guide_inv : Tensor,
                            y0                    : Tensor,
                            y0_inv                : Tensor,
                            lgw_mask              : Tensor,
                            lgw_mask_inv          : Tensor,
                            use_projection        : bool    = False,
                            channelwise           : bool    = False
                            ):
        
        avg, avg_inv = 0, 0
        lgw_mask_channels = lgw_mask.shape[1]
        lgw_mask_inv_channels = lgw_mask_inv.shape[1]
        
        for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
            if self.EO("lgw_cw_test") and (lgw_mask_channels > 1 or lgw_mask_inv_channels > 1):
                c_ = c % (lgw_mask_channels if lgw_mask_channels > 1 else lgw_mask_inv_channels)
            else:
                c_ = 0
            avg     += torch.norm(lgw_mask    [b][c_] * data_[row][b][c]   -   lgw_mask    [b][c_] * y0    [b][c])
            avg_inv += torch.norm(lgw_mask_inv[b][c_] * data_[row][b][c]   -   lgw_mask_inv[b][c_] * y0_inv[b][c])
            
        avg     /= x_0.shape[1]
        avg_inv /= x_0.shape[1]
        
        for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
            if channelwise:
                ratio     = torch.nan_to_num(torch.norm(lgw_mask    [b][c_] * data_[row][b][c] - lgw_mask    [b][c_] * y0    [b][c])   /   avg,     0)
                ratio_inv = torch.nan_to_num(torch.norm(lgw_mask_inv[b][c_] * data_[row][b][c] - lgw_mask_inv[b][c_] * y0_inv[b][c])   /   avg_inv, 0)
            else:
                ratio     = 1.
                ratio_inv = 1.
                    
            if self.EO("slerp_epsilon_guide"):
                if eps_substep_guide[b][c].sum() != 0:
                    eps_[row][b][c] = slerp_tensor(ratio * lgw_mask[b][0], eps_[row][b][c], eps_substep_guide[b][c])
                if eps_substep_guide_inv[b][c].sum() != 0:
                    eps_[row][b][c] = slerp_tensor(ratio_inv * lgw_mask_inv[b][0], eps_[row][b][c], eps_substep_guide_inv[b][c])
            else:
                eps_[row][b][c]            = eps_[row][b][c]   +   ratio * lgw_mask[b][0] * (eps_substep_guide[b][c] - eps_[row][b][c])   +   ratio_inv * lgw_mask_inv[b][0] * (eps_substep_guide_inv[b][c] - eps_[row][b][c])
            
            if use_projection:
                if self.EO("slerp_epsilon_guide"):
                    if eps_substep_guide[b][c].sum() != 0:
                        eps_row_lerp = slerp_tensor(self.mask[b][0], eps_[row][b][c], eps_substep_guide[b][c])
                    if eps_substep_guide_inv[b][c].sum() != 0:
                        eps_row_lerp = slerp_tensor((1-self.mask[b][0]), eps_[row][b][c], eps_substep_guide_inv[b][c])
                else:
                    eps_row_lerp           = eps_[row][b][c]   +          self.mask[b][0] * (eps_substep_guide[b][c] - eps_[row][b][c])   +              (1-self.mask[b][0]) * (eps_substep_guide_inv[b][c] - eps_[row][b][c]) # should this ever be self.mask_inv?

                eps_collinear_eps_lerp = get_collinear (eps_[row][b][c], eps_row_lerp)
                eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp   , eps_[row][b][c])

                eps_sum                = eps_collinear_eps_lerp + eps_lerp_ortho_eps


                if self.EO("slerp_epsilon_guide"):
                    if eps_substep_guide[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio * lgw_mask[b][0], eps_[row][b][c], eps_sum)
                    if eps_substep_guide_inv[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio_inv * lgw_mask_inv[b][0], eps_[row][b][c], eps_sum)
                else:
                    eps_[row][b][c]        = eps_[row][b][c]   +   ratio * lgw_mask[b][0] * (eps_sum                 - eps_[row][b][c])   +   ratio_inv * lgw_mask_inv[b][0] * (eps_sum                     - eps_[row][b][c])
            else:
                if self.EO("slerp_epsilon_guide"):
                    if eps_substep_guide[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio * lgw_mask[b][0], eps_[row][b][c], eps_substep_guide[b][c])
                    if eps_substep_guide_inv[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio_inv * lgw_mask_inv[b][0], eps_[row][b][c], eps_substep_guide_inv[b][c])
                else:
                    eps_[row][b][c]        = eps_[row][b][c]   +   ratio * lgw_mask[b][0] * (eps_substep_guide[b][c] - eps_[row][b][c])   +   ratio_inv * lgw_mask_inv[b][0] * (eps_substep_guide_inv[b][c] - eps_[row][b][c])
                
        return eps_

    
    def normalize_inputs(self, x:Tensor, y0:Tensor, y0_inv:Tensor):
        """
        Modifies and returns 'x' by matching its mean and/or std to y0 and/or y0_inv.
        Controlled by extra_options.

        Returns:
            - x      (modified)
            - y0     (may be modified to match mean and std from y0_inv)
            - y0_inv (unchanged)
        """
        if self.guide_mode == "epsilon_guide_mean_std_from_bkg":
            y0 = normalize_latent(y0, y0_inv)

        input_norm = self.EO("input_norm", "")
        input_std  = self.EO("input_std", 1.0)
                
        if input_norm == "input_ch_mean_set_std_to":
            x = normalize_latent(x, set_std=input_std)

        if input_norm == "input_ch_set_std_to":
            x = normalize_latent(x, set_std=input_std, mean=False)
                
        if input_norm == "input_mean_set_std_to":
            x = normalize_latent(x, set_std=input_std,             channelwise=False)
            
        if input_norm == "input_std_set_std_to":
            x = normalize_latent(x, set_std=input_std, mean=False, channelwise=False)
        
        return x, y0, y0_inv



def apply_frame_weights(mask, frame_weights, normalize=False):
    original_mask_mean = mask.mean()
    if frame_weights is not None:
        for f in range(mask.shape[2]):
            frame_weight = frame_weights[f]
            mask[..., f:f+1, :, :] *= frame_weight
        if normalize:
            mask_mean = mask.mean()
            mask *= (original_mask_mean / mask_mean)



def prepare_mask(x, mask, LGW_MASK_RESCALE_MIN) -> tuple[torch.Tensor, bool]:
    if mask is None:
        mask = torch.ones_like(x[:,0:1,...])
        LGW_MASK_RESCALE_MIN = False
        return mask, LGW_MASK_RESCALE_MIN

    # For flat/packed tensors, expect mask to already be compatible or use ones
    if x.ndim == 3:
        if mask.numel() == x.numel() or mask.numel() == x.shape[-1]:
            return mask.reshape(1, 1, -1).expand_as(x).to(x.dtype), LGW_MASK_RESCALE_MIN
        else:
            # Mask incompatible with flat tensor, use ones
            RESplain(f"Warning: mask shape {mask.shape} incompatible with flat tensor shape {x.shape}, using ones mask instead.")
            return torch.ones_like(x), False

    target_height = x.shape[-2]
    target_width  = x.shape[-1]

    spatial_mask = None
    if x.ndim == 5 and mask.shape[0] > 1 and mask.ndim < 4:
        target_frames = x.shape[-3]
        spatial_mask = mask.unsqueeze(0).unsqueeze(0)  # [B, H, W] -> [1, 1, B, H, W]
        spatial_mask = F.interpolate(spatial_mask, 
                                    size=(target_frames, target_height, target_width), 
                                    mode='trilinear', 
                                    align_corners=False)  # [1, 1, F, H, W]
        repeat_shape = [1]  # batch
        for i in range(1, x.ndim - 3):
            repeat_shape.append(x.shape[i])
        repeat_shape.extend([1, 1, 1])  # frames, height, width
    elif mask.ndim == 4: #temporal mask batch
        mask = F.interpolate(mask, size=(target_height, target_width), mode='bilinear', align_corners=False)
        mask = mask.repeat(x.shape[-4],1,1,1)
        mask.unsqueeze_(0)

    else:
        spatial_mask = mask.unsqueeze(1)
        spatial_mask = F.interpolate(spatial_mask, size=(target_height, target_width), mode='bilinear', align_corners=False)

        while spatial_mask.ndim < x.ndim:
            spatial_mask = spatial_mask.unsqueeze(2)
        
        repeat_shape = [1]  # batch
        for i in range(1, x.ndim - 2):
            repeat_shape.append(x.shape[i])
        repeat_shape.extend([1, 1])  # height and width
        repeat_shape[1] = 1                                   # only need one channel for masks

    if spatial_mask is not None:
        mask = spatial_mask.repeat(*repeat_shape).to(x.dtype)
        
        del spatial_mask
    return mask, LGW_MASK_RESCALE_MIN
    
def apply_temporal_smoothing(tensor, temporal_smoothing):
    if temporal_smoothing <= 0 or tensor.ndim != 5:
        return tensor

    kernel_size = 5
    padding = kernel_size // 2
    temporal_kernel = torch.tensor(
        [0.1, 0.2, 0.4, 0.2, 0.1],
        device=tensor.device, dtype=tensor.dtype
    ) * temporal_smoothing
    temporal_kernel[kernel_size//2] += (1 - temporal_smoothing)
    temporal_kernel = temporal_kernel / temporal_kernel.sum()

    # resahpe for conv1d
    b, c, f, h, w = tensor.shape
    data_flat = tensor.permute(0, 1, 3, 4, 2).reshape(-1, f)

    # apply smoohting
    data_smooth = F.conv1d(
        data_flat.unsqueeze(1),
        temporal_kernel.view(1, 1, -1),
        padding=padding
    ).squeeze(1)

    return data_smooth.view(b, c, h, w, f).permute(0, 1, 4, 2, 3)

def get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, row_offset, rk_type, b=None, c=None):
    s_in = x_0.new_ones([x_0.shape[0]])
    
    if b is not None and c is not None:  
        index = (b, c)
    elif b is not None: 
        index = (b,)
    else: 
        index = ()

    if RK_Method_Beta.is_exponential(rk_type):
        eps_row     =  y0    [index] -  x_0[index]
        eps_row_inv =  y0_inv[index] -  x_0[index]
    else:
        eps_row     = (x_[row][index] - y0    [index]) / (s_[row] * s_in) # was row+row_offset before for x_!!   not right...     also? potential issues here with x_[row+1] being RK.rows+2 with gauss-legendre_2s 1 imp step 1 imp substep
        eps_row_inv = (x_[row][index] - y0_inv[index]) / (s_[row] * s_in)
    
    return eps_row, eps_row_inv

def get_guide_epsilon(x_0, x_, y0, sigma, rk_type, b=None, c=None):
    s_in = x_0.new_ones([x_0.shape[0]])
    
    if b is not None and c is not None:  
        index = (b, c)
    elif b is not None: 
        index = (b,)
    else: 
        index = ()

    if RK_Method_Beta.is_exponential(rk_type):
        eps     = y0    [index] - x_0[index]
    else:
        eps     = (x_[index] - y0    [index]) / (sigma * s_in)
    
    return eps



@torch.no_grad
def noise_cossim_guide_tiled(x_list, guide, cossim_mode="forward", tile_size=2, step=0):

    guide_tiled = rearrange(guide, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)

    x_tiled_list = [
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for x in x_list
    ]
    x_tiled_stack = torch.stack([x_tiled[0] for x_tiled in x_tiled_list])  # [n_x, n_tiles, c, h, w]

    guide_flat = guide_tiled[0].view(guide_tiled.shape[1], -1).unsqueeze(0)  # [1, n_tiles, c*h*w]
    x_flat = x_tiled_stack.view(x_tiled_stack.size(0), x_tiled_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]

    cossim_tmp_all = F.cosine_similarity(x_flat, guide_flat, dim=-1)  # [n_x, n_tiles]

    if cossim_mode == "forward":
        indices = cossim_tmp_all.argmax(dim=0) 
    elif cossim_mode == "reverse":
        indices = cossim_tmp_all.argmin(dim=0) 
    elif cossim_mode == "orthogonal":
        indices = torch.abs(cossim_tmp_all).argmin(dim=0) 
    elif cossim_mode == "forward_reverse":
        if step % 2 == 0:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_forward":
        if step % 2 == 1:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "orthogonal_reverse":
        if step % 2 == 0:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_orthogonal":
        if step % 2 == 1:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    else:
        target_value = float(cossim_mode)
        indices = torch.abs(cossim_tmp_all - target_value).argmin(dim=0)  

    x_tiled_out = x_tiled_stack[indices, torch.arange(indices.size(0))]  # [n_tiles, c, h, w]

    x_tiled_out = x_tiled_out.unsqueeze(0) 
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)

    return x_detiled


@torch.no_grad
def noise_cossim_eps_tiled(x_list, eps, noise_list, cossim_mode="forward", tile_size=2, step=0):

    eps_tiled = rearrange(eps, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
    x_tiled_list = [
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for x in x_list
    ]
    noise_tiled_list = [
        rearrange(noise, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)
        for noise in noise_list
    ]

    noise_tiled_stack = torch.stack([noise_tiled[0] for noise_tiled in noise_tiled_list])  # [n_x, n_tiles, c, h, w]
    eps_expanded = eps_tiled[0].view(eps_tiled.shape[1], -1).unsqueeze(0)  # [1, n_tiles, c*h*w]
    noise_flat = noise_tiled_stack.view(noise_tiled_stack.size(0), noise_tiled_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]
    cossim_tmp_all = F.cosine_similarity(noise_flat, eps_expanded, dim=-1)  # [n_x, n_tiles]

    if cossim_mode == "forward":
        indices = cossim_tmp_all.argmax(dim=0)  
    elif cossim_mode == "reverse":
        indices = cossim_tmp_all.argmin(dim=0) 
    elif cossim_mode == "orthogonal":
        indices = torch.abs(cossim_tmp_all).argmin(dim=0) 
    elif cossim_mode == "orthogonal_pos":
        positive_mask = cossim_tmp_all > 0
        positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
        indices = positive_tmp.argmin(dim=0)
    elif cossim_mode == "orthogonal_neg":
        negative_mask = cossim_tmp_all < 0
        negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
        indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_posneg":
        if step % 2 == 0:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_negpos":
        if step % 2 == 1:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "forward_reverse":
        if step % 2 == 0:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_forward":
        if step % 2 == 1:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "orthogonal_reverse":
        if step % 2 == 0:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_orthogonal":
        if step % 2 == 1:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    else:
        target_value = float(cossim_mode)
        indices = torch.abs(cossim_tmp_all - target_value).argmin(dim=0)
    #else:
    #    raise ValueError(f"Unknown cossim_mode: {cossim_mode}")

    x_tiled_stack = torch.stack([x_tiled[0] for x_tiled in x_tiled_list])  # [n_x, n_tiles, c, h, w]
    x_tiled_out = x_tiled_stack[indices, torch.arange(indices.size(0))]  # [n_tiles, c, h, w]

    x_tiled_out = x_tiled_out.unsqueeze(0)  # restore batch dim
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)
    return x_detiled



@torch.no_grad
def noise_cossim_guide_eps_tiled(x_0, x_list, y0, noise_list, cossim_mode="forward", tile_size=2, step=0, sigma=None, rk_type=None):

    x_tiled_stack = torch.stack([
        rearrange(x, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)[0]
        for x in x_list
    ])  # [n_x, n_tiles, c, h, w]
    eps_guide_stack = torch.stack([
        rearrange(x - y0, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)[0]
        for x in x_list
    ])  # [n_x, n_tiles, c, h, w]
    del x_list

    noise_tiled_stack = torch.stack([
        rearrange(noise, "b c (h t1) (w t2) -> b (t1 t2) c h w", t1=tile_size, t2=tile_size)[0]
        for noise in noise_list
    ])  # [n_x, n_tiles, c, h, w]
    del noise_list

    noise_flat = noise_tiled_stack.view(noise_tiled_stack.size(0), noise_tiled_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]
    eps_guide_flat = eps_guide_stack.view(eps_guide_stack.size(0), eps_guide_stack.size(1), -1)  # [n_x, n_tiles, c*h*w]

    cossim_tmp_all = F.cosine_similarity(noise_flat, eps_guide_flat, dim=-1)  # [n_x, n_tiles]
    del noise_tiled_stack, noise_flat, eps_guide_stack, eps_guide_flat

    if cossim_mode == "forward":
        indices = cossim_tmp_all.argmax(dim=0) 
    elif cossim_mode == "reverse":
        indices = cossim_tmp_all.argmin(dim=0) 
    elif cossim_mode == "orthogonal":
        indices = torch.abs(cossim_tmp_all).argmin(dim=0) 
    elif cossim_mode == "orthogonal_pos":
        positive_mask = cossim_tmp_all > 0
        positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
        indices = positive_tmp.argmin(dim=0)
    elif cossim_mode == "orthogonal_neg":
        negative_mask = cossim_tmp_all < 0
        negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
        indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_posneg":
        if step % 2 == 0:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "orthogonal_negpos":
        if step % 2 == 1:
            positive_mask = cossim_tmp_all > 0
            positive_tmp = torch.where(positive_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('inf')))
            indices = positive_tmp.argmin(dim=0)
        else:
            negative_mask = cossim_tmp_all < 0
            negative_tmp = torch.where(negative_mask, cossim_tmp_all, torch.full_like(cossim_tmp_all, float('-inf')))
            indices = negative_tmp.argmax(dim=0)
    elif cossim_mode == "forward_reverse":
        if step % 2 == 0:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_forward":
        if step % 2 == 1:
            indices = cossim_tmp_all.argmax(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "orthogonal_reverse":
        if step % 2 == 0:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    elif cossim_mode == "reverse_orthogonal":
        if step % 2 == 1:
            indices = torch.abs(cossim_tmp_all).argmin(dim=0)
        else:
            indices = cossim_tmp_all.argmin(dim=0)
    else:
        target_value = float(cossim_mode)
        indices = torch.abs(cossim_tmp_all - target_value).argmin(dim=0)  

    x_tiled_out = x_tiled_stack[indices, torch.arange(indices.size(0))]  # [n_tiles, c, h, w]
    del x_tiled_stack

    x_tiled_out = x_tiled_out.unsqueeze(0)  
    x_detiled = rearrange(x_tiled_out, "b (t1 t2) c h w -> b c (h t1) (w t2)", t1=tile_size, t2=tile_size)

    return x_detiled







class NoiseStepHandlerOSDE:
    def __init__(self, x, eps=None, data=None, x_init=None, guide=None, guide_bkg=None):
        self.noise = None
        self.x = x
        self.eps = eps
        self.data = data
        self.x_init = x_init
        self.guide = guide
        self.guide_bkg = guide_bkg
        
        self.eps_list = None

        self.noise_cossim_map = {
            "eps_orthogonal":              [self.noise, self.eps],
            "eps_data_orthogonal":         [self.noise, self.eps, self.data],

            "data_orthogonal":             [self.noise, self.data],
            "xinit_orthogonal":            [self.noise, self.x_init],
            
            "x_orthogonal":                [self.noise, self.x],
            "x_data_orthogonal":           [self.noise, self.x, self.data],
            "x_eps_orthogonal":            [self.noise, self.x, self.eps],

            "x_eps_data_orthogonal":       [self.noise, self.x, self.eps, self.data],
            "x_eps_data_xinit_orthogonal": [self.noise, self.x, self.eps, self.data, self.x_init],
            
            "x_eps_guide_orthogonal":      [self.noise, self.x, self.eps, self.guide],
            "x_eps_guide_bkg_orthogonal":  [self.noise, self.x, self.eps, self.guide_bkg],
            
            "noise_orthogonal":            [self.noise, self.x_init],
            
            "guide_orthogonal":            [self.noise, self.guide],
            "guide_bkg_orthogonal":        [self.noise, self.guide_bkg],
        }

    def check_cossim_source(self, source):
        return source in self.noise_cossim_map

    def get_ortho_noise(self, noise, prev_noises=None, max_iter=100, max_score=1e-7, NOISE_COSSIM_SOURCE="eps_orthogonal"):
        
        if NOISE_COSSIM_SOURCE not in self.noise_cossim_map:
            raise ValueError(f"Invalid NOISE_COSSIM_SOURCE: {NOISE_COSSIM_SOURCE}")
        
        self.noise_cossim_map[NOISE_COSSIM_SOURCE][0] = noise

        params = self.noise_cossim_map[NOISE_COSSIM_SOURCE]
        
        noise = get_orthogonal_noise_from_channelwise(*params, max_iter=max_iter, max_score=max_score)
        
        return noise





# NOTE: NS AND SUBSTEP ADDED!
def handle_tiled_etc_noise_steps(
                                x_0,
                                x,
                                x_prenoise,
                                x_init,
                                eps,
                                denoised,
                                y0,
                                y0_inv,
                                step,
                                rk_type,
                                RK,
                                NS,
                                SUBSTEP,
                                sigma_up,
                                sigma,
                                sigma_next,
                                alpha_ratio,
                                s_noise,
                                noise_mode,
                                SDE_NOISE_EXTERNAL,
                                sde_noise_t,
                                NOISE_COSSIM_SOURCE,
                                NOISE_COSSIM_MODE,
                                noise_cossim_tile_size,
                                noise_cossim_iterations,
                                extra_options):
    
    EO = ExtraOptions(extra_options)
    
    x_tmp          = []
    cossim_tmp     = []
    noise_tmp_list = []
    
    if step > EO("noise_cossim_end_step", MAX_STEPS):
        NOISE_COSSIM_SOURCE       = EO("noise_cossim_takeover_source"    , "eps")
        NOISE_COSSIM_MODE         = EO("noise_cossim_takeover_mode"      , "forward"              )
        noise_cossim_tile_size    = EO("noise_cossim_takeover_tile"      , noise_cossim_tile_size )
        noise_cossim_iterations   = EO("noise_cossim_takeover_iterations", noise_cossim_iterations)
        
    for i in range(noise_cossim_iterations):
        #x_tmp.append(NS.swap_noise(x_0, x, sigma, sigma, sigma_next, ))
        x_tmp.append(NS.add_noise_post(x, sigma_up, sigma, sigma_next, alpha_ratio, s_noise, noise_mode, SDE_NOISE_EXTERNAL, sde_noise_t)    )#y0, lgw, sigma_down are currently unused
        noise_tmp = x_tmp[i] - x
        if EO("noise_noise_zscore_norm"):
            noise_tmp = normalize_zscore(noise_tmp, channelwise=False, inplace=True)
        if EO("noise_noise_zscore_norm_cw"):
            noise_tmp = normalize_zscore(noise_tmp, channelwise=True,  inplace=True)
        if EO("noise_eps_zscore_norm"):
            eps       = normalize_zscore(eps,       channelwise=False, inplace=True)
        if EO("noise_eps_zscore_norm_cw"):
            eps       = normalize_zscore(eps,       channelwise=True,  inplace=True)
            
        if   NOISE_COSSIM_SOURCE in ("eps_tiled", "guide_epsilon_tiled", "guide_bkg_epsilon_tiled", "iig_tiled"):
            noise_tmp_list.append(noise_tmp)
        if   NOISE_COSSIM_SOURCE == "eps":
            cossim_tmp.append(get_cosine_similarity(eps, noise_tmp))
        if   NOISE_COSSIM_SOURCE == "eps_ch":
            cossim_total = torch.zeros_like(eps[0][0][0][0])
            for ch in range(eps.shape[1]):
                cossim_total += get_cosine_similarity(eps[0][ch], noise_tmp[0][ch])
            cossim_tmp.append(cossim_total)
        elif NOISE_COSSIM_SOURCE == "data":
            cossim_tmp.append(get_cosine_similarity(denoised, noise_tmp))
        elif NOISE_COSSIM_SOURCE == "latent":
            cossim_tmp.append(get_cosine_similarity(x_prenoise, noise_tmp))
        elif NOISE_COSSIM_SOURCE == "x_prenoise":
            cossim_tmp.append(get_cosine_similarity(x_prenoise, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "x":
            cossim_tmp.append(get_cosine_similarity(x, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "x_data":
            cossim_tmp.append(get_cosine_similarity(denoised, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "x_init_vs_noise":
            cossim_tmp.append(get_cosine_similarity(x_init, noise_tmp))
        elif NOISE_COSSIM_SOURCE == "mom":
            cossim_tmp.append(get_cosine_similarity(denoised, x + sigma_next*noise_tmp))
        elif NOISE_COSSIM_SOURCE == "guide":
            cossim_tmp.append(get_cosine_similarity(y0, x_tmp[i]))
        elif NOISE_COSSIM_SOURCE == "guide_bkg":
            cossim_tmp.append(get_cosine_similarity(y0_inv, x_tmp[i]))
            
    if step < EO("noise_cossim_start_step", 0):
        x = x_tmp[0]

    elif (NOISE_COSSIM_SOURCE == "eps_tiled"):
        x = noise_cossim_eps_tiled(x_tmp, eps, noise_tmp_list, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step)
    elif (NOISE_COSSIM_SOURCE == "guide_epsilon_tiled"):
        x = noise_cossim_guide_eps_tiled(x_0, x_tmp, y0, noise_tmp_list, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step, sigma=sigma, rk_type=rk_type)
    elif (NOISE_COSSIM_SOURCE == "guide_bkg_epsilon_tiled"):
        x = noise_cossim_guide_eps_tiled(x_0, x_tmp, y0_inv, noise_tmp_list, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step, sigma=sigma, rk_type=rk_type)
    elif (NOISE_COSSIM_SOURCE == "guide_tiled"):
        x = noise_cossim_guide_tiled(x_tmp, y0, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size, step=step)
    elif (NOISE_COSSIM_SOURCE == "guide_bkg_tiled"):
        x = noise_cossim_guide_tiled(x_tmp, y0_inv, cossim_mode=NOISE_COSSIM_MODE, tile_size=noise_cossim_tile_size)
    else:
        for i in range(len(x_tmp)):
            if   (NOISE_COSSIM_MODE == "forward") and (cossim_tmp[i] == max(cossim_tmp)):
                x = x_tmp[i]
                break
            elif (NOISE_COSSIM_MODE == "reverse") and (cossim_tmp[i] == min(cossim_tmp)):
                x = x_tmp[i]
                break
            elif (NOISE_COSSIM_MODE == "orthogonal") and (abs(cossim_tmp[i]) == min(abs(val) for val in cossim_tmp)):
                x = x_tmp[i]
                break
            elif (NOISE_COSSIM_MODE != "forward") and (NOISE_COSSIM_MODE != "reverse") and (NOISE_COSSIM_MODE != "orthogonal"):
                x = x_tmp[0]
                break
    return x





def get_masked_epsilon_projection(x_0, x_, eps_, y0, y0_inv, s_, row, row_offset, rk_type, LG, step):
    
    eps_row, eps_row_inv = get_guide_epsilon_substep(x_0, x_, y0, y0_inv, s_, row, row_offset, rk_type)
    eps_row_lerp = eps_[row]   +   LG.mask * (eps_row-eps_[row])   +   (1-LG.mask) * (eps_row_inv-eps_[row])
    eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
    eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])
    eps_sum = eps_collinear_eps_lerp + eps_lerp_ortho_eps
    lgw_mask, lgw_mask_inv = LG.get_masks_for_step(step)
    eps_substep_guide = eps_[row] + lgw_mask * (eps_sum - eps_[row]) + lgw_mask_inv * (eps_sum - eps_[row])
    return eps_substep_guide



