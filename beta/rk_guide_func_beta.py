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
                            get_slerp_weight_for_cossim, normalize_latent, hard_light_blend, slerp_tensor, get_orthogonal_noise_from_channelwise

from .rk_method_beta import RK_Method_Beta
from .constants      import MAX_STEPS

#from ..latents import hard_light_blend, normalize_latent



class LatentGuide:
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
                ):
        
        self.dtype                    = dtype
        self.device                   = device
        self.model                    = model

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

        self.guide_mode                = ""
        self.max_steps                 = MAX_STEPS
        self.mask                      = None
        self.mask_inv                  = None
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
        
        self.extra_options            = extra_options
        self.EO                       = ExtraOptions(extra_options)


    def init_guides(self,
            x:Tensor,
            RK_IMPLICIT:bool,
            guides:Optional[Tensor]=None,
            noise_sampler:Optional["NoiseGeneratorSubclass"]=None,
            batch_num:int=0,
            sigma_init = None,
            guide_inversion_y0 = None,
            guide_inversion_y0_inv = None,
        ) -> Tensor:
        
        latent_guide_weight           = 0.0
        latent_guide_weight_inv       = 0.0
        latent_guide_weight_sync      = 0.0
        latent_guide_weight_sync_inv  = 0.0
        latent_guide_weight_mean      = 0.0
        latent_guide_weight_adain     = 0.0
        latent_guide_weight_attninj   = 0.0
        latent_guide_weight_style_pos = 0.0
        latent_guide_weight_style_neg = 0.0

        latent_guide_weights           = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_inv       = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_sync      = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_sync_inv  = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_mean      = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_adain     = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_attninj   = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_style_pos = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)
        latent_guide_weights_style_neg = torch.zeros_like(self.sigmas, dtype=self.dtype, device=self.device)

        latent_guide           = None
        latent_guide_inv       = None
        latent_guide_mean      = None
        latent_guide_adain     = None
        latent_guide_attninj   = None
        latent_guide_style_pos = None
        latent_guide_style_neg = None
        

        if guides is not None:
            self.guide_mode                = guides.get("guide_mode", "none")
            
            if self.guide_mode.startswith("inversion"):
                self.guide_mode = self.guide_mode.replace("inversion", "epsilon", 1)
            else:
                self.SAMPLE   = True
                self.UNSAMPLE = False
            
            latent_guide_weight            = guides.get("weight_masked",    0.)
            latent_guide_weight_inv        = guides.get("weight_unmasked",  0.)
            latent_guide_weight_sync       = guides.get("weight_masked_sync", 0.)
            latent_guide_weight_sync_inv   = guides.get("weight_unmasked_sync", 0.)
            latent_guide_weight_mean       = guides.get("weight_mean",      0.)
            latent_guide_weight_adain      = guides.get("weight_adain",     0.)
            latent_guide_weight_attninj    = guides.get("weight_attninj",   0.)
            latent_guide_weight_style_pos  = guides.get("weight_style_pos", 0.)
            latent_guide_weight_style_neg  = guides.get("weight_style_neg", 0.)
            #latent_guide_synweight_style_pos  = guides.get("synweight_style_pos", 0.)
            #latent_guide_synweight_style_neg  = guides.get("synweight_style_neg", 0.)

            latent_guide_weights           = guides.get("weights_masked")
            latent_guide_weights_inv       = guides.get("weights_unmasked")
            latent_guide_weights_sync      = guides.get("weights_masked_sync")
            latent_guide_weights_sync_inv  = guides.get("weights_unmasked_sync")
            latent_guide_weights_mean      = guides.get("weights_mean")
            latent_guide_weights_adain     = guides.get("weights_adain")
            latent_guide_weights_attninj   = guides.get("weights_attninj")
            latent_guide_weights_style_pos = guides.get("weights_style_pos")
            latent_guide_weights_style_neg = guides.get("weights_style_neg")
            #latent_guide_synweights_style_pos = guides.get("synweights_style_pos")
            #latent_guide_synweights_style_neg = guides.get("synweights_style_neg")

            latent_guide                   = guides.get("guide_masked")
            latent_guide_inv               = guides.get("guide_unmasked")
            latent_guide_mean              = guides.get("guide_mean")
            latent_guide_adain             = guides.get("guide_adain")
            latent_guide_attninj           = guides.get("guide_attninj")
            latent_guide_style_pos         = guides.get("guide_style_pos")
            latent_guide_style_neg         = guides.get("guide_style_neg")

            self.mask                      = guides.get("mask")
            self.mask_inv                  = guides.get("unmask")
            self.mask_mean                 = guides.get("mask_mean")
            self.mask_adain                = guides.get("mask_adain")
            self.mask_attninj              = guides.get("mask_attninj")
            self.mask_style_pos            = guides.get("mask_style_pos")
            self.mask_style_neg            = guides.get("mask_style_neg")

            scheduler_                     = guides.get("weight_scheduler_masked")
            scheduler_inv_                 = guides.get("weight_scheduler_unmasked")
            scheduler_sync_                     = guides.get("weight_scheduler_masked_sync")
            scheduler_sync_inv_                 = guides.get("weight_scheduler_unmasked_sync")
            scheduler_mean_                = guides.get("weight_scheduler_mean")
            scheduler_adain_               = guides.get("weight_scheduler_adain")
            scheduler_attninj_             = guides.get("weight_scheduler_attninj")
            scheduler_style_pos_           = guides.get("weight_scheduler_style_pos")
            scheduler_style_neg_           = guides.get("weight_scheduler_style_neg")

            start_steps_                   = guides.get("start_step_masked",   0)
            start_steps_inv_               = guides.get("start_step_unmasked", 0)
            start_steps_sync_                   = guides.get("start_step_masked_sync",   0)
            start_steps_sync_inv_               = guides.get("start_step_unmasked_sync", 0)
            start_steps_mean_              = guides.get("start_step_mean",     0)
            start_steps_adain_             = guides.get("start_step_adain",    0)
            start_steps_attninj_           = guides.get("start_step_attninj",  0)
            start_steps_style_pos_         = guides.get("start_step_style_pos", 0)
            start_steps_style_neg_         = guides.get("start_step_style_neg", 0)

            steps_                         = guides.get("end_step_masked",     1)
            steps_inv_                     = guides.get("end_step_unmasked",   1)
            steps_sync_                         = guides.get("end_step_masked_sync",     1)
            steps_sync_inv_                     = guides.get("end_step_unmasked_sync",   1)
            
            steps_mean_                    = guides.get("end_step_mean",       1)
            steps_adain_                   = guides.get("end_step_adain",      1)
            steps_attninj_                 = guides.get("end_step_attninj",    1)
            steps_style_pos_               = guides.get("end_step_style_pos",  1)
            steps_style_neg_               = guides.get("end_step_style_neg",  1)

            self.guide_cossim_cutoff_         = guides.get("cutoff_masked",       1.)
            self.guide_bkg_cossim_cutoff_     = guides.get("cutoff_unmasked",     1.)
            self.guide_mean_cossim_cutoff_    = guides.get("cutoff_mean",         1.)
            self.guide_adain_cossim_cutoff_   = guides.get("cutoff_adain",        1.)
            self.guide_attninj_cossim_cutoff_ = guides.get("cutoff_attninj",      1.)
            self.guide_style_pos_cossim_cutoff_ = guides.get("cutoff_style_pos",  1.)
            self.guide_style_neg_cossim_cutoff_ = guides.get("cutoff_style_neg",  1.)

            self.SYNC_SEPARATE = False
            if scheduler_sync_ is not None:
                self.SYNC_SEPARATE = True
            self.SYNC_SEPARATE = True
            if scheduler_sync_ is None and scheduler_ is not None:

                latent_guide_weight_sync      = latent_guide_weight
                latent_guide_weight_sync_inv  = latent_guide_weight_sync_inv
                latent_guide_weights_sync     = latent_guide_weights
                latent_guide_weights_sync_inv = latent_guide_weights_inv
                
                scheduler_sync_               = scheduler_
                scheduler_sync_inv_           = scheduler_sync_inv_
                
                start_steps_sync_             = start_steps_
                start_steps_sync_inv_         = start_steps_inv_
                
                steps_sync_                   = steps_
                steps_sync_inv_               = steps_inv_
                
                
                



            if self.mask     is not None and self.mask.shape    [0] > 1 and self.VIDEO is False:
                self.mask     = self.mask    [batch_num].unsqueeze(0)
            if self.mask_inv is not None and self.mask_inv.shape[0] > 1 and self.VIDEO is False:
                self.mask_inv = self.mask_inv[batch_num].unsqueeze(0)
                
            if self.guide_mode.startswith("fully_") and not RK_IMPLICIT:
                self.guide_mode = self.guide_mode[6:]   # fully_pseudoimplicit is only supported for implicit samplers, default back to pseudoimplicit

            guide_sigma_shift = self.EO("guide_sigma_shift", 0.0)                                                                         # effectively hardcoding shift to 0 !!!!!!
            
            if latent_guide_weights is None and scheduler_ is not None:# and self.guide_mode != "none":
                total_steps          = steps_ - start_steps_
                latent_guide_weights = get_sigmas(self.model, scheduler_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend              = torch.zeros(start_steps_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights = torch.cat((prepend, latent_guide_weights.to(self.device)), dim=0)
                
            if latent_guide_weights_inv is None and scheduler_inv_ is not None:# and self.guide_mode != "none":
                total_steps              = steps_inv_ - start_steps_inv_
                latent_guide_weights_inv = get_sigmas(self.model, scheduler_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                  = torch.zeros(start_steps_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_inv = torch.cat((prepend, latent_guide_weights_inv.to(self.device)), dim=0)
                
                
                
            if latent_guide_weights_sync is None and scheduler_sync_ is not None:# and self.guide_mode != "none":
                total_steps          = steps_sync_ - start_steps_sync_
                latent_guide_weights_sync = get_sigmas(self.model, scheduler_sync_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend              = torch.zeros(start_steps_sync_,                               dtype=self.dtype, device=self.device)
                latent_guide_weights_sync = torch.cat((prepend, latent_guide_weights_sync.to(self.device)), dim=0)
                
            if latent_guide_weights_sync_inv is None and scheduler_sync_inv_ is not None:# and self.guide_mode != "none":
                total_steps              = steps_sync_inv_ - start_steps_sync_inv_
                latent_guide_weights_sync_inv = get_sigmas(self.model, scheduler_sync_inv_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                  = torch.zeros(start_steps_sync_inv_,                               dtype=self.dtype, device=self.device) 
                latent_guide_weights_sync_inv = torch.cat((prepend, latent_guide_weights_sync_inv.to(self.device)), dim=0)
                
            latent_guide_weights_sync     = 1 - latent_guide_weights_sync     if latent_guide_weights_sync     is not None else latent_guide_weights
            latent_guide_weights_sync_inv = 1 - latent_guide_weights_sync_inv if latent_guide_weights_sync_inv is not None else latent_guide_weights_inv
            latent_guide_weight_sync      = 1 - latent_guide_weight_sync
            latent_guide_weight_sync_inv  = 1 - latent_guide_weight_sync_inv# these are more intuitive to use if these are reversed... so that sync weight = 1.0 means "maximum guide strength"
            
            if latent_guide_weights_mean is None and scheduler_mean_ is not None:
                total_steps               = steps_mean_ - start_steps_mean_
                latent_guide_weights_mean = get_sigmas(self.model, scheduler_mean_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                   = torch.zeros(start_steps_mean_,                                                        dtype=self.dtype, device=self.device) 
                latent_guide_weights_mean = torch.cat((prepend, latent_guide_weights_mean.to(self.device)), dim=0)
            
            if latent_guide_weights_adain is None and scheduler_adain_ is not None:
                total_steps                = steps_adain_ - start_steps_adain_
                latent_guide_weights_adain = get_sigmas(self.model, scheduler_adain_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                    = torch.zeros(start_steps_adain_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_adain = torch.cat((prepend, latent_guide_weights_adain.to(self.device)), dim=0)
            
            if latent_guide_weights_attninj is None and scheduler_attninj_ is not None:
                total_steps                = steps_attninj_ - start_steps_attninj_
                latent_guide_weights_attninj = get_sigmas(self.model, scheduler_attninj_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                    = torch.zeros(start_steps_attninj_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_attninj = torch.cat((prepend, latent_guide_weights_attninj.to(self.device)), dim=0)
            
            if latent_guide_weights_style_pos is None and scheduler_style_pos_ is not None:
                total_steps                = steps_style_pos_ - start_steps_style_pos_
                latent_guide_weights_style_pos = get_sigmas(self.model, scheduler_style_pos_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                    = torch.zeros(start_steps_style_pos_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_style_pos = torch.cat((prepend, latent_guide_weights_style_pos.to(self.device)), dim=0)
            
            if latent_guide_weights_style_neg is None and scheduler_style_neg_ is not None:
                total_steps                = steps_style_neg_ - start_steps_style_neg_
                latent_guide_weights_style_neg = get_sigmas(self.model, scheduler_style_neg_, total_steps, 1.0, shift=guide_sigma_shift).to(dtype=self.dtype, device=self.device) / self.sigma_max
                prepend                    = torch.zeros(start_steps_style_neg_,                                                         dtype=self.dtype, device=self.device) 
                latent_guide_weights_style_neg = torch.cat((prepend, latent_guide_weights_style_neg.to(self.device)), dim=0)
            
            if scheduler_ != "constant" or latent_guide_weights is not None:
                latent_guide_weights      = initialize_or_scale(latent_guide_weights,      latent_guide_weight,      self.max_steps)
            if scheduler_inv_ != "constant" or latent_guide_weights_inv is not None:
                latent_guide_weights_inv  = initialize_or_scale(latent_guide_weights_inv,  latent_guide_weight_inv,  self.max_steps)
                
            if scheduler_sync_ != "constant" or latent_guide_weights_sync is not None:
                latent_guide_weights_sync      = initialize_or_scale(latent_guide_weights_sync,      latent_guide_weight_sync,      self.max_steps)
            if scheduler_sync_inv_ != "constant" or latent_guide_weights_sync_inv is not None:
                latent_guide_weights_sync_inv  = initialize_or_scale(latent_guide_weights_sync_inv,  latent_guide_weight_sync_inv,  self.max_steps)
                
            if scheduler_mean_ != "constant" or latent_guide_weights_mean is not None:
                latent_guide_weights_mean = initialize_or_scale(latent_guide_weights_mean, latent_guide_weight_mean, self.max_steps)
            if scheduler_adain_ != "constant" or latent_guide_weights_adain is not None:
                latent_guide_weights_adain = initialize_or_scale(latent_guide_weights_adain, latent_guide_weight_adain, self.max_steps)
            if scheduler_attninj_ != "constant" or latent_guide_weights_attninj is not None:
                latent_guide_weights_attninj = initialize_or_scale(latent_guide_weights_attninj, latent_guide_weight_attninj, self.max_steps)
            if scheduler_style_pos_ != "constant" or latent_guide_weights_style_pos is not None:
                latent_guide_weights_style_pos = initialize_or_scale(latent_guide_weights_style_pos, latent_guide_weight_style_pos, self.max_steps)
            if scheduler_style_neg_ != "constant" or latent_guide_weights_style_neg is not None:
                latent_guide_weights_style_neg = initialize_or_scale(latent_guide_weights_style_neg, latent_guide_weight_style_neg, self.max_steps)

            latent_guide_weights      [steps_      :] = 0
            latent_guide_weights_inv  [steps_inv_  :] = 0
            latent_guide_weights_sync      [steps_sync_      :] = 1
            latent_guide_weights_sync_inv  [steps_sync_inv_  :] = 1
            latent_guide_weights_mean [steps_mean_ :] = 0
            latent_guide_weights_adain[steps_adain_:] = 0
            latent_guide_weights_attninj[steps_attninj_:] = 0
            latent_guide_weights_style_pos[steps_style_pos_:] = 0
            latent_guide_weights_style_neg[steps_style_neg_:] = 0

        
        self.lgw       = F.pad(latent_guide_weights,       (0, self.max_steps), value=0.0)
        self.lgw_inv   = F.pad(latent_guide_weights_inv,   (0, self.max_steps), value=0.0)
        self.lgw_sync       = F.pad(latent_guide_weights_sync,       (0, self.max_steps), value=1.0)
        self.lgw_sync_inv   = F.pad(latent_guide_weights_sync_inv,   (0, self.max_steps), value=1.0)
        self.lgw_mean  = F.pad(latent_guide_weights_mean,  (0, self.max_steps), value=0.0)
        self.lgw_adain = F.pad(latent_guide_weights_adain, (0, self.max_steps), value=0.0)
        self.lgw_attninj = F.pad(latent_guide_weights_attninj, (0, self.max_steps), value=0.0)
        self.lgw_style_pos = F.pad(latent_guide_weights_style_pos, (0, self.max_steps), value=0.0)
        self.lgw_style_neg = F.pad(latent_guide_weights_style_neg, (0, self.max_steps), value=0.0)

        
        mask, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask, self.LGW_MASK_RESCALE_MIN)
        self.mask = mask.to(dtype=self.dtype, device=self.device)

        if self.mask_inv is not None:
            mask_inv, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_inv, self.LGW_MASK_RESCALE_MIN)
            self.mask_inv = mask_inv.to(dtype=self.dtype, device=self.device)
        else:
            self.mask_inv = (1-self.mask)
            
        mask_style_pos, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_style_pos, self.LGW_MASK_RESCALE_MIN)
        self.mask_style_pos = mask_style_pos.to(dtype=self.dtype, device=self.device)
        
        mask_style_neg, self.LGW_MASK_RESCALE_MIN = prepare_mask(x, self.mask_style_neg, self.LGW_MASK_RESCALE_MIN)
        self.mask_style_neg = mask_style_neg.to(dtype=self.dtype, device=self.device)
    
        if latent_guide is not None:
            self.HAS_LATENT_GUIDE = True
            if type(latent_guide) is dict:
                if latent_guide    ['samples'].shape[0] > 1:
                    latent_guide['samples']     = latent_guide    ['samples'][batch_num].unsqueeze(0)
                latent_guide_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide) is torch.Tensor:
                latent_guide_samples = latent_guide.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide)}")

            if self.VIDEO and latent_guide_samples.shape[2] == 1:
                latent_guide_samples = latent_guide_samples.repeat(1, 1, x.shape[2], 1, 1)

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

        if latent_guide_inv is not None:
            self.HAS_LATENT_GUIDE_INV = True
            if type(latent_guide_inv) is dict:
                if latent_guide_inv['samples'].shape[0] > 1:
                    latent_guide_inv['samples'] = latent_guide_inv['samples'][batch_num].unsqueeze(0)
                latent_guide_inv_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_inv['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_inv) is torch.Tensor:
                latent_guide_inv_samples = latent_guide_inv.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_inv)}")

            if self.VIDEO and latent_guide_inv_samples.shape[2] == 1:
                latent_guide_inv_samples = latent_guide_inv_samples.repeat(1, 1, x.shape[2], 1, 1)

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
                if latent_guide_mean['samples'].shape[0] > 1:
                    latent_guide_mean['samples'] = latent_guide_mean['samples'][batch_num].unsqueeze(0)
                latent_guide_mean_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_mean['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_mean) is torch.Tensor:
                latent_guide_mean_samples = latent_guide_mean.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_mean)}")

            if self.VIDEO and latent_guide_mean_samples.shape[2] == 1:
                latent_guide_mean_samples = latent_guide_mean_samples.repeat(1, 1, x.shape[2], 1, 1)

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
                if latent_guide_adain['samples'].shape[0] > 1:
                    latent_guide_adain['samples'] = latent_guide_adain['samples'][batch_num].unsqueeze(0)
                latent_guide_adain_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_adain['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_adain) is torch.Tensor:
                latent_guide_adain_samples = latent_guide_adain.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_adain)}")

            if self.VIDEO and latent_guide_adain_samples.shape[2] == 1:
                latent_guide_adain_samples = latent_guide_adain_samples.repeat(1, 1, x.shape[2], 1, 1)

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
                if latent_guide_attninj['samples'].shape[0] > 1:
                    latent_guide_attninj['samples'] = latent_guide_attninj['samples'][batch_num].unsqueeze(0)
                latent_guide_attninj_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_attninj['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_attninj) is torch.Tensor:
                latent_guide_attninj_samples = latent_guide_attninj.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_attninj)}")

            if self.VIDEO and latent_guide_attninj_samples.shape[2] == 1:
                latent_guide_attninj_samples = latent_guide_attninj_samples.repeat(1, 1, x.shape[2], 1, 1)

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
                if latent_guide_style_pos['samples'].shape[0] > 1:
                    latent_guide_style_pos['samples'] = latent_guide_style_pos['samples'][batch_num].unsqueeze(0)
                latent_guide_style_pos_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_style_pos['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_style_pos) is torch.Tensor:
                latent_guide_style_pos_samples = latent_guide_style_pos.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_style_pos)}")

            if self.VIDEO and latent_guide_style_pos_samples.shape[2] == 1:
                latent_guide_style_pos_samples = latent_guide_style_pos_samples.repeat(1, 1, x.shape[2], 1, 1)

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
                if latent_guide_style_neg['samples'].shape[0] > 1:
                    latent_guide_style_neg['samples'] = latent_guide_style_neg['samples'][batch_num].unsqueeze(0)
                latent_guide_style_neg_samples = self.model.inner_model.inner_model.process_latent_in(latent_guide_style_neg['samples']).clone().to(dtype=self.dtype, device=self.device)
            elif type(latent_guide_style_neg) is torch.Tensor:
                latent_guide_style_neg_samples = latent_guide_style_neg.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(f"Invalid latent type: {type(latent_guide_style_neg)}")

            if self.VIDEO and latent_guide_style_neg_samples.shape[2] == 1:
                latent_guide_style_neg_samples = latent_guide_style_neg_samples.repeat(1, 1, x.shape[2], 1, 1)

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
                self.y0 = guide_inversion_y0
            else:
                self.y0     = noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min).to(dtype=self.dtype, device=self.device)
                self.y0     = normalize_zscore(self.y0,     channelwise=True, inplace=True)
                self.y0    *= self.sigma_max
                
            if guide_inversion_y0_inv is not None:
                self.y0_inv = guide_inversion_y0_inv
            else:
                self.y0_inv = noise_sampler(sigma=self.sigma_max, sigma_next=self.sigma_min).to(dtype=self.dtype, device=self.device)
                self.y0_inv = normalize_zscore(self.y0_inv, channelwise=True, inplace=True)
                self.y0_inv*= self.sigma_max

            
        if self.VIDEO and self.frame_weights_mgr is not None:
            num_frames = x.shape[2]
            self.frame_weights     = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights', num_frames)
            self.frame_weights_inv = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights_inv', num_frames)
            
        x, self.y0, self.y0_inv = self.normalize_inputs(x, self.y0, self.y0_inv)       # ???

        return x

    def prepare_weighted_masks(self, step:int, lgw_type="default") -> Tuple[Tensor, Tensor]:
        if lgw_type == "sync":
            lgw_     = self.lgw_sync    [step]
            lgw_inv_ = self.lgw_sync_inv[step]
        else:
            lgw_     = self.lgw    [step]
            lgw_inv_ = self.lgw_inv[step]
        
        mask     = torch.ones_like(self.y0) if self.mask     is None else self.mask
        mask_inv = torch.zeros_like(mask  ) if self.mask_inv is None else self.mask_inv

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

        if self.VIDEO and self.frame_weights_mgr:
            num_frames = lgw_mask.shape[2]
            if self.HAS_LATENT_GUIDE:
                frame_weights = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights', num_frames, step)
                apply_frame_weights(lgw_mask, frame_weights, normalize_frame_weights_per_step)
            if self.HAS_LATENT_GUIDE_INV:
                frame_weights_inv = self.frame_weights_mgr.get_frame_weights_by_name('frame_weights_inv', num_frames, step)
                apply_frame_weights(lgw_mask_inv, frame_weights_inv, normalize_frame_weights_per_step_inv)

        return lgw_mask.to(self.device), lgw_mask_inv.to(self.device)



    def get_cossim_adjusted_lgw_masks(self, data:Tensor, step:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        if self.HAS_LATENT_GUIDE:
            y0     = self.y0.clone()
        else:
            y0     = torch.zeros_like(data)
            
        if self.HAS_LATENT_GUIDE_INV:
            y0_inv = self.y0_inv.clone()
        else:
            y0_inv = torch.zeros_like(data)

        if y0.shape[0] > 1:                                    # this is for changing the guide on a per-step basis
            y0 = y0[min(step, y0.shape[0]-1)].unsqueeze(0)
        
        lgw_mask, lgw_mask_inv = self.get_masks_for_step(step)
        
        y0_cossim, y0_cossim_inv  = 1.0, 1.0
        if self.HAS_LATENT_GUIDE:
            y0_cossim     = get_pearson_similarity(data, y0,     mask=lgw_mask)
        if self.HAS_LATENT_GUIDE_INV:
            y0_cossim_inv = get_pearson_similarity(data, y0_inv, mask=lgw_mask_inv)
        
        #if y0_cossim < self.guide_cossim_cutoff_ or y0_cossim_inv < self.guide_bkg_cossim_cutoff_:
        if y0_cossim     >= self.guide_cossim_cutoff_:
            lgw_mask     *= 0
        if y0_cossim_inv >= self.guide_bkg_cossim_cutoff_:
            lgw_mask_inv *= 0
        
        return y0, y0_inv, lgw_mask, lgw_mask_inv












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
                                            sigmas                      : Tensor,
                                            NS                                  ,
                                            RK                                  ,
                                            pseudoimplicit_row_weights  : Tensor,
                                            pseudoimplicit_step_weights : Tensor,
                                            full_iter                   : int,
                                            BONGMATH                    : bool,
                                            ):
        
        if "pseudoimplicit" not in self.guide_mode or (self.lgw[step] == 0 and self.lgw_inv[step] == 0):
            return x_0, x_, eps_, None, None
        
        sigma = sigmas[step]

        if self.s_lying_ is not None:
            if row >= len(self.s_lying_):
                return x_0, x_, eps_, None, None
        
        if self.guide_mode.startswith("fully_"):
            data_cossim_test = denoised_prev
        else:
            data_cossim_test = data_[row]
            
        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_cossim_test, step)
        
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
            
            sub_sigma_2 = NS.sub_sigma - maxmin_ratio * (NS.sub_sigma * pseudoimplicit_row_weights[row] * pseudoimplicit_step_weights[full_iter] * self.lgw[step])

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
        
        if "fully_pseudoimplicit" not in self.guide_mode or (self.lgw[step] == 0 and self.lgw_inv[step] == 0):
            return x_0, x_, eps_ 
        
        sigma = sigmas[step]
        
        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(denoised_prev, step)
        
        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return x_0, x_, eps_
        

        # PREPARE FULLY PSEUDOIMPLICIT GUIDES
        if self.guide_mode in {"fully_pseudoimplicit", "fully_pseudoimplicit_cw", "fully_pseudoimplicit_projection", "fully_pseudoimplicit_projection_cw"} and (self.lgw[step] > 0 or self.lgw_inv[step] > 0):
            x_lying_   = x_.clone()
            eps_lying_ = eps_.clone()
            s_lying_   = []
            
            for r in range(RK.rows):
                
                NS.set_sde_substep(r, RK.multistep_stages, eta_substep, overshoot_substep, s_noise_substep)

                maxmin_ratio      = (NS.sub_sigma - RK.sigma_min) / NS.sub_sigma
                fully_sub_sigma_2 =  NS.sub_sigma - maxmin_ratio * (NS.sub_sigma * pseudoimplicit_row_weights[r] * pseudoimplicit_step_weights[full_iter] * self.lgw[step])
                
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
                                                row,
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
                                frame_targets : Optional[Tensor] = None,
                                ):
        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return x_row

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_row, step)
        
        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return x_row

        if self.VIDEO and self.frame_weights_mgr is not None and frame_targets is None:
            num_frames = data_row.shape[2]
            frame_targets = self.frame_weights_mgr.get_frame_weights_by_name('frame_targets', num_frames, step)
            if frame_targets is None:
                frame_targets = torch.tensor(self.EO("frame_targets", [1.0]))
            frame_targets = torch.clamp(frame_targets, 0.0, 1.0).to(self.device)

        if self.guide_mode in {"data", "data_projection", "lure", "lure_projection"}:
            if frame_targets is None:
                x_row = self.get_data_substep(x_row, data_row, y0, y0_inv, lgw_mask, lgw_mask_inv, step, sigma_row)
            else:
                t_dim = x_row.shape[-3]
                for t in range(t_dim): #temporal dimension
                    frame_target = float(frame_targets[t] if len(frame_targets) > t else frame_targets[-1])
                    x_row[...,t:t+1,:,:] = self.get_data_substep(
                                                                x_row       [...,t:t+1,:,:], 
                                                                data_row    [...,t:t+1,:,:],
                                                                y0          [...,t:t+1,:,:], 
                                                                y0_inv      [...,t:t+1,:,:], 
                                                                lgw_mask    [...,t:t+1,:,:], 
                                                                lgw_mask_inv[...,t:t+1,:,:], 
                                                                step, 
                                                                sigma_row, 
                                                                frame_target)
        
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
    def process_guides_eps_substep(self,
                                x_0           : Tensor,
                                x_row         : Tensor,
                                data_row      : Tensor,
                                eps_row       : Tensor,
                                step          : int,
                                sigma         : Tensor,
                                sigma_down    : Tensor,
                                sigma_row     : Tensor,
                                frame_targets : Optional[Tensor] = None,
                                RK=None,
                                ):
        
        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return eps_row

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_row, step)
        
        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return eps_row

        if self.VIDEO and data_row.ndim == 5 and frame_targets is None:
            num_frames = data_row.shape[2]
            frame_targets = self.frame_weights_mgr.get_frame_weights_by_name('frame_targets', num_frames, step)
            if frame_targets is None:
                frame_targets = self.EO("frame_targets", [1.0])
            frame_targets = torch.clamp(frame_targets, 0.0, 1.0)
            
        eps_y0     = torch.zeros_like(x_0)
        eps_y0_inv = torch.zeros_like(x_0)
        
        if self.HAS_LATENT_GUIDE:
            eps_y0     = RK.get_guide_epsilon(x_0, x_row, y0,     sigma, sigma_row, sigma_down, None)  
            
        if self.HAS_LATENT_GUIDE_INV:
            eps_y0_inv = RK.get_guide_epsilon(x_0, x_row, y0_inv, sigma, sigma_row, sigma_down, None)  

        if self.guide_mode in {"epsilon", "epsilon_projection"}:
            if frame_targets is None:
                eps_row = self.get_eps_substep(eps_row, eps_y0, eps_y0_inv, lgw_mask, lgw_mask_inv, step, sigma_row)
            else:
                t_dim = x_row.shape[-3]
                for t in range(t_dim): #temporal dimension
                    frame_target = float(frame_targets[t] if len(frame_targets) > t else frame_targets[-1])
                    eps_row[...,t:t+1,:,:] = self.get_eps_substep(
                                                                eps_row     [...,t:t+1,:,:],
                                                                eps_y0      [...,t:t+1,:,:], 
                                                                eps_y0_inv  [...,t:t+1,:,:], 
                                                                lgw_mask    [...,t:t+1,:,:], 
                                                                lgw_mask_inv[...,t:t+1,:,:], 
                                                                step, 
                                                                sigma_row, 
                                                                frame_target)
                    
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
                                row           :  int,
                                step          :  int,
                                sigma         : Tensor,
                                sigma_next    : Tensor,
                                sigma_down    : Tensor,
                                s_            : Tensor,
                                epsilon_scale :  float,
                                RK,
                                ):
        
        if not self.HAS_LATENT_GUIDE and not self.HAS_LATENT_GUIDE_INV:
            return eps_, x_

        y0, y0_inv, lgw_mask, lgw_mask_inv = self.get_cossim_adjusted_lgw_masks(data_[row], step)
        
        if not (lgw_mask.any() != 0 or lgw_mask_inv.any() != 0):  # cossim score too similar! deactivate guide for this step
            return eps_, x_ 

        if self.EO(["substep_eps_ch_mean_std", "substep_eps_ch_mean", "substep_eps_ch_std", "substep_eps_mean_std", "substep_eps_mean", "substep_eps_std"]):
            eps_orig = eps_.clone()
        
        if self.EO("dynamic_guides_mean_std"):
            y_shift, y_inv_shift = normalize_latent([y0, y0_inv], [data_, data_])
            y0 = y_shift
            if self.EO("dynamic_guides_inv"):
                y0_inv = y_inv_shift

        if self.EO("dynamic_guides_mean"):
            y_shift, y_inv_shift = normalize_latent([y0, y0_inv], [data_, data_], std=False)
            y0 = y_shift
            if self.EO("dynamic_guides_inv"):
                y0_inv = y_inv_shift



        if "data_old" == self.guide_mode:
            y0_tmp = y0.clone()
            if self.HAS_LATENT_GUIDE:
                y0_tmp = (1-lgw_mask) * data_[row] + lgw_mask * y0
                y0_tmp = (1-lgw_mask_inv) * y0_tmp + lgw_mask_inv * y0_inv
            x_[row+1] = y0_tmp + eps_[row]
            
        if self.guide_mode == "data_old_projection":

            d_lerp             = data_[row]   +   lgw_mask * (y0-data_[row])   +   lgw_mask_inv * (y0_inv-data_[row])
            
            d_collinear_d_lerp = get_collinear(data_[row], d_lerp)  
            d_lerp_ortho_d     = get_orthogonal(d_lerp, data_[row])  
            
            data_[row]         = d_collinear_d_lerp + d_lerp_ortho_d
            
            x_[row+1]          = data_[row] + eps_[row] * sigma
            


            #elif (self.UNSAMPLE or self.guide_mode in {"epsilon", "epsilon_cw", "epsilon_projection", "epsilon_projection_cw"}) and (self.lgw[step] > 0 or self.lgw_inv[step] > 0):
        elif self.guide_mode in {"epsilon", "epsilon_cw", "epsilon_projection", "epsilon_projection_cw"} and (self.lgw[step] > 0 or self.lgw_inv[step] > 0):
            if sigma_down < sigma   or   s_[row] < RK.sigma_max:
                                
                eps_substep_guide     = torch.zeros_like(x_0)
                eps_substep_guide_inv = torch.zeros_like(x_0)
                
                if self.HAS_LATENT_GUIDE:
                    eps_substep_guide     = RK.get_guide_epsilon(x_0, x_[row], y0,     sigma, s_[row], sigma_down, epsilon_scale)  
                    
                if self.HAS_LATENT_GUIDE_INV:
                    eps_substep_guide_inv = RK.get_guide_epsilon(x_0, x_[row], y0_inv, sigma, s_[row], sigma_down, epsilon_scale)  

                tol_value = self.EO("tol", -1.0)
                if tol_value >= 0:
                    for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
                        current_diff       = torch.norm(data_[row][b][c] - y0    [b][c]) 
                        current_diff_inv   = torch.norm(data_[row][b][c] - y0_inv[b][c]) 
                        
                        lgw_scaled         = torch.nan_to_num(1-(tol_value/current_diff),     0)
                        lgw_scaled_inv     = torch.nan_to_num(1-(tol_value/current_diff_inv), 0)
                        
                        lgw_tmp            = min(self.lgw[step]    , lgw_scaled)
                        lgw_tmp_inv        = min(self.lgw_inv[step], lgw_scaled_inv)

                        lgw_mask_clamp     = torch.clamp(lgw_mask,     max=lgw_tmp)
                        lgw_mask_clamp_inv = torch.clamp(lgw_mask_inv, max=lgw_tmp_inv)

                        eps_[row][b][c]    = eps_[row][b][c] + lgw_mask_clamp[b][c] * (eps_substep_guide[b][c] - eps_[row][b][c]) + lgw_mask_clamp_inv[b][c] * (eps_substep_guide_inv[b][c] - eps_[row][b][c])
                
                elif self.guide_mode in {"epsilon"}: 
                    #eps_[row] = slerp(lgw_mask.mean().item(), eps_[row], eps_substep_guide)
                    if self.EO("slerp_epsilon_guide"):
                        if eps_substep_guide.sum() != 0:
                            eps_[row] = slerp_tensor(lgw_mask, eps_[row], eps_substep_guide)
                        if eps_substep_guide_inv.sum() != 0:
                            eps_[row] = slerp_tensor(lgw_mask_inv, eps_[row], eps_substep_guide_inv)
                    else:
                        eps_[row] = eps_[row] + lgw_mask * (eps_substep_guide - eps_[row]) + lgw_mask_inv * (eps_substep_guide_inv - eps_[row])
                    
                    #eps_[row] = slerp_barycentric(eps_[row].norm(), eps_substep_guide.norm(), eps_substep_guide_inv.norm(), 1-lgw_mask-lgw_mask_inv, lgw_mask, lgw_mask_inv)
                    
                elif self.guide_mode in {"epsilon_projection"}:
                    if self.EO("slerp_epsilon_guide"):
                        if eps_substep_guide.sum() != 0:
                            eps_row_slerp = slerp_tensor(self.mask, eps_[row], eps_substep_guide)
                        if eps_substep_guide_inv.sum() != 0:
                            eps_row_slerp = slerp_tensor((1-self.mask), eps_row_slerp, eps_substep_guide_inv)

                        eps_collinear_eps_slerp = get_collinear(eps_[row], eps_row_slerp)
                        eps_slerp_ortho_eps     = get_orthogonal(eps_row_slerp, eps_[row])

                        eps_sum                = eps_collinear_eps_slerp + eps_slerp_ortho_eps

                        eps_[row] = slerp_tensor(lgw_mask, eps_[row] , eps_sum)
                        eps_[row] = slerp_tensor(lgw_mask_inv, eps_[row], eps_sum)
                    else:
                        eps_row_lerp           = eps_[row]   +   self.mask * (eps_substep_guide-eps_[row])   +   (1-self.mask) * (eps_substep_guide_inv-eps_[row])

                        eps_collinear_eps_lerp = get_collinear(eps_[row], eps_row_lerp)
                        eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp, eps_[row])

                        eps_sum                = eps_collinear_eps_lerp + eps_lerp_ortho_eps

                        eps_[row]              = eps_[row] + lgw_mask * (eps_sum - eps_[row]) + lgw_mask_inv * (eps_sum - eps_[row])
                    
                    
                    #eps_row_slerp          = eps_[row]   +   self.mask * (eps_substep_guide-eps_[row])   +   (1-self.mask) * (eps_substep_guide_inv-eps_[row])

                    
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
            eps_[row] = apply_temporal_smoothing(eps_[row], temporal_smoothing)
            
        if self.EO("substep_eps_ch_mean_std"):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row])
        if self.EO("substep_eps_ch_mean"):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], std=False)
        if self.EO("substep_eps_ch_std"):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], mean=False)
        if self.EO("substep_eps_mean_std"):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], channelwise=False)
        if self.EO("substep_eps_mean"):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], std=False, channelwise=False)
        if self.EO("substep_eps_std"):
            eps_[row] = normalize_latent(eps_[row], eps_orig[row], mean=False, channelwise=False)
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
        for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
            avg     += torch.norm(lgw_mask    [b][c] * data_[row][b][c]   -   lgw_mask    [b][c] * y0    [b][c])
            avg_inv += torch.norm(lgw_mask_inv[b][c] * data_[row][b][c]   -   lgw_mask_inv[b][c] * y0_inv[b][c])
            
        avg     /= x_0.shape[1]
        avg_inv /= x_0.shape[1]
        
        for b, c in itertools.product(range(x_0.shape[0]), range(x_0.shape[1])):
            if channelwise:
                ratio     = torch.nan_to_num(torch.norm(lgw_mask    [b][c] * data_[row][b][c] - lgw_mask    [b][c] * y0    [b][c])   /   avg,     0)
                ratio_inv = torch.nan_to_num(torch.norm(lgw_mask_inv[b][c] * data_[row][b][c] - lgw_mask_inv[b][c] * y0_inv[b][c])   /   avg_inv, 0)
            else:
                ratio     = 1.
                ratio_inv = 1.
                    
            if self.EO("slerp_epsilon_guide"):
                if eps_substep_guide[b][c].sum() != 0:
                    eps_[row][b][c] = slerp_tensor(ratio * lgw_mask[b][c], eps_[row][b][c], eps_substep_guide[b][c])
                if eps_substep_guide_inv[b][c].sum() != 0:
                    eps_[row][b][c] = slerp_tensor(ratio_inv * lgw_mask_inv[b][c], eps_[row][b][c], eps_substep_guide_inv[b][c])
            else:
                eps_[row][b][c]            = eps_[row][b][c]   +   ratio * lgw_mask[b][c] * (eps_substep_guide[b][c] - eps_[row][b][c])   +   ratio_inv * lgw_mask_inv[b][c] * (eps_substep_guide_inv[b][c] - eps_[row][b][c])
            
            if use_projection:
                if self.EO("slerp_epsilon_guide"):
                    if eps_substep_guide[b][c].sum() != 0:
                        eps_row_lerp = slerp_tensor(self.mask[b][c], eps_[row][b][c], eps_substep_guide[b][c])
                    if eps_substep_guide_inv[b][c].sum() != 0:
                        eps_row_lerp = slerp_tensor((1-self.mask[b][c]), eps_[row][b][c], eps_substep_guide_inv[b][c])
                else:
                    eps_row_lerp           = eps_[row][b][c]   +          self.mask[b][c] * (eps_substep_guide[b][c] - eps_[row][b][c])   +              (1-self.mask[b][c]) * (eps_substep_guide_inv[b][c] - eps_[row][b][c]) # should this ever be self.mask_inv?

                eps_collinear_eps_lerp = get_collinear (eps_[row][b][c], eps_row_lerp)
                eps_lerp_ortho_eps     = get_orthogonal(eps_row_lerp   , eps_[row][b][c])

                eps_sum                = eps_collinear_eps_lerp + eps_lerp_ortho_eps


                if self.EO("slerp_epsilon_guide"):
                    if eps_substep_guide[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio * lgw_mask[b][c], eps_[row][b][c], eps_sum)
                    if eps_substep_guide_inv[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio_inv * lgw_mask_inv[b][c], eps_[row][b][c], eps_sum)
                else:
                    eps_[row][b][c]        = eps_[row][b][c]   +   ratio * lgw_mask[b][c] * (eps_sum                 - eps_[row][b][c])   +   ratio_inv * lgw_mask_inv[b][c] * (eps_sum                     - eps_[row][b][c])
            else:
                if self.EO("slerp_epsilon_guide"):
                    if eps_substep_guide[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio * lgw_mask[b][c], eps_[row][b][c], eps_substep_guide[b][c])
                    if eps_substep_guide_inv[b][c].sum() != 0:
                        eps_[row][b][c] = slerp_tensor(ratio_inv * lgw_mask_inv[b][c], eps_[row][b][c], eps_substep_guide_inv[b][c])
                else:
                    eps_[row][b][c]        = eps_[row][b][c]   +   ratio * lgw_mask[b][c] * (eps_substep_guide[b][c] - eps_[row][b][c])   +   ratio_inv * lgw_mask_inv[b][c] * (eps_substep_guide_inv[b][c] - eps_[row][b][c])
                
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
        mask = torch.ones_like(x)
        LGW_MASK_RESCALE_MIN = False
        return mask, LGW_MASK_RESCALE_MIN
    
    target_height = x.shape[-2]
    target_width = x.shape[-1]

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
