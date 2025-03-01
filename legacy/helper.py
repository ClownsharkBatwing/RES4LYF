import re
import torch
from comfy.samplers import SCHEDULER_NAMES
import torch.nn.functional as F
from ..res4lyf import RESplain


def get_extra_options_kv(key, default, extra_options):

    match = re.search(rf"{key}\s*=\s*([a-zA-Z0-9_.+-]+)", extra_options)
    if match:
        value = match.group(1)
    else:
        value = default
    return value

def get_extra_options_list(key, default, extra_options):

    match = re.search(rf"{key}\s*=\s*([a-zA-Z0-9_.,+-]+)", extra_options)
    if match:
        value = match.group(1)
    else:
        value = default
    return value

def extra_options_flag(flag, extra_options):
    return bool(re.search(rf"{flag}", extra_options))

def safe_get_nested(d, keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

def is_video_model(model):
    is_video_model = False
    try :
        is_video_model = 'video'  in model.inner_model.inner_model.model_config.unet_config['image_model'] or \
                         'cosmos' in model.inner_model.inner_model.model_config.unet_config['image_model']
    except:
        pass
    return is_video_model

def is_RF_model(model):
    from comfy import model_sampling
    modelsampling = model.inner_model.inner_model.model_sampling
    return isinstance(modelsampling, model_sampling.CONST)



def lagrange_interpolation(x_values, y_values, x_new):

    if not isinstance(x_values, torch.Tensor):
        x_values = torch.tensor(x_values, dtype=torch.get_default_dtype())
    if x_values.ndim != 1:
        raise ValueError("x_values must be a 1D tensor or a list of scalars.")

    if not isinstance(x_new, torch.Tensor):
        x_new = torch.tensor(x_new, dtype=x_values.dtype, device=x_values.device)
    if x_new.ndim == 0:
        x_new = x_new.unsqueeze(0)

    if isinstance(y_values, list):
        y_values = torch.stack(y_values, dim=0)
    if y_values.ndim < 1:
        raise ValueError("y_values must have at least one dimension (the sample dimension).")

    n = x_values.shape[0]
    if y_values.shape[0] != n:
        raise ValueError(f"Mismatch: x_values has length {n} but y_values has {y_values.shape[0]} samples.")

    m = x_new.shape[0]
    result_shape = (m,) + y_values.shape[1:]
    result = torch.zeros(result_shape, dtype=y_values.dtype, device=y_values.device)

    for i in range(n):
        Li = torch.ones_like(x_new, dtype=y_values.dtype, device=y_values.device)
        xi = x_values[i]
        for j in range(n):
            if i == j:
                continue
            xj = x_values[j]
            Li = Li * ((x_new - xj) / (xi - xj))
        extra_dims = (1,) * (y_values.ndim - 1)
        Li = Li.view(m, *extra_dims)
        result = result + Li * y_values[i]

    return result


def get_cosine_similarity_manual(a, b):
    return (a * b).sum() / (torch.norm(a) * torch.norm(b))



def get_cosine_similarity(a, b):
    if a.dim() == 5 and b.dim() == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)


def get_pearson_similarity(a, b):
    a = a.mean(dim=(-2,-1))
    b = b.mean(dim=(-2,-1))
    if a.dim() == 5 and b.dim() == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)



def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor


def has_nested_attr(obj, attr_path):
    attrs = attr_path.split('.')
    for attr in attrs:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True

def get_res4lyf_scheduler_list():
    scheduler_names = SCHEDULER_NAMES.copy()
    if "beta57" not in scheduler_names:
        scheduler_names.append("beta57")
    return scheduler_names

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c


def get_collinear_alt(x, y):

    y_flat = y.view(y.size(0), -1).clone()
    x_flat = x.view(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat

    return x_proj_y.view_as(x)


def get_collinear(x, y):

    y_flat = y.view(y.size(0), -1).clone()
    x_flat = x.view(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat

    return x_proj_y.view_as(x)


def get_orthogonal(x, y):

    y_flat = y.view(y.size(0), -1).clone()
    x_flat = x.view(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat
    
    x_ortho_y = x_flat - x_proj_y 

    return x_ortho_y.view_as(x)


# pytorch slerp implementation from https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like
from torch.linalg import norm

# adapted to PyTorch from:
# https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
# most of the extra complexity is to support:
# - many-dimensional vectors
# - v0 or v1 with last dim all zeroes, or v0 ~colinear with v1
#   - falls back to lerp()
#   - conditional logic implemented with parallelism rather than Python loops
# - many-dimensional tensor for t
#   - you can ask for batches of slerp outputs by making t more-dimensional than the vectors
#   -   slerp(
#         v0:   torch.Size([2,3]),
#         v1:   torch.Size([2,3]),
#         t:  torch.Size([4,1,1]), 
#       )
#   - this makes it interface-compatible with lerp()
def slerp(v0: FloatTensor, v1: FloatTensor, t: float|FloatTensor, DOT_THRESHOLD=0.9995):
  '''
  Spherical linear interpolation
  Args:
    v0: Starting vector
    v1: Final vector
    t: Float value between 0.0 and 1.0
    DOT_THRESHOLD: Threshold for considering the two vectors as
                            colinear. Not recommended to alter this.
  Returns:
      Interpolation vector between v0 and v1
  '''
  assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

  # Normalize the vectors to get the directions and angles
  v0_norm: FloatTensor = norm(v0, dim=-1)
  v1_norm: FloatTensor = norm(v1, dim=-1)

  v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
  v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

  # Dot product with the normalized vectors
  dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
  dot_mag: FloatTensor = dot.abs()

  # if dp is NaN, it's because the v0 or v1 row was filled with 0s
  # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
  gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
  can_slerp: LongTensor = ~gotta_lerp

  t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
  t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
  out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

  # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
  if gotta_lerp.any():
    lerped: FloatTensor = lerp(v0, v1, t)

    out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

  # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
  if can_slerp.any():

    # Calculate initial angle between v0 and v1
    theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
    sin_theta_0: FloatTensor = theta_0.sin()
    # Angle at timestep t
    theta_t: FloatTensor = theta_0 * t
    sin_theta_t: FloatTensor = theta_t.sin()
    # Finish the slerp algorithm
    s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
    s1: FloatTensor = sin_theta_t / sin_theta_0
    slerped: FloatTensor = s0 * v0 + s1 * v1

    out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)
  
  return out




class OptionsManager:
    APPEND_OPTIONS = {"extra_options"}

    def __init__(self, options_inputs=None):
        self.options_list = options_inputs or []
        self._merged_dict = None

    def add_option(self, option):
        """Add a single options dictionary"""
        if option is not None:
            self.options_list.append(option)
            self._merged_dict = None # invalidate cached merged options

    @property
    def merged(self):
        """Get merged options with proper priority handling"""
        if self._merged_dict is None:
            self._merged_dict = {}

            special_string_options = {
                key: [] for key in self.APPEND_OPTIONS
            }

            for options_dict in self.options_list:
                if options_dict is not None:
                    for key, value in options_dict.items():
                        if key in self.APPEND_OPTIONS and value:
                            special_string_options[key].append(value)
                        elif isinstance(value, dict):
                            # Deep merge dictionaries
                            if key not in self._merged_dict:
                                self._merged_dict[key] = {}

                            if isinstance(self._merged_dict[key], dict):
                                self._deep_update(self._merged_dict[key], value)
                            else:
                                self._merged_dict[key] = value.copy()
                        else:
                            self._merged_dict[key] = value

            # append special case string options (e.g. extra_options)
            for key, value in special_string_options.items():
                if value:
                    self._merged_dict[key] = "\n".join(value)

        return self._merged_dict

    def get(self, key, default=None):
        return self.merged.get(key, default)

    def _deep_update(self, target_dict, source_dict):

        for key, value in source_dict.items():
            if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
                # recursive dict update
                self._deep_update(target_dict[key], value)
            else:
                target_dict[key] = value

    def __getitem__(self, key):
        """Allow dictionary-like access to options"""
        return self.merged[key]

    def __contains__(self, key):
        """Allow 'in' operator for options"""
        return key in self.merged

    def as_dict(self):
        """Return the merged options as a dictionary"""
        return self.merged.copy()

    def __bool__(self):
        """Return True if there are any options"""
        return len(self.options_list) > 0 and any(opt is not None for opt in self.options_list)

    def debug_print_options(self):
        for i, options_dict in enumerate(self.options_list):
            RESplain(f"Options {i}:", debug=True)
            if options_dict is not None:
                for key, value in options_dict.items():
                    RESplain(f"  {key}: {value}", debug=True)
            else:
                RESplain("  None", "\n", debug=True)
