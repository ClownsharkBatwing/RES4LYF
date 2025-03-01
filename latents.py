import torch
import torch.nn.functional as F




# TENSOR PROJECTION OPS

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

def get_collinear(x, y):
    return get_collinear_flat(x, y).view_as(x)

def get_orthogonal(x, y):
    x_flat = x.view(x.size(0), -1).clone()
    x_ortho_y = x_flat - get_collinear_flat(x, y)  
    return x_ortho_y.view_as(x)

def get_collinear_flat(x, y):

    y_flat = y.view(y.size(0), -1).clone()
    x_flat = x.view(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat

    return x_proj_y



# TENSOR NORMALIZATION OPS

def latent_normalize_channels(x):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std  = x.std (dim=(-2, -1), keepdim=True)
    return  (x - mean) / std

def latent_stdize_channels(x):
    std  = x.std (dim=(-2, -1), keepdim=True)
    return  x / std

def latent_meancenter_channels(x):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return  x - mean



# TENSOR INTERPOLATION OPS

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



# this is silly...
def normalize_latent(target, source=None, mean=True, std=True, set_mean=None, set_std=None, channelwise=True):
    target = target.clone()
    source = source.clone() if source is not None else None
    def normalize_single_latent(single_target, single_source=None):
        y = torch.zeros_like(single_target)
        for b in range(y.shape[0]):
            if channelwise:
                for c in range(y.shape[1]):
                    single_source_mean = single_source[b][c].mean() if set_mean is None else set_mean
                    single_source_std  = single_source[b][c].std()  if set_std  is None else set_std
                    
                    if mean and std:
                        y[b][c] = (single_target[b][c] - single_target[b][c].mean()) / single_target[b][c].std()
                        if single_source is not None:
                            y[b][c] = y[b][c] * single_source_std + single_source_mean
                    elif mean:
                        y[b][c] = single_target[b][c] - single_target[b][c].mean()
                        if single_source is not None:
                            y[b][c] = y[b][c] + single_source_mean
                    elif std:
                        y[b][c] = single_target[b][c] / single_target[b][c].std()
                        if single_source is not None:
                            y[b][c] = y[b][c] * single_source_std
            else:
                single_source_mean = single_source[b].mean() if set_mean is None else set_mean
                single_source_std  = single_source[b].std()  if set_std  is None else set_std
                
                if mean and std:
                    y[b] = (single_target[b] - single_target[b].mean()) / single_target[b].std()
                    if single_source is not None:
                        y[b] = y[b] * single_source_std + single_source_mean
                elif mean:
                    y[b] = single_target[b] - single_target[b].mean()
                    if single_source is not None:
                        y[b] = y[b] + single_source_mean
                elif std:
                    y[b] = single_target[b] / single_target[b].std()
                    if single_source is not None:
                        y[b] = y[b] * single_source_std
        return y

    if isinstance(target, (list, tuple)):
        if source is not None:
            assert isinstance(source, (list, tuple)) and len(source) == len(target), \
                "If target is a list/tuple, source must be a list/tuple of the same length."
            return [normalize_single_latent(t, s) for t, s in zip(target, source)]
        else:
            return [normalize_single_latent(t) for t in target]
    else:
        return normalize_single_latent(target, source)



def hard_light_blend(base_latent, blend_latent):
    if base_latent.sum() == 0 and base_latent.std() == 0:
        return base_latent
    
    blend_latent = (blend_latent - blend_latent.min()) / (blend_latent.max() - blend_latent.min())

    positive_mask = base_latent >= 0
    negative_mask = base_latent < 0
    
    positive_latent = base_latent * positive_mask.float()
    negative_latent = base_latent * negative_mask.float()

    positive_result = torch.where(blend_latent < 0.5,
                                  2 * positive_latent * blend_latent,
                                  1 - 2 * (1 - positive_latent) * (1 - blend_latent))

    negative_result = torch.where(blend_latent < 0.5,
                                  2 * negative_latent.abs() * blend_latent,
                                  1 - 2 * (1 - negative_latent.abs()) * (1 - blend_latent))
    
    negative_result = -negative_result

    combined_result = positive_result * positive_mask.float() + negative_result * negative_mask.float()

    #combined_result *= base_latent.max()
    
    ks  = combined_result
    ks2 = torch.zeros_like(base_latent)
    for n in range(base_latent.shape[1]):
        ks2[0][n] = (ks[0][n]) / ks[0][n].std()
        ks2[0][n] = (ks2[0][n] * base_latent[0][n].std())
    combined_result = ks2
    
    return combined_result



