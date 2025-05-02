import torch
import torch.nn.functional as F



# TENSOR PROJECTION OPS

def get_cosine_similarity_manual(a, b):
    return (a * b).sum() / (torch.norm(a) * torch.norm(b))

def get_cosine_similarity(a, b, mask=None, dim=0):
    if a.ndim == 5 and b.ndim == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
        
    if mask is not None:
        return F.cosine_similarity((mask * a).flatten(), (mask * b).flatten(), dim=dim)
    else:
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=dim)
    
def get_pearson_similarity(a, b, mask=None, dim=0, norm_dim=None):
    if a.ndim == 5 and b.ndim == 5 and b.shape[2] == 1:
        b = b.expand(-1, -1, a.shape[2], -1, -1)
    
    if norm_dim is None:
        if   a.ndim == 4:
            norm_dim=(-2,-1)
        elif a.ndim == 5:
            norm_dim=(-4,-2,-1)
    
    a = a - a.mean(dim=norm_dim, keepdim=True)
    b = b - b.mean(dim=norm_dim, keepdim=True)
    
    if mask is not None:
        return F.cosine_similarity((mask * a).flatten(), (mask * b).flatten(), dim=dim)
    else:
        return F.cosine_similarity(a.flatten(), b.flatten(), dim=dim)
    
    
    
def get_collinear(x, y):
    return get_collinear_flat(x, y).reshape_as(x)

def get_orthogonal(x, y):
    x_flat = x.reshape(x.size(0), -1).clone()
    x_ortho_y = x_flat - get_collinear_flat(x, y)  
    return x_ortho_y.view_as(x)

def get_collinear_flat(x, y):

    y_flat = y.reshape(y.size(0), -1).clone()
    x_flat = x.reshape(x.size(0), -1).clone()

    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    x_proj_y = torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat

    return x_proj_y



def get_orthogonal_noise_from_channelwise(*refs, max_iter=500, max_score=1e-15):
    noise, *refs = refs
    noise_tmp = noise.clone()
    #b,c,h,w = noise.shape
    if (noise.ndim == 4):
        b,ch,h,w = noise.shape
    elif (noise.ndim == 5):
        b,ch,t,h,w = noise.shape
    
    for i in range(max_iter):
        noise_tmp = gram_schmidt_channels_optimized(noise_tmp, *refs)
        
        cossim_scores = []
        for ref in refs:
            #for c in range(noise.shape[-3]):
            for c in range(ch):
                cossim_scores.append(get_cosine_similarity(noise_tmp[0][c], ref[0][c]).abs())
            cossim_scores.append(get_cosine_similarity(noise_tmp[0], ref[0]).abs())
            
        if max(cossim_scores) < max_score:
            break
    
    return noise_tmp



def gram_schmidt_channels_optimized(A, *refs):
    if (A.ndim == 4):
        b,c,h,w = A.shape
    elif (A.ndim == 5):
        b,c,t,h,w = A.shape

    A_flat = A.view(b, c, -1)  
    
    for ref in refs:
        ref_flat = ref.view(b, c, -1).clone()  

        ref_flat /= ref_flat.norm(dim=-1, keepdim=True) 

        proj_coeff = torch.sum(A_flat * ref_flat, dim=-1, keepdim=True)  
        projection = proj_coeff * ref_flat 

        A_flat -= projection

    return A_flat.view_as(A)



# Efficient implementation equivalent to the following:
def attention_weights(
    query, 
    key, 
    attn_mask=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight


def attention_weights_orig(q, k):
    # implementation of in-place softmax to reduce memory req
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores.div_(math.sqrt(q.size(-1)))
    torch.exp(scores, out=scores)
    summed = torch.sum(scores, dim=-1, keepdim=True)
    scores /= summed
    return scores.nan_to_num_(0.0, 65504., -65504.)


# calculate slerp ratio needed to hit a target cosine similarity score
def get_slerp_weight_for_cossim(cos_sim, target_cos):
    # assumes unit vector matrices used for cossim
    import math
    c = cos_sim
    T = target_cos
    K = 1 - c

    A = K**2 - 2 * T**2 * K
    B = 2 * (1 - c) * (c + T**2)
    C = c**2 - T**2

    if abs(A) < 1e-8: # nearly collinear
        return 0.5  # just mix 50:50

    disc = B**2 - 4*A*C
    if disc < 0:
        return None  # no valid solution... blow up somewhere to get user's attention

    sqrt_disc = math.sqrt(disc)
    w1 = (-B + sqrt_disc) / (2 * A)
    w2 = (-B - sqrt_disc) / (2 * A)

    candidates = [w for w in [w1, w2] if 0 <= w <= 1]
    if candidates:
        return candidates[0]
    else:
        return max(0.0, min(1.0, w1))



def get_slerp_ratio(cos_sim_A, cos_sim_B, target_cos):
    import math
    alpha = math.acos(cos_sim_A)
    beta  = math.acos(cos_sim_B)
    delta = math.acos(target_cos)
    
    if abs(beta - alpha) < 1e-6:
        return 0.5
    
    t = (delta - alpha) / (beta - alpha)
    t = max(0.0, min(1.0, t))
    return t

def find_slerp_ratio_grid(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, E: torch.Tensor,
                            target_ratio: float = 1.0, num_samples: int = 100) -> float:
    """
    Finds the interpolation parameter t (in [0,1]) for which:
       f(t) = cos(slerp(t, A, B), D) - target_ratio * cos(slerp(t, A, B), E)
    is minimized in absolute value.
    
    Instead of requiring a sign change for bisection, we sample t values uniformly and pick the one that minimizes |f(t)|.
    """
    ts = torch.linspace(0.0, 1.0, steps=num_samples, device=A.device, dtype=A.dtype)
    best_t   = 0.0
    best_val = float('inf')
    for t_val in ts:
        t_tensor = torch.tensor(t_val, dtype=A.dtype, device=A.device)
        C        = slerp_tensor(t_tensor, A, B)
        diff     = get_pearson_similarity(C, D) - target_ratio * get_pearson_similarity(C, E)
        if abs(diff) < best_val:
            best_val = abs(diff)
            best_t   = t_val
    return best_t



def compute_slerp_ratio_for_target(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor, target: float) -> float:
    """
    Given three unit vectors A, B, and D (all assumed to be coplanar)
    and a target cosine similarity (target) for the slerp result C with D,
    compute the interpolation parameter t such that:
        C = slerp(t, A, B)
        and cos(C, D) ≈ target.

    Args:
        A: Tensor of shape (D,), starting vector.
        B: Tensor of shape (D,), ending vector.
        D: Tensor of shape (D,), the reference vector.
        target: Desired cosine similarity between C and D.

    Returns:
        t: A float between 0 and 1.
    """
    A = A / (A.norm() + 1e-8)
    B = B / (B.norm() + 1e-8)
    D = D / (D.norm() + 1e-8)
    
    alpha = math.acos(max(-1.0, min(1.0, float(torch.dot(D, A))))) # angel between D and A
    beta  = math.acos(max(-1.0, min(1.0, float(torch.dot(D, B))))) # angle between D and B
    
    delta = math.acos(max(-1.0, min(1.0, target))) # target cosine similarity... angle etc...
    
    if abs(beta - alpha) < 1e-6:
        return 0.5
    
    t = (delta - alpha) / (beta - alpha)
    t = max(0.0, min(1.0, t))
    return t



# TENSOR NORMALIZATION OPS

def normalize_zscore(x, channelwise=False, inplace=False):
    if inplace:
        if channelwise:
            return x.sub_(x.mean(dim=(-2,-1), keepdim=True)).div_(x.std(dim=(-2,-1), keepdim=True))
        else:
            return x.sub_(x.mean()).div_(x.std())
    else:
        if channelwise:
            return (x - x.mean(dim=(-2,-1), keepdim=True) / x.std(dim=(-2,-1), keepdim=True))
        else:
            return (x - x.mean()) / x.std()

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

def line_intersection(a: torch.Tensor, d1: torch.Tensor, b: torch.Tensor, d2: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Computes the intersection (or closest point average) of two lines in R^D.
    
    The first line is defined by:  L1: x = a + t * d1
    The second line is defined by: L2: x = b + s * d2
    
    If the lines do not exactly intersect, this function returns the average of the closest points.
    
    a, d1, b, d2: Tensors of shape (D,) or with an extra batch dimension (B, D).
    Returns: Tensor of shape (D,) or (B, D) representing the intersection (or midpoint of closest approach).
    """
    # Compute dot products
    d1d1 = (d1 * d1).sum(dim=-1, keepdim=True)  # shape (B,1) or (1,)
    d2d2 = (d2 * d2).sum(dim=-1, keepdim=True)
    d1d2 = (d1 * d2).sum(dim=-1, keepdim=True)
    
    r = b - a  # shape (B, D) or (D,)
    r_d1 = (r * d1).sum(dim=-1, keepdim=True)
    r_d2 = (r * d2).sum(dim=-1, keepdim=True)
    
    # Solve for t and s:
    # t * d1d1 - s * d1d2 = r_d1
    # t * d1d2 - s * d2d2 = r_d2
    # Solve using determinants:
    denom = d1d1 * d2d2 - d1d2 * d1d2
    # Avoid division by zero
    denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    t = (r_d1 * d2d2 - r_d2 * d1d2) / denom
    s = (r_d1 * d1d2 - r_d2 * d1d1) / denom
    
    point1 = a + t * d1
    point2 = b + s * d2
    # If they intersect exactly, point1 and point2 are identical.
    # Otherwise, return the midpoint of the closest points.
    return (point1 + point2) / 2

def slerp_direction(t: float, u0: torch.Tensor, u1: torch.Tensor, DOT_THRESHOLD=0.9995) -> torch.Tensor:
    dot = (u0 * u1).sum(-1).clamp(-1.0, 1.0) #u0, u1 are unit vectors... should not be affected by clamp
    if dot.item() > DOT_THRESHOLD: # u0, u1 nearly aligned, fallback to lerp
        return torch.lerp(u0, u1, t)
    theta_0     = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t     = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0          = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1          = sin_theta_t / sin_theta_0
    return s0 * u0 + s1 * u1

def magnitude_aware_interpolation(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:

    m0 = v0.norm(dim=-1, keepdim=True)
    m1 = v1.norm(dim=-1, keepdim=True)

    u0 = v0 / (m0 + 1e-8)
    u1 = v1 / (m1 + 1e-8)
    
    u = slerp_direction(t, u0, u1)
    
    m = (1 - t) * m0 + t * m1 # tinerpolate magnitudes linearly
    return m * u


def slerp_tensor(val: torch.Tensor, low: torch.Tensor, high: torch.Tensor, dim=-3) -> torch.Tensor:
    #dim = (2,3)
    if low.ndim == 4 and low.shape[-3] > 1:
        dim=-3
    elif low.ndim == 5 and low.shape[-3] > 1:
        dim=-4
    elif low.ndim == 2:
        dim=(-2,-1)
        
    if type(val) == float:
        val = torch.Tensor([val]).expand_as(low).to(low.dtype).to(low.device)
        
    if val.shape != low.shape:
        val = val.expand_as(low)
        
    low_norm = low / (torch.norm(low, dim=dim, keepdim=True))
    high_norm = high / (torch.norm(high, dim=dim, keepdim=True))
    
    dot = (low_norm * high_norm).sum(dim=dim, keepdim=True).clamp(-1.0, 1.0)
    
    #near = ~(-0.9995 < dot < 0.9995) #dot > 0.9995 or dot < -0.9995
    near = dot > 0.9995
    opposite = dot < -0.9995

    condition = torch.logical_or(near, opposite)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)

    if val.ndim < low.ndim:
        val = val.unsqueeze(dim)
    
    factor_low = torch.sin((1 - val) * omega) / so
    factor_high = torch.sin(val * omega) / so

    res = factor_low * low + factor_high * high
    res = torch.where(condition, low * (1 - val) + high * val, res)
    return res




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
    
    t_batch_dim_count: int = max(0, t.ndim-v0.ndim) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.ndim))
    
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




def make_checkerboard(tile_size: int, num_tiles: int, dtype=torch.float16, device="cpu"):
    pattern = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    board = pattern.repeat(num_tiles // 2 + 1, num_tiles // 2 + 1)[:num_tiles, :num_tiles]
    board_expanded = board.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
    return board_expanded



