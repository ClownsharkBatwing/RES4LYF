import re
import torch
from comfy.samplers import SCHEDULER_NAMES

def get_extra_options_kv(key, default, extra_options):

    match = re.search(rf"{key}\s*=\s*([a-zA-Z0-9_.+-]+)", extra_options)
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


def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor
    
def get_res4lyf_scheduler_list():
    scheduler_names = SCHEDULER_NAMES.copy()
    if "beta57" not in scheduler_names:
        scheduler_names.append("beta57")
    return scheduler_names
    