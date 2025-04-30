import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar, List

import re
import functools

from comfy.samplers import SCHEDULER_NAMES

from .res4lyf import RESplain




# EXTRA_OPTIONS OPS

class ExtraOptions():
    def __init__(self, extra_options):
        self.extra_options = extra_options
        self.mute          = False
        
    def __call__(self, option, default=None, ret_type=None, match_all_flags=False):
        if isinstance(option, (tuple, list)):
            if match_all_flags:
                return all(self(single_option, default, ret_type) for single_option in option)
            else:
                return any(self(single_option, default, ret_type) for single_option in option)

        if default is None: # get flag
            pattern = rf"^(?:{re.escape(option)}\s*$|{re.escape(option)}=)"
            return bool(re.search(pattern, self.extra_options, flags=re.MULTILINE))
        elif ret_type is None:
            ret_type = type(default)
        
            if ret_type.__module__ != "builtins":
                mod = __import__(default.__module__)
                ret_type = lambda v: getattr(mod, v, None)
        
        if ret_type == list:
            pattern = rf"^{re.escape(option)}\s*=\s*([a-zA-Z0-9_.,+-]+)\s*$"
            match   = re.search(pattern, self.extra_options, flags=re.MULTILINE)
            
            if match:
                value = match.group(1)
                if not self.mute:
                    RESplain("Set extra_option: ", option, "=", value)
            else:
                value = default
                
            if type(value) == str: 
                value = value.split(',')
            
                if type(default[0]) == type:
                    ret_type = default[0]
                else:
                    ret_type = type(default[0])
                
                value = [ret_type(value[_]) for _ in range(len(value))]
        
        else:
            pattern = rf"^{re.escape(option)}\s*=\s*([a-zA-Z0-9_.+-]+)\s*$"
            match = re.search(pattern, self.extra_options, flags=re.MULTILINE)
            if match:
                if ret_type == bool:
                    value_str = match.group(1).lower()
                    value = value_str in ("true", "1", "yes", "on")
                else:
                    value = ret_type(match.group(1))
                if not self.mute:
                    RESplain("Set extra_option: ", option, "=", value)
            else:
                value = default
        
        return value




def extra_options_flag(flag, extra_options):
    pattern = rf"^(?:{re.escape(flag)}\s*$|{re.escape(flag)}=)"
    return bool(re.search(pattern, extra_options, flags=re.MULTILINE))

def get_extra_options_kv(key, default, extra_options, ret_type=None):
    ret_type = type(default) if ret_type is None else ret_type

    pattern = rf"^{re.escape(key)}\s*=\s*([a-zA-Z0-9_.+-]+)\s*$"
    match = re.search(pattern, extra_options, flags=re.MULTILINE)
    
    if match:
        value = match.group(1)
    else:
        value = default
        
    return ret_type(value)

def get_extra_options_list(key, default, extra_options, ret_type=None):
    default = [default] if type(default) != list else default
    
    #ret_type = type(default)    if ret_type is None else ret_type
    ret_type = type(default[0]) if ret_type is None else ret_type

    pattern = rf"^{re.escape(key)}\s*=\s*([a-zA-Z0-9_.,+-]+)\s*$"
    match   = re.search(pattern, extra_options, flags=re.MULTILINE)
    
    if match:
        value = match.group(1)
    else:
        value = default
    
    if type(value) == str:
        value = value.split(',')
    
    value = [ret_type(value[_]) for _ in range(len(value))]
        
    return value



class OptionsManager:
    APPEND_OPTIONS = {"extra_options"}

    def __init__(self, options, **kwargs):
        self.options_list = []
        if options is not None:
            self.options_list.append(options)

        for key, value in kwargs.items():
            if key.startswith('options') and value is not None:
                self.options_list.append(value)

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

    def update(self, key_or_dict, value=None):
        """Update options with a single key-value pair or a dictionary"""
        if value is not None or isinstance(key_or_dict, (str, list)):
            # single key-value update
            key_path = key_or_dict
            if isinstance(key_path, str):
                key_path = key_path.split('.')

            update_dict = {}
            current = update_dict

            for i, key in enumerate(key_path[:-1]):
                current[key] = {}
                current = current[key]

            current[key_path[-1]] = value

            self.add_option(update_dict)
        else:
            # dictionary update
            flat_updates = {}

            def _flatten_dict(d, prefix=""):
                for key, value in d.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        _flatten_dict(value, full_key)
                    else:
                        flat_updates[full_key] = value

            _flatten_dict(key_or_dict)

            for key_path, value in flat_updates.items():
                self.update(key_path, value)  # Recursive call

        return self

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





# MISCELLANEOUS OPS

def has_nested_attr(obj, attr_path):
    attrs = attr_path.split('.')
    for attr in attrs:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True

def safe_get_nested(d, keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d

class AlwaysTrueList:
    def __contains__(self, item):
        return True

def parse_range_string(s):
    if "all" in s:
        return AlwaysTrueList()

    result = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        val = float(part) if '.' in part else int(part)
        result.append(val)
    return result

def parse_range_string_int(s):
    if "all" in s:
        return AlwaysTrueList()
    
    result = []
    for part in s.split(','):
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end) + 1))
        elif part.strip() != '':
            result.append(int(part))
    return result




# COMFY OPS

def is_video_model(model):
    is_video_model = False
    try :
        is_video_model =    'video'  in model.inner_model.inner_model.model_config.unet_config['image_model'] or \
                            'cosmos' in model.inner_model.inner_model.model_config.unet_config['image_model'] or \
                            'wan2'   in model.inner_model.inner_model.model_config.unet_config['image_model'] or \
                            'ltxv'   in model.inner_model.inner_model.model_config.unet_config['image_model']    
    except:
        pass
    return is_video_model

def is_RF_model(model):
    from comfy import model_sampling
    modelsampling = model.inner_model.inner_model.model_sampling
    return isinstance(modelsampling, model_sampling.CONST)

def get_res4lyf_scheduler_list():
    scheduler_names = SCHEDULER_NAMES.copy()
    if "beta57" not in scheduler_names:
        scheduler_names.append("beta57")
    return scheduler_names

def move_to_same_device(*tensors):
    if not tensors:
        return tensors
    device = tensors[0].device
    return tuple(tensor.to(device) for tensor in tensors)

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)
    return c





# MISC OPS

def initialize_or_scale(tensor, value, steps):
    if tensor is None:
        return torch.full((steps,), value)
    else:
        return value * tensor


def pad_tensor_list_to_max_len(tensors: List[torch.Tensor], dim: int = -2) -> List[torch.Tensor]:
    """Zero-pad each tensor in `tensors` along `dim` up to their common maximum length."""
    max_len = max(t.shape[dim] for t in tensors)
    padded = []
    for t in tensors:
        cur = t.shape[dim]
        if cur < max_len:
            pad_shape = list(t.shape)
            pad_shape[dim] = max_len - cur
            zeros = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
            t = torch.cat((t, zeros), dim=dim)
        padded.append(t)
    return padded



class PrecisionTool:
    def __init__(self, cast_type='fp64'):
        self.cast_type = cast_type

    def cast_tensor(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.cast_type not in ['fp64', 'fp32', 'fp16']:
                return func(*args, **kwargs)

            target_device = None
            for arg in args:
                if torch.is_tensor(arg):
                    target_device = arg.device
                    break
            if target_device is None:
                for v in kwargs.values():
                    if torch.is_tensor(v):
                        target_device = v.device
                        break
            
        # recursively zs_recast tensors in nested dictionaries
            def cast_and_move_to_device(data):
                if torch.is_tensor(data):
                    if self.cast_type == 'fp64':
                        return data.to(torch.float64).to(target_device)
                    elif self.cast_type == 'fp32':
                        return data.to(torch.float32).to(target_device)
                    elif self.cast_type == 'fp16':
                        return data.to(torch.float16).to(target_device)
                elif isinstance(data, dict):
                    return {k: cast_and_move_to_device(v) for k, v in data.items()}
                return data

            new_args = [cast_and_move_to_device(arg) for arg in args]
            new_kwargs = {k: cast_and_move_to_device(v) for k, v in kwargs.items()}
            
            return func(*new_args, **new_kwargs)
        return wrapper

    def set_cast_type(self, new_value):
        if new_value in ['fp64', 'fp32', 'fp16']:
            self.cast_type = new_value
        else:
            self.cast_type = 'fp64'

precision_tool = PrecisionTool(cast_type='fp64')




class FrameWeightsManager:
    def __init__(self):
        self.frame_weights = None
        self.frame_weights_inv = None
        self.dynamics = "linear"
        self.dynamics_inv = "linear"
        self.schedule = "moderate_early"
        self.schedule_inv = "moderate_early"
        self.scale = 0.5
        self.scale_inv = 0.5
        self.is_reversed = False
        self.is_reversed_inv = False
        self.dtype = torch.float64
        self.device = torch.device('cpu')
        
    def set_device_and_dtype(self, device=None, dtype=None):
        """Set the device and dtype for generated weights"""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self
    
    def _generate_frame_weights(self, num_frames, dynamics, schedule, scale, is_reversed, frame_weights):
        if "fast" in schedule:
            rate_factor = 0.25
        elif "slow" in schedule:
            rate_factor = 1.0
        else: # moderate
            rate_factor = 0.5

        if "early" in schedule:
            start_change_factor = 0.0
        elif "late" in schedule:
            start_change_factor = 0.2
        else:
            start_change_factor = 0.0

        change_frames = max(round(num_frames * rate_factor), 2)
        change_start = round(num_frames * start_change_factor)
        low_value = 1.0 - scale

        if frame_weights is not None:
            weights = torch.cat([frame_weights, torch.full((num_frames,), frame_weights[-1])])
            weights = weights[:num_frames]
        else:
            if dynamics == "constant":
                weights = self._generate_constant_schedule(change_frames, low_value)
            elif dynamics == "linear":
                weights = self._generate_linear_schedule(change_frames, low_value)
            elif dynamics == "ease_out":
                weights = self._generate_easeout_schedule(change_frames, low_value)
            elif dynamics == "ease_in":
                weights = self._generate_easein_schedule(change_frames, low_value)
            else:
                raise ValueError(f"Invalid schedule: {dynamics}")

        # Prepend with change_start frames of 1.0 (unless constant)
        if dynamics != "constant":
            weights = torch.cat([torch.full((change_start,), 1.0), weights])
        # Fill remaining with final value
        weights = torch.cat([weights, torch.full((num_frames,), weights[-1])])
        weights = weights[:num_frames]

        weights = torch.flip(weights, [0]) if is_reversed else weights
        weights = weights.to(dtype=self.dtype, device=self.device)

        return weights

    def get_frame_weights_inv(self, num_frames):
        return self._generate_frame_weights(
            num_frames,
            self.dynamics_inv,
            self.schedule_inv,
            self.scale_inv,
            self.is_reversed_inv,
            self.frame_weights_inv
        )
    
    def get_frame_weights(self, num_frames):
        return self._generate_frame_weights(
            num_frames,
            self.dynamics,
            self.schedule,
            self.scale,
            self.is_reversed,
            self.frame_weights
        )

    def _generate_constant_schedule(self, timepoints, low_value):
        """constant schedule with the scale as the low weight"""
        return torch.ones(timepoints) * low_value
    
    def _generate_linear_schedule(self, timepoints, low_value):
        """linear schedule from 1 to the low weight
        1.0 |^
            | \
            |  \
            |   \
            |    \
            |     \
            |      \
            |       \
        low |        \
        0.0 +----------
             0        1
                time
        """
        return torch.linspace(1, low_value, timepoints)
    
    def _generate_easeout_schedule(self, timepoints, low_value):
        """exponential schedule from 1 to the low weight
        1.0 |^\_
            |   \\_
            |     \\__
            |        \\_
            |          \\__
            |             \\____
            |                   \\________
        low |                             \______
        0.0 +----------------------------------------
             0                                     1
                              time
        """
        timepoints = max(timepoints, 4)
        t = torch.linspace(0, 1, timepoints, dtype=self.dtype, device=self.device)
        weights = torch.pow(low_value, t)
        return weights

    def _generate_easein_schedule(self, timepoints, low_value):
        """a monomial power schedule from 1 to the low weight
        1.0 |^--_____
            |         \_____
            |                \____
            |                      \___
            |                          \__
            |                             \_
            |                               \
        low |                                 \
        0.0 +----------------------------------
             0                                1
                            time
        """
        timepoints = max(timepoints, 4)
        t = torch.linspace(0, 1, timepoints, dtype=self.dtype, device=self.device)
        weights = 1 - (1 - low_value) * torch.pow(t, 2)
        return weights
    
    