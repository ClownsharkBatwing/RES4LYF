import re


def get_extra_options_kv(key, default, extra_options):
    match = re.search(rf"{key}\s*=\s*([a-zA-Z0-9_]+)", extra_options)
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


