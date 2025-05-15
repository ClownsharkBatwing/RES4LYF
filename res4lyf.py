# Code adapted from https://github.com/pythongosssss/ComfyUI-Custom-Scripts

import asyncio
import os
import json
import shutil
import inspect
import aiohttp
import math
import comfy.model_sampling
import comfy.samplers
from aiohttp import web
from server import PromptServer
from tqdm import tqdm


CONFIG_FILE_NAME = "res4lyf.config.json"
DEFAULT_CONFIG_FILE_NAME = "web/js/res4lyf.default.json"
config = None

using_RES4LYF_time_snr_shift = False
original_time_snr_shift = comfy.model_sampling.time_snr_shift

def time_snr_shift_RES4LYF(alpha, t):
    if using_RES4LYF_time_snr_shift and get_config_value("updatedTimestepScaling", False):
        out = math.exp(alpha) / (math.exp(alpha) + (1 / t - 1) ** 1.0)
    else:
        out = original_time_snr_shift(alpha, t)
    return out

display_sampler_category = False

def get_display_sampler_category():
    global display_sampler_category
    return display_sampler_category
    
@PromptServer.instance.routes.post("/reslyf/settings")
async def update_settings(request):
    try:
        json_data = await request.json()
        setting = json_data.get("setting")
        value = json_data.get("value")

        if setting:
            save_config_value(setting, value)
            
            if setting == "updatedTimestepScaling":
                global using_RES4LYF_time_snr_shift
                using_RES4LYF_time_snr_shift = value
                if ( using_RES4LYF_time_snr_shift is True ):
                    RESplain("Using RES4LYF time SNR shift")
                else:
                    RESplain("Disabled RES4LYF time SNR shift")
            elif setting == "displayCategory":
                global display_sampler_category
                display_sampler_category = value
                if ( display_sampler_category is True ):
                    RESplain("Displaying sampler category", debug=True)
                else:
                    RESplain("Not displaying sampler category", debug=True)


        return web.Response(status=200)
    except Exception as e:
        return web.Response(status=500, text=str(e))

@PromptServer.instance.routes.post("/reslyf/log")
async def log_message(request):
    try:
        json_data = await request.json()
        log_text = json_data.get("log")
        
        if log_text:
            RESplain(log_text, debug=True)
            return web.Response(status=200)
        else:
            return web.Response(status=400, text="No log text provided")
    except Exception as e:
        return web.Response(status=500, text=str(e))
    
original_calculate_sigmas = comfy.samplers.calculate_sigmas

def calculate_sigmas_RES4LYF(model_sampling, scheduler_name, steps):
    if scheduler_name == "beta57":
        sigmas = comfy.samplers.beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)
    else:
        return original_calculate_sigmas(model_sampling, scheduler_name, steps)
    return sigmas

def init(check_imports=None):
    RESplain("Init")

    # initialize display category
    global display_sampler_category
    display_sampler_category = get_config_value("displayCategory", False)
    if ( display_sampler_category is True ):
        RESplain("Displaying sampler category", debug=True)

    # Initialize using_RES4LYF_time_snr_shift from config (deprecated, disabled by default)
    global using_RES4LYF_time_snr_shift
    using_RES4LYF_time_snr_shift = get_config_value("updatedTimestepScaling", False)
    if using_RES4LYF_time_snr_shift:
        comfy.model_sampling.time_snr_shift = time_snr_shift_RES4LYF
        RESplain("Using RES4LYF time SNR shift but this is deprecated and will be disabled at some completely unpredictable point in the future")

    # monkey patch comfy.samplers.calculate_sigmas with custom implementation
    comfy.samplers.calculate_sigmas = calculate_sigmas_RES4LYF
    if "beta57" not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES = comfy.samplers.SCHEDULER_NAMES + ["beta57"]
    if "beta57" not in comfy.samplers.KSampler.SCHEDULERS:
        comfy.samplers.KSampler.SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ["beta57"]

    return True


def save_config_value(key, value):
    config = get_extension_config()
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

    config_path = get_ext_dir(CONFIG_FILE_NAME)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def get_config_value(key, default=None, throw=False):
    config = get_extension_config()
    keys = key.split(".")
    d = config
    for k in keys[:-1]:
        if k not in d:
            if throw:
                raise KeyError("Configuration key missing: " + key)
            else:
                return default
        d = d[k]
    return d.get(keys[-1], default)


def is_debug_logging_enabled():
    logging_enabled = get_config_value("enableDebugLogs", False)
    return logging_enabled

def RESplain(*args, debug='info'):
    if isinstance(debug, bool):
        type = 'debug' if debug else 'info'
    else:
        type = debug

    if type == 'debug' and not is_debug_logging_enabled():
        return
    
    if not args:
        return

    name = get_extension_config()["name"]

    message = " ".join(map(str, args))

    if type != 'debug' and type != 'warning':
        print(f"({name}) {message}")
    else:
        print(f"({name} {type}) {message}")

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def merge_default_config(config, default_config):
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            config[key] = merge_default_config(config.get(key, {}), value)
    return config

def get_extension_config(reload=False):
    global config
    if not reload and config is not None:
        return config

    config_path = get_ext_dir(CONFIG_FILE_NAME)
    default_config_path = get_ext_dir(DEFAULT_CONFIG_FILE_NAME)
    
    if os.path.exists(default_config_path):
        with open(default_config_path, "r") as f:
            default_config = json.loads(f.read())
    else:
        default_config = {}

    if not os.path.exists(config_path):
        config = default_config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    else:
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        config = merge_default_config(config, default_config)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    return config


def get_comfy_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(inspect.getfile(PromptServer))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_web_ext_dir():
    config = get_extension_config()
    name = config["name"]
    dir = get_comfy_dir("web/extensions/res4lyf")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, name)
    return dir


def link_js(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass
    try:
        os.symlink(src, dst)
        return True
    except:
        import logging
        logging.exception('')
        return False


def is_junction(path):
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False


def install_js():
    src_dir = get_ext_dir("web/js")
    if not os.path.exists(src_dir):
        RESplain("No JS")
        return

    should_install = should_install_js()
    if should_install:
        RESplain("it looks like you're running an old version of ComfyUI that requires manual setup of web files, it is recommended you update your installation.", "warning", True)
    dst_dir = get_web_ext_dir()
    linked = os.path.islink(dst_dir) or is_junction(dst_dir)
    if linked or os.path.exists(dst_dir):
        if linked:
            if should_install:
                RESplain("JS already linked")
            else:
                os.unlink(dst_dir)
                RESplain("JS unlinked, PromptServer will serve extension")
        elif not should_install:
            shutil.rmtree(dst_dir)
            RESplain("JS deleted, PromptServer will serve extension")
        return
    
    if not should_install:
        RESplain("JS skipped, PromptServer will serve extension")
        return
    
    if link_js(src_dir, dst_dir):
        RESplain("JS linked")
        return

    RESplain("Copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def should_install_js():
    return not hasattr(PromptServer.instance, "supports") or "custom_nodes_from_web" not in PromptServer.instance.supports

def get_async_loop():
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_http_session():
    loop = get_async_loop()
    return aiohttp.ClientSession(loop=loop)


async def download(url, stream, update_callback=None, session=None):
    close_session = False
    if session is None:
        close_session = True
        session = get_http_session()
    try:
        async with session.get(url) as response:
            size = int(response.headers.get('content-length', 0)) or None

            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                perc = 0
                async for chunk in response.content.iter_chunked(2048):
                    stream.write(chunk)
                    progressbar.update(len(chunk))
                    if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
                        last = perc
                        perc = round(progressbar.n / progressbar.total, 2)
                        if perc != last:
                            last = perc
                            await update_callback(perc)
    finally:
        if close_session and session is not None:
            await session.close()


async def download_to_file(url, destination, update_callback=None, is_ext_subpath=True, session=None):
    if is_ext_subpath:
        destination = get_ext_dir(destination)
    with open(destination, mode='wb') as f:
        download(url, f, update_callback, session)


def wait_for_async(async_fn, loop=None):
    res = []

    async def run_async():
        r = await async_fn()
        res.append(r)

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    loop.run_until_complete(run_async())

    return res[0]


def update_node_status(client_id, node, text, progress=None):
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    PromptServer.instance.send_sync("res4lyf/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)


async def update_node_status_async(client_id, node, text, progress=None):
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    await PromptServer.instance.send("res4lyf/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)


def get_config_value(key, default=None, throw=False):
    split = key.split(".")
    obj = get_extension_config()
    for s in split:
        if s in obj:
            obj = obj[s]
        else:
            if throw:
                raise KeyError("Configuration key missing: " + key)
            else:
                return default
    return obj


def is_inside_dir(root_dir, check_path):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isabs(check_path):
        check_path = os.path.abspath(os.path.join(root_dir, check_path))
    return os.path.commonpath([check_path, root_dir]) == root_dir


def get_child_dir(root_dir, child_path, throw_if_outside=True):
    child_path = os.path.abspath(os.path.join(root_dir, child_path))
    if is_inside_dir(root_dir, child_path):
        return child_path
    if throw_if_outside:
        raise NotADirectoryError(
            "Saving outside the target folder is not allowed.")
    return None
