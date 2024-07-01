# RES4LYF

Better description to come later...

These nodes are for advanced manipulation of sigmas, latents, and various advanced sampling methods, including scheduling all parameters for the RES sampler (the code for which was adapted from https://github.com/Clybius/ComfyUI-Extra-Samplers) via a node called "ClownSampler". A variety of new noise types have also been added, which are available as noise sampler options and in "SharkSampler" (which is effectively an advanced version of SamplerCustomNoise with a number of added noise types).

**Installation: 
**If you are using a venv, you will need to first run from within your ComfyUI folder (that contains your "venv" folder):

source venv/bin/activate
(if you are on Linux)

venv\Scripts\activate
(if you are on Windows)

Then, "cd" into your "custom_nodes" folder and run the following commands:

git clone https://github.com/ClownsharkBatwing/RES4LYF/

cd RES4LYF

If you are using a venv, run these commands:

pip install -r requirements.txt

pip install opensimplex --no-deps

Alternatively, if you are using the portable version of ComfyUI you will need to replace "pip" with the path to your embedded pip executable. For example, on Windows:

X:\path\to\your\comfy_portable_folder\python_embedded\Scripts\pip.exe install -r requirements.txt

X:\path\to\your\comfy_portable_folder\python_embedded\Scripts\pip.exe install opensimplex --no-deps



