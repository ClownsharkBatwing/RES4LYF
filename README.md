# RES4LYF

Better description to come later...

These nodes are for advanced manipulation of sigmas, latents, and various advanced sampling methods, including scheduling all parameters for the RES sampler (the code for which was adapted from https://github.com/Clybius/ComfyUI-Extra-Samplers) via a node called "ClownSampler". A variety of new noise types have also been added, which are available as noise sampler options and in "SharkSampler" (which is effectively an advanced version of SamplerCustomNoise with a number of added noise types).

Installation: 
"cd" into your ComfyUI/custom_nodes folder and run the following commands:

git clone https://github.com/ClownsharkBatwing/RES4LYF/
cd RES4LYF
pip install -r requirements
pip install opensimplex --no-deps

Bear in mind that if you are using the portable version of ComfyUI you will need to replace "pip" with the full path to your embedded pip.exe, and if you are using a venv, you will need to first run:

source venv/bin/activate
(if you are on Linux)

venv\Scripts\activate
(if you are on Windows)

