# RES4LYF

These nodes are for advanced manipulation of sigmas, latents, and various advanced sampling methods. At the heart of the repository is the "ClownSampler", which was specifically designed to support both rectified flow and probability flow models. It features 25 different selectible samplers all available in both ODE or SDE modes with 17 noise types, 6 noise scaling modes, and options for implicit Runge-Kutta sampling refinement steps. Several new samplers are implemented, including RES_3S, RES_3M, and RES_2M. Additionally, img2img capabilities include both latent image guidance or unsampling/resampling (via rectified noise inversion). 

**INSTALLATION:** 

If you are using a venv, you will need to first run from within your ComfyUI folder (that contains your "venv" folder):

_Linux:_

source venv/bin/activate

_Windows:_

venv\Scripts\activate

_Then, "cd" into your "custom_nodes" folder and run the following commands:_

git clone https://github.com/ClownsharkBatwing/RES4LYF/

cd RES4LYF

_If you are using a venv, run these commands:_

pip install -r requirements.txt

_Alternatively, if you are using the portable version of ComfyUI you will need to replace "pip" with the path to your embedded pip executable. For example, on Windows:_

X:\path\to\your\comfy_portable_folder\python_embedded\Scripts\pip.exe install -r requirements.txt
