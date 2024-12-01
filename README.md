# RES4LYF

At the heart of this repository is the "ClownsharKSampler", which was specifically designed to support both rectified flow and probability flow models. It features 40 different selectible samplers (30 explicit, 7 fully implicit, 3 diagonally implicit) all available in both ODE or SDE modes with 20 noise types, 7 noise scaling modes, and options for implicit Runge-Kutta sampling refinement steps. Several new explicit samplers are implemented, most notably RES_2M, RES_3S, and RES_5S. Additionally, img2img capabilities include both latent image guidance and unsampling/resampling (via new forms of rectified noise inversion). 

A particular emphasis of this project has been to facilitate modulating parameters vs. time, which can facilitate large gains in image quality from the sampling process. To this end, a wide variety of sigma, latent, and noise manipulation nodes are included. 

Much of this work remains experimental and is subject to further changes.

# CURRENT FLUX WORKFLOWS

**TXT2IMG Workflow:**
This uses my amateur cell phone lora, which is freely available (https://huggingface.co/ClownsharkBatwing/CSBW_Style/blob/main/amateurphotos_1_amateurcellphonephoto_recapt2.safetensors). It significantly reduces the plastic, blurred look of Flux Dev.
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20flux.png)
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20WF%20flux.png)

**INPAINTING Workflow:**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/inpainting%20flux.png)
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/inpainting%20WF%20flux.png)

****DUAL GUIDES WITH MASKS Workflow:**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20masked%20flux.png)
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20masked%20WF%20flux.png)

# PREVIOUS WORKFLOWS
**THE FOLLOWING WORKFLOWS ARE FOR A PREVIOUS VERSION OF THE NODE.** 
These will still work! You will, however, need to manually delete and recreate the sampler and guide nodes and input the settings as they appear in the screenshots. The layout of the nodes has been changed slightly. To replicate their behavior precisely, add to the new extra_options box in ClownsharKSampler: truncate_conditioning=true (if that setting was used in the screenshot for the node).

![image](https://github.com/user-attachments/assets/a55ec484-1339-45a2-bcc4-76934f4648d4)

**TXT2IMG Workflow:** 

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20SD35M.png)

**TXT2IMG Workflow (Latent Image Guides):**
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20guided%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20guided%20SD35M.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20guided%20SD35M%20input.png

**TXT2IMG Workflow (Dual Guides with Masking):**
![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M.png)

Input images and mask:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20input1.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20input2.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/txt2img%20dual%20guides%20with%20mask%20SD35M%20mask.png

**IMG2IMG Workflow (Unsampling):** 

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SD35L%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SD35L.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SD35L%20input.png

**IMG2IMG Workflow (Unsampling with SDXL):**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SDXL%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SDXL.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20unsampling%20SDXL%20input.png

**IMG2IMG Workflow (Unsampling with latent image guide):**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20guided%20unsampling%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20guided%20unsampling%20SD35M.png)

Input image:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20guided%20unsampling%20SD35M%20input.png

**IMG2IMG Workflow (Unsampling with dual latent image guides and masking):**

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20output.png)

![image](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M.png)

Input images and mask:
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20input1.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20input2.png
https://github.com/ClownsharkBatwing/RES4LYF/blob/main/workflows/img2img%20dual%20guided%20masked%20unsampling%20SD35M%20mask.png

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

