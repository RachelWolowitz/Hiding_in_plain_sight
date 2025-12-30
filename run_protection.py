import warnings
warnings.filterwarnings("ignore")
import sys
import os 
import random
import argparse
import re
sys.path.append(".")
sys.path.append("..")
from protection import  ddim_sample_adv_momentum
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torch.backends import cudnn
import numpy as np 
from PIL import Image
from geoclip import GeoCLIP

parser = argparse.ArgumentParser()
parser.add_argument('--res', default=512, type=int, help='Input image resized resolution') 
parser.add_argument('--SD_name', default="stabilityai/stable-diffusion-2-base", type=str)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--images_root', default="./images", type=str,
                    help='The images root directory')
parser.add_argument('--timesteps', type=int, default=100)
parser.add_argument('--start_step', default=95, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--ddim_eta', type=float, default=0.0)
parser.add_argument('--iterations', type=int, default=2,help='The number of attack iterations')
parser.add_argument('--s', type=float, default=0.7,help='adversarial weight for each step in an iteration')
parser.add_argument('--a', type=float, default=0.5,help='adversarial weight for last step in an iteration')
parser.add_argument('--eta', type=float, default=0.0)
parser.add_argument('--LAMBDA_EDGE', type=float, default=0.0)
parser.add_argument('--beta', type=float, default=0.5,help='momentum decay factor')
parser.add_argument('--save_dir', type=str, default='/home/mlsnrs/data/wyn/NAS/GEO/out/adv_image/')
args = parser.parse_args()

def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    ## Stable Diffusion Model Loading
    SD_name = args.SD_name
    model = StableDiffusionPipeline.from_pretrained(
        SD_name,
        local_files_only=False, 
        device_map="balanced",
    )
    model.enable_attention_slicing(slice_size="auto")
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.enable_vae_slicing()
    model.enable_vae_tiling()
    model.vae.eval()
    model.unet.eval()
    model.text_encoder.eval()

    vic_model = GeoCLIP()
    vic_model.eval()
    
    timesteps = args.timesteps
    scale = args.scale  
    res = args.res
    start_step = args.start_step
    iterations = args.iterations
    eta = args.eta

    ## Load Images and Prompts
    image_dir = args.images_root
    image_paths = find_images(image_dir)

    adv_images = []
    images = []

    for ind, image_path in enumerate(image_paths):
        model.vae.zero_grad()
        model.unet.zero_grad()
        model.text_encoder.zero_grad()

        tmp_image = Image.open(image_path).convert('RGB')

        adv_image = ddim_sample_adv_momentum(
            model,
            label,
            vic_model,
            prompt=[""], 
            save_path=os.path.join(save_dir, str(ind).rjust(4, '0')),
            timesteps=timesteps, 
            guidance_scale=scale,
            image=tmp_image,
            ind=ind,
            res=res, 
            start_step=start_step,
            eta=eta,
            iterations=iterations,
            LAMBDA_EDGE=args.LAMBDA_EDGE,
            s=args.s, # latents = (latents + s * v_inner.float())
            a=args.a, # pri_latents = (pri_latents + a * gradient.float())
            beta=args.beta, # v_inner = beta * v_inner + (1 - beta) * gradient
            )

        adv_image = adv_image.astype(np.float32) / 255.0
        adv_images.append(adv_image[None].transpose(0, 3, 1, 2))

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        images.append(tmp_image)
            
if __name__ == '__main__':
    main()




