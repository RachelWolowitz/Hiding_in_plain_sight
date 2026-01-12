#!/bin/bash
export CUDA_VISIBLE_DEVICES=[GPUS]

python run_protection.py --res 512 \
 --SD_name "stabilityai/stable-diffusion-2-base" \
 --scale 0.0 \
 --images_root /path/to/images_root \ 
 --beta 0.5 \
 --timesteps 50 \
 --start_step 30 \
 --iterations 15 \
 --s 1.0 \
 --a 0.5 \
 --eta 0.0 \
 --LAMBDA_EDGE 1.0 \
 --save_dir /path/to/save_dir
