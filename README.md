---
license: mit
---
# Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision–Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

**NOTE**: To prevent potetial harm, we release our source code only *upon request for research purposes*.

## Overview
Recent advances in multimodal large reasoning models (MLRMs) have endowed multimodal large language models with impressive performance on complex visual understanding tasks.
However, these capabilities also introduce serious geolocation privacy risks: adversaries can exploit MLRMs to analyze images shared on social media, and infer precise user locations by leveraging subtle visual cues. 
As image sharing becomes increasingly pervasive, this emerging attack surface threatens to expose sensitive personal information, such as home addresses, daily routines, and frequently visited places, thereby enabling downstream risks including stalking and physical harm.
In this work, we propose a diffusion-based geolocation privacy protection framework that injects robust, perceptually realistic, and black-box transferable perturbations into user images, effectively preventing MLRMs from inferring precise geographic locations.
We first conduct an in-depth analysis of refusal-based protection mechanisms widely deployed by MLRM providers and empirically demonstrate their vulnerability to jailbreak attacks. 
To achieve robust protection directly at the image level, we leverage gradient guidance from GeoCLIP, a geolocation-specialized CLIP model, and introduce semantic-preserving perturbations in the latent space during diffusion reverse process.
The resulting protected images provide highly transferable protection across commercial MLRM services while maintaining strong visual quality and usability. 
Moreover, we support granular geolocation privacy control, allowing users to flexibly regulate the level of location disclosure (e.g., city-, region-, or country-level). By integrating diffusion-based inpainting, we extend geolocation gradient guidance to precisely edit the visual cues exploited by MLRMs, producing more fine-grained and semantically consistent modifications than standard inpainting baselines.
Extensive experiments on five latest commercial MLRM APIs demonstrate the effectiveness of our framework, inducing location prediction deviations of over 1000 km and reducing 1 km–level geolocation accuracy to below 5\%.

## Usage
We provide the code files for diffusion-based perturbation, MLRM response generation, and evalation metrics in this repository. The main implementation is in `run_protection.py` and `protection.py`.


The **recommended usage** is as follows:

1. Prepare the environment.
```
cd Hiding_in_plain_sight
pip install -r requirements.txt
```

2. Prepare the *DoxBench* and *Street View* dataset according to the official repository: [DoxBench](https://huggingface.co/datasets/MomoUchi/DoxBench), [Street View](https://github.com/njspyx/location-inference).
   
3. Run the diffusion-based perturbation defense:
```
./run_protection.sh
```

4. Prepare your API and generate MLRM responses with protected images:
```
python ./generation/v3.py
```

5. Evaluate MLRM responses for geolocation leakage:
```
python ./evaluation/distance.py
python ./evaluation/deviation.py
```

## Acknowledgement
This repo is based on the codebase of [Venom](https://github.com/huizhg/VENOM). We sincerely thank the contributors for their valuable work.