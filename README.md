# Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision–Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

🎉 **NEWS:** Our paper has been accepted to **NDSS 2027**!
**NOTE**: To prevent potetial harm, we release our source code only *upon request for research purposes*.

## Overview
Multimodal large reasoning models (MLRMs) have demonstrated remarkable capabilities in complex visual understanding.
However, this very power introduces a critical yet underexplored privacy threat: adversaries can exploit MLRMs to precisely infer users' geographic locations from casually shared photographs, by performing structured reasoning over subtle visual cues such as architectural styles, vegetation, and lighting conditions.
This capability exposes sensitive personal information including home addresses and daily routines, enabling severe real-world harms including stalking, surveillance, and targeted harassment.

In this work, we present a systematic study of MLRM-driven \textit{geolocation privacy leakage}.
We first reveal that refusal-based safeguards are critically insufficient, as carefully crafted jailbreak prompts can raise model response rates to 100\%. 
We further identify that existing defenses, which inject imperceptible perturbations into shared images, suffer from structural limitations intrinsic to their pixel-space optimization, resulting in degraded black-box transferability and pronounced visual artifacts.

Motivated by these findings, we propose a diffusion-based framework for geolocation privacy protection that addresses these failure modes at root.
By injecting perturbations into the \textit{latent space} of a diffusion model during reverse sampling, our method operates directly on high-level semantic representations, improving the effectiveness-utility trade-off by construction.
We further anchor the optimization with GeoCLIP, a model explicitly aligned with GPS coordinates, to 
target the geographic semantic signals that MLRMs exploit for location inference, achieving stronger black-box transferability without perceptible image degradation. 
We additionally extend the framework with diffusion inpainting mechanisms for granular, user-configurable geographic disclosure, enabling selective modification of location cues at country or region granularity.

Extensive experiments on five leading commercial MLRM APIs (e.g., GPT-5 and Claude Opus 4.5) validate the effectiveness of our framework, which substantially increases location prediction errors and reduces 1,km-level leakage accuracy to below 5\%.
Our approach consistently outperforms baselines in protection efficacy and image utility, while demonstrating robustness against image transformation and purification attacks.

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

## Citation
If you find this work useful, please cite:
```bibtex
@article{wang2026hiding,
  title={Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models},
  author={Wang, Yining and Li, Xi and Zhang, Mi and Zhang, Xiaohan and You, Xiaoyu and Qian, Zhenxing and Wen, Mi},
  journal={arXiv preprint arXiv:2609.21363},
  year={2026}
}

## Acknowledgement
This repo is based on the codebase of [Venom](https://github.com/huizhg/VENOM). We sincerely thank the contributors for their valuable work.
