import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import optim
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
import gc 
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
import os

process_latent = T.Compose([
    T.Resize(
        size=224, 
        interpolation=InterpolationMode.BICUBIC,  # 官方用双三次插值
        antialias=True  # 抗锯齿（PyTorch 1.10+支持）
    ),
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float),
    T.Normalize((0.481, 0.458, 0.408), (0.269, 0.261, 0.276))
])

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def preprocess(image, ind, res, device):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].to(device)
    return 2.0 * image - 1.0

def encoder(image, ind, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, ind, res, generator.device)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)

def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8)
    return latent, latents

def tensor2PIL(img_tensor):
    one = img_tensor[0].detach().cpu()

    from torchvision.transforms.functional import to_pil_image
    pil_img = to_pil_image(one)
    return pil_img

def sobel_edge(x):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sobel_y.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])

    edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return edge

def edge_consistency_loss(x0_t, x0_prev, alpha=1.0):
    e1 = torch.tanh(alpha * sobel_edge(x0_t))
    e2 = torch.tanh(alpha * sobel_edge(x0_prev))
    return (e1 - e2).abs().mean()

def diffusion_step(model, latents, context, t, guidance_scale, extra_step_kwargs):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]

    del latents_input, noise_pred
    gc.collect()
    return latents

def diffusion_separate_step(model, latents, context, t, guidance_scale, extra_step_kwargs):
    uncond_embeddings = context[0].unsqueeze(0)
    text_embeddings = context[1].unsqueeze(0)

    with torch.no_grad():
        noise_pred_uncond = model.unet(
            latents, 
            t,
            encoder_hidden_states=uncond_embeddings
        )["sample"]  

        del uncond_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    with torch.no_grad():
        noise_pred_text = model.unet(
            latents,  
            t,
            encoder_hidden_states=text_embeddings
        )["sample"]

    del text_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = model.scheduler.step(
        noise_pred, t, latents, **extra_step_kwargs
    )["prev_sample"]

    del noise_pred_uncond, noise_pred_text, noise_pred
    torch.cuda.empty_cache()
    gc.collect()

    return latents

@torch.no_grad()
def ddim_reverse_sample(image, ind, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    batch_size = 1
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids)[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids)[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, ind, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    return latents, all_latents

def patch_compute(
    model,
    latents_n, 
):
    max_patch = 5
    B, C, H, W = latents_n.shape
    assert H % max_patch == 0 and W % max_patch == 0, "H, W 必须能被 max_patch 整除"

    h_splits = H // max_patch
    w_splits = W // max_patch

    image = torch.zeros((B, 3, H * 8, W * 8), device="cpu")  

    for i in range(h_splits):
        for j in range(w_splits):
            h_start = i * max_patch
            h_end = h_start + max_patch
            w_start = j * max_patch
            w_end = w_start + max_patch

            latent_patch = latents_n[:, :, h_start:h_end, w_start:w_end]
            decoded_patch = model.vae.decode(latent_patch, return_dict=False)[0].cpu()

            ph_start = h_start * 8
            ph_end = h_end * 8
            pw_start = w_start * 8
            pw_end = w_end * 8

            image[:, :, ph_start:ph_end, pw_start:pw_end] = decoded_patch

            del decoded_patch, latent_patch
            gc.collect()
            torch.cuda.empty_cache()
    
    return image

def ddim_sample_adv_momentum(
        model,
        label,
        vic_model,
        prompt,
        save_path,
        timesteps:int=200,
        guidance_scale:float=2.5,
        image=None,
        ind=None,
        res=512,
        start_step=150,
        eta:float=0.0,
        iterations=5,
        LAMBDA_EDGE=0,
        s:float=1.0, # step
        a:float=0.5, # step
        beta:float=0.5 # momentum strength
        ):

    model.scheduler.set_timesteps(timesteps)
    timesteps_tensor = model.scheduler.timesteps
    extra_step_kwargs = model.prepare_extra_step_kwargs(None, eta)
    total_steps = len(timesteps_tensor)

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    height = width = res
    ori_image = image
    ### DDIM reverse
    latent, inversion_latents = ddim_reverse_sample(image, ind, prompt, model,
                                                    timesteps,
                                                    0, res=height)

    inversion_latents = inversion_latents[::-1]
    orig_latents_list = [x.detach().clone() for x in inversion_latents]

    batch_size = len(prompt)
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids)[0]
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids)[0]

    latent, latents = init_latent(latent, model, height, width, batch_size)
    pri_latents = latent.detach().to(model.device).requires_grad_(True)
    context_no_tune = torch.cat([uncond_embeddings, text_embeddings])
    gps_gallery = vic_model.gps_gallery

    s_ori = s

    #### interation
    for k in tqdm(range(iterations)):   
        latents = pri_latents.detach().requires_grad_(True)

        v_inner = torch.zeros_like(latents)
        beta = beta
        adv_guidance = True

        #### DDIM reverse
        for ind, t in tqdm(
            enumerate(model.scheduler.timesteps[1 + start_step - 1:]),
            desc= "Sampling",
            total=total_steps - start_step,
            leave=False,
        ):
            index = total_steps - start_step - ind - 1

            if k > 2:
                adv_guidance = True

            latents = diffusion_separate_step(model, latents, context_no_tune, t, guidance_scale, extra_step_kwargs)

            if adv_guidance == False:
                continue

            if index > 0 and adv_guidance:
                with torch.enable_grad():
                    latents_n = latents.detach().requires_grad_(True)
                    latents_n = latents_n / model.vae.config.scaling_factor

                    torch.cuda.empty_cache()
                    
                    image = model.vae.decode(latents_n, return_dict=False)[0]
                    image = image.to(latents_n.device)
                    image = model.image_processor.postprocess(image, output_type='pt')
                    image = process_latent(image).cpu()

                    image_features = vic_model.image_encoder(image)
                    image_features = F.normalize(image_features, dim=1)

                    ## target location
                    gps_data = torch.tensor([[-35.2809, 149.1300]], dtype=torch.float32)
                    location_features = vic_model.location_encoder(gps_data)
                    location_features = torch.mean(location_features, dim=0)
                    location_features = F.normalize(location_features, dim=0)

                    tau = 0.07
                    sim = (image_features @ location_features.T) / tau

                    ## TV loss
                    batch_size, channels, height, width = image.shape
                    tv_h = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
                    tv_w = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
                    tv_loss = (tv_h + tv_w) / (batch_size * channels * height * width)

                    LAMBDA_TV = 5
                    if ind == 0:
                        image_prev = image
                        gradient = torch.autograd.grad(sim + LAMBDA_TV * tv_loss, latents_n)[0]
                        print(sim, tv_loss)
                    elif ind >= 1:
                        edge_loss = edge_consistency_loss(image, image_prev)
                        gradient = torch.autograd.grad(sim + LAMBDA_TV * tv_loss + LAMBDA_EDGE * edge_loss, latents_n)[0]
                        print(sim, tv_loss, edge_loss)
                    
                    if ind == 0:
                        v_inner = gradient
                    v_inner = beta * v_inner + (1 - beta) * gradient

                latents = latents + s * v_inner.float()

                del image, gradient
                gc.collect()
                torch.cuda.empty_cache()
        
        image = latent2image(model.vae, latents.detach())
        perturbed = image.astype(np.float32) / 255 
        image = (perturbed * 255).astype(np.uint8)

        with torch.enable_grad():
            latents_n = latents.detach().to(model.device).requires_grad_(True)
            latents_n = latents_n / model.vae.config.scaling_factor

            image = model.vae.decode(latents_n, return_dict=False)[0]
            image = image.to(latents_n.device)
            image.requires_grad_(True)
            image = model.image_processor.postprocess(image, output_type='pt')
            image = process_latent(image).cpu()

            image_features = vic_model.image_encoder(image)
            image_features = F.normalize(image_features, dim=1)

            tau = 0.07
            sim = (image_features @ location_features.T) / tau

            gradient = torch.autograd.grad(sim, latents_n)[0]

        pri_latents = (pri_latents + a * gradient.float())
       
        del gradient, latents_n, image
        gc.collect()
        torch.cuda.empty_cache()

    image = latent2image(model.vae, latents.detach())
    perturbed = image.astype(np.float32) / 255 
    image = (perturbed * 255).astype(np.uint8)
    if save_path is not None:
        save_path = save_path + prompt[0] + "_protect.png"
        to_pil_image(image[0]).save(save_path)

    del perturbed
    gc.collect()
    torch.cuda.empty_cache()

    return image[0]