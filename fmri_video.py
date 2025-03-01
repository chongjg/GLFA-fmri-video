import argparse
import logging
import inspect
import math
import os
import gc
import copy

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers

from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from einops import rearrange

from models.unet_3d_condition import UNet3DConditionModel
from models.lora_handler import LoraHandler, LORA_VERSIONS
from models.utils import *
from models.eval_metrics import (ssim_score_only, 
                          img_classify_metric, 
                          video_classify_metric)
import imageio.v3 as iio

already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def create_output_folders(output_dir, config):
    # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"{config['kwargs']['NAME']}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDPMScheduler.from_config(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }
    
def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list): 
            params = create_optim_params(
                params=itertools.chain(*model), 
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue
            
        if is_lora and  condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)
    
    return optimizer_params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)

def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    global already_printed_trainables
    acc = []
    unfrozen_params = 0
    
    if trainable_modules is not None:
        unlock_all = any([name == 'all' for name in trainable_modules])
        if unlock_all:
            model.requires_grad_(True)
            unfrozen_params = len(list(model.parameters()))
        else:
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                for tm in trainable_modules:
                    if all([tm in name, name not in acc, 'lora' not in name]):
                        param.requires_grad_(is_enabled)
                        acc.append(name)
                        unfrozen_params += 1
                        
    if unfrozen_params > 0 and not already_printed_trainables:
        already_printed_trainables = True 
        print(f"{unfrozen_params} params have been processed.")

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def should_sample(global_step, validation_steps, validation_data):
    return (global_step % validation_steps == 0 or global_step == 1)  \
    and validation_data.sample_preview

def save_pipe(
        path, 
        global_step,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        output_dir,
        lora_manager: LoraHandler,
        unet_target_replace_module=None,
        text_target_replace_module=None,
        is_checkpoint=False,
        save_pretrained_model=True
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

    # Save the dtypes so we can continue training at the same precision.
    u_dtype, t_dtype, v_dtype = unet.dtype, text_encoder.dtype, vae.dtype 

   # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_save = copy.deepcopy(unet.cpu())
    text_encoder_save = copy.deepcopy(text_encoder.cpu())

    unet_out = copy.deepcopy(accelerator.unwrap_model(unet_save, keep_fp32_wrapper=False))
    text_encoder_out = copy.deepcopy(accelerator.unwrap_model(text_encoder_save, keep_fp32_wrapper=False))

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder_out,
        vae=vae,
    ).to(torch_dtype=torch.float32)
    
    lora_manager.save_lora_weights(model=pipeline, save_path=save_path, step=global_step)

    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    if is_checkpoint:
        unet, text_encoder = accelerator.prepare(unet, text_encoder)
        models_to_cast_back = [(unet, u_dtype), (text_encoder, t_dtype), (vae, v_dtype)]
        [x[0].to(accelerator.device, dtype=x[1]) for x in models_to_cast_back]

    logger.info(f"Saved model at {save_path} on step {global_step}")
    
    del pipeline
    del unet_out
    del text_encoder_out
    torch.cuda.empty_cache()
    gc.collect()


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 

def run_metrics(output_dir, device):
    metrics = {}
    gt_list = []
    pred_list = []
    print('loading validation results ...')
    for i in range(400):
        gif = iio.imread(os.path.join(output_dir, f'fmri-dataset-test-{i}.mp4'), index=None)
        pred, gt = np.split(gif, 2, axis=2)
        pred = pred[:,2:-2,2:-1]
        gt = gt[:,2:-2,1:-2]
        gt_list.append(gt)
        pred_list.append(pred)
    print('validation results loaded.')

    gt_list = np.stack(gt_list)
    pred_list = np.stack(pred_list)

    print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

    # image classification scores
    n_way = [2,50]
    num_trials = 100
    top_k = 1
    # video classification scores
    acc_list, std_list = video_classify_metric(
                                        pred_list,
                                        gt_list,
                                        n_way = n_way,
                                        top_k=top_k,
                                        num_trials=num_trials,
                                        num_frames=gt_list.shape[1],
                                        return_std=True,
                                        device=device
                                        )
    for i, nway in enumerate(n_way):
        print(f'video classification score ({nway}-way): {np.mean(acc_list[i])} +- {np.mean(std_list[i])}')
        metrics[f'video {nway}-way mean'] = np.mean(acc_list[i])
        metrics[f'video {nway}-way std'] = np.mean(np.mean(std_list[i]))

    acc_aver = [[] for i in range(len(n_way))]
    acc_std  = [[] for i in range(len(n_way))]
    ssim_aver = []
    ssim_std  = []
    for i in range(pred_list.shape[1]):

        # ssim scores
        ssim_scores, std = ssim_score_only(pred_list[:, i], gt_list[:, i])
        ssim_aver.append(ssim_scores)
        ssim_std.append(std)

        print(f'ssim score: {ssim_scores}, std: {std}')
        
        acc_list, std_list = img_classify_metric(
                                            pred_list[:, i], 
                                            gt_list[:, i], 
                                            n_way = n_way, 
                                            top_k=top_k, 
                                            num_trials=num_trials, 
                                            return_std=True,
                                            device=device)
        for idx, nway in enumerate(n_way):
            acc_aver[idx].append(np.mean(acc_list[idx]))
            acc_std[idx].append(np.mean(std_list[idx]))
            print(f'img classification score ({nway}-way): {np.mean(acc_list[idx])} +- {np.mean(std_list[idx])}')

    print('----------------- average results -----------------')
    print(f'average ssim score: {np.mean(ssim_aver)}, std: {np.mean(ssim_std)}')
    metrics[f'image ssim mean'] = np.mean(ssim_aver)
    metrics[f'image ssim std'] = np.mean(ssim_std)
    for i, nway in enumerate(n_way):
        print(f'average img classification score ({nway}-way): {np.mean(acc_aver[i])} +- {np.mean(acc_std[i])}')
        metrics[f'image {nway}-way mean'] = np.mean(acc_aver[i])
        metrics[f'image {nway}-way std'] = np.mean(acc_std[i])
    return metrics

@torch.inference_mode()
def validation_inference(
    output_dir,
    validation_data,
    Model,
    pipeline,
    fmri_encoder: fMRI_encoder,
    test_dataset: torch.utils.data.Dataset,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    seed: int=666,
):

    output_dir = output_dir
    width=validation_data.width
    height=validation_data.height
    num_frames=validation_data.num_frames
    num_inference_steps=validation_data.num_inference_steps
    guidance_scale=validation_data.guidance_scale

    if seed is not None:
        torch.manual_seed(seed)

    window_size = Model['window_size']
    with torch.autocast(device, dtype=torch.float):

        with torch.no_grad():
            fmri_negative_embedding = fmri_encoder(test_dataset.aver_fmri[:window_size][None].to('cuda'))['fmri_embedding']
            torch.cuda.empty_cache()

        for global_step, batch in enumerate(test_dataloader):
            if global_step % 100 == 0:
                print(f'---------------- step {global_step} / {len(test_dataloader)} ---------------- ')

            torch.cuda.empty_cache()

            prompt = batch['text_prompt'][0]
            if global_step == 0:
                with torch.no_grad():
                    fmri_embedding = fmri_encoder(batch['image'][:,0].to('cuda'))['fmri_embedding']
                    negative_embedding = fmri_negative_embedding.repeat(fmri_embedding.shape[0],1,1)

            with torch.no_grad():
                video_frames = pipeline(
                    fmri_embedding,
                    negative_embedding=negative_embedding,#text_negative_embedding,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_type='pt',
                ).frames

                video_frames = video_frames.clamp(-1, 1).add(1).div(2).cpu()
                pixel_values = (batch['pixel_values'].transpose(1, 2).cpu() + 1.) / 2.

                for frame_id in range(video_frames.shape[0]):
                    out_file = f"{output_dir}/fmri-dataset-test-{global_step+frame_id}.mp4"
                    video_frame = video_frames[frame_id:frame_id+1]
                    pixel_value = pixel_values[frame_id:frame_id+1]
                    
                    save_videos_grid(torch.cat([video_frame, pixel_value]), out_file, fps=num_frames//4)
    
    return run_metrics(output_dir, device)

def main(
    pretrained_model_path: str,
    output_dir: str,
    Model: Dict,
    Data: Dict,
    validation_data: Dict,
    shuffle: bool = True,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = None, # Eg: ("attn1", "attn2")
    extra_unet_params = None,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    train_text_encoder: bool = False,
    use_offset_noise: bool = False,
    rescale_schedule: bool = False,
    offset_noise_strength: float = 0.1,
    # extend_dataset: bool = False,
    cache_latents: bool = False,
    lora_version: LORA_VERSIONS = LORA_VERSIONS[0],
    save_lora_for_webui: bool = False,
    only_lora_for_webui: bool = False,
    lora_bias: str = 'none',
    use_unet_lora: bool = False,
    use_text_lora: bool = False,
    unet_lora_modules: Tuple[str] = ["ResnetBlock2D"],
    text_encoder_lora_modules: Tuple[str] = ["CLIPEncoderLayer"],
    save_pretrained_model: bool = True,
    lora_rank: int = 16,
    lora_path: str = '',
    lora_unet_dropout: float = 0.1,
    logger_type: str = 'tensorboard',
    **kwargs
):

    train_batch_size = train_batch_size // len(Data['subjects'])

    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
       output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path)
    fmri_encoder = fMRI_encoder(Model, func_kernel_size=Model['func_kernel_size'], global_align=Model['global_align'])

    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    # Use LoRA if enabled.  
    lora_manager = LoraHandler(
        version=lora_version, 
        use_unet_lora=use_unet_lora,
        use_text_lora=use_text_lora,
        save_for_webui=save_lora_for_webui,
        only_for_webui=only_lora_for_webui,
        unet_replace_modules=unet_lora_modules,
        text_encoder_replace_modules=text_encoder_lora_modules,
        lora_bias=lora_bias
    )

    unet_lora_params, unet_negation = lora_manager.add_lora_to_model(
        use_unet_lora, unet, lora_manager.unet_replace_modules, lora_unet_dropout, lora_path, r=lora_rank) 
    if use_unet_lora:
        print(unet)

    # Create parameters to optimize over with a condition (if "condition" is true, optimize it)
    extra_unet_params = extra_unet_params if extra_unet_params is not None else {}

    trainable_modules_available = trainable_modules is not None
    
    optim_params = [
        param_optim(unet, trainable_modules_available, extra_params=extra_unet_params, negation=unet_negation),
        param_optim(unet_lora_params, use_unet_lora, is_lora=True,
                        extra_params={**{"lr": learning_rate}, **extra_unet_params}
                    )
    ]

    params = create_optimizer_params(optim_params, learning_rate)

    for name, param in fmri_encoder.named_parameters():
        params.append(
            {
                "name": name,
                "params": param,
                "lr": (Model['lr_encoder'] if 'fmri_encoder.transformer.' in name else (Model['lr_cond_stage'])),
            }
        )
    
    # Create Optimizer
    optimizer = optimizer_cls(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    train_dataset = fd_fmri_video_dataset(
        path=Data['path'],
        video_size=Data['video_size'],
        fmri_size=Data['fmri_size'],
        window_size=Model['window_size'],
        phase='train',
        stage='fmri-video',
        subjects=Data['subjects'],
    )
    test_dataset = fd_fmri_video_dataset(
        path=Data['path'],
        video_size=Data['video_size'],
        fmri_size=Data['fmri_size'],
        window_size=Model['window_size'],
        phase='test',
        stage='fmri-video',
        subjects=Data['test_subjects'],
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=6,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer,train_dataloader, lr_scheduler, text_encoder, fmri_encoder = accelerator.prepare(
        unet, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
        text_encoder,
        fmri_encoder,
    )

    # Use Gradient Checkpointing if enabled.
    unet_and_text_g_c(
        unet, 
        text_encoder, 
        gradient_checkpointing, 
        text_encoder_gradient_checkpointing
    )
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [vae, fmri_encoder]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # Fix noise schedules to predcit light and dark areas if available.
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("fmri2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    fmri_mask = (train_dataset.aver_fmri.mean(0) != 0).flatten()
    def finetune_unet(batch, train_encoder=False):
        nonlocal use_offset_noise
        nonlocal rescale_schedule

        BN = batch['image'].shape[0]
        num_sbjs = batch['image'].shape[1]
        
        # Check if we are training the text encoder
        unet.train()
        fmri_encoder.train()
        
        # Unfreeze UNET Layers
        if global_step == 0: 
            handle_trainable_modules(
                unet, 
                trainable_modules, 
                is_enabled=True,
                negation=unet_negation
            )
            text_encoder.eval()
            text_encoder.requires_grad_(False)
            fmri_encoder.requires_grad_(True)
            cast_to_gpu_and_type([fmri_encoder], accelerator, torch.float32)

        # Convert videos to latent space
        pixel_values = batch["pixel_values"]

        if not cache_latents:
            latents = tensor_to_vae_latent(pixel_values.to('cuda', dtype=torch.float16), vae)
        else:
            latents = pixel_values
        latents = latents.repeat(num_sbjs,1,1,1,1)

        # Sample noise that we'll add to the latents
        use_offset_noise = use_offset_noise and not rescale_schedule
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        # Sample a random timestep for each video
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
        # *Potentially* Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if kwargs.get('eval_train', False):
            unet.eval()
            text_encoder.eval()
            fmri_encoder.eval()
            
        batch['image'] = batch['image'].to('cuda')
        outputs = []
        reg_loss = 0.
        align_loss = 0.
        align_embedding_loss = torch.tensor(0.)
        align_cnt = 0
        for subid in range(num_sbjs):
            outputs.append(fmri_encoder(batch['image'][:,subid]))
            if Model['global_align']:
                reg_loss = reg_loss + F.mse_loss(outputs[subid]['func_align_fmri'].reshape(BN*Model['window_size'], -1)[:, fmri_mask], batch['image'][:,subid].reshape(BN*Model['window_size'], -1)[:, fmri_mask], reduction="mean")
                for pre_sub in range(subid):
                    align_cnt += 1
                    align_loss = align_loss + F.mse_loss(outputs[subid]['func_align_fmri'].reshape(BN*Model['window_size'], -1)[:, fmri_mask], outputs[pre_sub]['func_align_fmri'].reshape(BN*Model['window_size'], -1)[:, fmri_mask], reduction="mean")
                    # align_embedding_loss = align_embedding_loss + F.mse_loss(outputs[subid]['fmri_embedding'], outputs[pre_sub]['fmri_embedding'], reduction="mean")
        if Model['global_align']:
            reg_loss = reg_loss / num_sbjs
            align_loss = align_loss / align_cnt
            # align_embedding_loss = align_embedding_loss / align_cnt

        encoder_hidden_states = torch.cat([item['fmri_embedding'] for item in outputs], 0)

        # Get the target for loss depending on the prediction type
        if noise_scheduler.prediction_type == "epsilon":
            target = noise

        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if Model['global_align']:
            print(f'step {global_step} -- training loss : {loss.item()}' + f'; align_loss : {align_loss.item()}; align_embedding_loss : {align_embedding_loss.item()}')
        else:
            print(f'step {global_step} -- training loss : {loss.item()}')# + (f'; align_loss : {align_loss.item()}; regloss : {regloss.item()}; align_embedding_loss : {align_embedding_loss.item()}' if align_loss_enable else ''))

        loss = loss + Model['reg_scale'] * reg_loss + Model['align_scale'] * align_loss + Model['align_embedding_scale'] * align_embedding_loss

        return loss, latents

    torch.cuda.empty_cache()

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        loss_list = []
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet, fmri_encoder):

                text_prompt = batch['text_prompt'][0]
                
                with accelerator.autocast():
                    loss, latents = finetune_unet(batch, train_encoder=train_text_encoder)
                    loss_list.append(loss.item())
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                try:
                    accelerator.backward(loss)

                    if max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            params_to_clip = list(unet.parameters()) + list(fmri_encoder.parameters())
                            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                            
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}") 
                    continue

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
            
                if global_step % checkpointing_steps == 0:
                    save_pipe(
                        pretrained_model_path, 
                        global_step, 
                        accelerator, 
                        unet, 
                        text_encoder, 
                        vae, 
                        output_dir, 
                        lora_manager,
                        unet_lora_modules,
                        text_encoder_lora_modules,
                        is_checkpoint=True,
                        save_pretrained_model=save_pretrained_model
                    )
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    save_model(fmri_encoder, opt=optimizer,
                        path=os.path.join(save_path, f'fmri-encoder-checkpoint'), epoch=epoch, iter=global_step)

                if should_sample(global_step, validation_steps, validation_data):
                    if global_step == 1: print("Performing validation prompt.")
                    if accelerator.is_main_process:
                        
                        with accelerator.autocast():
                            unet.eval()
                            text_encoder.eval()
                            fmri_encoder.eval()
                            unet_and_text_g_c(unet, text_encoder, False, False)
                            lora_manager.deactivate_lora_train([unet, text_encoder], True)    

                            pipeline = fMRIToVideoSDPipeline.from_pretrained(
                                pretrained_model_path,
                                text_encoder=text_encoder,
                                vae=vae,
                                unet=unet
                            )

                            diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                            pipeline.scheduler = diffusion_scheduler

                            validation_output_dir = os.path.join(output_dir, f"sample-validation/{global_step}")
                            metrics = validation_inference(
                                output_dir=validation_output_dir,
                                validation_data=validation_data,
                                Model=Model,
                                pipeline=pipeline,
                                fmri_encoder=fmri_encoder,
                                test_dataset=test_dataset,
                                test_dataloader=test_dataloader,
                            )
                            
                            prompt = text_prompt# if len(validation_data.prompt) <= 0 else validation_data.prompt
                            with torch.no_grad():
                                aver_fmri = train_dataset.aver_fmri[:Model['window_size']][None].to('cuda')

                                fmri_negative_embedding = fmri_encoder(aver_fmri)['fmri_embedding']
                                outputs = fmri_encoder(batch['image'].flatten(0,1).to('cuda'))
                                if Model['global_align']:
                                    func_align_fmri = outputs['func_align_fmri'].reshape(batch['image'].shape[0],batch['image'].shape[1],Model['window_size'],256,256).detach().cpu().numpy()

                                    # for batch_id in range(batch['image'].shape[0]):
                                    for batch_id in range(1):
                                        for subid in range(batch['image'].shape[1]):
                                            # for window_id in range(Model['window_size']):
                                            for window_id in range(1):
                                                for ori in range(2):
                                                    if ori == 1:
                                                        image = batch['image'][batch_id,subid,window_id,0].cpu().numpy()
                                                    else:
                                                        image = func_align_fmri[batch_id,subid,window_id]
                                                    mask = image == 0
                                                    norm_img = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255 # norm to [0, 255]
                                                    norm_img = cv2.applyColorMap(norm_img.astype(np.uint8), cv2.COLORMAP_JET)
                                                    norm_img[mask] = 0
                                                    os.makedirs(f'{output_dir}/samples/fmri/', exist_ok=True)
                                                    cv2.imwrite(f"{output_dir}/samples/fmri/step-{global_step}_batch-{batch_id}_{window_id}_sub-{subid}-{'ori' if ori==1 else 'pred'}.jpg", norm_img)

                                fmri_embedding = outputs['fmri_embedding'][:1]

                            curr_dataset_name = 'train'
                            save_filename = f"{global_step}_dataset-{curr_dataset_name}_{prompt[:50]}"
                            
                            with torch.no_grad():
                                out_file = f"{output_dir}/samples/fmri-{save_filename}.mp4"
                                video_frames = pipeline(
                                    fmri_embedding,
                                    negative_embedding=fmri_negative_embedding,
                                    prompt=prompt,
                                    width=validation_data.width,
                                    height=validation_data.height,
                                    num_frames=validation_data.num_frames,
                                    num_inference_steps=validation_data.num_inference_steps,
                                    guidance_scale=validation_data.guidance_scale,
                                    output_type='pt',
                                ).frames

                                video_frames = video_frames[0][None].clamp(-1, 1).add(1).div(2).cpu()
                                pixel_values = (batch['pixel_values'][0:1].transpose(1, 2).cpu() + 1.) / 2.
                                video_frames = torch.cat([video_frames, pixel_values])
                                
                                save_videos_grid(video_frames, out_file, fps=3)

                            del pipeline
                            torch.cuda.empty_cache()

                        logger.info(f"Saved a new sample to {out_file}")

                    unet_and_text_g_c(
                        unet, 
                        text_encoder, 
                        gradient_checkpointing, 
                        text_encoder_gradient_checkpointing
                    )

                    lora_manager.deactivate_lora_train([unet, text_encoder], False)    
                    torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
        print(f'epoch average loss : {np.mean(loss_list)}')

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
                pretrained_model_path, 
                global_step, 
                accelerator, 
                unet, 
                text_encoder, 
                vae, 
                output_dir, 
                lora_manager,
                unet_lora_modules,
                text_encoder_lora_modules,
                is_checkpoint=False,
                save_pretrained_model=save_pretrained_model
        )     
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/v2/0308-fmri-video-sub12-funcalign.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))

# CUDA_VISIBLE_DEVICES=5 python -u fmri_video.py --config ./configs/v2/fmri-video-sub12-funcalign.yaml > fmri-video-sub12-funcalign.out
