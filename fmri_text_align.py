import argparse
import logging
import inspect
import math
import os

from typing import Dict, Optional
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import diffusers
import transformers

from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock

from transformers.models.clip.modeling_clip import clip_loss

from models.utils import *

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
    out_dir = os.path.join(output_dir, f"{config['kwargs']['NAME']}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

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

def main(
    output_dir: str,
    Model: Dict,
    Data: Dict,
    loss_dictname: list,
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
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
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
    fmri_encoder = fMRI_encoder(Model)
    clip_model = CLIPModel_txt_img()

    # Freeze any necessary models
    freeze_models([clip_model])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, fmri_encoder)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    params = []

    for name, param in fmri_encoder.named_parameters():
        params.append(
            {
                "name": name, 
                "params": param, 
                "lr": (Model['lr_encoder'] if 'fmri_encoder.transformer.' in name else 
                       ((10 if 'logit_scale' in name else 1) * Model['lr_cond_stage'])
                    )
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
        stage='fmri-text',
        subjects=Data['subjects'],
        data_aug=Data['data_aug'],
    )
    test_dataset = fd_fmri_video_dataset(
        path=Data['path'],
        video_size=Data['video_size'],
        fmri_size=Data['fmri_size'],
        window_size=Model['window_size'],
        phase='test',
        stage='fmri-text',
        subjects=Data['subjects'],
        data_aug=False,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Prepare everything with our `accelerator`.
    optimizer,train_dataloader, lr_scheduler, fmri_encoder, clip_model = accelerator.prepare(
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
        fmri_encoder,
        clip_model,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [clip_model]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("fmri2text-fine-tune")

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

    def fmri_text_align(batch):
        if global_step == 0:
            clip_model.eval()
            clip_model.requires_grad_(False)
            fmri_encoder.train()
            fmri_encoder.requires_grad_(True)
            cast_to_gpu_and_type([fmri_encoder], accelerator, torch.float32)

        if kwargs.get('eval_train', False):
            clip_model.eval()
            fmri_encoder.eval()

        with torch.no_grad():
            text_latent, image_latent = clip_model(batch['text_prompt'], batch['pixel_values'].to('cuda', dtype=torch.float16))
            sim3 = image_latent.float().detach() @ text_latent.float().detach().t() * 100
            image_text_loss = clip_loss(sim3)

        batch['image'] = batch['image'].to('cuda')
        outputs = []
        sim1_list = []
        sim2_list = []
        for subid in range(batch['image'].shape[1]):
            outputs.append(fmri_encoder(batch['image'][:,subid])) # i th subject
            sim1_list.append(outputs[subid]['fmri_latent'] @ image_latent.float().t() * fmri_encoder.logit_scale.exp())
            sim2_list.append(outputs[subid]['fmri_latent'] @ text_latent.float().t()  * fmri_encoder.logit_scale.exp())

        image_loss = 0
        for sim1 in sim1_list:
            image_loss = image_loss + clip_loss(sim1) / len(sim1_list)
        text_loss  = 0
        for sim2 in sim2_list:
            text_loss = text_loss + clip_loss(sim2) / len(sim2_list)

        loss = {
            'clip_image_loss' : image_loss,
            'clip_text_loss' : text_loss,
            'image_text_loss' : image_text_loss,
        }
        loss_item = {}
        output_log = f'step {global_step} -- '
        for key, value in loss.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            loss_item[key] = value
            output_log += f' {key} : {value};'
        print(output_log)

        return loss, loss_item

    for epoch in range(first_epoch, num_train_epochs):
        print(f'fmri logit_scale : {fmri_encoder.logit_scale.exp().item()}')

        train_loss = 0.0
        train_loss_list = []
        train_clip_image_loss_list = []
        train_clip_text_loss_list = []
        train_image_text_loss_list = []

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(fmri_encoder):
                if global_step > 0 and global_step % 500 == 0:
                    test_clip_image_loss_list = []
                    test_clip_text_loss_list = []
                    test_image_text_loss_list = []
                    print('-----------------------------------------')
                    print('----------------test start---------------')
                    print('-----------------------------------------')
                    fmri_encoder.eval()
                    for test_step, test_batch in enumerate(test_dataloader):
                        with torch.no_grad():
                            with accelerator.autocast():
                                loss_dict, loss_dict_item = fmri_text_align(test_batch)
                                test_clip_image_loss_list.append(loss_dict_item['clip_image_loss'])
                                test_clip_text_loss_list.append(loss_dict_item['clip_text_loss'])
                                test_image_text_loss_list.append(loss_dict_item['image_text_loss'])
                    fmri_encoder.train()
                    print('-----------------------------------------')
                    print(f'test set clip image loss  : {np.mean(test_clip_image_loss_list)}')
                    print(f'test set clip text loss  : {np.mean(test_clip_text_loss_list)}')
                    print(f'test set image text loss  : {np.mean(test_image_text_loss_list)}')
                    print('-----------------------------------------')

                with accelerator.autocast():
                    loss_dict, loss_dict_item = fmri_text_align(batch)
                    train_clip_image_loss_list.append(loss_dict_item['clip_image_loss'])
                    train_clip_text_loss_list.append(loss_dict_item['clip_text_loss'])
                    train_image_text_loss_list.append(loss_dict_item['image_text_loss'])

                    loss = 0
                    for dict_name in loss_dictname:
                        loss = loss + (0.5 if 'clip' in dict_name else 1) * loss_dict[dict_name]
                    train_loss_list.append(loss.item())
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                try:
                    accelerator.backward(loss)

                    if max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            params_to_clip = list(fmri_encoder.parameters())
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
                    save_model(fmri_encoder, opt=optimizer,
                        path=os.path.join(output_dir, f'fmri-encoder-checkpoint'), epoch=epoch, iter=global_step, prefix=str(global_step))

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
        print(f'epoch {epoch} average loss {np.mean(train_loss_list)}')
        print(f'epoch {epoch} average clip image loss {np.mean(train_clip_image_loss_list)}')
        print(f'epoch {epoch} average clip text loss {np.mean(train_clip_text_loss_list)}')
        print(f'epoch {epoch} average image text loss {np.mean(train_image_text_loss_list)}')

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/v2/fmri-text-sub12.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))

# CUDA_VISIBLE_DEVICES=0 python -u fmri_text_align.py --config ./configs/v2/fmri-text-sub12.yaml > fmri-text-sub12.out
