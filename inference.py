import argparse
import os
import warnings

from diffusers import DPMSolverMultistepScheduler, UNet3DConditionModel

from fmri_video import handle_memory_attention, load_primary_models
from models.lora import inject_inferable_lora

from models.utils import *

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet  # This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder="unet")

    pipe = fMRIToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet.disable_gradient_checkpointing()
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()

    inject_inferable_lora(pipe, lora_path, r=lora_rank)

    return pipe

@torch.inference_mode()
def inference(
    config,
    model: str,
    fmri_encoder: fMRI_encoder,
    test_dataset: torch.utils.data.Dataset,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    seed: int=666,
    curr_dataset_name='test',
):

    output_dir = config.output_dir
    width = config.inference_cfg['width']
    height = config.inference_cfg['height']
    num_frames = config.inference_cfg['num_frames']
    num_inference_steps = config.inference_cfg['num_inference_steps']
    guidance_scale = config.inference_cfg['guidance_scale']

    if seed is not None:
        torch.manual_seed(seed)

    window_size = config.Model['window_size']
    with torch.autocast(device, dtype=torch.half):
        pipeline = initialize_pipeline(model, device, xformers, sdp, lora_path=model+'lora', lora_rank=4)

        with torch.no_grad():
            fmri_negative_embedding = fmri_encoder(test_dataset.aver_fmri[:window_size][None].to('cuda'))['fmri_embedding']
            torch.cuda.empty_cache()

        for global_step, batch in enumerate(test_dataloader):
            print(f'---------------- step {global_step} / {len(test_dataloader)} ---------------- ')

            torch.cuda.empty_cache()

            prompt = batch['text_prompt'][0]
            print(prompt)
            with torch.no_grad():
                fmri_embedding = fmri_encoder(batch['image'][:,0].to('cuda'))['fmri_embedding']
                negative_embedding = fmri_negative_embedding.repeat(fmri_embedding.shape[0],1,1)

            curr_dataset_name = curr_dataset_name #batch['dataset']
            
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
                    out_file = f"{output_dir}/samples/fmri-dataset-{curr_dataset_name}-{global_step*config.test_batch_size+frame_id}.mp4"
                    video_frame = video_frames[frame_id:frame_id+1]
                    pixel_value = pixel_values[frame_id:frame_id+1]
                    
                    save_videos_grid(torch.cat([video_frame, pixel_value]), out_file, fps=config.inference_cfg['num_frames']//4)
                    
if __name__ == "__main__":

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_file", type=str, default='./configs/v2/inference_config-sub12-30k.yaml', help="path to config file")
    parser.add_argument("-p", "--phase", type=str, default="test", help="phase : test set or training set")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    args = parser.parse_args()
    # fmt: on
    config = Config(args.config_file)

    os.makedirs(config.output_dir, exist_ok=True)

    fmri_encoder = fMRI_encoder(config.Model, func_kernel_size=config.Model['func_kernel_size'], global_align=config.Model['global_align']).cuda()
    fmri_encoder.eval()

    curr_dataset_name = args.phase

    test_dataset = fd_fmri_video_dataset(
        path=config.Data['path'],
        video_size=config.Data['video_size'],
        fmri_size=config.Data['fmri_size'],
        window_size=config.Model['window_size'],
        phase=curr_dataset_name,
        subjects=config.Data['subjects'],
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    inference(
        config,
        model=config.pretrained_model_path,
        fmri_encoder=fmri_encoder,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        curr_dataset_name=curr_dataset_name,
    )

# CUDA_VISIBLE_DEVICES=0 python -u inference.py --config_file ./configs/v2/inference_config-sub12-funcalign-22k.yaml

# CUDA_VISIBLE_DEVICES=3 python -u run_metrics.py ./outputs-inference/fmri-video-sub12-funcalign-16k/ > ./outputs-inference/fmri-video-sub12-funcalign-16k/test.out