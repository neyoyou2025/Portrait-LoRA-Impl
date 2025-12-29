import argparse

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from utils import BLOCKS, filter_lora, scale_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, required=True, help="LoRA prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--style_LoRA", type=str, default=None, help="path for the style LoRA"
    )
    parser.add_argument(
        "--style_alpha", type=float, default=1., help="alpha parameter to scale the style LoRA weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("/home/asus/data/init_ckpt/stable-diffusion-xl-base-1-0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")

    # Get Style LoRA SD
    if args.style_LoRA is not None:
        style_LoRA_sd, _ = pipeline.lora_state_dict(args.style_LoRA)
        style_LoRA = filter_lora(style_LoRA_sd, BLOCKS['style'])
        style_LoRA = scale_lora(style_LoRA, args.style_alpha)
    else:
        style_LoRA = {}

    # Merge LoRAs
    res_lora = {**style_LoRA}

    # Load
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)

    # Generate
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

    import os
    os.makedirs(args.output_path, exist_ok=True)
    # Save
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{i}.jpg')
