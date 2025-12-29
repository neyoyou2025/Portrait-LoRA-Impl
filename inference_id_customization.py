import argparse
from functools import partial
import gc
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from utils import BLOCKS, filter_lora, scale_lora
from src.ram import EmbeddingManager
from src.sdxl_text_encoder_1 import FrozenCLIPEmbedder
from src.face_encoder import FaceEncoder
from src.sdxl_text_encoder_2 import FrozenCLIPEmbedderWithProjection
from diffusers import StableDiffusionXLPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, default="", help="fuse prompt"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the images"
    )
    parser.add_argument(
        "--content_LoRA", type=str, default=None, help="path for the content LoRA"
    )
    parser.add_argument(
        "--content_alpha", type=float, default=1., help="alpha parameter to scale the content LoRA weights"
    )
    parser.add_argument(
        "--face_alpha", type=float, default=1., help="alpha parameter to scale the ram face weights"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=4, help="number of images per prompt"
    )
    parser.add_argument(
        "--face_img", type=str, default=None, help="face img"
    )
    parser.add_argument(
        "--embedding_manager_ckpt", type=str, default=None, help="path for embedding manager"
    )
    parser.add_argument(
        "--text_encoder_ckpt", type=str, default=None, help="path for sdxl text encoder 1"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = StableDiffusionXLPipeline.from_pretrained("/home/asus/data/init_ckpt/stable-diffusion-xl-base-1-0",
                                                         vae=vae,
                                                         torch_dtype=torch.float16).to("cuda")
    text_encoder = FrozenCLIPEmbedder(
            version="/home/asus/data/init_ckpt/stable-diffusion-xl-base-1-0",
            device="cuda",
            revision=False,
            tokenizer=pipeline.tokenizer,
            instant_prompt="*"
        )
    text_encoder.text_encoder.load_state_dict(torch.load(args.text_encoder_ckpt)) 
    text_encoder.to(device="cuda", dtype=torch.float16)
    text_encoder_two = FrozenCLIPEmbedderWithProjection(
        version="/home/asus/data/init_ckpt/stable-diffusion-xl-base-1-0",
        device="cuda",
        revision=False,
        tokenizer=pipeline.tokenizer_2,
        instant_prompt=args.prompt
    )
    text_encoder_two.to(device="cuda", dtype=torch.float16)
    face_encoder = FaceEncoder(version="/home/asus/data/init_ckpt/clip-vit-large-patch14", device="cuda", weight_dtype=torch.float16)
    id_embeddings, image_embeds = face_encoder.get_id_embedding(args.face_img)
    del face_encoder
    gc.collect()
    torch.cuda.empty_cache()
    if(args.embedding_manager_ckpt is not None):
        embedding_manager = EmbeddingManager(pipeline.tokenizer, torch.float16, 
                                         placeholder_strings=["*"], dim=768, weight=args.face_alpha)
        embedding_manager.load(args.embedding_manager_ckpt)
        embedding_manager.to(device="cuda", dtype=torch.float16)
    
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids
    
    one_text_input_ids = tokenize_prompt(pipeline.tokenizer, args.prompt + " The facial features are * ")
    pipeline.text_encoder = text_encoder
    pipeline.text_encoder.forward = partial(pipeline.text_encoder.forward, input_idss=one_text_input_ids.to(device=text_encoder.device), id_embedding=id_embeddings[0], image_embedding=image_embeds[0], embedding_manager=embedding_manager)
    pipeline.text_encoder_2 = text_encoder_two.text_encoder
    pipeline.text_encoder_2.forward = text_encoder_two.text_encoder.forward
    pipeline.text_encoder_2.forward = partial(pipeline.text_encoder_2.forward)
    # Get Content LoRA SD
    if args.content_LoRA is not None:
        content_LoRA_sd, _ = pipeline.lora_state_dict(args.content_LoRA)
        content_LoRA = filter_lora(content_LoRA_sd, BLOCKS['content'])
        content_LoRA = scale_lora(content_LoRA, args.content_alpha)
    else:
        content_LoRA = {}

    # Merge LoRAs
    res_lora = {**content_LoRA}
    # Load
    pipeline.load_lora_into_unet(res_lora, None, pipeline.unet)
    # Generate
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

    import os
    os.makedirs(args.output_path, exist_ok=True)
    # Save
    for i, img in enumerate(images):
        img.save(f'{args.output_path}/{i}.jpg')
