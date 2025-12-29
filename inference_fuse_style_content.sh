#!/bin/bash  


STYLE_LORA="lora/style/checkpoint-1000"
CONTENT_LORA="lora/content"
PORTRAIT_IMG="example/content"
PROMPT="A woman with long, flowing blonde hair, wearing a dark sweater in <s> style."
OUTPUT_DIR="result"
echo $CONTENT_LORA
echo $STYLE_LORA
echo $PORTRAIT_IMG
echo $PROMPT
echo $OUTPUT_DIR

# 启动推理
python inference_fuse_style_content.py \
    --prompt="$PROMPT" \
    --output_path="$OUTPUT_DIR" \
    --face_img="$PORTRAIT_IMG" \
    --style_alpha=1.5 \
    --face_alpha=0.5 \
    --style_LoRA="$STYLE_LORA" \
    --content_LoRA="$CONTENT_LORA/checkpoint-1500" \
    --embedding_manager_ckpt="$CONTENT_LORA/embedding_manager_1500.pth" \
    --text_encoder_ckpt="$CONTENT_LORA/text_encoder_path_1500.pth"
