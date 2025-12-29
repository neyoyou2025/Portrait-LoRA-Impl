#!/bin/bash  

CONTENT_LORA="lora/content"
PORTRAIT_IMG="example/content"
PROMPT="A woman with long, flowing blonde hair, wearing a dark sweater and glasses in photo style."
OUTPUT_DIR="result_id_customization"
echo $CONTENT_LORA
echo $PORTRAIT_IMG
echo $PROMPT
echo $OUTPUT_DIR

# 启动推理
python inference_id_customization.py \
    --prompt="$PROMPT" \
    --output_path="$OUTPUT_DIR" \
    --face_img="$PORTRAIT_IMG" \
    --face_alpha=1 \
    --content_LoRA="$CONTENT_LORA/checkpoint-1500" \
    --embedding_manager_ckpt="$CONTENT_LORA/embedding_manager_1500.pth" \
    --text_encoder_ckpt="$CONTENT_LORA/text_encoder_path_1500.pth"
