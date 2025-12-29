#!/bin/bash  

STYLE_LORA="lora/style"
PROMPT="A cat in <s> style."
OUTPUT_DIR="result_style_customization"
echo $STYLE_LORA
echo $PROMPT
echo $OUTPUT_DIR

# 启动推理
python inference_style_customization.py \
    --prompt="$PROMPT" \
    --output_path="$OUTPUT_DIR" \
    --style_LoRA="$STYLE_LORA/checkpoint-1000" \
