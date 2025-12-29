# Portrait-LoRA

This is the official PyTorch code for our paper.

<!-- ![Framework](./main.png) -->

#### 0. Preparation

The folder contains our complete source code, and we have provided a set of sample images for testing.

Create a new conda environment and install dependencies：
```
conda create -n p_lora python=3.8
conda activate p_lora
pip install -r requirements.txt  
```

#### 1. Train

##### 1.1 Data Preparation

Our method is implemented through one-shot fine-tuning. This file provides an example: the image in the `./example/style` directory is style reference image used for extracting style feature, while the `./example/content` directory is portrait reference image used for extracting content feature.

##### 1.2 Model Preparation

The baseline of our method is implemented on top of [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14). Our code provides an automatic download feature. However, if any issues occur during the download, you can also click the download link to download the files manually.

##### 1.3 Style Feature Extraction
Parameters can be edited in `./train_style_disentanglement.sh`, such as batch size (`train_batch_size`), learning rate (`learning_rate`), content-enhanced prompt (`instance_prompt`) and so on, executing this file to train the style LoRA.

```
bash train_style_disentanglement.sh
```

##### 1.4 Content Feature Extraction
Similarly, you can modify the parameters in `./train_content_disentanglement.sh` and train content LoRA and RAM by executing that file.

```
bash train_content_disentanglement.sh
```

#### 2. Test
##### 2.1 Text-driven Style Transfer(Style Customization)
Execute the following commands to verify the results of style customization.

```
bash inference_style_customization.sh
```

##### 2.2 Face Edit(ID Customization)
Execute the following commands to verify the results of ID customization.

```
bash inference_id_customization.sh
```

##### 2.3 Portrait style transfer
Execute the following commands to verify the results of portrait style transfer.

```
bash inference_fuse_style_content.sh
```