# EasyGen
The official code for paper "Making Multimodal Generation Easier: When Diffusion Models Meet LLMs"

![image](https://github.com/zxy556677/EasyGen/blob/main/asset/easygen2.png)

We present EasyGen, an efficient model designed to enhance multimodal understanding and generation by harnessing the capabilities of diffusion models and large language models (LLMs). Unlike existing multimodal models that predominately depend on encoders like CLIP or ImageBind and need ample amounts of training data to bridge the gap between modalities, EasyGen is built upon a bidirectional conditional diffusion model named BiDiffuser, which promotes more efficient interactions between modalities. EasyGen handles image-to-text generation by integrating BiDiffuser and an LLM via a simple projection layer. Unlike most existing multimodal models that are limited to generating text responses, EasyGen can also facilitate text-to-image generation by leveraging the LLM to create textual descriptions, which can be interpreted by BiDiffuser to generate appropriate visual responses. Extensive quantitative and qualitative experiments demonstrate the effectiveness of EasyGen, whose training can be easily achieved in a lab setting.

![image](https://github.com/zxy556677/EasyGen/blob/main/asset/easygen1.png)

| Model | EasyGen | InstructBLIP | BLIP2 | LLaVA | Emu |
|----------|----------|-----------|-----------|---|---|
| Training Images | 173K | 16M | 129M | 753K | 2B |
| Image-Captioning | 145.7 | 140.7 | 145.2 | 30.0 | 117.7 |

The performance is evaluated on the MS-COCO Karpathy dataset and measured by the CIDEr metric.

# Dependency

```
pip install -r requirements.txt
```

# Pretrain (feature alignment)

bash train_vicuna_7B.sh

```
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=20008 train_mem.py \
    --model_name_or_path /home/data2/xiangyu/Code/EasyGen/Tuning_for_LLaVA_only_MLP \
    --tune_mlp True \
    --freeze_backbone True \
    --freeze_mlp False \
    --data_path data/dummy_conversation.json \
    --bf16 True \
    --output_dir pretrain_only_MLP \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 150000 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --remove_unused_columns False \
```

fastchat/train/train.py

line 703: 
```
train_dataset = pre_dataset + caption_dataset
```

# Instruct-tuning

bash train_vicuna_7B.sh

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=20008 train_mem.py \
    --model_name_or_path /home/data2/xiangyu/Code/EasyGen/Tuning_for_LLaVA_only_MLP \
    --tune_mlp True \
    --freeze_backbone False \
    --freeze_mlp False \
    --data_path data/dummy_conversation.json \
    --bf16 True \
    --output_dir pretrain_only_MLP \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 150000 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
```
fastchat/train/train.py

line 703: 
```
train_dataset = qa_dataset + dialog_dataset + vqav2_dataset + train_dataset + llava_dataset
```

# Lora

We also provide the Lora method to train EasyGen. To use lora, please run
```
bash train_vicuna_7B_lora.sh
```
Also, you need to change the 10 line in train_mem.py

```
from fastchat.train.train_lora import train
```

The inference code of lora also are different, please use:

```
python -m fastchat.serve.inference_llama
```

<img src="asset/cmd_example.gif" width="70%">

# Download weights

You can download our trained models from:

https://huggingface.co/xiangyu556677/EasyGen


# Inference

By using this command, EasyGen can do image ground conversation:
```
python -m fastchat.serve.inference_llama
```
Before using this command, please download lora_weight and LLM's original weight from https://huggingface.co/xiangyu556677/EasyGen. Also, you need to change the line 671, 677 and 682 to your own root. As for BiDiffuser's weight, please according to [UniDiffuser](https://github.com/thu-ml/unidiffuser) to download relevant weight (such as AutoKL and clip's weight) and change the line 649 (the weight of [BiDiffuser](https://huggingface.co/xiangyu556677/EasyGen)) to your own root.
By using this command, EasyGen is trained on multimodal dialogue conversation and can generate images:
```
python -m fastchat.serve.inference_easygen
```
# Acknowledgement

+ [UniDiffuser](https://github.com/thu-ml/unidiffuser) The diffusion module of EasyGen, BiDiffuser, is developed based on UniDiffuser!
+ [FastChat](https://github.com/lm-sys/FastChat) This repository is built upon FastChat!




