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
#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \


