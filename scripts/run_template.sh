deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma" \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --save_total_limit 1 \
    --save_steps 0.2 \
    --layers_to_transform 16 \
    --learning_rate 2e-4 \
    --truncation_method right \
    --length_assign_method method_4 \
    --chat_template "template_with_eos"
