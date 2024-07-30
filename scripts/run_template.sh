export WANDB_API_KEY="2f787b5e67ea6f6970182ba8f57f0a61060d7d12"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero3.json \
    --lora_enable False \
    --output_dir "a100_gemma_template_with_token_num_eos_FT_1e-5_warmup0.05_scoreX10" \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --save_total_limit 1 \
    --save_steps 0.2 \
    --layers_to_transform 0 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --truncation_method right \
    --length_assign_method method_2 \
    --chat_template "template_with_token_num_eos"


deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero3.json \
    --lora_enable False \
    --output_dir "a100_gemma_template_with_token_num_eos_FT_1e-5_warmup0.1_scoreX10" \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --save_total_limit 1 \
    --save_steps 0.2 \
    --layers_to_transform 0 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --truncation_method right \
    --length_assign_method method_2 \
    --chat_template "template_with_token_num_eos"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero3.json \
    --lora_enable False \
    --output_dir "a100_gemma_template_with_token_num_eos_FT_8e-5_warmup0.05_scoreX10" \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --save_total_limit 1 \
    --save_steps 0.2 \
    --layers_to_transform 0 \
    --learning_rate 8e-5 \
    --warmup_ratio 0.05 \
    --truncation_method right \
    --length_assign_method method_2 \
    --chat_template "template_with_token_num_eos"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero3.json \
    --lora_enable False \
    --output_dir "a100_gemma_template_with_token_num_eos_FT_3e-5_warmup0.05_scoreX10" \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 4 \
    --save_total_limit 1 \
    --save_steps 0.2 \
    --layers_to_transform 0 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.05 \
    --truncation_method right \
    --length_assign_method method_2 \
    --chat_template "template_with_token_num_eos"
