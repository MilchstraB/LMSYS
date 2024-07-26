export WANDB_API_KEY="2f787b5e67ea6f6970182ba8f57f0a61060d7d12"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_template" \
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


deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_template" \
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
    --chat_template "template"


deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_template" \
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
    --chat_template "chat_template_with_token_num"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_template" \
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
    --chat_template "chat_template"



deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_template" \
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
    --length_assign_method method_3 \
    --chat_template "template_with_token_num"


deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_length" \
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
    --length_assign_method method_2 \
    --chat_template "template_with_token_num"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_length" \
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
    --length_assign_method method_1 \
    --chat_template "template_with_token_num"

deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\"]" \
    --output_dir "a100_gemma_template_lora32/64" \
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
    --chat_template "chat_template"