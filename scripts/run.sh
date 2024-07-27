deepspeed train.py \
    --model_name_or_path ./cache/gemma2 \
    --model_max_length 2048 \
    --instruction "" \
    --prompt_template "<prompt>: <\P>" \
    --a_template "\n<response_a>: <\A>" \
    --b_template "\n<response_b>: <\B>" \
    --add_eos_token True \
    --use_chat_template False \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --output_dir ./output/gemma2-ft \
    --run_name gemma2-ft-baseline \
    --deepspeed ./scripts/zero3.json \
    --lora_enable False \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_dora False \
    --gradient_checkpointing True \
    --lora_target '["down_proj", "q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "gate_proj"]' \
    --eval_steps 0.2 \
    --eval_strategy "steps" \
    --bf16_full_eval True \
    --group_by_length False \
    --debug_fast_test False \
    --label_smoothing_factor 0.0 \
    --warmup_steps 20 \
    --logging_steps 0.005 \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --show_length False