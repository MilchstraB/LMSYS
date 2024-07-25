
deepspeed train.py \
    --model_name_or_path google/gemma-2-9b-it \
    --model_max_length 2048 \
    --instruction "" \
    --prompt_template "<prompt>: <\P>" \
    --a_template "\n<response_a>: <\A>" \
    --b_template "\n<response_b>: <\B>" \
    --add_eos_token False \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --output_dir ./output/gemma2_baseline \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_dora False \
    --gradient_checkpointing True \
    --lora_target "all-linear" \
    --eval_steps 0.2 \
    --eval_strategy "steps" \
    --bf16_full_eval True \
    --output_dir "a100_gemma" \
    --group_by_length False \
    --debug_fast_test False \
    --label_smoothing_factor 0.0 \
    --warmup_ratio 0.05 \
    --logging_steps 0.005 \
    --report_to "wandb" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --show_length False
