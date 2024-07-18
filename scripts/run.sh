deepspeed train.py \
    --output_dir ./output/gemma2_exp7 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --model_name_or_path ./cache/gemma2 \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard


deepspeed train.py \
    --output_dir ./output/gemma2_exp8 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --model_name_or_path ./cache/gemma2 \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard

deepspeed train.py \
    --output_dir ./output/gemma2_exp9 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_target '["q_proj", "k_proj", "v_proj"]' \
    --model_name_or_path ./cache/gemma2 \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard

deepspeed train.py \
    --output_dir ./output/gemma2_exp10 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_target '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]' \
    --model_name_or_path ./cache/gemma2 \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard

deepspeed train.py \
    --output_dir ./output/gemma2_exp11 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./cache/gemma2 \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --label_smoothing_factor 0.1

deepspeed train.py \
    --output_dir ./output/gemma2_exp12 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./cache/gemma2 \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --freeze_layers 16

python3 /h3cstore_nt/daiyi.zhu/temp/train.py
