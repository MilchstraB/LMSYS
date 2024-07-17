deepspeed train.py \
    --output_dir ./output/gemma2_baseline \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /home/share/pyz/model_weight/gemma-2-9b-it \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard

deepspeed train.py \
    --output_dir ./output/gemma2_baseline \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /home/share/pyz/model_weight/gemma-2-9b-it \
    --train_data_path data/split/train.csv \
    --val_data_path data/split/val.csv \
    --test_data_path data/split/test.csv \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.05 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to tensorboard
