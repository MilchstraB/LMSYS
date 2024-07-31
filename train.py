import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, log_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from torch.optim import AdamW
from trainer import MyTrainer, get_optimizer_grouped_parameters
from text_process import TextProcessorV2

os.environ["WANDB_PROJECT"] = "LMSYS_Text_ClS"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="google/gemma-2b"
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    chat_template: str = field(default="template")
    truncation_method: str = field(default="left")
    length_assign_method: str = field(default="method_2")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/split/train.csv.gz", metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default="data/split/val.csv.gz", metadata={"help": "Path to the validation data."}
    )
    test_data_path: str = field(
        default="data/split/test.csv.gz", metadata={"help": "Path to the test data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    filter_long_text: str = field(default=None)


    llrd_enable: bool = field(default=False)
    score_lr: float = field(default=1e-4)

    lora_enable: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target: str = field(default="all-linear")
    layers_to_transform: int = field(default=0)
    use_dora: bool = field(default=False)

    gradient_checkpointing: bool = field(default=True)
    eval_steps: float = field(default=0.2)
    eval_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    bf16_full_eval: bool = field(default=True)
    output_dir: str = field(default="output")
    group_by_length: bool = field(default=False)
    debug_fast_test: bool = field(default=False)

    label_smoothing_factor: float = field(default=0.0)
    warmup_ratio: float = field(default=0.05)
    logging_steps: float = field(default=0.005)
    report_to: str = field(default="wandb")

    torch_compile: bool = field(default=True)
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use. [4, 8, 16]"})
    device_map: str = field(default="cuda")


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


def filter_length_ds(
    ds: Dataset,
    max_length,
    filter_method: str = "pab",
):

    if filter_method == "pab":
        return ds.filter(
            lambda x: x["original_prompt_length"]
            + x["original_response_a_length"]
            + x["original_response_b_length"]
            < max_length
        )
    elif filter_method == "pa":
        return ds.filter(
            lambda x: x["original_prompt_length"] + x["original_response_a_length"]
            < max_length
        )
    else:
        raise ValueError(f"{filter_method} error!")


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(
        training_args.output_dir, training_args.run_name
    )
    try:
        training_args.lora_target = eval(training_args.lora_target)
    except:
        training_args.lora_target = training_args.lora_target
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device_map},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
        torch_dtype=compute_dtype,
        **bnb_model_from_pretrained_args,
    )
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model.enable_input_require_grads()
    model.config.use_cache = False

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_target,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            layers_to_transform=[
                i for i in range(42) if i >= int(training_args.layers_to_transform)
            ],
            task_type=TaskType.SEQ_CLS,
            use_dora=training_args.use_dora,
        )
        model = get_peft_model(model, lora_config)

    # prepare data
    train_dataset = Dataset.from_csv(data_args.train_data_path)
    if training_args.filter_long_text:
        train_dataset = filter_length_ds(
            train_dataset,
            model_args.model_max_length,
            filter_method=training_args.filter_long_text,
        )
        print(
            f"Filter max length: {model_args.model_max_length}, total training data: {len(train_dataset)}"
        )
    val_dataset = Dataset.from_csv(data_args.val_data_path)
    test_dataset = Dataset.from_csv(data_args.test_data_path)

    if training_args.debug_fast_test:
        train_dataset = train_dataset.select(range(5))
        val_dataset = val_dataset.select(range(5))
        test_dataset = test_dataset.select(range(20))

    preprocess = TextProcessorV2(
        tokenizer=tokenizer,
        max_length=model_args.model_max_length,
        truncation_method=model_args.truncation_method,
        length_assign_method=model_args.length_assign_method,
        chat_template=model_args.chat_template,
    )

    # Save EOS Token For result check
    if model_args.length_assign_method == "method_4":
        add_eos_token = True
    elif "<eos>" in preprocess.chat_template:
        add_eos_token = True
    else:
        add_eos_token = False
    hyper_parameter = {
        "truncation_method": model_args.truncation_method,
        "length_assign_method": model_args.length_assign_method,
        "chat_template": preprocess.chat_template,
        "model_max_length": model_args.model_max_length,
        "add_eos_token": add_eos_token,
    }
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    save_path = os.path.join(training_args.output_dir, "hyper_parameter.json")
    with open(save_path, "w") as f:
        json.dump(hyper_parameter, f)
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        num_proc=8,
    )
    val_dataset = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False,
        num_proc=8,
    )
    test_dataset = test_dataset.map(
        preprocess,
        batched=True,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
        num_proc=8,
    )

    trainer = MyTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)    )
    if training_args.llrd_enable:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model,
            base_lr=training_args.learning_rate,
            score_lr=training_args.score_lr,
        )
        optimizer = AdamW(optimizer_grouped_parameters)
        trainer.optimizer = optimizer
    trainer.log({"text_process_parameter": hyper_parameter})
    trainer.train()

    val_result = trainer.evaluate(val_dataset, metric_key_prefix="val")

    test_result = trainer.evaluate(test_dataset, metric_key_prefix="test")
    with open(os.path.join(training_args.output_dir, "result.json"), "w") as f:
        json.dump([test_result, val_result], f)


if __name__ == "__main__":
    train()