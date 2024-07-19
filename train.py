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
)

from utils import CustomTokenizer

os.environ["WANDB_PROJECT"] = "LMSYS_Text_ClS"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/data/share/pyz/model_weight/gemma-2-9b-it"
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    instruction: str = field(
        default="Now I will give you a prompt and two responses. You should choose the better response.\n"
    )
    prompt_template: dict = field(default="Prompt: <\P>")
    a_template: str = field(default="Response of A: <\A>")
    b_template: str = field(default="Response of B: <\B>")
    add_eos_token: bool = field(default=False)


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/split/train.csv", metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default="data/split/val.csv", metadata={"help": "Path to the validation data."}
    )
    test_data_path: str = field(
        default="data/split/test.csv", metadata={"help": "Path to the test data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = "none"
    lora_target = field(default="all-linear")
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    use_dora: bool = field(default=False)

    gradient_checkpointing: bool = field(default=True)
    eval_steps = field(default=0.2)
    eval_strategy = field(default="steps")
    eval_on_start = field(default=True)
    bf16_full_eval = field(default=True)
    output_dir = field(default="gemma_beseline_debug")
    group_by_length = field(default=False)
    debug_fast_test = field(default=False)

    label_smoothing_factor = field(default=0.0)
    warmup_ratio = field(default=0.05)
    logging_steps = field(default=0.005)
    report_to = field(default="wandb")


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
        add_eos_token=model_args.add_eos_token,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=3
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
            layers_to_transform=training_args.layers_to_transform,
            task_type=TaskType.SEQ_CLS,
            use_dora=training_args.use_dora,
        )
        model = get_peft_model(model, lora_config)

    # prepare data
    train_dataset = Dataset.from_csv(data_args.train_data_path)
    val_dataset = Dataset.from_csv(data_args.val_data_path)
    test_dataset = Dataset.from_csv(data_args.test_data_path)

    if training_args.debug_fast_test:
        train_dataset = train_dataset.select(range(5))
        val_dataset = val_dataset.select(range(5))
        test_dataset = test_dataset.select(range(20))
    preprocess = CustomTokenizer(
        tokenizer,
        max_length=model_args.model_max_length,
        prompt_template=model_args.prompt_template,
        a_template=model_args.a_template,
        b_template=model_args.b_template,
        instruction=model_args.instruction,
    )
    train_dataset = train_dataset.map(
        preprocess, batched=True, remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        preprocess, batched=True, remove_columns=val_dataset.column_names
    )
    test_dataset = test_dataset.map(
        preprocess, batched=True, remove_columns=test_dataset.column_names
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    test_result = trainer.evaluate(test_dataset, metric_key_prefix="test")
    with open(os.path.join(training_args.output_dir, "result.json"), "w") as f:
        json.dump(test_result, f)


if __name__ == "__main__":
    train()
