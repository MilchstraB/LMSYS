import transformers
from transformers import (
    TrainingArguments,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Optional, List, Union, Dict
from dataclasses import dataclass, field
from sklearn.metrics import log_loss, accuracy_score

import torch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/share/pyz/model_weight/gemma-2-9b-it"
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    test_data_path: str = field(
        default=None, metadata={"help": "Path to the test data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    lora_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    gradient_checkpointing: bool = True
    lora_target = "all-linear"
    eval_steps = 0.2
    eval_strategy = "epoch"
    eval_on_start = True
    bf16_full_eval = True
    output_dir = "gemma_beseline_debug"
    resume_from_checkpoint = None
    group_by_length = True
    debug_fast_test = False
    label_smoothing_factor = 0.0


class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = [
            "\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]
        ]
        response_b = [
            "\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]
        ]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        labels = []
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)
        return {**tokenized, "labels": labels}

    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))


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
        model_args.model_name_or_path, padding_side="right", use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=3, torch_dtype=torch.bfloat16
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
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)

    # prepare data
    train_dataset = Dataset.from_csv(data_args.train_data_path)
    val_dataset = Dataset.from_csv(data_args.val_data_path)
    test_dataset = Dataset.from_csv(data_args.test_data_path)

    if training_args.debug_fast_test:
        train_dataset = train_dataset.select(range(5))
        val_dataset = val_dataset.select(range(5))
    encode = CustomTokenizer(tokenizer, max_length=model_args.model_max_length)
    train_dataset = train_dataset.map(
        encode, batched=True, remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        encode, batched=True, remove_columns=val_dataset.column_names
    )
    test_dataset = test_dataset.map(
        encode, batched=True, remove_columns=test_dataset.column_names
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset={"val_ds": val_dataset},
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    print(trainer.evaluate(test_dataset, metric_key_prefix="test"))


if __name__ == "__main__":
    train()
