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
from text_process import TextProcessorV2

os.environ["WANDB_PROJECT"] = "LMSYS_Text_ClS"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
from transformers import (
    Trainer,
)
from dataclasses import dataclass, field
from transformers.trainer import *
import torch.nn.functional as F

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


@dataclass
class MyLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    def __init__(self, smooth_factor):
        self.smooth_factor = smooth_factor
        self.criterion = nn.CrossEntropyLoss()
    def __call__(self, model_output, labels, num_classes=3):
        logits = model_output["logits"]
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
        smooth_factor = self.smooth_factor
        # 0: a wins, 1: b wins, 2: tie

        # label smoothing
        # if a/b wins, only smooth to tie. For example, if a wins, original label is 0, smooth to [0.9, 0.1, 0.0]
        # if tie, smooth to both a and b wins. For example, if tie, original label is 2, smooth to [0.05, 0.05, 0.9]
        # Initialize smoothed labels with the same shape as one-hot labels
        smoothed_labels = torch.zeros_like(one_hot_labels)

        # Apply smoothing
        for i in range(num_classes):
                if i == 0:  # Case where original label is 0 (a wins)
                    smoothed_labels[:, i] = torch.max(smoothed_labels[:, i], one_hot_labels[:, i] * (1 - smooth_factor))
                    smoothed_labels[:, 2] = torch.max(smoothed_labels[:, 2], one_hot_labels[:, i] * smooth_factor)
                elif i == 1:  # Case where original label is 1 (b wins)
                    smoothed_labels[:, i] = torch.max(smoothed_labels[:, i], one_hot_labels[:, i] * (1 - smooth_factor))
                    smoothed_labels[:, 2] = torch.max(smoothed_labels[:, 2], one_hot_labels[:, i] * smooth_factor)
                else:  # Case where original label is 2 (tie)
                    smoothed_labels[:, i] = torch.max(smoothed_labels[:, i], one_hot_labels[:, i] * (1 - smooth_factor))
                    smoothed_labels[:, 0] = torch.max(smoothed_labels[:, 0], one_hot_labels[:, i] * smooth_factor / 2)
                    smoothed_labels[:, 1] = torch.max(smoothed_labels[:, 1], one_hot_labels[:, i] * smooth_factor / 2)

        loss = self.criterion(logits, smoothed_labels)

        return loss        



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="google/gemma-2-9b-it"
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

    lora_enable: bool = field(default=False)
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

    class MyTrainer(Trainer):
        # only change compute_loss
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_smoother = MyLabelSmoother(self.args.label_smoothing_factor) if self.args.label_smoothing_factor > 0 else None
        def create_optimizer(self):
            global model
            para_group = [
            {
            "params": [p for n, p in model.named_parameters() if "score" in n],
            "lr": self.args.learning_rate,
            "weight_decay": 0.0,
            },
            {
            "params": [p for n, p in model.named_parameters() if "score" not in n],
            "lr": self.args.learning_rate * 10,
            "weight_decay": 0.0,
            },
            ]
            optimizer = torch.optim.AdamW(para_group)
            return optimizer
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.

            Subclass and override for custom behavior.
            """
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    print("We are using MyLabelSmoother")
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
    trainer = MyTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.log({"text_process_parameter": hyper_parameter})
    trainer.train()

    val_result = trainer.evaluate(val_dataset, metric_key_prefix="val")

    test_result = trainer.evaluate(test_dataset, metric_key_prefix="test")
    with open(os.path.join(training_args.output_dir, "result.json"), "w") as f:
        json.dump([test_result, val_result], f)


if __name__ == "__main__":
    train()
