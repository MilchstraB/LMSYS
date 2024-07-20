import time
from dataclasses import dataclass
import copy
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, log_loss
from tqdm import trange
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from utils import CustomTokenizer


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size, max_length):
    a_win, b_win, tie = [], [], []
    new_df = copy.deepcopy(df)
    for start_idx in trange(0, len(new_df), batch_size):
        end_idx = min(start_idx + batch_size, len(new_df))
        tmp = new_df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()

        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())

    new_df["winner_model_a"] = a_win
    new_df["winner_model_b"] = b_win
    new_df["winner_tie"] = tie

    return new_df


@dataclass
class Config:
    model_name_or_path = "/data/share/pyz/model_weight/gemma-2-9b-it"
    lora_dir = "output/gemma2_baseline/checkpoint-1435"
    model_max_length = 2048
    batch_size = 4
    device = torch.device("cuda")
    test_data_path = "data/split/test.csv"
    tta = True  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>

    # prompt_template = "Prompt: <\P>"
    # a_template = "Response of A: <\A>"
    # b_template = "Response of B: <\B>"
    # instruction = "Now I will give you a prompt and two responses. You should choose the better response.\n"
    prompt_template = "<prompt>: <\P>"
    a_template = "\n<response_a>: <\A>"
    b_template = "\n<response_b>: <\B>"
    instruction = ""


def calculate_metrics(predictions_df, true_labels_df):
    """
    Calculate log loss and accuracy between predictions and true labels.

    Parameters:
    predictions_df (pd.DataFrame): DataFrame containing predicted probabilities.
    true_labels_df (pd.DataFrame): DataFrame containing true labels.

    Returns:
    tuple: (average log loss, accuracy)
    """
    # Ensure the DataFrames are aligned on the index
    predictions_df = predictions_df.set_index("id")
    true_labels_df = true_labels_df.set_index("id")

    # Extract true labels as one-hot encoded vectors
    true_labels = true_labels_df[
        ["winner_model_a", "winner_model_b", "winner_tie"]
    ].values

    # Extract predicted probabilities
    predicted_probabilities = predictions_df[
        ["winner_model_a", "winner_model_b", "winner_tie"]
    ].values

    # Calculate log loss
    avg_log_loss = log_loss(true_labels, predicted_probabilities)

    # Extract true labels as class indices for accuracy calculation
    true_label_indices = np.argmax(true_labels, axis=1)
    predicted_label_indices = np.argmax(predicted_probabilities, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(true_label_indices, predicted_label_indices)

    return avg_log_loss, accuracy


if __name__ == "__main__":
    cfg = Config()
    tokenizer = GemmaTokenizerFast.from_pretrained(
        cfg.model_name_or_path, padding_side="right", use_fast=True
    )

    preprocess = CustomTokenizer(
        tokenizer,
        max_length=cfg.model_max_length,
        prompt_template=cfg.prompt_template,
        a_template=cfg.a_template,
        b_template=cfg.b_template,
        instruction=cfg.instruction,
    )
    raw_dataset = Dataset.from_csv(cfg.test_data_path)
    test_dataset = raw_dataset.map(preprocess, batched=True)
    data = pd.DataFrame(test_dataset.to_dict())

    data["length"] = data["input_ids"].apply(len)

    aug_test_dataset = raw_dataset.rename_columns(
        {"response_a": "response_b", "response_b": "response_a"}
    )
    aug_test_dataset = aug_test_dataset.map(preprocess, batched=True)

    aug_data = pd.DataFrame(aug_test_dataset.to_dict())
    aug_data["length"] = aug_data["input_ids"].apply(len)
    # Load base model on GPU 0
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma2ForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        num_labels=3,
        quantization_config=bnb_config,
        device_map=cfg.device,
        use_cache=False,
    )
    model = PeftModel.from_pretrained(model, cfg.lora_dir)

    # sort by input length to fully leverage dynaminc padding
    data = data.sort_values("length", ascending=False)
    result_df = inference(
        data,
        model,
        cfg.device,
        batch_size=cfg.batch_size,
        max_length=cfg.model_max_length,
    )
    proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values
    print(calculate_metrics(result_df, data))

    ensemble_proba = None
    if cfg.tta:
        aug_data = aug_data.sort_values(
            "length", ascending=False
        )  # sort by input length to boost speed
        tta_result_df = inference(
            aug_data,
            model,
            cfg.device,
            batch_size=cfg.batch_size,
            max_length=cfg.model_max_length,
        )
        # recall TTA's order is flipped
        tta_proba = tta_result_df[
            ["winner_model_b", "winner_model_a", "winner_tie"]
        ].values
        # average original result and TTA result.
        ensemble_proba = (proba + tta_proba) / 2

    ensembel_result_df = copy.deepcopy(result_df)
    ensembel_result_df.loc[:, "winner_model_a"] = ensemble_proba[:, 0]
    ensembel_result_df.loc[:, "winner_model_b"] = ensemble_proba[:, 1]
    ensembel_result_df.loc[:, "winner_tie"] = ensemble_proba[:, 2]
    submission_df = ensembel_result_df[
        ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    ]
    submission_df.to_csv("preds_gemma.csv", index=False)

    log_loss_result = calculate_metrics(submission_df, data)
    print(log_loss_result)
