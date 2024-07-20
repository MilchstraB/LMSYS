import numpy as np


def process_func_2(example, tokenizer):
    "Truncation: head + tail. From https://arxiv.org/abs/1905.05583"
    MAX_LENGTH = 1844  # 1.8K
    inputs = [
        "User prompt: ",
        example["prompt"],
        "\n\nModel A :\n",
        example["response_a"],
        "\n\n--------\n\nModel B:\n",
        example["response_b"],
    ]

    input_ids, attention_mask = [], []
    total_length = 0

    for e in inputs:
        token_ids = tokenizer(e, return_tensors="np")
        input_ids.append(token_ids["input_ids"][0])
        attention_mask.append(token_ids["attention_mask"][0])
        total_length += len(token_ids["input_ids"][0])

    if total_length > MAX_LENGTH:
        truncation_length = total_length - MAX_LENGTH
        len_a = len(input_ids[3])
        len_b = len(input_ids[-1])
        len_p = len(input_ids[1])

        ratio_a = len_a / (len_p + len_a + len_b)
        ratio_b = len_b / (len_p + len_a + len_b)

        trunc_a = int(truncation_length * ratio_a)
        trunc_b = int(truncation_length * ratio_b)
        trunc_p = truncation_length - trunc_a - trunc_b

        def truncate_seq(seq, trunc_len):
            head_len = trunc_len // 2
            tail_len = trunc_len - head_len
            return np.concatenate((seq[:head_len], seq[-tail_len:]))

        input_ids[1] = truncate_seq(input_ids[1], len_p - trunc_p)
        attention_mask[1] = truncate_seq(attention_mask[1], len_p - trunc_p)
        input_ids[3] = truncate_seq(input_ids[3], len_a - trunc_a)
        attention_mask[3] = truncate_seq(attention_mask[3], len_a - trunc_a)
        input_ids[-1] = truncate_seq(input_ids[-1], len_b - trunc_b)
        attention_mask[-1] = truncate_seq(attention_mask[-1], len_b - trunc_b)

    return {
        "input_ids": np.concatenate(input_ids),
        "attention_mask": np.concatenate(attention_mask),
        "labels": example["label"],
    }
