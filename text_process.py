from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from transformers import AutoTokenizer


def parse_text(text: str) -> list:
    return eval(text, {"null": ""})


templates_dict = {
    "chat_template_with_token_num": """<bos><start_of_turn>user
<prompt>: {prompt}\n
<response_a> ({a_word_num} words): {response_a}\n
<response_b> ({b_word_num} words): {response_b}
<end_of_turn>
<start_of_turn>model
""",
    "chat_template": """<bos><start_of_turn>user
<prompt>: {prompt}\n
<response_a>: {response_a}\n
<response_b>: {response_b}
<end_of_turn>
<start_of_turn>model
""",
    "template": """<bos><prompt>: {prompt}\n
<response_a>: {response_a}\n
<response_b>: {response_b}
""",
    "template_with_token_num": """<bos><prompt>: {prompt}\n
<response_a> ({a_word_num} words): {response_a}\n
<response_b> ({b_word_num} words): {response_b}
""",
    "template_with_token_num_eos": """<bos><prompt>: {prompt}\n
<response_a> ({a_word_num} words): {response_a}\n
<response_b> ({b_word_num} words): {response_b}
<eos>""",
    "template_with_eos": """<bos><prompt>: {prompt}\n
<response_a>: {response_a}\n
<response_b>: {response_b}
<eos>""",
    "non_sp_token_template": """<prompt>: {prompt}\n
<response_a>: {response_a}\n
<response_b>: {response_b}
""",
}


class TextProcessorV2:
    def __init__(
        self,
        truncation_method: str,
        length_assign_method: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        chat_template: Optional[str] = None,
        get_labels: Optional[bool] = True,
    ):
        """
        Initializes the TextProcessor object.

        Args:
            truncation_method (str): The method used for truncating text.
            length_assign_method (str): The method used for assigning length to text.
            tokenizer (AutoTokenizer): The tokenizer object used for tokenization.
            max_length (int): The maximum length of the processed text.
            chat_template (Optional[str], optional): The chat template to be used. Defaults to None.
            get_labels (Optional[bool], optional): Whether to retrieve labels. Defaults to True. [For Inference, set to False.]
        """
        self.chat_template = templates_dict["chat_template_with_token_num"]
        if chat_template is not None and chat_template in templates_dict:
            self.chat_template = templates_dict[chat_template]
        elif isinstance(chat_template, str):
            self.chat_template = chat_template
            print(f"[WEARING]: The chat_template set as: {self.chat_template}")
        else:
            raise ValueError("Chat template not supported")

        self.truncation_method = truncation_method
        self.length_assign_method = length_assign_method
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.get_labels = get_labels

    def preprocess_batch(
        self, batch_data: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        batch_prompt = [" ".join(parse_text(t)).strip() for t in batch_data["prompt"]]
        batch_response_a = [
            " ".join(parse_text(t)).strip() for t in batch_data["response_a"]
        ]
        batch_response_b = [
            " ".join(parse_text(t)).strip() for t in batch_data["response_b"]
        ]
        return batch_prompt, batch_response_a, batch_response_b

    def compute_token_num(self, text: str) -> int:
        return len(
            self.tokenizer(text, add_special_tokens=False, truncation=False)[
                "input_ids"
            ]
        )

    def format_texts(
        self,
        batch_prompt: List[str],
        batch_response_a: List[str],
        batch_response_b: List[str],
        response_a_token_num: List[int],
        response_b_token_num: List[int],
    ) -> List[str]:
        texts = []
        for prompt, response_a, response_b, a_num, b_num in zip(
            batch_prompt,
            batch_response_a,
            batch_response_b,
            response_a_token_num,
            response_b_token_num,
        ):
            if "a_word_num" in self.chat_template:
                text = self.chat_template.format(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                    a_word_num=a_num,
                    b_word_num=b_num,
                )
            else:
                text = self.chat_template.format(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                )
            texts.append(text)
        return texts

    def get_part_capacity(
        self,
        prompt_token_num: int,
        response_a_token_num: int,
        response_b_token_num: int,
        cur_max_token_capacity: int,
    ) -> tuple:
        if self.length_assign_method == "method_1":
            response_token_capacity = max(cur_max_token_capacity - prompt_token_num, 0)
            prompt_capacity = min(prompt_token_num, cur_max_token_capacity)
            response_a_capacity = int(
                response_token_capacity
                * response_a_token_num
                / (response_a_token_num + response_b_token_num)
            )
            response_b_capacity = response_token_capacity - response_a_capacity
        elif self.length_assign_method == "method_2":
            total_tokens = (
                prompt_token_num + response_a_token_num + response_b_token_num
            )
            prompt_capacity = int(
                cur_max_token_capacity * prompt_token_num / total_tokens
            )
            response_a_capacity = int(
                cur_max_token_capacity * response_a_token_num / total_tokens
            )
            response_b_capacity = (
                cur_max_token_capacity - prompt_capacity - response_a_capacity
            )
        elif self.length_assign_method == "method_3":
            response_token_capacity = max(cur_max_token_capacity - prompt_token_num, 0)
            prompt_capacity = min(prompt_token_num, cur_max_token_capacity)
            response_a_capacity = response_token_capacity // 2
            response_b_capacity = response_token_capacity - response_a_capacity
        else:
            raise ValueError("Method not supported")
        return prompt_capacity, response_a_capacity, response_b_capacity

    def __call__(self, batch_data):
        """
        Preprocesses the text data in the batch and computes the token numbers for each part of the text.
        Then, it calculates the maximum token capacity for the data and assigns token capacities to different parts of the text based on the specified length assignment method.
        Finally, it truncates the text if necessary, encodes it into input_ids and attention_mask, and returns the final input.

        Args:
            batch_data (dict): A dictionary containing the batch data with keys "prompt", "response_a", and "response_b".

        Returns:
            dict: A dictionary containing the final input with keys "input_ids" and "attention_mask".

        self.truncation_method 可以为 [left, right]，表示prompt从哪部分截断
        self.length_assign_method 可以为 [method_1, method_2, method_3]，表示分配长度的方法
            - 方法一：prompt全部保留，response_a和response_b按长度分配
            - 方法二：prompt，response_a, response_b都按长度分配
            - 方法三：prompt全部保留，response_a和response_b平分
            - 方法四：原先的方法，直接截断response_b
            ...
        """
        batch_prompt, batch_response_a, batch_response_b = self.preprocess_batch(
            batch_data
        )
        final_input = defaultdict(list)
        if self.get_labels:
            final_input["labels"] = self.extract_labels(batch_data)
        prompt_token_num = np.array([self.compute_token_num(p) for p in batch_prompt])
        response_a_token_num = np.array(
            [self.compute_token_num(r) for r in batch_response_a]
        )
        response_b_token_num = np.array(
            [self.compute_token_num(r) for r in batch_response_b]
        )
        p_len, a_len, b_len = [], [], []
        for i in range(len(batch_prompt)):
            p_len.append(prompt_token_num[i])
            a_len.append(response_a_token_num[i])
            b_len.append(response_b_token_num[i])

        final_input["original_prompt_length"] = p_len
        final_input["original_response_a_length"] = a_len
        final_input["original_response_b_length"] = b_len
        if self.length_assign_method == "method_4":
            texts = self.format_texts(
                batch_prompt,
                batch_response_a,
                batch_response_b,
                response_a_token_num,
                response_b_token_num,
            )
            tokenized = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=False,
                add_special_tokens=True,
            )
            token_length = [len(t) for t in tokenized["input_ids"]]
            self.tokenizer.add_eos_token = True
            tokenized_truncation = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
            )
            for key in tokenized_truncation:
                final_input[key] = tokenized_truncation[key]
            final_input["token_length"] = token_length
            return final_input

        concat_batch_text = self.format_texts(
            batch_prompt,
            batch_response_a,
            batch_response_b,
            response_a_token_num,
            response_b_token_num,
        )
        concat_batch_text_token_num = np.array(
            [self.compute_token_num(text) for text in concat_batch_text]
        )

        other_part_token_num = (
            concat_batch_text_token_num
            - prompt_token_num
            - response_a_token_num
            - response_b_token_num
        )
        max_token_capacity = self.max_length - other_part_token_num - 10
        token_length = []
        for i, token_num in enumerate(concat_batch_text_token_num):

            if token_num > self.max_length:
                prompt_capacity, response_a_capacity, response_b_capacity = (
                    self.get_part_capacity(
                        prompt_token_num[i],
                        response_a_token_num[i],
                        response_b_token_num[i],
                        max_token_capacity[i],
                    )
                )
                if self.truncation_method in ["left", "right"]:

                    self.tokenizer.truncation_side = self.truncation_method
                    prompt = self.tokenizer(
                        batch_prompt[i],
                        max_length=max(prompt_capacity, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    response_a = self.tokenizer(
                        batch_response_a[i],
                        max_length=max(response_a_capacity, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    response_b = self.tokenizer(
                        batch_response_b[i],
                        max_length=max(response_b_capacity, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )

                    prompt_text = self.tokenizer.decode(prompt["input_ids"]).strip()
                    response_a_text = self.tokenizer.decode(
                        response_a["input_ids"]
                    ).strip()
                    response_b_text = self.tokenizer.decode(
                        response_b["input_ids"]
                    ).strip()

                    text = self.chat_template.format(
                        prompt=prompt_text,
                        response_a=response_a_text,
                        response_b=response_b_text,
                        a_word_num=response_a_token_num[i],
                        b_word_num=response_b_token_num[i],
                    )
                elif self.truncation_method == "middle":
                    self.tokenizer.truncation_side = "left"
                    prompt = self.tokenizer(
                        batch_prompt[i],
                        max_length=max(prompt_capacity//2 - 1, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    response_a = self.tokenizer(
                        batch_response_a[i],
                        max_length=max(response_a_capacity//2 - 1, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    response_b = self.tokenizer(
                        batch_response_b[i],
                        max_length=max(response_b_capacity//2 - 1, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    prompt_text_end = self.tokenizer.decode(prompt["input_ids"]).strip()
                    response_a_text_end = self.tokenizer.decode(
                        response_a["input_ids"]
                    ).strip()
                    response_b_text_end = self.tokenizer.decode(
                        response_b["input_ids"]
                    ).strip()

                    self.tokenizer.truncation_side = "right"
                    prompt = self.tokenizer(
                        batch_prompt[i],
                        max_length=max(prompt_capacity - prompt_capacity//2 - 1, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    response_a = self.tokenizer(
                        batch_response_a[i],
                        max_length=max(response_a_capacity - response_a_capacity//2 - 1, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )
                    response_b = self.tokenizer(
                        batch_response_b[i],
                        max_length=max(response_b_capacity - response_b_capacity//2 - 1, 0),
                        truncation=True,
                        add_special_tokens=False,
                    )

                    prompt_text_start = self.tokenizer.decode(prompt["input_ids"]).strip()
                    response_a_text_start = self.tokenizer.decode(
                        response_a["input_ids"]
                    ).strip()
                    response_b_text_start = self.tokenizer.decode(
                        response_b["input_ids"]
                    ).strip()

                    prompt_text = prompt_text_start + " ... " + prompt_text_end
                    response_a_text = response_a_text_start + " ... " + response_a_text_end
                    response_b_text = response_b_text_start + " ... " + response_b_text_end


                    text = self.chat_template.format(
                        prompt=prompt_text,
                        response_a=response_a_text,
                        response_b=response_b_text,
                        a_word_num=response_a_token_num[i],
                        b_word_num=response_b_token_num[i],
                    )
                else:
                    raise ValueError("Truncation method not supported")
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=False,
                    add_special_tokens=False,
                )
                assert len(inputs["input_ids"]) <= self.max_length
                token_length.append(len(inputs["input_ids"]))
            else:
                inputs = self.tokenizer(
                    concat_batch_text[i],
                    max_length=self.max_length,
                    truncation=False,
                    add_special_tokens=False,
                )
                assert len(inputs["input_ids"]) <= self.max_length
                token_length.append(len(inputs["input_ids"]))
            for key in inputs:
                final_input[key].append(inputs[key])
        final_input["token_length"] = token_length
        self.tokenizer.truncation_side = "right"

        return final_input

    def extract_labels(self, batch_data: Dict[str, List[str]]) -> List[int]:
        labels = [
            0 if a_win else 1 if b_win else 2
            for a_win, b_win in zip(
                batch_data["winner_model_a"], batch_data["winner_model_b"]
            )
        ]
        return labels
