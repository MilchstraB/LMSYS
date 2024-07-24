from typing import Dict, List, Optional, Union

import numpy as np
from transformers import AutoTokenizer


def parse_text(text: str) -> list:
    return eval(text, {"null": ""})


default_chat_template = """<bos><start_of_turn>user
{prompt}\n
<response_a> ({a_word_num} words): {response_a}\n
<response_b> ({b_word_num} words): {response_b}
<end_of_turn>
<start_of_turn>model
"""


class TextProcessorV2:
    def __init__(
        self,
        truncation_method: str,
        length_assign_method: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        chat_template: Optional[str] = None,
    ):
        self.truncation_method = truncation_method
        self.length_assign_method = length_assign_method
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template

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
            ...
        """

    def __call__(self, batch_data):
        """ """
        # 预处理文本：
        batch_prompt = [" ".join(parse_text(t)).strip() for t in batch_data["prompt"]]
        batch_response_a = [
            " ".join(parse_text(t)).strip() for t in batch_data["response_a"]
        ]
        batch_response_b = [
            " ".join(parse_text(t)).strip() for t in batch_data["response_b"]
        ]

        def compute_token_num(text: str):
            return len(
                self.tokenizer(text, add_special_tokens=False, truncation=False)[
                    "input_ids"
                ]
            )

        # compute token num of prompt, response_a, response_b
        a = [compute_token_num(p) for p in batch_prompt]
        prompt_token_num = np.array([compute_token_num(p) for p in batch_prompt])
        response_a_token_num = np.array(
            [compute_token_num(p) for p in batch_response_a]
        )
        response_b_token_num = np.array(
            [compute_token_num(p) for p in batch_response_b]
        )

        # 使用caht_template拼接所有文本 如果"a_word_num"在chat_template中则format中添加这个参数
        concat_batch_text = []
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
                concat_batch_text.append(text)
            else:
                text = self.chat_template.format(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                )
                concat_batch_text.append(text)
        concat_batch_text_token_num = np.array(
            [compute_token_num(text) for text in concat_batch_text]
        )
        # 除了prompt, response_a, response_b之外的其他部分的token num
        other_part_token_num = (
            concat_batch_text_token_num
            - prompt_token_num
            - response_a_token_num
            - response_b_token_num
        )

        # 计算最多容纳多少数据token, 10作为余量
        max_token_capacity = self.max_length - other_part_token_num - 10

        def get_part_capacity(
            prompt_token_num,
            response_a_token_num,
            response_b_token_num,
            cur_max_token_capacity,
        ):
            # 按比例分配给各部分
            # 方法一：prompt全部保留，response_a和response_b按长度分配
            if self.length_assign_method == "method_1":
                response_token_capacity = max(
                    cur_max_token_capacity - prompt_token_num, 0
                )
                prompt_capacity = min(prompt_token_num, cur_max_token_capacity)
                response_a_capacity = int(
                    response_token_capacity
                    * response_a_token_num
                    / (response_a_token_num + response_b_token_num)
                )
                response_b_capacity = response_token_capacity - response_a_capacity
            # 方法二：prompt，response_a, response_b都按长度分配:
            elif self.length_assign_method == "method_2":
                prompt_capacity = int(
                    cur_max_token_capacity
                    * prompt_token_num
                    / (prompt_token_num + response_a_token_num + response_b_token_num)
                )
                response_a_capacity = int(
                    cur_max_token_capacity
                    * response_a_token_num
                    / (prompt_token_num + response_a_token_num + response_b_token_num)
                )
                response_b_capacity = (
                    cur_max_token_capacity - prompt_capacity - response_a_capacity
                )
            # 方法三：prompt全部保留，response_a和response_b平分
            elif self.length_assign_method == "method_3":
                response_token_capacity = max(
                    cur_max_token_capacity - prompt_token_num, 0
                )
                prompt_capacity = min(prompt_token_num, cur_max_token_capacity)
                response_a_capacity = int(response_token_capacity * 0.5)
                response_b_capacity = response_token_capacity - response_a_capacity
            else:
                raise ValueError("Method not supported")
            return prompt_capacity, response_a_capacity, response_b_capacity

        final_input = {"input_ids": [], "attention_mask": []}
        for i, token_num in enumerate(concat_batch_text_token_num):
            if token_num > self.max_length:
                prompt_capacity, response_a_capacity, response_b_capacity = (
                    get_part_capacity(
                        prompt_token_num[i],
                        response_a_token_num[i],
                        response_b_token_num[i],
                        max_token_capacity[i],
                    )
                )
                # capacity -1 留一定的余地，防止超长
                if self.truncation_method in ["left", "right"]:
                    # 保留最后面的文本内容
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
                else:
                    raise ValueError("Truncation method not supported")
                # decode them to text
                prompt_text = self.tokenizer.decode(prompt["input_ids"]).strip()
                response_a_text = self.tokenizer.decode(response_a["input_ids"]).strip()
                response_b_text = self.tokenizer.decode(response_b["input_ids"]).strip()

                # concat them if "a_word_num" in chat_template
                if "a_word_num" in self.chat_template:
                    text = self.chat_template.format(
                        prompt=prompt_text,
                        response_a=response_a_text,
                        response_b=response_b_text,
                        a_word_num=response_a_token_num[i],
                        b_word_num=response_a_token_num[i],
                    )
                else:
                    text = self.chat_template.format(
                        prompt=prompt_text,
                        response_a=response_a_text,
                        response_b=response_b_text,
                    )
                # encode the text to input_ids and attention_mask
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=False,
                    add_special_tokens=False,
                )
                if len(inputs["input_ids"]) > self.max_length:
                    # print(response_a_text)
                    print(prompt_capacity, response_a_capacity, response_b_capacity)
                    print("=" * 80)
                    # print(response_b_text)
                    print(text)
                assert (
                    len(inputs["input_ids"]) <= self.max_length
                ), f"{len(inputs['input_ids'])}"
                final_input["input_ids"].append(inputs["input_ids"])
                final_input["attention_mask"].append(inputs["attention_mask"])
            else:
                inputs = self.tokenizer(
                    concat_batch_text[i],
                    max_length=self.max_length,
                    truncation=False,
                    add_special_tokens=False,
                )
                assert len(inputs["input_ids"]) <= self.max_length
                final_input["input_ids"].append(inputs["input_ids"])
                final_input["attention_mask"].append(inputs["attention_mask"])
        self.tokenizer.truncation_side = "right"
        return final_input
