from typing import Any
from transformers import PreTrainedTokenizerBase


class CustomTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        prompt_template: str,
        a_template: str,
        b_template: str,
        instruction: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.b_template = b_template
        self.a_template = a_template
        self.instruction = instruction

    def __call__(self, batch: dict) -> dict:
        prompt = [
            self.prompt_template.replace("<\P>", self.process_text(t))
            for t in batch["prompt"]
        ]
        response_a = [
            self.a_template.replace("<\A>", self.process_text(t))
            for t in batch["response_a"]
        ]
        response_b = [
            self.b_template.replace("<\B>", self.process_text(t))
            for t in batch["response_b"]
        ]
        texts = [
            self.instruction + "\n".join(sample)
            for sample in zip(prompt, response_a, response_b)
        ]

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