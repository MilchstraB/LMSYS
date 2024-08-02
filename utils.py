import re
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer


def get_optimizer_grouped_parameters(model, base_lr, score_lr, weight_decay):
    no_decay = ["bias", "layernorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "score" in n and not any(nd in n for nd in no_decay)],
            "lr": score_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "score" in n and any(nd in n for nd in no_decay)],
            "lr": score_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "score" not in n and not any(nd in n for nd in no_decay)],
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "score" not in n and any(nd in n for nd in no_decay)],
            "lr": base_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        device = logits.device
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1.2], dtype=torch.bfloat16).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def is_only_whitespace(s):
    return re.fullmatch(r'\s*', s) is not None


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


class CustomTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        prompt_template: str,
        a_template: str,
        b_template: str,
        instruction: str,
        show_length: bool = False,
        use_chat_template: bool = False,
        switch: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.b_template = b_template
        self.a_template = a_template
        self.instruction = instruction
        self.show_length = show_length
        self.use_chat_template = use_chat_template
        self.switch = switch

    def __call__(self, batch: dict) -> dict:
        if self.show_length == False:
            response_a = [
                self.a_template.replace("<\A>", self.process_text(t))
                for t in batch["response_a"]
            ]
            response_b = [
                self.b_template.replace("<\B>", self.process_text(t))
                for t in batch["response_b"]
            ]
        else:
            response_a = [
                self.a_template.replace("<\A>", self.process_text(t)).replace(
                    "<response_a>:",
                    f"<response_a> ({self.process_text(t).count(' ')} words):",
                )
                for t in batch["response_a"]
            ]
            response_b = [
                self.b_template.replace("<\B>", self.process_text(t)).replace(
                    "<response_b>:",
                    f"<response_b> ({self.process_text(t).count(' ')} words):",
                )
                for t in batch["response_b"]
            ]
        prompt = [
            self.prompt_template.replace("<\P>", self.process_text(t))
            for t in batch["prompt"]
        ]

        if self.switch:
            response_a, response_b = response_b, response_a
        
        texts = [
            self.instruction + "\n".join(sample)
            for sample in zip(prompt, response_a, response_b)
        ]

        add_special_tokens = not self.use_chat_template
        if self.use_chat_template:
            texts = [
                "<bos><start_of_turn>user\n<<content>><end_of_turn>\n<start_of_turn>model\n".replace(
                    "<<content>>", t
                )
                for t in texts
            ]
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=False,
            add_special_tokens=add_special_tokens,
        )
        token_length = [len(t) for t in tokenized["input_ids"]]

        tokenized_truncation = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        labels = []
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)

        if self.switch:
            mapping = [1, 0, 2]
            labels = [mapping[e] for e in labels]
        return {**tokenized_truncation, "labels": labels, "token_length": token_length}

    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))


class ConvTokenizer:
    """Concat the text with turn format: [prompt_1, response_a_1, response_b_1, prompt_2, response_a_2, response_b_2, ...]
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        prompt = [self.process_text(t) for t in batch["prompt"]]
        response_a = [self.process_text(t) for t in batch["response_a"]]
        response_b = [self.process_text(t) for t in batch["response_b"]]

        texts = []
        for turns_p, turns_a, turns_b in zip(prompt, response_a, response_b):
            conversation, round = (
                "Which one of the chatbots below did a better job responding to the user request? Or were they tied?",
                1,
            )
            for p, a, b in zip(turns_p, turns_a, turns_b):
                if is_only_whitespace(p) or is_only_whitespace(a) or is_only_whitespace(b):
                    continue
                conversation += f"\n\n~~~~~~~~~~ ROUND {round} ~~~~~~~~~~"
                conversation += f"\n\n<prompt>: {p}"
                conversation += f"\n\n<response_a>: {a}"
                conversation += f"\n\n<response_b>: {b}"
                round += 1
            texts.append(conversation)

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
        return eval(text, {"null": ""})
