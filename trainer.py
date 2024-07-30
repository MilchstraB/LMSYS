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

    

class MyTrainer(Trainer):
    # only change compute_loss
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoother = MyLabelSmoother(self.args.label_smoothing_factor) if self.args.label_smoothing_factor > 0 else None

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