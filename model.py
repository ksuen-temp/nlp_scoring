from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput


@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    probs: torch.FloatTensor = None


class ModelForTokenClassification(PreTrainedModel):
    def __init__(self, backbone, config) -> None:
        super().__init__(config)
        self.backbone = backbone
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, 1)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))
            loss = torch.masked_select(loss, labels.view(-1, 1) > -1).mean()

        return TokenClassifierOutput(
            loss=loss,
            probs=logits[:, :, 0].sigmoid(),
        )
