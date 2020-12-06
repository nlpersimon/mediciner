from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict



def get_active_labels(labels, active_loss, pad_label):
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(pad_label).type_as(labels)
            )
    return active_labels


class BertWithMRC(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.W_start = nn.Linear(config.hidden_size, 2)
        self.W_end = nn.Linear(config.hidden_size, 2)
        self.init_weights()
    
    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor=None,
                start_tensor: torch.Tensor=None,
                end_tensor: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            start_tensor: (batch_size, max_seq_len)
            end_tensor: (batch_size, max_seq_len)
        
        Returns:
            outputs: {loss: (1,) if start_tensor and end_tensor are not None,
                      start_logits: (batch_size, 2),
                      end_logits: (batch_size, 2)}
        """
        bert_outputs = self.bert(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 return_dict=True,
                                 output_hidden_states=True)
        bert_logits = bert_outputs['hidden_states'][-1]
        start_logits = self.W_start(bert_logits)
        end_logits = self.W_end(bert_logits)
        outputs = {}
        if start_tensor is not None and end_tensor is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = ((attention_mask == 1) & (token_type_ids == 1)).view(-1)
            pad_label = loss_fct.ignore_index
            active_starts = get_active_labels(start_tensor, active_loss, pad_label)
            active_ends = get_active_labels(end_tensor, active_loss, pad_label)
            loss = loss_fct(torch.cat([start_logits.view(-1, 2), end_logits.view(-1, 2)]),
                            torch.cat([active_starts.view(-1), active_ends.view(-1)]))
            outputs['loss'] = loss
        outputs['start_logits'] = start_logits
        outputs['end_logits'] = end_logits
        return outputs
        
