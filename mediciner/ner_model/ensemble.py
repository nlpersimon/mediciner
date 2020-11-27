from transformers import BertForTokenClassification
import torch
import torch.nn as nn



def get_last_hidden_states(bert: BertForTokenClassification,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask)
    last_hidden_states = outputs['hidden_states'][-1]
    return last_hidden_states


class BertEnsemble(nn.Module):
    def __init__(self,
                 bert_1: BertForTokenClassification,
                 bert_2: BertForTokenClassification,
                 num_labels: int) -> None:
        super().__init__()
        self.bert_1 = bert_1.eval()
        self.bert_2 = bert_2.eval()
        self.num_labels = num_labels
        self.classifier = nn.Linear(bert_1.config.hidden_size + bert_2.config.hidden_size,
                                    num_labels)
        self._freeze_bert()
        self._init_classifier()
    
    def _freeze_bert(self) -> None:
        for bert in [self.bert_1, self.bert_2]:
            for param in bert.parameters():
                param.requires_grad = False
        return
    
    def _init_classifier(self) -> None:
        for name, params in self.classifier.named_parameters():
            if 'bias' in name:
                nn.init.constant_(params, 0.0)
            else:
                nn.init.xavier_uniform_(params)
        return
        
    def forward(self,
                input_ids_1: torch.Tensor,
                attention_mask_1: torch.Tensor,
                input_ids_2: torch.Tensor,
                attention_mask_2) -> torch.Tensor:
        h_1 = get_last_hidden_states(self.bert_1, input_ids_1, attention_mask_1)
        h_2 = get_last_hidden_states(self.bert_2, input_ids_2, attention_mask_2)
        logits = self.classifier(torch.cat([h_1, h_2], dim=-1))
        return logits