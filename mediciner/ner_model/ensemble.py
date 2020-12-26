from transformers import BertForTokenClassification
import torch
import torch.nn as nn
from typing import Dict
import os
import json



def get_last_hidden_states(bert: BertForTokenClassification,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = bert(input_ids, attention_mask)
    last_hidden_states = outputs['hidden_states'][-1]
    return last_hidden_states


class BertEnsemble(nn.Module):
    def __init__(self,
                 bert_1_path: str,
                 bert_2_path: str,
                 num_labels: int,
                 dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.bert_1_path = os.path.abspath(bert_1_path)
        self.bert_2_path = os.path.abspath(bert_2_path)
        bert_1 = BertForTokenClassification.from_pretrained(bert_1_path,
                                                            return_dict=True,
                                                            output_hidden_states=True,
                                                            num_labels=num_labels)
        bert_2 = BertForTokenClassification.from_pretrained(bert_2_path,
                                                            return_dict=True,
                                                            output_hidden_states=True,
                                                            num_labels=num_labels)
        self.bert_1 = bert_1.eval()
        self.bert_2 = bert_2.eval()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
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
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_1 = get_last_hidden_states(self.bert_1, input_ids, attention_mask)
        h_2 = get_last_hidden_states(self.bert_2, input_ids, attention_mask)
        logits = self.classifier(self.dropout(torch.cat([h_1, h_2], dim=-1)))
        outputs = {
            'logits': logits
        }
        return outputs

    def save_trained(self, save_dir_path: str) -> None:
        if not os.path.isdir(save_dir_path):
            os.mkdir(save_dir_path)
        
        config = {
            'bert_1_path': self.bert_1_path,
            'bert_2_path': self.bert_2_path,
            'num_labels': self.num_labels
        }
        with open(os.path.join(save_dir_path, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        torch.save(self.classifier.state_dict(), os.path.join(save_dir_path, 'classifier.pt'))
        return
    
    @classmethod
    def load_trained(cls, load_dir_path: str) -> 'BertEnsemble':
        with open(os.path.join(load_dir_path, 'config.json')) as f:
            config = json.load(f)
        
        bert_ensemble = cls(config['bert_1_path'],
                            config['bert_2_path'],
                            int(config['num_labels']))
        classifier = nn.Linear(bert_ensemble.classifier.in_features, bert_ensemble.classifier.out_features)
        classifier.load_state_dict(torch.load(os.path.join(load_dir_path, 'classifier.pt')))
        bert_ensemble.classifier = classifier.eval()
        return bert_ensemble