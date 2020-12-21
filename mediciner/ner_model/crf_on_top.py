from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torchcrf import CRF


class BertWithCRF(BertPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_wise_ff = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 return_dict=True)
        last_hidden_state = self.dropout(bert_outputs['last_hidden_state'])
        outputs = {}
        emissions = self.position_wise_ff(last_hidden_state)
        if labels is not None:
            loss = self.crf(emissions, labels, attention_mask.byte(), 'token_mean')
            outputs['loss'] = -1 * loss
        outputs['emissions'] = emissions
        return outputs