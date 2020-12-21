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