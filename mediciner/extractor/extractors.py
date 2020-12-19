from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import BertForTokenClassification
from tokenizers import Encoding, BertWordPieceTokenizer
import torch
from typing import List, Tuple
# from seqeval.metrics.sequence_labeling import get_entities
from .utils import get_entities
from .utils import encodings_to_features, translate_labels_to_tags, adjust_pred_tags, unpad_labels
from ..dataset.corpus_labeler import label_to_tag




@dataclass
class Entity:
    start: int
    end: int
    text: str
    type: str


class BaseExtractor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod 
    def extract_entities(self, texts: List[str]) -> List[List[Entity]]:
        pass


class BertExtractor(BaseExtractor):
    def __init__(self, bert_model: BertForTokenClassification,
                       tokenizer: BertWordPieceTokenizer,
                       max_input_len: int,
                       device: torch.device=torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.bert_model = bert_model
        self.bert_model.eval()
        self.bert_model.to(device)
        self.tokenizer = tokenizer
        self.padding_id = self.tokenizer.token_to_id('[PAD]')
        self.max_input_len = max_input_len
        
    def extract_entities(self, texts: List[str]) -> List[List[Entity]]:
        encodings = [self.tokenizer.encode(text) for text in texts]
        input_ids, attention_mask = encodings_to_features(encodings, self.padding_id, self.max_input_len)
        pred_labels_batch = self.predict_labels(input_ids.to(self.device), attention_mask.to(self.device))
        pred_tags_batch = [translate_labels_to_tags(pred_labels, len(text), encoding.offsets)
                           for pred_labels, text, encoding in zip(pred_labels_batch, texts, encodings)]
        pred_tags_batch = [adjust_pred_tags(pred_tags)
                           for pred_tags in pred_tags_batch]
        entities = [[Entity(start, end + 1, text[start:(end + 1)], ent_type) for ent_type, start, end in get_entities(pred_tags, True)]
                    for text, pred_tags in zip(texts, pred_tags_batch)]
        return entities

    def predict_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
        pred_labels = outputs['logits'].argmax(dim=2).to('cpu')
        torch.cuda.empty_cache()
        pred_labels = unpad_labels(pred_labels, attention_mask)
        return pred_labels