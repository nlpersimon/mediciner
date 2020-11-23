from dataclasses import dataclass
from transformers import BertForTokenClassification
from tokenizers import Encoding, BertWordPieceTokenizer
import torch
from typing import List, Tuple
from seqeval.metrics.sequence_labeling import get_entities
from .utils import encodings_to_features, translate_labels_to_tags, adjust_pred_tags, unpad_labels
from ..dataset.corpus_labeler import label_to_tag
from ..dataset.processor import Example




@dataclass
class Entity:
    start: int
    end: int
    text: str
    type: str


class BertExtractor(object):
    def __init__(self, bert_model: BertForTokenClassification,
                       tokenizer: BertWordPieceTokenizer,
                       max_input_len: int,
                       device: torch.device=torch.device('cpu')) -> None:
        self.device = device
        self.bert_model = bert_model
        self.bert_model.eval()
        self.bert_model.to(device)
        self.tokenizer = tokenizer
        self.padding_id = self.tokenizer.token_to_id('[PAD]')
        self.max_input_len = max_input_len
        
    def extract_entities(self, examples: List[Example]) -> List[List[Entity]]:
        encodings = [self.tokenizer.encode(example.content) for example in examples]
        input_ids, attention_mask = encodings_to_features(encodings, self.padding_id, self.max_input_len)
        pred_labels_batch = self.predict_labels(input_ids.to(self.device), attention_mask.to(self.device))
        pred_tags_batch = []
        for pred_labels, example, encoding in zip(pred_labels_batch, examples, encodings):
            center_start, center_end = example.center_span
            pred_tags = translate_labels_to_tags(pred_labels, len(example.content), encoding.offsets)[center_start:center_end]
            pred_tags_batch.append(adjust_pred_tags(pred_tags))
        entities = []
        for example, pred_tags in zip(examples, pred_tags_batch):
            center_start, center_end = example.center_span
            center_sentence = example.content[center_start:center_end]
            example_ents = []
            for ent_type, start, end in get_entities(pred_tags):
                example_ents.append(Entity(start, end + 1, center_sentence[start:(end + 1)], ent_type))
            entities.append(example_ents)
        return entities

    def predict_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
        pred_labels = outputs['logits'].argmax(dim=2).to('cpu')
        torch.cuda.empty_cache()
        pred_labels = unpad_labels(pred_labels, attention_mask)
        return pred_labels