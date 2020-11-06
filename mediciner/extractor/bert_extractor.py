from dataclasses import dataclass
from transformers import BertForTokenClassification
from tokenizers import Encoding, BertWordPieceTokenizer
import torch
from typing import List, Tuple
from seqeval.metrics.sequence_labeling import get_entities
from ..dataset.corpus_labeler import label_to_tag



def encodings_to_features(encodings: List[Encoding],
                          padding_id: int,
                          max_input_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids_list, attention_mask_list = [], []
    for encoding in encodings:
        clipped_input_ids = encoding.ids[:max_input_len]
        clipped_att_mask = encoding.attention_mask[:max_input_len]
        padding_len = max_input_len - len(clipped_input_ids)
        padded_input_ids = clipped_input_ids + [padding_id] * padding_len
        padded_att_mask = clipped_att_mask + [0] * padding_len
        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_att_mask)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.float)
    return (input_ids, attention_mask)

def unpad_labels(labels: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    unpadded_labels = [example_labels[example_att == 1]
                       for example_labels, example_att in zip(labels, attention_mask)]
    return unpadded_labels

def translate_labels_to_tags(pred_labels: torch.Tensor,
                             input_len: int,
                             token_spans: List[Tuple[int, int]]) -> List[str]:
    input_tags = ['O'] * input_len
    for (start, end), label in zip(token_spans, pred_labels):
        input_tags[start:end] = expand_tag(label_to_tag(int(label)), end - start)
    return input_tags

def expand_tag(tag: str, n_char: int) -> List[str]:
    if tag[0] == 'B':
        _, ent_type = tag.split('-')
        expanded_tags = [tag] + ([f'I-{ent_type}'] * (n_char - 1))
    else:
        expanded_tags = [tag] * n_char
    return expanded_tags

def adjust_pred_tags(pred_tags: List[str]) -> List[str]:
    adjusted_tags, prev_type = [], ''
    for tag in pred_tags:
        if tag == 'X':
            tag = f'I-{prev_type}' if prev_type else 'O'
        prev_type = tag[2:]
        adjusted_tags.append(tag)
    return adjusted_tags


@dataclass
class Entity:
    start: int
    end: int
    text: str
    type: str


class BertExtractor(object):
    def __init__(self, bert_model: BertForTokenClassification,
                       tokenizer: BertWordPieceTokenizer,
                       max_input_len: int) -> None:
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.padding_id = self.tokenizer.token_to_id('[PAD]')
        self.max_input_len = max_input_len
        
    def extract_entities(self, texts: List[str]) -> List[List[Entity]]:
        encodings = [self.tokenizer.encode(text) for text in texts]
        input_ids, attention_mask = encodings_to_features(encodings, self.padding_id, self.max_input_len)
        pred_labels_batch = self.predict_labels(input_ids, attention_mask)
        pred_tags_batch = [translate_labels_to_tags(pred_labels, len(text), encoding.offsets)
                           for pred_labels, text, encoding in zip(pred_labels_batch, texts, encodings)]
        pred_tags_batch = [adjust_pred_tags(pred_tags)
                           for pred_tags in pred_tags_batch]
        entities = [[Entity(start, end + 1, text[start:(end + 1)], ent_type) for ent_type, start, end in get_entities(pred_tags)]
                    for text, pred_tags in zip(texts, pred_tags_batch)]
        return entities

    def predict_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.bert_model(input_ids, attention_mask)
        pred_labels = outputs['logits'].argmax(dim=2)
        pred_labels = unpad_labels(pred_labels, attention_mask)
        return pred_labels