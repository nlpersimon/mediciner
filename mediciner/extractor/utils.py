from collections import Counter
from tokenizers import Encoding
import torch
from typing import List, Tuple
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

def get_entities(tags: List[str], strict=False) -> List[Tuple[str, int, int]]:
    entities, type_stack = [], []
    start, end = 0, 0
    prev_tag, b_exist = 'O', False
    tags = tags + ['O']
    for idx, tag in enumerate(tags):
        if is_end_of_prev_chunk(tag) and type_stack:
            ent_type = get_argmax_type(type_stack)
            entities.append((ent_type, start, end))
            type_stack, b_exist = [], False
        if is_start_of_next_chunk(prev_tag, tag, strict):
            start = idx
            b_exist = True
            type_stack.append(tag[2:])
        elif is_inside_of_next_chunk(tag, b_exist):
            type_stack.append(tag[2:])
        end, prev_tag = idx, tag
    return entities

def is_end_of_prev_chunk(tag: str) -> bool:
    return tag == 'O' or tag[0] == 'B'

def get_argmax_type(type_stack: List[str]) -> str:
    type_count = Counter(type_stack)
    ent_type = max(type_count, key=lambda t: type_count[t])
    return ent_type

def is_start_of_next_chunk(previous_tag, current_tag, strict):
    inside_as_start = previous_tag[0] == 'O' and current_tag[0] == 'I'
    return (inside_as_start and not strict) or (current_tag[0] == 'B')

def is_inside_of_next_chunk(current_tag, b_exist):
    return current_tag[0] == 'I' and b_exist