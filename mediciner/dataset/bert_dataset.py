import torch
import torch.nn as nn
from tokenizers import Encoding, BertWordPieceTokenizer
from typing import Dict, List, Tuple
import os
import pandas as pd
import re
from .corpus_labeler import label_corpus, tag_to_label
from .utils import read_corpus, read_ents_table, adjust_labels_by_encoding




class BertDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_corpus_dir: str,
                       path_to_ents_table: str,
                       tokenizer: BertWordPieceTokenizer,
                       max_input_len: int) -> None:
        self.corpus = read_corpus(path_to_corpus_dir)
        self.ents_table = read_ents_table(path_to_ents_table).set_index(['article_id', 'sentence_id'])
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.input_ids, self.attention_mask, self.labels = self.convert_corpus_to_features()
    
    def convert_corpus_to_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def add_features():
            padded_input_ids, padded_att_mask, padded_labels = self.pad_features(pack_input_ids,
                                                                                 pack_att_mask,
                                                                                 pack_labels)
            input_ids.append(padded_input_ids)
            attention_mask.append(padded_att_mask)
            labels.append(padded_labels)
            return

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for article_id, sentences in self.corpus.items():
            pack_input_ids: List[int] = []
            pack_att_mask: List[int] = []
            pack_labels: List[int] = []
            for sentence_id, sentence in enumerate(sentences):
                sent_input_ids, sent_att_mask, sent_labels = self.get_sentence_features(article_id, sentence_id, sentence)
                if (len(pack_input_ids) + len(sent_input_ids)) > self.max_input_len:
                    add_features()
                    pack_input_ids, pack_att_mask, pack_labels = [], [], []
                pack_input_ids.extend(sent_input_ids)
                pack_att_mask.extend(sent_att_mask)
                pack_labels.extend(sent_labels)
            if pack_input_ids:
                add_features()
        features = (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.float),
                    torch.tensor(labels, dtype=torch.long))
        return features

    def get_sentence_features(self, article_id: int, sentence_id: int, sentence: str) -> Tuple[List[int], List[int], List[int]]:
        encoding = self.tokenizer.encode(sentence)
        ent_spans = self.get_ent_spans(article_id, sentence_id)
        labels = label_corpus(sentence, ent_spans)
        adjusted_labels = adjust_labels_by_encoding(encoding, labels, tag_to_label('X'))
        return (encoding.ids[1:-1][:self.max_input_len],
                encoding.attention_mask[1:-1][:self.max_input_len],
                adjusted_labels[:self.max_input_len])
    
    def get_ent_spans(self, article_id: int, sentence_id: int) -> List:
        try:
            ent_spans = [(row.start_position, row.end_position, row.entity_type)
                         for row in self.ents_table.loc[(article_id, sentence_id)].itertuples()]
        except KeyError:
            ent_spans = []
        return ent_spans
    
    def pad_features(self,
                     input_ids: List[int],
                     attention_mask: List[int],
                     labels: List[int]) -> Tuple[List[int], List[int], List[int]]:
        padding_len = self.max_input_len - len(input_ids)
        cls_id, sep_id, pad_id = [self.tokenizer.token_to_id(token) for token in ('[CLS]', '[SEP]', '[PAD]')]
        special_token_mask, special_token_label = 1, tag_to_label('O')
        pad_mask, pad_label = 0, tag_to_label('O')
        padded_input_ids = [cls_id] + input_ids + [sep_id] + [pad_id] * padding_len 
        padded_att_mask = [special_token_mask] + attention_mask + [special_token_mask] + [pad_mask] * padding_len
        padded_labels = [special_token_label] + labels + [special_token_label] + [pad_label] * padding_len
        return (padded_input_ids, padded_att_mask, padded_labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.labels[idx])

    def __len__(self):
        return len(self.input_ids)