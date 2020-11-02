import torch
import torch.nn as nn
from tokenizers import Encoding, BertWordPieceTokenizer
from typing import Dict, List, Tuple
import os
import pandas as pd
import re
from .corpus_labeler import CorpusLabeler



def read_corpus(path_to_corpus_dir: str) -> Dict[int, List[str]]:
    files = filter(lambda fn: fn[-4:] == '.txt', os.listdir(path_to_corpus_dir))
    corpus = {}
    for file in files:
        article_id = int(re.findall('-(\d+).', file)[0])
        with open(os.path.join(path_to_corpus_dir, file), 'r') as f:
            corpus[article_id] = f.read().splitlines()
    return corpus

def read_ents_table(path_to_ents_table: str) -> pd.DataFrame:
    ents_table = pd.read_csv(path_to_ents_table)
    return ents_table

def adjust_labels_by_encoding(encoding: Encoding, labels: List[int], subword_label: int) -> List[int]:
    adjusted_labels = []
    for token, (start, end) in zip(encoding.tokens, encoding.offsets):
        if _is_special_token(token):
            continue
        label = subword_label if _is_subword(token) else labels[start]
        adjusted_labels.append(label)
    return adjusted_labels

def _is_special_token(token: str) -> bool:
    return token in ('[CLS]', '[SEP]')

def _is_subword(token: str) -> bool:
    return token[:2] == '##'


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_corpus_dir: str,
                       path_to_ents_table: str,
                       tokenizer: BertWordPieceTokenizer,
                       corpus_labeler: CorpusLabeler,
                       max_input_len: int) -> None:
        self.corpus = read_corpus(path_to_corpus_dir)
        self.ents_table = read_ents_table(path_to_ents_table).set_index(['article_id', 'sentence_id'])
        self.tokenizer = tokenizer
        self.corpus_labeler = corpus_labeler
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
                if (len(pack_input_ids) + len(sent_input_ids)) >= self.max_input_len:
                    add_features()
                    pack_input_ids, pack_att_mask, pack_labels = [], [], []
                pack_input_ids.extend(sent_input_ids)
                pack_att_mask.extend(sent_att_mask)
                pack_labels.extend(sent_labels)
            if pack_input_ids:
                add_features()
        features = (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.uint8),
                    torch.tensor(labels, dtype=torch.long))
        return features

    def get_sentence_features(self, article_id: int, sentence_id: int, sentence: str) -> Tuple[List[int], List[int], List[int]]:
        encoding = self.tokenizer.encode(sentence)
        ent_spans = self.get_ent_spans(article_id, sentence_id)
        labels = self.corpus_labeler.label_corpus(sentence, ent_spans)
        adjusted_labels = adjust_labels_by_encoding(encoding, labels, 1)
        return (encoding.ids[1:-1][:self.max_input_len],
                encoding.attention_mask[1:-1][:self.max_input_len],
                adjusted_labels[:self.max_input_len])
    
    def get_ent_spans(self, article_id: int, sentence_id: int) -> List:
        try:
            ent_spans = [(row.start_position, row.end_position, row.entity_type)
                         for row in self.ents_table.loc[(article_id, sentence_id)].itertuples()]
        except:
            ent_spans = []
        return ent_spans
    
    def pad_features(self,
                     input_ids: List[int],
                     attention_mask: List[int],
                     labels: List[int]) -> Tuple[List[int], List[int], List[int]]:
        padding_len = self.max_input_len - len(input_ids)
        cls_id, sep_id, pad_id = [self.tokenizer.token_to_id(token) for token in ('[CLS]', '[SEP]', '[PAD]')]
        outside_id = self.corpus_labeler.tag_to_label['O']
        padded_input_ids = [cls_id] + input_ids + [sep_id] + [pad_id] * padding_len 
        padded_att_mask = [1] + attention_mask + [1] + [0] * padding_len
        padded_labels = [outside_id] + labels + [outside_id] + [outside_id] * padding_len
        return (padded_input_ids, padded_att_mask, padded_labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.labels[idx])

    def __len__(self):
        return len(self.input_ids)