import torch
import torch.nn as nn
from tokenizers import Encoding
from typing import Dict, List, Tuple
import os
import pandas as pd
import re



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
    def __init__(self, path_to_corpus_dir: str, path_to_ents_table: str) -> None:
        self.corpus = read_corpus(path_to_corpus_dir)
        self.ents_table = read_ents_table(path_to_ents_table)
        self.input_ids, self.attention_mask, self.labels = self.convert_corpus_to_features()
    
    def convert_corpus_to_features(self) -> Tuple[torch.Tensor]:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError