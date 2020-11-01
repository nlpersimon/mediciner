import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import pandas as pd



def read_corpus(path_to_corpus_dir: str) -> Dict[int, List[str]]:
    raise NotImplementedError

def read_ents_table(path_to_ents_table: str) -> pd.DataFrame:
    raise NotImplementedError


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