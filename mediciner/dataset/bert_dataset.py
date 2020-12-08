import torch
from typing import Tuple
from .corpus_labeler import tag_to_label
from .utils import read_corpus, read_ents_table
from .processor import BertProcessor




class BertDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_corpus_dir: str,
                       path_to_ents_table: str,
                       processor: BertProcessor,
                       set_type: str) -> None:
        self.corpus = read_corpus(path_to_corpus_dir)
        self.ents_table = None
        if path_to_ents_table:
            self.ents_table = read_ents_table(path_to_ents_table).set_index(['article_id', 'sentence_id'])
        self.processor = processor
        self.set_type = set_type
        self.input_ids, self.attention_mask, self.labels = self.convert_corpus_to_feature_tensors()

    def convert_corpus_to_feature_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.processor.convert_corpus_to_features(self.corpus, self.ents_table, self.set_type)
        input_ids, attention_mask, labels = [], [], []
        max_input_len = self.processor.max_input_len + 2  # +2 for [CLS] and [SEP]
        pad_id, pad_mask, pad_label = self.processor.tokenizer.token_to_id('[PAD]'), 0, tag_to_label('O')
        for feature in features:
            padding_len = max_input_len - len(feature.input_ids)
            input_ids.append(feature.input_ids + [pad_id] * padding_len)
            attention_mask.append(feature.attention_mask + [pad_mask] * padding_len)
            if feature.labels is not None:
                labels.append(feature.labels + [pad_label] * padding_len)
            else:
                labels.append([pad_label] * max_input_len)
        feature_tensors = (torch.tensor(input_ids, dtype=torch.long),
                           torch.tensor(attention_mask, dtype=torch.float),
                           torch.tensor(labels, dtype=torch.long))
        return feature_tensors

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.labels[idx])

    def __len__(self):
        return len(self.input_ids)


class BertWithMRCDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_corpus_dir: str,
                       path_to_ents_table: str,
                       processor: BertProcessor,
                       set_type: str) -> None:
        self.corpus = read_corpus(path_to_corpus_dir)
        self.ents_table = None
        if path_to_ents_table:
            self.ents_table = read_ents_table(path_to_ents_table).set_index(['article_id', 'sentence_id'])
        self.processor = processor
        self.processor.set_type = set_type
        self.set_type = set_type
        self.input_ids, self.token_type_ids, self.attention_mask, self.start_tensor, self.end_tensor, self.entype_ids = self.convert_corpus_to_feature_tensors()

    def convert_corpus_to_feature_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.processor.convert_corpus_to_features(self.corpus, self.ents_table, self.set_type)
        input_ids, attention_mask, token_type_ids, start_tensor, end_tensor, entype_ids = [], [], [], [], [], []
        max_input_len = self.processor.max_input_len + 2  # +2 for [CLS] and [SEP]
        pad_id, pad_mask, pad_label = self.processor.tokenizer.token_to_id('[PAD]'), 0, tag_to_label('O')
        for feature in features:
            padding_len = max_input_len - len(feature.input_ids)
            input_ids.append(feature.input_ids + [pad_id] * padding_len)
            attention_mask.append(feature.attention_mask + [pad_mask] * padding_len)
            token_type_ids.append(feature.token_type_ids + [1] * padding_len)
            if feature.labels is not None:
                labels = feature.labels + [pad_label] * padding_len
                start_tensor.append([1 if label == 1 else 0 for label in labels])
                end_tensor.append([1 if label == 2 else 0 for label in labels])
            else:
                start_tensor.append([0] * max_input_len)
                end_tensor.append([0] * max_input_len)
            entype_ids.append(feature.entype_id)
        feature_tensors = (torch.tensor(input_ids, dtype=torch.long),
                           torch.tensor(token_type_ids, dtype=torch.long),
                           torch.tensor(attention_mask, dtype=torch.float),
                           torch.tensor(start_tensor, dtype=torch.long),
                           torch.tensor(end_tensor, dtype=torch.long),
                           torch.tensor(entype_ids, dtype=torch.long))
        return feature_tensors

    def __getitem__(self, idx):
        return (self.input_ids[idx],
                self.token_type_ids[idx],
                self.attention_mask[idx],
                self.start_tensor[idx],
                self.end_tensor[idx],
                self.entype_ids[idx])

    def __len__(self):
        return len(self.input_ids)