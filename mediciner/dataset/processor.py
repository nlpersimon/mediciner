from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import accumulate
from tokenizers import BertWordPieceTokenizer
import tqdm
from typing import List, Union, Tuple, Dict, Generator
import pandas as pd
from .corpus_labeler import label_corpus, tag_to_label
from .utils import adjust_labels_by_encoding



@dataclass
class Example:
    id: str
    content: str
    center_span: Tuple[int, int]
    start: int=-1
    end: int=-1
    labels: Union[None, List[int]]=None


@dataclass
class Feature:
    id: str
    input_ids: List[int]
    attention_mask: List[int]
    labels: Union[None, List[int]]=None


def get_ent_spans(ents_table, article_id: int, sentence_id: int) -> List:
        try:
            ent_spans = [(row.start_position, row.end_position, row.entity_type)
                         for row in ents_table.loc[(article_id, sentence_id)].itertuples()]
        except KeyError:
            ent_spans = []
        return ent_spans


class BertProcessor(ABC):
    def __init__(self, max_input_len: int, tokenizer: BertWordPieceTokenizer) -> None:
        self.max_input_len = max_input_len
        self.tokenizer = tokenizer

    def convert_corpus_to_features(self, corpus: Dict[int, List[str]],
                                   ents_table: Union[pd.DataFrame, None]=None,
                                   set_type: str='default') -> List[Feature]:
        examples = self.convert_corpus_to_examples(corpus, ents_table, set_type)
        features = [self.convert_example_to_feature(example)
                    for example in tqdm.tqdm(examples, desc=f'convert {set_type} examples to {set_type} features')]
        return features
    
    def convert_corpus_to_examples(self, corpus: Dict[int, List[str]],
                                   ents_table: Union[pd.DataFrame, None]=None,
                                   set_type: str='default') -> List[Example]:
        examples = []
        for article_id, article in tqdm.tqdm(corpus.items(),
                                             desc=f'convert {set_type} corpus to {set_type} examples'):
            dataset_id = f'{set_type}-{article_id}'
            article_examples = self.convert_article_to_examples(article, dataset_id, ents_table)
            examples.extend(article_examples)
        return examples
    
    def convert_example_to_feature(self, example: Example) -> Feature:
        encoding = self.tokenizer.encode(example.content)
        labels = None
        if example.labels is not None:
            labels = adjust_labels_by_encoding(encoding, example.labels, tag_to_label('X'), tag_to_label('P'))
            # add the labels for [CLS] and [SEP]
            labels = [tag_to_label('O')] + labels + [tag_to_label('O')]
        assert len(encoding.ids) == len(encoding.attention_mask)
        if labels is not None:
            assert len(encoding.ids) == len(labels)
        feature = Feature(example.id, encoding.ids, encoding.attention_mask, labels)
        return feature
    
    @abstractmethod
    def convert_article_to_examples(self, article: List[str],
                                    dataset_id: str,
                                    ents_table: Union[pd.DataFrame, None]=None) -> List[Example]:
        pass
    
    def get_clipped_offsets(self, sentence: str) -> List[Tuple[int, int]]:
        encoding = self.tokenizer.encode(sentence)
        clipped_offsets = encoding.offsets[1:-1][:self.max_input_len]
        return clipped_offsets



class BertSentProcessor(BertProcessor):
    def __init__(self, max_input_len: int, tokenizer: BertWordPieceTokenizer) -> None:
        super().__init__(max_input_len, tokenizer)
    
    def convert_article_to_examples(self, article: List[str],
                                    dataset_id: str,
                                    ents_table: Union[pd.DataFrame, None]=None) -> List[Example]:
        sentences_with_id = [(sent_id, sent) for sent_id, sent in enumerate(article)]
        start, examples = 0, []
        _, article_id = dataset_id.split('-')
        while sentences_with_id:
            sentences, clipped_sent_spans = self.collect_sentences_and_clip_spans(sentences_with_id)
            clipped_sents_labels = None
            if ents_table is not None:
                clipped_sents_labels = []
                for (sent_id, sent), (sent_start, sent_end) in zip(sentences, clipped_sent_spans):
                    ent_spans = get_ent_spans(ents_table, int(article_id), sent_id)
                    clipped_sents_labels +=  label_corpus(sent, ent_spans)[sent_start:sent_end]
            clipped_sentences = ''.join([sent[sent_start:sent_end]
                                         for (_, sent), (sent_start, sent_end) in zip(sentences, clipped_sent_spans)])
            end = start + sum(len(sent) for _, sent in sentences)
            center_span = (0, len(clipped_sentences))
            examples.append(Example(dataset_id, clipped_sentences, center_span, start, end, clipped_sents_labels))
            start = end
        return examples
    
    @abstractmethod
    def collect_sentences_and_clip_spans(self, sentences_with_id: List[Tuple[int, str]]) -> Tuple[List[Tuple[int, str]],
                                                                                                  List[Tuple[int, int]]]:
        pass


class BertMultiSentProcessor(BertSentProcessor):
    def __init__(self, max_input_len: int, tokenizer: BertWordPieceTokenizer) -> None:
        super().__init__(max_input_len, tokenizer)
    
    def collect_sentences_and_clip_spans(self, sentences_with_id: List[Tuple[int, str]]) -> Tuple[List[Tuple[int, str]],
                                                                                                  List[Tuple[int, int]]]:
        hold_len, sentences, clip_spans = 0, [], []
        while hold_len < self.max_input_len and sentences_with_id:
            sent_id, sentence = sentences_with_id[0]
            clipped_offsets = self.get_clipped_offsets(sentence)
            n_tokens = len(clipped_offsets)
            if (n_tokens + hold_len) > self.max_input_len:
                break
            sentences_with_id.pop(0)
            (clip_start, _), (_, clip_end) = clipped_offsets[0], clipped_offsets[-1]
            sentences.append((sent_id, sentence))
            clip_spans.append((clip_start, clip_end))
            hold_len += n_tokens
        return (sentences, clip_spans)


class BertUniSentProcessor(BertSentProcessor):
    def __init__(self, max_input_len: int, tokenizer: BertWordPieceTokenizer) -> None:
        super().__init__(max_input_len, tokenizer)

    def collect_sentences_and_clip_spans(self, sentences_with_id: List[Tuple[int, str]]) -> Tuple[List[Tuple[int, str]],
                                                                                                  List[Tuple[int, int]]]:
        sent_id, sentence = sentences_with_id[0]
        clipped_offsets = self.get_clipped_offsets(sentence)
        (clip_start, _), (_, clip_end) = clipped_offsets[0], clipped_offsets[-1]
        sentences_with_id.pop(0)
        return ([(sent_id, sentence)], [(clip_start, clip_end)])


class BertWindowProcessor(BertProcessor):
    def __init__(self, max_input_len: int, window_size: int, tokenizer: BertWordPieceTokenizer) -> None:
        super().__init__(max_input_len, tokenizer)
        self.window_size = window_size
    
    def convert_article_to_examples(self, article: List[str],
                                    dataset_id: str,
                                    ents_table: Union[pd.DataFrame, None]=None) -> List[Example]:
        _, article_id = dataset_id.split('-')
        windows = self.generate_windows(article)
        padding_label = tag_to_label('P')
        examples = []
        for center_sent_id, ((start, end), (left_context, center_sent, right_context)) in enumerate(windows):
            clipped_sent_labels = None
            window_text = left_context + center_sent + right_context
            (clipped_start, _), *_, (_, clipped_end) = self.get_clipped_offsets(window_text)
            if ents_table is not None:
                left_labels, right_labels = [padding_label] * len(left_context), [padding_label] * len(right_context)
                ent_spans = get_ent_spans(ents_table, int(article_id), center_sent_id)
                center_labels = label_corpus(center_sent, ent_spans)
                labels = left_labels + center_labels + right_labels
                clipped_sent_labels = labels[clipped_start:clipped_end]
            clipped_window_text = window_text[clipped_start:clipped_end]

            original_center_start = len(left_context)
            original_center_end = original_center_start + len(center_sent)
            center_start, center_end = min(original_center_start, clipped_end), min(original_center_end, clipped_end)
            center_span = (center_start, center_end)

            examples.append(Example(dataset_id, clipped_window_text, center_span, start, end, clipped_sent_labels)) 
        return examples

    def generate_windows(self, article: List[str]) -> Generator[Tuple[Tuple[int, int], Tuple[str, str, str]],
                                                                None,
                                                                None]:
        sent_starts = list(accumulate([0] + [len(sent) for sent in article[:-1]]))
        for center_idx, center_sent in enumerate(article):
            left_bound = max(0, center_idx - self.window_size)
            right_bound = center_idx + self.window_size + 1
            left_context = ''.join(article[left_bound:center_idx])
            right_context = ''.join(article[(center_idx + 1):right_bound])
            window = (left_context, center_sent, right_context)
            start = sent_starts[center_idx]
            end = start + len(center_sent)
            original_span = (start, end)
            yield (original_span, window)