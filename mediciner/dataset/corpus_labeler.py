from typing import List, Dict, Tuple

class CorpusLabeler(object):
    def __init__(self, entity_types: List[str]) -> None:
        self.entity_types = entity_types
        self.tag_to_label = self.construct_tag_to_label()
    
    def construct_tag_to_label(self) -> Dict[str, int]:
        tag_to_label = {
            'O': 0,
            'X': 1,
        }
        label_begin, label_inner = 2, 3
        for ent_type in self.entity_types:
            tag_to_label[f'B-{ent_type}'] = label_begin
            tag_to_label[f'I-{ent_type}'] = label_inner
            label_begin = label_inner + 1
            label_inner = label_begin + 1
        return tag_to_label
    
    def label_corpus(self, corpus: str, entity_spans: List[Tuple[int, int, str]]) -> List[int]:
        corpus_tags = self.tag_corpus(corpus, entity_spans)
        corpus_labels = [self.tag_to_label[tag] for tag in corpus_tags]
        return corpus_labels
    
    def tag_corpus(self, corpus: str, entity_spans: List[Tuple[int, int, str]]) -> List[str]:
        corpus_tags = ['O'] * len(corpus)
        for begin, end, ent_type in entity_spans:
            tag_begin, tag_inner = f'B-{ent_type}', f'I-{ent_type}'
            assert tag_begin in self.tag_to_label and tag_inner in self.tag_to_label
            corpus_tags[begin] = tag_begin
            corpus_tags[(begin + 1):end] = [tag_inner] * (end - begin - 1)
        return corpus_tags