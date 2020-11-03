from typing import List, Dict, Tuple


def construct_TAG_TO_LABEL(entity_types) -> Dict[str, int]:
    TAG_TO_LABEL = {
        'O': 0,
        'X': 1,
    }
    label_begin, label_inner = 2, 3
    for ent_type in entity_types:
        TAG_TO_LABEL[f'B-{ent_type}'] = label_begin
        TAG_TO_LABEL[f'I-{ent_type}'] = label_inner
        label_begin = label_inner + 1
        label_inner = label_begin + 1
    return TAG_TO_LABEL

ENTITY_TYPES = (
    'time',
    'med_exam',
    'profession',
    'name',
    'location',
    'family',
    'ID',
    'clinical_event',
    'education',
    'money',
    'contact',
    'organization'
)
TAG_TO_LABEL = construct_TAG_TO_LABEL(ENTITY_TYPES)
LABEL_TO_TAG = {label: tag for tag, label in TAG_TO_LABEL.items()}

def tag_to_label(tag: str) -> int:
    return TAG_TO_LABEL[tag]

def label_to_tag(label: int) -> str:
    return LABEL_TO_TAG[label]

def label_corpus(corpus: str, entity_spans: List[Tuple[int, int, str]]) -> List[int]:
    corpus_tags = tag_corpus(corpus, entity_spans)
    corpus_labels = [tag_to_label(tag) for tag in corpus_tags]
    return corpus_labels

def tag_corpus(corpus: str, entity_spans: List[Tuple[int, int, str]]) -> List[str]:
    corpus_tags = ['O'] * len(corpus)
    for begin, end, ent_type in entity_spans:
        tag_begin, tag_inner = f'B-{ent_type}', f'I-{ent_type}'
        assert tag_begin in TAG_TO_LABEL and tag_inner in TAG_TO_LABEL
        corpus_tags[begin] = tag_begin
        corpus_tags[(begin + 1):end] = [tag_inner] * (end - begin - 1)
    return corpus_tags