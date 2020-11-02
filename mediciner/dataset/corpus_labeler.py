from typing import List, Dict

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