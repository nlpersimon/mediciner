from dataclasses import dataclass
import pandas as pd
from tokenizers import BertWordPieceTokenizer
import tqdm
from typing import List, Dict
from .extractor.bert_extractor import BertExtractor, Entity
from .dataset.processor import BertProcessor, Example



@dataclass
class Paragraph:
    start: int
    end: int
    content: str


def sentences_to_paragraphs(sentences: List[str],
                            max_paragraph_len: int,
                            tokenizer: BertWordPieceTokenizer) -> List[Paragraph]:
    parag_start, parag_end, paragraphs = 0, 0, []
    hold_parag, hold_parag_len = '', 0
    for sentence in sentences:
        encoding = tokenizer.encode(sentence)
        # [1:-1] to exclude [CLS] and [SEP]
        clipped_offsets = encoding.offsets[1:-1][:max_paragraph_len]
        (sent_start, _), (_, sent_end) = clipped_offsets[0], clipped_offsets[-1]
        clipped_sentence = sentence[sent_start:sent_end]
        n_tokens = len(clipped_offsets)
        if (hold_parag_len + n_tokens) > max_paragraph_len:
            paragraphs.append(Paragraph(parag_start, parag_end, hold_parag))
            hold_parag, hold_parag_len = '', 0
            parag_start = parag_end
        hold_parag, hold_parag_len = hold_parag + clipped_sentence, hold_parag_len + n_tokens
        parag_end += len(sentence)
    if hold_parag:
        paragraphs.append(Paragraph(parag_start, parag_end, hold_parag))
    return paragraphs

def build_ents_table(corpus: Dict[int, List[str]],
                     processor: BertProcessor,
                     bert_extractor: BertExtractor,
                     batch_size: int=32) -> pd.DataFrame:
    examples = processor.convert_corpus_to_examples(corpus)
    packed_examples = [examples[i:(i + batch_size)] for i in range(0, len(examples), batch_size)]
    ents_matrix = []
    for pack in tqdm.tqdm(packed_examples, desc='extract entities from the corpus'):
        example_ents = extract_entities(pack, bert_extractor)
        for example, ents in zip(pack, example_ents):
            _, article_id = example.id.split('-')
            ent_rows = [[int(article_id), ent.start, ent.end, ent.text, ent.type]
                        for ent in ents]
            ents_matrix.extend(ent_rows)
    ents_table = pd.DataFrame(ents_matrix,
                              columns=['article_id', 'start_position', 'end_position', 'entity_text', 'entity_type'])
    return ents_table

def extract_entities(examples: List[Example], bert_extractor: BertExtractor) -> List[List[Entity]]:
    example_entities = bert_extractor.extract_entities(examples)
    entities = [[Entity(ent.start + example.start, ent.end + example.start, ent.text, ent.type)
                 for ent in ents]
                for example, ents in zip(examples, example_entities)]
    return entities