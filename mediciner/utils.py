from dataclasses import dataclass
from tokenizers import BertWordPieceTokenizer
from typing import List



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