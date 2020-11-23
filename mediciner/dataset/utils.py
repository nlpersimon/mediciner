from tokenizers import Encoding
from typing import List, Dict
import os
import pandas as pd
import re
from .corpus_labeler import tag_to_label




def read_corpus(path_to_corpus_dir: str) -> Dict[int, List[str]]:
    """ 從包含對話檔案的資料夾中把所有對話檔案讀進來。
    
    掃描 path_to_corpus_dir，得到所有檔案名稱之後，將這些檔案讀進來，並按照換行符號切開之後存進 dict。
    由於 article_id 就包含在檔名，所以從檔名中取出 article_id，作為 dict 的 key。

    Args:
        path_to_corpus_dir: 比如 data/dialogue_lined/train
    
    Returns:
        corpus: key = article_id, value = [內容1, 內容2, ...]
    """
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

def adjust_labels_by_encoding(encoding: Encoding, labels: List[int], subword_label: int, padding_label: int) -> List[int]:
    """ 將原始的 labels 依據 BertWordPieceTokenizer 輸出的 encoding 做調整（調至相同長度）。

    我們在製作 labels 時是以 "字元" 作為標記單位，但 tokenizer 遇到英文單字的時候可能會將它切成若干份 subword，
    比如 negative -> ne ##ga ##tive。這時它的原始 labels [0, 0, 0, 0, 0, 0, 0, 0] 就要調整成為 [0, 1, 1]，
    其中 0 代表 'O'，1 代表 'X'，也就是 outside tag 與 subword tag。
    假設 negative 對應的 labels 為 [2, 3, 3, 3, 3, 3, 3, 3]，那麼就要隨著 subwords 調整為 [2, 3, 3]。
    因此我們可以透過 encoding 取得 token 在原始輸入中的位置，進而取得這個 token 第一個字元的位置，這樣就能取得它的原始 label。
    取第一個字元對應的 label 是因為通常第一個字元對應到的 label 都是最重要的，比如上例中它對應到 entity 的 beginning。

    Args:
        encoding: BertWordPieceTokenizer 對原始輸入文字做斷詞後的輸出，它是一個特殊的資料結構，包含了 tokens, token 在輸入中的位置,
                  以及它用來輸入 Bert model 需要的 features (token ids and attention mask)
        labels: 原始輸入文字對應的 labels
        subword_label: subword 對應的 label

    Returns:
        adjusted_labels: 調整後的 labels，長度與 encoding 中的 token 數量相等，供模型學習
    """
    adjusted_labels = []
    for token, (start, end) in zip(encoding.tokens, encoding.offsets):
        if _is_special_token(token):
            continue
        if _is_subword(token) and labels[start] != padding_label:
            adjusted_labels.append(subword_label)
        else:
            adjusted_labels.append(labels[start])
    return adjusted_labels

def _is_special_token(token: str) -> bool:
    return token in ('[CLS]', '[SEP]')

def _is_subword(token: str) -> bool:
    return token[:2] == '##'