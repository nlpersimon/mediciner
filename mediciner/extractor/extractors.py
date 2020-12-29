from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import BertForTokenClassification
from tokenizers import Encoding, BertWordPieceTokenizer
import torch
from typing import List, Tuple
import re
# from seqeval.metrics.sequence_labeling import get_entities
from .utils import get_entities
from .utils import encodings_to_features, translate_labels_to_tags, adjust_pred_tags, unpad_labels
from ..dataset.corpus_labeler import label_to_tag
from ..ner_model import BertWithCRF




@dataclass
class Entity:
    start: int
    end: int
    text: str
    type: str


class BaseExtractor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod 
    def extract_entities(self, texts: List[str]) -> List[List[Entity]]:
        pass


class BertExtractor(BaseExtractor):
    def __init__(self, bert_model: BertForTokenClassification,
                       tokenizer: BertWordPieceTokenizer,
                       max_input_len: int,
                       device: torch.device=torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.bert_model = bert_model
        self.bert_model.eval()
        self.bert_model.to(device)
        self.tokenizer = tokenizer
        self.padding_id = self.tokenizer.token_to_id('[PAD]')
        self.max_input_len = max_input_len
        
    def extract_entities(self, texts: List[str]) -> List[List[Entity]]:
        encodings = [self.tokenizer.encode(text) for text in texts]
        input_ids, attention_mask = encodings_to_features(encodings, self.padding_id, self.max_input_len)
        pred_labels_batch = self.predict_labels(input_ids.to(self.device), attention_mask.to(self.device))
        pred_tags_batch = [translate_labels_to_tags(pred_labels, len(text), encoding.offsets)
                           for pred_labels, text, encoding in zip(pred_labels_batch, texts, encodings)]
        pred_tags_batch = [adjust_pred_tags(pred_tags)
                           for pred_tags in pred_tags_batch]
        entities = [[Entity(start, end + 1, text[start:(end + 1)], ent_type) for ent_type, start, end in get_entities(pred_tags, True)]
                    for text, pred_tags in zip(texts, pred_tags_batch)]
        return entities

    def predict_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
        pred_labels = outputs['logits'].argmax(dim=2).to('cpu')
        torch.cuda.empty_cache()
        pred_labels = unpad_labels(pred_labels, attention_mask)
        return pred_labels


class BertWithCRFExtractor(BertExtractor):
    def __init__(self, bert_model: BertWithCRF,
                       tokenizer: BertWordPieceTokenizer,
                       max_input_len: int,
                       device: torch.device=torch.device('cpu')) -> None:
        super().__init__(bert_model, tokenizer, max_input_len, device)
    
    def predict_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask)
            pred_labels = self.bert_model.crf.decode(outputs['emissions'], attention_mask.byte())
        torch.cuda.empty_cache()
        pred_labels = [torch.tensor(labels) for labels in pred_labels]
        return pred_labels


def strQ2B(s):
    """把字串全形轉半形"""
    rstring = ""
    for uchar in s:
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        rstring += chr(u_code)
    return rstring


class RuleExtractor(BaseExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.clinical_events = {'h1n1', 'sars', 'covid-19', '武漢肺炎', '新冠', '新冠肺炎', '八仙塵爆', '高雄氣爆'}
        self.clinical_events_pattern = '|'.join(self.clinical_events)
        self.professions = {'房務員', '麵包店', '設計業', '餐飲業', '替代役', '工程師'}
        self.prof_pattern = '|'.join(self.professions)
        self.organizations = {'CDC', 'ＣＤＣ'}
        self.org_pattern = '|'.join(self.organizations)
        self.med_pattern = '[\d零一二三四五六七八九]+[.．][多幾]|[\d零一二三四五六七八九０１２３４５６７８９]+[.．][\d零一二三四五六七八九０１２３４５６７８９]'
        self.contact_list = {'line', '臉書', 'ｌｉｎｅ', 'grindr', 'hornet', 'tinder', 'ig','instagram', 'badboyisme', 'www.prep.gov', 'e?-?mail', '09\d{8}'}
        self.contact_pattern = '|'.join(self.contact_list)
        self.time_pattern = '|'.join([
            '中秋節?',
            '潑水節?',
            '清明節?',
            '重陽節?',
            '聖誕節?',
            '耶誕節?',
            '七夕',
            '兒童節',
            '雙十節',
            '國慶日?',
            '除夕',
            '中元節?',
            '冬至']
        )
    
    def extract_entities(self, texts: List[str]) -> List[List[Entity]]:
        entities = []
        for text in texts:
            text_ents = []
            text_ents += self.extract_ID_ents(text)
            text_ents += self.extract_clinical_event_ents(text)
            text_ents += self.extract_profession_ents(text)
            text_ents += self.extract_education_ents(text)
            entities.append(text_ents)
        return entities
    
    def extract_time_ents(self, text: str) -> List[Entity]:
        time_ents = []
        for match in re.finditer(self.time_pattern, strQ2B(text.lower())):
            start, end = match.span()
            ent_text = text[start:end]
            if '/' not in ent_text:
                time_ents.append(Entity(start, end, ent_text, 'time'))
            else:
                month, day = ent_text.split('/')
                if (1 <= int(month) <= 12) and (1 <= int(day) <= 31):
                    time_ents.append(Entity(start, end, ent_text, 'time'))
        return time_ents
    
    def extract_ID_ents(self, text: str) -> List[Entity]:
        ID_ents = []
        for match in re.finditer('[a-zA-Z]\d{9,10}|第\d號|病例(號碼)?\d+', text):
            start, end = match.start(), match.end()
            ent_text = text[start:end]
            if ent_text[:4] == '病例號碼':
                start += 4
            elif ent_text[:2] == '病例':
                start += 2
            ID_ents.append(Entity(start, end, text[start:end], 'ID'))
        return ID_ents

    def extract_clinical_event_ents(self, text: str) -> List[Entity]:
        clinical_event_ents = [Entity(match.start(), match.end(), text[match.start():match.end()], 'clinical_events')
                                for match in re.finditer(self.clinical_events_pattern, strQ2B(text.lower()))]

        return clinical_event_ents

    def extract_profession_ents(self, text: str) -> List[Entity]:
        profession_ents = []
        for match in re.finditer(self.prof_pattern, text):
            start, end = match.span()
            ent_text = text[start:end]
            if ent_text == '外送' and text[min(0, start - 1)] == '另':
                continue
            profession_ents.append(Entity(start, end, ent_text, 'profession'))
        
        return profession_ents

    def extract_organization_ents(self, text: str) -> List[Entity]:
        organization_ents = [Entity(match.start(), match.end(), text[match.start():match.end()], 'organization')
                             for match in re.finditer(self.org_pattern, text)]
        return organization_ents

    def extract_education_ents(self, text: str) -> List[Entity]:
        edu_ents = [Entity(match.start(), match.end(), text[match.start():match.end()], 'education')
                    for match in re.finditer('法律系|化學系|交通大學|高雄大學', text)]
        return edu_ents
    
    def extract_med_exam_ents(self, text: str) -> List[Entity]:
        med_ents = [Entity(match.start(), match.end(), text[match.start():match.end()], 'med_exam')
                    for match in re.finditer(self.med_pattern, strQ2B(text.lower()))]
        return med_ents
    
    def extract_contact_ents(self, text: str) -> List[Entity]:
        con_ents = [Entity(match.start(), match.end(), text[match.start():match.end()], 'contact')
                    for match in re.finditer(self.contact_pattern, strQ2B(text.lower()))]
        return con_ents