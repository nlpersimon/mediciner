from .corpus_labeler import ENTITY_TYPES

ENTYPE_TO_QUERY = {
    'time': '所有日期、時間、年齡等，例如：出生年月日、看診時間',
    'med_exam': '所有醫療檢查報告、影像報告的數值',
    'profession': '所有任職公司名稱、任職單位等',
    'name': '所有的姓名、綽號、社群/通訊軟體使用者名稱、個人於團體中的代號等',
    'location': '所有地址、商店名、建築物名稱、景點',
    'family': '所有個人的家庭成員關係',
    'ID': '所有跟個人有關的編號，例如：身分證號碼、證件號碼、卡號、病歷號等',
    'clinical_event': '所有廣為人知的臨床事件',
    'education': '所有個人的就學經歷或學歷，如系所、程度',
    'money': '所有金額，例如：看診金額、個人負擔金額、自費金額',
    'contact': '所有電話號碼、傳真號碼、信箱、IP 位址、網址、網站名稱',
    'organization': '所有個人參與的組織、團體、社團等等的名稱',
    'others': '所有其他跟個人隱私有關，可以關聯到當事人的內容'
}
ID_TO_ENTYPE = {idx: entype for idx, entype in enumerate(ENTITY_TYPES)}
ENTYPE_TO_ID = {entype: idx for idx, entype in ID_TO_ENTYPE.items()}


def entype_to_query(entity_type: str) -> str:
    return ENTYPE_TO_QUERY[entity_type]

def entype_to_id(entity_type: str) -> int:
    return ENTYPE_TO_ID[entity_type]

def id_to_entype(id: int) -> str:
    return ID_TO_ENTYPE[id]