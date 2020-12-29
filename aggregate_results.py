import re
import pandas as pd
from collections import defaultdict

train_ents = pd.read_csv('data/dialogue_lined/train_entities.csv')
time_ent_set = set(train_ents.query('entity_type == "time"')['entity_text'].unique())

def read_pred(file_path):
    pred = pd.read_csv(file_path, delimiter='\t')
    return pred

def add_predictions(dst_pred, src_pred, filter_dst_first=False, filter_src=True):
    if filter_dst_first:
        dst_pred = filter_results(dst_pred, filter_others=False)
    dst_ent_range_dict = build_ent_range_dict(dst_pred)
    filtered_src_pred = filter_ents(src_pred, dst_ent_range_dict)
    if filter_src:
        filtered_src_pred = filter_results(filtered_src_pred, filter_others=True)
    dst_pred = pd.concat([dst_pred, filtered_src_pred], ignore_index=True)
    dst_pred = (dst_pred.sort_values(['article_id', 'start_position'])
                        .reset_index(drop=True))
    return dst_pred

def build_ent_range_dict(ent_table):
    ent_range_dict = defaultdict(lambda: [])
    for row in ent_table.itertuples():
        ent_range_dict[row.article_id].append(range(row.start_position, row.end_position))
    return ent_range_dict

def filter_ents(source_ent_table, ent_range_dict):
    keep_rows = []
    for row in source_ent_table.itertuples():
        if not any(row.start_position in rng for rng in ent_range_dict[row.article_id]):
            if row.entity_type != 'time':
                keep_rows.append(row)
            elif row.entity_text in time_ent_set:
                keep_rows.append(row)
    return pd.DataFrame(keep_rows).drop(columns=['Index'])

def filter_results(ent_table, filter_others=True):
    filter_time_indices = get_filter_time_indices(ent_table)
    filter_med_exam_indices = get_filter_med_exam_indices(ent_table)
    filter_money_indices = get_filter_money_indices(ent_table)
    drop_indices = filter_time_indices + filter_med_exam_indices + filter_money_indices
    if filter_others:
        filter_other_exam_indices = get_filter_other_indices(ent_table)
        drop_indices += filter_other_exam_indices
    return (ent_table.drop(index=drop_indices)
                     .reset_index(drop=True))

def get_filter_time_indices(ent_table):
    time_ents = ent_table.query('entity_type == "time"')
    #time_ents = time_ents[time_ents['entity_text'].str.len() == 1 & ~time_ents['entity_text'].str.contains('[\d一二三四五六七八九零]')]
    #time_ents = time_ents[time_ents['entity_text'].str.len() == 1]
    #time_ents = time_ents.query(f'entity_text not in {time_ent_set}')
    time_ents = time_ents[(time_ents['entity_text'].str.len() == 1) | (time_ents['entity_text'] == '禮拜')]
    return time_ents.index.tolist()

def get_filter_med_exam_indices(ent_table):
    med_exam_ents = ent_table.query('entity_type == "med_exam"')
    # TODO: 加入全形數字
    not_med_exam_ents = med_exam_ents[~med_exam_ents['entity_text'].str.contains('[\d０１２３４５６７８９一二三四五六七八九零十百佰千仟萬億兩倆]')]
    #phone_no = ('119', '110')
    #phone_ents = med_exam_ents.query(f'entity_text in {phone_no}')
    return not_med_exam_ents.index.tolist()# + phone_ents.index.tolist()

def get_filter_money_indices(ent_table):
    money_ents = ent_table.query('entity_type == "money"')
    # TODO: 加入全形數字
    money_ents = money_ents[~money_ents['entity_text'].str.contains('[\d０１２３４５６７８９一二三四五六七八九零十兩倆百佰千仟萬億]')]
    return money_ents.index.tolist()

def get_filter_other_indices(ent_table):
    other_types = (
    'profession',
    'name',
    'location',
    'family',
    'ID',
    'clinical_event',
    'education',
    'contact',
    'organization',
    'others'
    )
    other_indices = []
    for ent_type in other_types:
        ents = ent_table.query(f'entity_type == "{ent_type}"')
        possible_texts = train_ents.query(f'entity_type == "{ent_type}"')['entity_text'].unique().tolist()
        indices = ents.query(f'entity_text not in {possible_texts}').index.tolist()
        other_indices += indices
    return other_indices

def filter_time_freq_1(ent_table):
    cnts = ent_table.query('entity_type == "time"')['entity_text'].value_counts()
    drop_times = cnts[cnts == 1].index.tolist() + ['前一天', '那一天']
    drop_indices = ent_table.query(f'entity_type == "time" and entity_text in {drop_times}').index
    return ent_table.drop(index=drop_indices).reset_index(drop=True)

def correctify_id_type(ents_df):
    ents_df_copy = ents_df.copy()
    for i in range(ents_df.shape[0]):
        if re.match('[a-zA-Z]\d{8,}', ents_df.iloc[i, -2]):
            ents_df_copy.iloc[i, -1] = 'ID'
    return ents_df_copy

def main():
    ensemble = read_pred('lastday_predictions/test-multi-sents-ensemble-test-member-lr2.25e-5-self-lr8e-3-ep15.tsv')
    roberta_large = read_pred('lastday_predictions/test-multi-sents-roberta-wwm-ext-large-lr1e-5-ep120.tsv')
    roberta = read_pred('lastday_predictions/test-multi-sents-roberta-wwm-ext-lr2.25e-5.tsv')
    bert = read_pred('lastday_predictions/test-multi-sents-test-further-pretrain-lr1e-5.tsv')
    rule = read_pred('lastday_predictions/rule-based.tsv')
    
    roblg_ensemble = add_predictions(roberta_large, ensemble, True)
    roblg_ensemble_roberta = add_predictions(roblg_ensemble, roberta, True)
    roblg_ensemble_roberta_bert = add_predictions(roblg_ensemble_roberta, bert, True)
    roblg_ensemble_roberta_bert_rule = add_predictions(roblg_ensemble_roberta_bert, rule, True, False)
    roblg_ensemble_roberta_bert_rule = correctify_id_type(roblg_ensemble_roberta_bert_rule)
    roblg_ensemble_roberta_bert_rule.to_csv('lastday_predictions/roblg-ensemble-roberta-bert-rule.tsv',
                                        index=False,
                                        sep='\t')
    
    return

if __name__ == '__main__':
    main()