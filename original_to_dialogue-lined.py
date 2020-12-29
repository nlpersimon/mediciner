import pandas as pd
import re
import tqdm
import os


def build_dataset(file_paths, set_type, character_pattern):
    dst_dir = f'data/dialogue_lined/{set_type}'
    dialogue_span_table = []
    for file_path in tqdm.tqdm(file_paths):
        with open(file_path, 'r') as f:
            dialogue = doc_to_dialogue(f.read(), character_pattern)
        _, file_name = os.path.split(file_path)
        cur_table = write_out_and_build_table(dialogue, os.path.join(dst_dir, file_name))
        dialogue_span_table.append(cur_table)
    dialogue_span_table = pd.concat(dialogue_span_table).reset_index(drop=True)
    return dialogue_span_table

def doc_to_dialogue(doc_text, character_pattern):
    cha_iters = list(re.finditer(character_pattern, doc_text))
    dialogue = [doc_text[cur_cha.start():next_cha.start()]
                for cur_cha, next_cha in zip(cha_iters, cha_iters[1:])]
    dialogue += [doc_text[cha_iters[-1].start():]]
    return dialogue

def write_out_and_build_table(dialogue, out_path):
    article_id = int(re.findall('-(\d+)\.', out_path)[0])
    table = build_table(article_id, dialogue)
    with open(out_path, 'w') as f:
        f.write('\n'.join(dialogue) + '\n')
    return table

def build_table(article_id, dialogue):
    doc_text = ''.join(dialogue)
    start, table = 0, []
    for sent_id, sent in enumerate(dialogue):
        end = start + len(sent)
        assert doc_text[start:end] == sent
        table.append([article_id, sent_id, start, end])
        start = end
    table = pd.DataFrame(table, columns=['article_id', 'sentence_id', 'start', 'end'])
    return table

def build_dataset_sent_ents_table(dataset_dialogue_span_table, dataset_ents_table):
    dialogue_group = dataset_dialogue_span_table.groupby('article_id')
    ents_group = dataset_ents_table.groupby('article_id')
    dataset_sent_ents_table = []
    for article_id, ents_indices in ents_group.groups.items():
        dialogue_indices = dialogue_group.groups[article_id]
        cur_table = build_article_sent_ents_table(dataset_dialogue_span_table.iloc[dialogue_indices],
                                                  dataset_ents_table.iloc[ents_indices])
        dataset_sent_ents_table.append(cur_table)
    dataset_sent_ents_table = pd.concat(dataset_sent_ents_table)
    return dataset_sent_ents_table

def build_article_sent_ents_table(article_dialogue_span_table, article_ents_table):
    pos2sent_id = {pos: row.sentence_id for row in article_dialogue_span_table.itertuples()
                                        for pos in range(row.start, row.end)}
    article_dialogue_span_table = article_dialogue_span_table.set_index('sentence_id')
    article_sent_ents_table = []
    for row in article_ents_table.itertuples():
        start_sent_id, end_sent_id = pos2sent_id[row.start_position], pos2sent_id[row.end_position - 1]
        assert start_sent_id == end_sent_id
        original_start = article_dialogue_span_table.loc[start_sent_id]['start']
        ent_start, ent_end = row.start_position - original_start, row.end_position - original_start
        article_sent_ents_table.append([row.article_id,
                                        start_sent_id,
                                        ent_start,
                                        ent_end,
                                        row.entity_text,
                                        row.entity_type])
    columns = ['article_id', 'sentence_id', 'start_position', 'end_position', 'entity_text', 'entity_type']
    article_sent_ents_table = pd.DataFrame(article_sent_ents_table, columns=columns)
    return article_sent_ents_table

def main():
    diag_characters = [
        '民\w?',
        '女?醫師[AB]?',
        '護理師[AB]?',
        '個?管師',
        '家屬[12B]?',
        '不確定人物',
        '生',
        '眾'
    ]
    character_pattern = '|'.join([c + '：' for c in diag_characters])
    
    sample_dir = 'data/original/sample'
    sample_files = [os.path.join(sample_dir, fn) for fn in os.listdir(sample_dir) if fn[-3:] == 'txt']
    sample_dialogue_span_table = build_dataset(sample_files, 'sample', character_pattern)
    
    sample_ents = pd.read_csv('data/original/sample_entities.csv')
    sample_sent_ents_table = build_dataset_sent_ents_table(sample_dialogue_span_table, sample_ents)
    sample_sent_ents_table.to_csv('data/dialogue_lined/sample_entities.csv', index=False)
    
    train_dir = 'data/original/train'
    train_files = [os.path.join(train_dir, fn) for fn in os.listdir(train_dir) if fn[-3:] == 'txt']
    train_dialogue_span_table = build_dataset(train_files, 'train', character_pattern)
    
    train_ents = pd.read_csv('data/original/train_entities.csv')
    train_sent_ents_table = build_dataset_sent_ents_table(train_dialogue_span_table, train_ents)
    train_sent_ents_table.to_csv('data/dialogue_lined/train_entities.csv', index=False)
    
    dev_dir = 'data/original/dev'
    dev_files = [os.path.join(dev_dir, fn) for fn in os.listdir(dev_dir) if fn[-3:] == 'txt']
    dev_dialogue_span_table = build_dataset(dev_files, 'dev', character_pattern)
    
    test_dir = 'data/original/test'
    test_files = [os.path.join(test_dir, fn) for fn in os.listdir(test_dir) if fn[-3:] == 'txt']
    test_dialogue_span_table = build_dataset(test_files, 'test', character_pattern)
    
    return

if __name__ == '__main__':
    main()