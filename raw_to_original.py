import pandas as pd
import tqdm




def write_to_file(set_type, txt_dir, csv_dir, raw_docs):
    ent_data = None
    for raw_doc in tqdm.tqdm(raw_docs):
        doc_text, cur_data = extract_doc_and_data(raw_doc)
        assert cur_data['article_id'].nunique() == 1
        with open(f'{txt_dir}/{set_type}-{cur_data["article_id"].iloc[0]}.txt', 'w') as f:
            f.write(doc_text)
        ent_data = pd.concat([ent_data, cur_data])
    ent_data.to_csv(f'{csv_dir}/{set_type}_entities.csv', index=False)
    return ent_data

def extract_doc_and_data(raw_doc):
    doc_text, _, data_text = raw_doc.partition('\n')
    data = [row.split('\t') for row in data_text.splitlines()]
    data = pd.DataFrame(data[1:], columns=data[0])
    return (doc_text, data)

def main():
    with open('data/raw/SampleData_deid.txt', 'r') as f:
        samples = [txt for txt in f.read().split('\n\n--------------------\n\n') if txt]
    sample_ents = write_to_file('sample', 'data/original/sample', 'data/original', samples)
    
    with open('data/raw/train_2.txt', 'r') as f:
        train = [txt for txt in f.read().split('\n\n--------------------\n\n') if txt]
    train_ents = write_to_file('train', 'data/original/train', 'data/original', train)
    
    with open('data/raw/development_2.txt', 'r') as f:
        dev = [txt for txt in f.read().split('\n\n--------------------\n\n') if txt]
    for article in dev:
        article_id, content = article.split('\n')
        article_id = article_id.split(': ')[-1]
        with open(f'data/original/dev/dev-{article_id}.txt', 'w') as f:
            f.write(content)
            
    with open('data/raw/test.txt', 'r') as f:
        test = [txt for txt in f.read().split('\n\n--------------------\n\n') if txt]
    for article in test:
        article_id, content = article.split('\n')
        article_id = article_id.split(': ')[-1]
        with open(f'data/original/test/test-{article_id}.txt', 'w') as f:
            f.write(content)

    return

if __name__ == '__main__':
    main()