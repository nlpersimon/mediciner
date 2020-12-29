from itertools import chain
from tokenizers import BertWordPieceTokenizer
import os
from mediciner.dataset import read_corpus
from mediciner.dataset import BertProcessor, BertMultiSentProcessor


def write_examples(examples, file_name):
    with open(file_name, 'w') as f:
        for example in examples:
            f.write(example.content + '\n')
    return

def main():
    train_corpus = read_corpus('data/dialogue_lined/train')
    dev_corpus = read_corpus('data/dialogue_lined/dev')
    test_corpus = read_corpus('data/dialogue_lined/test')
    
    tokenizer = BertWordPieceTokenizer('vocab/bert-base-chinese-vocab.txt')
    processor = BertMultiSentProcessor(510, tokenizer)
    
    train_examples = processor.convert_corpus_to_examples(train_corpus)
    dev_examples = processor.convert_corpus_to_examples(dev_corpus)
    test_examples = processor.convert_corpus_to_examples(test_corpus)
    
    dst_dir = 'data/dialogue_lined/multi-sents-further-pretrain/'
    write_examples(chain(train_examples, test_examples), os.path.join(dst_dir, 'train_test_dialogues.txt'))
    
    return

if __name__ == '__main__':
    main()