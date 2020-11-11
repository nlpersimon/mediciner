"""
Usage:
    build_entity_table.py --path-to-corpus-dir=<dir> --path-to-vocab=<file> --path-to-model-dir=<dir> --path-to-output=<file> [options]

Options:
    -h --help                               show this screen.
    --path-to-corpus-dir=<dir>              path to the corpus directory
    --path-to-vocab=<file>                  path to the vocabulary for tokenizer
    --path-to-model-dir=<dir>               path to the fine-tuned model directory
    --path-to-output=<file>                 path to the built entity table
    --gpu=<int>                             use GPU [default: -1]
    --mode=<str>                            multi-sents or uni-sent [default: multi-sents]
    --batch-size=<int>                      batch size for inferencing [default: 8]
    --max-input-len=<int>                   max length of input sequence [default: 510]
"""
from docopt import docopt
from transformers import BertForTokenClassification
from tokenizers import BertWordPieceTokenizer
import torch
from mediciner.dataset.processor import BertProcessor
from mediciner.dataset.utils import read_corpus
from mediciner.extractor.bert_extractor import BertExtractor
from mediciner.utils import build_ents_table






def main():
    args = docopt(__doc__)
    tokenizer = BertWordPieceTokenizer(str(args['--path-to-vocab']))
    max_input_len = int(args['--max-input-len'])
    sent_processor = BertProcessor(max_input_len, tokenizer, mode=str(args['--mode']))
    bert_model = BertForTokenClassification.from_pretrained(str(args['--path-to-model-dir']))
    device_no = int(args['--gpu'])
    device = torch.device(f'cuda:{device_no}') if device_no > -1 else torch.device('cpu')
    bert_extractor = BertExtractor(bert_model, tokenizer, max_input_len, device)
    corpus = read_corpus(str(args['--path-to-corpus-dir']))
    ents_table = build_ents_table(corpus, sent_processor, bert_extractor, batch_size=int(args['--batch-size']))
    ents_table.to_csv(str(args['--path-to-output']), index=False, sep='\t')
    return

if __name__ == '__main__':
    main()