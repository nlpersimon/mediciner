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
    --ensemble                              ensemble or not
    --crf                                   model with CRF or not
"""
from docopt import docopt
from transformers import BertForTokenClassification
from tokenizers import BertWordPieceTokenizer
import torch
from mediciner.dataset.processor import BertMultiSentProcessor, BertUniSentProcessor
from mediciner.dataset.utils import read_corpus
from mediciner.extractor.extractors import BertExtractor, BertWithCRFExtractor
from mediciner.utils import build_ents_table
from mediciner.ner_model import BertEnsemble, BertWithCRF






def main():
    args = docopt(__doc__)

    processors = {
        'multi-sents': BertMultiSentProcessor,
        'uni-sent': BertUniSentProcessor
    }

    tokenizer = BertWordPieceTokenizer(str(args['--path-to-vocab']))
    max_input_len = int(args['--max-input-len'])
    processor_constructor = processors[str(args['--mode'])]
    processor = processor_constructor(max_input_len, tokenizer)
    if args['--ensemble']:
        bert_model = BertEnsemble.load_trained(str(args['--path-to-model-dir']))
    elif args['--crf']:
        bert_model = BertWithCRF.from_pretrained(str(args['--path-to-model-dir']))
    else:
        bert_model = BertForTokenClassification.from_pretrained(str(args['--path-to-model-dir']))
    device_no = int(args['--gpu'])
    device = torch.device(f'cuda:{device_no}') if device_no > -1 else torch.device('cpu')
    if args['--crf']:
        bert_extractor = BertWithCRFExtractor(bert_model, tokenizer, max_input_len, device)
    else:
        bert_extractor = BertExtractor(bert_model, tokenizer, max_input_len, device)
    corpus = read_corpus(str(args['--path-to-corpus-dir']))
    ents_table = build_ents_table(corpus, processor, bert_extractor, batch_size=int(args['--batch-size']))
    ents_table.to_csv(str(args['--path-to-output']), index=False, sep='\t')
    return

if __name__ == '__main__':
    main()