"""
Usage:
    train_ensemble.py --path-to-train-corpus-dir=<dir> --path-to-train-ents-table=<file> --path-to-saving-model=<file> --path-to-vocab=<file> --path-to-bert1=<dir> --path-to-bert2=<dir>[options]

Options:
    -h --help                               show this screen.
    --path-to-train-corpus-dir=<dir>        path to the training corpus directory
    --path-to-train-ents-table=<file>       path to the training entities table for the corpus
    --path-to-saving-model=<file>           path to the fine-tuned model should be saved
    --path-to-vocab=<file>                  path to the vocabulary for tokenizer
    --path-to-bert1=<dir>                   path to the first trained bert model
    --path-to-bert2=<dir>                   path to the second trained bert model
    --path-to-val-corpus-dir=<dir>          path to the validation corpus directory [default: ]
    --path-to-val-ents-table=<file>         path to the validation entities table for the corpus [default: ]
    --gpu=<int>                             use GPU [default: -1]
    --mode=<str>                            multi-sents or uni-sent [default: multi-sents]
    --ideal-batch-size=<int>                batch size that you want to use to update the model [default: 32]
    --actual-batch-size=<int>               batch size that your gpu or memory can hold, need to be smaller than --ideal-batch-size [default: 8]
    --max-epochs=<int>                      max number of epochs [default: 20]
    --max-input-len=<int>                   max length of input sequence [default: 510]
    --optimizer=<str>                       optimizer to train the model [default: AdaBelief]
    --learning-rate=<float>                 learning rate for optimizer [default: 3e-5]
    --weight-decay=<float>                  weight decay rate for updating parameters [default: 0.0]
    --lr-scheduler=<str>                    learning rate scheduler [default: ]
    --seed=<int>                            random seed to sample typos and sentences [default: 1]
"""
from docopt import docopt
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
import os
import pytorch_lightning as pl
from mediciner.dataset.processor import BertMultiSentProcessor, BertUniSentProcessor, BertProcessor
from mediciner.dataset.corpus_labeler import TAG_TO_LABEL
from mediciner.train import (BertLightning,
                             prepare_dataloader,
                             prepare_trainer)
from mediciner.ner_model import BertEnsemble



def collect_hparams(args: dict) -> dict:
    hparams = {
        'mode': str(args['--mode']),
        'pretrained-model': f'{str(args["--path-to-bert1"])}|{str(args["--path-to-bert2"])}',
        'max-input-len': int(args['--max-input-len']),
        'batch-size': int(args['--ideal-batch-size']),
        'max-epochs': int(args['--max-epochs']),
        'optimizer': str(args['--optimizer']),
        'learning-rate': float(args['--learning-rate']),
        'weight-decay': float(args['--weight-decay']),
        'lr-scheduler': str(args['--lr-scheduler'])
    }
    return hparams

def main():
    args = docopt(__doc__)

    pl.seed_everything(int(args['--seed']))

    tokenizer = BertWordPieceTokenizer(str(args['--path-to-vocab']))

    processors = {
        'multi-sents': BertMultiSentProcessor,
        'uni-sent': BertUniSentProcessor
    }

    processor_constructor = processors[str(args['--mode'])]
    processor = processor_constructor(int(args['--max-input-len']), tokenizer)
    train_dataloader, val_dataloader = prepare_dataloader(processor, tokenizer, args)
    

    ensemble_model = BertEnsemble(str(args['--path-to-bert1']),
                                  str(args['--path-to-bert2']),
                                  len(TAG_TO_LABEL))
    hparams = collect_hparams(args)
    accum_steps = int(int(args['--ideal-batch-size']) / int(args['--actual-batch-size']))
    hparams['n-iters-an-epoch'] = int(len(train_dataloader) / accum_steps)
    bert_lightning = BertLightning(ensemble_model, hparams, use_logger=True)
    
    trainer = prepare_trainer(args)
    
    print('hyper parameters:')
    print(hparams)
    
    trainer.fit(bert_lightning, train_dataloader, val_dataloader)

    bert_lightning.bert_model.save_trained(str(args['--path-to-saving-model']))

    with open(os.path.join(str(args['--path-to-saving-model']), 'hparams.txt'), 'w') as f:
        for hparam, value in hparams.items():
            f.write(f'{hparam}: {value}\n')
    return


if __name__ == '__main__':
    main()