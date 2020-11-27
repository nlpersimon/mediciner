"""
Usage:
    train.py --path-to-train-corpus-dir=<dir> --path-to-train-ents-table=<file> --path-to-saving-model=<file> --path-to-vocab=<file> [options]

Options:
    -h --help                               show this screen.
    --path-to-train-corpus-dir=<dir>        path to the training corpus directory
    --path-to-train-ents-table=<file>       path to the training entities table for the corpus
    --path-to-saving-model=<file>           path to the fine-tuned model should be saved
    --path-to-vocab=<file>                  path to the vocabulary for tokenizer
    --bert-name=<str>                       name of bert model you want to fine-tune [default: bert-base-chinese]
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
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertForTokenClassification
from typing import Tuple, List, Union
import random
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from mediciner.dataset.bert_dataset import BertDataset
from mediciner.dataset.processor import BertMultiSentProcessor, BertUniSentProcessor, BertProcessor
from mediciner.dataset.corpus_labeler import TAG_TO_LABEL
from mediciner.train.lightning import BertLightning



class TensorDataset(Dataset):
    def __init__(self, tensor: List[Tuple[torch.Tensor]]) -> None:
        self.tensor = tensor
    
    def __getitem__(self, idx):
        return self.tensor[idx]
    
    def __len__(self):
        return len(self.tensor)


def prepare_dataloader(processor: BertProcessor,
                       tokenizer: BertWordPieceTokenizer,
                       args: dict) -> Tuple[DataLoader, Union[None, DataLoader]]:
    batch_size = int(args['--actual-batch-size'])
    train_dataset = BertDataset(str(args['--path-to-train-corpus-dir']),
                                str(args['--path-to-train-ents-table']),
                                processor,
                                'train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    val_dataloader: Union[None, DataLoader] = None
    if args['--path-to-val-corpus-dir']:
        val_dataset = BertDataset(str(args['--path-to-val-corpus-dir']),
                                  str(args['--path-to-val-ents-table']),
                                  processor,
                                  'val')
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    return (train_dataloader, val_dataloader)

def collect_hparams(args: dict) -> dict:
    hparams = {
        'mode': str(args['--mode']),
        'pretrained-model': str(args['--bert-name']),
        'max-input-len': int(args['--max-input-len']),
        'batch-size': int(args['--ideal-batch-size']),
        'max-epochs': int(args['--max-epochs']),
        'optimizer': str(args['--optimizer']),
        'learning-rate': float(args['--learning-rate']),
        'weight-decay': float(args['--weight-decay']),
        'lr-scheduler': str(args['--lr-scheduler'])
    }
    return hparams

def prepare_trainer(args: dict) -> pl.Trainer:
    gpu_usage = [int(args['--gpu'])] if int(args['--gpu']) > -1 else 0
    *_, saving_model_name = str(args['--path-to-saving-model']).split('/')
    logger = TensorBoardLogger('lightning_logs', saving_model_name)
    accum_steps = int(int(args['--ideal-batch-size']) / int(args['--actual-batch-size']))
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs=int(args['--max-epochs']),
                        #limit_val_batches=0.0,
                        gpus=gpu_usage,
                        accumulate_grad_batches=accum_steps,
                        #gradient_clip_val=float(args['--clip-grad']),
                        deterministic=True,
                        terminate_on_nan=True,
                        logger=logger,
                        callbacks=[lr_monitor])
    return trainer


def main():
    args = docopt(__doc__)

    pl.seed_everything(int(args['--seed']))

    model_name = str(args['--bert-name'])
    tokenizer = BertWordPieceTokenizer(str(args['--path-to-vocab']))

    processors = {
        'multi-sents': BertMultiSentProcessor,
        'uni-sent': BertUniSentProcessor
    }

    processor_constructor = processors[str(args['--mode'])]
    processor = processor_constructor(int(args['--max-input-len']), tokenizer)
    train_dataloader, val_dataloader = prepare_dataloader(processor, tokenizer, args)
    

    bert_model = BertForTokenClassification.from_pretrained(model_name,
                                                            return_dict=True,
                                                            num_labels=len(TAG_TO_LABEL))
    hparams = collect_hparams(args)
    accum_steps = int(int(args['--ideal-batch-size']) / int(args['--actual-batch-size']))
    hparams['n-iters-an-epoch'] = int(len(train_dataloader) / accum_steps)
    bert_lightning = BertLightning(bert_model, hparams, use_logger=True)
    
    trainer = prepare_trainer(args)
    
    print('hyper parameters:')
    print(hparams)
    
    trainer.fit(bert_lightning, train_dataloader, val_dataloader)

    bert_lightning.bert_model.save_pretrained(str(args['--path-to-saving-model']))

    with open(os.path.join(str(args['--path-to-saving-model']), 'hparams.txt'), 'w') as f:
        for hparam, value in hparams.items():
            f.write(f'{hparam}: {value}\n')
    return


if __name__ == '__main__':
    main()