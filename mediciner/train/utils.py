from torch.utils.data import DataLoader, Dataset
from tokenizers import BertWordPieceTokenizer
from typing import Tuple, List, Union
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ..dataset.processor import BertProcessor
from ..dataset.bert_dataset import BertDataset


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
                        gradient_clip_val=float(args['--grad-clip-val']),
                        deterministic=True,
                        terminate_on_nan=True,
                        logger=logger,
                        callbacks=[lr_monitor])
    return trainer