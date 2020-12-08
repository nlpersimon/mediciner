import pytorch_lightning as pl
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch_optimizer import RAdam
import torch
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from ..dataset.corpus_labeler import tag_to_label, label_to_tag
from ..dataset.mrc import id_to_entype
from ..ner_model import BertWithMRC



class BertLightning(pl.LightningModule):
    def __init__(self, bert_model: BertForTokenClassification, hparams: dict, use_logger: bool = False) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.use_logger = False
        self.hparams = hparams
        self._subword_label = tag_to_label('X')
        self._outside_label = tag_to_label('O')

    def forward(self, x):
        input_ids, attention_mask = x
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        loss = self.compute_loss(logits, labels, attention_mask)
        #self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=self.use_logger)
        if self.trainer.lr_schedulers:
            scheduler = self.trainer.lr_schedulers[0]
            param_groups = scheduler['scheduler'].optimizer.param_groups
            lr = param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=self.use_logger)
        return loss
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 把原始程式碼計算 loss 的部分拉出來做是為了 assign ignore_index 給 cross_entropy
        # 這樣 window format 就可以直接在 context sentence 的部分將 label 設為 tag_to_label('P')
        # 從而避免計算 context sentences 的 loss
        # Only keep active parts of the loss
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.bert_model.num_labels)
        # ignore context labels
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(tag_to_label('P')).type_as(labels)
        )
        loss = F.cross_entropy(active_logits, active_labels, ignore_index=tag_to_label('P'))
        return loss

    def configure_optimizers(self):
        no_decay = ('bias', 'gamma', 'beta')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams['weight-decay']},
            {'params': [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer_ops = {
            'AdamW': AdamW,
            'RAdam': RAdam,
            'AdaBelief': AdaBelief
        }
        optimizer_constructor = optimizer_ops[self.hparams['optimizer']]
        if self.hparams['optimizer'] == 'AdaBelief':
            optimizer = optimizer_constructor(
                optimizer_grouped_parameters,
                lr=self.hparams['learning-rate'],
                eps=1e-16,
                betas=(0.9,0.999),
                weight_decouple=True,
                rectify=True)
        else:
            optimizer = optimizer_constructor(optimizer_grouped_parameters, lr=self.hparams['learning-rate'])

        if self.use_logger:
            self.logger.experiment.add_hparams(hparams_dict=self.hparams)

        if self.hparams['lr-scheduler']:
            scheduler = self.configure_lr_scheduler(optimizer, self.hparams['lr-scheduler'])
            return [optimizer], [scheduler]

        return optimizer

    def configure_lr_scheduler(self, optimizer, lr_scheduler_name):
        if lr_scheduler_name == 'cyclic':
            lr_scheduler = self.get_cyclic_lr_scheduler(optimizer)
        else:
            total_training_steps = self.hparams['n-iters-an-epoch'] * self.hparams['max-epochs']
            lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, 0, total_training_steps),
                            'name': 'linear lr',
                            'interval': 'step'}
        return lr_scheduler
            
    
    def get_cyclic_lr_scheduler(self, optimizer):
        max_lr = self.hparams['learning-rate'] 
        base_lr = self.hparams.get('base-lr', max_lr / 3)
        step_size = self.hparams['n-iters-an-epoch'] * 4
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                                       base_lr=base_lr,
                                                                       max_lr=max_lr,
                                                                       step_size_up=step_size,
                                                                       cycle_momentum=False),
                        'name': 'cyclic lr',
                        'interval': 'step'}
        return lr_scheduler

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self((input_ids, attention_mask))
        pred_labels = outputs['logits'].argmax(dim=2)
        true_tags, pred_tags = [], []
        for true, pred, att_mask in zip(labels, pred_labels, attention_mask):
            select_bool = (att_mask == 1) & (true != self._subword_label)
            pred[pred == self._subword_label] = self._outside_label
            true_tags.append([label_to_tag(int(label)) for label in true[select_bool]])
            pred_tags.append([label_to_tag(int(label)) for label in pred[select_bool]])
        return (true_tags, pred_tags)

    def validation_epoch_end(self, validation_step_outputs):
        entity_type_metrics = self.compute_entity_type_metrics(validation_step_outputs)
        for ent_type, metrics in entity_type_metrics.items():
            self.logger.experiment.add_scalars(ent_type,
                                               {'precision': metrics['precision'],
                                                'recall': metrics['recall'],
                                                'f1-score': metrics['f1-score']},
                                               self.current_epoch)
        return
    
    def compute_entity_type_metrics(self, validation_step_outputs):
        true_tags, pred_tags = [], []
        for step_true, step_pred in validation_step_outputs:
            true_tags += step_true
            pred_tags += step_pred
        entity_type_metrics = classification_report(true_tags,
                                                    pred_tags,
                                                    mode='strict',
                                                    scheme=IOB2,
                                                    output_dict=True)
        return entity_type_metrics


class BertWithMRCLightning(BertLightning):
    def __init__(self, bert_mrc_model: BertWithMRC, hparams: dict, use_logger: bool = False) -> None:
        super().__init__(bert_mrc_model, hparams, use_logger)
    
    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        return outputs
    
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, start_tensor, end_tensor, entype_ids = batch
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  start_tensor=start_tensor,
                                  end_tensor=end_tensor)
        loss = outputs['loss']
        if self.trainer.lr_schedulers:
            scheduler = self.trainer.lr_schedulers[0]
            param_groups = scheduler['scheduler'].optimizer.param_groups
            lr = param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=self.use_logger)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        每個 batch 都是一個 paragraph 的各個 entity type query
        且需要另外提供 ent_type_ids 以供對回去 entity type labels
        """
        input_ids, token_type_ids, attention_mask, start_tensor, end_tensor, ent_type_ids = batch
        outputs = self((input_ids, token_type_ids, attention_mask))
        pred_starts = outputs['start_logits'].argmax(dim=2)
        pred_ends = outputs['end_logits'].argmax(dim=2)
        input_len = ((attention_mask[0] == 1) & (token_type_ids[0] == 1)).sum()
        true_tags, pred_tags = ['O'] * input_len, ['O'] * input_len
        for true_s, true_e, pred_s, pred_e, tkt_ids, att_mask, etp_id in zip(start_tensor,
                                                                             end_tensor,
                                                                             pred_starts,
                                                                             pred_ends,
                                                                             token_type_ids,
                                                                             attention_mask,
                                                                             ent_type_ids):
            select_bool = (att_mask == 1) & (tkt_ids == 1)
            self.fill_tags(true_tags, true_s[select_bool], true_e[select_bool], int(etp_id))
            self.fill_tags(pred_tags, pred_s[select_bool], pred_e[select_bool], int(etp_id))
        return ([true_tags], [pred_tags])
    
    def fill_tags(self, raw_tags, starts, ends, ent_type_id):
        starts = torch.where(starts == 1)[0]
        ends = torch.where(ends == 1)[0]
        ent_chunks = sorted([('S', int(s)) for s in starts] + [('E', int(e)) for e in ends],
                            key=lambda x: x[1])
        prev_btp, prev_idx = '', -1
        for bound_type, idx in ent_chunks:
            if bound_type == 'S' and raw_tags[idx] == 'O':
                raw_tags[idx] = f'B-{id_to_entype(ent_type_id)}'
            elif prev_btp == 'S' and all(tag == 'O' for tag in raw_tags[(prev_idx + 1):(idx + 1)]):
                raw_tags[(prev_idx + 1):(idx + 1)] = [f'I-{id_to_entype(ent_type_id)}'] * (idx - prev_idx)
            prev_btp, prev_idx = bound_type, idx
        return
        