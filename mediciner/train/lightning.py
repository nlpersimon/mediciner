import pytorch_lightning as pl
from transformers import BertForTokenClassification, AdamW
from torch_optimizer import RAdam
from adabelief_pytorch import AdaBelief
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from ..dataset.corpus_labeler import tag_to_label, label_to_tag



class BertLightning(pl.LightningModule):
    def __init__(self, bert_model: BertForTokenClassification, use_logger: bool = False) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.use_logger = False
        self._subword_label = tag_to_label('X')
        self._outside_label = tag_to_label('O')

    def forward(self, x):
        input_ids, attention_mask = x
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=self.use_logger)
        return loss

    def configure_optimizers(self):
        no_decay = ('bias', 'gamma', 'beta')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        #optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        #optimizer = RAdam(optimizer_grouped_parameters, lr=3e-5)
        optimizer = AdaBelief(optimizer_grouped_parameters,
                              lr=3e-5,
                              eps=1e-16,
                              betas=(0.9,0.999),
                              weight_decouple=True,
                              rectify=True)
        return optimizer

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