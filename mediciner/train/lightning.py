import pytorch_lightning as pl
from transformers import BertForTokenClassification, AdamW



class BertLightning(pl.LightningModule):
    def __init__(self, bert_model: BertForTokenClassification, use_logger: bool = False) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.use_logger = False

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
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in self.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        return optimizer

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError