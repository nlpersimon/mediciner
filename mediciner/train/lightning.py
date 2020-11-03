import pytorch_lightning as pl



class BertLightning(pl.LightningModule):
    def __init__(self, bert_model):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError