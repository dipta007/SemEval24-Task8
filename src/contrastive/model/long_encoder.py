import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class LongEncoder(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder

    def forward(self, text):
        model_output = self.encoder(**text).last_hidden_state
        sen_embedding = model_output[:, 0, :]
        return sen_embedding

