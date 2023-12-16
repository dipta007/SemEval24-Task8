import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class DocEncoder(pl.LightningModule):
    def __init__(self, config, sen_encoder, doc_encoder):
        super().__init__()
        self.config = config
        self.sen_encoder = sen_encoder
        self.doc_encoder = doc_encoder

    def forward(self, text):
        enc_out = self.sen_encoder(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        enc_out = enc_out.last_hidden_state[:, 0, :]
        doc_sen_emb = enc_out.view(self.config.batch_size, -1, self.sen_encoder.config.hidden_size)

        doc_emb = self.doc_encoder(inputs_embeds=doc_sen_emb, attention_mask=text['doc_attention_mask'])
        doc_emb = doc_emb.last_hidden_state[:, 0, :]
        return doc_emb