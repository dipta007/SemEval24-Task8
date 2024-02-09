import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class Encoder(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder

    def forward(self, text):
        model_output = self.encoder(**text)
        sen_embedding = self.mean_pooling(model_output, text['attention_mask'])
        sen_embedding = F.normalize(sen_embedding, p=2, dim=-1)
        return sen_embedding

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
