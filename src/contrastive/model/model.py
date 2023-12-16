import torch
from torch import nn
import lightning.pytorch as pl
from transformers import AutoModel
from .encoder import Encoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

THRSHOLD = 0.5

class ContrastiveModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        sen_encoder = AutoModel.from_pretrained(self.config.model_name)
        # self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)
        self.encoder = Encoder(config, sen_encoder)
        self.classifier = nn.Sequential(
            nn.Linear(sen_encoder.config.hidden_size, 1),
            nn.Sigmoid()
        )

        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.bce_loss = nn.BCELoss()

    def get_metrics(self, preds, labels):
        preds_flat = preds.view(-1).detach().cpu().numpy()
        preds_flat[preds_flat >= THRSHOLD] = 1
        labels_flat = labels.view(-1).detach().cpu().numpy()

        preds_flat = preds_flat.astype(int)
        labels_flat = labels_flat.astype(int)

        acc = accuracy_score(labels_flat, preds_flat)
        precision = precision_score(labels_flat, preds_flat)
        recall = recall_score(labels_flat, preds_flat)
        f1 = f1_score(labels_flat, preds_flat)
        micro_f1 = f1_score(labels_flat, preds_flat, average='micro')
        macro_f1 = f1_score(labels_flat, preds_flat, average='macro')

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }

    def forward(self, text, gen_text, label):
        text_embedding = self.encoder(text)
        gen_text_embedding = self.encoder(gen_text)
        con_loss = self.contrastive_loss(text_embedding, gen_text_embedding, label)

        text_pred = self.classifier(text_embedding)
        gen_text_pred = self.classifier(gen_text_embedding)

        text_label = label.clone()
        text_label[text_label == -1] = 0.0
        text_bce_loss = self.bce_loss(text_pred.view(-1), text_label.view(-1))

        gen_text_label = torch.ones_like(label, dtype=torch.float32)
        gen_text_bce_loss = self.bce_loss(gen_text_pred.view(-1), gen_text_label.view(-1))

        loss = con_loss + text_bce_loss + gen_text_bce_loss

        log_dict = {
            "loss": loss,
            "con_loss": con_loss,
            "text_bce_loss": text_bce_loss,
            "gen_text_bce_loss": gen_text_bce_loss,
        }

        text_metrics = self.get_metrics(text_pred, text_label)
        gen_text_metrics = self.get_metrics(gen_text_pred, gen_text_label)

        for key, value in text_metrics.items():
            log_dict[f"text_{key}"] = value
        
        for key, value in gen_text_metrics.items():
            log_dict[f"gen_text_{key}"] = value

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        text, gen_text, label = batch
        loss, log_dict = self(text, gen_text, label)

        for key, value in log_dict.items():
            self.log(f"train/{key}", value, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        text, gen_text, label = batch
        loss, log_dict = self(text, gen_text, label)

        for key, value in log_dict.items():
            self.log(f"valid/{key}", value, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer