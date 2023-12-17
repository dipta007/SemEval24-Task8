import torch
from torch import nn
import lightning.pytorch as pl
from transformers import AutoModel
from .encoder import Encoder
from .doc_encoder import DocEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

THRSHOLD = 0.5


class ContrastiveModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        sen_encoder = AutoModel.from_pretrained(self.config.model_name)
        if self.config.encoder_type == "sen":
            self.encoder = Encoder(config, sen_encoder)
        elif self.config.encoder_type == "doc":
            doc_encoder = AutoModel.from_pretrained(self.config.model_name)
            self.encoder = DocEncoder(config, sen_encoder, doc_encoder)
        else:
            raise ValueError("Encoder type not found")
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.cls_dropout),
            nn.Linear(sen_encoder.config.hidden_size, sen_encoder.config.hidden_size), nn.Tanh(),
            nn.Dropout(self.config.cls_dropout),
            nn.Linear(sen_encoder.config.hidden_size, 1), nn.Sigmoid()
        )

        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.bce_loss = nn.BCELoss()

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

        loss = (
            self.config.loss_weight_con * con_loss
            + self.config.loss_weight_text * text_bce_loss
            + self.config.loss_weight_gen_text * gen_text_bce_loss
        )

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
        text, gen_text, label, _ = batch
        loss, log_dict = self(text, gen_text, label)

        for key, value in log_dict.items():
            self.log(f"train/{key}", value, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, gen_text, label, _ = batch
        loss, log_dict = self(text, gen_text, label)

        for key, value in log_dict.items():
            self.log(f"valid/{key}", value, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        text, _, _, ids = batch

        text_embedding = self.encoder(text)
        cls = self.classifier(text_embedding)
        cls = cls.view(-1)
        cls = cls.detach().cpu().numpy()
        cls[cls >= THRSHOLD] = 1
        cls[cls < THRSHOLD] = 0
        return ids, cls

    def get_metrics(self, preds, labels):
        preds_flat = preds.view(-1).detach().cpu().numpy()
        preds_flat[preds_flat >= THRSHOLD] = 1
        preds_flat[preds_flat < THRSHOLD] = 0
        labels_flat = labels.view(-1).detach().cpu().numpy()

        preds_flat = preds_flat.astype(int)
        labels_flat = labels_flat.astype(int)

        acc = accuracy_score(labels_flat, preds_flat)
        precision = precision_score(labels_flat, preds_flat, zero_division=1)
        recall = recall_score(labels_flat, preds_flat, zero_division=1)
        f1 = f1_score(labels_flat, preds_flat, zero_division=1)
        micro_f1 = f1_score(labels_flat, preds_flat, average="micro", zero_division=1)
        macro_f1 = f1_score(labels_flat, preds_flat, average="macro", zero_division=1)

        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer