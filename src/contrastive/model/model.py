import torch
from torch import nn
import lightning.pytorch as pl
from transformers import AutoModel
from .encoder import Encoder
from .doc_encoder import DocEncoder
from .long_encoder import LongEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from dataloader.sen_sim import MODELS

THRSHOLD = 0.5


class ContrastiveModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        sen_encoder = AutoModel.from_pretrained(self.config.model_name)
        if self.config.encoder_type == "sen":
            if self.config.model_name.index("longformer") != -1:
                self.encoder = LongEncoder(config, sen_encoder)
            else:
                self.encoder = Encoder(config, sen_encoder)
        elif self.config.encoder_type == "doc":
            doc_encoder = AutoModel.from_pretrained(self.config.model_name)
            self.encoder = DocEncoder(config, sen_encoder, doc_encoder)
        else:
            raise ValueError("Encoder type not found")
        
        cls_act = nn.Tanh() if self.config.cls_act == "tanh" else nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.cls_dropout),
            nn.Linear(sen_encoder.config.hidden_size, sen_encoder.config.hidden_size),
            cls_act,
            nn.Dropout(self.config.cls_dropout),
            nn.Linear(sen_encoder.config.hidden_size, len(MODELS)),
        )

        self.contrastive_loss = nn.TripletMarginLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pos, neg, pos_label, neg_label):
        pos_emb_1 = self.encoder(pos)
        pos_emb_2 = self.encoder(pos)

        neg_emb_1 = self.encoder(neg)
        neg_emb_2 = self.encoder(neg)

        con_loss_1 = self.contrastive_loss(pos_emb_1, pos_emb_2, neg_emb_1)
        con_loss_2 = self.contrastive_loss(pos_emb_1, pos_emb_2, neg_emb_2)
        con_loss_3 = self.contrastive_loss(neg_emb_1, neg_emb_2, pos_emb_1)
        con_loss_4 = self.contrastive_loss(neg_emb_1, neg_emb_2, pos_emb_2)

        pos_con_loss = (con_loss_1 + con_loss_2) / 2.0
        neg_con_loss = (con_loss_3 + con_loss_4) / 2.0

        pos_pred_1 = self.classifier(pos_emb_1)
        pos_pred_2 = self.classifier(pos_emb_2)
        neg_pred_1 = self.classifier(neg_emb_1)
        neg_pred_2 = self.classifier(neg_emb_2)

        pos_ce_loss_1 = self.ce_loss(pos_pred_1, pos_label)
        pos_ce_loss_2 = self.ce_loss(pos_pred_2, pos_label)
        neg_ce_loss_1 = self.ce_loss(neg_pred_1, neg_label)
        neg_ce_loss_2 = self.ce_loss(neg_pred_2, neg_label)

        pos_ce_loss = (pos_ce_loss_1 + pos_ce_loss_2) / 2.0
        neg_ce_loss = (neg_ce_loss_1 + neg_ce_loss_2) / 2.0

        loss = (
            self.config.lw_pos_con * pos_con_loss
            + self.config.lw_neg_con * neg_con_loss
            + self.config.lw_pos_ce * pos_ce_loss
            + self.config.lw_neg_ce * neg_ce_loss
        )

        log_dict = {
            "loss": loss,
            "pos_con_loss": pos_con_loss,
            "neg_con_loss": neg_con_loss,
            "pos_ce_loss": pos_ce_loss,
            "neg_ce_loss": neg_ce_loss,
        }

        pos_metrics = self.get_metrics(pos_pred_1, pos_label)
        neg_metrics = self.get_metrics(neg_pred_1, neg_label)

        for key, value in pos_metrics.items():
            log_dict[f"pos_{key}"] = value

        for key, value in neg_metrics.items():
            log_dict[f"neg_{key}"] = value

        log_dict["mean_acc"] = (pos_metrics["acc"] + neg_metrics["acc"]) / 2.0

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        pos, neg, pos_label, neg_label, _, _ = batch
        loss, log_dict = self(pos, neg, pos_label, neg_label)

        for key, value in log_dict.items():
            self.log(
                f"train/{key}", value, prog_bar=True, batch_size=self.config.batch_size
            )
        return loss

    def validation_step(self, batch, batch_idx):
        pos, neg, pos_label, neg_label, _, _ = batch
        loss, log_dict = self(pos, neg, pos_label, neg_label)

        for key, value in log_dict.items():
            self.log(
                f"valid/{key}", value, prog_bar=True, batch_size=self.config.batch_size
            )
        return loss

    def predict_step(self, batch, batch_idx):
        pos, neg, pos_label, neg_label, pos_ids, neg_ids = batch

        text_embedding = self.encoder(pos)
        cls = self.classifier(text_embedding)
        cls = torch.softmax(cls, dim=-1)
        cls = torch.argmax(cls, dim=-1)
        cls = cls.view(-1)
        cls = cls.detach().cpu().numpy()
        return pos_ids, cls

    def get_metrics(self, preds, labels):
        preds = torch.softmax(preds, dim=-1)
        preds_flat = torch.argmax(preds, dim=-1).detach().cpu().numpy()
        labels_flat = labels.view(-1).detach().cpu().numpy()

        preds_flat = preds_flat.astype(int)
        labels_flat = labels_flat.astype(int)

        acc = accuracy_score(labels_flat, preds_flat)
        micro_f1 = f1_score(labels_flat, preds_flat, average="micro", zero_division=1)
        macro_f1 = f1_score(labels_flat, preds_flat, average="macro", zero_division=1)

        return {
            "acc": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        return optimizer
