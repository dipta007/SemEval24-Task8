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
        sen_encoder = AutoModel.from_pretrained(
            self.config.model_name, hidden_dropout_prob=self.config.enc_dropout
        )
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
            nn.LayerNorm(sen_encoder.config.hidden_size)
            if self.config.normalization == "before"
            else nn.Identity(),
            cls_act,
            nn.LayerNorm(sen_encoder.config.hidden_size)
            if self.config.normalization == "after"
            else nn.Identity(),
            nn.Dropout(self.config.cls_dropout),
            nn.Linear(sen_encoder.config.hidden_size, len(MODELS)),
        )

        self.contrastive_loss = (
            nn.TripletMarginLoss()
            if self.config.ssup == 1
            else nn.CosineEmbeddingLoss()
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, data):
        embs_1 = []
        for d in data:
            embs_1.append(self.encoder(d))

        preds = []
        for emb in embs_1:
            pred = self.classifier(emb)
            preds.append(pred)

        # ? Contrastive loss
        con_loss = []
        if self.config.ssup == 0:
            for i in range(len(embs_1)):
                for j in range(i + 1, len(embs_1)):
                    labels = torch.ones(embs_1[i].shape[0]).long().to(self.device) * -1
                    curr_con_loss = self.contrastive_loss(embs_1[i], embs_1[j], labels)
                    con_loss.append(curr_con_loss)
        else:
            embs_3 = []
            for d in data:
                embs_3.append(self.encoder(d))

            for i in range(len(embs_1)):
                emb_2 = self.encoder(data[i])
                for j in range(i + 1, len(embs_1)):
                    curr_con_loss = self.contrastive_loss(embs_1[i], emb_2, embs_3[j])
                    con_loss.append(curr_con_loss)
        con_loss = torch.stack(con_loss).mean()

        # ? Classification loss
        ce_loss = []
        for i in range(len(preds)):
            labels = torch.ones(preds[i].shape[0]).long().to(self.device) * i
            curr_ce_loss = self.ce_loss(preds[i], labels)
            ce_loss.append(curr_ce_loss)
        ce_loss = torch.stack(ce_loss).mean()

        # ? Total loss
        loss = (
            self.config.con_loss_weight * con_loss
            + self.config.ce_loss_weight * ce_loss
        )

        # ? Metrics
        log_dict = {
            "loss": loss,
            "con_loss": con_loss,
            "ce_loss": ce_loss,
        }

        metrics = []
        for i in range(len(preds)):
            pred = preds[i]
            label = torch.ones(pred.shape[0]).long().to(self.device) * i
            metrics.append(self.get_metrics(pred, label))

        for key in metrics[0].keys():
            cum_value = 0
            for i in range(len(metrics)):
                cum_value += metrics[i][key]
            log_dict[f"{key}"] = cum_value / len(metrics)

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        data, _ = batch
        loss, log_dict = self(data)

        for key, value in log_dict.items():
            self.log(
                f"train/{key}",
                value,
                prog_bar=True,
                batch_size=self.config.batch_size,
                sync_dist=self.config.ddp,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        data, _ = batch
        loss, log_dict = self(data)

        for key, value in log_dict.items():
            self.log(
                f"valid/{key}",
                value,
                prog_bar=True,
                batch_size=self.config.batch_size,
                sync_dist=self.config.ddp,
            )
        return loss

    def predict_step(self, batch, batch_idx):
        data, ids = batch

        text_embedding = self.encoder(data[0])
        cls = self.classifier(text_embedding)
        cls = torch.softmax(cls, dim=-1)
        cls = torch.argmax(cls, dim=-1)
        cls = cls.view(-1)
        cls = cls.detach().cpu().numpy()
        return ids[0], cls

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
