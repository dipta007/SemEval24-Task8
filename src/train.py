import lightning.pytorch as pl
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from dataloader.sen_sim import ContrastiveDataModule
from model.model import ContrastiveModel
from lightning.pytorch.loggers import WandbLogger
import base_config as config
import shutil
import os
import wandb

wandb.login()

config = config.get_config()
print(config)

device = "gpu" if torch.cuda.is_available() else "cpu"
print("\ndevice :", device)
config.device = "cuda" if device == "gpu" else "cpu"


def main():
    monitoring_metric = "valid/text_acc"
    monitoring_mode = "max"
    checkpoint_dir = (
        f"/nfs/ada/ferraro/users/sroydip1/semeval24/task8/checkpoints/{config.exp_name}"
    )
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    L.seed_everything(config.seed)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model_{}={{{}:.2f}}".format(
                monitoring_metric.replace("/", "_"), monitoring_metric
            ),
            auto_insert_metric_name=False,
            monitor=f"{monitoring_metric}",
            mode=monitoring_mode,
            verbose=True,
            save_top_k=1,
            save_on_train_epoch_end=False,
            enable_version_counter=False,
        ),
        EarlyStopping(
            monitor=f"{monitoring_metric}",
            mode=monitoring_mode,
            verbose=True,
            patience=10,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(leave=True),
    ]
    loggers = [
        WandbLogger(
            entity="gcnssdvae",
            project="sem8",
            log_model=False,
            name=config.exp_name if config.exp_name != 'sweep' else None,
            tags=["paper"],
        )
    ]
    if config.debug:
        loggers = False

    if loggers:
        loggers[0].experiment.define_metric("valid/text_acc", summary="max")

    print("Loading data")
    datamodule = ContrastiveDataModule(config)

    print("Loading model")
    model = ContrastiveModel(config, datamodule.tokenizer)
    print("Training")
    trainer = pl.Trainer(
        accelerator=device,
        logger=loggers,
        callbacks=callbacks,
        fast_dev_run=False,
        val_check_interval=config.validate_every,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches // config.batch_size if config.batch_size < config.accumulate_grad_batches else 1,
        log_every_n_steps=1,
        overfit_batches=config.overfit if config.overfit != 0 else 0.0,
    )

    print("Fitting")
    trainer.fit(model=model, datamodule=datamodule)



if __name__ == "__main__":
    main()