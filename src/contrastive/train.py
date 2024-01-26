import lightning.pytorch as pl
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from dataloader.doc_sim import ContrastiveDocDataModule
from dataloader.sen_sim_pair import ContrastiveDataModule
from model.model import ContrastiveModel
from lightning.pytorch.loggers import WandbLogger
import base_config as config
import shutil
import wandb

wandb.login()

config = config.get_config()
print(config)

device = "gpu" if torch.cuda.is_available() else "cpu"
print("\ndevice :", device)
config.device = "cuda" if device == "gpu" else "cpu"


def main():
    monitoring_metric = "valid/acc"
    monitoring_mode = "max"
    checkpoint_dir = (
        f"/nfs/ada/ferraro/users/sroydip1/semeval24/task8/subtaskB/{config.exp_name}"
    )
    if config.exp_name != 'sweep':
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
            save_top_k=(1 if config.exp_name != 'sweep' else 0),
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
            project="sem8B",
            log_model=False,
            name=config.exp_name if config.exp_name != 'sweep' else None,
        )
    ]
    if config.debug:
        loggers = False

    if loggers:
        loggers[0].experiment.define_metric("valid/acc", summary="max")

    print("Loading data")
    if config.encoder_type == "sen":
        datamodule = ContrastiveDataModule(config)
    elif config.encoder_type == "doc":
        datamodule = ContrastiveDocDataModule(config)
    else:
        raise ValueError("Encoder type not found")

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
        reload_dataloaders_every_n_epochs=1
    )

    print("Fitting")
    trainer.fit(model=model, datamodule=datamodule)



if __name__ == "__main__":
    main()