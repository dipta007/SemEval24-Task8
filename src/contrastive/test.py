import sys
import jsonlines
import torch
from model.model import ContrastiveModel
import os
import lightning.pytorch as pl
from dataloader.sen_sim_pair import ContrastiveDataModule
from dataloader.doc_sim import ContrastiveDocDataModule


ROOT_DIR = "/nfs/ada/ferraro/users/sroydip1/semeval24/task8/subtaskB"

if len(sys.argv) < 2:
    print("Please provide experiment name!")
    sys.exit()

final = ""
if len(sys.argv) > 2:
    final = "_final"

exp_name = sys.argv[1]
files = os.listdir(f"{ROOT_DIR}/{exp_name}")

if len(files) == 0:
    print("No checkpoints found!")
    sys.exit()

index = 0
if len(files) > 1:
    print("Multiple checkpoints found!")
    for i, file in enumerate(files):
        print(f"{i}: {file}")
    index = int(input("Enter checkpoint index: "))

file_name = files[index]

model = ContrastiveModel.load_from_checkpoint(
    f"{ROOT_DIR}/{exp_name}/{file_name}",
)

config = model.config
config.batch_size = 16

print(config)

model.eval()

if config.encoder_type == "doc":
    datamodule = ContrastiveDocDataModule(config)
elif config.encoder_type == "sen":
    datamodule = ContrastiveDataModule(config)
else:
    raise ValueError(f"Invalid encoder type: {config.encoder_type}")

datamodule.setup(f"test{final}")

trainer = pl.Trainer()

ids_predicitons = trainer.predict(model, datamodule.test_dataloader())


ids, predicitons = zip(*ids_predicitons)

ids = [id.clone().detach() for id in ids]
predicitons = [torch.tensor(pred) for pred in predicitons]

ids = torch.cat(ids)
predicitons = torch.cat(predicitons)

print(ids.shape)
print(predicitons.shape)

out_dir = "./out"
if final:
    out_dir = "./final"

with jsonlines.open(f"{out_dir}/out_{exp_name}.jsonl", "w") as writer:
    for id, pred in zip(ids, predicitons):
        writer.write({"id": id.item(), "label": pred.item()})

if not final:
    cmd = f"PYTHONPATH=../../ python ../../subtaskB/scorer/scorer.py --pred_file_path=./out/out_{exp_name}.jsonl --gold_file_path=./data/SubtaskB/subtaskB_dev.jsonl"
    os.system(cmd)
