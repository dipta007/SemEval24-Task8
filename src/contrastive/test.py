import sys
import jsonlines
import torch
from model.model import ContrastiveModel
import os
import lightning.pytorch as pl
from dataloader import ContrastiveDataModule


ROOT_DIR = "/nfs/ada/ferraro/users/sroydip1/semeval24/task8/checkpoints"

if len(sys.argv) < 2:
    print("Please provide experiment name!")
    sys.exit()

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
config.batch_size = 1

model.eval()

datamodule = ContrastiveDataModule(config)
datamodule.setup("test")

trainer = pl.Trainer()

ids_predicitons = trainer.predict(model, datamodule.test_dataloader())

ids, predicitons = zip(*ids_predicitons)

ids = torch.tensor(ids)
predicitons = torch.tensor(predicitons).squeeze(-1)

print(ids.shape)
print(predicitons.shape)

with jsonlines.open(f"./out/subtask_a_monolingual.jsonl", "w") as writer:
    for id, pred in zip(ids, predicitons):
        writer.write({"id": id.item(), "label": pred.item()})


cmd = f"PYTHONPATH=../../ python ../../subtaskA/scorer/scorer.py --pred_file_path=./out/subtask_a_monolingual.jsonl --gold_file_path=../../data/SubtaskA/subtaskA_dev_monolingual.jsonl"
os.system(cmd)
