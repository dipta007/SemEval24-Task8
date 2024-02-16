import sys
import jsonlines
import torch
from model.model import ContrastiveModel
import os
from pprint import pprint
import lightning.pytorch as pl
from dataloader.sen_sim import ContrastiveDataModule


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
print("="*50)
print("==>Config: ")
pprint(vars(config))
print("="*50)
config.batch_size = 16


print("==>Config: ")
pprint(vars(config))
print("="*50)

model.eval()
datamodule = ContrastiveDataModule(config)
datamodule.setup("test_final")

trainer = pl.Trainer()

ids_predicitons = trainer.predict(model, datamodule.test_dataloader())


ids, predicitons = zip(*ids_predicitons)

ids = [id.clone().detach() for id in ids]
predicitons = [torch.tensor(pred) for pred in predicitons]

ids = torch.cat(ids)
predicitons = torch.cat(predicitons)


# print(ids.shape)
# print(predicitons.shape)

with jsonlines.open(f"./tmp_{exp_name}.jsonl", "w") as writer:
    for id, pred in zip(ids, predicitons):
        writer.write({"id": id.item(), "label": pred.item()})

print(f"==>Scoring for {exp_name}")

# for test
cmd = f"PYTHONPATH=../ python ./subtaskA/scorer/scorer.py --pred_file_path=./tmp_{exp_name}.jsonl --gold_file_path=./data/aug_data/SubtaskA/monolingual/ibm/test.jsonl"

# for test_final
cmd = f"PYTHONPATH=../ python ./subtaskA/scorer/scorer.py --pred_file_path=./tmp_{exp_name}.jsonl --gold_file_path=./data/SubtaskA/subtaskA_test_monolingual.jsonl"

cmd += f" > ./data/final_test/{exp_name}.txt"
os.system(cmd)
print("="*50)
print()
