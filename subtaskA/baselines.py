import os
from tqdm.auto import tqdm


CMD = """
python baseline/transformer_baseline.py -tr ../data/SubtaskA/subtaskA_train_monolingual.jsonl -t ../data/SubtaskA/subtaskA_test_monolingual.jsonl -p ../data/out/baselines/{file_path}.jsonl -sb A -m={model_name} -bsz={batch_size} > ./score/{model_name}.log
"""

MODELS = [
    "jpwahle/longformer-base-plagiarism-detection",
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
]

for model_name in tqdm(MODELS):
    try:
        print(f"Running {model_name}")
        batch_size = 8 if "large" in model_name else 16
        batch_size = 2 if "longformer" in model_name else batch_size

        cmd = CMD.format(file_path=model_name.replace("/", "_"), model_name=model_name, batch_size=batch_size)
        print(model_name)
        os.system(cmd)
        os.system(f"rm -rf ./{model_name}")
    except Exception as e:
        print(f"Error in {model_name}: {e}")


