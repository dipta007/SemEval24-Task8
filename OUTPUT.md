#### Longformer-4096
```
python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path ./longout.jsonl --subtask A --model allenai/longformer-base-4096
```
batch_size = 2

INFO : Prediction file format is correct
INFO : macro-F1=0.62535 micro-F1=0.66580        accuracy=0.66580

Output file: `./data/output/longout.jsonl`

#### Longformer-plagarism baseline
```
python3 subtaskA/baseline/transformer_baseline.py --train_file_path ./data/SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path ./data/SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path ./longpara.jsonl --subtask A --model jpwahle/longformer-base-plagiarism-detection
```
batch_size = 2

INFO : Prediction file format is correct
INFO : macro-F1=0.65383 micro-F1=0.68620        accuracy=0.68620

Output file: `./data/output/longpara.jsonl`