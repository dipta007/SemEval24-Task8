![Python Version](https://badgen.net/pypi/python/black)
![GitHub Contributors](https://img.shields.io/github/contributors/dipta007/SemEval24-Task8?style=plastic)
![GitHub Stars](https://img.shields.io/github/stars/dipta007/SemEval24-Task8?style=plastic)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/dipta007/SemEval24-Task8?style=plastic)
![GitHub Forks](https://img.shields.io/github/forks/dipta007/SemEval24-Task8?style=plastic)
![GitHub Last Commit](https://img.shields.io/github/last-commit/dipta007/SemEval24-Task8?style=plastic)
![GitHub Top Language](https://img.shields.io/github/languages/top/dipta007/SemEval24-Task8?style=plastic)
![GitHub Commit Activity](https://img.shields.io/github/commit-activity/m/dipta007/SemEval24-Task8?style=plastic)
![GitHub Followers](https://img.shields.io/github/followers/dipta007?style=plastic)

# HU at SemEval-2024 Task 8A: Can Contrastive Learning Learn Embeddings to Detect Machine-Generated Text?

This is the official implementation of our final submission on SemEval 2024, Task 8.
Paper is available on [arXiv](https://arxiv.org/abs/2402.11815).


## Run Locally

Clone

```bash
  git clone https://github.com/dipta007/SemEval24-task8
```

Go to the project directory

```bash
  cd SemEval24-task8
```

Install dependencies

```bash
  conda env create -f environment.yml 
  conda activate sem24_task8
```

Download Data
```
 gdown https://drive.google.com/drive/folders/1FrhMQ5QvMgaeSgcBmZbk7l_GbU-ga99P -O ./data --folder
```

Run trainer

```bash
  python src/train.py --exp_name=EXP_NAME
```


## Final Model Hyperparameters
```
 'accumulate_grad_batches': 16,
 'batch_size': 2,
 'cls_dropout': 0.6,
 'encoder_type': 'sen',
 'loss_weight_con': 0.7,
 'loss_weight_gen_text': 0.1,
 'loss_weight_text': 0.8,
 'lr': 1e-05,
 'max_doc_len': 64,
 'max_epochs': -1,
 'max_sen_len': 4096,
 'model_name': 'jpwahle/longformer-base-plagiarism-detection',
 'seed': 42,
 'validate_every': 0.04,
 'weight_decay': 0.0
```