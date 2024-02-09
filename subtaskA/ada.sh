#!/bin/bash

#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sroydip1+ada@umbc.edu       # Where to send mail
#SBATCH -D .
#SBATCH --job-name="sem8B"
#SBATCH --output=run/sem8B.log
#SBATCH --error=run/sem8B.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=70000
#SBATCH --time=240:00:00
#SBATCH --constraint=rtx_6000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

v=$(git status --porcelain | wc -l)
if [[ $v -gt 200 ]]; then
    echo "Error: uncommited changes" >&2
    exit 1
else
    echo "Success: No uncommited changes"
    # echo "CMD:" $@ ++debug=False
    # $@ ++debug=False
    python baselines.py
fi