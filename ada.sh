#!/bin/bash

#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sroydip1+ada@umbc.edu       # Where to send mail
#SBATCH -D .
#SBATCH --job-name="Fsem8"
#SBATCH --output=run/Fsem8.log
#SBATCH --error=run/Fsem8.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --time=240:00:00
#SBATCH --constraint=rtx_8000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

v=$(git status --porcelain | wc -l)
if [[ $v -gt 200 ]]; then
    echo "Error: uncommited changes" >&2
    exit 1
else
    echo "Success: No uncommited changes"
    $@
fi