#!/bin/bash
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sroydip1@umbc.edu             # Where to send mail
#SBATCH -D .
#SBATCH --job-name=jupyter
#SBATCH --output=./run/jupyter.log
#SBATCH --error=./run/jupyter.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --time=72:00:00
#SBATCH --constraint=rtx_6000                   # NULL (12GB), rtx_6000 (24GB), rtx_8000 (48GB)

port=8888
node=$(hostname -s)
user=$(whoami)

jupyter-lab --no-browser --port=${port} --ip=${node}