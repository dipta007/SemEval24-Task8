#!/bin/bash
#SBATCH --mail-type=ALL                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sroydip1+ada@umbc.edu       # Where to send mail
#SBATCH -D .
#SBATCH --job-name="jupyter"
#SBATCH --output=log/output/jupyter.log
#SBATCH --error=log/error/jupyter.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --time=240:00:00

port=8888
node=$(hostname -s)
user=$(whoami)


jupyter-lab --no-browser --port=${port} --ip=${node}

# ssh -N -L <local_port>:<node_nodelist(g12)>:<port> <user>@<server>