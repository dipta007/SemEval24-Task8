#!/bin/bash
for (( i=1; i<=$1; i++ ))
do
    echo "Running $i"
    sbatch ada.sh wandb agent $2
done