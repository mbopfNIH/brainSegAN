#!/bin/bash

#SBATCH --workdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=brainSegAN
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx1

#export PATH=/slurm_storage/public/cuda8.0/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/slurm_storage/public/cuda8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/slurm_storage/public/cuda9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/slurm_storage/public/cuda9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Dynamically create outputs directory
mkdir outputs_$SLURM_JOB_ID

env | sort
nvidia-smi
nvcc --version
python train_adversarial.py --cuda --batchsize 4 --niter 200 --outpath ./outputs_$SLURM_JOB_ID
