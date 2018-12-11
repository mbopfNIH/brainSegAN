#!/bin/bash

#SBATCH --workdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=brainSegAN
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx1 

export PATH=/slurm_storage/public/cuda8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/slurm_storage/public/cuda8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Dynamically create outputs directory
mkdir outputs_$SLURM_JOB_ID

#env | sort
#nvidia-smi
#nvcc --version
time python train_adversarial.py --cuda --batchsize $batchsize --decay $decay --lr $lr --niter $niter --outpath ./outputs_$SLURM_JOB_ID
