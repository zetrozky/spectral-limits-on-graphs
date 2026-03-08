#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gpus=1
#SBATCH --time=0-00:10:00
#SBATCH --output=verify_app_d-%j.out
#SBATCH --error=verify_app_d-%j.err
#SBATCH --job-name=verify_app_d

#SBATCH --container-image="nvcr.io#nvidia/pytorch:22.12-py3"
#SBATCH --container-workdir=/workspace

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)/python/

echo "Running experiment..."
cd python/GCN
python3 audit_noise.py
mv audit_2_noise.png audit_2_noise_${SLURM_JOB_ID:-local}.png

echo "Job finished at $(date)"
