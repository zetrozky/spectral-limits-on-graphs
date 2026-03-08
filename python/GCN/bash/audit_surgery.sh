#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gpus=1
#SBATCH --time=0-00:05:00
#SBATCH --output=audit_surgery-%j.out
#SBATCH --error=audit_surgery-%j.err
#SBATCH --job-name=audit_surgery

#SBATCH --container-image="nvcr.io#nvidia/pytorch:22.12-py3"
#SBATCH --container-workdir=/workspace

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)/python/GCN

echo "Running experiment..."
cd python/GCN
python3 audit_surgery.py
mv audit_3_surgery.png audit_3_surgery_${SLURM_JOB_ID:-local}.png

echo "Job finished at $(date)"
