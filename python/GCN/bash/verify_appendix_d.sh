#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gpus=1
#SBATCH --time=0-00:15:00
#SBATCH --output=verify_app_d-%j.out
#SBATCH --error=verify_app_d-%j.err
#SBATCH --job-name=verify_app_d

#SBATCH --container-image="nvcr.io#nvidia/pytorch:22.12-py3"
#SBATCH --container-workdir=/workspace

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)/python/GCN

echo "Running experiment..."
cd python/GCN
python3 verify_appendix_d.py "$@"

echo "Job finished at $(date)"
