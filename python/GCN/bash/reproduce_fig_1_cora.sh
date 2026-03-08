#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gpus=1
#SBATCH --time=0-00:20:00
#SBATCH --output=reproduce_fig_1_cora-%j.out
#SBATCH --error=reproduce_fig_1_cora-%j.err
#SBATCH --job-name=reproduce_fig_1_cora

#SBATCH --container-image="nvcr.io#nvidia/pytorch:22.12-py3"
#SBATCH --container-workdir=/workspace

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)/python/GCN

echo "Running experiment..."
cd python/GCN
python3 reproduce_fig_1_cora.py

if [ -f "figure_1_cora_reproduction_fixed.png" ]; then
    mv figure_1_cora_reproduction_fixed.png figure_1_cora_reproduction_fixed_${SLURM_JOB_ID:-local}.png
fi

echo "Job finished at $(date)"
