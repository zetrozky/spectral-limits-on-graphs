#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gpus=1
#SBATCH --time=0-00:20:00
#SBATCH --output=reproduce_fig_2d-%j.out
#SBATCH --error=reproduce_fig_2d-%j.err
#SBATCH --job-name=reproduce_fig_2d

#SBATCH --container-image="nvcr.io#nvidia/pytorch:22.12-py3"
#SBATCH --container-workdir=/workspace

echo "Job started on $(hostname) at $(date)"

export PYTHONPATH=$PYTHONPATH:$(pwd)/python/GCN

echo "Running experiment..."
cd python/GCN
python3 reproduce_fig_2d.py "$@"

if [ -f "figure_2d.png" ]; then
    mv figure_2d.png figure_2d_${SLURM_JOB_ID:-local}.png
fi

echo "Job finished at $(date)"
