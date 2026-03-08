#!/bin/bash

#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gpus=1
#SBATCH --time=0-00:05:00
#SBATCH --output=fig3b_repro-%j.out
#SBATCH --error=fig3b_repro-%j.err
#SBATCH --job-name=fig3b_run

#SBATCH --container-image="nvcr.io#nvidia/pytorch:22.12-py3"
#SBATCH --container-workdir=/workspace
# usage: sbatch --container-mounts="$PWD:/workspace" trainerscript.sh

# --- Start of execution ---
echo "Job started on $(hostname) at $(date)"

echo "Running reproduce_fig_3b.py..."

python python/GCN/reproduce_fig_3b.py
mv figure_3b_simulation.png figure_3b_simulation_${SLURM_JOB_ID:-local}.png

echo "Job finished at $(date)"
