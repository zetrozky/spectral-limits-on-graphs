# Evaluating Shi et al. (PNAS 2024): Spectral Limits on Graphs

This repo contains code to reproduce and extend the results of Shi et al. (PNAS 2024).
Made for the Analysis of new phenomena in Machine/Deep Learning Course, Winter 2025/2026 by Avantica Vempati.

## 1. Reproductions (Section 2)

- `reproduce_fig_3b.py`: Generates the regularized CSBM double descent curve (Figure 3B from Shi et al./ Figure 1).
- `reproduce_fig_1_cora.py`: Reproduces Cora benchmark (Figure 1 from Shi et al.).
- `reproduce_fig_2d.py`: Reproduces the noise baseline on Chameleon (Figure 2d from Shi et al.)

## 2. Extensions & Audits (Section 3)

- `verify_appendix_d.py`: $c$-sweep, MLP limit divergence (Figure 2).
- `audit_noise.py`: Structural noise injection, MLP boundary condition (Figure 3).
- `audit_surgery.py`: O(n^3) Band-Stop Spectral filter (Figure 4) and evaluation on Platonov deduplicated Squirrel dataset.

## Requirements

This codebase requires PyTorch and PyTorch Geometric. To install the dependencies locally or within your container:

```bash
pip install -r requirements.txt
```

## 3. Instructions

Run the shell scripts using sbatch on a SLURM cluster.

```bash
sbatch --container-mounts="$PWD:/workspace" python/GCN/bash/script.sh --args
```

Modify the scripts as needed to adjust parameters or run them on your specific SLURM cluster.

Folder structure was preserved from original paper to make life easier.
