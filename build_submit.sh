#!/usr/bin/env bash
# SLURM batch job to build torchcomms-sparse on Perlmutter.
# Usage: sbatch build_submit.sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu
#SBATCH --qos=premium
#SBATCH --time=02:00:00
#SBATCH --account=m5083_g
#SBATCH --job-name=build_torchcomms
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

set -euo pipefail

module load pytorch/2.8.0
module load nccl/2.24.3
module load cudatoolkit/12.9
# module load PrgEnv-gnu
# module load cray-mpich
# module load craype-accel-nvidia80

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "$SCRIPT_DIR"
mkdir -p logs

bash "$SCRIPT_DIR/build_perlmutter.sh"
