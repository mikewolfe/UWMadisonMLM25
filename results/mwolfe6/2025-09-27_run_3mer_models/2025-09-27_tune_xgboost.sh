#!/bin/bash

#SBATCH -N 1
#SBATCH -c 20
#SBATCH -t 2-00:00:00
#SBATCH --mem=400G
#SBATCH -o batch_xgboost_models.%j.out
#SBATCH -e batch_xgboost_models.%j.err

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export TMPDIR=/mnt/scratch/group/sjmcilwain/mwolfe6/UWMadisonMLM25/results/mwolfe6/2025-09-27_run_3mer_models/tmp

Rscript 2025-09-27_tune_xgboost.R 2025-09-27_test20 20 20
