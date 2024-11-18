#!/bin/bash
#SBATCH --job-name=bro-serpentin # Job name
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#SBATCH --tasks=1 # Number of tasks
#SBATCH --gpus-per-node=1 # Require GPUs
#SBATCH --time=0-00:10 # Maximum runtime (D-HH:MM)
#SBATCH --nodelist=calypso[0,1]

./run_serpentin