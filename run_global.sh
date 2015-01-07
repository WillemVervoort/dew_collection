#!/bin/bash -l
#SBATCH -J dewpot
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 14:00:00
#SBATCH -n 16
#SBATCH --mem-per-cpu=4000
#SBATCH -p parallel

srun $USERAPPL/bin/jug execute daily_dew_jug.py --jugdir=jugdata

/appl/bin/used_slurm_resources.bash
