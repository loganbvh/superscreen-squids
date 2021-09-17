#!/bin/bash

#SBATCH --job-name=squid_image
#SBATCH --array=0-95
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --mem=6G

echo "slurm script:"
echo "$(<$0)"

echo "python script:"
echo "$(<image_row.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

python image_row.py \
    --outdir=$HOME/superscreen-squids/scripts/squid-image/results/ \
    --squid-height=0.25 \
    --min-triangles=8000 \
    --x_range="-7, 5" \
    --y_range="-7, 5" \
    --iterations=2

mkdir $HOME/superscreen-squids/scripts/results/$SLURM_ARRAY_JOB_ID/logs
mv slurm-$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID.out $HOME/superscreen-squids/scripts/results/$SLURM_ARRAY_JOB_ID/logs/