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

outdir=$HOME/superscreen-squids/scripts/squid-image/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir/logs

python image_row.py \
    --outdir=$outdir \
    --squid-height=0.05 \
    --min-triangles=8000 \
    --x_range="-15, 15" \
    --y_range="-15, 15" \
    --iterations=2

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
