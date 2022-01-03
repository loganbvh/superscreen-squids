#!/bin/bash

#SBATCH --job-name=squid_image
#SBATCH --array=0-80
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

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
    --squid-height=-0.035 \
    --min-triangles=13000 \
    --optimesh-steps=50 \
    --x_range="-4, 4" \
    --y_range="-5, 5" \
    --iterations=5

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
