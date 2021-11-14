#!/bin/bash

#SBATCH --job-name=squid_image_leads
#SBATCH --array=0-65
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

echo "slurm script:"
echo "$(<$0)"

echo "python script:"
echo "$(<image_row_leads.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

outdir=$HOME/superscreen-squids/scripts/squid-image/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir/logs

python image_row_leads.py \
    --outdir=$outdir \
    --squid-height=-0.05 \
    --sample-min-triangles=12000 \
    --squid-min-triangles=10000 \
    --sample-temperature=4.0 \
    --x_range="-12, 12" \
    --y_range="-12, 12" \
    --iterations=10

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
