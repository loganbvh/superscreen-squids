#!/bin/bash

#SBATCH --job-name=psf
#SBATCH --array=0-80
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=6G

echo "slurm script:"
echo "$(<$0)"

echo "python script:"
echo "$(<psf_row.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

outdir=$HOME/superscreen-squids/scripts/psf/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir/logs

python psf_row.py \
    --outdir=$outdir \
    --squid-type="ibm-small" \
    --squid-height=0.175 \
    --align-layers="middle" \
    --min-points=5000 \
    --x-range="-2, 2" \
    --y-range="-2, 2" \
    --iterations=4

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
