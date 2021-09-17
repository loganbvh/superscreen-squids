#!/bin/bash

#SBATCH --job-name=squid_image
#SBATCH --array=0-47
#SBATCH --output=slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G

echo "slurm script:"
echo "$(<$0)"

echo "python script:"
echo "$(<image_row.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

python image_row.py \
    --outdir=$SCRATCH/image_squid/ \
    --squid_height=0 \
    --min_triangles=8000 \
    --x_range="-7, 5" \
    --y_range="-7, 5" \
    --iterations=2
