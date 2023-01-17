#!/bin/bash
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbhg-delta-gpu
#SBATCH --time=00:05:00      # hh:mm:ss for the job
##SBATCH --mail-user=hrsbrnn2@illinois.edu
##SBATCH --mail-type="END"

 
module purge 
module load cuda gcc anaconda3
module list  
echo "job is starting on `hostname`"
srun python main_script.py $nx $nz $mass $half_model $other_half $spin

