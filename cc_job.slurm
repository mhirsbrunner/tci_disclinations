#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --time=01:00:00      # hh:mm:ss for the job
#SBATCH --mail-user=hrsbrnn2@illinois.edu
#SBATCH --mail-type="START","END"

 
module purge 
module load gcc anaconda3
module list  
echo "job is starting on `hostname`"
srun python cc_main_script.py $nx $nz $mass $disc_type $half_sign $spin

