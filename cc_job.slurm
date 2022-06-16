#!/bin/bash
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --time=04:00:00      # hh:mm:ss for the job
#SBATCH --mail-user=hrsbrnn2@illinois.edu
#SBATCH --mail-type="START","END"

 
module purge
module load gcc anaconda/3

.  /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate tci_env

echo "job is starting on `hostname`"
srun python cc_main_script.py $nz $nx $mass $disc_type $half_sign $spin

