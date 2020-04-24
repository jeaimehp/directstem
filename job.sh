#!/bin/bash
#SBATCH -J myjob           # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 64              # Total # of mpi tasks
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=kcahill@osc.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

module list
module load python2
pwd
date

# Launch MPI code... 

time ibrun python2 pi_csv.py         # Use ibrun instead of mpirun or mpiexec
