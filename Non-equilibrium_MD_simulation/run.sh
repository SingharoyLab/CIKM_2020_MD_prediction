# Run NAMD script for 5 replicas for SMD of 100-Alanine
# Contact: Daipayan Sarkar
# Email: dsarkar@asu.edu

export run_index=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

# load your NAMD module or path to NAMD executable
module load namd/2.13-mpi 

# launch mutiple namd replica jobs
mpirun -np 8 namd2 smd.constvel.namd >& 100ala-smd-${run_index}.out
