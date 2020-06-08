#!/usr/bin/bash
# Job submission script for 5 replicas for SMD of 100-Alanine
# Contact: Daipayan Sarkar
# Email: dsarkar@asu.edu

if [ -z ${array} ] ; then
    array=000-004
fi


sbatch \
    --job-name=smd-slow-100ala --array=${array} \
    ${slurm_depend:+--depend=afterany:${slurm_depend}} \
    --partition=parallel -q normal \
    -N 1 \
    -n 8 \
    --tasks=$((1*8)) \
    --ntasks-per-core=1 --exclusive \
    --time=24:00:00 \
<<EOF
#!/usr/bin/bash

cd \${SLURM_SUBMIT_DIR}

exec bash run.sh

EOF
