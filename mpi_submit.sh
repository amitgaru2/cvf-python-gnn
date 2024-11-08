#!/bin/bash
#SBATCH --account=cvf-analysis
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agaru@uwyo.edu
#SBATCH --output=cvf_%A.log
#SBATCH --nodes=4
#SBATCH --mem=1000G
#SBATCH --partition=beartooth-hugemem


echo "JobID:         "$SLURM_JOB_ID
echo "JobName:       "$SLURM_JOB_NAME
echo "Partition:     "$SLURM_JOB_PARTITION
echo "Nodes:         "$SLURM_JOB_NUM_NODES
echo "NodeList:      "$SLURM_JOB_NODELIST
echo "Tasks per Node:"$SLURM_TASKS_PER_NODE
echo "CPUs per Node: "$SLURM_JOB_CPUS_PER_NODE
echo "CPUs per Task: "$SLURM_CPUS_PER_TASK
echo "CPUs on Node:  "$SLURM_CPUS_ON_NODE
echo "WorkingDir:    "$SLURM_SUBMIT_DIR

echo "Loading Modules..."
module load miniconda3/23.11.0

echo "Activating virtualenv..."
#conda activate dml_mpi
conda activate /gscratch/agaru/test1234

echo "which python..."
which python

echo "Python version..."
python --version

echo "Creating hostlist file..."
hostlist --expand $SLURM_JOB_NODELIST > hostlist
cat hostlist | while read line; do echo ${line} slots=16; done > hostlist_w_slots

cd 'cvf-analysis'
mpirun --hostfile ../hostlist python graph_coloring_v2_mpi.py graph_powerlaw_cluster_graph_n14

echo "Done!"
