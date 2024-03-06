#!/bin/bash
#SBATCH --account=cvf-analysis
#SBATCH --time=10:00:00
#SSBATCH --mail-type=ALL
#SSBATCH --mail-user=agaru@uwyo.edu
#SBATCH --output=cvf_%A.log
#SBATCH --nodes=1
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
module load gcc/11.2.0 python/3.10.8

echo "Running the script..."
cd coloring
python graph_coloring_node_effect.py

echo "Done!"
