#!/bin/bash
#SBATCH --account=cvf-analysis
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agaru@uwyo.edu
#SBATCH --output=cvf_%A.log
#SBATCH --nodes=6
#SBATCH --mem=128G
#SBATCH --partition=teton-gpu
#SBATCH --gres=gpu:1

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
module load miniconda3/24.3.0

echo "Activating virtualenv..."
conda activate /project/cvf-analysis/agaru/envs/cvf-env

echo "which python..."
which python

echo "Python version..."
python --version

echo "Creating hostlist file..."
hostlist --expand $SLURM_JOB_NODELIST >hostlist
cat hostlist | while read line; do echo ${line} slots=16; done >hostlist_w_slots

cd 'cvf-analysis'
mpirun --hostfile ../hostlist python cvf_analysis.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n7_d4

echo "Done!"
