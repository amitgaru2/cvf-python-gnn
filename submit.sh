#!/bin/bash
#SBATCH --account=cvf-analysis
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=agaru@uwyo.edu
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
#cd coloring
#python graph_coloring_node_effect.py

cd 'cvf-analysis'
#python main.py --program maximal_matching --graph_names graph_1 graph_2 
#python main.py --program maximal_matching --graph_names graph_3 graph_4
#python main.py --program maximal_matching --graph_names graph_5 graph_6 
python main.py -f --program maximal_independent_set --graph-names graph_1 graph_2 graph_3 graph_4 graph_5 graph_6 graph_7 graph_8
#python main.py --program coloring --graph-names graph_6 graph_7
#python main.py --program maximal_matching --graph_names graph_1 --sample-size 10000
#python main.py --program maximal_matching --graph_names graph_1 graph_2 graph_3 graph_4 graph_5 graph_6 graph_7 graph_8

echo "Done!"
