#!/bin/bash
#SBATCH --account=cvf-analysis
#SBATCH --time=72:00:00
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
#python main.py --program maximal_matching --graph-names graph_1 graph_2 graph_3 graph_4 graph_5 graph_6 graph_7 graph_8
#python main.py --program graph_coloring -f --graph-names graph_6b
#python main.py --program maximal_matching -f --graph-names graph_6b
#python main.py -f --program maximal_independent_set --graph-names graph_1 graph_2 graph_3 graph_4 graph_5 graph_6 graph_7 graph_8 graph_6b
#python main.py --program graph_coloring --graph-names graph_3 graph_4 graph_5 graph_8
#python main.py --program maximal_matching --graph_names graph_1 --sample-size 10000
#python main.py --program maximal_matching --graph_names graph_1 graph_2 graph_3 graph_4 graph_5 graph_6 graph_7 graph_8

#python3 main.py --program maximal_matching --graph-names graph_5 graph_6 graph_7 graph_8
#python3 main.py --program dijkstra_token_ring --graph-names 13 14 15 16 17 18 19
#python3 main.py --program maximal_matching --graph-names graph_1
#python3 main.py --program maximal_matching --graph-names graph_2
#python3 main.py --program maximal_matching --graph-names graph_3
#python3 main.py --program maximal_matching --graph-names graph_4
#python3 main.py --program maximal_matching --graph-names graph_5
#python3 main.py --program maximal_matching --graph-names graph_6
#python3 main.py --program maximal_matching --graph-names graph_7
#python3 main.py --program maximal_matching --graph-names graph_8
python3 main.py --program linear_regression -f --graph-names test_lr_graph_1 --logging DEBUG --config-file matrix_4

echo "Done!"
