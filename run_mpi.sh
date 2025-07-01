#!/usr/bin/zsh
set -e

host=$(hostname)

echo "Hostname: "$host" | Rank: "$RANK" Shell: "$SHELL

source ~/anaconda3/etc/profile.d/conda.sh

conda activate cvf

cd gnn

# # python cvf_analysis_mpi.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n7_d4

# # exec python cvf_analysis_mpi.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n6
python "$@"
