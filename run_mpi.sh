#!/usr/bin/zsh
set -e

echo "Hostname: "$(hostname)" | Shell: "$SHELL

source ~/anaconda3/etc/profile.d/conda.sh

git pull origin gnn5 --rebase

conda activate cvf

cd gnn

# python cvf_analysis_mpi.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n7_d4

python cvf_analysis_mpi.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n7
