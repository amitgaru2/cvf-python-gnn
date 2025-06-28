#!/bin/bash
set -ex

conda init

conda activate cvf 

python cvf_analysis_mpi.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n7_d4
