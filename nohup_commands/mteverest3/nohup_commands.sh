#!/bin/bash
set -ex

GC_PROG="graph_coloring"
DTR_PROG="dijkstra_token_ring"
MM_PROG="maximal_matching"


GC_MODEL="lstm_trained_at_2025_08_29_23_08"
DTR_MODEL="lstm_trained_at_2025_08_29_23_28"
MM_MODEL="lstm_trained_at_2025_08_30_05_53"

CVF_ANALYSISI_DIR="cvf_analysis"
GNN_DIR="gnn"

cd $GNN_DIR

python cvf_analysis.py --model $MM_MODEL --program $MM_PROG --graph-names graph_random_regular_graph_n9_d2
