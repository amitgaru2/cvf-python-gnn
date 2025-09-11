#!/bin/bash
set -ex

GC_PROG="graph_coloring"
DTR_PROG="dijkstra_token_ring"
MM_PROG="maximal_matching"


GC_MODEL="lstm_trained_at_2025_08_29_23_08"
DTR_MODEL="lstm_trained_at_2025_08_29_23_28"
MM_MODEL="lstm_trained_at_2025_08_30_05_53"

CVF_ANALYSIS_DIR="cvf-analysis"
GNN_DIR="gnn"

# cd $GNN_DIR

cd $CVF_ANALYSIS_DIR
python main.py --program $MM_PROG --graph-names graph_powerlaw_cluster_graph_n7 -ml

