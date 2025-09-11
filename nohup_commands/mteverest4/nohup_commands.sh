#!/bin/bash
set -ex

GC_PROG="graph_coloring"
DTR_PROG="dijkstra_token_ring"
MM_PROG="maximal_matching"


GC_MODEL="lstm_trained_at_2025_08_29_23_08"
DTR_MODEL="lstm_trained_at_2025_08_29_23_28"
MM_MODEL="lstm_trained_at_2025_08_30_05_53"

cd gnn
python cvf_analysis.py --model $MM_MODEL --program maximal_matching --graph-names star_graph_n11
