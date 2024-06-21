#!/bin/bash
set -e
source venv/bin/activate
cd cvf-analysis
#python3 main.py --program maximal_matching --graph-names graph_5 graph_6 graph_7 graph_8
python3 main.py --program dijkstra_token_ring --graph-names 3 4 5 6 7 8 9 10 11 12
python3 main.py --program dijkstra_token_ring -f --graph-names 3 4 5 6 7 8 9 10 11 12
