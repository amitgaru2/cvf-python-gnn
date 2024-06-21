#!/bin/bash
set -e
source venv/bin/activate
cd cvf-analysis
#python3 main.py --program maximal_matching --graph-names graph_5 graph_6 graph_7 graph_8
#python3 main.py --program dijkstra_token_ring --graph-names 13 14 15 16 17 18 19
python3 main.py --program dijkstra_token_ring -f --graph-names 13 14 
