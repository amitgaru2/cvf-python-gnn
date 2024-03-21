#!/bin/bash
set -e
source venv/bin/activate
cd cvf-analysis
python3 main.py --program maximal_matching --graph-names graph_5 graph_6 graph_7 graph_8
