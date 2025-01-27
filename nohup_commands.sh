#!/bin/bash
set -e
# source venv/bin/activate
# cd cvf-analysis
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
# python3 main.py --program linear_regression -f --graph-names test_lr_graph_1 --logging DEBUG
# python graph_coloring_v2.py

# cd gnn
# python vanilla_gcn_generic.py

cd simulations/cvf-analysis
python main.py --program graph_coloring --sched 0 --no-sim 100000 --fault-prob 0.5 --graph-names graph_powerlaw_cluster_graph_n30
python main.py --program graph_coloring --sched 1 --no-sim 100000 --fault-prob 0.5 --graph-names graph_powerlaw_cluster_graph_n30
python main.py --program graph_coloring --sched 1 -me --no-sim 100000 --fault-prob 0.5 --graph-names graph_powerlaw_cluster_graph_n30
#python main.py --program graph_coloring --sched 0 --no-sim 100000 --fault-prob 0.5 --graph-names graph_powerlaw_cluster_graph_n29
#python main.py --program graph_coloring --sched 1 --no-sim 100000 --fault-prob 0.5 --graph-names graph_powerlaw_cluster_graph_n29
#python main.py --program graph_coloring --sched 1 -me --no-sim 100000 --fault-prob 0.5 --graph-names graph_powerlaw_cluster_graph_n29
