#!/bin/bash
set -ex

# conda activate cvf

# cd cvf-analysis/v2
# python main.py --program maximal_matching --graph-names graph_random_regular_graph_n8_d4 -ml

# cd simulations
# python simulate.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 1 --graph-names implicit_graph_n8 --fault-prob 1.0 --simulation-type controlled_at_node --controlled-at-node 7
# python simulate.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 1 --graph-names implicit_graph_n8 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 0
# python simulate.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 1 --graph-names implicit_graph_n8 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 2
# python simulate.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 1 --graph-names implicit_graph_n8 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 7
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type random
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node --controlled-at-node 0
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node --controlled-at-node 3
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node --controlled-at-node 6
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 0
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 3
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names star_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 6
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names graph_powerlaw_cluster_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 1
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names graph_powerlaw_cluster_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 4
# python simulate.py --program maximal_matching --sched 0 --no-sim 500000 --fault-interval 1 --graph-names graph_powerlaw_cluster_graph_n7 --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 5

# cd gnn
# python cvf_analysis.py lstm_trained_at_2025_05_12_21_31 dijkstra implicit_graph_n14
# python cvf_analysis.py lstm_trained_at_2025_04_26_14_01 coloring graph_random_regular_graph_n10_d3
# python cvf_analysis.py lstm_trained_at_2025_05_15_11_29 maximal_matching star_graph_n7
# python cvf_analysis.py lstm_trained_at_2025_05_15_11_29 maximal_matching graph_powerlaw_cluster_graph_n7
# python cvf_analysis.py lstm_trained_at_2025_05_15_11_29 maximal_matching graph_random_regular_graph_n7_d4

# # # # graphs=("star_graph_n7" "graph_powerlaw_cluster_graph_n7" "graph_random_regular_graph_n7_d4" "star_graph_n13" "graph_powerlaw_cluster_graph_n8" "graph_random_regular_graph_n8_d4" "star_graph_n15" "graph_powerlaw_cluster_graph_n9")
# graphs=("star_graph_n7" "graph_powerlaw_cluster_graph_n7")
# # graphs=("star_graph_n7")
# # graphs=("implicit_graph_n5" "implicit_graph_n6" "implicit_graph_n7" "implicit_graph_n8" "implicit_graph_n9" "implicit_graph_n10")
# joined_graphs_args="${graphs[@]}"

# epochs=10
# batch_size=1024
# hidden_size=32

# python main.py --program graph_coloring --sched 0 --no-sim 100000 --fault-interval 1 --graph-names graph_7 --fault-prob 1.0
# python main.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 1 --graph-names implicit_graph_n5 --fault-prob 1.0
# python main.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 2 --graph-names implicit_graph_n5 --fault-prob 1.0
# python main.py --program dijkstra_token_ring --sched 0 --no-sim 500000 --fault-interval 4 --graph-names implicit_graph_n5 --fault-prob 1.0

cd gnn

# # # # # graphs=("star_graph_n7" "graph_powerlaw_cluster_graph_n7" "graph_random_regular_graph_n7_d4" "star_graph_n13" "graph_powerlaw_cluster_graph_n8" "graph_random_regular_graph_n8_d4" "star_graph_n15" "graph_powerlaw_cluster_graph_n9")
# graphs=("graph_random_regular_graph_n4_d2" "graph_random_regular_graph_n5_d4" "graph_random_regular_graph_n6_d3")
# # # graphs=("graph_powerlaw_cluster_graph_n6" "graph_random_regular_graph_n6_d3")
# # # # # # # graphs=("implicit_graph_n5" "implicit_graph_n6" "implicit_graph_n7" "implicit_graph_n8" "implicit_graph_n9" "implicit_graph_n10")
# joined_graphs_args="${graphs[@]}"

# epochs=50
# batch_size=256
# hidden_size=32

# python lstm_scratch.py \
#     --program maximal_matching \
#     --epochs $epochs \
#     --batch-size $batch_size \
#     --hidden-size $hidden_size \
#     --num-layers 2 \
#     --graph-names $joined_graphs_args

# # python gcn_scratch.py \
# #     --epochs $epochs \
# #     --batch-size $batch_size \
# #     --hidden-size $hidden_size \
# #     --graph-names $joined_graphs_args

# # cd gnn/new_ideas
# #
# # python bert_scratch.py
# # python transformer_w_same_node_seql.py 50

# cd gnn

# python cvf_analysis.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n4_d2
# python cvf_analysis.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n5_d4
# python cvf_analysis.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n6_d3
# python cvf_analysis.py --model lstm_trained_at_2025_06_25_19_06 --program maximal_matching --graph-names graph_powerlaw_cluster_graph_n7
python cvf_analysis.py --model lstm_trained_at_2025_06_26_12_01 --program maximal_matching --graph-names graph_random_regular_graph_n7_d4
