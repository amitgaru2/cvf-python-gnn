#!/bin/bash
set -ex

# conda activate cvf

# cd cvf-analysis/v2
# python main.py --program maximal_matching --graph-names graph_10 -ml

cd simulations

# PROGRAM="graph_coloring"
# PROGRAM="dijkstra_token_ring"
PROGRAM="maximal_matching"

GRAPH="graph_7"
# GRAPH="implicit_graph_n10"

NO_SIMS=10000
LIMIT_STEPS=100
FAULT_INTERVALS=(2 3 4 5)
SIMULATION_TYPE="controlled_at_node_amit_v2"

for FI in "${FAULT_INTERVALS[@]}"; do
    python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 0 --limit-steps $LIMIT_STEPS
    python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 4 --limit-steps $LIMIT_STEPS
    python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 9 --limit-steps $LIMIT_STEPS
done

GRAPH="graph_10"

for FI in "${FAULT_INTERVALS[@]}"; do
    python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 0 --limit-steps $LIMIT_STEPS
    python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 4 --limit-steps $LIMIT_STEPS
    python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 9 --limit-steps $LIMIT_STEPS
done

# SIMULATION_TYPE="controlled_at_node_amit_v2"

# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 0 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 4 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-node 9 --limit-steps $LIMIT_STEPS

# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type random --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 0 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 1 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 2 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 3 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 4 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 5 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 6 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 7 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 7 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 8 --limit-steps $LIMIT_STEPS
# python simulate.py --program $PROGRAM --sched 0 --no-sim $NO_SIMS --fault-interval $FAULT_INTERVAL --graph-names $GRAPH --fault-prob 1.0 --simulation-type controlled_at_node_duong --controlled-at-node 9 --limit-steps $LIMIT_STEPS

# cd gnn
# python cvf_analysis.py lstm_trained_at_2025_05_12_21_31 dijkstra implicit_graph_n14
# python cvf_analysis.py lstm_trained_at_2025_04_26_14_01 coloring graph_random_regular_graph_n10_d3
# python cvf_analysis.py lstm_trained_at_2025_05_15_11_29 $PROGRAM star_graph_n7
# python cvf_analysis.py lstm_trained_at_2025_05_15_11_29 $PROGRAM graph_powerlaw_cluster_graph_n7
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

# cd gnn

# # # # # graphs=("star_graph_n7" "graph_powerlaw_cluster_graph_n7" "graph_random_regular_graph_n7_d4" "star_graph_n13" "graph_powerlaw_cluster_graph_n8" "graph_random_regular_graph_n8_d4" "star_graph_n15" "graph_powerlaw_cluster_graph_n9")
# graphs=("star_graph_n4" "star_graph_n5")
# # # # # graphs=("graph_powerlaw_cluster_graph_n6" "graph_random_regular_graph_n6_d3")
# # # # # # # # # graphs=("implicit_graph_n5" "implicit_graph_n6" "implicit_graph_n7" "implicit_graph_n8" "implicit_graph_n9" "implicit_graph_n10")
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
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n4
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n5
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n6
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n7
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n8
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_17_35 --program maximal_matching --graph-names star_graph_n9
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_13_36 --program maximal_matching --graph-names graph_powerlaw_cluster_graph_n5 graph_powerlaw_cluster_graph_n6
# python cvf_analysis.py --model lstm_trained_at_2025_06_27_13_36 --program maximal_matching --graph-names graph_powerlaw_cluster_graph_n5 graph_powerlaw_cluster_graph_n6
# mpirun -n 4 python cvf_analysis_mpi.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n7_d4

# python cvf_analysis.py --model lstm_trained_at_2025_04_26_14_01 --program coloring --graph-names graph_10
# mpirun --hostfile hosts.txt ./run_mpi.sh cvf_analysis_mpi.py --model lstm_trained_at_2025_06_26_20_32 --program maximal_matching --graph-names graph_random_regular_graph_n8_d4
