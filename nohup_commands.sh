#!/bin/bash
set -ex

GRAPH_COLORING_PROGRAM="graph_coloring"
DIJKSTRA_TOKEN_PROGRAM="dijkstra_token_ring"
MAX_MATCHING_PROGRAM="maximal_matching"


# cd cvf-analysis

# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names star_graph_n10 -ml
# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n10_d4 -ml
# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n11_d4 -ml
# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n12_d4 -ml

# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n9_d4 -ml
# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n10_d4 -ml
# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n11_d4 -ml
# python main.py --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n12_d4 -ml


# cd simulations

# python automate.py

# # # PROGRAM="graph_coloring"
# DT_PROGRAM="dijkstra_token_ring"
# # PROGRAM="maximal_matching"

# # GRAPH="graph_7"
# DT_GRAPH="implicit_graph_n10"

# MM_GRAPH="graph_7"

# NO_SIMS=100000
# LIMIT_STEPS=200
# FAULT_INTERVALS=(1)
# SIMULATION_TYPE="controlled_at_node_amit_v2"

# for FI in "${FAULT_INTERVALS[@]}"; do
#     python simulate.py --program $PROGRAM --no-sim $NO_SIMS --fault-interval $FI  --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-nodes 8 9 --node-sel-strategy random --limit-steps $LIMIT_STEPS
#     python simulate.py --program $PROGRAM --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-nodes 8 9 --node-sel-strategy round-robin --limit-steps $LIMIT_STEPS
#     python simulate.py --program $PROGRAM --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-nodes 8 9 --node-sel-strategy reduced-wt --limit-steps $LIMIT_STEPS
# done

# for FI in "${FAULT_INTERVALS[@]}"; do
#     python simulate.py --program $PROGRAM --no-sim $NO_SIMS --fault-interval $FI  --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-nodes 4 5 --node-sel-strategy random --limit-steps $LIMIT_STEPS
#     python simulate.py --program $PROGRAM --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-nodes 4 5 --node-sel-strategy round-robin --limit-steps $LIMIT_STEPS
#     python simulate.py --program $PROGRAM --no-sim $NO_SIMS --fault-interval $FI --graph-names $GRAPH --fault-prob 1.0 --simulation-type $SIMULATION_TYPE --controlled-at-nodes 4 5 --node-sel-strategy reduced-wt --limit-steps $LIMIT_STEPS
# done

# for FI in "${FAULT_INTERVALS[@]}"; do
#     python simulate_v2.py --program graph_coloring --faulty-edges 0,2 2,1 --no-sim 10000 --fault-interval $FI --graph-names graph_20 --limit-steps $LIMIT_STEPS
#     python simulate_v2.py --program graph_coloring --faulty-edges 0,2 2,1 1,0 --no-sim 10000 --fault-interval $FI --graph-names graph_20 --limit-steps $LIMIT_STEPS
#     python simulate_v2.py --program graph_coloring --faulty-edges 0,1 1,2 2,3 --no-sim 10000 --fault-interval $FI --graph-names graph_21 --limit-steps $LIMIT_STEPS
#     python simulate_v2.py --program graph_coloring --faulty-edges 0,1 1,2 2,3 3,0 --no-sim 10000 --fault-interval $FI --graph-names graph_21 --limit-steps $LIMIT_STEPS
# done

# for FI in "${FAULT_INTERVALS[@]}"; do
#     python simulate_v2.py --program $DT_PROGRAM --faulty-edges 4,5 5,4 3,4 4,3 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 1,2 2,1 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 2,3 3,2 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 3,4 4,3 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 4,5 5,4 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 5,6 6,5 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 6,7 7,6 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 7,8 8,7 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 8,9 9,8 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
#     # python simulate_v2.py --program $DT_PROGRAM --faulty-edges 9,0 0,9 --no-sim $NO_SIMS --fault-interval $FI --graph-names $DT_GRAPH --limit-steps $LIMIT_STEPS
# done


# for FI in "${FAULT_INTERVALS[@]}"; do
#     python simulate_v2.py --program $PROGRAM --faulty-edges 1,0 --no-sim 10000 --fault-interval $FI --graph-names $DT_GRAPH --limit-steps 100
#     python simulate_v2.py --program $PROGRAM --faulty-edges 0,9 8,9 --no-sim 10000 --fault-interval $FI --graph-names $DT_GRAPH --limit-steps 100
#     python simulate_v2.py --program $PROGRAM --faulty-edges 4,5 5,4 --no-sim 10000 --fault-interval $FI --graph-names $DT_GRAPH --limit-steps 100
#     python simulate_v2.py --program $PROGRAM --faulty-edges 2,3 3,2 --no-sim 10000 --fault-interval $FI --graph-names $DT_GRAPH --limit-steps 100
# done

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

cd gnn

# graphs=("star_graph_n8" "star_graph_n9" "star_graph_n10" \
#         "graph_powerlaw_cluster_graph_n5" "graph_powerlaw_cluster_graph_n6" "graph_powerlaw_cluster_graph_n7" \
#         "graph_random_regular_graph_n7_d2" "graph_random_regular_graph_n7_d4" "graph_random_regular_graph_n8_d2")

graphs=("implicit_graph_n8" "implicit_graph_n9" "implicit_graph_n10" "implicit_graph_n11")

# graphs=("star_graph_n7" "star_graph_n8" \
#         "graph_powerlaw_cluster_graph_n5" "graph_powerlaw_cluster_graph_n6" \
#         "graph_random_regular_graph_n6_d2" "graph_random_regular_graph_n7_d2")


joined_graphs_args="${graphs[@]}"

epochs=50
batch_size=64
hidden_size=32

python lstm_scratch.py \
    --program $DIJKSTRA_TOKEN_PROGRAM \
    --epochs $epochs \
    --batch-size $batch_size \
    --hidden-size $hidden_size \
    --num-layers 2 \
    --graph-names $joined_graphs_args


graphs=("star_graph_n6" "star_graph_n7" \
        "graph_powerlaw_cluster_graph_n5" "graph_powerlaw_cluster_graph_n6" \
        "graph_random_regular_graph_n6_d2" "graph_random_regular_graph_n7_d2")


python lstm_scratch.py \
    --program $MAX_MATCHING_PROGRAM \
    --epochs 25 \
    --batch-size $batch_size \
    --hidden-size $hidden_size \
    --num-layers 2 \
    --graph-names $joined_graphs_args

# python gcn_scratch.py \
#     --program $GRAPH_COLORING_PROGRAM \
#     --epochs $epochs \
#     --batch-size $batch_size \
#     --hidden-size $hidden_size \
#     --graph-names $joined_graphs_args


# for joined_graphs_args in "${graphs[@]}"; do

#     python gcn_scratch.py \
#         --program $MAX_MATCHING_PROGRAM \
#         --epochs $epochs \
#         --batch-size $batch_size \
#         --hidden-size $hidden_size \
#         --graph-names $joined_graphs_args

# done
