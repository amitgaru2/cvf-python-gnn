#!/bin/bash
set -ex

GC_PROG="graph_coloring"
DTR_PROG="dijkstra_token_ring"
MM_PROG="maximal_matching"

GC_MODEL="lstm_trained_at_2025_08_29_23_08"
DTR_MODEL="lstm_trained_at_2025_08_29_23_28"
MM_MODEL="lstm_trained_at_2025_08_30_05_53"

#cd cvf-analysis
#python main.py --program maximal_matching --graph-names graph_powerlaw_cluster_graph_n9 --generate-test-data-ml

cd gnn

epochs=25
batch_size=64
hidden_size=32

graphs=("star_graph_n8" "star_graph_n9" \
        "graph_powerlaw_cluster_graph_n5" "graph_powerlaw_cluster_graph_n6" \
        "graph_random_regular_graph_n7_d2" "graph_random_regular_graph_n7_d4")

joined_graphs_args="${graphs[@]}"

# python lstm_scratch.py \
#     --program $GRAPH_COLORING_PROGRAM \
#     --epochs 25 \
#     --batch-size $batch_size \
#     --hidden-size $hidden_size \
#     --num-layers 2 \
#     --graph-names $joined_graphs_args

#python gcn_scratch.py \
#     --program $GC_PROG \
#     --epochs 25 \
#     --batch-size $batch_size \
#     --hidden-size $hidden_size \
#     --graph-names $joined_graphs_args


graphs=("implicit_graph_n8" "implicit_graph_n9" "implicit_graph_n10")


joined_graphs_args="${graphs[@]}"

# # python lstm_scratch.py \
# #     --program $DIJKSTRA_TOKEN_PROGRAM \
# #     --epochs 50 \
# #     --batch-size $batch_size \
# #     --hidden-size $hidden_size \
# #     --num-layers 2 \
# #     --graph-names $joined_graphs_args


# python gcn_scratch.py \
#     --program $DTR_PROG\
#     --epochs 50 \
#     --batch-size $batch_size \
#     --hidden-size $hidden_size \
#     --graph-names $joined_graphs_args


# graphs=("star_graph_n4" "star_graph_n5" "star_graph_n6" "star_graph_n7" "star_graph_n8" \
#         "graph_powerlaw_cluster_graph_n4" "graph_powerlaw_cluster_graph_n5" "graph_powerlaw_cluster_graph_n6" \
#         "graph_random_regular_graph_n5_d4" "graph_random_regular_graph_n6_d4")

graphs=("star_graph_n8" "graph_powerlaw_cluster_graph_n6" "graph_random_regular_graph_n6_d2")

joined_graphs_args="${graphs[@]}"

# python lstm_scratch.py \
#     --program $MAX_MATCHING_PROGRAM \
#     --epochs 50 \
#     --batch-size $batch_size \
#     --hidden-size $hidden_size \
#     --num-layers 2 \
#     --graph-names $joined_graphs_args

python gcn_scratch.py \
    --program $MM_PROG \
    --epochs 25 \
    --batch-size $batch_size \
    --hidden-size $hidden_size \
    --graph-names $joined_graphs_args


# for joined_graphs_args in "${graphs[@]}"; do

#     python gcn_scratch.py \
#         --program $MAX_MATCHING_PROGRAM \
#         --epochs $epochs \
#         --batch-size $batch_size \
#         --hidden-size $hidden_size \
#         --graph-names $joined_graphs_args

# done


# testing

# gc


# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM star_graph_n10
# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM star_graph_n11
# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM star_graph_n12
# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_powerlaw_cluster_graph_n7
# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_powerlaw_cluster_graph_n8
# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_random_regular_graph_n8_d2
# python lstm_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_random_regular_graph_n8_d4

# python cvf_analysis.py --model $model --program $GRAPH_COLORING_PROGRAM --graph-names graph_powerlaw_cluster_graph_n7 graph_powerlaw_cluster_graph_n8 graph_random_regular_graph_n8_d2 graph_random_regular_graph_n8_d4
# python plot_cvf.py --model $model --program $GRAPH_COLORING_PROGRAM --graph-names graph_powerlaw_cluster_graph_n7 graph_powerlaw_cluster_graph_n8 graph_random_regular_graph_n8_d2 graph_random_regular_graph_n8_d4
# python cvf_analysis.py --model $model --program $GRAPH_COLORING_PROGRAM --graph-names graph_random_regular_graph_n9_d4
# python plot_cvf.py --model $model --program $GRAPH_COLORING_PROGRAM --graph-names star_graph_n13 graph_powerlaw_cluster_graph_n9 graph_random_regular_graph_n9_d4


# model=gcn_trained_at_2025_08_29_23_13

# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM star_graph_n10
# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM star_graph_n11
# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM star_graph_n12
# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_powerlaw_cluster_graph_n7
# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_powerlaw_cluster_graph_n8
# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_random_regular_graph_n8_d2
# python gcn_scratch_test.py $model $GRAPH_COLORING_PROGRAM graph_random_regular_graph_n8_d4


# python cvf_analysis.py --model $model --program $GRAPH_COLORING_PROGRAM --graph-names star_graph_n10 star_graph_n11 star_graph_n12 star_graph_n13 graph_powerlaw_cluster_graph_n7 graph_powerlaw_cluster_graph_n8 graph_powerlaw_cluster_graph_n9 graph_random_regular_graph_n8_d2 graph_random_regular_graph_n8_d4 graph_random_regular_graph_n9_d4
# python plot_cvf.py --model $model --program $GRAPH_COLORING_PROGRAM --graph-names star_graph_n10 star_graph_n11 star_graph_n12 star_graph_n13 graph_powerlaw_cluster_graph_n7 graph_powerlaw_cluster_graph_n8 graph_powerlaw_cluster_graph_n9 graph_random_regular_graph_n8_d2 graph_random_regular_graph_n8_d4 graph_random_regular_graph_n9_d4


# # dtr
# model=lstm_trained_at_2025_08_29_23_28

# python lstm_scratch_test.py $model $DIJKSTRA_TOKEN_PROGRAM implicit_graph_n10
# python lstm_scratch_test.py $model $DIJKSTRA_TOKEN_PROGRAM implicit_graph_n11
# python lstm_scratch_test.py $model $DIJKSTRA_TOKEN_PROGRAM implicit_graph_n12

# python cvf_analysis.py --model $model --program $DIJKSTRA_TOKEN_PROGRAM --graph-names implicit_graph_n14 implicit_graph_n15
# python plot_cvf.py --model $model --program $DIJKSTRA_TOKEN_PROGRAM --graph-names implicit_graph_n13 implicit_graph_n14 implicit_graph_n15


# model=gcn_trained_at_2025_08_29_23_41

# python gcn_scratch_test.py $model $DIJKSTRA_TOKEN_PROGRAM implicit_graph_n10
# python gcn_scratch_test.py $model $DIJKSTRA_TOKEN_PROGRAM implicit_graph_n11
# python gcn_scratch_test.py $model $DIJKSTRA_TOKEN_PROGRAM implicit_graph_n12


# python cvf_analysis.py --model $model --program $DIJKSTRA_TOKEN_PROGRAM --graph-names implicit_graph_n13 implicit_graph_n14 implicit_graph_n15
# # python cvf_analysis.py --model $model --program $DIJKSTRA_TOKEN_PROGRAM --graph-names implicit_graph_n15
# python plot_cvf.py --model $model --program $DIJKSTRA_TOKEN_PROGRAM --graph-names implicit_graph_n13 implicit_graph_n14 implicit_graph_n15


# mm

# mm_model=lstm_trained_at_2025_08_30_05_53

# # python lstm_scratch_test.py $model $MAX_MATCHING_PROGRAM star_graph_n9
# # python lstm_scratch_test.py $model $MAX_MATCHING_PROGRAM star_graph_n10
# python lstm_scratch_test.py $model $MAX_MATCHING_PROGRAM graph_powerlaw_cluster_graph_n7
# python lstm_scratch_test.py $model $MAX_MATCHING_PROGRAM graph_random_regular_graph_n7_d2

# python cvf_analysis.py --model $model --program $MAX_MATCHING_PROGRAM --graph-names star_graph_n10

# model=gcn_trained_at_2025_08_30_06_35

# python gcn_scratch_test.py $model $MAX_MATCHING_PROGRAM star_graph_n9
# python gcn_scratch_test.py $model $MAX_MATCHING_PROGRAM star_graph_n10
# python gcn_scratch_test.py $model $MAX_MATCHING_PROGRAM graph_powerlaw_cluster_graph_n7
# python gcn_scratch_test.py $model $MAX_MATCHING_PROGRAM graph_random_regular_graph_n7_d2

# python cvf_analysis.py --model $mm_model --program $MAX_MATCHING_PROGRAM --graph-names graph_powerlaw_cluster_graph_n8
# python cvf_analysis.py --model $mm_model --program $MAX_MATCHING_PROGRAM --graph-names graph_random_regular_graph_n8_d2


# python cvf_analysis.py --model lstm_trained_at_2025_08_29_23_28 --program dijkstra_token_ring --graph-names implicit_graph_n12 implicit_graph_n13

# python cvf_analysis.py --model $mm_model --program $MAX_MATCHING_PROGRAM --graph-names star_graph_n10
