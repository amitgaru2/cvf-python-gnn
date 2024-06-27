#!/bin/bash
set -e

echo "Generating merged plots for Program: \"${1}\" | Graph: \"${2}\""
# $1 could be maximal_matching | dijkstra_token_ring
# $2 is graph name

TEMP_DIR=temp
mkdir -p ${TEMP_DIR}

ANALYSIS_PLOT_LOCATION="analysis/plots/${1}"
NODE_VS_MAX_CVF_PLOTS_LOCATION="${ANALYSIS_PLOT_LOCATION}/node_vs_max_cvf_effect"
NODE_VS_CVF_PLOTS_LOCATION="${ANALYSIS_PLOT_LOCATION}/node_vs_cvf_effect"
NODE_VS_ACCUMULATED_CVF_PLOTS_LOCATION="${ANALYSIS_PLOT_LOCATION}/node_vs_accumulated_cvf_effect"

NODE_VS_ACCUMULATED_CVF_PLOTS_MERGE_FILELIST="${TEMP_DIR}/accumulated_cvf_plots_${1}_${2}.txt"
NODE_VS_ACCUMULATED_CVF_PLOTS_MERGE_PLOT=${TEMP_DIR}/accumulated_cvf_plots__${1}__${2}.png

NODE_VS_MAX_CVF__CVF__ACCUMULATED_CVF_MERGE_PLOT=${TEMP_DIR}/node_vs_max_cvf__cvf__accumulated_cvf__${1}__${2}.png

ls -r ${NODE_VS_ACCUMULATED_CVF_PLOTS_LOCATION}/node__vs__accumulated_severe_cvf_effect_gte_*__full__${1}__${2}.png > "${NODE_VS_ACCUMULATED_CVF_PLOTS_MERGE_FILELIST}"
# vertically append
convert $(cat $NODE_VS_ACCUMULATED_CVF_PLOTS_MERGE_FILELIST) -append ${NODE_VS_ACCUMULATED_CVF_PLOTS_MERGE_PLOT}

# vertically append
convert ${NODE_VS_MAX_CVF_PLOTS_LOCATION}/node_vs_max_rank_effect__full__${1}__${2}.png \
        ${NODE_VS_CVF_PLOTS_LOCATION}/node__vs__rank_effect_gte_*__full__${1}__${2}.png \
        ${NODE_VS_ACCUMULATED_CVF_PLOTS_MERGE_PLOT} \
        -append ${NODE_VS_MAX_CVF__CVF__ACCUMULATED_CVF_MERGE_PLOT}

if [[ "$1" == "dijkstra_token_ring" ]];
then
convert plots/${1}/rank_effect_by_node__full__${1}__${2}.png \
        ${NODE_VS_MAX_CVF__CVF__ACCUMULATED_CVF_MERGE_PLOT} \
        -gravity center \
        +append merged_plots/${1}_merged_plot_${2}.png
else
convert graph_images/${2}.png \
        plots/${1}/rank_effect_by_node__full__${1}__${2}.png \
        ${NODE_VS_MAX_CVF__CVF__ACCUMULATED_CVF_MERGE_PLOT} \
        -gravity center \
        +append merged_plots/${1}_merged_plot_${2}.png
fi

# rm merged_plots/max_matching_results_graph_6.png
# rm ${ACCUMULATED_CVF_PLOTS_MERGE_FILELIST}
