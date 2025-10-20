#!/bin/bash
set -ex

cd riak_client

./init_experiment.sh graph_dblp_coauthorship_n317080
# ./init_experiment.sh graph_youtube_connection_n1134890
# ./init_experiment.sh graph_youtube_connection_n3223585
# ./init_experiment.sh complete_graph_n100
