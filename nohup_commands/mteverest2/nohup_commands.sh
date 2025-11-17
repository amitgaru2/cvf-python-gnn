#!/bin/bash
set -ex

cd riak_client

# ./init_experiment.sh graph_dblp_coauthorship_n317080
# ./verify_results.sh graph_dblp_coauthorship_n317080
# ./init_experiment.sh graph_youtube_connection_n1134890
# ./init_experiment.sh graph_youtube_connection_n3223585
# ./init_experiment.sh complete_graph_n100
# ./init_experiment.sh complete_graph_n100
# ./verify_results.sh complete_graph_n100
# ./init_experiment.sh complete_graph_n50

ansible-playbook -i inventory.ini playbook.yml -e "graph=graph_dblp_coauthorship_n317080"
# ansible-playbook -i inventory.ini verify_playbook.yml -e "graph=graph_dblp_coauthorship_n317080" -e "date=" -e "job_id="
