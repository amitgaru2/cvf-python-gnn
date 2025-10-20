#!/bin/bash
# ===================================================
# This script is meant to be run in the local machine
# ===================================================
set -eu

# variables defined to be changed
GRAPH_NAME="$1"

source commons_experiment.sh

CLIENT_COPY_FILES=("run_nohup.sh" "nohup_commands.sh" "custom_logger.py" "riak_helpers.py" "client_helpers.py" "graph_helpers.py" "graph_coloring.py" "verify_graph_coloring.py")
# CLIENT_COPY_DIRS=("graphs")
CLIENT_COPY_DIRS=()

BUCKET__PROPS__N_VAL=3
BUCKET__PROPS__R=2
BUCKET__PROPS__W=2
BUCKET__PROPS__DW=2
# end of variables to be changed

# copy the client script to all client machines
for client in "${CLIENT_MACHINES[@]}"; do
    echo "Setting up client machine: $client."
    ssh "$client" "rm -rf ~/research/client_scripts"
    ssh "$client" "mkdir -p ~/research/client_scripts"
    ssh "$client" "mkdir -p ~/research/client_scripts/graphs"
    echo "Copying files and directories to $client."
    for file in "${CLIENT_COPY_FILES[@]}"; do
        scp "$file" "$client:~/research/client_scripts/"
    done
    for dir in "${CLIENT_COPY_DIRS[@]}"; do
        scp -r "$dir" "$client:~/research/client_scripts/"
    done
    scp graphs/${GRAPH_NAME}.txt "$client:~/research/client_scripts/graphs/"
    echo -e "Done copying files and directories to $client.\n"
done


NUM_CLIENTS=${#CLIENT_MACHINES[@]}

for client_id in "${!CLIENT_MACHINES[@]}"; do
    client="${CLIENT_MACHINES[$client_id]}"
    CLIENT_SCRIPT=$(cat <<EOF
python3 graph_coloring.py --graph-name "$GRAPH_NAME" --client-id "$client_id" --num-clients "$NUM_CLIENTS"
echo -e "\nVerifying results...\n"
python3 verify_graph_coloring.py --graph-name "$GRAPH_NAME" --client-id "$client_id" --num-clients "$NUM_CLIENTS"
EOF
)
    echo -e "Creating client script for $client.\n"
    ssh "$client" "cat << 'EOF' > ~/research/client_scripts/client_script.sh
${CLIENT_SCRIPT}
EOF"
    echo -e "Done creating client script for $client.\n"
done


SERVER_MACHINES_ENV=$(IFS=';'; echo "${SERVER_MACHINES[*]}")

# prepare the database script
# echo -e "Preparing the database.\n"
# RIAK_SERVER_URLS="${SERVER_MACHINES_ENV}" python prepare_database.py --graph-name "${GRAPH_NAME}"
# echo -e "Done preparing the database.\n"

echo -e "Update bucket properties:\n"
curl -X PUT -H "Content-Type: application/json" -d "{\"props\":{\"n_val\":${BUCKET__PROPS__N_VAL}, \"r\":${BUCKET__PROPS__R}, \"w\":${BUCKET__PROPS__W}, \"dw\":${BUCKET__PROPS__DW}}}" http://${SERVER_MACHINES[0]}/buckets/graph_coloring__${GRAPH_NAME}/props
echo -e "Done updating bucket properties.\n"

echo -e "Bucket properties:"
curl http://${SERVER_MACHINES[0]}/buckets/graph_coloring__${GRAPH_NAME}/props
echo -e "\n"

# execute the client script on all client machines
timePrefix=$(date +"%H_%M")
Clients_Job_ID="""$timePrefix""_"$(shuf -i 10000-99999 -n 1)
for client_id in "${!CLIENT_MACHINES[@]}"; do
    client="${CLIENT_MACHINES[$client_id]}"
    echo "Executing run_nohup.sh on $client."
    ssh "$client" "cd ~/research/client_scripts && chmod +x run_nohup.sh && RIAK_SERVER_URLS=\"${SERVER_MACHINES_ENV}\" ./run_nohup.sh ${Clients_Job_ID} ${GRAPH_NAME} ${client_id} ${#CLIENT_MACHINES[@]}"
    echo -e "Done executing run_nohup.sh on $client.\n"
done

echo "=== Experiment initialization complete. ==="
