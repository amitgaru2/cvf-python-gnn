#!/bin/bash
set -eu

# variables defined to be changed
GRAPH_NAME="$1"

source commons_experiment.sh

CLIENT_COPY_FILES=("run_nohup.sh" "nohup_commands.sh" "custom_logger.py" "riak_helpers.py" "client_helpers.py" "graph_helpers.py" "populate_data_client.py")
# CLIENT_COPY_DIRS=("graphs")
CLIENT_COPY_DIRS=()


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
python3 populate_data_client.py --graph-name "$GRAPH_NAME" --client-id "$client_id" --num-clients "$NUM_CLIENTS"
EOF
)
    echo -e "Creating client script for $client.\n"
    ssh "$client" "cat << 'EOF' > ~/research/client_scripts/client_script.sh
${CLIENT_SCRIPT}
EOF"
    echo -e "Done creating client script for $client.\n"
done

SERVER_MACHINES_ENV=$(IFS=';'; echo "${SERVER_MACHINES[*]}")

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
