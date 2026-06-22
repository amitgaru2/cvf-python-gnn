#!/bin/bash
# ===================================================
# This script is meant to be run in the local machine
# ===================================================
set -eu

# variables defined to be changed
GRAPH_NAME="$1"

source commons_experiment.sh

CLIENT_COPY_FILES=("verify_graph_coloring.py")
CLIENT_COPY_DIRS=()

# copy the client script to all client machines
for client in "${CLIENT_MACHINES[@]}"; do
    echo "Copying files and directories to $client."
    for file in "${CLIENT_COPY_FILES[@]}"; do
        scp "$file" "$client:~/research/client_scripts/"
    done
    for dir in "${CLIENT_COPY_DIRS[@]}"; do
        scp -r "$dir" "$client:~/research/client_scripts/"
    done
    echo -e "Done copying files and directories to $client.\n"
done


for client_id in "${!CLIENT_MACHINES[@]}"; do
    client="${CLIENT_MACHINES[$client_id]}"
    CLIENT_SCRIPT=$(cat <<EOF
python3 verify_graph_coloring.py --graph-name "$GRAPH_NAME" --client-id "$client_id" --num-clients "$NUM_CLIENTS"
echo -e "\nDone execution...\n"
EOF
)
    echo -e "Creating client script for $client.\n"
    ssh "$client" "cat << 'EOF' > ~/research/client_scripts/client_script.sh
${CLIENT_SCRIPT}
EOF"
    echo -e "Done creating client script for $client.\n"
done


# execute the client script on all client machines
timePrefix=$(date +"%H_%M")
Clients_Job_ID="""$timePrefix""_"$(shuf -i 10000-99999 -n 1)
for client_id in "${!CLIENT_MACHINES[@]}"; do
    client="${CLIENT_MACHINES[$client_id]}"
    echo "Executing run_nohup.sh on $client."
    ssh "$client" "cd ~/research/client_scripts && chmod +x run_nohup.sh && RIAK_SERVER_URLS=\"${SERVER_MACHINES_ENV}\" ./run_nohup.sh ${Clients_Job_ID} ${GRAPH_NAME} ${client_id} ${NUM_CLIENTS}"
    echo -e "Done executing run_nohup.sh on $client.\n"
done

echo "=== Verification initialization complete. ==="
