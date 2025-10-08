#!/bin/bash
# ===================================================
# This script is meant to be run in the local machine
# ===================================================
set -eu

# variables defined to be changed
SERVER_MACHINES=("manaslu5.uwyo.edu" "manaslu6.uwyo.edu" "manaslu7.uwyo.edu")
CLIENT_MACHINES=("yangra1.uwyo.edu" "yangra2.uwyo.edu" "yangra3.uwyo.edu")

CLIENT_COPY_FILES=("run_nohup.sh" "nohup_commands.sh" "riak_helpers.py" "client_helpers.py" "graph_coloring.py")
CLIENT_COPY_DIRS=("graphs")
# end of variables to be changed

# copy the client script to all client machines
for client in "${CLIENT_MACHINES[@]}"; do
    echo "Setting up client machine: $client."
    ssh "$client" "rm -rf ~/research/client_scripts"
    ssh "$client" "mkdir -p ~/research/client_scripts"
    echo "Copying files and directories to $client."
    for file in "${CLIENT_COPY_FILES[@]}"; do
        scp "$file" "$client:~/research/client_scripts/"
    done
    for dir in "${CLIENT_COPY_DIRS[@]}"; do
        scp -r "$dir" "$client:~/research/client_scripts/"
    done
    echo -e "Done copying files and directories to $client.\n"
done


SERVER_MACHINES_ENV=$(IFS=';'; echo "${SERVER_MACHINES[*]}")

# prepare the database script
RIAK_SERVER_URLS="${SERVER_MACHINES_ENV}" python prepare_db.py

# execute the client script on all client machines
timePrefix=$(date +"%H_%M")
Clients_Job_ID="""$timePrefix""_"$(shuf -i 10000-99999 -n 1)
for client_id in "${!CLIENT_MACHINES[@]}"; do
    client="${CLIENT_MACHINES[$client_id]}"
    echo "Executing run_nohup.sh on $client."
    ssh "$client" "cd ~/research/client_scripts && chmod +x run_nohup.sh && RIAK_SERVER_URLS=\"${SERVER_MACHINES_ENV}\" ./run_nohup.sh ${Clients_Job_ID} ${client_id} ${#CLIENT_MACHINES[@]}"
    echo -e "Done executing run_nohup.sh on $client.\n"
done

echo "=== Experiment initialization complete. ==="
