#!/bin/bash
# ===================================================
# This script is meant to be run in the local machine
# ===================================================
set -eu

# variables defined to be changed
GRAPH_NAME="$1"

SERVER_MACHINES=("localhost:8098")
CLIENT_MACHINES=("lhotse3.uwyo.edu")


CLIENT_COPY_FILES=("run_nohup.sh" "nohup_commands.sh")
# CLIENT_COPY_DIRS=("graphs")
CLIENT_COPY_DIRS=("_build")

BUCKET__PROPS__N_VAL=3
BUCKET__PROPS__R=2
BUCKET__PROPS__W=2
BUCKET__PROPS__DW=2
# end of variables to be changed

echo -e "Compiling the project...\n"
rebar3 compile
echo -e "Done compiling the project.\n"



SERVER_MACHINES_ENV=$(IFS=';'; echo "${SERVER_MACHINES[*]}")

ERL_SCRIPT_PREFIX=""
ERL_SCRIPT_SUFFIX="init stop"

# prepare the database script
echo -e "Preparing the database.\n"
RIAK_SERVER_URLS="${SERVER_MACHINES_ENV}" erl -pa _build/default/lib/riak_client_project/ebin -pa _build/default/lib/jsx/ebin -noshell -s prepare_db -graph-name "${GRAPH_NAME}" -s init stop
echo -e "Done preparing the database.\n"

echo -e "Update bucket properties:\n"
curl -X PUT -H "Content-Type: application/json" -d "{\"props\":{\"n_val\":${BUCKET__PROPS__N_VAL}, \"r\":${BUCKET__PROPS__R}, \"w\":${BUCKET__PROPS__W}, \"dw\":${BUCKET__PROPS__DW}}}" http://${SERVER_MACHINES[0]}/buckets/graph_coloring__${GRAPH_NAME}/props
echo -e "Done updating bucket properties.\n"

echo -e "Bucket properties:"
curl http://${SERVER_MACHINES[0]}/buckets/graph_coloring__${GRAPH_NAME}/props
echo -e "\n"

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
    # scp graphs/${GRAPH_NAME}.txt "$client:~/research/client_scripts/graphs/"
    echo -e "Done copying files and directories to $client.\n"
done

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
