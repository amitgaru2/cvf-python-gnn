#!/bin/bash
set -eu
# ===================================================
source commons_experiment.sh

# CLIENT_MACHINES=("yangra2.uwyo.edu" "yangra3.uwyo.edu")
dateDir="25_10_18"
jobID="19_57_21613"
# ===================================================
# collect the client logs from all client machines
for client in "${CLIENT_MACHINES[@]}"; do
    hostName=$(echo $client | cut -d'.' -f1)
    mkdir -p "./client_script_logs/${hostName}/${dateDir}"
    echo "Fetching from client machine: $client."
    scp "$client:~/research/client_script_logs/${hostName}/${dateDir}/${jobID}.log" ./client_script_logs/${hostName}/${dateDir}/${jobID}.log
    cat ./client_script_logs/${hostName}/${dateDir}/${jobID}.log | grep Total || true
    echo -e "Done fetching from $client.\n"
done
echo "=== Client logs collection complete. ==="
