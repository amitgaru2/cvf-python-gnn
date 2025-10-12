#!/bin/bash
set -eu
# ===================================================
CLIENT_MACHINES=("yangra1.uwyo.edu" "yangra2.uwyo.edu" "yangra3.uwyo.edu" "manaslu9.uwyo.edu" "manaslu10.uwyo.edu" "manaslu11.uwyo.edu" "manaslu12.uwyo.edu")
dateDir="25_10_12"
jobID="06_43_80143"
# ===================================================
# collect the client logs from all client machines
for client in "${CLIENT_MACHINES[@]}"; do
    hostName=$(echo $client | cut -d'.' -f1)
    mkdir -p "./client_script_logs/${hostName}/${dateDir}"
    echo "Fetching from client machine: $client."
    scp "$client:~/research/client_script_logs/${hostName}/${dateDir}/${jobID}.log" ./client_script_logs/${hostName}/${dateDir}/${jobID}.log
    echo -e "Done fetching from $client.\n"
done
echo "=== Client logs collection complete. ==="
