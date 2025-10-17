#!/bin/bash
set -eu
# ===================================================
CLIENT_MACHINES=("yangra1.uwyo.edu" "yangra2.uwyo.edu" "yangra3.uwyo.edu" "yangra4.uwyo.edu" \
                 "yangra5.uwyo.edu" "yangra6.uwyo.edu" "yangra7.uwyo.edu" "yangra8.uwyo.edu" \
                 "yangra9.uwyo.edu" "yangra10.uwyo.edu" "yangra11.uwyo.edu" \
                 "nuptse1.uwyo.edu" "nuptse2.uwyo.edu" "nuptse3.uwyo.edu" "nuptse4.uwyo.edu" \
                 "manaslu3.uwyo.edu" "manaslu4.uwyo.edu" \
                 "manaslu9.uwyo.edu" "manaslu10.uwyo.edu" "manaslu11.uwyo.edu" "manaslu12.uwyo.edu")

# CLIENT_MACHINES=("nuptse1.uwyo.edu")
dateDir="25_10_16"
jobID="01_12_18748"
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
