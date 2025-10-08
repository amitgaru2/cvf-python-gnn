#!/bin/bash
set -eu
export PYTHONDONTWRITEBYTECODE=1

jobID="$1"
clientID="$2"
NoOfClients="$3"
echo "Job ID: ""$jobID"
echo "Client ID: ""$clientID"
echo "Number of Clients: ""$NoOfClients"
echo "RIAK_SERVER_URLS: ""$RIAK_SERVER_URLS"

dateDir=$(date +"%y_%m_%d")
hostName=$(hostname | sed 's/\./_/g')
logLocationDir="../client_script_logs/${hostName}/${dateDir}/"
mkdir -p "$logLocationDir"
logLocation="${logLocationDir}""$jobID"".log"

echo "Log location: ""$logLocation"

commanLocation="nohup_commands.sh"
tempFileLocation="temp.sh"
cp ${commanLocation} ${tempFileLocation}
sed -i '2,${/^#/d}' ${tempFileLocation}
sed -i '/^$/d' ${tempFileLocation}
chmod +x ${tempFileLocation}

echo "Started at : "$(date)
nohup ./${tempFileLocation} ${clientID} ${NoOfClients} > "$logLocation" 2>&1 <&- &
