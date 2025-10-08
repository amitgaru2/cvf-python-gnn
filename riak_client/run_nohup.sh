#!/bin/bash
set -eu
export PYTHONDONTWRITEBYTECODE=1

jobID="$1"
GraphName="$2"
ClientId="$3"
NoOfClients="$4"
echo "Job ID: ""$jobID"
echo "Graph Name: ""$GraphName"
echo "Client ID: ""$ClientId"
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
nohup ./${tempFileLocation} ${GraphName} ${ClientId} ${NoOfClients} > "$logLocation" 2>&1 <&- &
