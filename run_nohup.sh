#!/bin/bash
set -e
export PYTHONDONTWRITEBYTECODE=1
dateTime=$(date +"%y_%m_%d_%H_%M")
jobID="""$dateTime""__"$(shuf -i 1000-9999 -n 1)
echo "Job ID: ""$jobID"
hostName=$(hostname | sed 's/\./_/g')
mkdir -p "nohup_logs/${hostName}"
logLocation="nohup_logs/${hostName}/""$jobID"".log"
echo "Log location: ""$logLocation"
echo "Started at : "$(date)
nohup ./nohup_commands.sh > "$logLocation" 2>&1 <&- &
# command_pid=$!
# wait $command_pid && echo "Process completed successfully!" || echo "Process failed!"
