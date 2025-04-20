#!/bin/bash
set -e
export PYTHONDONTWRITEBYTECODE=1
dateTime=$(date +"%Y-%m-%d__%H:%M:%S")
jobID=$(shuf -i 10000-99999 -n 1)__$dateTime
echo "Job ID: ""$jobID"
nohup ./nohup_commands.sh > nohup_logs/cvf_"$jobID".log 2>&1 <&- &
