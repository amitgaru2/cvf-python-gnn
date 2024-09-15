#!/bin/bash
set -e
export PYTHONDONTWRITEBYTECODE=1
jobID=$(shuf -i 10000-99999 -n 1)
echo "Job ID: ""$jobID"
nohup ./nohup_commands.sh > cvf_"$jobID".log 2>&1 <&- &
