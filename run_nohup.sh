#!/bin/bash
set -eu
echo "CVF Project root DIR: "$CVF_PROJECT_DIR
export PYTHONDONTWRITEBYTECODE=1
dateDir=$(date +"%y_%m_%d")
timePrefix=$(date +"%H_%M")
jobID="""$timePrefix""_"$(shuf -i 10000-99999 -n 1)
echo "Job ID: ""$jobID"
hostName=$(hostname | sed 's/\./_/g')
mkdir -p "nohup_logs/${hostName}/${dateDir}"
logLocation="nohup_logs/${hostName}/${dateDir}/""$jobID"".log"
echo "Log location: ""$logLocation"
echo "Started at : "$(date)
commanLocation="nohup_commands/${hostName}/nohup_commands.sh"
tempFileLocation="temp.sh"
cp ${commanLocation} ${tempFileLocation}
sed -i '2,${/^#/d}' ${tempFileLocation}
sed -i '/^$/d' ${tempFileLocation}
chmod +x ${tempFileLocation}
nohup ./${tempFileLocation} > "$logLocation" 2>&1 <&- &
# command_pid=$!
# wait $command_pid && echo "Process completed successfully!" || echo "Process failed!"
