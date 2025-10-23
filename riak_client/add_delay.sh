#!/bin/bash
set -eu

source commons_experiment.sh

DELAY_MS="$1"

# copy the client script to all client machines
# for item in "${SERVER_MACHINES[@]}"; do
# 	server="${item%%:*}"
#     echo "Adding tc delay to machine: $server."
#     ssh "$server" "sudo tc qdisc add dev eno1 root netem delay ${DELAY_MS}ms"
#     echo -e "Done adding tc delay to machine: $server.\n"
# done

# verify servers are running
# for item in "${SERVER_MACHINES[@]}"; do
# 	server="${item%%:*}"
#     echo "Verifying tc delay on machine: $server."
#     ssh "$server" "sudo tc qdisc show dev eno1"
#     echo -e "Done verifying tc delay on machine: $server.\n"
# done

# # reset tc delays
for item in "${SERVER_MACHINES[@]}"; do
	server="${item%%:*}"
    echo "Resetting tc delay on machine: $server."
    ssh "$server" "sudo tc qdisc del dev eno1 root"
    echo -e "Done resetting tc delay on machine: $server.\n"
done


# # verify servers are running
for item in "${SERVER_MACHINES[@]}"; do
	server="${item%%:*}"
    echo "Verifying tc delay on machine: $server."
    ssh "$server" "sudo tc qdisc show dev eno1"
    echo -e "Done verifying tc delay on machine: $server.\n"
done
