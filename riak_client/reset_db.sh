#!/bin/bash
set -eu

SERVER_MACHINES=("manaslu5.uwyo.edu:8098" "manaslu6.uwyo.edu:8098" "manaslu7.uwyo.edu:8098" "manaslu8.uwyo.edu:8098")

# copy the client script to all client machines
for server in "${SERVER_MACHINES[@]}"; do
    echo "Cleaning up machine: $server."
    ssh "$server" "~/research/riak/_build/rel/rel/riak/bin/riak stop"
    ssh "$server" "rm -rf ~/research/riak/_build/rel/rel/riak/data/bitcask/*"
    ssh "$server" "~/research/riak/_build/rel/rel/riak/bin/riak daemon"
    echo -e "Done cleaning up machine: $server.\n"
done