#!/bin/bash
set -eu

SERVER_MACHINES=("manaslu5.uwyo.edu" "manaslu6.uwyo.edu" "manaslu7.uwyo.edu" "manaslu8.uwyo.edu")

# copy the client script to all client machines
for server in "${SERVER_MACHINES[@]}"; do
    echo "Cleaning up machine: $server."
    ssh "$server" "~/research/riak/_build/rel/rel/riak/bin/riak stop"
    ssh "$server" "rm -rf ~/research/riak/_build/rel/rel/riak/data/bitcask/*"
    echo "Contents of bitcask after deletion (should be empty):"
    ssh "$server" "ls ~/research/riak/_build/rel/rel/riak/data/bitcask"
    ssh "$server" "~/research/riak/_build/rel/rel/riak/bin/riak daemon"
    echo -e "Done cleaning up machine: $server.\n"
done

# verify servers are running
for server in "${SERVER_MACHINES[@]}"; do
    echo "Verifying riak status on machine: $server."
    ssh "$server" "~/research/riak/_build/rel/rel/riak/bin/riak status"
    echo -e "Done verifying riak status on machine: $server.\n"
done