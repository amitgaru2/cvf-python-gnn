#!/bin/bash
set -ex

clientID="$1"
NoOfClients="$2"
echo "Client ID: ""$clientID"
echo "Number of Clients: ""$NoOfClients"
echo "RIAK_SERVER_URLS: ""$RIAK_SERVER_URLS"

python graph_coloring_client.py --client_id "$clientID" --num_clients "$NoOfClients"
