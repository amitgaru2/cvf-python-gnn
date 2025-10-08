#!/bin/bash
set -ex

GraphName="$1"
ClientID="$2"
NoOfClients="$3"
echo "Graph Name: ""$GraphName"
echo "Client ID: ""$ClientID"
echo "Number of Clients: ""$NoOfClients"
echo "RIAK_SERVER_URLS: ""$RIAK_SERVER_URLS"

python3 graph_coloring.py --graph-name "$GraphName" --client-id "$ClientID" --num-clients "$NoOfClients"

echo "Done execution!"