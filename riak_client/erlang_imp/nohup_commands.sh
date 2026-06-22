#!/bin/bash
set -ex

GraphName="$1"
ClientID="$2"
NoOfClients="$3"
echo "Graph Name: ""$GraphName"
echo "Client ID: ""$ClientID"
echo "Number of Clients: ""$NoOfClients"
echo "RIAK_SERVER_URLS: ""$RIAK_SERVER_URLS"

erl -pa _build/default/lib/riak_client_project/ebin -pa _build/default/lib/jsx/ebin -noshell -s graph_coloring -graph-name "$GraphName" -client-id "$ClientID" -num-clients "$NoOfClients" -s init stop
echo -e "\nVerifying results...\n"
erl -pa _build/default/lib/riak_client_project/ebin -pa _build/default/lib/jsx/ebin -noshell -s verify_graph_coloring -graph-name "$GraphName" -client-id "$ClientID" -num-clients "$NoOfClients" -s init stop

echo "Done execution!"