-module('verify_graph_coloring').
-export([start/0]).
-include("graph.hrl").

start() ->
    my_logger:setup(),
    Args = init:get_arguments(),
    case
        {
            proplists:get_value('graph-name', Args),
            proplists:get_value('client-id', Args),
            proplists:get_value('num-clients', Args)
        }
    of
        {undefined, _, _} ->
            usage();
        {_, undefined, _} ->
            usage();
        {_, _, undefined} ->
            usage();
        {GraphName, ClientIdStr, NumClientsStr} ->
            {ok, ClientId} = graph_coloring:safe_int(ClientIdStr),
            {ok, NumClients} = graph_coloring:safe_int(NumClientsStr),
            my_logger:info(
                io_lib:format(
                    "Starting graph coloring with graph: ~s, client ID: ~p, number of clients: ~p",
                    [GraphName, ClientId, NumClients]
                )
            ),
            Graph = graph:get_graph(GraphName),
            my_logger:info(
                io_lib:format("Loaded graph: ~s", [graph:to_string(Graph)])
            ),
            Partition = partition:get_partition_for_client(
                graph:nodes(Graph), ClientId, NumClients
            ),
            my_logger:info(
                io_lib:format("Client ~p has partition: ~p", [ClientId, Partition])
            ),
            main(Graph, Partition)
    end.

usage() ->
    my_logger:warning(
        io_lib:format(
            "Usage: erl -noshell -s verify_graph_coloring start -graph-name <name> -client-id <id> -num-clients <num> -s init stop",
            []
        )
    ),
    halt(1).

verify([], _, _) ->
    true;
verify([Nbr | Rest], SelfColor, Graph) ->
    NbrColor = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node_~p__val", [Nbr])
    ),
    case NbrColor of
        undefined ->
            my_logger:warning(
                io_lib:format("Neighbor node ~p has no color assigned!", [Nbr])
            ),
            false;
        SelfColor ->
            false;
        _ ->
            verify(Rest, SelfColor, Graph)
    end.

verifyNode([], _, Acc) ->
    Acc;
verifyNode([Node | Rest], Graph, {Passed, Failed}) ->
    Color = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node_~p__val", [Node])
    ),
    case Color of
        undefined ->
            my_logger:warning(
                io_lib:format("Node ~p has no color assigned!", [Node])
            ),
            verifyNode(Rest, Graph, {Passed, Failed + 1});
        _ ->
            Neighbors = graph_coloring:get_lexically_ordered_neighbors(Graph, Node),
            case verify(Neighbors, Color, Graph) of
                true ->
                    verifyNode(Rest, Graph, {Passed + 1, Failed});
                false ->
                    my_logger:warning(
                        io_lib:format("Coloring conflict found at node ~p!", [Node])
                    ),
                    verifyNode(Rest, Graph, {Passed, Failed + 1})
            end
    end.

main(Graph, Partition) ->
    {PassedCount, FailedCount} = verifyNode(Partition, Graph, {0, 0}),
    io:format("Passed: ~p, Failed: ~p~n", [PassedCount, FailedCount]),
    my_logger:info(
        io_lib:format(
            "Verification complete. Passed: ~p, Failed: ~p",
            [PassedCount, FailedCount]
        )
    ),
    ok.
