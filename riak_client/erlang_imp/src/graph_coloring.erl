-module(graph_coloring).
-export([start/0]).
-record(graph, {adj, name}).

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
            {ok, ClientId} = safe_int(ClientIdStr),
            {ok, NumClients} = safe_int(NumClientsStr),
            my_logger:info(
                io_lib:format(
                    "Starting graph coloring with graph: ~s, client ID: ~p, number of clients: ~p",
                    [GraphName, ClientId, NumClients]
                )
            ),
            Graph = graph:get_graph(GraphName),
            my_logger:info(
                io_lib:format("Loaded graph: ~s~n", [graph:to_string(Graph)])
            ),
            main(Graph, ClientId, NumClients),
            halt(0)
    end,
    ok.

usage() ->
    my_logger:warning(
        lists:flatten(
            io_lib:format(
                "Usage: erl -noshell -s graph_coloring start -graph-name <name> -client-id <id> -num-clients <num> -s init stop",
                []
            )
        )
    ),
    halt(1).

safe_int(Str) when is_list(Str) ->
    FlatStr = lists:flatten(Str),
    try
        {ok, list_to_integer(FlatStr)}
    catch
        _:_ -> {error, invalid_integer}
    end.

main(Graph, ClientId, NumClients) ->
    take_step(Graph),
    ok.

get_lexically_ordered_neighbors(Graph, Node) ->
    Response = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node~p__meta", [Node])
    ),
    Neighbors = maps:get(<<"nbrs">>, Response),
    SortedNeighbors = lists:sort(Neighbors),
    my_logger:info(
        io_lib:format("Node ~p has lexically ordered neighbors: ~p~n", [Node, SortedNeighbors])
    ),
    ok.

take_step_each_node(Graph, Node) ->
    Neighbors = graph:neighbors(Graph, Node),
    my_logger:info(io_lib:format("Node ~p has neighbors: ~p~n", [Node, Neighbors])),
    SelfColor = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node~p__val", [Node])
    ),
    my_logger:info(io_lib:format("Node ~p has color: ~p~n", [Node, SelfColor])),
    get_lexically_ordered_neighbors(Graph, Node),
    ok.

take_step(Graph) ->
    lists:foreach(fun(Node) -> take_step_each_node(Graph, Node) end, graph:nodes(Graph)),
    ok.
