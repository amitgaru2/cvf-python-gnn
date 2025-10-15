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
            Partition = partition:get_partition_for_client(
                graph:nodes(Graph), ClientId, NumClients
            ),
            main(Graph, Partition),
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

main(Graph, Partition) ->
    take_step(Graph, Partition),
    ok.

get_lexically_ordered_neighbors(Graph, Node) ->
    Response = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node~p__meta", [Node])
    ),
    Neighbors = maps:get(<<"nbrs">>, Response),
    lists:sort(Neighbors).

get_i_j_ordering(Node, Nbr) when Node < Nbr ->
    {Node, Nbr};
get_i_j_ordering(Node, Nbr) ->
    {Nbr, Node}.

loop_until(ConditionFun) ->
    case ConditionFun() of
        true ->
            loop_until(ConditionFun);
        false ->
            ok
    end.

get_pet_lock(Graph, Node, Nbr) ->
    Side = Node,
    OtherSide = Nbr,
    {i, j} = get_i_j_ordering(Node, Nbr),
    riak_client:put_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("L_FLAG_~p_~p_~p", [i, j, Side]),
        true
    ),
    riak_client:put_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("L_TURN_~p_~p", [i, j]),
        Side
    ),
    ConditionFun = fun() ->
        Turn = riak_client:get_request_riak(
            io_lib:format("graph_coloring__~s", [Graph#graph.name]),
            io_lib:format("L_TURN_~p_~p", [i, j])
        ),
        FlagOtherSide = riak_client:get_request_riak(
            io_lib:format("graph_coloring__~s", [Graph#graph.name]),
            io_lib:format("L_FLAG_~p_~p_~p", [i, j, OtherSide])
        ),
        (Turn == Side) and (FlagOtherSide == true)
    end,
    loop_until(ConditionFun),
    ok.

release_pet_lock(Graph, Node, Nbr) ->
    Side = Node,
    {i, j} = get_i_j_ordering(Node, Nbr),
    riak_client:put_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("L_FLAG_~p_~p_~p", [i, j, Side]),
        false
    ),
    ok.

get_neighbor_color([], Graph, Node, Partition, {NbrColors, AcquiredLocks}) ->
    {NbrColors, AcquiredLocks};
get_neighbor_color([Nbr | Rest], Graph, Node, Partition, {NbrColors, AcquiredLocks}) ->
    case lists:member(Nbr, Partition) of
        true ->
            NewAcquiredLocks = AcquiredLocks;
        false ->
            get_pet_lock(Graph, Node, Nbr),
            NewAcquiredLocks = [Nbr | AcquiredLocks]
    end,
    NbrColor = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node~p__val", [Nbr])
    ),
    NewNbrColors = [NbrColor | NbrColors],
    get_neighbor_color(Rest, Graph, Node, Partition, {NewNbrColors, NewAcquiredLocks}).

take_step_each_node(Graph, Node, Partition) ->
    SelfColor = riak_client:get_request_riak(
        io_lib:format("graph_coloring__~s", [Graph#graph.name]),
        io_lib:format("node~p__val", [Node])
    ),
    my_logger:info(io_lib:format("Node ~p has color: ~p~n", [Node, SelfColor])),
    SortedNeighbors = get_lexically_ordered_neighbors(Graph, Node),
    LockAcquired = [],
    {NbrColors, AcquiredLocks} = get_neighbor_color(
        SortedNeighbors, Graph, Node, Partition, LockAcquired
    ),
    case lists:member(SelfColor, NbrColors) of
        true ->
            NewColor = lists:min(lists:seq(0, length(NbrColors)) -- NbrColors),
            riak_client:put_request_riak(
                io_lib:format("graph_coloring__~s", [Graph#graph.name]),
                io_lib:format("node~p__val", [Node]),
                NewColor
            ),
            ok;
        false ->
            ok
    end,
    % release locks
    [release_pet_lock(Graph, Node, Nbr) || Nbr <- AcquiredLocks],
    ok.

take_step(Graph, Partition) ->
    lists:foreach(fun(Node) -> take_step_each_node(Graph, Node, Partition) end, graph:nodes(Graph)),
    ok.
