-module(graph_coloring).
-export([start/0]).

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
                lists:flatten(
                    io_lib:format(
                        "Starting graph coloring with graph: ~s, client ID: ~p, number of clients: ~p~n",
                        [GraphName, ClientId, NumClients]
                    )
                )
            ),
            Graph = graph:get_graph(GraphName),
            main(Graph, ClientId, NumClients),
            halt(0)
    end,
    ok.

usage() ->
    my_logger:warning(
        "Usage: erl -noshell -s graph_coloring start -graph-name <name> -client-id <id> -num-clients <num> -s init stop"
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
    my_logger:info(
        lists:flatten(
            io_lib:format("Loaded graph: ~s~n", [graph:to_string(Graph)])
        )
    ),
    ok.
