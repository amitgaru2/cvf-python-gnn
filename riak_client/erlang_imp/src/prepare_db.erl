-module(prepare_db).
-export([start/0]).

-include_lib("stdlib/include/assert.hrl").

-import(riak_client, [
    put_request_riak/3,
    get_request_riak/2,
    delete_request_riak/2
]).

-define(RIAK_BUCKET_PREFIX, "graph_coloring").
-define(RIAK_NODE_KEY_PREFIX, "node_").

start() ->
    Args = init:get_arguments(),
    case proplists:get_value('graph-name', Args) of
        undefined ->
            usage();
        GraphNameList ->
            % ["graphname"] -> "graphname"
            GraphName = lists:flatten(GraphNameList),
            main(GraphName)
    end.

usage() ->
    my_logger:error(
        io_lib:format("Usage: erl -noshell -s prepare_db start -graph-name <name> -s init stop~n")
    ),
    halt(1).

main(GraphName) ->
    my_logger:info(io_lib:format("Locating graph ~s...", [GraphName])),
    Graph = graph:get_graph(GraphName),
    case Graph of
        undefined ->
            my_logger:error(io_lib:format("Graph ~s not found.", [GraphName])),
            halt(1);
        _ ->
            my_logger:info(io_lib:format("Found graph: ~s", [graph:to_string(Graph)])),
            RiakBucketName = ?RIAK_BUCKET_PREFIX ++ "__" ++ GraphName,
            my_logger:info(
                io_lib:format("Cleaning existing data in bucket ~s...", [RiakBucketName])
            ),
            delete_data(RiakBucketName),
            my_logger:info(io_lib:format("Cleanup done.", [])),
            % init_data(RiakBucketName, Graph),
            my_logger:info(io_lib:format("Database preparation done.", [])),
            halt(0)
    end.

init_data(RiakBucketName, Graph) ->
    init_graph_data(RiakBucketName, Graph),
    init_config_data(RiakBucketName, Graph).

delete_data(RiakBucketName) ->
    io:format("Deleting Riak bucket: ~s~n", [RiakBucketName]),
    KeysResult = riak_client:get_request_riak(RiakBucketName, undefined, "keys=true"),
    case KeysResult of
        #{<<"keys">> := Keys} ->
            lists:foreach(
                fun(Key) ->
                    riak_client:delete_request_riak(RiakBucketName, binary_to_list(Key))
                end,
                Keys
            );
        _ ->
            io:format("No keys found for bucket ~s.~n", [RiakBucketName])
    end.

read_and_log_data(RiakBucketName) ->
    io:format("Reading Riak bucket: ~s~n", [RiakBucketName]),
    KeysResult = riak_client:get_request_riak(RiakBucketName, undefined, "keys=true"),
    case KeysResult of
        #{<<"keys">> := Keys} ->
            lists:foreach(
                fun(KeyBin) ->
                    Key = binary_to_list(KeyBin),
                    case string:prefix(Key, ?RIAK_NODE_KEY_PREFIX) of
                        nomatch ->
                            ok;
                        _ ->
                            Value = riak_client:get_request_riak(RiakBucketName, Key),
                            io:format("Key: ~s, Value: ~p~n", [Key, Value])
                    end
                end,
                Keys
            );
        _ ->
            io:format("No keys found.~n")
    end.

init_graph_data(RiakBucketName, Graph) ->
    my_logger:info(io_lib:format("Writing initial graph data...", [])),
    Ns = graph:nodes(Graph),
    lists:foreach(
        fun(N) ->
            NodeKey = io_lib:format("~s~p__meta", [?RIAK_NODE_KEY_PREFIX, N]),
            Meta = #{
                nbrs => graph:neighbors(Graph, N)
            },
            my_logger:info(
                io_lib:format("Writing node ~p metadata ~p to Riak with key ~s", [
                    N, Meta, lists:flatten(NodeKey)
                ])
            ),
            riak_client:put_request_riak(RiakBucketName, lists:flatten(NodeKey), Meta)
        end,
        Ns
    ).

init_config_data(RiakBucketName, Graph) ->
    Ns = lists:sort(graph:nodes(Graph)),
    InitConfig = [rand:uniform(graph:degree(Graph, N)) - 1 || N <- Ns],
    my_logger:info(io_lib:format("Writing initial configuration: ~p", [InitConfig])),
    NodeCount = graph:number_of_nodes(Graph),
    lists:foreach(
        fun(I) ->
            NodeKey = io_lib:format("~s~p__val", [?RIAK_NODE_KEY_PREFIX, I]),
            Value = lists:nth(I + 1, InitConfig),
            case riak_client:put_request_riak(RiakBucketName, lists:flatten(NodeKey), Value) of
                true ->
                    ok;
                false ->
                    io:format("Failed to write node ~p to Riak.~n", [I]),
                    halt(1)
            end
        end,
        lists:seq(0, NodeCount - 1)
    ).
