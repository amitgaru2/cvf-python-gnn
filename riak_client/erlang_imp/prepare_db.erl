-module(prepare_db).
-export([main/2, delete_data/1, read_and_log_data/1, init_graph_data/2, init_config_data/2, start/0]).

-include_lib("stdlib/include/assert.hrl").

-import(riak_client, [
    put_request_riak/3,
    get_request_riak/2,
    delete_request_riak/2
]).

-define(RIAK_BUCKET_PREFIX, "graph_coloring").
-define(RIAK_NODE_KEY_PREFIX, "node_").

%% -------------------------------------------------------------
%% Entry point
%% -------------------------------------------------------------

start() ->
    %% Mimic Python's argparse
    Args = init:get_arguments(),
    case proplists:get_value("graph-name", Args) of
        undefined ->
            io:format("Usage: erl -noshell -s prepare_db start --graph-name <name> -s init stop~n"),
            halt(1);
        GraphNameList ->
            GraphName = lists:flatten(GraphNameList),
            run(GraphName)
    end.

run(GraphName) ->
    logger:info("Locating graph ~s...~n", [GraphName]),
    Graph = graph:get_graph(GraphName),
    case Graph of
        undefined ->
            io:format("Graph ~s not found.~n", [GraphName]),
            halt(1);
        _ ->
            io:format("Found graph: ~s~n", [graph:to_string(Graph)]),
            RiakBucketName = ?RIAK_BUCKET_PREFIX ++ "__" ++ GraphName,
            io:format("Cleaning existing data in bucket ~s...~n", [RiakBucketName]),
            delete_data(RiakBucketName),
            io:format("Cleanup done.~n"),
            main(RiakBucketName, Graph),
            io:format("Database preparation done.~n"),
            halt(0)
    end.

%% -------------------------------------------------------------
%% Delete data in Riak bucket
%% -------------------------------------------------------------

delete_data(RiakBucketName) ->
    io:format("Deleting Riak bucket: ~s~n", [RiakBucketName]),
    KeysResult = riak_client:get_request_riak(RiakBucketName, undefined, #{"keys" => "true"}),
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

%% -------------------------------------------------------------
%% Read and log data
%% -------------------------------------------------------------

read_and_log_data(RiakBucketName) ->
    io:format("Reading Riak bucket: ~s~n", [RiakBucketName]),
    KeysResult = riak_client:get_request_riak(RiakBucketName, undefined, #{"keys" => "true"}),
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

%% -------------------------------------------------------------
%% Write graph metadata
%% -------------------------------------------------------------

init_graph_data(RiakBucketName, Graph) ->
    io:format("Writing initial graph data...~n"),
    Ns = graph:nodes(Graph),
    lists:foreach(
        fun(N) ->
            NodeKey = io_lib:format("~s~p__meta", [?RIAK_NODE_KEY_PREFIX, N]),
            Meta = #{
                <<"nbrs">> => graph:neighbors(Graph, N)
            },
            riak_client:put_request_riak(RiakBucketName, lists:flatten(NodeKey), Meta)
        end,
        Ns
    ).

%% -------------------------------------------------------------
%% Write initial configuration values
%% -------------------------------------------------------------

init_config_data(RiakBucketName, Graph) ->
    Ns = lists:sort(graph:nodes(Graph)),
    InitConfig = [rand:uniform(graph:degree(Graph, N)) - 1 || N <- Ns],
    io:format("Writing initial configuration: ~p~n", [InitConfig]),
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

%% -------------------------------------------------------------
%% Main orchestration
%% -------------------------------------------------------------

main(RiakBucketName, Graph) ->
    init_graph_data(RiakBucketName, Graph),
    init_config_data(RiakBucketName, Graph).
