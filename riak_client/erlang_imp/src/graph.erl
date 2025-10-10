-module(graph).
-export([
    new/2,
    nodes/1,
    degree/2,
    number_of_nodes/1,
    neighbors/2,
    get_graph/1,
    to_string/1
]).

-define(GRAPHS_DIR, "graphs").

-record(graph, {
    adj :: map(),
    name :: string()
}).

%% -------------------------------------------------------------
%% Constructor
%% -------------------------------------------------------------

new(AdjacencyDict, Name) when is_map(AdjacencyDict), is_list(Name) ->
    #graph{adj = AdjacencyDict, name = Name};
new(_, _) ->
    error({type_error, "Input must be a dictionary (map)"}).

%% -------------------------------------------------------------
%% Accessor functions
%% -------------------------------------------------------------

nodes(#graph{adj = Adj}) ->
    maps:keys(Adj).

degree(#graph{adj = Adj}, Node) ->
    case maps:get(Node, Adj, undefined) of
        undefined ->
            error({value_error, io_lib:format("Node ~p does not exist in the graph", [Node])});
        Neighbors ->
            length(Neighbors)
    end.

number_of_nodes(#graph{adj = Adj}) ->
    maps:size(Adj).

neighbors(#graph{adj = Adj}, Node) ->
    case maps:get(Node, Adj, undefined) of
        undefined ->
            error({value_error, io_lib:format("Node ~p does not exist in the graph", [Node])});
        Neighbors ->
            Neighbors
    end.

to_string(#graph{name = Name, adj = Adj}) ->
    io_lib:format("Graph(name=~s, nodes=~p)", [Name, maps:keys(Adj)]).

%% -------------------------------------------------------------
%% Load graph from file
%% -------------------------------------------------------------

get_graph(GraphName) ->
    FullPath = filename:join(?GRAPHS_DIR, GraphName ++ ".txt"),
    io:format("Locating Graph: ~s~n", [GraphName]),
    case file:read_file(FullPath) of
        {ok, Bin} ->
            Lines = string:split(binary_to_list(Bin), "\n", all),
            GraphDict = parse_lines(Lines, #{}),
            #graph{adj = GraphDict, name = GraphName};
        {error, enoent} ->
            io:format("Graph file: ~s not found! Skipping the graph.~n", [FullPath]),
            undefined;
        {error, Reason} ->
            io:format("Error reading file ~s: ~p~n", [FullPath, Reason]),
            undefined
    end.

%% -------------------------------------------------------------
%% Helper to parse adjacency list lines
%% -------------------------------------------------------------

parse_lines([], Acc) ->
    Acc;
parse_lines(["" | Rest], Acc) ->
    parse_lines(Rest, Acc);
parse_lines([Line | Rest], Acc) ->
    case string:tokens(Line, " \t") of
        [] ->
            parse_lines(Rest, Acc);
        Tokens ->
            Integers = [list_to_integer(T) || T <- Tokens],
            [Node | Edges] = Integers,
            parse_lines(Rest, maps:put(Node, Edges, Acc))
    end.
