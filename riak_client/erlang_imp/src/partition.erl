-module(partition).
-export([get_partition_for_client/3]).

%% Partition a list of numbers into M partitions using a hash-like method.
consistent_partition(Numbers, M) ->
    EmptyPartitions = lists:duplicate(M, []),
    distribute(Numbers, M, EmptyPartitions).

%% Helper: recursively distribute numbers into partitions
distribute([], _M, Partitions) ->
    Partitions;
distribute([Num | Rest], M, Partitions) ->
    % similar to Python's hash(num) % M
    % Index = 0 to M-1
    Index = erlang:phash2(Num, M),
    UpdatedPartitions = add_to_partition(Partitions, Index, Num),
    distribute(Rest, M, UpdatedPartitions).

%% Updating nested list at Index
add_to_partition(Partitions, Index, Num) ->
    {Left, [Current | Right]} = lists:split(Index, Partitions),
    NewPartition = [Num | Current],
    Left ++ [NewPartition | Right].

%% Get the partition assigned to a given client (by index)
get_partition_for_client(GraphNodes, ClientId, M) ->
    Partitions = consistent_partition(GraphNodes, M),
    % ClientId is 0-based, lists:nth is 1-based
    lists:nth(ClientId + 1, Partitions).
