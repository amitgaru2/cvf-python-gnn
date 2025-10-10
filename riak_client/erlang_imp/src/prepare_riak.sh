#!/usr/bin/env escript
%%! -smp enable -sname prepare_riak

main(Args) ->
    %% Call your logger setup
    %% my_logger:setup(),

    %% Parse arguments (simple example)
    io:format("Preparing Riak with arguments: ~p~n", [Args]),
    case Args of
        ["--graph-name", GraphName] ->
            io:format("Graph name: ~s~n", [GraphName]),
            %% Here you would call your Riak initialization functions
            ok;
        _ ->
            io:format("Usage: ./prepare_riak --graph-name <name>~n"),
            halt(1)
    end.
