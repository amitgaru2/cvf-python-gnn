%%-----------------------------------------------------------
%% Module: my_logger
%% Purpose: Reusable logging module mimicking Python logging
%%-----------------------------------------------------------
-module(my_logger).
-export([setup/0, info/1, warning/1, error/1, debug/1]).

%%-----------------------------------------------------------
%% Setup function: add console handler with custom formatter
%%-----------------------------------------------------------
setup() ->
    %% Avoid adding duplicate handlers
    logger:add_handler(console, logger_std_h, #{level => info}),
    ok.

%%-----------------------------------------------------------
%% Logging functions
%%-----------------------------------------------------------
debug(Msg) ->
    logger:debug(Msg).

info(Msg) ->
    logger:info(Msg).

warning(Msg) ->
    logger:warning(Msg).

error(Msg) ->
    logger:error(Msg).

%%-----------------------------------------------------------
%% Custom formatter function
%%-----------------------------------------------------------
format(Level, Msg, _Time, _Meta) ->
    %% Get timestamp as "YYYY-MM-DD HH:MM:SS"
    TimeStr = timestamp_string(),
    Formatted = io_lib:format("~s - ~p - ~s", [TimeStr, Level, lists:flatten(Msg)]),
    Formatted.

%%-----------------------------------------------------------
%% Generate timestamp string like Python's "%Y-%m-%d %H:%M:%S"
%%-----------------------------------------------------------
timestamp_string() ->
    {{Year, Month, Day}, {Hour, Min, Sec}} = calendar:local_time(),
    io_lib:format(
        "~4..0B-~2..0B-~2..0B ~2..0B:~2..0B:~2..0B",
        [Year, Month, Day, Hour, Min, Sec]
    ).

my_formatter(LogEvent, _Config) ->
    Level = maps:get(level, LogEvent),
    Msg = io_lib:format("~p: ~p~n", [Level, maps:get(msg, LogEvent)]),
    Msg.
