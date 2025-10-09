%%-----------------------------------------------------------
%% Module: my_logger
%% Purpose: Reusable logging module mimicking Python logging
%%-----------------------------------------------------------
-module(my_logger).
-export([setup/0, info/1, warning/1, error/1]).

%%-----------------------------------------------------------
%% Setup function: add console handler with custom formatter
%%-----------------------------------------------------------
setup() ->
    %% Avoid adding duplicate handlers
    case
        logger:add_handler(console, logger_std_h, #{
            level => info,
            config => #{formatter => fun my_logger:format/4}
        })
    of
        {error, {already_exists, console}} -> ok;
        ok -> ok
    end.

%%-----------------------------------------------------------
%% Logging functions
%%-----------------------------------------------------------
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
