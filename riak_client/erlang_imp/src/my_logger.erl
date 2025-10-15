-module(my_logger).
-export([setup/0, info/1, warning/1, error/1, debug/1]).

setup() ->
    ok = logger:add_handler(console, logger_std_h, #{}),
    logger:set_primary_config(level, info),
    ok.

debug(Msg) ->
    logger:debug(Msg).

info(Msg) ->
    logger:info(Msg).

warning(Msg) ->
    logger:warning(Msg).

error(Msg) ->
    logger:error(Msg).
