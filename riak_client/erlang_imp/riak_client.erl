-module(riak_client).
-export([
    get_random_riak_base_url/0,
    put_request_riak/3,
    get_request_riak/2,
    get_request_riak/3,
    delete_request_riak/2,
    init/0
]).

-define(DEFAULT_RIAK_URLS, "localhost:8098").
-define(RIAK_BUCKET_PREFIX, "graph_coloring").
-define(RIAK_NODE_KEY_PREFIX, "node_").
-define(RIAK_PETERSON_LCK_FLAG_KEY_PREFIX, "L_FLAG_").
-define(RIAK_PETERSON_LCK_TURN_KEY_PREFIX, "L_TURN_").

%% ------------------------------------------------------------------
%% Initialization: read environment and start dependencies
%% ------------------------------------------------------------------

init() ->
    application:ensure_all_started(inets),
    application:ensure_all_started(crypto),
    application:ensure_all_started(public_key),
    RIAK_URLS = os:getenv("RIAK_SERVER_URLS", ?DEFAULT_RIAK_URLS),
    URLList = string:tokens(RIAK_URLS, ";"),
    put(riak_urls, [ "http://" ++ U || U <- URLList ]),
    io:format("Using RIAK_BASE_URLS: ~p~n", [get(riak_urls)]),
    ok.

get_random_riak_base_url() ->
    Urls = get(riak_urls),
    case Urls of
        undefined -> init(), get_random_riak_base_url();
        [] -> "http://localhost:8098";
        _ ->
            RandomIndex = rand:uniform(length(Urls)),
            lists:nth(RandomIndex, Urls)
    end.

%% ------------------------------------------------------------------
%% PUT request
%% ------------------------------------------------------------------

put_request_riak(BucketName, Key, Value) ->
    BaseUrl = get_random_riak_base_url(),
    Url = io_lib:format("~s/buckets/~s/keys/~s", [BaseUrl, BucketName, Key]),
    Json = jsx:encode(Value),
    Headers = [{"Content-Type", "application/json"}],
    case httpc:request(put, {lists:flatten(Url), Headers, "application/json", Json}, [], []) of
        {ok, {{_, 200, _}, _RespHeaders, Body}} ->
            io:format("Wrote to ~s.~n", [Url]),
            io:format("Response: ~s~n", [Body]),
            true;
        {ok, {{_, Code, _}, _, Body}} ->
            io:format("Error writing (~p): ~s~n", [Code, Body]),
            false;
        {error, Reason} ->
            io:format("HTTP error: ~p~n", [Reason]),
            false
    end.

%% ------------------------------------------------------------------
%% GET request
%% ------------------------------------------------------------------

get_request_riak(BucketName, Key) ->
    get_request_riak(BucketName, Key, []).

get_request_riak(BucketName, Key, Params) ->
    BaseUrl = get_random_riak_base_url(),
    URL = case Key of
        undefined -> io_lib:format("~s/buckets/~s/keys", [BaseUrl, BucketName]);
        _ -> io_lib:format("~s/buckets/~s/keys/~s", [BaseUrl, BucketName, Key])
    end,
    FullURL = lists:flatten(URL),
    case httpc:request(get, {FullURL, []}, [], [{params, Params}]) of
        {ok, {{_, 200, _}, _Headers, Body}} ->
            io:format("Read from ~s.~n", [FullURL]),
            case catch jsx:decode(Body, [return_maps]) of
                {'EXIT', _} -> Body;
                Json -> Json
            end;
        {ok, {{_, 404, _}, _, _}} ->
            io:format("Key '~s' not found in bucket '~s'.~n", [Key, BucketName]),
            undefined;
        {ok, {{_, Code, _}, _, Body}} ->
            io:format("Error ~p: ~s~n", [Code, Body]),
            undefined;
        {error, Reason} ->
            io:format("HTTP error: ~p~n", [Reason]),
            undefined
    end.

%% ------------------------------------------------------------------
%% DELETE request
%% ------------------------------------------------------------------

delete_request_riak(BucketName, Key) ->
    BaseUrl = get_random_riak_base_url(),
    URL = io_lib:format("~s/buckets/~s/keys/~s", [BaseUrl, BucketName, Key]),
    FullURL = lists:flatten(URL),
    case httpc:request(delete, {FullURL, []}, [], []) of
        {ok, {{_, 200, _}, _, Body}} ->
            io:format("Deleted key '~s' from bucket '~s'.~n", [Key, BucketName]),
            io:format("Response: ~s~n", [Body]),
            true;
        {ok, {{_, Code, _}, _, Body}} ->
            io:format("Error deleting (~p): ~s~n", [Code, Body]),
            false;
        {error, Reason} ->
            io:format("HTTP error: ~p~n", [Reason]),
            false
    end.
