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

init() ->
    my_logger:setup(),
    application:ensure_all_started(inets),
    application:ensure_all_started(crypto),
    application:ensure_all_started(public_key),
    RIAK_URLS = os:getenv("RIAK_SERVER_URLS", ?DEFAULT_RIAK_URLS),
    URLList = string:tokens(RIAK_URLS, ";"),
    put(riak_urls, ["http://" ++ U || U <- URLList]),
    my_logger:info(io_lib:format("Using RIAK_BASE_URLS: ~p.", [get(riak_urls)])),
    ok.

get_random_riak_base_url() ->
    Urls = get(riak_urls),
    case Urls of
        undefined ->
            init(),
            get_random_riak_base_url();
        [] ->
            "http://localhost:8098";
        _ ->
            RandomIndex = rand:uniform(length(Urls)),
            lists:nth(RandomIndex, Urls)
    end.

put_request_riak(BucketName, Key, Value) ->
    BaseUrl = get_random_riak_base_url(),
    Url = io_lib:format("~s/buckets/~s/keys/~s", [BaseUrl, BucketName, Key]),
    FullURL = lists:flatten(Url),
    Json = jsx:encode(Value),
    my_logger:info(io_lib:format("PUT to ~s with value: ~s", [lists:flatten(Url), Json])),
    Headers = [{"Content-Type", "application/json"}],
    case httpc:request(put, {FullURL, Headers, "application/json", Json}, [], []) of
        {ok, {{_, Code, _}, _RespHeaders, Body}} when Code >= 200, Code < 300 ->
            my_logger:info(io_lib:format("PUT to URL: ~s the value: ~p", [FullURL, Value])),
            my_logger:debug(io_lib:format("Response: ~s.", [Body])),
            true;
        {ok, {{_, Code, _}, _, Body}} ->
            my_logger:error(io_lib:format("Error writing (~p): ~s.", [Code, Body])),
            false;
        {error, Reason} ->
            my_logger:error(io_lib:format("HTTP error: ~p.", [Reason])),
            false
    end.

get_request_riak(BucketName, Key) ->
    get_request_riak(BucketName, Key, []).

get_request_riak(BucketName, Key, _ParamStr) ->
    BaseUrl = get_random_riak_base_url(),
    URL =
        case Key of
            undefined -> io_lib:format("~s/buckets/~s/keys", [BaseUrl, BucketName]);
            _ -> io_lib:format("~s/buckets/~s/keys/~s", [BaseUrl, BucketName, Key])
        end,
    FullURL =
        case _ParamStr of
            undefined ->
                lists:flatten(URL);
            _ ->
                io_lib:format("~s?~s", [lists:flatten(URL), _ParamStr])
        end,
    case httpc:request(get, {FullURL, []}, [], []) of
        {ok, {{_, Code, _}, _Headers, Body}} when Code >= 200, Code < 300 ->
            my_logger:info(io_lib:format("Read from ~s.", [FullURL])),
            case catch jsx:decode(list_to_binary(Body), [return_maps]) of
                {'EXIT', Reason} ->
                    my_logger:error(io_lib:format("Error decoding JSON: ~p.", [Reason])),
                    Body;
                Json ->
                    Json
            end;
        {ok, {{_, 404, _}, _, _}} ->
            my_logger:warning(
                io_lib:format("Key '~s' not found in bucket '~s'.", [Key, BucketName])
            ),
            undefined;
        {ok, {{_, Code, _}, _, Body}} ->
            my_logger:debug(io_lib:format("Error ~p: ~s.", [Code, Body])),
            undefined;
        {error, Reason} ->
            my_logger:error(io_lib:format("HTTP error: ~p.", [Reason])),
            undefined
    end.

delete_request_riak(BucketName, Key) ->
    BaseUrl = get_random_riak_base_url(),
    URL = io_lib:format("~s/buckets/~s/keys/~s", [BaseUrl, BucketName, Key]),
    FullURL = lists:flatten(URL),
    case httpc:request(delete, {FullURL, []}, [], []) of
        {ok, {{_, Code, _}, _RespHeaders, Body}} when Code >= 200, Code < 300 ->
            my_logger:info(io_lib:format("Delete from ~s.", [FullURL])),
            my_logger:info(
                io_lib:format("Deleted key '~s' from bucket '~s'.", [Key, BucketName])
            ),
            my_logger:debug(io_lib:format("Response: ~s.", [Body])),
            true;
        {ok, {{_, Code, _}, _, Body}} ->
            my_logger:error(io_lib:format("Error deleting (~p): ~s.", [Code, Body])),
            false;
        {error, Reason} ->
            my_logger:error(io_lib:format("HTTP error: ~p.", [Reason])),
            false
    end.
