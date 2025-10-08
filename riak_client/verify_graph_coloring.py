import time

from custom_logger import logger
from graph_helpers import get_graph
from graph_coloring import get_args_parser
from client_helpers import get_partition_for_client
from riak_helpers import (
    RIAK_BUCKET_PREFIX,
    RIAK_NODE_KEY_PREFIX,
    get_request_riak,
)


def verify(graph, node):
    self_color = get_request_riak(
        RIAK_BUCKET_NAME,
        f"{RIAK_NODE_KEY_PREFIX}{node}__val",
    )
    for nbr in graph.neighbors(node):
        nbr_color = get_request_riak(
            RIAK_BUCKET_NAME,
            f"{RIAK_NODE_KEY_PREFIX}{nbr}__val",
        )
        if self_color == nbr_color:
            return False

    return True


def main(graph):
    failed_count = 0
    passed_count = 0

    for node in CLIENT_NODES:
        passed = verify(graph, node)
        if passed:
            passed_count += 1
            logger.info(f"Node {node} passed verification.")
        else:
            failed_count += 1
            logger.error(f"Node {node} failed verification.")

    logger.info(
        f"Verification complete: Total : {len(CLIENT_NODES)} | Passed: {passed_count} | Failed: {failed_count}."
    )


if __name__ == "__main__":
    start_time = time.time()
    args = get_args_parser()
    graph_name = args.graph_name
    graph = get_graph(graph_name)
    logger.info(f"Found graph {graph}.")
    if args.client_id >= args.num_clients or args.client_id < 0:
        raise Exception("Client ID must be in the range [0, num_clients-1].")
    client_id = args.client_id
    num_clients = args.num_clients
    CLIENT_NODES = get_partition_for_client(graph, client_id, num_clients)
    logger.info(f"Client {client_id} handling nodes: {CLIENT_NODES}.")
    RIAK_BUCKET_NAME = f"{RIAK_BUCKET_PREFIX}__{graph_name}"
    logger.info(f"Using Riak bucket: {RIAK_BUCKET_NAME}")
    main(graph)
    logger.info(f"Total time taken: {time.time() - start_time} seconds.")
