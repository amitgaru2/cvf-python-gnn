"""prepare database in parallel"""

import sys
import argparse

from riak_client.python_client.riak_helpers import (
    RIAK_BUCKET_PREFIX,
    get_request_riak,
    put_request_riak,
    delete_request_riak,
    RIAK_NODE_KEY_PREFIX,
)

from custom_logger import logger
from graph_helpers import get_graph
from client_helpers import get_partition_for_client


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    return args


def delete_data():
    logger.info(f"Deleting Riak bucket: {RIAK_BUCKET_NAME}")

    keys = get_request_riak(RIAK_BUCKET_NAME, "", params={"keys": "true"})["keys"]
    for key in keys:
        delete_request_riak(RIAK_BUCKET_NAME, key)


# def read_and_log_data(riak_bucket_name):
#     logger.info(f"Reading Riak bucket: {riak_bucket_name}")

#     keys = get_request_riak(riak_bucket_name, "", params={"keys": "true"})["keys"]
#     for key in keys:
#         if key.startswith(RIAK_NODE_KEY_PREFIX):
#             value = get_request_riak(riak_bucket_name, key)
#             logger.info(f"Key: {key}, Value: {value}")


def init_graph_data(graph):
    logger.info(f"Writing initial graph data.")

    for n in CLIENT_NODES:
        node_key = f"{RIAK_NODE_KEY_PREFIX}{n}__meta"
        meta = {"nbrs": list(graph.neighbors(n))}
        put_request_riak(RIAK_BUCKET_NAME, node_key, meta)


def init_config_data():
    # init_config = tuple(
    #     random.choice(range(graph.degree(n))) for n in sorted(graph.nodes())
    # )  # random initial configuration
    init_config = tuple(0 for _ in CLIENT_NODES)  # all zeros initial configuration
    logger.info(f"Writing initial configuration: {init_config}.")

    for i, n in enumerate(CLIENT_NODES):
        node_key = f"{RIAK_NODE_KEY_PREFIX}{n}__val"
        success = put_request_riak(RIAK_BUCKET_NAME, node_key, init_config[i])
        if not success:
            logger.error(
                f"Failed to write node {n} to Riak with initial value {init_config[i]}."
            )
            sys.exit(1)


def main(graph):
    init_graph_data(graph)
    init_config_data()


if __name__ == "__main__":
    args = get_args_parser()
    graph_name = args.graph_name
    graph = get_graph(graph_name)
    if graph is None:
        logger.error(f"Graph {graph_name} not found.")
        sys.exit(1)
    logger.info(f"Found graph {graph}.")
    # read_and_log_data(riak_bucket_name)
    RIAK_BUCKET_NAME = f"{RIAK_BUCKET_PREFIX}__{graph_name}"
    logger.info(f"Using Riak bucket: {RIAK_BUCKET_NAME}")
    delete_data()
    logger.info(f"Cleanup done.")
    if args.client_id >= args.num_clients or args.client_id < 0:
        raise Exception("Client ID must be in the range [0, num_clients-1].")
    client_id = args.client_id
    num_clients = args.num_clients
    CLIENT_NODES = get_partition_for_client(graph, client_id, num_clients)
    logger.info(f"Client {client_id} handling nodes: {CLIENT_NODES}.")
    main(graph)
    logger.info(f"Database preparation done.")
