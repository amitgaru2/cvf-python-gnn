import sys
import random
import argparse

from riak_helpers import (
    RIAK_BUCKET_PREFIX,
    delete_request_riak,
    get_request_riak,
    put_request_riak,
    RIAK_NODE_KEY_PREFIX,
)

from custom_logger import logger
from graph_helpers import get_graph


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-name",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def delete_data(riak_bucket_name):
    logger.info(f"Deleting Riak bucket: {riak_bucket_name}")

    keys = get_request_riak(riak_bucket_name, "", params={"keys": "true"})["keys"]
    for key in keys:
        delete_request_riak(riak_bucket_name, key)


def read_and_log_data(riak_bucket_name):
    logger.info(f"Reading Riak bucket: {riak_bucket_name}")

    keys = get_request_riak(riak_bucket_name, "", params={"keys": "true"})["keys"]
    for key in keys:
        if key.startswith(RIAK_NODE_KEY_PREFIX):
            value = get_request_riak(riak_bucket_name, key)
            logger.info(f"Key: {key}, Value: {value}")


def init_graph_data(riak_bucket_name, graph):
    logger.info(f"Writing initial graph data.")

    for n in graph.nodes():
        node_key = f"{RIAK_NODE_KEY_PREFIX}{n}__meta"
        meta = {"nbrs": list(graph.neighbors(n))}
        put_request_riak(riak_bucket_name, node_key, meta)


def init_config_data(riak_bucket_name, graph):
    init_config = tuple(
        random.choice(range(graph.degree(n))) for n in sorted(graph.nodes())
    )
    logger.info(f"Writing initial configuration: {init_config}.")

    for i in range(graph.number_of_nodes()):
        node_key = f"{RIAK_NODE_KEY_PREFIX}{i}__val"
        success = put_request_riak(riak_bucket_name, node_key, init_config[i])
        if not success:
            logger.error(f"Failed to write node {i} to Riak.")
            sys.exit(1)


def main(riak_bucket_name, graph):
    init_graph_data(riak_bucket_name, graph)
    init_config_data(riak_bucket_name, graph)


if __name__ == "__main__":
    args = get_args_parser()
    graph_name = args.graph_name
    graph = get_graph(graph_name)
    if graph is None:
        logger.error(f"Graph {graph_name} not found.")
        sys.exit(1)
    logger.info(f"Found graph {graph}.")
    riak_bucket_name = f"{RIAK_BUCKET_PREFIX}__{graph_name}"
    read_and_log_data(riak_bucket_name)
    # logger.info(f"Cleaning existing data in the bucket {riak_bucket_name}.")
    # delete_data(riak_bucket_name)
    # logger.info(f"Cleanup done.")
    # main(riak_bucket_name, graph)
    # logger.info(f"Database preparation done.")

