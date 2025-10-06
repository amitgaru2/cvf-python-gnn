import os
import sys

import requests

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)


from custom_logger import logger
from command_line_helpers import get_graph_v2


RING_SIZE = 8
RIAK_BASE_URL = "http://localhost:8098"
RIAK_BUCKET_PREFIX = "graph_coloring"
RIAK_NODE_KEY_PREFIX = "node_"
RIAK_PETERSON_LCK_KEY_PREFIX = "L_"


def put_request_riak(bucket_name, key, value):
    """
    Implements the equivalent of:
      curl -XPUT \
        -H "Content-Type: application/json" \
        -d '{"name":"BAlice2"}' \
        http://127.0.0.1:8098/buckets/users/keys/user1
    """
    url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys/{key}"
    headers = {"Content-Type": "application/json"}
    data = value

    try:
        response = requests.put(url, json=data, headers=headers)
        response.raise_for_status()
        logger.info(f"Wrote {{ {key}: {value} }} to the Bucket: {bucket_name}.")
        logger.debug(f"Success: {response.status_code}")
        if response.text:
            logger.debug("Response body:", response.text)
        return True
    except requests.HTTPError as err:
        logger.error(f"HTTP error: {err}")
        logger.error("Status code:", err.response.status_code)
        logger.error("Response:", err.response.text)
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def init_pet_lock_data(riak_bucket_name, graph):
    logger.info(f"Writing initial Peterson lock data.")

    for n in graph.nodes():
        for nbr in graph.neighbors(n):
            if n < nbr:
                node_key = f"{RIAK_PETERSON_LCK_KEY_PREFIX}{n}_{nbr}"
                meta = {"flag_0": False, "flag_1": False, "turn": None}
                put_request_riak(riak_bucket_name, node_key, meta)


def init_graph_data(riak_bucket_name, graph):
    logger.info(f"Writing initial graph data.")

    for n in graph.nodes():
        node_key = f"{RIAK_NODE_KEY_PREFIX}{n}__meta"
        meta = {"nbrs": list(graph.neighbors(n))}
        put_request_riak(riak_bucket_name, node_key, meta)


def init_config_data(riak_bucket_name, graph):
    init_config = tuple(0 for _ in range(graph.number_of_nodes()))
    logger.info(f"Writing initial configuration: {init_config}")

    for i in range(graph.number_of_nodes()):
        node_key = f"{RIAK_NODE_KEY_PREFIX}{i}__val"
        success = put_request_riak(riak_bucket_name, node_key, init_config[i])
        if not success:
            logger.error(f"Failed to write node {i} to Riak.")
            sys.exit(1)


def init_data(riak_bucket_name, graph):
    init_graph_data(riak_bucket_name, graph)
    init_config_data(riak_bucket_name, graph)
    init_pet_lock_data(riak_bucket_name, graph)


def invert_dict(d):
    grouped = {}

    for key, value in d.items():
        grouped.setdefault(value, []).append(key)

    return grouped


def main(graph_name):
    graph_name = sys.argv[1]
    graph = get_graph_v2(graph_name)
    logger.info(f"Loaded graph {graph}.")

    init_config = tuple(0 for _ in range(graph.number_of_nodes()))
    logger.info(f"Writing initial config: {init_config}")

    riak_bucket_name = f"{RIAK_BUCKET_PREFIX}__{graph_name}"
    logger.info(f"Using Riak bucket: {riak_bucket_name}")

    init_data(riak_bucket_name, graph)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python graph_coloring.py <graph_name>")
        sys.exit(1)

    graph_name = sys.argv[1]
    main(graph_name)
