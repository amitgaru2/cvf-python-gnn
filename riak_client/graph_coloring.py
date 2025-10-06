import os
import sys

import requests

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)


from custom_logger import logger
from command_line_helpers import get_graph_v2


RIAK_BASE_URL = "http://localhost:8098"
RIAK_BUCKET_PREFIX = "graph_coloring"
RIAK_NODE_KEY_PREFIX = "node_"


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
        logger.info(f"Wrote {{ {key}: {value} }} to {bucket_name}.")
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


def init_data(riak_bucket_name, graph):
    init_config = tuple(0 for _ in range(graph.number_of_nodes()))
    logger.info(f"Writing initial config: {init_config}")

    for i in range(graph.number_of_nodes()):
        node_key = f"{RIAK_NODE_KEY_PREFIX}{i}"
        success = put_request_riak(riak_bucket_name, node_key, init_config[i])
        if not success:
            logger.error(f"Failed to write node {i} to Riak.")
            sys.exit(1)


def update_node_color(riak_bucket_name, node_index, color):
    pass


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
