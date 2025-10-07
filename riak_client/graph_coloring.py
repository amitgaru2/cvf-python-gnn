import os
import sys
import json
import argparse

import requests

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)


from custom_logger import logger
from command_line_helpers import get_graph_v2


RING_SIZE = 8
RIAK_BASE_URL = "http://localhost:8098"
RIAK_BUCKET_PREFIX = "graph_coloring"
RIAK_NODE_KEY_PREFIX = "node_"
RIAK_PETERSON_LCK_FLAG_KEY_PREFIX = "L_FLAG_"
RIAK_PETERSON_LCK_TURN_KEY_PREFIX = "L_TURN_"


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--client-partition-nodes",
        type=int,
        nargs="+",
        help="list of nodes that belongs to the current client's partition",
        required=True,
    )

    parser.add_argument(
        "--delete-data",
        action="store_true",
        help="Delete all data in the Riak bucket for the specified graph and exit.",
    )

    args = parser.parse_args()

    return args


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


def get_request_riak(bucket_name, key, params={}):
    if key:
        url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys/{key}"
    else:
        url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        try:
            value = response.json()  # attempt to parse JSON
        except json.JSONDecodeError:
            value = response.text  # fallback to raw text
    elif response.status_code == 404:
        logger.error(f"Key '{key}' not found in bucket '{bucket_name}'.")
        value = None
    else:
        logger.error(f"Error {response.status_code}: {response.text}")
        value = None
    return value


def delete_request_riak(bucket_name, key):
    url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys/{key}"
    try:
        response = requests.delete(url)
        response.raise_for_status()
        logger.info(f"Success deleting key '{key}' from bucket '{bucket_name}'.")
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


def get_pet_lock(riak_bucket_name, i, j, side):
    """
    i < j
    side: i or j
    """
    logger.info(f"Acquiring lock for edge ({i}, {j}) by node {side}.")
    put_request_riak(
        riak_bucket_name, f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{side}", True
    )
    put_request_riak(
        riak_bucket_name, f"{RIAK_PETERSON_LCK_TURN_KEY_PREFIX}{i}_{j}", side
    )
    while True:
        turn = get_request_riak(
            riak_bucket_name, f"{RIAK_PETERSON_LCK_TURN_KEY_PREFIX}{i}_{j}"
        )
        flag_otherside = get_request_riak(
            riak_bucket_name, f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{j}"
        )
        if flag_otherside and turn == side:
            break

    return True


def release_pet_lock(riak_bucket_name, i, j, side):
    """
    i < j
    side: i or j
    """
    logger.info(f"Releasing lock for edge ({i}, {j}) by node {side}.")
    put_request_riak(
        riak_bucket_name, f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{side}", False
    )


def check_all_members_of(lst, *values):
    return not (set(values) - set(lst))


def init_pet_lock_data(riak_bucket_name, graph, client_partition_nodes):
    logger.info(f"Writing initial Peterson lock data.")

    for n in graph.nodes():
        for nbr in graph.neighbors(n):
            if not check_all_members_of(client_partition_nodes, n, nbr):
                (i, j) = (n, nbr) if n < nbr else (nbr, n)
                node_key = f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{n}"
                put_request_riak(riak_bucket_name, node_key, False)
                turn_key = f"{RIAK_PETERSON_LCK_TURN_KEY_PREFIX}{i}_{j}"
                put_request_riak(riak_bucket_name, turn_key, -1)


def init_graph_data(riak_bucket_name, graph):
    logger.info(f"Writing initial graph data.")

    for n in graph.nodes():
        node_key = f"{RIAK_NODE_KEY_PREFIX}{n}__meta"
        meta = {"nbrs": list(graph.neighbors(n))}
        put_request_riak(riak_bucket_name, node_key, meta)


def init_config_data(riak_bucket_name, graph):
    init_config = tuple(0 for _ in range(graph.number_of_nodes()))
    logger.info(f"Writing initial configuration: {init_config}.")

    for i in range(graph.number_of_nodes()):
        node_key = f"{RIAK_NODE_KEY_PREFIX}{i}__val"
        success = put_request_riak(riak_bucket_name, node_key, init_config[i])
        if not success:
            logger.error(f"Failed to write node {i} to Riak.")
            sys.exit(1)


def init_data(riak_bucket_name, graph, client_partition_nodes):
    init_graph_data(riak_bucket_name, graph)
    init_config_data(riak_bucket_name, graph)
    init_pet_lock_data(riak_bucket_name, graph, client_partition_nodes)


def invert_dict(d):
    grouped = {}

    for key, value in d.items():
        grouped.setdefault(value, []).append(key)

    return grouped


def check_client_partition_nodes(client_partition_nodes, graph):
    if len(set(client_partition_nodes)) != len(client_partition_nodes):
        raise Exception("Client partition nodes contain duplicates.")

    if set(client_partition_nodes) - set(graph.nodes()):
        raise Exception("Client partition nodes not in graph nodes.")


def get_lexically_ordered_neighbor_i_j(n):
    neighbors = get_request_riak(
        f"{RIAK_BUCKET_PREFIX}__{graph_name}", f"{RIAK_NODE_KEY_PREFIX}{n}__meta"
    )["nbrs"]

    neighbors.sort()

    for nbr in neighbors:
        if not check_all_members_of(client_partition_nodes, n, nbr):
            (i, j) = (n, nbr) if n < nbr else (nbr, n)
            yield True, nbr, (i, j)
        else:
            yield False, nbr, (None, None)


def take_step_each_node(graph, n, client_partition_nodes):
    neighbor_colors = set()
    lock_acquired_for = []
    for lock_req, nbr, (i, j) in get_lexically_ordered_neighbor_i_j(n):
        if lock_req:
            get_pet_lock(f"{RIAK_BUCKET_PREFIX}__{graph_name}", i, j, n)
            lock_acquired_for.append((i, j))

        nbr_color = get_request_riak(
            f"{RIAK_BUCKET_PREFIX}__{graph_name}",
            f"{RIAK_NODE_KEY_PREFIX}{nbr}__val",
        )
        neighbor_colors.add(nbr_color)

    new_color = min({k for k in range(graph.degree(n) + 1)} - neighbor_colors)
    put_request_riak(
        f"{RIAK_BUCKET_PREFIX}__{graph_name}",
        f"{RIAK_NODE_KEY_PREFIX}{n}__val",
        new_color,
    )

    # release lock
    for i, j in lock_acquired_for:
        release_pet_lock(f"{RIAK_BUCKET_PREFIX}__{graph_name}", i, j, n)


def take_step(graph, client_partition_nodes):
    for n in client_partition_nodes:
        take_step_each_node(graph, n, client_partition_nodes)


def main(graph_name, client_partition_nodes):
    graph = get_graph_v2(graph_name)
    logger.info(f"Loaded graph {graph}.")

    check_client_partition_nodes(client_partition_nodes, graph)

    riak_bucket_name = f"{RIAK_BUCKET_PREFIX}__{graph_name}"
    logger.info(f"Using Riak bucket: {riak_bucket_name}")

    init_data(riak_bucket_name, graph, client_partition_nodes)
    take_step(graph, client_partition_nodes)


def delete_data(graph_name):
    riak_bucket_name = f"{RIAK_BUCKET_PREFIX}__{graph_name}"
    logger.info(f"Deleting Riak bucket: {riak_bucket_name}")

    keys = get_request_riak(riak_bucket_name, "", params={"keys": "true"})["keys"]
    for key in keys:
        delete_request_riak(riak_bucket_name, key)


if __name__ == "__main__":
    args = get_args_parser()
    graph_name = args.graph_name
    if args.delete_data:
        delete_data(graph_name)
        sys.exit(0)
    client_partition_nodes = args.client_partition_nodes
    main(graph_name, client_partition_nodes)
