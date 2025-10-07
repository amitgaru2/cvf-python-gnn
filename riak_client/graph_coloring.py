import os
import sys
import argparse

from client_helpers import get_partition_for_client
from riak_helpers import (
    get_request_riak,
    put_request_riak,
    RIAK_NODE_KEY_PREFIX,
    RIAK_BUCKET_PREFIX,
    RIAK_PETERSON_LCK_FLAG_KEY_PREFIX,
    RIAK_PETERSON_LCK_TURN_KEY_PREFIX,
)

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)


from custom_logger import logger
from command_line_helpers import get_graph_v2


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


def get_i_j_ordering(node, nbr):
    (i, j) = (node, nbr) if node < nbr else (nbr, node)
    return (i, j)


def get_pet_lock(node, nbr):
    side = node
    other_side = nbr
    (i, j) = get_i_j_ordering(node, nbr)
    logger.info(f"Acquiring lock for edge ({i}, {j}) by node {side}.")
    put_request_riak(
        RIAK_BUCKET_NAME, f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{side}", True
    )
    put_request_riak(
        RIAK_BUCKET_NAME, f"{RIAK_PETERSON_LCK_TURN_KEY_PREFIX}{i}_{j}", side
    )
    while True:
        turn = get_request_riak(
            RIAK_BUCKET_NAME, f"{RIAK_PETERSON_LCK_TURN_KEY_PREFIX}{i}_{j}"
        )
        flag_otherside = get_request_riak(
            RIAK_BUCKET_NAME,
            f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{other_side}",
        )
        if turn == side and flag_otherside is True:
            continue  # wait
        else:
            break

    return True


def release_pet_lock(node, nbr):
    side = node
    (i, j) = get_i_j_ordering(node, nbr)
    logger.info(f"Releasing lock for edge ({i}, {j}) by node {side}.")
    put_request_riak(
        RIAK_BUCKET_NAME, f"{RIAK_PETERSON_LCK_FLAG_KEY_PREFIX}{i}_{j}_{side}", False
    )


def check_all_members_of(lst, *values):
    return not (set(values) - set(lst))


def get_lexically_ordered_neighbors(node):
    neighbors = get_request_riak(
        f"{RIAK_BUCKET_PREFIX}__{graph_name}", f"{RIAK_NODE_KEY_PREFIX}{node}__meta"
    )["nbrs"]
    neighbors.sort()
    for nbr in neighbors:
        yield nbr


def take_step_each_node(graph, node):
    neighbor_colors = set()
    lock_acquired_for = []
    for nbr in get_lexically_ordered_neighbors(node):
        lock_req = nbr not in CLIENT_NODES
        if lock_req:
            get_pet_lock(node, nbr)
            lock_acquired_for.append(nbr)

        nbr_color = get_request_riak(
            f"{RIAK_BUCKET_PREFIX}__{graph_name}",
            f"{RIAK_NODE_KEY_PREFIX}{nbr}__val",
        )
        neighbor_colors.add(nbr_color)

    new_color = min({k for k in range(graph.degree(node) + 1)} - neighbor_colors)
    put_request_riak(
        f"{RIAK_BUCKET_PREFIX}__{graph_name}",
        f"{RIAK_NODE_KEY_PREFIX}{node}__val",
        new_color,
    )

    # release lock
    for nbr in lock_acquired_for:
        release_pet_lock(node, nbr)


def take_step(graph):
    for node in CLIENT_NODES:
        take_step_each_node(graph, node)


def main(graph):
    take_step(graph)


if __name__ == "__main__":
    args = get_args_parser()
    graph_name = args.graph_name
    graph = get_graph_v2(graph_name)
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
