import os
import sys
import math
import time

import numpy as np
import pandas as pd

from collections import defaultdict
from functools import reduce, wraps


from custom_logger_mpi import logger
from custom_mpi import comm, program_node_rank


class Rank:
    def __init__(self, L, C, M):
        self.L = L
        self.C = C
        self.M = M

    def add_cost(self, val):
        self.L += val
        self.C += 1
        self.M = max(val, self.M)

    def __str__(self) -> str:
        return f"L: {self.L}, C: {self.C}, M: {self.M}"

    __repr__ = __str__


GlobalAvgRank = defaultdict(lambda: 0)
GlobalMaxRank = defaultdict(lambda: 0)
GlobalAvgRankEffect = defaultdict(lambda: 0)
GlobalAvgNodeRankEffect = {}

GlobalTimeTrackFunction = {}


def time_track(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time
        if func.__name__ in GlobalTimeTrackFunction:
            GlobalTimeTrackFunction[func.__name__] += total_time
        else:
            GlobalTimeTrackFunction[func.__name__] = total_time
        return result

    return inner


graphs_dir = "graphs"
graph_names = [sys.argv[1]]


def start(graphs_dir, graph_name):
    # logger.info('Started for Graph: "%s".', graph_name)
    full_path = os.path.join(graphs_dir, f"{graph_name}.txt")
    if not os.path.exists(full_path):
        # logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
        exit()

    graph = {}
    with open(full_path, "r") as f:
        line = f.readline()
        while line:
            node_edges = [int(i) for i in line.split()]
            node = node_edges[0]
            edges = node_edges[1:]
            graph[node] = set(edges)
            line = f.readline()

    return graph


class GraphColoringS:
    def __init__(self) -> None:
        self.graph = start(graphs_dir, graph_names[0])
        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}

        self.possible_node_values = [
            set(range(self.degree_of_nodes[node] + 1)) for node in self.nodes
        ]
        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.total_configs = reduce(
            lambda x, y: x * y, self.possible_node_values_length
        )
        logger.info(f"Total configs: {self.total_configs:,}.")

        # rank map
        self.global_rank_map = {}
        self.analysed_rank_count = 0

        self.possible_values = list(
            set([j for i in self.possible_node_values for j in i])
        )
        self.possible_values.sort()
        self.possible_values_indx_str = {
            v: str(i) for i, v in enumerate(self.possible_values)
        }  # mapping from value to index

        self.initialize_helpers()

        self.sample_size = 1000
        self.mpi_config = {}

        self.config_allocation()

    def config_allocation(self):
        equal_samples = self.total_configs // comm.size
        cursor = equal_samples + (self.total_configs - equal_samples * comm.size)
        self.mpi_config[0] = {
            "start": 0,
            "end": cursor,
        }
        for i in range(1, comm.size):
            new_cursor = cursor + equal_samples
            self.mpi_config[i] = {"start": cursor, "end": new_cursor}
            cursor = new_cursor

    @property
    def samples(self):
        return range(
            self.mpi_config[program_node_rank]["start"],
            self.mpi_config[program_node_rank]["end"],
        )

    def initialize_helpers(self):
        possible_node_values_len_rev = self.possible_node_values_length[::-1]

        self.base_n_to_decimal_multiplier = [1]
        for x in possible_node_values_len_rev[:-1]:
            self.base_n_to_decimal_multiplier.append(
                self.base_n_to_decimal_multiplier[-1] * x
            )

        self.base_n_to_decimal_multiplier_rev = self.base_n_to_decimal_multiplier[::-1]

    def base_n_to_decimal(self, base_n_str):
        value = 0
        length = len(base_n_str)
        for i in range(length):
            digit = int(base_n_str[length - i - 1])
            value += self.base_n_to_decimal_multiplier[i] * digit
        return value  # base 10, not fractional value

    @time_track
    def config_to_indx(self, config):
        config_to_indx_str = "".join(self.possible_values_indx_str[i] for i in config)
        result = self.base_n_to_decimal(config_to_indx_str)
        return result

    @time_track
    def indx_to_config(self, indx: int):
        s = []
        for multiplier in self.base_n_to_decimal_multiplier_rev:
            if indx >= multiplier:
                indx_val = indx // multiplier
                s.append(indx_val)
                indx -= indx_val * multiplier
            else:
                s.append(0)
        return tuple(s)

    def start(self):
        self.find_rank()
        self.reduce_rank()
        if program_node_rank == 0:
            self.save_rank()
        # self.find_rank_effect()
        # self.save_rank_effect()

    @time_track
    def _find_min_possible_color(self, colors):
        for i in range(len(colors) + 1):
            if i not in colors:
                return i

    @time_track
    def _get_program_transitions(self, start_state):
        program_transitions = []
        for position, color in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            neighbor_colors = set(start_state[i] for i in self.graph[position])
            if color not in neighbor_colors:  # is different color
                continue
            transition_color = self._find_min_possible_color(neighbor_colors)
            if color != transition_color:
                perturb_state = tuple(
                    [
                        *start_state[:position],
                        transition_color,
                        *start_state[position + 1 :],
                    ]
                )
                program_transitions.append(self.config_to_indx(perturb_state))
                # may be yield can save memory

        return program_transitions

    @time_track
    def is_invariant(self, config):
        for node, color in enumerate(config):
            for dest_node in self.graph[node]:
                if config[dest_node] == color:
                    return False
        return True

    def dfs(self, path: list[int]):
        indx = path[-1]
        if indx in self.global_rank_map:
            return

        config = self.indx_to_config(indx)
        if self.is_invariant(config):
            self.analysed_rank_count += 1
            self.global_rank_map[indx] = np.array([0, 1, 0])
            return

        self.global_rank_map[indx] = np.array([0, 0, 0])
        rank = self.global_rank_map[indx]
        for child_indx in self._get_program_transitions(config):
            self.dfs([*path, child_indx])
            rank_child = self.global_rank_map[child_indx]
            rank[0] += rank_child[0] + rank_child[1]
            rank[1] += rank_child[1]
            rank[2] = max(rank[2], rank_child[2] + 1)

        # post visit
        self.analysed_rank_count += 1

    def find_rank(self):
        for i in self.samples:
            if i not in self.global_rank_map:
                self.dfs([i])
                logger.debug(f"Analysed {self.analysed_rank_count:,} configurations.")

        for indx in self.global_rank_map:
            # remove duplicates for rank calculation
            if (
                self.mpi_config[program_node_rank]["start"]
                <= indx
                < self.mpi_config[program_node_rank]["end"]
            ):
                rank = self.global_rank_map[indx]
                avg_rank = math.ceil(rank[0] / rank[1])
                GlobalAvgRank[avg_rank] += 1
                GlobalMaxRank[rank[2]] += 1

    def reduce_rank(self):
        global GlobalAvgRank

        def _reducer(x, y):
            for rank, cnt in y.items():
                if rank in x:
                    x[rank] += cnt
                else:
                    x[rank] = cnt
            return x

        all_global_avg_rank = comm.gather(dict(GlobalAvgRank), root=0)
        if program_node_rank == 0:
            GlobalAvgRank = reduce(_reducer, all_global_avg_rank)

    def save_rank(self):
        df = pd.DataFrame(
            {"rank": GlobalAvgRank.keys(), "count": GlobalAvgRank.values()}
        )
        df.sort_values(by="rank").reset_index(drop=True).to_csv(
            os.path.join("new_results", f"ranks_avg__{graph_names[0]}.csv")
        )

        # max
        # df = pd.DataFrame(
        #     {"rank": GlobalMaxRank.keys(), "count": GlobalMaxRank.values()}
        # )
        # df.sort_values(by="rank").reset_index(drop=True).to_csv(
        #     os.path.join("new_results", f"ranks_max__{graph_names[0]}.csv")
        # )


def main():
    coloring = GraphColoringS()
    coloring.start()
    logger.info("%s", GlobalAvgRank)
    time_tracking = {k: round(v, 2) for k, v in GlobalTimeTrackFunction.items()}
    logger.info("%s", time_tracking)


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info("Total time taken: %s seconds.", round(time.time() - start_time, 4))
