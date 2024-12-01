import csv
import os
import sys
import math
import time

import numpy as np
import pandas as pd

from collections import defaultdict
from functools import reduce, wraps


from custom_logger import logger


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


GlobalRankMap = defaultdict(lambda: Rank(L=0, C=0, M=0))
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


class DijkstraTokenRing:
    def __init__(self) -> None:
        self.graph = start(graphs_dir, graph_names[0])
        self.nodes = list(self.graph.keys())

        self.possible_node_values = [{0, 1, 2} for _ in self.nodes]
        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.total_configs = reduce(
            lambda x, y: x * y, self.possible_node_values_length
        )
        logger.info(f"Total configs: {self.total_configs:,}.")

        # rank map
        self.global_rank_map = np.full([self.total_configs, 3], None)
        self.analysed_rank_count = 0

        self.possible_values = list(
            set([j for i in self.possible_node_values for j in i])
        )
        self.possible_values.sort()
        self.possible_values_indx_str = {
            v: str(i) for i, v in enumerate(self.possible_values)
        }  # mapping from value to index

        self.initialize_helpers()
        self.initialize_problem_helpers()

    def initialize_problem_helpers(self):
        self.bottom = 0
        self.top = len(self.nodes) - 1

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
        self.save_rank()

    # program specific methods
    def __bottom_eligible_update(self, state):
        _state = list(state[:])
        _state[self.bottom] = (state[self.bottom] - 1) % 3
        return self.config_to_indx(tuple(_state))

    def __top_eligible_update(self, state):
        _state = list(state[:])
        _state[self.top] = (state[self.top - 1] + 1) % 3
        return self.config_to_indx(tuple(_state))

    def __other_eligible_update(self, state, idx, L_or_R_idx):
        _state = list(state[:])
        _state[idx] = state[L_or_R_idx]
        return self.config_to_indx(tuple(_state))

    # end program specific meethods

    @time_track
    def _get_program_transitions(self, start_state):
        program_transitions = []
        if (start_state[self.bottom] + 1) % 3 == start_state[self.bottom + 1]:
            program_transitions.append(self.__bottom_eligible_update(start_state))

        if (
            start_state[self.top - 1] == start_state[self.bottom]
            and (start_state[self.top - 1] + 1) % 3 != start_state[self.top]
        ):
            program_transitions.append(self.__top_eligible_update(start_state))

        for i in range(self.bottom + 1, self.top):
            if (start_state[i] + 1) % 3 == start_state[i - 1]:
                program_transitions.append(
                    self.__other_eligible_update(start_state, i, i - 1)
                )

            if (start_state[i] + 1) % 3 == start_state[i + 1]:
                program_transitions.append(
                    self.__other_eligible_update(start_state, i, i + 1)
                )

        return program_transitions

    @time_track
    def is_invariant(self, config):
        eligible_rules = 0

        if (config[self.bottom] + 1) % 3 == config[self.bottom + 1]:
            eligible_rules += 1

        if (
            config[self.top - 1] == config[self.bottom]
            and (config[self.top - 1] + 1) % 3 != config[self.top]
        ):
            eligible_rules += 1

        for i in range(self.bottom + 1, self.top):
            if eligible_rules > 1:
                return False

            if (config[i] + 1) % 3 == config[i - 1]:
                eligible_rules += 1

            if (config[i] + 1) % 3 == config[i + 1]:
                eligible_rules += 1

        return eligible_rules == 1

    def dfs(self, path: list[int]):
        indx = path[-1]

        if self.global_rank_map[indx, 0] is not None:
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
        for i in range(self.total_configs):
            if self.global_rank_map[i, 0] is None:
                self.dfs([i])
                logger.debug(f"Analysed {self.analysed_rank_count:,} configurations.")

        for indx in range(self.total_configs):
            rank = self.global_rank_map[indx]
            avg_rank = math.ceil(rank[0] / rank[1])
            GlobalAvgRank[avg_rank] += 1
            GlobalMaxRank[rank[2]] += 1

    def save_rank(self):
        df = pd.DataFrame(
            {"rank": GlobalAvgRank.keys(), "count": GlobalAvgRank.values()}
        )
        df.sort_values(by="rank").reset_index(drop=True).to_csv(
            os.path.join("new_results", f"ranks_avg__{graph_names[0]}.csv")
        )

        # max
        df = pd.DataFrame(
            {"rank": GlobalMaxRank.keys(), "count": GlobalMaxRank.values()}
        )
        df.sort_values(by="rank").reset_index(drop=True).to_csv(
            os.path.join("new_results", f"ranks_max__{graph_names[0]}.csv")
        )

    def find_rank_effect(self):
        for indx in range(self.total_configs):
            frm_config = self.indx_to_config(indx)
            for position, color in enumerate(frm_config):
                for perturb_color in self.possible_node_values[position] - {color}:
                    perturb_state = tuple(
                        [
                            *frm_config[:position],
                            perturb_color,
                            *frm_config[position + 1 :],
                        ]
                    )
                    to_indx = self.config_to_indx(perturb_state)
                    rank_effect = math.ceil(
                        self.global_rank_map[indx, 0] / self.global_rank_map[indx, 1]
                    ) - math.ceil(
                        self.global_rank_map[to_indx, 0]
                        / self.global_rank_map[to_indx, 1]
                    )
                    GlobalAvgRankEffect[rank_effect] += 1
                    if position not in GlobalAvgNodeRankEffect:
                        GlobalAvgNodeRankEffect[position] = defaultdict(lambda: 0)
                    GlobalAvgNodeRankEffect[position][rank_effect] += 1

    def save_rank_effect(self):
        df = pd.DataFrame(
            {
                "rank effect": GlobalAvgRankEffect.keys(),
                "count": GlobalAvgRankEffect.values(),
            }
        )
        df.sort_values(by="rank effect").reset_index(drop=True).to_csv(
            os.path.join("new_results", f"rank_effects_avg__{graph_names[0]}.csv")
        )

        df = pd.DataFrame.from_dict(GlobalAvgNodeRankEffect, orient="index")
        df.fillna(0, inplace=True)
        df = df.reindex(sorted(df.columns), axis=1)
        df.index.name = "node"
        df.sort_index(inplace=True)
        df.astype("int64").to_csv(
            os.path.join(
                "new_results", f"rank_effects_by_node_avg__{graph_names[0]}.csv"
            )
        )


def main():
    dijkstra = DijkstraTokenRing()
    dijkstra.start()
    logger.info("%s", GlobalAvgRank)
    time_tracking = {k: round(v, 2) for k, v in GlobalTimeTrackFunction.items()}
    logger.info("%s", time_tracking)

    writer = csv.DictWriter(
        open(f"{graph_names[0]}_config_rank_dataset.csv", "w"),
        fieldnames=["config", "rank"],
    )
    writer.writeheader()
    for k, v in enumerate(dijkstra.global_rank_map):
        writer.writerow(
            {
                "config": [i for i in dijkstra.indx_to_config(k)],
                "rank": math.ceil(v[0] / v[1]),
            }
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info("Total time taken: %s seconds.", round(time.time() - start_time, 4))
