import os
import math

import numpy as np
import pandas as pd

from typing import List
from functools import reduce
from collections import defaultdict


from custom_logger import logger


class CVFAnalysisV2:
    def __init__(self, graph_name, graph) -> None:
        self.graph_name = graph_name
        self.graph = graph
        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}

        self.possible_node_values = self.get_possible_node_values()

        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.total_configs = reduce(
            lambda x, y: x * y, self.possible_node_values_length
        )
        logger.info(f"Total configs: {self.total_configs:,}.")

        # rank
        self.global_avg_rank = defaultdict(lambda: 0)
        self.global_max_rank = defaultdict(lambda: 0)

        # rank effects
        self.global_avg_rank_effect = defaultdict(lambda: 0)
        self.global_avg_node_rank_effect = {}

        # rank map
        self.init_global_rank_map()
        self.analysed_rank_count = 0

        self.possible_values = list(
            set([j for i in self.possible_node_values for j in i])
        )
        self.possible_values.sort()
        self.possible_values_indx_str = {
            v: str(i) for i, v in enumerate(self.possible_values)
        }  # mapping from value to index

        self.initialize_helpers()
        self.initialize_program_helpers()

    def get_possible_node_values(self) -> List:
        raise NotImplemented

    def init_global_rank_map(self):
        """override this when not needed like for simulation"""
        self.global_rank_map = np.full([self.total_configs, 3], None)  # L, C, M

    def initialize_helpers(self):
        possible_node_values_len_rev = self.possible_node_values_length[::-1]

        self.base_n_to_decimal_multiplier = [1]
        for x in possible_node_values_len_rev[:-1]:
            self.base_n_to_decimal_multiplier.append(
                self.base_n_to_decimal_multiplier[-1] * x
            )

        self.base_n_to_decimal_multiplier_rev = self.base_n_to_decimal_multiplier[::-1]

    def initialize_program_helpers(self):
        pass

    def base_n_to_decimal(self, base_n_str):
        value = 0
        length = len(base_n_str)
        for i in range(length):
            digit = int(base_n_str[length - i - 1])
            value += self.base_n_to_decimal_multiplier[i] * digit
        return value  # base 10, not fractional value

    def config_to_indx(self, config):
        config_to_indx_str = "".join(self.possible_values_indx_str[i] for i in config)
        result = self.base_n_to_decimal(config_to_indx_str)
        return result

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
        self.find_rank_effect()
        self.save_rank_effect()

    def _get_program_transitions(self, start_state):
        raise NotImplemented

    def is_invariant(self, config):
        raise NotImplemented

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
            self.global_avg_rank[avg_rank] += 1
            self.global_max_rank[rank[2]] += 1

    def save_rank(self):
        df = pd.DataFrame(
            {
                "rank": self.global_avg_rank.keys(),
                "count": self.global_avg_rank.values(),
            }
        )
        df.sort_values(by="rank").reset_index(drop=True).to_csv(
            os.path.join("results", f"ranks_avg__{self.graph_name}.csv")
        )

        # max
        df = pd.DataFrame(
            {
                "rank": self.global_max_rank.keys(),
                "count": self.global_max_rank.values(),
            }
        )
        df.sort_values(by="rank").reset_index(drop=True).to_csv(
            os.path.join("results", f"ranks_max__{self.graph_name}.csv")
        )

    def find_rank_effect(self):
        for indx in range(self.total_configs):
            frm_config = self.indx_to_config(indx)
            for position, value in enumerate(frm_config):
                for perturb_value in self.possible_node_values[position] - {value}:
                    perturb_state = tuple(
                        [
                            *frm_config[:position],
                            perturb_value,
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
                    self.global_avg_rank_effect[rank_effect] += 1
                    if position not in self.global_avg_node_rank_effect:
                        self.global_avg_node_rank_effect[position] = defaultdict(
                            lambda: 0
                        )
                    self.global_avg_node_rank_effect[position][rank_effect] += 1

    def save_rank_effect(self):
        df = pd.DataFrame(
            {
                "rank effect": self.global_avg_rank_effect.keys(),
                "count": self.global_avg_rank_effect.values(),
            }
        )
        df.sort_values(by="rank effect").reset_index(drop=True).to_csv(
            os.path.join("results", f"rank_effects_avg__{self.graph_name}.csv")
        )

        df = pd.DataFrame.from_dict(self.global_avg_node_rank_effect, orient="index")
        df.fillna(0, inplace=True)
        df = df.reindex(sorted(df.columns), axis=1)
        df.index.name = "node"
        df.sort_index(inplace=True)
        df.astype("int64").to_csv(
            os.path.join("results", f"rank_effects_by_node_avg__{self.graph_name}.csv")
        )
