import os
import csv
import math

import numpy as np
import pandas as pd

from functools import reduce
from typing import List, Tuple
from collections import defaultdict


from custom_logger import logger


class ProgramData:
    def __init__(self, val: int):
        self.val = val
        self.data = self.val

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return str(self.data)

    __repr__ = __str__


class CVFAnalysisV2:
    results_dir = ""

    def __init__(
        self,
        graph_name: str,
        graph: dict,
        generate_data_ml: bool = False,
        generate_data_embedding: bool = False,
    ) -> None:
        self.graph_name = graph_name
        self.graph = graph
        self.generate_data_ml = generate_data_ml
        self.generate_data_embedding = generate_data_embedding
        self.pt_graph_adj_list = {}

        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}

        self.possible_node_values, self.possible_node_values_mapping = (
            self.get_possible_node_values()
        )

        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.total_configs = reduce(
            lambda x, y: x * y, self.possible_node_values_length
        )
        logger.info(f"Total configs: {self.total_configs:,}.")

        self.total_invariants = 0

        # rank
        self.global_avg_rank = defaultdict(lambda: 0)
        self.global_max_rank = defaultdict(lambda: 0)

        # node's program transitions
        self.global_pt = defaultdict(lambda: 0)

        # rank effects
        self.global_avg_rank_effect = defaultdict(lambda: 0)
        self.global_avg_node_rank_effect = {}

        # rank map
        self.init_global_rank_map()
        self.analysed_rank_count = 0

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
        config_to_indx_str = [str(indx) for indx in config]
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
        logger.info("Total Invariants: %s.", self.total_invariants)
        self.save_rank()
        self.find_rank_effect()
        self.save_rank_effect()
        if self.generate_data_ml:
            self.generate_dataset_for_ml()
        if self.generate_data_embedding:
            self.generate_dataset_for_embedding()

    def _get_program_transitions(self, start_state):
        raise NotImplemented

    def is_invariant(self, config: Tuple[int]):
        raise NotImplemented

    def dfs(self, path: list[int]):
        indx = path[-1]

        if self.global_rank_map[indx, 0] is not None:
            return

        config = self.indx_to_config(indx)
        if self.is_invariant(config):
            self.pt_graph_adj_list[indx] = [indx]
            self.total_invariants += 1
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
            os.path.join(
                "results", self.results_dir, f"ranks_avg__{self.graph_name}.csv"
            )
        )

        # max
        df = pd.DataFrame(
            {
                "rank": self.global_max_rank.keys(),
                "count": self.global_max_rank.values(),
            }
        )
        df.sort_values(by="rank").reset_index(drop=True).to_csv(
            os.path.join(
                "results", self.results_dir, f"ranks_max__{self.graph_name}.csv"
            )
        )

    def find_rank_effect(self):
        for indx in range(self.total_configs):
            frm_config = self.indx_to_config(indx)
            for position, value in enumerate(frm_config):
                for perturb_value in set(
                    range(len(self.possible_node_values[position]))
                ) - {value}:
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
            os.path.join(
                "results", self.results_dir, f"rank_effects_avg__{self.graph_name}.csv"
            )
        )

        # node's rank effect
        df = pd.DataFrame.from_dict(self.global_avg_node_rank_effect, orient="index")
        df.fillna(0, inplace=True)
        df = df.reindex(sorted(df.columns), axis=1)
        df.index.name = "node"
        df.sort_index(inplace=True)
        df.astype("int64").to_csv(
            os.path.join(
                "results",
                self.results_dir,
                f"rank_effects_by_node_avg__{self.graph_name}.csv",
            )
        )

    def generate_dataset_for_ml(self):
        writer = csv.DictWriter(
            open(
                os.path.join(
                    "datasets",
                    self.results_dir,
                    f"{self.graph_name}_config_rank_dataset.csv",
                ),
                "w",
            ),
            fieldnames=["config", "L", "C", "M"],
        )
        writer.writeheader()
        for k, v in enumerate(self.global_rank_map):
            writer.writerow(
                {
                    "config": list(self.indx_to_config(k)),
                    "L": v[0],
                    "C": v[1],
                    "M": v[2],
                }
            )

    def save_node_pt(self):
        df = pd.DataFrame.from_dict(self.global_pt, orient="index")
        df.fillna(0, inplace=True)
        df = df.reindex(sorted(df.columns), axis=1)
        df.index.name = "node"
        df.sort_index(inplace=True)
        df.astype("int64").to_csv(
            os.path.join(
                "results",
                self.results_dir,
                f"pts_by_node_avg__{self.graph_name}.csv",
            )
        )

    def generate_dataset_for_embedding(self):
        with open(
            os.path.join(
                "datasets", self.results_dir, f"{self.graph_name}_pt_adj_list.txt"
            ),
            "w",
        ) as f:
            for indx in range(self.total_configs):
                f.write(",".join(str(i) for i in self.pt_graph_adj_list[indx]))
                f.write("\n")
