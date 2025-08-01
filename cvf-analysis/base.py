import os
import csv
import math

import numpy as np
import pandas as pd

from functools import reduce
from typing import List, Tuple
from collections import defaultdict


from custom_logger import logger


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        new_instance = super(Singleton, cls).__call__(*args, **kwargs)
        if new_instance not in cls._instances:
            cls._instances[new_instance] = new_instance
        return cls._instances[new_instance]


class ProgramTransitionTreeNode(metaclass=Singleton):

    def __init__(self, indx):
        self.indx = indx
        self.children = []
        self.parents = []

    def add_child(self, child: "ProgramTransitionTreeNode"):
        self.children.append(child)

    def add_parent(self, parent: "ProgramTransitionTreeNode"):
        self.parents.append(parent)

    def is_disjoint(self) -> bool:
        return len(self.parents) == 0

    def __hash__(self):
        return hash(self.indx)

    def __eq__(self, other):
        return self.indx == other.indx

    def __str__(self):
        return f"{self.__class__.__name__}: {self.indx}"

    __repr__ = __str__


class ProgramData:
    N_VARS = 1

    def __init__(self, val: int):
        self.data = (val,)

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return str(self.data)

    __repr__ = __str__


class CVFAnalysisV2:
    results_dir = ""
    DataKlass = ProgramData

    def __init__(
        self,
        graph_name: str,
        graph: dict,
        extra_kwargs: dict = dict(),
        generate_data_ml: bool = False,
        generate_data_embedding: bool = False,
        generate_test_data_ml: bool = False,
    ) -> None:
        self.graph_name = graph_name
        self.graph = graph
        self.generate_data_ml = generate_data_ml
        self.generate_data_embedding = generate_data_embedding
        self.generate_test_data_ml = generate_test_data_ml
        self.pt_graph_adj_list = {None: set()}
        self.extra_kwargs = extra_kwargs

        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}

        self.pre_initialize_program_helpers()

        self.possible_node_values, self.possible_node_values_mapping = (
            self.get_possible_node_values()
        )

        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.total_configs = reduce(
            lambda x, y: x * y, self.possible_node_values_length
        )
        logger.info(f"Total configs: {self.total_configs:,}.")

        self.total_invariants = 0

        # node's program transitions count
        self.global_pt = defaultdict(lambda: 0)

        # config -> successors / program transitions for ml
        self.config_successors = {}

        if not self.generate_test_data_ml:
            self.init_rank_calculation_related_configs()

        self.initialize_helpers()
        self.initialize_program_helpers()

    def pre_initialize_program_helpers(self):
        pass

    def get_possible_node_values(self) -> List:
        raise NotImplemented

    def init_rank_calculation_related_configs(self):
        """not needed when generating ML test data"""
        # rank
        self.global_avg_rank = defaultdict(lambda: 0)
        self.global_max_rank = defaultdict(lambda: 0)

        # rank effects
        self.global_avg_rank_effect = defaultdict(lambda: 0)
        self.global_avg_node_rank_effect = {}

        # rank map
        self.init_global_rank_map()
        self.analysed_rank_count = 0

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
        if self.generate_test_data_ml:
            self.start_test_data_generation_ml()
            return

        self.find_rank()
        logger.info("Total Invariants: %s.", f"{self.total_invariants:,}")
        self.save_rank()
        self.find_rank_effect()
        self.save_rank_effect()
        if self.generate_data_ml:
            self.generate_dataset_for_ml()
        if self.generate_data_embedding:
            self.generate_dataset_for_embedding()

    def start_test_data_generation_ml(self):
        logger.info("Generating test data for ML.")
        self.find_successors()
        self.generate_test_dataset_for_ml()

    def _get_program_transitions_as_configs(self, start_state: Tuple[int]):
        raise NotImplemented

    def _get_program_transitions(self, start_state):
        program_transitions = []
        for position, perturb_state in self._get_program_transitions_as_configs(
            start_state
        ):
            if perturb_state is not None:
                program_transitions.append(self.config_to_indx(perturb_state))
            else:
                program_transitions.append(perturb_state)

        if not program_transitions:
            print(
                "not foudn for", start_state, self.get_actual_config_values(start_state)
            )
        return program_transitions

    def is_invariant(self, config: Tuple[int]):
        raise NotImplemented

    def dfs(self, path: list[int]):
        indx = path[-1]

        successors = []

        if self.global_rank_map[indx, 0] is not None:
            return

        config = self.indx_to_config(indx)
        if self.is_invariant(config):
            # print("invariant", indx)
            self.total_invariants += 1
            self.analysed_rank_count += 1
            self.global_rank_map[indx] = np.array([0, 1, 0])
            return

        self.global_rank_map[indx] = np.array([0, 0, 0])
        rank = self.global_rank_map[indx]
        pt_node = ProgramTransitionTreeNode(indx)
        for child_indx in self._get_program_transitions(config):
            successors.append(child_indx)
            if child_indx is None:
                continue
            child_node = ProgramTransitionTreeNode(child_indx)
            pt_node.add_child(child_node)
            child_node.add_parent(pt_node)
            self.dfs([*path, child_indx])
            rank_child = self.global_rank_map[child_indx]
            rank[0] += rank_child[0] + rank_child[1]
            rank[1] += rank_child[1]
            rank[2] = max(rank[2], rank_child[2] + 1)

        # post visit
        self.analysed_rank_count += 1
        self.config_successors[indx] = successors

    def find_successors(self):
        """for ML test dataset"""
        for indx in range(self.total_configs):
            successors = []
            config = self.indx_to_config(indx)
            if not self.is_invariant(config):
                for child_indx in self._get_program_transitions(config):
                    successors.append(child_indx)
            self.config_successors[indx] = successors

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

    def possible_perturbed_state_frm(self, frm_indx):
        frm_config = self.indx_to_config(frm_indx)
        for position, value in enumerate(frm_config):
            for perturb_value in set(
                range(self.possible_node_values_length[position])
            ) - {value}:
                perturb_state = tuple(
                    [
                        *frm_config[:position],
                        perturb_value,
                        *frm_config[position + 1 :],
                    ]
                )
                to_indx = self.config_to_indx(perturb_state)
                yield position, to_indx

    def find_rank_effect(self):
        for indx in range(self.total_configs):
            for position, to_indx in self.possible_perturbed_state_frm(indx):
                rank_effect = math.ceil(
                    self.global_rank_map[indx, 0] / self.global_rank_map[indx, 1]
                ) - math.ceil(
                    self.global_rank_map[to_indx, 0] / self.global_rank_map[to_indx, 1]
                )
                self.global_avg_rank_effect[rank_effect] += 1
                if position not in self.global_avg_node_rank_effect:
                    self.global_avg_node_rank_effect[position] = defaultdict(lambda: 0)
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

    def get_actual_config_node_values(self, position, value):
        """
        Given the mappings from the method `get_possible_node_values`, we use this method to decode the index value of the node in any given configuration.
        Eg: Configuration (2, 0, 1, 4) will map to the third, first, second, and fifth value from the possible values of the nodes 0, 1, 2, 3 respectively.
        """
        return self.possible_node_values[position][value]

    def get_actual_config_values(self, config):
        result = []
        for position, value in enumerate(config):
            result.append(self.get_actual_config_node_values(position, value))
        return tuple(result)

    def get_mapped_value_of_data(self, node, data):
        return self.possible_node_values_mapping[node][self.DataKlass(*data)]

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
            fieldnames=["config", "rank", "succ"],
        )
        writer.writeheader()
        for k, v in enumerate(self.global_rank_map):
            config = self.get_actual_config_values(self.indx_to_config(k))
            writer.writerow(
                {
                    "config": list(config),
                    "rank": math.ceil(v[0] / v[1]),
                    "succ": (
                        [
                            (
                                list(
                                    self.get_actual_config_values(
                                        self.indx_to_config(i)
                                    )
                                )
                                # if i is None
                                # else None
                            )
                            for i in self.config_successors[k]
                        ]
                        if k in self.config_successors
                        else []
                    ),
                }
            )

    def generate_test_dataset_for_ml(self):
        writer = csv.DictWriter(
            open(
                os.path.join(
                    "datasets",
                    self.results_dir,
                    f"{self.graph_name}_config_succ_dataset.csv",
                ),
                "w",
            ),
            fieldnames=["config", "succ"],
        )
        writer.writeheader()
        for k in range(self.total_configs):
            writer.writerow(
                {
                    "config": list(self.indx_to_config(k)),
                    "succ": (
                        [
                            list(self.indx_to_config(i))
                            for i in self.config_successors[k]
                        ]
                        if k in self.config_successors
                        else []
                    ),
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

            def _dfs(start_node: ProgramTransitionTreeNode, path):
                for child_node in start_node.children:
                    full_path = _dfs(child_node, [*path, child_node.indx])
                    if full_path is not None:
                        f.write(",".join(str(i) for i in full_path))
                        f.write("\n")

                if not start_node.children:
                    # invariant
                    if len(path) == 1:
                        f.write(",".join(str(i) for i in path))
                        f.write("\n")

                    return path

                return None

            for indx in range(self.total_configs):
                pt_node = ProgramTransitionTreeNode(indx)
                if pt_node.is_disjoint():
                    # do the DFS
                    _dfs(pt_node, [pt_node.indx])
