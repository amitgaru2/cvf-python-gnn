import os
import copy
import numpy as np
import pandas as pd

from functools import lru_cache
from cvf_analysis import CVFAnalysis, logger


class LinearRegressionFullAnalysis(CVFAnalysis):
    results_prefix = "linear_regression"
    results_dir = os.path.join("results", results_prefix)

    def __init__(self, graph_name, graph) -> None:
        super().__init__(graph_name, graph)
        self.learning_rate = 0.001
        self.slope_step_decimals = 1
        self.slope_step = 1 / (10**self.slope_step_decimals)
        self.min_slope = 0
        # self.max_slope = 4
        # self.actual_m = 3.0834764453827943
        # self.actual_b = -82.57574306316957
        # self.no_of_nodes = 4
        # self.df = pd.read_csv(
        #     "/home/agaru/research/cvf-python/linear_regression/SOCR-HeightWeight.csv"
        # )
        # self.df.rename(
        #     columns={"Height(Inches)": "X", "Weight(Pounds)": "y"}, inplace=True
        # )
        # self.doubly_stochastic_matrix_config = [
        #     [1 / 3, 1 / 6, 1 / 6, 1 / 3],
        #     [1 / 6, 1 / 6, 1 / 3, 1 / 3],
        #     [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        #     [1 / 3, 1 / 3, 1 / 6, 1 / 6],
        # ]
        self.max_slope = 1.0
        self.actual_m = 1.1204590846513611
        self.actual_b = -0.11847322643445737
        self.no_of_nodes = 3
        self.df = pd.read_csv(
            "/home/agaru/research/cvf-python/linear_regression/random-data.csv"
        )
        self.doubly_stochastic_matrix_config = [
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
            [1 / 6, 2 / 3, 1 / 6],
        ]
        self.node_data_partitions = np.array_split(self.df, self.no_of_nodes)
        for i, node_data in enumerate(self.node_data_partitions):
            self.df.loc[node_data.index, "node"] = i

        self.df["partition"] = -1
        for i in range(self.no_of_nodes):
            node_filter = self.df["node"] == i
            node_df = self.df[node_filter]
            partitions = self.__gen_test_data_partition_frm_df(
                self.no_of_nodes, node_df
            )
            for i, p in enumerate(partitions):
                self.df.loc[self.df.index.isin(p.index.values), "partition"] = i

    def __gen_test_data_partition_frm_df(self, partitions, df):
        shuffled = df.sample(frac=1)
        result = np.array_split(shuffled, partitions)
        return result

    def _start(self):
        self._gen_configurations()
        self._find_invariants()
        self._init_pts_rank()
        self._find_program_transitions_n_cvfs()
        # self._rank_all_states()
        # self._gen_save_rank_count()
        # self._calculate_pts_rank_effect()
        # self._calculate_cvfs_rank_effect()
        # self._gen_save_rank_effect_count()
        # self._gen_save_rank_effect_by_node_count()

    def _gen_configurations(self):
        self.configurations = {tuple([self.min_slope for _ in range(len(self.nodes))])}
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for node_pos in self.nodes:
            config_copy = copy.deepcopy(self.configurations)
            for i in np.round(
                np.arange(
                    self.min_slope + self.slope_step,
                    self.max_slope + self.slope_step,
                    self.slope_step,
                ),
                2,
            ):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def __get_adjusted_value(self, value):
        if value > self.max_slope:
            return self.max_slope

        if value < self.min_slope:
            return self.min_slope

        result = value

        if value / self.slope_step != 0:
            result = (value // self.slope_step) * self.slope_step

        if (value - result) > self.slope_step / 2:
            result = result + self.slope_step

        return result

    def _find_invariants(self):
        for state in self.configurations:
            for m in state:
                if not (
                    self.actual_m - self.slope_step / 2
                    < m
                    <= self.actual_m + self.slope_step / 2
                ):
                    break
            else:
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))

    def __forward(self, X, params):
        return params["m"] * X + params["c"]

    def __gradient_m(self, X, y, y_pred):
        N = len(y)
        return (-2 / N) * np.sum(X * (y - y_pred))

    def __get_node_data_df(self, node_id):
        return self.df[self.df["node"] == node_id]

    def __get_next_near_convergence_value(self, original_value, calculated_value):
        if calculated_value > original_value:
            return original_value + self.slope_step_decimals
        return original_value - self.slope_step_decimals

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        perturbed_m = dest_state[perturb_pos]
        original_m = start_state[perturb_pos]

        node_df = self.__get_node_data_df(perturb_pos)
        X_node = node_df["X"].array
        y_node = node_df["y"].array
        params = {"m": original_m, "c": 0}
        y_node_pred = self.__forward(X_node, params)
        grad_m = self.__gradient_m(X_node, y_node, y_node_pred)
        grad_m = np.round(grad_m, decimals=1)  # round off to 1
        doubly_st_mt = self.doubly_stochastic_matrix_config[perturb_pos]
        new_m = (
            sum(frac * start_state[i] for i, frac in enumerate(doubly_st_mt))
            - self.learning_rate * grad_m
        )
        ad_new_m = self.__get_adjusted_value(new_m)
        ad_new_m = np.round(ad_new_m, self.slope_step_decimals)
        if ad_new_m == original_m:
            ad_new_m = self.__get_next_near_convergence_value(original_m, new_m)
        return ad_new_m == perturbed_m

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        all_slope_values = set(
            np.round(
                np.arange(
                    self.min_slope, self.max_slope + self.slope_step, self.slope_step
                ),
                2,
            )
        )
        for position, val in enumerate(start_state):
            possible_slope_values = all_slope_values - {val}
            for perturb_val in possible_slope_values:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

        if not program_transitions:
            print("program transitions not found for", start_state)

        return program_transitions

    def _get_cvfs(self, start_state):
        cvfs = dict()
        all_slope_values = set(
            np.arange(self.min_slope, self.max_slope + self.slope_step, self.slope_step)
        )
        for position, val in enumerate(start_state):
            possible_slope_values = all_slope_values - {val}
            for perturb_val in possible_slope_values:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                cvfs[perturb_state] = (
                    position  # track the nodes to calculate its overall rank effect
                )

        return cvfs
