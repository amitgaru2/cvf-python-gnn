import os
import copy
import json
import pprint

import numpy as np
import pandas as pd

from cvf_analysis import CVFAnalysis, logger


class LinearRegressionFullAnalysis(CVFAnalysis):
    results_prefix = "linear_regression"
    results_dir = os.path.join("results", results_prefix)

    def __init__(self, graph_name, graph) -> None:
        super().__init__(graph_name, graph)

        self._temp_program_transitions = {}

        self.iterations = 100

        # self.learning_rate = 0.001
        # self.slope_step_decimals = 1
        # self.min_slope = np.float64(0.0)
        # self.max_slope = np.float64(1.1)
        # self.no_of_nodes = 3
        # self.df = pd.read_csv(
        #     "/home/agaru/research/cvf-python/linear_regression/random-data.csv"
        # )
        # self.doubly_stochastic_matrix_config = [
        #     [1 / 3, 1 / 3, 1 / 3],
        #     [1 / 3, 2 / 3, 0],
        #     [1 / 3, 0, 2 / 3],
        # ]
        # self.actual_m = 0.9
        # self.actual_b = -0.11847322643445737

        self.learning_rate = 0.0001
        self.stop_threshold = 0.0001
        self.slope_step = np.float64(0.025)
        self.slope_step_decimals = 3
        self.min_slope = np.float64(1.700)
        self.max_slope = np.float64(1.925)
        self.no_of_nodes = 4
        self.df = pd.read_csv(
            "/home/agaru/research/cvf-python/linear_regression/SOCR-HeightWeight.csv"
        )
        self.df.rename(
            columns={"Height(Inches)": "X", "Weight(Pounds)": "y"}, inplace=True
        )
        self.doubly_stochastic_matrix_config = [
            [1 / 2, 1 / 4, 1 / 8, 1 / 8],
            [1 / 4, 3 / 4, 0, 0],
            [1 / 8, 0, 7 / 8, 0],
            [1 / 8, 0, 0, 7 / 8],
        ]
        # self.actual_m = 3.08

        # self.slope_step = 1 / (10**self.slope_step_decimals)
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
        # self._init_pts_rank()
        # self._find_program_transitions()
        # self._find_program_transitions_v2()
        self._find_program_transitions_n_cvfs()
        self._init_pts_rank()
        self.__save_pts_to_file()
        self._rank_all_states()
        self._gen_save_rank_count()
        self._calculate_pts_rank_effect()
        self._calculate_cvfs_rank_effect()
        self._gen_save_rank_effect_count()
        self._gen_save_rank_effect_by_node_count()

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
                self.slope_step_decimals,
            ):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _find_invariants(self):
        # min_loss_sum = 1000000
        # min_loss_sum_state = None
        # for state in self.configurations:
        #     temp = 0
        #     for node, m in enumerate(state):
        #         node_df = self.__get_node_data_df(node)
        #         X_node = node_df["X"].array
        #         y_node = node_df["y"].array
        #         params = {"m": m, "c": 0}
        #         y_node_pred = self.__forward(X_node, params)
        #         loss = self.__loss_fn(y_node, y_node_pred)
        #         temp += loss

        #     if abs(temp) < min_loss_sum:
        #         min_loss_sum = abs(temp)
        #         min_loss_sum_state = state

        #     # for m in state:
        #     #     if not (
        #     #         self.actual_m - self.slope_step / 2
        #     #         < m
        #     #         <= self.actual_m + self.slope_step / 2
        #     #     ):
        #     #         break
        #     # else:
        #     #     self.invariants.add(state)

        # self.invariants.add(min_loss_sum_state)
        # print("Invariants", self.invariants, "min loss sum", min_loss_sum)
        logger.info("No. of Invariants: %s", len(self.invariants))

    def __forward(self, X, params):
        return params["m"] * X + params["c"]

    def __loss_fn(self, y, y_pred):
        N = len(y)
        return (1 / N) * sum((y[i] - y_pred[i]) ** 2 for i in range(N))

    def __gradient_m(self, X, y, y_pred):
        N = len(y)
        return (-2 / N) * np.sum(X * (y - y_pred))

    def __get_node_data_df(self, node_id):
        return self.df[self.df["node"] == node_id]

    def __clean_float_to_step_size_single(self, slope):
        # return np.trunc(slope*10**self.slope_step_decimals)/(10**self.slope_step_decimals)
        # return np.round(slope, self.slope_step_decimals)
        # if slope % np.int64(slope) >= 0.75:
        #     return np.round(slope, self.slope_step_decimals)
        # return np.trunc(slope * 10**self.slope_step_decimals) / (
        #     10**self.slope_step_decimals
        # )
        # 1.556 -> 1.50, 1.686 -> 1.60, 1.47 -> 1.45 { 0.05 }
        quotient = np.divide(slope, self.slope_step)
        if quotient == int(quotient):
            return np.round(slope, self.slope_step_decimals)
        return np.round(np.int64(quotient) * self.slope_step, self.slope_step_decimals)

    def __clean_float_to_step_size(self, node_slopes):
        result = []
        for slope in node_slopes:
            result.append(self.__clean_float_to_step_size_single(slope))

        return result

    def __copy_replace_indx_value(self, lst, indx, value):
        lst_copy = lst.copy()
        lst_copy[indx] = value
        return lst_copy

    def _get_program_transitions(self, start_state):
        program_transitions = set()

        node_params = list(start_state)
        for node_id in range(self.no_of_nodes):
            prev_m = node_params[node_id]
            for i in range(1, self.iterations + 1):
                node_df = self.__get_node_data_df(node_id)
                X_node = node_df["X"].array
                y_node = node_df["y"].array

                y_node_pred = self.__forward(X_node, {"m": prev_m, "c": 0})
                grad_m = self.__gradient_m(X_node, y_node, y_node_pred)

                doubly_st_mt = self.doubly_stochastic_matrix_config[node_id]

                start_state_cpy = list(start_state)
                start_state_cpy[node_id] = prev_m

                new_slope = (
                    sum(
                        frac * start_state_cpy[j]
                        for j, frac in enumerate(doubly_st_mt)
                    )
                    - self.learning_rate * grad_m
                )

                # if new_slope < self.min_slope:
                #     new_slope = self.min_slope

                if new_slope > self.max_slope:
                    new_slope = self.max_slope

                node_params[node_id] = new_slope

                if abs(prev_m - new_slope) <= self.stop_threshold:
                    break


        for node_id, new_slope in enumerate(node_params):
            new_slope_cleaned = self.__clean_float_to_step_size_single(new_slope)
            if new_slope_cleaned != self.__clean_float_to_step_size_single(
                start_state[node_id]
            ):
                new_node_params = self.__copy_replace_indx_value(
                    list(start_state), node_id, new_slope_cleaned
                )
                new_node_params = tuple(new_node_params)
                program_transitions.add(new_node_params)

        if not program_transitions:
            self.invariants.add(start_state)
            logger.debug("No program transition found for %s !", start_state)
            # logger.debug("Node params progress %s !", node_params_progress)
        # else:
        # logger.debug("%s : %s", start_state, program_transitions)

        return program_transitions

    def _get_cvfs(self, start_state):
        cvfs = dict()
        all_slope_values = set(
            np.round(
                np.arange(
                    self.min_slope,
                    np.round(
                        self.max_slope + self.slope_step, self.slope_step_decimals
                    ),
                    self.slope_step,
                ),
                self.slope_step_decimals,
            )
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

    def __save_pts_to_file(self):
        def _map_key(state):
            return json.dumps([float(k) for k in state])

        pts = {
            _map_key(state): list(pts["program_transitions"])
            for state, pts in self.pts_n_cvfs.items()
        }

        json.dump(pts, open("output.json", "w"))
