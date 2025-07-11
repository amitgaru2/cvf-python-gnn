import torch
import numpy as np

from typing import Tuple
from itertools import product

from base import CVFAnalysisV2, ProgramData
from lr_configs.config_adapter import LRConfig


class LinearRegressionData(ProgramData):
    def __init__(self, m: np.float64, c: np.float64):
        self.m = m  # slope
        self.c = c  # y-intercept
        self.data = (self.m, self.c)

    @staticmethod
    def get_m(data):
        return data[0]

    @staticmethod
    def get_c(data):
        return data[1]


class LinearRegressionCVFAnalysisV2(CVFAnalysisV2):
    results_dir = "linear_regression"

    def pre_initialize_program_helpers(self):
        self.config_file = "matrix_1"
        self.lr_config = LRConfig(self.config_file)

    def get_possible_node_values(self):
        """
        result: [ [0.5, 1.0, 1.5], [0.5, 1.0, 1.5], ... ]
        mapping: [ {0.5: 0, 1.0: 1, 1.5: 2}, {0.5: 0, 1.0: 1, 1.5: 2}, ... ]
        """
        result = []
        mapping = []
        possible_m_values = np.round(
            np.arange(
                self.lr_config.config.min_m + self.lr_config.config.m_step,
                self.lr_config.config.max_m + self.lr_config.config.m_step,
                self.lr_config.config.m_step,
            ),
            self.lr_config.config.m_step_decimals,
        )
        possible_c_values = np.round(
            np.arange(
                self.lr_config.config.min_c + self.lr_config.config.c_step,
                self.lr_config.config.max_c + self.lr_config.config.c_step,
                self.lr_config.config.c_step,
            ),
            self.lr_config.config.c_step_decimals,
        )
        possible_values_for_each_node = [
            i for i in product(possible_m_values, possible_c_values)
        ]  # same for all the nodes
        values_mapping_for_each_node = {
            v: i for i, v in enumerate(possible_values_for_each_node)
        }
        result = [possible_values_for_each_node for _ in self.nodes]
        mapping = [values_mapping_for_each_node for _ in self.nodes]
        return result, mapping

    """
    def __get_node_data_df(self, node_id):
        return self.config.df[self.config.df["node"] == node_id]

    def __clean_float_to_step_size_single(self, slope):
        quotient = np.divide(slope, self.config.slope_step)
        if quotient == int(quotient):
            return np.round(slope, self.config.slope_step_decimals)
        return np.round(
            np.int64(quotient) * self.config.slope_step, self.config.slope_step_decimals
        )

    def __copy_replace_indx_value(self, lst, indx, value):
        lst_copy = lst.copy()
        lst_copy[indx] = value
        return lst_copy

    def is_invariant(self, config: Tuple[int]):
        return super().is_invariant(config)


    def _get_program_transitions_as_configs(self, start_state):
        node_params = list(start_state)

        for node_id in range(self.nodes):
            for _ in range(1, self.config.iterations + 1):
                m = torch.tensor(node_params[node_id], requires_grad=True)
                c = torch.tensor(node_params[node_id], requires_grad=True)

                start_state_cpy = list(start_state)
                start_state_cpy[node_id] = m.item()

                node_df = self.__get_node_data_df(node_id)
                X_node = torch.tensor(node_df["X"].array)
                y = m * X_node
                y.backward()

                doubly_st_mt = self.doubly_stochastic_matrix_config[node_id]

                new_slope = (
                    sum(
                        frac * start_state_cpy[j] for j, frac in enumerate(doubly_st_mt)
                    )
                    - self.learning_rate * m.grad
                )

                if new_slope > self.config.max_slope:
                    new_slope = self.config.max_slope

                node_params[node_id] = new_slope

                if abs(m - new_slope) <= self.config.iteration_stop_threshold:
                    break

        for node_id, new_slope in enumerate(node_params):
            new_slope_cleaned = self.__clean_float_to_step_size_single(new_slope)
            if new_slope_cleaned != start_state[node_id]:
                new_node_params = self.__copy_replace_indx_value(
                    list(start_state), node_id, new_slope_cleaned
                )
                new_node_params = tuple(new_node_params)
                yield node_id, new_node_params
    """


if __name__ == "__main__":
    import os
    import sys

    utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
    sys.path.append(utils_path)

    from command_line_helpers import get_graph

    graph_names = ["star_graph_n5"]
    for graph_name, graph in get_graph(graph_names):
        lr = LinearRegressionCVFAnalysisV2(graph_name, graph)
