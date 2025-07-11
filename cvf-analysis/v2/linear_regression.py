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

    def is_invariant(self, config: Tuple[int]):
        """if the m and c value lies in the given range of the values, in all the nodes, then consider the config as invariant"""
        actual_config = self.get_actual_config_values(config)
        for node_value in actual_config:
            if (
                self.lr_config.config.invariant[0][0]
                <= node_value[0]
                <= self.lr_config.config.invariant[1][0]
                and self.lr_config.config.invariant[0][1]
                <= node_value[1]
                <= self.lr_config.config.invariant[1][1]
            ):
                pass
            else:
                return False
        return True

    def __get_node_data_df(self, node_id):
        return self.lr_config.df[self.config.df["node"] == node_id]

    def __clean_to_step_size(self, slope):
        quotient = np.divide(slope, self.config.slope_step)
        if quotient == int(quotient):
            return np.round(slope, self.config.slope_step_decimals)
        return np.round(
            np.int64(quotient) * self.config.slope_step, self.config.slope_step_decimals
        )

    # def __copy_replace_indx_value(self, lst, indx, value):
    #     lst_copy = lst.copy()
    #     lst_copy[indx] = value
    #     return lst_copy

    def _get_program_transitions_as_configs(self, start_state):

        for i in range(len(self.nodes)):
            data = self.get_actual_config_node_values(i, start_state[i])

            for _ in range(1, self.config.iterations + 1):
                m = torch.tensor(data.m, requires_grad=True)
                c = torch.tensor(data.c, requires_grad=True)

                node_df = self.__get_node_data_df(i)
                X_node = torch.tensor(node_df["X"].array)
                y = m * X_node + c
                y.backward()

                # new values to update
                doubly_st_mt = self.lr_config.config.doubly_stochastic_matrix[i]
                new_m = (
                    sum(wt * data.m for wt in doubly_st_mt)
                    - self.learning_rate * m.grad
                )
                new_c = (
                    sum(wt * data.c for wt in doubly_st_mt)
                    - self.learning_rate * c.grad
                )

                # start_state_cpy = list(start_state)
                # start_state_cpy[node_id] = m.item()

                # new_slope = (
                #     sum(
                #         frac * start_state_cpy[j] for j, frac in enumerate(doubly_st_mt)
                #     )
                #     - self.learning_rate * m.grad
                # )

                # if new_slope > self.config.max_slope:
                #     new_slope = self.config.max_slope

                # node_params[node_id] = new_slope

                # if abs(m - new_slope) <= self.config.iteration_stop_threshold:
                #     break

        # for node_id, new_slope in enumerate(node_params):
        #     new_slope_cleaned = self.__clean_float_to_step_size_single(new_slope)
        #     if new_slope_cleaned != start_state[node_id]:
        #         new_node_params = self.__copy_replace_indx_value(
        #             list(start_state), node_id, new_slope_cleaned
        #         )
        #         new_node_params = tuple(new_node_params)
        #         yield node_id, new_node_params


if __name__ == "__main__":
    import os
    import sys

    utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
    sys.path.append(utils_path)

    from command_line_helpers import get_graph

    graph_names = ["star_graph_n4"]
    for graph_name, graph in get_graph(graph_names):
        lr = LinearRegressionCVFAnalysisV2(graph_name, graph)
        for v in lr.possible_node_values[0]:
            mapped_v = lr.possible_node_values_mapping[0][v]
            if lr.is_invariant([mapped_v for _ in lr.nodes]):
                print(v, mapped_v)
