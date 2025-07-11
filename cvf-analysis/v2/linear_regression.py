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
            LinearRegressionData(*i)
            for i in product(possible_m_values, possible_c_values)
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
        return self.lr_config.df[self.lr_config.df["node"] == node_id]

    def __clean_m_to_step_size(self, m):
        quotient = np.divide(m, self.lr_config.config.m_step)
        if quotient == int(quotient):
            return np.round(m, self.lr_config.config.m_step_decimals)
        return np.round(
            np.int64(quotient) * self.lr_config.config.m_step,
            self.lr_config.config.m_step_decimals,
        )

    def __clean_c_to_step_size(self, c):
        quotient = np.divide(c, self.lr_config.config.c_step)
        if quotient == int(quotient):
            return np.round(c, self.lr_config.config.c_step_decimals)
        return np.round(
            np.int64(quotient) * self.lr_config.config.c_step,
            self.lr_config.config.c_step_decimals,
        )

    def _get_program_transitions_as_configs(self, start_state):

        for position in range(len(self.nodes)):
            data = self.get_actual_config_node_values(position, start_state[position])

            # for _ in range(1, self.lr_config.config.iterations + 1):
            m = torch.tensor(data.m, requires_grad=True)
            c = torch.tensor(data.c, requires_grad=True)

            node_df = self.__get_node_data_df(position)
            X_node = torch.tensor(node_df["X"].array)
            y_true = torch.tensor(node_df["y"].array)
            y_pred = m * X_node + c
            loss = ((y_pred - y_true) ** 2).mean()
            loss.backward()

            # new values to update
            doubly_st_mt = self.lr_config.config.doubly_stochastic_matrix[position]
            new_m = (
                sum(wt * data.m for wt in doubly_st_mt)
                - self.lr_config.config.learning_rate * m.grad
            )
            new_c = (
                sum(wt * data.c for wt in doubly_st_mt)
                - self.lr_config.config.learning_rate * c.grad
            )

            if new_m > self.lr_config.config.max_m:
                new_m = self.lr_config.config.max_m

            if new_c > self.lr_config.config.max_c:
                new_c = self.lr_config.config.max_c

            # need to first normalize and check the values
            new_m = self.__clean_m_to_step_size(new_m)
            new_c = self.__clean_c_to_step_size(new_c)
            #

            perturb_node_val_indx = self.possible_node_values_mapping[position][
                LinearRegressionData(new_m, new_c)
            ]

            if perturb_node_val_indx != start_state[position]:
                perturb_state = tuple(
                    [
                        *start_state[:position],
                        perturb_node_val_indx,
                        *start_state[position + 1 :],
                    ]
                )

                yield position, perturb_state


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
            # if lr.is_invariant([mapped_v for _ in lr.nodes]):
            #     print(v, mapped_v)
            print("mapped_v", mapped_v)
            for i in lr._get_program_transitions_as_configs(
                [mapped_v for _ in lr.nodes]
            ):
                print("i", i)
                print(lr.get_actual_config_values(i[1]))
            break
