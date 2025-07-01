import torch
import numpy as np

from base import ProgramData, CVFAnalysisV2
from lr_configs.config_adapter import LRConfig


class LinearRegressionCVFAnalysisV2(CVFAnalysisV2):
    results_dir = "linear_regression"

    def initialize_program_helpers(self):
        self.config = LRConfig.generate_config(self.config_file)
        self.cache = {"p": {}, "q": {}, "r": {}}

    def __get_node_data_df(self, node_id):
        return self.config.df[self.config.df["node"] == node_id]

    def get_possible_node_values(self):
        result = list()
        for _ in self.nodes:
            possible_values = np.round(
                np.arange(
                    self.config.min_slope + self.config.slope_step,
                    self.config.max_slope + self.config.slope_step,
                    self.config.slope_step,
                ),
                self.config.slope_step_decimals,
            )
            result.append(tuple(possible_values))

        return result, []

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

    def is_invariant(self, config):
        return super().is_invariant(config)

    def _get_program_transitions(self, start_state):
        program_transitions = []
        node_params = list(start_state)

        for node_id in range(self.nodes):
            for _ in range(1, self.config.iterations + 1):
                m = torch.tensor(node_params[node_id], requires_grad=True)

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

                if abs(m - new_slope) <= self.config.stop_threshold:
                    break
            # else:
            #     logger.debug(
            #         "Couldn't converge node %s for the state %s", node_id, start_state
            #     )

        for node_id, new_slope in enumerate(node_params):
            new_slope_cleaned = self.__clean_float_to_step_size_single(new_slope)
            if new_slope_cleaned != start_state[node_id]:
                new_node_params = self.__copy_replace_indx_value(
                    list(start_state), node_id, new_slope_cleaned
                )
                new_node_params = tuple(new_node_params)
                program_transitions.append(new_node_params)

        # if not program_transitions:
        #     self._add_to_invariants(start_state)
        #     # logger.debug("No program transition found for %s !", start_state)

        return program_transitions
