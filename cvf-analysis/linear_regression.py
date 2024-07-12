import os
import copy

import numpy as np

from itertools import combinations

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


class LinearRegressionFullAnalysis(CVFAnalysis):
    results_prefix = "linear_regression"
    results_dir = os.path.join("results", results_prefix)

    slope_step = 0.5
    min_slope = 0
    max_slope = 4

    actual_slope = 3

    def __init__(self, graph_name, graph) -> None:
        super().__init__(graph_name, graph)
        self.doubly_stochastic_matrix_config = [
            [1 / 3, 1 / 6, 1 / 6, 1 / 3],
            [1 / 6, 1 / 6, 1 / 3, 1 / 3],
            [1 / 6, 1 / 3, 1 / 3, 1 / 6],
            [1 / 3, 1 / 3, 1 / 6, 1 / 6],
        ]

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
            for i in np.arange(
                self.min_slope + self.slope_step,
                self.max_slope + self.slope_step,
                self.slope_step,
            ):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    # def _get_adjusted_value(self, value):
    #     if value / self.slope_step == 0:
    #         return value

    #     temp = (value // self.slope_step) * self.slope_step

    #     result = temp
    #     if (value - temp) > self.slope_step / 2:
    #         result = temp + self.slope_step

    #     if result > self.max_slope:
    #         return self.max_slope

    #     if result < self.min_slope:
    #         return self.min_slope

    def _find_invariants(self):
        for state in self.configurations:
            for m in state:
                if not (
                    self.actual_slope - self.slope_step / 2
                    < m
                    <= self.actual_slope + self.slope_step / 2
                ):
                    break
            else:
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))

    def __forward(X, params):
        return [params["m"] * i + params["c"] for i in X]

    def __loss_fn(y, y_pred):
        N = len(y)
        return (1 / N) * sum((y[i] - y_pred[i]) ** 2 for i in range(N))

    def __r2_score(y, y_mean, y_pred):
        N = len(y)
        rss = sum((y[i] - y_pred[i]) ** 2 for i in range(N))
        tss = sum((y[i] - y_mean) ** 2 for i in range(N))
        r2 = 1 - rss / tss
        return r2

    def __gradient_m(X, y, y_pred):
        N = len(y)
        return (-2 / N) * sum((X[i] * (y[i] - y_pred[i])) for i in range(N))

    def __gradient_c(y, y_pred):
        N = len(y)
        return (-2 / N) * sum((y[i] - y_pred[i]) for i in range(N))

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        perturbed_m = dest_state[perturb_pos]
        original_m = start_state[perturb_pos]

        if abs(original_m - self.actual_slope) >= abs(perturbed_m - self.actual_slope):
            return True

        return False

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        all_slope_values = set(
            np.arange(self.min_slope, self.max_slope + self.slope_step, self.slope_step)
        )
        for position, val in enumerate(start_state):
            possible_slope_values = all_slope_values - {val}
            for perturb_val in possible_slope_values:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

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
