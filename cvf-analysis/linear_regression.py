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
    max_slope = 3

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
        # self._find_invariants()
        # self._init_pts_rank()
        # self._find_program_transitions_n_cvfs()
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
