import os
import importlib

import pandas as pd


class Config:

    def __init__(
        self,
        learning_rate,
        stop_threshold,
        slope_step,
        min_slope,
        max_slope,
        data_path,
        doubly_stochastic_matrix,
    ) -> None:
        self.learning_rate = learning_rate
        self.stop_threshold = stop_threshold
        self.slope_step = slope_step
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.df = pd.read_csv(
            os.path.join(
                os.getenv("CVF_CODE_ROOT", "/"),
                data_path,
            )
        )
        self.doubly_stochastic_matrix = doubly_stochastic_matrix
        self.no_of_nodes = len(self.doubly_stochastic_matrix)

    @classmethod
    def generate_config(cls, config_file):
        config_module = importlib.import_module(f"{config_file}")
        return Config(
            learning_rate=config_module.learning_rate,
            stop_threshold=config_module.stop_threshold,
            min_slope=config_module.min_slope,
            max_slope=config_module.max_slope,
            data_path=config_module.data_path,
            doubly_stochastic_matrix=config_module.doubly_stochastic_matrix,
        )
