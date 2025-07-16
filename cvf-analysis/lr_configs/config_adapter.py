import os
import importlib

import numpy as np
import pandas as pd


from custom_logger import logger


class LRConfig:

    def __init__(self, config_file) -> None:
        config_module = importlib.import_module(f"lr_configs.{config_file}")
        self._check_doubly_stochastic_matrix(config_module.doubly_stochastic_matrix)
        self.config = config_module

        self.no_of_nodes = len(self.config.doubly_stochastic_matrix)
        self.df = pd.read_csv(os.path.join("lr_configs", self.config.data_path))

        # distributed data among the available nodes
        self.df = self.df.sample(
            frac=1, random_state=self.config.df_random_state
        ).reset_index(drop=True)
        self.node_data_partitions = np.array_split(self.df, self.no_of_nodes)
        for i, node_data in enumerate(self.node_data_partitions):
            self.df.loc[node_data.index, "node"] = i

        self._preprocessing_dataset()

    def _preprocessing_dataset(self):
        pass

    @classmethod
    def _check_doubly_stochastic_matrix(cls, doubly_stochastic_matrix):
        arr = np.array(doubly_stochastic_matrix)
        try:
            for i, row in enumerate(arr):
                assert np.sum(row) == 1, f"Row index {i}"
                assert np.sum(arr[:, i]) == 1, f"Column index {i}"
        except Exception as e:
            logger.exception("Invalid doubly stochastic matrix!")
            exit(1)

        logger.info("Doubly stochastic matrix validated!")
