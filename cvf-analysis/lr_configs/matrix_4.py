import os
import numpy as np


iterations = 100

learning_rate = 0.0001
stop_threshold = 0.0001
slope_step = np.float64(0.025)
slope_step_decimals = 3
min_slope = np.float64(0.800)
max_slope = np.float64(1.900)

no_of_nodes = 4

file_path = os.path.join(
    os.getenv("CVF_CODE_ROOT", "/"),
    "linear_regression",
    "SOCR-HeightWeight.csv",
)

ds_matrix_config_id = 4

doubly_stochastic_matrix_config =  [
    [1 / 8, 1/8, 1/8, 1/8, 1/8, 1 / 8, 1 / 8, 1 / 8],
    [1 / 8, 7 / 8, 0, 0, 0, 0, 0, 0],
    [1 / 8, 0, 7 / 8, 0, 0, 0, 0, 0],
    [1 / 8, 0, 0, 7 / 8, 0, 0, 0, 0],
    [1 / 8, 0, 0, 0, 7 / 8, 0, 0, 0],
    [1 / 8, 0, 0, 0, 0, 7 / 8, 0, 0],
    [1 / 8, 0, 0, 0, 0, 0,7 / 8, 0],
    [1 / 8, 0, 0, 0, 0, 0, 0, 7 / 8],
]
