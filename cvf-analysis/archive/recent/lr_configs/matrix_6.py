import os

import numpy as np

learning_rate = 0.0001
stop_threshold = 0.0001
slope_step = np.float64(0.025)
slope_step_decimals = 3
min_slope = np.float64(1.300)
max_slope = np.float64(1.900)

data_path = os.path.join(
    "linear_regression", "SOCR-HeightWeight-XY.csv"
)  # from code root

iterations = 100


df_random_state = 60


doubly_stochastic_matrix = [
    [1 / 4, 1 / 8, 1 / 4, 1 / 8, 1 / 4],
    [1 / 8, 4 / 8, 0, 3 / 8, 0],
    [1 / 4, 0, 3 / 4, 0, 0],
    [1 / 8, 3 / 8, 0, 4 / 8, 0],
    [1 / 4, 0, 0, 0, 3 / 4],
]

# 0: keeping 0.25 sharing 0.75
# 1: keeping 0.5 sharing 0.5
# 2: keeping 0.75 sharing 0.25
# 3: keeping 0.5 sharing 0.5
# 4: keeping 0.75 sharing 0.25
