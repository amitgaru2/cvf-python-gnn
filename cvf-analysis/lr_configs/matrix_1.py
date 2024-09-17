import os

import numpy as np
import pandas as pd


learning_rate = 0.0001
stop_threshold = 0.0001
slope_step = np.float64(0.025)
slope_step_decimals = 3
min_slope = np.float64(0.800)
max_slope = np.float64(1.900)
df = pd.read_csv(
    os.path.join(
        os.getenv("CVF_CODE_ROOT", "/"),
        "linear_regression",
        "SOCR-HeightWeight.csv",
    )
)  # /home/agaru/research/cvf-python/
df.rename(columns={"Height(Inches)": "X", "Weight(Pounds)": "y"}, inplace=True)
df.drop("Index", axis=1, inplace=True)

doubly_stochastic_matrix = [
    [1 / 2, 1 / 4, 1 / 8, 1 / 8],
    [1 / 4, 3 / 4, 0, 0],
    [1 / 8, 0, 7 / 8, 0],
    [1 / 8, 0, 0, 7 / 8],
]
