import os

import numpy as np

learning_rate = 0.001
iteration_stop_threshold = 0.0001  # stop when, either complete `iterations` or there is no improvement above this threshold

data_path = os.path.join("datasets", "SOCR-HeightWeight-XY.csv")

# got this by fitting our data at https://www.graphpad.com/quickcalcs/linear2/
invariant_m = 3.042 - 0.1672, 3.042 + 0.1672
invariant_c = -79.54 - 11.38, -79.54 + 11.38
invariant = (invariant_m[0], invariant_c[0]), (invariant_m[1], invariant_c[1])

# m configs
m_step = np.float64(0.025)
m_step_decimals = 3
min_m = np.float64(2.500)
max_m = np.float64(3.500)

# c configs
c_step = np.float64(10.0)
c_step_decimals = 1
min_c = np.float64(-100.0)
max_c = np.float64(0.0)


iterations = 100

doubly_stochastic_matrix = [
    [1 / 2, 1 / 4, 1 / 8, 1 / 8],
    [1 / 4, 3 / 4, 0, 0],
    [1 / 8, 0, 7 / 8, 0],
    [1 / 8, 0, 0, 7 / 8],
]

df_random_state = 36
