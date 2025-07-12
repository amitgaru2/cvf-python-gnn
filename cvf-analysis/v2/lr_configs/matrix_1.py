import os

import numpy as np

learning_rate = 0.01
iteration_stop_threshold = 0.0001  # stop when, either complete `iterations` or there is no improvement above this threshold

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "custom_data__m105__c0.csv")

# got this by fitting our data at https://www.graphpad.com/quickcalcs/linear2/
invariant_m = 103,107
invariant_c = -0.5, 0.5
invariant = (invariant_m[0], invariant_c[0]), (invariant_m[1], invariant_c[1])

# m configs
m_step = np.float64(0.5)
m_step_decimals = 1
min_m = np.float64(100)
max_m = np.float64(110)

# c configs
c_step = np.float64(0.2)
c_step_decimals = 1
min_c = np.float64(0)
max_c = np.float64(0)

iterations = 100

doubly_stochastic_matrix = [
    [1 / 2, 1 / 4, 1 / 8, 1 / 8],
    [1 / 4, 3 / 4, 0, 0],
    [1 / 8, 0, 7 / 8, 0],
    [1 / 8, 0, 0, 7 / 8],
]

df_random_state = 36
