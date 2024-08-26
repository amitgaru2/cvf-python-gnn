import os
import copy
import json
import pprint

import numpy as np
import pandas as pd

from cvf_analysis import CVFAnalysis, logger


class LinearRegressionFullAnalysis(CVFAnalysis):
    results_prefix = "linear_regression"
    results_dir = os.path.join("results", results_prefix)

    def __init__(self, graph_name, graph) -> None:
        super().__init__(graph_name, graph)

        self.learning_rate = 0.001
        self.iterations = 100

        self.slope_step_decimals = 1
        self.min_slope = np.float64(0.0)
        self.max_slope = np.float64(1.1)
        self.no_of_nodes = 3
        self.df = pd.read_csv(
            "/home/agaru/research/cvf-python/linear_regression/random-data.csv"
        )
        self.doubly_stochastic_matrix_config = [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 2 / 3, 0],
            [1 / 3, 0, 2 / 3],
        ]
        self.actual_m = 0.9
        self.actual_b = -0.11847322643445737

        # self.slope_step_decimals = 1
        # self.min_slope = 1.7
        # self.max_slope = 2.3
        # self.no_of_nodes = 4
        # self.df = pd.read_csv(
        #     "/home/amitgaru2/research/cvf-python/linear_regression/SOCR-HeightWeight.csv"
        # )
        # self.df.rename(
        #     columns={"Height(Inches)": "X", "Weight(Pounds)": "y"}, inplace=True
        # )
        # self.doubly_stochastic_matrix_config = [
        #     [1 / 2, 1 / 4, 1 / 8, 1 / 8],
        #     [1 / 4, 3 / 4, 0, 0],
        #     [1 / 8, 0, 7 / 8, 0],
        #     [1 / 8, 0, 0, 7 / 8],
        # ]
        # self.actual_m = 3.08

        self.slope_step = 1 / (10**self.slope_step_decimals)
        self.node_data_partitions = np.array_split(self.df, self.no_of_nodes)
        for i, node_data in enumerate(self.node_data_partitions):
            self.df.loc[node_data.index, "node"] = i

        self.df["partition"] = -1
        for i in range(self.no_of_nodes):
            node_filter = self.df["node"] == i
            node_df = self.df[node_filter]
            partitions = self.__gen_test_data_partition_frm_df(
                self.no_of_nodes, node_df
            )
            for i, p in enumerate(partitions):
                self.df.loc[self.df.index.isin(p.index.values), "partition"] = i

    def __gen_test_data_partition_frm_df(self, partitions, df):
        shuffled = df.sample(frac=1)
        result = np.array_split(shuffled, partitions)
        return result

    def _start(self):
        self._gen_configurations()
        self._find_invariants()
        self._init_pts_rank()
        # self._find_program_transitions()
        self._find_program_transitions_v2()
        # self._find_program_transitions_n_cvfs()
        # self.__save_pts_to_file()
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
            for i in np.round(
                np.arange(
                    self.min_slope + self.slope_step,
                    self.max_slope + self.slope_step,
                    self.slope_step,
                ),
                2,
            ):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    # def __get_adjusted_value(self, value):
    #     if value > self.max_slope:
    #         return self.max_slope

    #     if value < self.min_slope:
    #         return self.min_slope

    #     result = value

    #     if result / self.slope_step != 0:
    #         result = (result // self.slope_step) * self.slope_step

    #     if (value - result) > self.slope_step / 2:
    #         result = result + self.slope_step

    #     return result

    def _find_invariants(self):
        min_loss_sum = 1000000
        min_loss_sum_state = None
        for state in self.configurations:
            temp = 0
            for node, m in enumerate(state):
                node_df = self.__get_node_data_df(node)
                X_node = node_df["X"].array
                y_node = node_df["y"].array
                params = {"m": m, "c": 0}
                y_node_pred = self.__forward(X_node, params)
                loss = self.__loss_fn(y_node, y_node_pred)
                temp += loss

            if abs(temp) < min_loss_sum:
                min_loss_sum = abs(temp)
                min_loss_sum_state = state

            # for m in state:
            #     if not (
            #         self.actual_m - self.slope_step / 2
            #         < m
            #         <= self.actual_m + self.slope_step / 2
            #     ):
            #         break
            # else:
            #     self.invariants.add(state)

        self.invariants.add(min_loss_sum_state)
        print("Invariants", self.invariants, "min loss sum", min_loss_sum)
        logger.info("No. of Invariants: %s", len(self.invariants))

    def __forward(self, X, params):
        return params["m"] * X + params["c"]

    def __loss_fn(self, y, y_pred):
        N = len(y)
        return (1 / N) * sum((y[i] - y_pred[i]) ** 2 for i in range(N))

    def __gradient_m(self, X, y, y_pred):
        N = len(y)
        return (-2 / N) * np.sum(X * (y - y_pred))

    def __get_node_data_df(self, node_id):
        return self.df[self.df["node"] == node_id]

    def __clean_float_to_step_size_single(self, slope):
        return np.round(slope, self.slope_step_decimals)

    def __clean_float_to_step_size(self, node_slopes):
        result = []
        for slope in node_slopes:
            result.append(self.__clean_float_to_step_size_single(slope))

        return result

    def __copy_replace_indx_value(self, lst, indx, value):
        lst_copy = lst.copy()
        lst_copy[indx] = value
        return lst_copy

    def _find_program_transitions(self):
        node_params = [self.min_slope for i in range(self.no_of_nodes)]
        program_transitions = []
        for i in range(1, self.iterations + 1):
            prev_node_params = node_params.copy()
            for node_id in range(self.no_of_nodes):
                m_node = node_params[node_id]

                node_df = self.__get_node_data_df(node_id)
                X_node = node_df["X"].array
                y_node = node_df["y"].array

                y_node_pred = self.__forward(X_node, {"m": m_node, "c": 0})
                grad_m = self.__gradient_m(X_node, y_node, y_node_pred)

                doubly_st_mt = self.doubly_stochastic_matrix_config[node_id]

                node_params[node_id] = (
                    sum(
                        frac * prev_node_params[i]
                        for i, frac in enumerate(doubly_st_mt)
                    )
                    - self.learning_rate * grad_m
                )

            program_transitions.append(prev_node_params)

        # find step by given self.slope_step; remove redundant state transitions
        actual_program_transitions = []
        prev_trans = program_transitions[0]
        prev_trans_cleaned = self.__clean_float_to_step_size(prev_trans)
        actual_program_transitions.append(prev_trans_cleaned)
        for trans in program_transitions[1:]:
            trans_cleaned = self.__clean_float_to_step_size(trans)
            if prev_trans_cleaned != trans_cleaned:
                actual_program_transitions.append(trans_cleaned)
            prev_trans_cleaned = trans_cleaned

        pprint.pprint(actual_program_transitions)
        print(len(actual_program_transitions))

    def _find_program_transitions_v2(self):
        searched_states = set()
        program_transitions = {
            tuple([self.min_slope for i in range(self.no_of_nodes)]): []
        }
        actual_program_transitions = []
        left_tobe_searched_node_params = program_transitions.keys() - searched_states
        while left_tobe_searched_node_params:
            actual_node_params = list(left_tobe_searched_node_params)[0]
            node_params = list(actual_node_params)
            for i in range(1, 10 + 1):
                prev_node_params = node_params.copy()
                for node_id in range(self.no_of_nodes):
                    m_node = prev_node_params[node_id]

                    node_df = self.__get_node_data_df(node_id)
                    X_node = node_df["X"].array
                    y_node = node_df["y"].array

                    y_node_pred = self.__forward(X_node, {"m": m_node, "c": 0})
                    grad_m = self.__gradient_m(X_node, y_node, y_node_pred)

                    doubly_st_mt = self.doubly_stochastic_matrix_config[node_id]

                    new_slope = (
                        sum(
                            frac * prev_node_params[j]
                            for j, frac in enumerate(doubly_st_mt)
                        )
                        - self.learning_rate * grad_m
                    )
                    new_slope_cleaned = self.__clean_float_to_step_size_single(
                        new_slope
                    )
                    if new_slope_cleaned != self.__clean_float_to_step_size_single(
                        node_params[node_id]
                    ):
                        new_node_params = self.__copy_replace_indx_value(
                            prev_node_params, node_id, new_slope_cleaned
                        )
                        new_node_params = tuple(
                            self.__clean_float_to_step_size(new_node_params)
                        )
                        actual_program_transitions.append(new_node_params)
                    else:
                        node_params[node_id] = new_slope

                if actual_program_transitions:
                    program_transitions[actual_node_params].extend(
                        actual_program_transitions
                    )
                    for pt in actual_program_transitions:
                        if pt not in program_transitions:
                            program_transitions[pt] = []
                    actual_program_transitions = []
                    break
            else:
                print("No program transition found for", actual_node_params)

            searched_states.add(actual_node_params)
            left_tobe_searched_node_params = (
                program_transitions.keys() - searched_states
            )

        pprint.pprint(program_transitions)

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        all_slope_values = set(
            np.round(
                np.arange(
                    self.min_slope, self.max_slope + self.slope_step, self.slope_step
                ),
                2,
            )
        )

        if start_state in self.invariants:
            return program_transitions

        invariant = list(self.invariants)[0]

        for position, val in enumerate(start_state):
            if val == invariant[position]:
                continue
            node_df = self.__get_node_data_df(position)
            X_node = node_df["X"].array
            y_node = node_df["y"].array
            params = {"m": val, "c": 0}
            y_node_pred = self.__forward(X_node, params)
            grad_m = self.__gradient_m(X_node, y_node, y_node_pred)

            # m_pred_add = np.round(val + self.slope_step, self.slope_step_decimals)
            # L_add = self.__newton_raphson_L(0, m_pred_add, val, grad_m)

            # m_pred_sub = np.round(val - self.slope_step, self.slope_step_decimals)
            # L_sub = self.__newton_raphson_L(0, m_pred_sub, val, grad_m)

            # perturb_state = list(start_state)
            # perturb_val = None
            # if L_add <= 0 and L_sub <= 0:
            #     # no program transition
            #     pass
            # elif L_add >= 0 and L_sub >= 0:
            #     min_L = min(L_add, L_sub)
            #     perturb_val = m_pred_add if min_L == L_add else m_pred_sub
            # elif L_add <= 0:
            #     # program transition to negative step
            #     perturb_val = m_pred_sub
            # else:
            #     # program transition to positive step
            #     perturb_val = m_pred_add

            # if perturb_val is not None:
            #     # print("position =", position, "perturb_val =", perturb_val)
            #     perturb_state[position] = perturb_val
            #     perturb_state = tuple(perturb_state)
            #     program_transitions.add(perturb_state)

            # possible_slope_values = all_slope_values - {val}
            # for perturb_val in possible_slope_values:
            #     perturb_state = list(start_state)
            #     perturb_state[position] = perturb_val
            #     perturb_state = tuple(perturb_state)
            #     delta, is_pt = self._is_program_transition(
            #         position, start_state, perturb_state, grad_m
            #     )
            #     if is_pt:
            #         program_transitions.add(perturb_state)
            #     else:
            #         if (
            #             np.round(abs(perturb_val - val), self.slope_step_decimals)
            #             <= self.slope_step
            #         ):
            #             deltas[position][perturb_state] = delta

        # if not program_transitions:
        #     print("program transitions not found for", start_state)
        #     input()
        # else:
        #     if self.invariants & program_transitions:
        #         program_transitions = self.invariants & program_transitions

        return program_transitions

    def __newton_raphson_L(self, L0, m_pred, m, grad_m):
        def f_L(L):
            return m_pred - m + L * grad_m

        def f_L_der(L):
            return grad_m

        numerator = f_L(L0)
        denominator = f_L_der(L0)

        while numerator != 0:
            new_L0 = L0 - numerator / denominator
            numerator = f_L(new_L0)
            denominator = f_L_der(new_L0)
            if abs(new_L0 - L0) <= 0.00001:
                L0 = new_L0
                break
            L0 = new_L0

        return L0

    def _get_cvfs(self, start_state):
        cvfs = dict()
        all_slope_values = set(
            np.round(
                np.arange(
                    self.min_slope, self.max_slope + self.slope_step, self.slope_step
                ),
                self.slope_step_decimals,
            )
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

    def __save_pts_to_file(self):
        def _map_key(state):
            return json.dumps([float(k) for k in state])

        pts = {
            _map_key(state): list(pts["program_transitions"])
            for state, pts in self.pts_n_cvfs.items()
        }

        json.dump(pts, open("output.json", "w"))
