import os
import math
import random

import torch
import numpy as np

from typing import Tuple

from base import (
    ProgramData,
    CVFAnalysisV2,
    create_dir_if_not_exists,
    CHUNK_CONFIG_RATIO,
    logger,
)


class MaximalMatchingData(ProgramData):
    N_VARS = 2

    def __init__(self, p: int, m: bool):
        self.data = (p, m)

    @property
    def p(self):
        return self.data[0]

    @property
    def m(self):
        return self.data[1]


class MaximalMatchingCVFAnalysisV2(CVFAnalysisV2):
    """(p, m) binded as a single variable."""

    results_dir = "maximal_matching"
    DataKlass = MaximalMatchingData

    def get_possible_node_values(self):
        """include m values as well"""
        result = []
        mapping = []
        for position in self.nodes:
            possible_values = list()
            for neighbor in [None, *self.graph[position]]:
                for m in (False, True):
                    possible_values.append(MaximalMatchingData(neighbor, m))

            mapping.append({v: i for i, v in enumerate(possible_values)})
            result.append(tuple(possible_values))

        return result, mapping

    def is_invariant(self, state: Tuple[int]):
        """check invariant"""

        def _pr_married(j, config):
            for i in self.graph[j]:
                if (
                    self.get_actual_config_node_values(i, state[i]).p == j
                    and config.p == i
                ):
                    return True
            return False

        for j, indx in enumerate(state):
            # update m.j
            config = self.get_actual_config_node_values(j, indx)
            if config.m != _pr_married(j, config):
                return False

            # accept a proposal
            if config.m == _pr_married(j, config):
                if config.p is None:
                    for i in self.graph[j]:
                        if self.get_actual_config_node_values(i, state[i]).p == j:
                            return False

                    for k in self.graph[j]:
                        if (
                            self.get_actual_config_node_values(k, state[k]).p is None
                            and k < j
                            and not self.get_actual_config_node_values(k, state[k]).m
                        ):
                            return False
                else:
                    i = config.p
                    if self.get_actual_config_node_values(i, state[i]).p != j and (
                        self.get_actual_config_node_values(i, state[i]).m or j <= i
                    ):
                        return False

        # print("Invariant", [self.possible_node_values[i][j] for i, j in enumerate(state)])
        return True

    def _is_program_transition(self, i, start_state, dest_state) -> bool:
        """https://inria.hal.science/inria-00127899/document#page=8.52"""

        config = self.get_actual_config_node_values(i, start_state[i])
        dest_config = self.get_actual_config_node_values(i, dest_state[i])

        def _pr_married(_i):
            for j in self.graph[_i]:
                config_j = self.get_actual_config_node_values(j, start_state[j])
                if config.p == j and config_j.p == _i:
                    return True
            return False

        # Update
        if config.m != _pr_married(i):
            if dest_config.m == _pr_married(i):
                return True

        # Marriage
        if config.m == _pr_married(i) and config.p is None:
            for j in self.graph[i]:
                config_j = self.get_actual_config_node_values(j, start_state[j])
                if config_j.p == i:
                    if dest_config.p == j:
                        return True

        # Seduction
        max_j = -1
        if config.m == _pr_married(i) and config.p is None:
            for k in self.graph[i]:
                config_k = self.get_actual_config_node_values(k, start_state[k])
                if config_k.p == i:
                    break
            else:
                for j in self.graph[i]:
                    config_j = self.get_actual_config_node_values(j, start_state[j])
                    if config_j.p is None and j > i and not config_j.m:
                        max_j = max(max_j, j)

        if max_j >= 0 and dest_config.p == max_j:
            return True

        # Abandonment
        if config.m == _pr_married(i):
            if config.p is not None:
                j = config.p
                config_j = self.get_actual_config_node_values(j, start_state[j])
                if config_j.p != i and (config_j.m or j <= i):
                    if dest_config.p is None:
                        return True

        return False

    def _is_program_transition_v2(
        self, node, prev_value, new_value, neighbors_w_values
    ) -> bool:
        """https://inria.hal.science/inria-00127899/document#page=8.52"""

        config = self.get_actual_config_node_values(node, prev_value)
        new_config = self.get_actual_config_node_values(node, new_value)

        # PRmarried (i) = ∃j ∈ N (i) : (pi = j and pj = i)
        pr_married_node = False
        for j in self.graph[node]:
            config_j = self.get_actual_config_node_values(j, neighbors_w_values[j])
            if config.p == j and config_j.p == node:
                pr_married_node = True
                break

        # Update
        if config.m != pr_married_node:
            # if there is change in m and the current m is not equivalent to pr_married_node, then it is a transition
            if new_config.m == pr_married_node:
                logger.debug("Update")
                return True

        # Marriage
        if config.m == pr_married_node and config.p is None:
            # config.m is True => pr_married => True => config.p != None; so this case doesn't apply
            # config.m is False; config.p is None then the new p should be the node `j` that has p = node
            for j in self.graph[node]:
                config_j = self.get_actual_config_node_values(j, neighbors_w_values[j])
                if config_j.p == node:
                    if new_config.p == j:
                        logger.debug("Marriage")
                        return True

        # Seduction
        max_j = -1
        # seduce the neighbor that has the highest index among the neighbors that has index higher than self
        # 0 can seduce 1, 2, 3 and selects 3 if all eligible; 1, 2, 3 cannot seduce 0
        if config.m == pr_married_node and config.p is None:
            for k in self.graph[node]:
                config_k = self.get_actual_config_node_values(k, neighbors_w_values[k])
                if config_k.p == node:
                    break
            else:
                for j in self.graph[node]:
                    config_j = self.get_actual_config_node_values(
                        j, neighbors_w_values[j]
                    )
                    if config_j.p is None and j > node and not config_j.m:
                        max_j = max(max_j, j)

        if max_j >= 0 and new_config.p == max_j:
            logger.debug("Seduction")
            return True

        # Abandonment
        if config.m == pr_married_node:
            if config.p is not None:
                j = config.p
                config_j = self.get_actual_config_node_values(j, neighbors_w_values[j])
                if config_j.p != node and (config_j.m or j <= node):
                    # there is another nbr j that points to node but the neighbor p doesn't point to node
                    if new_config.p is None:
                        logger.debug("Abandonment")
                        return True

        return False

    def _get_program_transitions_as_configs(self, start_state):
        for position, node_val_indx in enumerate(start_state):
            data = self.get_actual_config_node_values(position, node_val_indx)
            current_p_value = data.p
            current_m_value = data.m

            possible_config_p_val = {
                i.p for i in self.possible_node_values[position]
            } - {current_p_value}

            for perturb_p_val in possible_config_p_val:
                perturb_node_val_indx = self.possible_node_values_mapping[position][
                    MaximalMatchingData(perturb_p_val, current_m_value)
                ]
                perturb_state = tuple(
                    [
                        *start_state[:position],
                        perturb_node_val_indx,
                        *start_state[position + 1 :],
                    ]
                )
                if self._is_program_transition(position, start_state, perturb_state):
                    yield position, perturb_state
                    break

            possible_config_m_val = {True, False} - {current_m_value}
            for perturb_m_val in possible_config_m_val:
                perturb_node_val_indx = self.possible_node_values_mapping[position][
                    MaximalMatchingData(current_p_value, perturb_m_val)
                ]
                perturb_state = tuple(
                    [
                        *start_state[:position],
                        perturb_node_val_indx,
                        *start_state[position + 1 :],
                    ]
                )
                if self._is_program_transition(position, start_state, perturb_state):
                    yield position, perturb_state
                    break

    def _evaluate_perturbed_pr_married(self, position, state):
        if self.get_actual_config_node_values(position, state[position]).p is None:
            return [False]
        return [True, False]

    def possible_perturbed_state_frm(self, frm_indx):
        frm_config = self.indx_to_config(frm_indx)
        for position, value in enumerate(frm_config):
            config = self.get_actual_config_node_values(position, value)
            for a_pr_married_value in self._evaluate_perturbed_pr_married(
                position, frm_config
            ):
                perturb_node_val_indxs = []
                if config.m is not a_pr_married_value:
                    perturb_node_val_indxs.append(
                        self.possible_node_values_mapping[position][
                            MaximalMatchingData(config.p, a_pr_married_value)
                        ]
                    )
                else:
                    if config.p is None:
                        for j in self.graph[position]:
                            perturb_node_val_indxs.append(
                                self.possible_node_values_mapping[position][
                                    MaximalMatchingData(j, a_pr_married_value)
                                ]
                            )
                    else:
                        perturb_node_val_indxs.append(
                            self.possible_node_values_mapping[position][
                                MaximalMatchingData(None, a_pr_married_value)
                            ]
                        )

                if perturb_node_val_indxs:
                    for perturb_node_val_indx in perturb_node_val_indxs:
                        perturb_state = tuple(
                            [
                                *frm_config[:position],
                                perturb_node_val_indx,
                                *frm_config[position + 1 :],
                            ]
                        )
                        to_indx = self.config_to_indx(perturb_state)
                        yield position, to_indx

    def _get_next_value_given_nbrs(self, node, node_value, neighbors_w_values):
        """designed for simulation v2"""
        # for position, node_val_indx in enumerate(start_state):
        data = self.get_actual_config_node_values(node, node_value)
        current_p_value = data.p
        current_m_value = data.m

        possible_config_p_val = {i.p for i in self.possible_node_values[node]} - {
            current_p_value
        }

        choices = []
        for perturb_p_val in possible_config_p_val:
            next_value = self.possible_node_values_mapping[node][
                MaximalMatchingData(perturb_p_val, current_m_value)
            ]
            if self._is_program_transition_v2(
                node, node_value, next_value, neighbors_w_values
            ):
                choices.append((next_value, 0))  # changed the first var
                break

        possible_config_m_val = {True, False} - {current_m_value}
        for perturb_m_val in possible_config_m_val:
            next_value = self.possible_node_values_mapping[node][
                MaximalMatchingData(current_p_value, perturb_m_val)
            ]
            if self._is_program_transition_v2(
                node, node_value, next_value, neighbors_w_values
            ):
                choices.append((next_value, 1))  # changed the second var
                break

        if choices:
            return random.choice(choices)

        return None, None

    def generate_dataset_for_ml_v2(self):
        chunk_dataset_dir = os.path.join(
            "datasets",
            self.results_dir,
            f"{self.graph_name}_config_rank_dataset",
        )
        create_dir_if_not_exists(chunk_dataset_dir)

        def _save_chunk(chunk_id, X_all, y_all):
            torch.save(
                {
                    "X": torch.from_numpy(
                        X_all.reshape(X_all.shape[0], 2, -1).transpose(0, 2, 1)
                    ).float(),
                    "y": torch.from_numpy(np.array(y_all)).float(),
                },
                os.path.join(chunk_dataset_dir, f"chunk_{chunk_id:04d}.pt"),
            )

        def _get_p_encoding(p_value):
            if p_value is None:
                p_value = highest_p_value + 1

            p_value = np.array([p_value])
            p_encoded_value = np.eye(highest_p_value + 2)[p_value][0]
            return p_encoded_value

        def _get_m_encoding(m_value):
            m_encoded_value = np.array([1.0]) if m_value else np.array([0.0])
            return m_encoded_value

        def _get_p_m_encoding(p_value, m_value):
            return np.hstack((_get_p_encoding(p_value), _get_m_encoding(m_value)))

        def _get_encoded_config(config):
            return np.vstack([_get_p_m_encoding(v.data[0], v.data[1]) for v in config])

        highest_p_value = 15
        chunk_id = 0

        X_all = []
        y_all = []

        for k, v in enumerate(self.global_rank_map):
            y = np.array([math.ceil(v[0] / v[1])])
            config = _get_encoded_config(
                self.get_actual_config_values(self.indx_to_config(k))
            )
            if k in self.config_successors:
                succ = np.array(
                    [
                        _get_encoded_config(
                            self.get_actual_config_values(self.indx_to_config(i))
                        )
                        for i in self.config_successors[k]
                    ]
                )
                succ = np.mean(succ, axis=0)
            else:
                succ = np.full((config.shape[0], config.shape[1]), -1)

            X_w_pad = np.vstack((config, succ))
            X_all.append(X_w_pad)
            y_all.append(y)
            if (k + 1) % CHUNK_CONFIG_RATIO == 0:
                X_all = np.array(X_all)
                _save_chunk(chunk_id, X_all, y_all)
                X_all = []
                chunk_id += 1

        if X_all:
            X_all = np.array(X_all)
            _save_chunk(chunk_id, X_all, y_all)

        # torch.save(
        #     {
        #         "X": torch.from_numpy(
        #             X_all.reshape(X_all.shape[0], 2, -1).transpose(0, 2, 1)
        #         ).float(),
        #         "y": torch.from_numpy(np.array(y_all)).float(),
        #     },
        #     os.path.join(
        #         "datasets",
        #         self.results_dir,
        #         f"{self.graph_name}_config_rank_dataset.pt",
        #     ),
        # )

    def generate_test_dataset_for_ml_v2(self):
        chunk_dataset_dir = os.path.join(
            "datasets",
            self.results_dir,
            f"{self.graph_name}_config_rank_dataset",
        )
        create_dir_if_not_exists(chunk_dataset_dir)

        def _save_chunk(chunk_id, X_all):
            torch.save(
                {
                    "X": torch.from_numpy(
                        X_all.reshape(X_all.shape[0], 2, -1).transpose(0, 2, 1)
                    ).float(),
                },
                os.path.join(chunk_dataset_dir, f"chunk_{chunk_id:04d}.pt"),
            )

        def _get_p_encoding(p_value):
            if p_value is None:
                p_value = highest_p_value + 1

            p_value = np.array([p_value])
            p_encoded_value = np.eye(highest_p_value + 2)[p_value][0]
            return p_encoded_value

        def _get_m_encoding(m_value):
            m_encoded_value = np.array([1.0]) if m_value else np.array([0.0])
            return m_encoded_value

        def _get_p_m_encoding(p_value, m_value):
            return np.hstack((_get_p_encoding(p_value), _get_m_encoding(m_value)))

        def _get_encoded_config(config):
            return np.vstack([_get_p_m_encoding(v.data[0], v.data[1]) for v in config])

        highest_p_value = 15
        X_all = []
        chunk_id = 0

        for i, k in enumerate(range(self.total_configs), 1):
            config = _get_encoded_config(
                self.get_actual_config_values(self.indx_to_config(k))
            )
            if k in self.config_successors and self.config_successors[k]:
                succ = np.array(
                    [
                        _get_encoded_config(
                            self.get_actual_config_values(self.indx_to_config(i))
                        )
                        for i in self.config_successors[k]
                    ]
                )
                succ = np.mean(succ, axis=0)
            else:
                succ = np.full((config.shape[0], config.shape[1]), -1)

            X_wo_pad = np.vstack((config, succ))
            pad_length = 15 - len(self.nodes)
            X_w_pad = np.pad(
                X_wo_pad,
                pad_width=((0, 0), (0, pad_length)),
                mode="constant",
                constant_values=-1,
            )

            X_all.append(X_w_pad)
            if i % CHUNK_CONFIG_RATIO == 0:
                X_all = np.array(X_all)
                _save_chunk(chunk_id, X_all)
                X_all = []
                chunk_id += 1

        if X_all:
            X_all = np.array(X_all)
            _save_chunk(chunk_id, X_all)

        # torch.save(
        #     {
        #         "X": torch.from_numpy(
        #             X_all.reshape(X_all.shape[0], 2, -1).transpose(0, 2, 1)
        #         ).float(),
        #     },
        #     os.path.join(
        #         "datasets",
        #         self.results_dir,
        #         f"{self.graph_name}_config_rank_dataset.pt",
        #     ),
        # )


if __name__ == "__main__":
    import os
    import sys

    utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
    sys.path.append(utils_path)

    from command_line_helpers import get_graph

    graph_names = ["graph_2_node"]
    for graph_name, graph in get_graph(graph_names):
        cvf = MaximalMatchingCVFAnalysisV2(graph_name, graph)
        c0_nf = cvf.possible_node_values_mapping[0][MaximalMatchingData(None, False)]
        # c1_2t = cvf.possible_node_values_mapping[1][MaximalMatchingData(2, True)]
        # c2_nf = cvf.possible_node_values_mapping[2][MaximalMatchingData(None, False)]
        c0_1f = cvf.possible_node_values_mapping[0][MaximalMatchingData(1, False)]
        c1_0f = cvf.possible_node_values_mapping[1][MaximalMatchingData(0, False)]
        c1_0t = cvf.possible_node_values_mapping[1][MaximalMatchingData(0, True)]
        c1_nt = cvf.possible_node_values_mapping[1][MaximalMatchingData(None, True)]
        c1_nf = cvf.possible_node_values_mapping[1][MaximalMatchingData(None, False)]
        c0_1t = cvf.possible_node_values_mapping[0][MaximalMatchingData(1, True)]
        c0_nt = cvf.possible_node_values_mapping[0][MaximalMatchingData(None, True)]

        # cx = cvf.possible_node_values_mapping[1][MaximalMatchingData(None, True)]

        result = cvf._get_next_value_given_nbrs(0, c0_1f, {1: c1_nt})
        print(result)
        print(cvf.get_actual_config_values(config=(result[0], c1_nt)))

        # result = cvf._get_next_value_given_nbrs(0, c0_1f, {1: c1_2t})
        # print(result)
        # print(cvf.get_actual_config_values(config=(result[0], c1_2t)))

        # c3 = result[0]

        # result = cvf._get_next_value_given_nbrs(1, c2, {0: c3})
        # print(result)
        # print(cvf.get_actual_config_values(config=(c3, result[0])))
        # c4 = result[0]

        # result = cvf._get_next_value_given_nbrs(1, c4, {0: c3})
        # print(result)
        # print(cvf.get_actual_config_values(config=(c3, result[0])))

        # c5 = result[0]

        # result = cvf._get_next_value_given_nbrs(0, c3, {1: cx})
        # print(result)
        # print(cvf.get_actual_config_values(config=(result[0], cx)))

        # c6 = result[0]

        # result = cvf._get_next_value_given_nbrs(1, c5, {0: c6})
        # print(result)
        # print(cvf.get_actual_config_values(config=(c6, result[0])))

        # c7 = result[0]

        # result = cvf._get_next_value_given_nbrs(1, c7, {0: c6})
        # print(result)
        # print(cvf.get_actual_config_values(config=(c6, result[0])))

        # result3 = cvf._get_next_value_given_nbrs(1, result2, {0: result1})
        # print(result3)
        # print(cvf.get_actual_config_values(config=(result1, result3)))

        # result2 = cvf._get_next_value_given_nbrs(1, 0, {0: result1})
        # # print(result2)
        # print(cvf.get_actual_config_values(config=(result1, result2)))

        # result3 = cvf._get_next_value_given_nbrs(1, result2, {0: result1})
        # print(cvf.get_actual_config_values(config=(result1, result3)))

        # result4 = cvf._get_next_value_given_nbrs(0, result1, {1: 0})
        # print(result4)
        # print(cvf.get_actual_config_values(config=(result4, 0)))

        # result = cvf.get_actual_config_values(config=(3, 0, 1, 1))
        # print(result)
        # result = cvf.get_actual_config_values(config=(2, 0, 1, 1))
        # print(result)
        # result = cvf.get_actual_config_values(config=(2, 0, 1, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(2, 2, 1, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(3, 2, 1, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(3, 2, 0, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(3, 3, 0, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(4, 0, 2, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(4, 0, 3, 0))
        # print(result)
        # result = cvf.get_actual_config_values(config=(5, 0, 3, 0))
        # print(result)
