import random

from typing import Tuple
from base import ProgramData, CVFAnalysisV2


class MaximalMatchingData(ProgramData):
    def __init__(self, p: int, m: bool):
        self.p = p
        self.m = m
        self.data = (self.p, self.m)


class MaximalMatchingCVFAnalysisV2(CVFAnalysisV2):
    """Not complete"""

    results_dir = "maximal_matching"

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
                if self.possible_node_values[i][state[i]].p == j and config.p == i:
                    return True
            return False

        for j, indx in enumerate(state):
            # update m.j
            config = self.possible_node_values[j][indx]
            if config.m != _pr_married(j, config):
                return False

            # accept a proposal
            if config.m == _pr_married(j, config):
                if config.p is None:
                    for i in self.graph[j]:
                        if self.possible_node_values[i][state[i]].p == j:
                            return False

                    for k in self.graph[j]:
                        if (
                            self.possible_node_values[k][state[k]].p is None
                            and k < j
                            and not self.possible_node_values[k][state[k]].m
                        ):
                            return False
                else:
                    i = config.p
                    if self.possible_node_values[i][state[i]].p != j and (
                        self.possible_node_values[i][state[i]].m or j <= i
                    ):
                        return False

        # print("Invariant", [self.possible_node_values[i][j] for i, j in enumerate(state)])
        return True

    def _is_program_transition(self, i, start_state, dest_state) -> bool:
        """https://inria.hal.science/inria-00127899/document#page=8.52"""

        config = self.possible_node_values[i][start_state[i]]
        dest_config = self.possible_node_values[i][dest_state[i]]

        def _pr_married(_i):
            for j in self.graph[_i]:
                config_j = self.possible_node_values[j][start_state[j]]
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
                config_j = self.possible_node_values[j][start_state[j]]
                if config_j.p == i:
                    if dest_config.p == j:
                        return True

        # Seduction
        max_j = -1
        if config.m == _pr_married(i) and config.p is None:
            for k in self.graph[i]:
                config_k = self.possible_node_values[k][start_state[k]]
                if config_k.p == i:
                    break
            else:
                for j in self.graph[i]:
                    config_j = self.possible_node_values[j][start_state[j]]
                    if config_j.p is None and j > i and not config_j.m:
                        max_j = max(max_j, j)

        if max_j >= 0 and dest_config.p == max_j:
            return True

        # Abandonment
        if config.m == _pr_married(i):
            if config.p is not None:
                j = config.p
                config_j = self.possible_node_values[j][start_state[j]]
                if config_j.p != i and (config_j.m or j <= i):
                    if dest_config.p is None:
                        return True

        return False

    def _is_program_transition_v2(
        self, node, prev_value, new_value, neighbors_w_values
    ) -> bool:
        """https://inria.hal.science/inria-00127899/document#page=8.52"""

        config = self.possible_node_values[node][prev_value]
        dest_config = self.possible_node_values[node][new_value]

        def _pr_married(_i):
            for j in self.graph[_i]:
                config_j = self.possible_node_values[j][neighbors_w_values[j]]
                if config.p == j and config_j.p == _i:
                    return True
            return False

        # Update
        if config.m != _pr_married(node):
            if dest_config.m == _pr_married(node):
                return True

        # Marriage
        if config.m == _pr_married(node) and config.p is None:
            for j in self.graph[node]:
                config_j = self.possible_node_values[j][neighbors_w_values[j]]
                if config_j.p == node:
                    if dest_config.p == j:
                        return True

        # Seduction
        max_j = -1
        if config.m == _pr_married(node) and config.p is None:
            for k in self.graph[node]:
                config_k = self.possible_node_values[k][neighbors_w_values[k]]
                if config_k.p == node:
                    break
            else:
                for j in self.graph[node]:
                    config_j = self.possible_node_values[j][neighbors_w_values[j]]
                    if config_j.p is None and j > node and not config_j.m:
                        max_j = max(max_j, j)

        if max_j >= 0 and dest_config.p == max_j:
            return True

        # Abandonment
        if config.m == _pr_married(node):
            if config.p is not None:
                j = config.p
                config_j = self.possible_node_values[j][neighbors_w_values[j]]
                if config_j.p != node and (config_j.m or j <= node):
                    if dest_config.p is None:
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
        if self.possible_node_values[position][state[position]].p is None:
            return [False]
        return [True, False]

    def possible_perturbed_state_frm(self, frm_indx):
        frm_config = self.indx_to_config(frm_indx)
        for position, value in enumerate(frm_config):
            config = self.possible_node_values[position][value]
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
                choices.append(next_value)
                break

        possible_config_m_val = {True, False} - {current_m_value}
        for perturb_m_val in possible_config_m_val:
            next_value = self.possible_node_values_mapping[node][
                MaximalMatchingData(current_p_value, perturb_m_val)
            ]
            if self._is_program_transition_v2(
                node, node_value, next_value, neighbors_w_values
            ):
                choices.append(next_value)
                break

        if choices:
            return random.sample(choices, 1)[0]

        return None


if __name__ == "__main__":
    import os
    import sys

    utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
    sys.path.append(utils_path)

    from command_line_helpers import get_graph

    graph_names = ["star_graph_n4"]
    for graph_name, graph in get_graph(graph_names):
        cvf = MaximalMatchingCVFAnalysisV2(graph_name, graph)
        result = cvf._get_next_value_given_nbrs(0, 0, {0: 1, 2: 1})
        print(result)
