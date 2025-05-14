import math

from typing import Tuple
from collections import defaultdict
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

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        j = perturb_pos
        state = start_state
        config = self.possible_node_values[perturb_pos][state[perturb_pos]]
        dest_config = self.possible_node_values[perturb_pos][dest_state[perturb_pos]]

        def _pr_married(j, config):
            for i in self.graph[j]:
                if self.possible_node_values[i][state[i]].p == j and config.p == i:
                    return True
            return False

        # update m.j
        if config.m != _pr_married(j, config):
            if dest_config.m == _pr_married(j, config):
                return True
        else:
            if config.p is None:
                for i in self.graph[j]:
                    if (
                        self.possible_node_values[i][state[i]].p == j
                        and dest_config.p == i
                    ):
                        return True

                # make a proposal
                for i in self.graph[j]:
                    if self.possible_node_values[i][state[i]].p == j:
                        break
                else:
                    max_k = -1
                    for k in self.graph[j]:
                        if (
                            self.possible_node_values[k][state[k]].p is None
                            and k < j
                            and not self.possible_node_values[k][state[k]].m
                        ):
                            if k > max_k:
                                max_k = k

                    if max_k >= 0 and dest_config.p == max_k:
                        return True
            else:
                # withdraw a proposal
                i = config.p
                if self.possible_node_values[i][state[i]].p != j and (
                    self.possible_node_values[i][state[i]].m or j <= i
                ):
                    if dest_config.p is None:
                        return True

        return False

    def _get_program_transitions_as_configs(self, start_state):
        for position, node_val_indx in enumerate(start_state):
            current_p_value = self.possible_node_values[position][node_val_indx].p
            current_m_value = self.possible_node_values[position][node_val_indx].m

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
                    yield perturb_state

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
                    yield perturb_state

    def _evaluate_perturbed_pr_married(self, position, state):
        if self.possible_node_values[position][state[position]].p is None:
            return [False]
        return [True, False]

    def find_rank_effect(self):

        def _save_perturbation_implications(_position, _frm_indx, _perturb_state):
            to_indx = self.config_to_indx(_perturb_state)
            rank_effect = math.ceil(
                self.global_rank_map[_frm_indx, 0] / self.global_rank_map[_frm_indx, 1]
            ) - math.ceil(
                self.global_rank_map[to_indx, 0] / self.global_rank_map[to_indx, 1]
            )
            self.global_avg_rank_effect[rank_effect] += 1
            if _position not in self.global_avg_node_rank_effect:
                self.global_avg_node_rank_effect[_position] = defaultdict(lambda: 0)
            self.global_avg_node_rank_effect[_position][rank_effect] += 1

        for indx in range(self.total_configs):
            frm_config = self.indx_to_config(indx)
            for position, value in enumerate(frm_config):
                config = self.possible_node_values[position][value]
                for a_pr_married_value in self._evaluate_perturbed_pr_married(
                    position, frm_config
                ):
                    if config.m is not a_pr_married_value:
                        perturb_node_val_indx = self.possible_node_values_mapping[
                            position
                        ][MaximalMatchingData(config.p, a_pr_married_value)]
                        perturb_state = tuple(
                            [
                                *frm_config[:position],
                                perturb_node_val_indx,
                                *frm_config[position + 1 :],
                            ]
                        )
                        _save_perturbation_implications(position, indx, perturb_state)
                    else:
                        if config.p is None:
                            for nbr in self.graph[position]:
                                perturb_node_val_indx = (
                                    self.possible_node_values_mapping[position][
                                        MaximalMatchingData(nbr, a_pr_married_value)
                                    ]
                                )
                                perturb_state = tuple(
                                    [
                                        *frm_config[:position],
                                        perturb_node_val_indx,
                                        *frm_config[position + 1 :],
                                    ]
                                )
                                _save_perturbation_implications(
                                    position, indx, perturb_state
                                )
                        else:
                            perturb_node_val_indx = self.possible_node_values_mapping[
                                position
                            ][MaximalMatchingData(None, a_pr_married_value)]
                            perturb_state = tuple(
                                [
                                    *frm_config[:position],
                                    perturb_node_val_indx,
                                    *frm_config[position + 1 :],
                                ]
                            )
                            _save_perturbation_implications(
                                position, indx, perturb_state
                            )
