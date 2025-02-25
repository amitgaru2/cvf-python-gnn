import math

from collections import defaultdict

from base import ProgramData, CVFAnalysisV2


class MaximalIndependentSetCVFAnalysisV2(CVFAnalysisV2):
    """Not complete"""

    results_dir = "maximal_independent_set"
    IN = 1
    OUT = 0

    def get_possible_node_values(self):
        result = list()
        for _ in self.nodes:
            possible_values = [ProgramData(i) for i in [self.OUT, self.IN]]
            result.append(tuple(possible_values))

        return result, []

    def _I_lte_v_null(self, position, state):
        for nbr in self.graph[position]:
            if (
                self.degree_of_nodes[nbr] <= self.degree_of_nodes[position]
                and state[nbr] == self.IN
            ):
                return False
        return True

    def _check_if_none_eligible_process(self, state):
        """check invariant"""
        for position, config in enumerate(state):
            if config == self.OUT and self._I_lte_v_null(position, state):
                return False
            if config == self.IN and not self._I_lte_v_null(position, state):
                return False

    def is_invariant(self, state):
        for position, config in enumerate(state):
            if config == self.OUT and not any(
                state[nbr] == self.IN for nbr in self.graph[position]
            ):
                return False
            if config == self.IN and not all(
                state[nbr] == self.OUT for nbr in self.graph[position]
            ):
                return False

        return True

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        if start_state[perturb_pos] == self.OUT and self._I_lte_v_null(
            perturb_pos, start_state
        ):
            return dest_state[perturb_pos] == self.IN
        if start_state[perturb_pos] == self.IN and not self._I_lte_v_null(
            perturb_pos, start_state
        ):
            return dest_state[perturb_pos] == self.OUT
        return False

    def _get_program_transitions(self, start_state):
        program_transitions = []
        for position, _ in enumerate(start_state):
            possible_config_val = set(
                range(len(self.possible_node_values[position]))
            ) - {start_state[position]}
            for perturb_val in possible_config_val:
                perturb_node_val_indx = perturb_val
                perturb_state = tuple(
                    [
                        *start_state[:position],
                        perturb_node_val_indx,
                        *start_state[position + 1 :],
                    ]
                )
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.append(self.config_to_indx(perturb_state))

        return program_transitions

    # def _get_cvfs(self, start_state):
    #     """
    #     1. If the perturbation is from OUT to IN then it is always C.V.F.
    #     2. If the perturbation is from IN to OUT then it is C.V.F only if it has degree >= any of its neighbor.
    #     """
    #     cvfs = dict()
    #     for position, _ in enumerate(start_state):
    #         if start_state[position].val == self.OUT:
    #             perturb_state = list(copy.deepcopy(start_state))
    #             perturb_state[position].val = self.IN
    #             perturb_state = tuple(perturb_state)
    #             cvfs[perturb_state] = position
    #         else:
    #             for nbr in self.graph[position]:
    #                 if self.degree_of_nodes[nbr] <= self.degree_of_nodes[position]:
    #                     perturb_state = list(copy.deepcopy(start_state))
    #                     perturb_state[position].val = self.OUT
    #                     perturb_state = tuple(perturb_state)
    #                     cvfs[perturb_state] = position
    #                     break
    #     return cvfs

    def find_rank_effect(self):
        """
        1. If the perturbation is from OUT to IN then it is always C.V.F.
        2. If the perturbation is from IN to OUT then it is C.V.F only if it has degree >= any of its neighbor.
        """

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
                if value == self.OUT:
                    perturb_node_val_indx = self.IN
                    perturb_state = tuple(
                        [
                            *frm_config[:position],
                            perturb_node_val_indx,
                            *frm_config[position + 1 :],
                        ]
                    )
                    _save_perturbation_implications(position, indx, perturb_state)
                else:
                    for nbr in self.graph[position]:
                        if self.degree_of_nodes[nbr] <= self.degree_of_nodes[position]:
                            perturb_node_val_indx = self.OUT
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
                            break
