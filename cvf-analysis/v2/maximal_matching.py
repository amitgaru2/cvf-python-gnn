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
        for position in self.nodes:
            possible_values = list()
            for neighbor in [None, *self.graph[position]]:
                for m in (True, False):
                    possible_values.append(MaximalMatchingData(neighbor, m))

            result.append(tuple(possible_values))

        return result

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
                            self.possible_node_values[i][state[k]].p is None
                            and k < j
                            and not self.possible_node_values[i][state[k]].m
                        ):
                            return False
                else:
                    i = config.p
                    if self.possible_node_values[i][state[i]].p != j and (
                        self.possible_node_values[i][state[i]].m or j <= i
                    ):
                        return False

        return True

    def _get_program_transitions(self, start_state):
        program_transitions = []
        return program_transitions
