from base import CVFAnalysisV2


class MaximalMatchingCVFAnalysisV2(CVFAnalysisV2):
    """Not complete"""

    results_dir = "maximal_matching"

    def get_possible_node_values(self):
        """include m values as well"""
        result = []
        for position in self.nodes:
            result.append(
                set([-1, *self.graph[position]])
            )  # either None, or one of its neighbors
        return result

    def is_invariant(self, state):
        """check invariant"""

        def _pr_married(j, config):
            for i in self.graph[j]:
                if state[i].p == j and config.p == i:
                    return True
            return False

        for j, config in enumerate(state):
            # update m.j
            if config.m != _pr_married(j, config):
                return False

            # accept a proposal
            if config.m == _pr_married(j, config):
                if config.p is None:
                    for i in self.graph[j]:
                        if state[i].p == j:
                            return False

                    for k in self.graph[j]:
                        if state[k].p is None and k < j and not state[k].m:
                            return False
                else:
                    i = config.p
                    if state[i].p != j and (state[i].m or j <= i):
                        return False

        return True

    def _get_program_transitions(self, start_state):
        program_transitions = []
        return program_transitions
