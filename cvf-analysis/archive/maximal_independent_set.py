import os
import copy
import string

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


class Configuration:
    def __init__(self, val=0):
        self._val = val

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

    def __repr__(self):
        return f"<val: {self.val}>"


class MaximalSetIndependenceFullAnalysis(CVFAnalysis):
    results_prefix = "maximal_independent_set"
    results_dir = os.path.join("results", results_prefix)
    OUT = 0
    IN = 1

    def possible_values_of_node(self, position):
        return {self.OUT, self.IN}  # 0: out, 1: in

    def _gen_configurations(self):
        self.configurations = {
            tuple([Configuration(val=0) for i in range(len(self.nodes))])
        }
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for node_pos in self.nodes:
            config_copy = copy.deepcopy(self.configurations)
            for val in self.possible_values_of_node(node_pos):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = Configuration(val=val)
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _I_lte_v_null(self, position, state):
        for nbr in self.graph[position]:
            if (
                self.degree_of_nodes[nbr] <= self.degree_of_nodes[position]
                and state[nbr].val == self.IN
            ):
                return False
        return True

    def _check_if_none_eligible_process(self, state):
        """check invariant"""
        for position, config in enumerate(state):
            if config.val == self.OUT and self._I_lte_v_null(position, state):
                return False
            if config.val == self.IN and not self._I_lte_v_null(position, state):
                return False

    def _check_invariant(self, state):
        for position, config in enumerate(state):
            if config.val == self.OUT and not any(
                state[nbr].val == self.IN for nbr in self.graph[position]
            ):
                return False
            if config.val == self.IN and not all(
                state[nbr].val == self.OUT for nbr in self.graph[position]
            ):
                return False

        return True

    def _find_invariants(self):
        for state in self.configurations:
            if self._check_invariant(state):
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        if start_state[perturb_pos].val == self.OUT and self._I_lte_v_null(
            perturb_pos, start_state
        ):
            return dest_state[perturb_pos].val == self.IN
        if start_state[perturb_pos].val == self.IN and not self._I_lte_v_null(
            perturb_pos, start_state
        ):
            return dest_state[perturb_pos].val == self.OUT
        return False

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, _ in enumerate(start_state):
            possible_config_val = self.possible_values_of_node(position) - {
                start_state[position].val
            }
            for perturb_val in possible_config_val:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].val = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

        return program_transitions

    def _get_cvfs(self, start_state):
        """
        1. If the perturbation is from OUT to IN then it is always C.V.F.
        2. If the perturbation is from IN to OUT then it is C.V.F only if it has degree >= any of its neighbor.
        """
        cvfs = dict()
        for position, _ in enumerate(start_state):
            if start_state[position].val == self.OUT:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].val = self.IN
                perturb_state = tuple(perturb_state)
                cvfs[perturb_state] = position
            else:
                for nbr in self.graph[position]:
                    if self.degree_of_nodes[nbr] <= self.degree_of_nodes[position]:
                        perturb_state = list(copy.deepcopy(start_state))
                        perturb_state[position].val = self.OUT
                        perturb_state = tuple(perturb_state)
                        cvfs[perturb_state] = position
                        break
        return cvfs


class MaximalSetIndependencePartialAnalysis(
    PartialCVFAnalysisMixin, MaximalSetIndependenceFullAnalysis
):
    pass
