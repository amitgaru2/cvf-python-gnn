import os
import copy

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


class DijkstraTokenRingFullAnalysis(CVFAnalysis):
    results_prefix = "dijkstra_token_ring"
    results_dir = os.path.join("results", results_prefix)

    @staticmethod
    def gen_implicit_graph(no_nodes: int):
        if no_nodes < 3 or no_nodes > 26:
            raise Exception("no_nodes must be >= 3.")

        graph = {}
        for i in range(no_nodes):
            left = (i - 1) % no_nodes
            right = (i + 1) % no_nodes
            graph[i] = [left, right]
        return graph

    def _gen_configurations(self):
        self.configurations = {tuple([0 for i in range(len(self.nodes))])}
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for node_pos in self.nodes:
            config_copy = copy.deepcopy(self.configurations)
            for i in range(1, self.degree_of_nodes[node_pos] + 1):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _check_if_only_one_eligible_process(self, state):
        """check invariant"""
        bottom = 0
        top = len(state) - 1
        eligible_nodes = 0
        for i, node_state in enumerate(state):
            if i == bottom:
                if (node_state + 1) % 3 == state[i + 1]:
                    eligible_nodes += 1

            elif i == top:
                if state[i - 1] == state[0] and (state[i - 1] + 1) % 3 != node_state:
                    eligible_nodes += 1

            else:
                if (node_state + 1) % 3 == state[i - 1]:
                    eligible_nodes += 1

                if (node_state + 1) % 3 == state[i + 1]:
                    eligible_nodes += 1

        return eligible_nodes == 1

    def _find_invariants(self):
        for state in self.configurations:
            if self._check_if_only_one_eligible_process(state):
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        if start_state in self.invariants and dest_state in self.invariants:
            return False

        s_prev = start_state[perturb_pos]
        s_new = dest_state[perturb_pos]

        node = self.nodes[perturb_pos]

        neighbor_pos = self.graph[node]
        neighbor_states = [start_state[i] for i in neighbor_pos]
        left_state, right_state = neighbor_states

        if node == self.nodes[0]:  # bottom
            return (s_prev + 1) % 3 == right_state and s_new == (s_prev - 1) % 3

        elif node == self.nodes[-1]:  # top
            return (
                left_state == right_state
                and (left_state + 1) % 3 != s_prev
                and s_new == (left_state + 1) % 3
            )

        else:  # others
            if (s_prev + 1) % 3 == left_state:
                return s_new == left_state
            elif (s_prev + 1) % 3 == right_state:
                return s_new == right_state

        return False

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, _ in enumerate(start_state):
            possible_node_colors = set(
                range(self.degree_of_nodes[self.nodes[position]] + 1)
            ) - {start_state[position]}
            for perturb_val in possible_node_colors:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

        return program_transitions

    def _get_cvfs(self, start_state):
        cvfs = dict()
        for position, _ in enumerate(start_state):
            possible_node_colors = set(
                range(self.degree_of_nodes[self.nodes[position]] + 1)
            ) - {start_state[position]}
            for perturb_val in possible_node_colors:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                cvfs[perturb_state] = position

        return cvfs


class DijkstraTokenRingPartialAnalysis(
    PartialCVFAnalysisMixin, DijkstraTokenRingFullAnalysis
):
    pass
