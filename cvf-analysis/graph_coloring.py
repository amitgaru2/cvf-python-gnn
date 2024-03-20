import os
import copy

from itertools import combinations

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


class GraphColoringFullAnalysis(CVFAnalysis):
    results_prefix = "coloring"
    results_dir = os.path.join("results", results_prefix)

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

    def _find_invariants(self):
        for state in self.configurations:
            all_paths = combinations(range(len(state)), 2)
            for src, dest in all_paths:
                src_node, dest_node = self.nodes[src], self.nodes[dest]
                src_color, dest_color = state[src], state[dest]
                if dest_node in self.graph[src_node] and src_color == dest_color:
                    # found same color node between neighbors
                    break
            else:
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))

    def _find_min_possible_color(self, colors):
        for i in range(len(colors) + 1):
            if i not in colors:
                return i

    def _is_different_color(self, color, other_colors):
        """
        return True if "color" is different from all "other_colors"
        """
        for c in other_colors:
            if color == c:
                return False
        return True

    def _is_program_transition(self, perturb_pos, start_state, dest_state):
        if start_state in self.invariants and dest_state in self.invariants:
            return False

        neighbor_pos = [*self.graph[perturb_pos]]
        neighbor_colors = set(dest_state[i] for i in neighbor_pos)
        min_color = self._find_min_possible_color(neighbor_colors)

        return dest_state[perturb_pos] == min_color

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, val in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            neighbor_pos = [*self.graph[position]]
            neighbor_colors = set(start_state[i] for i in neighbor_pos)
            if self._is_different_color(val, neighbor_colors):
                continue

            # if the current node's color is not different among the neighbors => search for the program transitions possible
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
                cvfs[perturb_state] = (
                    position  # track the nodes to calculate its overall rank effect
                )

        return cvfs


class GraphColoringPartialAnalysis(PartialCVFAnalysisMixin, GraphColoringFullAnalysis):
    def get_cvfs(self, start_state):
        cvfs = {}
        for position, _ in enumerate(start_state):
            possible_node_colors = set(range(self.degree_of_nodes[position] + 1)) - {
                start_state[position]
            }
            for perturb_val in possible_node_colors:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                cvfs[perturb_state] = position

        return cvfs
