from base import ProgramData, CVFAnalysisV2


class GraphColoringCVFAnalysisV2(CVFAnalysisV2):
    results_dir = "graph_coloring"

    def get_possible_node_values(self):
        """mapping is same to the values in the nodes that is v in value is the index in the mapping."""
        result = list()
        for node in self.nodes:
            possible_values = [
                ProgramData(i) for i in range(self.degree_of_nodes[node] + 1)
            ]
            result.append(tuple(possible_values))

        return result, []

    @staticmethod
    def _find_min_possible_color(colors):
        for i in range(len(colors) + 1):
            if i not in colors:
                return i

    def is_invariant(self, config):
        for node, color in enumerate(config):
            for dest_node in self.graph[node]:
                if config[dest_node] == color:
                    return False
        return True

    def _get_next_value_given_nbrs(self, node, node_value, neighbors_w_values):
        """
        designed for simulation v2.
        The next color value is independent of the current node's value.
        don't select the minimum color if it is already different from the neighbors
        """
        if node_value not in set(neighbors_w_values.values()):
            return None  # already different
        next_color = self._find_min_possible_color(neighbors_w_values.values())
        return next_color if node_value != next_color else None

    def _get_program_transitions_as_configs(self, start_state):
        for position, color in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            neighbor_colors = set(start_state[i] for i in self.graph[position])
            if color in neighbor_colors:  # is different color
                transition_color = self._find_min_possible_color(neighbor_colors)
                if color != transition_color:
                    self.global_pt[position] += 1
                    perturb_state = tuple(
                        [
                            *start_state[:position],
                            transition_color,
                            *start_state[position + 1 :],
                        ]
                    )
                    yield position, perturb_state
