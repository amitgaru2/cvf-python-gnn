from base import CVFAnalysisV2


class GraphColoringCVFAnalysisV2(CVFAnalysisV2):
    def get_possible_node_values(self):
        return [set(range(self.degree_of_nodes[node] + 1)) for node in self.nodes]

    def _find_min_possible_color(self, colors):
        for i in range(len(colors) + 1):
            if i not in colors:
                return i

    def is_invariant(self, config):
        for node, color in enumerate(config):
            for dest_node in self.graph[node]:
                if config[dest_node] == color:
                    return False
        return True

    def _get_program_transitions(self, start_state):
        program_transitions = []
        for position, color in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            neighbor_colors = set(start_state[i] for i in self.graph[position])
            if color not in neighbor_colors:  # is different color
                continue
            transition_color = self._find_min_possible_color(neighbor_colors)
            if color != transition_color:
                perturb_state = tuple(
                    [
                        *start_state[:position],
                        transition_color,
                        *start_state[position + 1 :],
                    ]
                )
                program_transitions.append(self.config_to_indx(perturb_state))
                # may be yield can save memory

        return program_transitions
