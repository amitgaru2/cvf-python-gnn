from base import ProgramData, CVFAnalysisV2


class GraphColoringCVFAnalysisV2(CVFAnalysisV2):
    results_dir = "coloring"

    def get_possible_node_values(self):
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

    def start(self):
        super().start()
        # self.save_node_pt()

    def _get_program_transitions_as_configs(self, start_state):
        for position, color in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            neighbor_colors = set(start_state[i] for i in self.graph[position])
            if color not in neighbor_colors:  # is different color
                continue
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

                yield perturb_state

    def _get_program_transitions(self, start_state: tuple):
        program_transitions = []
        for perturb_state in self._get_program_transitions_as_configs(self):
            # indx = self.config_to_indx(start_state)
            # for position, color in enumerate(start_state):
            # # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            # neighbor_colors = set(start_state[i] for i in self.graph[position])
            # if color not in neighbor_colors:  # is different color
            #     continue
            # transition_color = self._find_min_possible_color(neighbor_colors)
            # if color != transition_color:
            #     self.global_pt[position] += 1
            #     perturb_state = tuple(
            #         [
            #             *start_state[:position],
            #             transition_color,
            #             *start_state[position + 1 :],
            #         ]
            #     )

            program_transitions.append(self.config_to_indx(perturb_state))
            # may be yield can save memory

        return program_transitions
