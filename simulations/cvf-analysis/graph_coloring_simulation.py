from simulation import SimulationMixin, Action

from graph_coloring import GraphColoringCVFAnalysisV2


class GraphColoringSimulation(SimulationMixin, GraphColoringCVFAnalysisV2):
    results_dir = "coloring"

    def get_all_eligible_actions(self, state):
        eligible_actions = []
        for position, color in enumerate(state):
            # check if node already has different color among the neighbors => If yes => not eligible to do anything
            neighbor_colors = set(state[i] for i in self.graph[position])
            if color not in neighbor_colors:  # is different color
                # considering the case where if the node has different color than neighboring node, regardless minimum or not, then it is not eligible
                continue
            transition_color = self._find_min_possible_color(neighbor_colors)
            if color != transition_color:
                eligible_actions.append(
                    Action(Action.UPDATE, position, [color, transition_color])
                )

        return eligible_actions
