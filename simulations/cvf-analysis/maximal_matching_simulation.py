from functools import reduce

from custom_logger import logger
from dijkstra_v2 import DijkstraTokenRing
from simulation import SimulationMixin, Action


class DijkstraSimulation(SimulationMixin, DijkstraTokenRing):

    def __init__(self, graph_name, graph) -> None:
        self.graph = graph
        self.graph_name = graph_name
        self.nodes = list(self.graph.keys())

        self.possible_node_values = [{0, 1, 2} for _ in self.nodes]
        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.total_configs = reduce(
            lambda x, y: x * y, self.possible_node_values_length
        )
        logger.info(f"Total configs: {self.total_configs:,}.")

        # rank map
        self.global_rank_map = None
        self.analysed_rank_count = 0

        self.possible_values = list(
            set([j for i in self.possible_node_values for j in i])
        )
        self.possible_values.sort()
        self.possible_values_indx_str = {
            v: str(i) for i, v in enumerate(self.possible_values)
        }  # mapping from value to index

        self.initialize_helpers()
        self.initialize_problem_helpers()

    def get_all_eligible_actions(self, state):
        eligible_actions = []
        if (state[self.bottom] + 1) % 3 == state[self.bottom + 1]:
            transition_value = (state[self.bottom] - 1) % 3
            eligible_actions.append(
                Action(
                    Action.UPDATE, self.bottom, [state[self.bottom], transition_value]
                )
            )

        if (
            state[self.top - 1] == state[self.bottom]
            and (state[self.top - 1] + 1) % 3 != state[self.top]
        ):
            transition_value = (state[self.top - 1] + 1) % 3
            eligible_actions.append(
                Action(Action.UPDATE, self.top, [state[self.top], transition_value])
            )

        for i in range(self.bottom + 1, self.top):
            if (state[i] + 1) % 3 == state[i - 1]:
                transition_value = state[i - 1]
                eligible_actions.append(
                    Action(
                        Action.UPDATE,
                        i,
                        [state[i], transition_value],
                    )
                )

            if (state[i] + 1) % 3 == state[i + 1]:
                transition_value = state[i + 1]
                eligible_actions.append(
                    Action(
                        Action.UPDATE,
                        i,
                        [state[i], transition_value],
                    )
                )

        # for position, color in enumerate(state):
        #     # check if node already has different color among the neighbors => If yes => not eligible to do anything
        #     neighbor_colors = set(state[i] for i in self.graph[position])
        #     if color not in neighbor_colors:  # is different color
        #         # considering the case where if the node has different color than neighboring node, regardless minimum or not, then it is not eligible
        #         continue
        #     transition_color = self._find_min_possible_color(neighbor_colors)
        #     if color != transition_color:
        #         eligible_actions.append(
        #             Action(Action.UPDATE, position, [color, transition_color])
        #         )

        return eligible_actions
