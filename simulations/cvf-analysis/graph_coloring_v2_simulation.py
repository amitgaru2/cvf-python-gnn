import csv
import sys
import math
import time
import random
import itertools
import numpy as np

from functools import reduce

from typing import List
from pprint import pprint

from custom_logger import logger
from simulation import SimulationMixin, Action, CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER
from graph_coloring_v2 import (
    GraphColoring,
    GlobalAvgRank,
    GlobalTimeTrackFunction,
)


class GraphColoringSimulation(SimulationMixin, GraphColoring):

    def __init__(self, graph_name, graph) -> None:
        self.graph_name = graph_name
        self.graph = graph

        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}

        self.possible_node_values = [
            set(range(self.degree_of_nodes[node] + 1)) for node in self.nodes
        ]
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


# def main():
#     logger.info("Graph %s", graph_names[0])
#     coloring = GraphColoringSimulation()
#     coloring.create_simulation_environment(
#         no_of_simulations=1000, scheduler=DISTRIBUTED_SCHEDULER, me=True
#     )
#     coloring.apply_fault_settings(fault_probability=0.75)
#     results = coloring.start_simulation()
#     results = np.array(results)
#     results = results.sum(axis=0)
#     logger.info("Results %s", results)
#     # print(coloring.generate_fault_weight(3))


# if __name__ == "__main__":
#     start_time = time.time()
#     main()
#     logger.info("Total time taken: %s seconds.", round(time.time() - start_time, 4))
