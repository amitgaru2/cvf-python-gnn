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


from simulation import SimulationMixin, Action, CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER
from graph_coloring_v2 import (
    GraphColoring,
    GlobalAvgRank,
    GlobalTimeTrackFunction,
    logger,
    start,
    graphs_dir,
)


graph_names = [sys.argv[1]]


class GraphColoringSimulation(SimulationMixin, GraphColoring):

    def __init__(self) -> None:
        self.graph = start(graphs_dir, graph_names[0])
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

    def get_random_state(self, avoid_invariant=False):
        def _inner():
            _state = []
            for i in range(len(self.nodes)):
                _state.append(random.choice(list(self.possible_node_values[i])))
            _state = tuple(_state)

            return _state

        state = _inner()
        if avoid_invariant:
            while self.is_invariant(state):
                state = _inner()

        return state

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

    def remove_conflicts(self, actions: List[Action]) -> List[Action]:
        checked_actions = []
        remaining_actions = actions[:]
        while remaining_actions:
            indx = random.randint(0, len(remaining_actions) - 1)
            action = remaining_actions[indx]
            # remove the conflicting actions from "action" i.e. remove all the actions that are neighbors to the process producing "action"
            neighbors = self.graph[action.process]
            remaining_actions.pop(indx)

            new_remaining_actions = []
            for i, act in enumerate(remaining_actions):
                if act.process not in neighbors:
                    new_remaining_actions.append(act)

            remaining_actions = new_remaining_actions[:]
            checked_actions.append(action)

        return checked_actions


def main():
    logger.info("Graph %s", graph_names[0])
    coloring = GraphColoringSimulation()
    coloring.create_simulation_environment(
        no_of_simulations=1000, scheduler=DISTRIBUTED_SCHEDULER, me=True
    )
    coloring.apply_fault_settings(fault_probability=0.75)
    results = coloring.start_simulation()
    results = np.array(results)
    results = results.sum(axis=0)
    logger.info("Results %s", results)
    # print(coloring.generate_fault_weight(3))


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info("Total time taken: %s seconds.", round(time.time() - start_time, 4))
