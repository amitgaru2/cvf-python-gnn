import csv
import sys
import math
import time
import random
import itertools

from typing import List

from simulation import SimulationMixin, Action, CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER
from graph_coloring_v2 import (
    GraphColoring,
    GlobalAvgRank,
    GlobalTimeTrackFunction,
    logger,
)


graph_names = [sys.argv[1]]


class GraphColoringSimulation(SimulationMixin, GraphColoring):

    def get_random_state(self):
        state = []
        for i in range(len(self.nodes)):
            state.append(random.choice(list(self.possible_node_values[i])))

        return tuple(state)

    # def find_eligible_nodes(self, state):
    #     eligible_nodes = []
    #     for position, color in enumerate(state):
    #         # check if node already has different color among the neighbors => If yes => not eligible to do anything
    #         neighbor_colors = set(state[i] for i in self.graph[position])
    #         if color not in neighbor_colors:  # is different color
    #             # considering the case where if the node has different color than neighboring node, regardless minimum or not, then it is not eligible
    #             continue
    #         transition_color = self._find_min_possible_color(neighbor_colors)
    #         if color != transition_color:
    #             eligible_nodes.append(position)

    #     return eligible_nodes

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
            neighbor_action_indexes = [
                i.process for i in remaining_actions if i.process in neighbors
            ]
            for i in neighbor_action_indexes:
                remaining_actions.pop(i)
            remaining_actions.pop(indx)
            checked_actions.append(action)

        return checked_actions

    # def get_pts_distributed_schedular_wo_me(
    #     self, state, n_subset_eligible_process=None
    # ):
    #     """
    #     n_subset_eligible_process = None => take all eligible nodes
    #     """
    #     eligible_nodes = self.find_eligible_nodes(state)
    #     program_transitions = []

    #     if eligible_nodes:
    #         if n_subset_eligible_process is None:
    #             n_subset_eligible_process = len(eligible_nodes)

    #         for eligible_node_cobmination in itertools.combinations(
    #             eligible_nodes, n_subset_eligible_process
    #         ):
    #             program_transitions.append(
    #                 self._get_distributed_program_transitions_for_nodes(
    #                     state, set(eligible_node_cobmination)
    #                 )
    #             )

    #     return program_transitions

    # def _get_distributed_program_transitions_for_nodes(self, state, nodes):
    #     program_transition = []
    #     for position, color in enumerate(state):
    #         transition_color = color
    #         if position in nodes:
    #             neighbor_colors = set(state[i] for i in self.graph[position])
    #             if color in neighbor_colors:  # is different color
    #                 transition_color = self._find_min_possible_color(neighbor_colors)

    #         program_transition.append(transition_color)

    #     return tuple(program_transition)

    # def _get_program_transitions_for_node(self, state, node):
    #     program_transitions = []
    #     color = state[node]
    #     neighbor_colors = set(state[i] for i in self.graph[node])
    #     # if color not in neighbor_colors:  # is different color
    #     #     continue
    #     transition_color = self._find_min_possible_color(neighbor_colors)
    #     if color != transition_color:
    #         perturb_state = tuple(
    #             [
    #                 *state[:node],
    #                 transition_color,
    #                 *state[node + 1 :],
    #             ]
    #         )
    #         # program_transitions.append(self.config_to_indx(perturb_state))
    #         program_transitions.append(perturb_state)
    #         # may be yield can save memory

    #     return program_transitions


def main():
    coloring = GraphColoringSimulation()
    coloring.create_simulation_environment(
        no_of_simulations=50, scheduler=CENTRAL_SCHEDULER, me=False
    )
    coloring.start_simulation()
    # coloring.generate_initial_random_state()
    # print(coloring.initial_state)
    # print(coloring.is_invariant(coloring.initial_state))
    # # print(coloring.find_eligible_nodes(coloring.initial_state))
    # print(
    #     coloring.get_pts_distributed_schedular_wo_me(
    #         coloring.initial_state, n_subset_eligible_process=None
    #     )
    # )
    # print(
    #     coloring._get_distributed_program_transitions_for_nodes(
    #         coloring.initial_state, nodes={0, 2}
    #     )
    # )
    # coloring.start()
    # logger.info("%s", GlobalAvgRank)
    # time_tracking = {k: round(v, 2) for k, v in GlobalTimeTrackFunction.items()}
    # logger.info("%s", time_tracking)

    # writer = csv.DictWriter(
    #     open(f"{graph_names[0]}_config_rank_dataset.csv", "w"),
    #     fieldnames=["config", "rank"],
    # )
    # writer.writeheader()
    # for k, v in enumerate(coloring.global_rank_map):
    #     writer.writerow(
    #         {
    #             "config": [i for i in coloring.indx_to_config(k)],
    #             "rank": math.ceil(v[0] / v[1]),
    #         }
    #     )


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info("Total time taken: %s seconds.", round(time.time() - start_time, 4))
