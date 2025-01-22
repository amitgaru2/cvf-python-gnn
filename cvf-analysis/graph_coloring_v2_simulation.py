import csv
import sys
import math
import time
import random

from graph_coloring_v2 import (
    GraphColoring,
    GlobalAvgRank,
    GlobalTimeTrackFunction,
    logger,
)


def get_next_cvf_state_distributed_schedular_wo_me(
    state, eligible_processes, n_subset_eligible_process
):
    pass


graph_names = [sys.argv[1]]


class GraphColoringForSimulation(GraphColoring):

    def generate_initial_random_state(self):
        self.initial_state = []
        for i in range(len(self.nodes)):
            self.initial_state.append(random.choice(list(self.possible_node_values[i])))

    def find_eligible_nodes(self, state):
        eligible_nodes = []
        for position, color in enumerate(state):
            # check if node already has different color among the neighbors => If yes => not eligible to do anything
            neighbor_colors = set(state[i] for i in self.graph[position])
            if color not in neighbor_colors:  # is different color
                # considering the case where if the node has different color than neighboring node, regardless minimum or not, then it is not eligible
                continue
            transition_color = self._find_min_possible_color(neighbor_colors)
            if color != transition_color:
                eligible_nodes.append(position)

        return eligible_nodes

    def get_pts_distributed_schedular_wo_me(self, state, n_subset_eligible_process=1):
        eligible_nodes = self.find_eligible_nodes(state)
        program_transitions = []

        if eligible_nodes:
            for node in eligible_nodes:
                program_transitions.extend(
                    self._get_program_transitions_for_node(state, node)
                )

        return program_transitions

    def _get_program_transitions_for_node(self, state, node):
        program_transitions = []
        color = state[node]
        neighbor_colors = set(state[i] for i in self.graph[node])
        # if color not in neighbor_colors:  # is different color
        #     continue
        transition_color = self._find_min_possible_color(neighbor_colors)
        if color != transition_color:
            perturb_state = tuple(
                [
                    *state[:node],
                    transition_color,
                    *state[node + 1 :],
                ]
            )
            # program_transitions.append(self.config_to_indx(perturb_state))
            program_transitions.append(perturb_state)
            # may be yield can save memory

        return program_transitions


def main():
    coloring = GraphColoringForSimulation()
    coloring.generate_initial_random_state()
    print(coloring.initial_state)
    print(coloring.is_invariant(coloring.initial_state))
    # print(coloring.find_eligible_nodes(coloring.initial_state))
    print(coloring.get_pts_distributed_schedular_wo_me(coloring.initial_state))
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
