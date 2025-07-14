"""
Based on https://github.com/amitgaru2/research-journals/blob/main/2025/results/July.md
"""

import os
import sys
import time
import random
from typing import List

from simulation import Action
from custom_logger import logger

sys.path.append(
    os.path.join(
        os.getenv("CVF_PROJECT_DIR", "/home/agaru/research/cvf-python-gnn"),
        "cvf-analysis",
        "v2",
    )
)

utils_path = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home/agaru/research/cvf-python-gnn"), "utils"
)
sys.path.append(utils_path)


class NodeVarHistory:

    def __init__(self, size=5):
        self.size = size
        self.hist = [None for _ in range(self.size)]
        self.cur_indx = -1

    def add_history(self, value):
        self.hist.append(value)
        if len(self.hist) > self.size:
            self.hist.pop(
                0
            )  # maintain the size of history to self.size by removing oldest element
        self.cur_indx += (
            1  # keep track of the number of history of variable added to the node
        )

    def get_history_from(self, indx):
        if indx > self.cur_indx:
            raise Exception(
                "Trying to access history index yet to be filled."
            )  # just to make sure we avoided this
        if self.cur_indx - indx > self.size:
            internal_indx = 0  # obsolette index trying to be accessed, so move it to the latest oldest in the history i.e. at index 0
        else:
            internal_indx = (self.size - 1) - (self.cur_indx - indx)
        return (
            self.hist[internal_indx:],
            (self.cur_indx + 1 - self.size) + internal_indx,
        )

    def get_latest_value(self):
        if self.cur_indx >= self.size:
            return self.hist[self.size - 1]  # the latest value in the list
        return self.hist[self.cur_indx]


class SimulationMixinV2:

    def init_edges(self):
        """edges are necessary since faults are now introduced in the edges rather than nodes"""
        self.edges = []  # size of 2M, M is the size of undirected edges in the graph
        for src, dest in self.graph:
            self.edges.append((src, dest))  # (src, dest) src being read by dest

        self.edges_indx = {
            e: i for i, e in enumerate(self.edges)
        }  # reverse lookup to get the index of edges {(1, 2): 0, (2, 1): 2, ...}

    def init_var_hist(self):
        self.nodes_hist = [
            NodeVarHistory() for _ in self.nodes
        ]  # history for each nodes; assuming each node has single variable.

    def init_stale_pointers(self):
        """points the neighboring history from where the variable is yet to be read"""
        for node in range(len(self.nodes)):
            self.nodes_sp[node] = [
                0 if edge[1] == node else None for edge in range(len(self.edges))
            ]  # sp values initialized to 0 (first value) for all the edges where the node is a part of the edge (reading the value) otherwise set null

    def create_simulation_environment(
        self, no_of_simulations: int, fault_interval: int, limit_steps: int
    ):
        """initialize configurations for the simulation"""
        self.no_of_simulations = no_of_simulations
        self.fault_interval = fault_interval
        self.limit_steps = limit_steps

        self.init_edges()
        self.init_var_hist()
        self.init_stale_pointers()

    def log_var_history(self, node, value):
        """log variable history for individual node"""
        self.nodes_hist[node].add_history(value)

    def log_state_to_history(self, state):
        """log the entire state to the history of individual nodes"""
        for node, value in enumerate(state):
            self.log_var_history(node, value)

    def get_random_state(self, avoid_invariant=False):
        """
        get a random initial state to start the simulation with.
            - avoid_invariant: True will return the random state that is not an Invariant.
            - avoid_invariant: False will return the random state that may be an Invariant.
        """

        def _inner():
            _indx = random.randint(0, self.total_configs - 1)
            _state = self.indx_to_config(_indx)
            return _indx, _state

        indx, state = _inner()
        if avoid_invariant:
            while self.is_invariant(state):  # from the base class
                indx, state = _inner()

        return indx, state

    def get_faulty_action(self, faulty_edges):
        """
        faults are introducted in faulty_edges only.
        one of the random fault given the faulty_edges is selected
        """
        eligible_actions_for_fault = []
        temp_state = []
        for edge in faulty_edges:
            sp_pointer_of_reader_node = self.nodes_sp[edge[1]]
            for v in self.nodes_hist[edge[0]].get_history_from(sp_pointer_of_reader_node):
                pass

        if not eligible_actions_for_fault:
            logger.warning(
                "No eligible action found for fault given faulty edges %s", faulty_edges
            )
            action = None
        else:
            action = self.get_one_random_value(eligible_actions_for_fault)
        return action

    def get_most_latest_state(self):
        """the values of the variables are latest for all the nodes"""
        return [nh.get_latest_value() for nh in self.nodes_hist]

    def get_all_eligible_actions(self, state):
        """get the program transitions from state."""
        eligible_actions = []
        for position, program_transition in self._get_program_transitions_as_configs(
            state
        ):
            eligible_actions.append(
                Action(
                    Action.UPDATE,
                    position,
                    [state[position], program_transition[position]],
                )
            )
        return eligible_actions

    def get_one_random_value(self, values: List):
        return random.sample(values, 1)

    def get_action(self, state):
        """
        select a random eligible action and the process for which that action occur will update its' sp pointer to all latest values
        """
        eligible_actions = self.get_all_eligible_actions()
        if not eligible_actions:
            logger.warning(
                "No eligible action for %s : %s",
                state,
                self.get_actual_config_values(state),
            )
            action = None
        else:
            action = self.get_one_random_value(eligible_actions)
            # the selected action of the node as a program transition will
            # imply that the node has sp pointer pointed to all the latest
            # values of its neighbors
            for nbr in self.graph[action.process]:
                self.nodes_sp[action.process][nbr] = self.nodes_hist[
                    nbr
                ].cur_indx  # set to latest pointer
        return action

    def execute(self, state, action):
        """execute the action in the state => update the value of a node (given by action) in the state"""
        return action.execute(state)

    def run_simulations(self, state, **simulation_kwargs):
        """core simulation logic for a single round of simulationn"""
        steps = 0
        last_fault_duration = 0
        faulty_edges = []

        while True:
            faulty_action = None
            if last_fault_duration + 1 >= self.fault_interval:
                # fault introduction
                faulty_action = self.get_faulty_action(faulty_edges)

            if faulty_action is not None:
                last_fault_duration = 0
            else:
                # program transition
                state = self.get_most_latest_state()
                action = self.get_action(state)
                self.execute(state, action)
                last_fault_duration += 1

            steps += 1
            if self.limit_steps and steps >= self.limit_steps:
                # limit steps explicitly to stop the non-convergent chain or limit the steps for convergence
                return steps, True

        return steps, False

    def start_simulation(self, **simulation_kwargs):
        """entrypoint of the simulation"""
        logger.info(
            "Simulation environment: No. of Simulations: %d",
            self.no_of_simulations,
        )
        results = []
        log_time = time.time()
        for i in range(1, self.no_of_simulations + 1):
            if i % 10_000 == 0:
                logger.info(
                    "Time taken: %ss, Running simulation round: %d",
                    round(time.time() - log_time, 4),
                    i,
                )
                log_time = time.time()
            _, state = self.get_random_state(avoid_invariant=True)
            self.log_state_to_history(state)
            inner_results = self.run_simulations(state, **simulation_kwargs)
            results.append([*inner_results])

        return results
