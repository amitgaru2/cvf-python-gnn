"""
Based on https://github.com/amitgaru2/research-journals/blob/main/2025/results/July.md
"""

import os
import sys
import time
import random

from collections import Queue

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
        """
        - num_vars = 1 for coloring, dijkstra
        - num_vars = 2 for max matching
        """
        self.num_vars = 1

        self.nodes_hist = []
        for _ in range(len(self.nodes)):
            self.nodes_hist.append(
                [NodeVarHistory() for _ in range(self.num_vars)]
            )  # queue for each variables in every nodes

    def init_stale_pointers(self):
        """points the neighboring history from where the variable is yet to be read"""
        for node in range(len(self.nodes)):
            for k in range(self.num_vars):
                self.nodes_sp[node][k] = [
                    0 if edge[1] == node else None for edge in range(len(self.edges))
                ]  # sp values initialized to 0 (first value) for all the edges where the node is a part of the edge (reading the value) otherwise set null

    def create_simulation_environment(self, no_of_simulations: int, limit_steps: int):
        """initialize configurations for the simulation"""
        self.no_of_simulations = no_of_simulations
        self.limit_steps = limit_steps

        self.init_edges()
        self.init_var_hist()
        self.init_stale_pointers()

    def log_var_history(self, node, var, value):
        self.nodes_hist[node][var].add_history(value)

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

    def run_simulations(self, state, **simulation_kwargs):
        """core simulation logic for a single round of simulationn"""
        step = 0
        last_fault_duration = 0

        return step

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
            inner_results = self.run_simulations(state, **simulation_kwargs)
            results.append([*inner_results])

        return results
