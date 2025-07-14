"""
Based on https://github.com/amitgaru2/research-journals/blob/main/2025/results/July.md
"""

import os
import sys
import time
import random
from typing import List
from itertools import product

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

    def get_history_frm(self, indx):
        if indx > self.cur_indx:
            raise Exception(
                "Trying to access history index yet to be filled."
            )  # just to make sure we avoided this

        if self.cur_indx - (self.size - 1) > indx:
            internal_indx = 0  # obsolette index trying to be accessed, so move it to the latest oldest in the history i.e. at index 0
        else:
            internal_indx = (self.size - 1) - (self.cur_indx - indx)

        start_indx = (self.cur_indx - (self.size - 1)) + internal_indx
        return zip(
            self.hist[internal_indx:],
            [start_indx + i for i in range(len(self.hist[internal_indx:]))],
        )

    def get_latest_value(self):
        return self.hist[self.size - 1]  # the latest value in the list

    def __str__(self):
        return f"{self.hist}"


class ActionV2:
    def __init__(self, node, previous_value, new_value, read_pointers):
        self.node = node
        self.previous_value = previous_value
        self.new_value = new_value
        self.read_pointers = read_pointers  # that points to the sp_pointers based on which new_value was calculated; {1: 2, 3: 2}

    def execute(self, var_history: NodeVarHistory, read_pointers: dict):
        var_history.add_history(self.new_value)
        read_pointers.update(self.read_pointers)

    def __str__(self) -> str:
        return f"{self.node}: {self.previous_value} - {self.new_value} ; {self.read_pointers}"

    __repr__ = __str__


class NodeReadValue:
    def __init__(self, node, reader_node, value, read_pointer):
        self.node = node
        self.reader_node = reader_node
        self.value = value
        self.read_pointer = (
            read_pointer  # the sp_pointer of the reader "reader_node" to node "node"
        )


class SimulationMixinV2:

    def init_edges(self):
        """edges are necessary since faults are now introduced in the edges rather than nodes"""
        self.edges = []  # size of 2M, M is the size of undirected edges in the graph
        for src, dests in self.graph.items():
            self.edges.extend(
                [(src, dest) for dest in dests]
            )  # (src, dest) src being read by dest

        # self.edges_indx = {
        #     e: i for i, e in enumerate(self.edges)
        # }  # reverse lookup to get the index of edges {(1, 2): 0, (2, 1): 2, ...}

    def init_var_hist(self):
        self.nodes_hist: List[NodeVarHistory] = [
            NodeVarHistory() for _ in self.nodes
        ]  # history for each nodes; assuming each node has single variable.

    def init_stale_pointers(self):
        """points the neighboring history from where the variable is yet to be read"""
        self.nodes_read_pointer = {}
        for node in range(len(self.nodes)):
            self.nodes_read_pointer[node] = {
                edge[0]: 0 for edge in self.edges if edge[1] == node
            }

        print("nodes_reader_pointer", self.nodes_read_pointer)

    def create_simulation_environment(
        self,
        no_of_simulations: int,
        faulty_edges: dict,
        fault_interval: int,
        limit_steps: int,
    ):
        """initialize configurations for the simulation"""
        self.no_of_simulations = no_of_simulations
        self.fault_interval = fault_interval
        self.limit_steps = limit_steps
        self.faulty_edges = faulty_edges

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

    def get_latest_value_of_node(self, node):
        return self.nodes_hist[node].get_latest_value()

    def get_faulty_action(self):
        """
        faults are introducted in faulty_edges only.
        one of the random fault given the faulty_edges is selected
        """
        node_w_value_n_nbr_values = {}
        for edge in self.faulty_edges:
            temp = []
            read_frm_node, reader_node = edge[0], edge[1]
            sp_pointer_of_reader_node = self.nodes_read_pointer[reader_node][
                read_frm_node
            ]
            for v, read_pointer in self.nodes_hist[read_frm_node].get_history_frm(
                sp_pointer_of_reader_node
            ):
                temp.append(NodeReadValue(read_frm_node, reader_node, v, read_pointer))

            if temp and reader_node not in node_w_value_n_nbr_values:
                node_w_value_n_nbr_values[reader_node] = [temp]
            else:
                node_w_value_n_nbr_values[reader_node].append(temp)

        # find all eligible updates
        eligible_actions_for_fault = []
        for node, nbr_read_values in node_w_value_n_nbr_values.items():
            nbr_combinations = product(*nbr_read_values)
            for nbr_comb in nbr_combinations:
                neighbors_w_values = {i.node: i.value for i in nbr_comb}
                next_val = self._get_next_value_given_nbrs(
                    node, self.get_latest_value_of_node(node), neighbors_w_values
                )  # from base cvf class
                if next_val is not None:
                    eligible_actions_for_fault.append(
                        ActionV2(
                            node,
                            previous_value=self.get_latest_value_of_node(node),
                            new_value=next_val,
                            read_pointers={i.node: i.read_pointer for i in nbr_comb},
                        )
                    )

        if not eligible_actions_for_fault:
            logger.warning(
                "No eligible action found for fault given faulty edges %s",
                self.faulty_edges,
            )
            action = None
        else:
            action = self.get_one_random_value(eligible_actions_for_fault)
        return action

    def get_most_latest_state(self):
        """the values of the variables are latest for all the nodes"""
        return [self.get_latest_value_of_node(i) for i in range(len(self.nodes))]

    def get_latest_reader_pointer_of_node(self, node):
        return self.nodes_hist[node].cur_indx

    def get_most_latest_state_reader_pointer(self, exclude_node=None):
        result = {i: self.get_latest_reader_pointer_of_node(i) for i in self.nodes}
        if exclude_node is not None:
            result.pop(exclude_node, None)
        return result

    def get_all_eligible_actions(self, state):
        """get the program transitions from state."""
        eligible_actions = []
        for node in self.nodes:
            neighbors_w_values = {}
            read_pointers = {}
            for nbr in self.graph[node]:
                neighbors_w_values[nbr] = state[nbr]
                read_pointers[nbr] = self.get_latest_reader_pointer_of_node(nbr)

            next_value = self._get_next_value_given_nbrs(
                node, state[node], neighbors_w_values
            )  # from base cvf class

            if next_value is not None:
                eligible_actions.append(
                    ActionV2(
                        node,
                        previous_value=state[node],
                        new_value=next_value,
                        read_pointers=read_pointers,
                    )
                )

        return eligible_actions

    def get_one_random_value(self, values: List):
        return random.sample(values, 1)[0]

    def get_action(self, state):
        """
        select a random eligible action and the process for which that action occur will update its' sp pointer to all latest values
        """
        eligible_actions = self.get_all_eligible_actions(state)
        if not eligible_actions:
            logger.warning(
                "No eligible action for %s : %s",
                state,
                self.get_actual_config_values(state),
            )
            action = None
        else:
            action = self.get_one_random_value(eligible_actions)
        return action

    def run_simulations(self, state):
        """core simulation logic for a single round of simulation"""
        last_fault_duration = 0
        print()
        for _ in range(self.limit_steps):
            faulty_action = None
            if last_fault_duration + 1 >= self.fault_interval:
                # fault introduction
                faulty_action = self.get_faulty_action()

            if faulty_action is not None:
                faulty_action.execute(
                    self.nodes_hist[faulty_action.node],
                    self.nodes_read_pointer[faulty_action.node],
                )
                print("fault happened at", faulty_action.node)
                print(
                    "new history at",
                    faulty_action.node,
                    self.nodes_hist[faulty_action.node],
                )
                print(
                    "new pointers at",
                    faulty_action.node,
                    self.nodes_read_pointer[faulty_action.node],
                )
                print("\n\n")
                last_fault_duration = 0
            else:
                # program transition
                state = self.get_most_latest_state()
                action = self.get_action(state)
                if action is not None:
                    action.execute(
                        self.nodes_hist[action.node],
                        self.nodes_read_pointer[action.node],
                    )
                    print("prog transition happened at", action.node)
                    print("new_history at", action.node, self.nodes_hist[action.node])
                    print(
                        "new_pointers at",
                        action.node,
                        self.nodes_read_pointer[action.node],
                    )
                    print("\n\n")
                    last_fault_duration += 1

        return True

    def start_simulation(self):
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
            logger.info("Selected initial state is %s", state)
            self.log_state_to_history(state)
            inner_results = self.run_simulations(state)
            results.append(inner_results)

        return results
