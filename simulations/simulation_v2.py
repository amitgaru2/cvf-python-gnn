"""
Based on https://github.com/amitgaru2/research-journals/blob/main/2025/July.md
"""

import os
import csv
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
    )
)

utils_path = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home/agaru/research/cvf-python-gnn"), "utils"
)
sys.path.append(utils_path)

from common_helpers import create_dir_if_not_exists


class NodeVarHistory:

    def __init__(self, size):
        self.size = size
        self.hist = [float("inf") for _ in range(self.size)]
        self.cur_indx = (
            -1
        )  # counter ; if 10 elements inserted in the history then cur_indx = 9

    def add_history(self, value):
        self.hist.append(value)
        if len(self.hist) > self.size:
            self.hist.pop(
                0
            )  # maintain the size of history to self.size by removing oldest element
        self.cur_indx += (
            1  # keep track of the number of history of variable added to the node
        )

    def get_internal_indx(self, indx):
        if self.cur_indx - (self.size - 1) > indx:
            internal_indx = 0  # obsolette index trying to be accessed, so move it to the latest oldest in the history i.e. at index 0
        else:
            internal_indx = (self.size - 1) - (self.cur_indx - indx)

        return internal_indx

    def get_history_frm(self, indx):
        if indx > self.cur_indx:
            raise Exception(
                "Trying to access history index yet to be filled."
            )  # just to make sure we avoided this

        internal_indx = self.get_internal_indx(indx)
        start_indx = (self.cur_indx - (self.size - 1)) + internal_indx
        return zip(
            self.hist[internal_indx:],
            [start_indx + i for i in range(len(self.hist[internal_indx:]))],
        )

    def get_history_at(self, indx):
        internal_indx = self.get_internal_indx(indx)
        cur_indx = (self.cur_indx - (self.size - 1)) + internal_indx
        return self.hist[internal_indx], cur_indx

    def get_latest_value(self):
        return self.hist[self.size - 1]  # the latest value in the list

    def __str__(self):
        return f"{self.hist}"

    __repr__ = __str__


class ActionV2:
    def __init__(
        self,
        node: int,
        previous_value,
        new_value,
        changed_var: int,
        read_pointers: dict,
    ):
        self.node = node  # the node on which the change happened
        self.previous_value = previous_value  # indexed state value
        self.new_value = new_value  # indexed state value
        self.changed_var = changed_var  # which variable changed between the previous_value -> new_value
        self.read_pointers = read_pointers  # that points to the sp_pointers based on which new_value was calculated; {1: 2, 3: 2}

    def execute(self, var_history: NodeVarHistory, read_pointers: dict):
        var_history[self.changed_var].add_history(self.new_value)
        read_pointers.update(self.read_pointers)

    def __str__(self) -> str:
        return f"N={self.node} PV={self.previous_value} NV={self.new_value} RP={self.read_pointers}"

    __repr__ = __str__


class NodeReadValue:
    def __init__(self, node, reader_node, value, read_pointer):
        self.node = node
        self.reader_node = reader_node
        self.value = value
        self.read_pointer = (
            read_pointer  # the sp_pointer of the reader "reader_node" to node "node"
        )

    def __str__(self):
        return f"{self.value}"

    __repr__ = __str__


class SimulationMixinV2:
    N_VARS = 1  # 1 is default, override in the simulation class

    def init_edges(self):
        """edges are necessary since faults are now introduced in the edges rather than nodes"""
        self.edges = []  # size of 2M, M is the size of undirected edges in the graph
        for src, dests in self.graph.items():
            self.edges.extend(
                [(src, dest) for dest in dests]
            )  # (src, dest) src being read by dest

    def init_pt_count(self):
        self.pt_count = {i: 0 for i in self.nodes}

    def init_var_hist(self):
        self.nodes_hist: List[NodeVarHistory] = [
            [NodeVarHistory(self.hist_size) for _ in range(self.N_VARS)]
            for _ in self.nodes
        ]  # history for each nodes; each node can have multiple variables like max-matching has 2 varialbes `p` and `m` treated separately.

    def init_stale_pointers(self):
        """points the neighboring history from where the variable is yet to be read"""
        self.nodes_read_pointer = {}
        for node in range(len(self.nodes)):
            self.nodes_read_pointer[node] = {
                edge[0]: [0 for _ in range(self.N_VARS)]
                for edge in self.edges
                if edge[1] == node
            }

    def create_simulation_environment(
        self,
        no_of_simulations: int,
        faulty_edges: dict,
        fault_interval: int,
        limit_steps: int,
        hist_size: int,
    ):
        """initialize configurations for the simulation"""
        self.no_of_simulations = no_of_simulations
        self.fault_interval = fault_interval
        self.limit_steps = limit_steps
        self.faulty_edges = faulty_edges
        self.hist_size = hist_size

        self.init_edges()

    def log_var_history(self, node: int, var: int, value):
        """log variable history for individual node"""
        self.nodes_hist[node][var].add_history(value)

    def log_state_to_history(self, state):
        """log the entire state to the history of individual nodes"""
        for node, value in enumerate(state):
            for var, act_value in enumerate(
                self.possible_node_values[node][value].data
            ):
                self.log_var_history(node, var, act_value)

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

    def get_latest_value_of_node_var(self, node: int, var: int):
        return self.nodes_hist[node][var].get_latest_value()

    def get_read_pointer_of_node_at_var(
        self,
        node: int,
        at: int,
        var: int,
    ):
        return self.nodes_read_pointer[node][at][var]

    def get_faulty_action(self):
        """
        faults are introducted in faulty_edges only.
        one of the random fault given the faulty_edges is selected
        """
        node_w_value_n_nbr_values = {}
        for edge in self.faulty_edges:
            temp = [[] for _ in range(self.N_VARS)]
            read_frm_node, reader_node = edge[0], edge[1]
            sp_pointer_of_reader_node = self.nodes_read_pointer[reader_node][
                read_frm_node
            ]
            for var in range(self.N_VARS):
                for v, read_pointer in self.nodes_hist[read_frm_node][
                    var
                ].get_history_frm(sp_pointer_of_reader_node[var]):
                    temp[var].append(
                        NodeReadValue(read_frm_node, reader_node, v, read_pointer)
                    )

            if any(temp):
                if reader_node not in node_w_value_n_nbr_values:
                    node_w_value_n_nbr_values[reader_node] = [temp]
                else:
                    node_w_value_n_nbr_values[reader_node].append(temp)

        # logger.debug("node_w_value_n_nbr_values %s", node_w_value_n_nbr_values)

        # find all eligible updates
        eligible_actions_for_fault = []
        for node, nbr_read_values in node_w_value_n_nbr_values.items():
            nbr_var_vals_combs = []
            for nbr_var_vals in nbr_read_values:
                # first combine internal variables' values then combine the combination with other nodes' combinations;
                # print("nbr_var_vals", nbr_var_vals)
                nbr_var_vals_combs.append(list(product(*nbr_var_vals)))
            nbr_combinations = product(*nbr_var_vals_combs)
            # print("nbr_combinations", list(nbr_combinations))
            for nbr_comb in nbr_combinations:
                neighbors_w_values = {
                    read_data[0].node: self.get_mapped_value_of_data(
                        read_data[0].node,
                        [read_data[var].value for var in range(self.N_VARS)],
                    )
                    for read_data in nbr_comb
                }
                read_pointers = {
                    read_data[0].node: [
                        read_data[var].read_pointer for var in range(self.N_VARS)
                    ]
                    for read_data in nbr_comb
                }
                # for those neighbors that are not the part of faulty edge
                # print("neighbors_w_values", neighbors_w_values)

                for nbr in self.graph[node]:
                    if nbr not in neighbors_w_values:
                        data = []
                        read_pointer = []
                        for var in range(self.N_VARS):
                            temp = self.nodes_hist[nbr][var].get_history_at(
                                self.get_read_pointer_of_node_at_var(node, nbr, var)
                            )
                            data.append(temp[0])
                            read_pointer.append(temp[1])
                        neighbors_w_values[nbr] = self.get_mapped_value_of_data(
                            nbr, data
                        )
                        read_pointers[nbr] = read_pointer

                # print("neighbors_w_values", neighbors_w_values)
                # print("read_pointers", read_pointers)

                current_value = self.get_mapped_value_of_data(
                    node,
                    [
                        self.get_latest_value_of_node_var(node, var)
                        for var in range(self.N_VARS)
                    ],
                )
                next_val, changed_var = self._get_next_value_given_nbrs(
                    node, current_value, neighbors_w_values
                )  # from base cvf class
                if next_val is not None:
                    eligible_actions_for_fault.append(
                        ActionV2(
                            node,
                            previous_value=self.get_latest_value_of_node_var(
                                node, changed_var
                            ),
                            new_value=self.get_actual_config_node_values(
                                node, next_val
                            ).data[changed_var],
                            changed_var=changed_var,
                            read_pointers=read_pointers,
                        )
                    )

        if not eligible_actions_for_fault:
            logger.debug(
                "No eligible action found for fault given faulty edges %s.",
                self.faulty_edges,
            )
            action = None
        else:
            action = self.get_one_random_value(eligible_actions_for_fault)
        return action

    def get_most_latest_state(self):
        """the values of the variables that are latest for all the nodes"""
        return [
            self.get_mapped_value_of_data(
                i, [self.get_latest_value_of_node_var(i, j) for j in range(self.N_VARS)]
            )
            for i in range(len(self.nodes))
        ]  # since node hist contains the actual values that needs to be combined as single state value at particular node index and the state value should be mapped back from the actual values of combination of variables

    def get_latest_reader_pointer_of_node_var(self, node, var):
        return self.nodes_hist[node][var].cur_indx

    def get_all_eligible_actions(self, state):
        """get the program transitions from state."""
        eligible_actions = []
        for node in self.nodes:
            neighbors_w_values = {}
            read_pointers = {}
            for nbr in self.graph[node]:
                neighbors_w_values[nbr] = state[nbr]
                read_pointers[nbr] = []
                for j in range(self.N_VARS):
                    read_pointers[nbr].append(
                        self.get_latest_reader_pointer_of_node_var(nbr, j)
                    )

            next_value, changed_var = self._get_next_value_given_nbrs(
                node, state[node], neighbors_w_values
            )  # from base cvf class

            if next_value is not None:
                eligible_actions.append(
                    ActionV2(
                        node,
                        previous_value=self.get_actual_config_node_values(
                            node, state[node]
                        ).data[changed_var],
                        new_value=self.get_actual_config_node_values(
                            node, next_value
                        ).data[changed_var],
                        changed_var=changed_var,
                        read_pointers=read_pointers,
                    )
                )

        return eligible_actions

    def get_one_random_value(self, values: List):
        return random.choice(values)

    def get_action(self, state):
        """
        select a random eligible action and the process for which that action occur will update its' sp pointer to all latest values
        """
        eligible_actions = self.get_all_eligible_actions(state)
        if not eligible_actions:
            logger.debug(
                "No eligible action for %s.\n",
                self.get_actual_config_values(state),
            )
            action = None
        else:
            action = self.get_one_random_value(eligible_actions)
        return action

    def log_pt_count(self, action):
        """
        log the program transition and aggregate it for current simulation round.
        """
        self.pt_count[action.node] += 1

    # def run_simulations(self):
    #     state = self.get_most_latest_state()
    #     action = self.get_action(state)
    #     print("state", state)
    #     print("action", action)

    #     return 0, True

    def run_simulations(self):
        """core simulation logic for a single round of simulation"""
        last_fault_duration = 0
        FAULT_NEXT_STEP = "f"
        next_step = None  # None means go it normal flow, do not interrupt anything
        # print()
        for step in range(1, self.limit_steps + 1):
            logger.debug("\nStep %s.", step)
            faulty_action = None
            if (
                next_step == FAULT_NEXT_STEP
                or last_fault_duration + 1 >= random.randint(*self.fault_interval)
            ):
                # fault introduction
                faulty_action = self.get_faulty_action()

                if faulty_action is None:
                    # Termination condition; when there is no any fault that can occur
                    logger.debug(
                        "Since no eligible action for the faults found. Terminating at step %s.",
                        step,
                    )
                    return step, False

                faulty_action.execute(
                    self.nodes_hist[faulty_action.node],
                    self.nodes_read_pointer[faulty_action.node],
                )
                logger.debug("Fault happened at %s.", faulty_action.node)
                logger.debug(
                    "New history at %s: %s",
                    faulty_action.node,
                    self.nodes_hist[faulty_action.node],
                )
                logger.debug(
                    "New pointers at %s: %s",
                    faulty_action.node,
                    self.nodes_read_pointer[faulty_action.node],
                )
                last_fault_duration = 0
                next_step = None
            else:
                # program transition
                # if program transition not found check in next round if fault can occur, do not terminate if no program transition found
                state = self.get_most_latest_state()
                action = self.get_action(state)
                if action is not None:
                    # there is possible program transition
                    action.execute(
                        self.nodes_hist[action.node],
                        self.nodes_read_pointer[action.node],
                    )
                    self.log_pt_count(action)
                    logger.debug("Prog transition happened at %s.", action.node)
                    logger.debug(
                        "New_history at %s: %s.",
                        action.node,
                        self.nodes_hist[action.node],
                    )
                    logger.debug(
                        "New_pointers at %s: %s",
                        action.node,
                        self.nodes_read_pointer[action.node],
                    )
                else:
                    # force next step to be a fault since all the preceeding steps will have no program transitions unless fault is executed.
                    next_step = FAULT_NEXT_STEP
                last_fault_duration += 1

        return step, True

    def prepare_simulation_round(self):
        self.init_pt_count()
        self.init_var_hist()
        self.init_stale_pointers()

    def start_simulation(self):
        """entrypoint of the simulation"""
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
            self.prepare_simulation_round()
            _, state = self.get_random_state(avoid_invariant=True)
            logger.debug("Selected initial state is %s.", state)
            self.log_state_to_history(state)
            inner_results = self.run_simulations()
            results.append([*inner_results, *self.pt_count.values()])

        return results

    def store_raw_result(self, result):
        faulty_edges_verb = (
            f"{"_".join(["-".join([str(i) for i in fe]) for fe in self.faulty_edges])}"
        )
        lim_steps_verb = f"{self.limit_steps}" if self.limit_steps else ""
        save_dir = os.path.join("results", self.results_dir)
        create_dir_if_not_exists(save_dir)

        file_path = os.path.join(
            save_dir,
            f"{self.graph_name}__FE_{faulty_edges_verb}__N{self.no_of_simulations}__FI_{"-".join([str(i) for i in self.fault_interval])}__H{self.hist_size}__L{lim_steps_verb}.csv",
        )
        f = open(
            file_path,
            "w",
            newline="",
        )  # from the base class
        logger.info("\nSaving result at %s", file_path)
        writer = csv.writer(f)
        headers = ["SN", "Steps", "Lmt Reach"]
        headers.extend([f"PT {i}" for i in self.nodes])
        writer.writerow(headers)
        for i, v in enumerate(result, 1):
            writer.writerow([i, *v])

    def aggregate_result(self, result, agg_file=None):
        step_sum = sum([v[0] for v in result])
        if agg_file is None:
            logger.info(f"Total steps taken {step_sum:,}.")
        else:
            f = open(agg_file, "a+")
            csv.writer(f).writerow([step_sum, *self.faulty_edges])
            f.close()
