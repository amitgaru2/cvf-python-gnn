import os
import csv
import sys
import time
import random

import numpy as np

from typing import List

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

from common_helpers import create_dir_if_not_exists

CENTRAL_SCHEDULER = 0
DISTRIBUTED_SCHEDULER = 1


class Action:
    UPDATE = 1

    def __init__(self, action_type, process, params):
        self.action_type = action_type
        self.process = process
        self.params = (
            params  # for update [current_value, new_value] or [from_value, to_value]
        )

    def execute(self, state):
        return {self.UPDATE: self.update}[self.action_type](state)

    def update(self, state):
        new_state = list(state)
        new_state[self.process] = self.params[1]  # update to new value
        return tuple(new_state)

    def __str__(self) -> str:
        return f"{self.process}: {self.params}"

    __repr__ = __str__


class SimulationMixin:
    highest_fault_weight = np.float32(0.8)
    least_fault_weight = np.float32(0.01)
    RANDOM_FAULT_SIMULATION_TYPE = "random"
    CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE = "controlled_at_node_amit_v1"
    CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_AMIT_V2 = "controlled_at_node_amit_v2"
    CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG = "controlled_at_node_duong"
    RANDOM_FAULT_START_AT_NODE_SIMULATION_TYPE = "random_start_at_node"

    SIMULATION_TYPES = [
        RANDOM_FAULT_SIMULATION_TYPE,
        CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE,
        CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG,
        RANDOM_FAULT_START_AT_NODE_SIMULATION_TYPE,
    ]

    def init_global_rank_map(self):
        """override this when not needed like for simulation"""
        self.global_rank_map = None

    def create_simulation_environment(
        self,
        simulation_type: str,
        no_of_simulations: int,
        scheduler: int,
        me: bool,
        limit_steps: int,
    ):
        self.no_of_simulations = no_of_simulations
        self.scheduler = scheduler
        self.me = me
        self.simulation_type = simulation_type
        self.limit_steps = limit_steps

    def apply_fault_settings(self, fault_probability: float, fault_interval: int):
        self.fault_probability = fault_probability
        self.fault_interval = fault_interval
        self.fault_weight = None

    def configure_fault_weight(self):
        other_fault_weight = (1.0 - self.highest_fault_weight) / (len(self.nodes) - 1)
        dsm = np.full((len(self.nodes), len(self.nodes)), other_fault_weight)
        np.fill_diagonal(dsm, self.highest_fault_weight)
        self.fault_weight = dsm

    def get_random_state(self, avoid_invariant=False):
        def _inner():
            _state = []
            for i in range(len(self.nodes)):  # from the base class
                _state.append(
                    random.choice(list(self.possible_node_values[i]))
                )  # from the base class
            _state = tuple(_state)

            return _state

        state = _inner()
        if avoid_invariant:
            while self.is_invariant(state):  # from the base class
                state = _inner()

        return state

    def get_random_state_v2(self, avoid_invariant=False):
        def _inner():
            _indx = random.randint(0, self.total_configs - 1)
            _state = self.indx_to_config(_indx)
            return _indx, _state

        indx, state = _inner()
        if avoid_invariant:
            while self.is_invariant(state):  # from the base class
                indx, state = _inner()

        return indx, state

    def get_all_eligible_actions(self, state):
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

    def get_actions(self, state):
        eligible_actions = self.get_all_eligible_actions(state)  # from the base class
        if self.scheduler == CENTRAL_SCHEDULER:
            actions = self.get_one_random_action(eligible_actions)
        else:
            actions = self.get_subset_of_actions(eligible_actions)
            if self.me:
                actions = self.remove_conflicts_betn_actions(actions)

        return actions

    def select_transitions_for_process(self, p, state, count):
        faulty_actions = []
        random_number = np.random.uniform()
        if random_number <= self.fault_probability:
            randomly_selected_processes = list(
                np.random.choice(
                    a=self.nodes,
                    p=p,
                    size=count,
                    replace=False,
                )
            )

            indx = self.config_to_indx(state)
            for p in randomly_selected_processes:
                possible_transition_indexes = [
                    i[1] for i in self.possible_perturbed_state_frm(indx) if i[0] == p
                ]
                if not possible_transition_indexes:
                    logger.warning(
                        "No any possible perturbation found for %s at state %s.",
                        p,
                        state,
                    )

                transition_indx = random.choice(possible_transition_indexes)
                transition_state = self.indx_to_config(transition_indx)
                faulty_actions.append(
                    Action(
                        Action.UPDATE,
                        p,
                        [state[p], transition_state[p]],
                    )
                )

        return faulty_actions

    def inject_fault_at_node(self, state, process):
        """Amit controlled version where given node has highest possibility of the fault."""
        fault_count = 1

        other_prob_wts = (1.0 - self.highest_fault_weight) / (len(self.nodes) - 1)
        p = [other_prob_wts for _ in range(len(self.nodes))]
        p[process] = self.highest_fault_weight
        p = np.array(p)
        p /= p.sum()

        faulty_actions = self.select_transitions_for_process(p, state, fault_count)

        return faulty_actions

    def inject_fault_at_node_v2(self, state, process):
        """Amit controlled version v2. Fault occurs at only targetted nodes."""
        fault_count = 1

        other_prob_wts = 0.0
        p = [other_prob_wts for _ in range(len(self.nodes))]
        p[process] = 1.0

        faulty_actions = self.select_transitions_for_process(p, state, fault_count)

        return faulty_actions

    def inject_least_fault_at_node(self, state, process):
        """Duong controlled version where given node has least possibility of the fault."""
        fault_count = 1

        other_prob_wts = (1.0 - self.least_fault_weight) / (len(self.nodes) - 1)
        p = [other_prob_wts for _ in range(len(self.nodes))]
        p[process] = self.least_fault_weight
        p = np.array(p)
        p /= p.sum()

        faulty_actions = self.select_transitions_for_process(p, state, fault_count)

        return faulty_actions

    def inject_fault_w_equal_prob(self, state):
        fault_count = 1
        faulty_actions = []

        random_number = np.random.uniform()
        if random_number <= self.fault_probability:
            randomly_selected_nodes = list(
                np.random.choice(
                    a=self.nodes,
                    size=fault_count,
                    replace=False,
                )
            )

            logger.debug("Selected random nodes %s.", randomly_selected_nodes)
            indx = self.config_to_indx(state)

            for p in randomly_selected_nodes:
                possible_transition_indexes = [
                    i[1] for i in self.possible_perturbed_state_frm(indx) if i[0] == p
                ]
                if not possible_transition_indexes:
                    logger.warning(
                        "No any possible perturbation found for %s at state %s.",
                        p,
                        state,
                    )
                    continue
                transition_indx = random.choice(possible_transition_indexes)
                transition_state = self.indx_to_config(transition_indx)
                faulty_actions.append(
                    Action(Action.UPDATE, p, [state[p], transition_state[p]])
                )

        return faulty_actions

    def remove_conflicts_betn_actions(self, actions: List[Action]) -> List[Action]:
        checked_actions = []
        remaining_actions = actions[:]
        while remaining_actions:
            indx = random.randint(0, len(remaining_actions) - 1)
            action = remaining_actions[indx]
            # remove the conflicting actions from "action" i.e. remove all the actions that are neighbors to the process producing "action"
            neighbors = self.graph[action.process]  # from the base class
            remaining_actions.pop(indx)

            new_remaining_actions = []
            for i, act in enumerate(remaining_actions):
                if act.process not in neighbors:
                    new_remaining_actions.append(act)

            remaining_actions = new_remaining_actions[:]
            checked_actions.append(action)

        return checked_actions

    def remove_conflicts_betn_processes(self, processes: List) -> List:
        checked_processes = []
        remaining_processes = processes[:]
        while remaining_processes:
            indx = random.randint(0, len(remaining_processes) - 1)
            process = remaining_processes[indx]
            neighbors = self.graph[process]  # from the base class
            remaining_processes.pop(indx)

            new_remaining_processes = []
            for p in remaining_processes:
                if p not in neighbors:
                    new_remaining_processes.append(p)

            remaining_processes = new_remaining_processes[:]
            checked_processes.append(process)

        return checked_processes

    def get_one_random_action(self, actions: List[Action]):
        return random.sample(actions, 1)

    def get_subset_of_actions(self, actions: List[Action]):
        count = len(actions)
        subset_size = random.randint(1, count)
        return random.sample(actions, subset_size)

    def get_steps_to_convergence(self, state):
        step = 0
        while not self.is_invariant(state):  # from the base class
            actions = self.get_actions(state)
            state = self.execute(state, actions)
            step += 1
        return step

    def get_faulty_actions_random(self, state, *others):
        faulty_actions = self.inject_fault_w_equal_prob(state)
        return faulty_actions

    def get_faulty_actions_random_start_at_node(self, state, process, step, *others):
        if step == 0:
            faulty_actions = self.inject_fault_at_node(state, process)
        else:
            faulty_actions = self.inject_fault_w_equal_prob(state)
        return faulty_actions

    def get_faulty_actions_controlled_at_node(self, state, process, *others):
        """
        process: process_id where the fault weight is concentrated
        """
        faulty_actions = self.inject_fault_at_node(state, process)
        return faulty_actions

    def get_faulty_actions_controlled_at_node_v2(self, state, process, *others):
        """
        process: process_id where the fault weight is concentrated
        """
        faulty_actions = self.inject_fault_at_node_v2(state, process)
        return faulty_actions

    def get_faulty_actions_controlled_at_node_duong(self, state, process, *others):
        """
        process: process_id where the fault weight is concentrated
        """
        faulty_actions = self.inject_least_fault_at_node(state, process)
        return faulty_actions

    def run_simulations(self, state, *extra_args):
        step = 0
        last_fault_duration = 0
        faulty_action_generator = {
            self.RANDOM_FAULT_SIMULATION_TYPE: self.get_faulty_actions_random,
            self.RANDOM_FAULT_START_AT_NODE_SIMULATION_TYPE: self.get_faulty_actions_random_start_at_node,
            self.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE: self.get_faulty_actions_controlled_at_node,
            self.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_AMIT_V2: self.get_faulty_actions_controlled_at_node_v2,
            self.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG: self.get_faulty_actions_controlled_at_node_duong,
        }[self.simulation_type]
        while not self.is_invariant(state):  # from the base class
            faulty_actions = []
            if last_fault_duration + 1 >= self.fault_interval:
                faulty_actions = faulty_action_generator(state, *extra_args, step)

            if faulty_actions:
                state = self.execute(state, faulty_actions)
                last_fault_duration = 0
            else:
                actions = self.get_actions(state)
                state = self.execute(state, actions)
                last_fault_duration += 1

            logger.debug("Next state: %s.", state)

            # last_fault_duration += 1
            step += 1
            if self.limit_steps and step >= self.limit_steps:
                # limit steps explicitly to stop the non-convergent chain or limit the steps for convergence
                logger.debug("Limit step reached!")
                return step, True

        return step, False

    def execute(self, state, actions: List[Action]):
        for action in actions:
            state = action.execute(state)

        return state

    def start_simulation(self, *simulation_type_args):
        logger.info(
            "Simulation environment: No. of Simulations: %d | Scheduler: %s | ME: %s",
            self.no_of_simulations,
            "DISTRIBUTED" if self.scheduler else "CENTRAL",
            self.me,
        )
        results = []
        log_time = time.time()
        for i in range(1, self.no_of_simulations + 1):
            if i % 50_000 == 0:
                logger.info(
                    "Time taken: %ss, Running simulation round: %d",
                    round(time.time() - log_time, 4),
                    i,
                )
                log_time = time.time()
            _, state = self.get_random_state_v2(avoid_invariant=True)
            # self.configure_fault_weight()
            inner_results = self.run_simulations(state, *simulation_type_args)
            results.append(inner_results)

        return results

    def store_raw_result(self, result, *simulation_type_args):
        simulation_type_args_verbose = "args_" + (
            "_".join(str(i) for i in simulation_type_args)
            if simulation_type_args
            else ""
        )
        limit_steps_verbose = f"limits_{self.limit_steps}" if self.limit_steps else ""
        save_dir = os.path.join("results", self.results_dir)
        create_dir_if_not_exists(save_dir)

        file_path = os.path.join(
            save_dir,
            f"{self.graph_name}__{self.scheduler}__{self.simulation_type}_{simulation_type_args_verbose}__{self.no_of_simulations}__{self.me}__{self.fault_interval}__{limit_steps_verbose}.csv",
        )
        f = open(
            file_path,
            "w",
            newline="",
        )  # from the base class
        logger.info("\nSaving result at %s", file_path)
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Steps", "Limit Reached"])
        for i, v in enumerate(result, 1):
            writer.writerow([i, *v])

    def aggregate_result(self, result):
        result = np.array(result)
        # _, bin_edges = np.histogram(result.flatten())
        # bin_edges = bin_edges.astype(int)
        bin_edges = [1, 2, 3, 4, 5, 10]
        bin_edges += [(i + 1) * 10 for i in range(1, np.max(result) // 10)]
        # bin_edges = [1, 5, 10, 15, 20]
        result = result.transpose()
        histogram = []
        for p in range(len(self.nodes)):
            hist, _ = np.histogram(result[p], bins=bin_edges)
            histogram.append(hist)
        return histogram, bin_edges

    def store_result(self, histogram, bin_edges):
        file_path = os.path.join(
            "results",
            self.results_dir,
            f"{self.graph_name}__{self.scheduler}__{self.no_of_simulations}__{self.me}__{self.fault_probability}__{self.highest_fault_weight:.2f}.csv",
        )
        logger.info("Saving result at %s", file_path)
        f = open(
            file_path,
            "w",
            newline="",
        )  # from the base class
        writer = csv.writer(f)
        writer.writerow(["Node", *bin_edges])
        for p, v in enumerate(histogram):
            writer.writerow([p, *v])  # from the base class
