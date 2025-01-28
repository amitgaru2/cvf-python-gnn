import csv
import os
import sys
import time
import random

import numpy as np

from typing import List

from custom_logger import logger

sys.path.append(
    os.path.join(os.getenv("CVF_PROJECT_DIR", "/home"), "cvf-analysis", "v2")
)

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
    results_dir = ""
    highest_fault_weight = np.float32(0.8)

    def init_global_rank_map(self):
        """override this when not needed like for simulation"""
        self.global_rank_map = None

    def create_simulation_environment(
        self, no_of_simulations: int, scheduler: int, me: bool
    ):
        self.no_of_simulations = no_of_simulations
        self.scheduler = scheduler
        self.me = me

    def apply_fault_settings(self, fault_probability: float):
        self.fault_probability = fault_probability
        self.fault_weight = None

    def configure_fault_weight(self, process):
        sm = np.full((len(self.nodes), len(self.nodes) + 1), np.nan)
        np.fill_diagonal(sm, self.highest_fault_weight)
        sm[:, -1] = np.float32(1.0 - self.highest_fault_weight)

        for i in self.nodes:
            temp = self.nodes[:]
            temp.pop(i)
            random.shuffle(temp)
            for j in temp:
                value = np.random.uniform(0, min(self.highest_fault_weight, sm[i, -1]))
                sm[i, j] = value
                sm[i, -1] -= value

            sm[i, j] += sm[i, -1]
            sm[i, -1] = 0

        self.fault_weight = sm[:, :-1]

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

    def get_actions(self, state):
        eligible_actions = self.get_all_eligible_actions(state)  # from the base class
        if self.scheduler == CENTRAL_SCHEDULER:
            actions = self.get_one_random_action(eligible_actions)
        else:
            actions = self.get_subset_of_actions(eligible_actions)
            if self.me:
                actions = self.remove_conflicts_betn_actions(actions)

        return actions

    def inject_fault(self, state, process):
        fault_count = 1
        faulty_actions = []
        if self.scheduler == DISTRIBUTED_SCHEDULER:
            fault_count = random.randint(1, len(self.nodes))  # from the base class
        random_number = np.random.uniform()
        if random_number <= self.fault_probability:
            # logger.info("Fault occurred at %s", process)
            # inject fault
            randomly_selected_processes = list(
                np.random.choice(
                    a=self.nodes,
                    p=self.fault_weight[process],
                    size=fault_count,
                    replace=False,
                )
            )
            if self.me:
                randomly_selected_processes = self.remove_conflicts_betn_processes(
                    randomly_selected_processes
                )

            for p in randomly_selected_processes:
                transition_value = random.choice(
                    list(self.possible_node_values[p] - {state[p]})
                )
                faulty_actions.append(
                    Action(Action.UPDATE, p, [state[p], transition_value])
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

    def run_simulations(self, state, process):
        """
        process: process_id where the fault weight is concentrated
        """
        # self.configure_fault_weight(process)
        step = 0
        while not self.is_invariant(state):  # from the base class
            # logger.info("State %s", state)
            faulty_actions = self.inject_fault(state, process)  # might be faulty or not
            if faulty_actions:
                state = self.execute(state, faulty_actions)
            else:
                actions = self.get_actions(state)
                state = self.execute(state, actions)
            step += 1

        # logger.info("State %s", state)
        return step

    def execute(self, state, actions: List[Action]):
        for action in actions:
            state = action.execute(state)

        return state

    def start_simulation(self):
        logger.info(
            "Simulation environment: No. of Simulations: %d | Scheduler: %s | ME: %s",
            self.no_of_simulations,
            "DISTRIBUTED" if self.scheduler else "CENTRAL",
            self.me,
        )
        results = []
        log_time = time.time()
        for i in range(1, self.no_of_simulations + 1):
            if i % 1000 == 0:
                logger.info(
                    "Time taken: %ss, Running simulation round: %d",
                    round(time.time() - log_time, 4),
                    i,
                )
                log_time = time.time()
            inner_results = []
            state = self.get_random_state(avoid_invariant=True)  # from the base class
            self.configure_fault_weight(0)
            # print(self.fault_weight)
            for process in range(len(self.nodes)):  # from the base class
                inner_results.append(self.run_simulations(state, process))

            results.append(inner_results)

        return results

    def aggregate_result(self, result):
        # print("result", result)
        result = np.array(result)
        _, bin_edges = np.histogram(result.flatten(), bins=10)
        bin_edges = bin_edges.astype(int)
        result = result.transpose()
        histogram = []
        for p in range(len(self.nodes)):
            hist, _ = np.histogram(result[p], bins=bin_edges)
            histogram.append(hist)
        return histogram, bin_edges

    def store_result(self, histogram, bin_edges):
        # fieldnames = ["Node", "Aggregated Steps"]
        f = open(
            os.path.join(
                "results",
                self.results_dir,
                f"{self.graph_name}__{self.scheduler}__{self.no_of_simulations}__{self.me}__{self.fault_probability}__{self.highest_fault_weight:.2f}.csv",
            ),
            "w",
            newline="",
        )  # from the base class
        writer = csv.writer(f)
        writer.writerow(["Node", *bin_edges])
        for p, v in enumerate(histogram):
            writer.writerow([p, *v])  # from the base class
