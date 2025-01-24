import time
import random

import numpy as np

from typing import List

from custom_logger import logger

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
        highest_fault_weight = np.float32(0.8)
        other_fault_weight = np.float32(
            (1 - highest_fault_weight) / (len(self.nodes) - 1)
        )  # from the base class
        fault_weight = np.array([other_fault_weight for _ in range(len(self.nodes))])
        fault_weight[process] = highest_fault_weight
        fault_weight /= fault_weight.sum()
        self.fault_weight = fault_weight

    def get_actions(self, state):
        eligible_actions = self.get_all_eligible_actions(state)  # from the base class
        if self.scheduler == CENTRAL_SCHEDULER:
            actions = self.get_one_random_action(eligible_actions)
        else:
            actions = self.get_subset_of_actions(eligible_actions)
            if self.me:
                actions = self.remove_conflicts(actions)  # from the base class

        return actions

    def inject_fault(self, state, process):
        fault_count = 1
        state_copy = list(state)
        if self.scheduler == DISTRIBUTED_SCHEDULER:
            fault_count = random.randint(1, len(self.nodes))  # from the base class
        random_number = np.random.uniform()
        if random_number <= self.fault_probability:
            # logger.info("Fault occurred at %s", process)
            # inject fault
            randomly_selected_processes = list(
                np.random.choice(
                    a=self.nodes, p=self.fault_weight, size=fault_count, replace=False
                )
            )
            if self.me:
                randomly_selected_processes = self.remove_conflicts_betn_processes(
                    randomly_selected_processes
                )

            for p in randomly_selected_processes:
                state_copy[p] = random.choice(list(self.possible_node_values[p]))

        return tuple(state_copy)

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
        self.configure_fault_weight(process)
        step = 0
        while not self.is_invariant(state):  # from the base class
            # logger.info("State %s", state)
            faulty_state = self.inject_fault(state, process)  # might be faulty or not
            if faulty_state != state:
                pass
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
            for process in range(len(self.nodes)):  # from the base class
                inner_results.append(self.run_simulations(state, process))

            results.append(inner_results)

        return results
