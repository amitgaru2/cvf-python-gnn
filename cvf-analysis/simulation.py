import random

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

    def apply_fault_settings(self, fault_probability: float, fault_weight: List):
        self.fault_probability = fault_probability
        self.fault_weight = fault_weight

    def generate_fault_weight(self):
        fault_weight = 0.8

    def get_actions(self, state):
        eligible_actions = self.get_all_eligible_actions(state)  # from the base class
        if self.scheduler == CENTRAL_SCHEDULER:
            actions = self.get_one_random_action(eligible_actions)
        else:
            actions = self.get_subset_of_actions(eligible_actions)
            if self.me:
                actions = self.remove_conflicts(actions)  # from the base class


        return actions

    def get_one_random_action(self, actions: List[Action]):
        return random.sample(actions, 1)

    def get_subset_of_actions(self, actions: List[Action]):
        count = len(actions)
        subset_size = random.randint(1, count)
        return random.sample(actions, subset_size)

    def run_simulations(self):
        state = self.get_random_state(avoid_invariant=True)  # from the base class
        step = 0
        while not self.is_invariant(state):  # from the base class
            logger.info("State %s", state)
            actions = self.get_actions(state)
            state = self.execute(state, actions)
            step += 1

        logger.info("State %s", state)
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
        for i in range(1, self.no_of_simulations + 1):
            logger.info("Running simulation round: %d", i)
            results.append(self.run_simulations())

        return results
