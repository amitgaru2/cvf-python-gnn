import random

from typing import List


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


class SimulationMixin:

    def create_simulation_environment(self, no_of_simulations, scheduler, me):
        self.no_of_simulations = no_of_simulations
        self.scheduler = scheduler
        self.me = me

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
        state = self.get_random_state()  # from the base class
        step = 0
        while not self.is_invariant(state):  # from the base class
            actions = self.get_actions(self.scheduler, self.me, state)
            state = self.execute(state, actions)
            step += 1

        return step

    # def get_all_eligible_actions(self, actions: List[Action]):
    #     pass

    def execute(self, state, actions: List[Action]):
        for action in actions:
            state = action.execute(state)

        return state

    def start(self):
        results = []
        for i in range(self.no_of_simulations):
            results[i] = self.run_simulations()

        return results
