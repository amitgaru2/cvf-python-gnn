from simulation import SimulationMixin, Action

from dijkstra import DijkstraTokenRingCVFAnalysisV2


class DijkstraSimulation(SimulationMixin, DijkstraTokenRingCVFAnalysisV2):
    results_dir = "dijkstra"

    def get_all_eligible_actions(self, state):
        eligible_actions = []
        if (state[self.bottom] + 1) % 3 == state[self.bottom + 1]:
            transition_value = (state[self.bottom] - 1) % 3
            eligible_actions.append(
                Action(
                    Action.UPDATE, self.bottom, [state[self.bottom], transition_value]
                )
            )

        if (
            state[self.top - 1] == state[self.bottom]
            and (state[self.top - 1] + 1) % 3 != state[self.top]
        ):
            transition_value = (state[self.top - 1] + 1) % 3
            eligible_actions.append(
                Action(Action.UPDATE, self.top, [state[self.top], transition_value])
            )

        for i in range(self.bottom + 1, self.top):
            if (state[i] + 1) % 3 == state[i - 1]:
                transition_value = state[i - 1]
                eligible_actions.append(
                    Action(
                        Action.UPDATE,
                        i,
                        [state[i], transition_value],
                    )
                )

            if (state[i] + 1) % 3 == state[i + 1]:
                transition_value = state[i + 1]
                eligible_actions.append(
                    Action(
                        Action.UPDATE,
                        i,
                        [state[i], transition_value],
                    )
                )

        return eligible_actions
