import random

from simulation import SimulationMixin, Action

from maximal_independent_set import MaximalIndependentSetCVFAnalysisV2


class MaximalIndependentSetSimulation(SimulationMixin, MaximalIndependentSetCVFAnalysisV2):

    def get_all_eligible_actions(self, state):
        eligible_actions = []

        for position, _ in enumerate(state):
            possible_config_val = set(
                range(len(self.possible_node_values[position]))
            ) - {state[position]}
            for perturb_val in possible_config_val:
                perturb_node_val_indx = perturb_val
                perturb_state = tuple(
                    [
                        *state[:position],
                        perturb_node_val_indx,
                        *state[position + 1 :],
                    ]
                )
                if self._is_program_transition(position, state, perturb_state):
                    eligible_actions.append(
                        Action(
                            Action.UPDATE,
                            position,
                            [state[position], perturb_state[position]],
                        )
                    )

        return eligible_actions

    def inject_fault_at_node(self, state, process):
        possible_actions = []

        for indx in range(self.total_configs):
            frm_config = self.indx_to_config(indx)
            for position, value in enumerate(frm_config):
                if value == self.OUT:
                    perturb_node_val_indx = self.IN
                    perturb_state = tuple(
                        [
                            *frm_config[:position],
                            perturb_node_val_indx,
                            *frm_config[position + 1 :],
                        ]
                    )
                    possible_actions.append(
                        Action(
                            Action.UPDATE, process, [state[process], perturb_state[process]]
                        )
                    )

                else:
                    for nbr in self.graph[position]:
                        if self.degree_of_nodes[nbr] <= self.degree_of_nodes[position]:
                            perturb_node_val_indx = self.OUT
                            perturb_state = tuple(
                                [
                                    *frm_config[:position],
                                    perturb_node_val_indx,
                                    *frm_config[position + 1 :],
                                ]
                            )
                            possible_actions.append(
                                Action(
                                    Action.UPDATE, process, [state[process], perturb_state[process]]
                                )
                            )
                            break

        return [random.choice(possible_actions)]
