from simulation import SimulationMixin, Action

from maximal_matching import MaximalMatchingCVFAnalysisV2, MaximalMatchingData


class MaximalMatchingSimulation(SimulationMixin, MaximalMatchingCVFAnalysisV2):
    results_dir = "maximal_matching"

    def get_all_eligible_actions(self, state):
        eligible_actions = []

        for position, node_val_indx in enumerate(state):
            current_p_value = self.possible_node_values[position][node_val_indx].p
            current_m_value = self.possible_node_values[position][node_val_indx].m

            possible_config_p_val = {
                i.p for i in self.possible_node_values[position]
            } - {current_p_value}

            for perturb_p_val in possible_config_p_val:
                perturb_node_val_indx = self.possible_node_values_mapping[position][
                    MaximalMatchingData(perturb_p_val, current_m_value)
                ]
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

            possible_config_m_val = {True, False} - {current_m_value}
            for perturb_m_val in possible_config_m_val:
                perturb_node_val_indx = self.possible_node_values_mapping[position][
                    MaximalMatchingData(current_p_value, perturb_m_val)
                ]
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
