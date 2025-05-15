import random

from simulation import SimulationMixin, Action

from maximal_matching import MaximalMatchingCVFAnalysisV2, MaximalMatchingData


class MaximalMatchingSimulation(SimulationMixin, MaximalMatchingCVFAnalysisV2):
    results_dir = "maximal_matching"

    # def get_all_eligible_actions(self, state):
    #     eligible_actions = []

    #     for position, node_val_indx in enumerate(state):
    #         current_p_value = self.possible_node_values[position][node_val_indx].p
    #         current_m_value = self.possible_node_values[position][node_val_indx].m

    #         possible_config_p_val = {
    #             i.p for i in self.possible_node_values[position]
    #         } - {current_p_value}

    #         for perturb_p_val in possible_config_p_val:
    #             perturb_node_val_indx = self.possible_node_values_mapping[position][
    #                 MaximalMatchingData(perturb_p_val, current_m_value)
    #             ]
    #             perturb_state = tuple(
    #                 [
    #                     *state[:position],
    #                     perturb_node_val_indx,
    #                     *state[position + 1 :],
    #                 ]
    #             )
    #             if self._is_program_transition(position, state, perturb_state):
    #                 eligible_actions.append(
    #                     Action(
    #                         Action.UPDATE,
    #                         position,
    #                         [state[position], perturb_state[position]],
    #                     )
    #                 )

    #         possible_config_m_val = {True, False} - {current_m_value}
    #         for perturb_m_val in possible_config_m_val:
    #             perturb_node_val_indx = self.possible_node_values_mapping[position][
    #                 MaximalMatchingData(current_p_value, perturb_m_val)
    #             ]
    #             perturb_state = tuple(
    #                 [
    #                     *state[:position],
    #                     perturb_node_val_indx,
    #                     *state[position + 1 :],
    #                 ]
    #             )
    #             if self._is_program_transition(position, state, perturb_state):
    #                 eligible_actions.append(
    #                     Action(
    #                         Action.UPDATE,
    #                         position,
    #                         [state[position], perturb_state[position]],
    #                     )
    #                 )

    #     return eligible_actions

    # def inject_fault_at_node(self, state, process):
    #     faulty_actions = []
    #     config = self.possible_node_values[process][state[process]]
    #     possible_actions = []
    #     for a_pr_married_value in self._evaluate_perturbed_pr_married(process, state):
    #         if config.m is not a_pr_married_value:
    #             perturb_node_val_indx = self.possible_node_values_mapping[process][
    #                 MaximalMatchingData(config.p, a_pr_married_value)
    #             ]
    #             perturb_state = tuple(
    #                 [
    #                     *state[:process],
    #                     perturb_node_val_indx,
    #                     *state[process + 1 :],
    #                 ]
    #             )

    #             possible_actions.append(
    #                 Action(
    #                     Action.UPDATE, process, [state[process], perturb_state[process]]
    #                 )
    #             )

    #         else:
    #             if config.p is None:
    #                 for nbr in self.graph[process]:
    #                     perturb_node_val_indx = self.possible_node_values_mapping[
    #                         process
    #                     ][MaximalMatchingData(nbr, a_pr_married_value)]
    #                     perturb_state = tuple(
    #                         [
    #                             *state[:process],
    #                             perturb_node_val_indx,
    #                             *state[process + 1 :],
    #                         ]
    #                     )

    #                     possible_actions.append(
    #                         Action(
    #                             Action.UPDATE,
    #                             process,
    #                             [state[process], perturb_state[process]],
    #                         )
    #                     )

    #             else:
    #                 perturb_node_val_indx = self.possible_node_values_mapping[process][
    #                     MaximalMatchingData(None, a_pr_married_value)
    #                 ]
    #                 perturb_state = tuple(
    #                     [
    #                         *state[:process],
    #                         perturb_node_val_indx,
    #                         *state[process + 1 :],
    #                     ]
    #                 )
    #                 possible_actions.append(
    #                     Action(
    #                         Action.UPDATE,
    #                         process,
    #                         [state[process], perturb_state[process]],
    #                     )
    #                 )

    #     faulty_actions.append(random.choice(possible_actions))
    #     return faulty_actions

    def get_all_eligible_actions(self, state):
        for position, pt in self._get_program_transitions_as_configs(state):
            pass
        pass
