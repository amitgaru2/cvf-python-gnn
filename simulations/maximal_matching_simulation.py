from simulation_v2 import SimulationMixinV2
from simulation import SimulationMixin, Action

from maximal_matching import MaximalMatchingCVFAnalysisV2


class MaximalMatchingSimulation(SimulationMixin, MaximalMatchingCVFAnalysisV2):
    results_dir = "maximal_matching"

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


class MaximalMatchingSimulationV2(SimulationMixinV2, MaximalMatchingCVFAnalysisV2):
    results_dir = "maximal_matching_v2"


class MaximalMatchingSimulationSepVarV2(
    SimulationMixinV2, MaximalMatchingCVFAnalysisV2
):
    N_VARS = 2
    results_dir = "maximal_matching_v2_sep_var"
