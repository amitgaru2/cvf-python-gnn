from base import CVFAnalysisV2, ProgramData


class DijkstraTokenRingCVFAnalysisV2(CVFAnalysisV2):
    results_dir = "dijkstra_token_ring"

    # def get_possible_node_values(self):
    #     return [{0, 1, 2} for _ in self.nodes]

    def get_possible_node_values(self):
        result = list()
        for _ in self.nodes:
            possible_values = [ProgramData(i) for i in [0, 1, 2]]
            result.append(tuple(possible_values))

        return result, []

    def initialize_program_helpers(self):
        self.bottom = 0
        self.top = len(self.nodes) - 1

    # program specific methods
    def __bottom_eligible_update(self, state):
        _state = list(state[:])
        _state[self.bottom] = (state[self.bottom] - 1) % 3
        return tuple(_state)

    def __top_eligible_update(self, state):
        _state = list(state[:])
        _state[self.top] = (state[self.top - 1] + 1) % 3
        return tuple(_state)

    def __other_eligible_update(self, state, idx, L_or_R_idx):
        _state = list(state[:])
        _state[idx] = state[L_or_R_idx]
        return tuple(_state)

    def _get_program_transitions_as_configs(self, start_state):
        yielded = set()
        if (start_state[self.bottom] + 1) % 3 == start_state[self.bottom + 1]:
            pt_state = self.__bottom_eligible_update(start_state)
            yield self.bottom, pt_state
            yielded.add(pt_state)

        if (
            start_state[self.top - 1] == start_state[self.bottom]
            and (start_state[self.top - 1] + 1) % 3 != start_state[self.top]
        ):
            pt_state = self.__top_eligible_update(start_state)
            if pt_state not in yielded:
                yield self.top, pt_state
                yielded.add(pt_state)

        for i in range(self.bottom + 1, self.top):
            if (start_state[i] + 1) % 3 == start_state[i - 1]:
                pt_state = self.__other_eligible_update(start_state, i, i - 1)
                if pt_state not in yielded:
                    yield i, pt_state
                    yielded.add(pt_state)

            if (start_state[i] + 1) % 3 == start_state[i + 1]:
                pt_state = self.__other_eligible_update(start_state, i, i + 1)
                if pt_state not in yielded:
                    yield i, pt_state
                    yielded.add(pt_state)

    def is_invariant(self, config):
        eligible_rules = 0

        if (config[self.bottom] + 1) % 3 == config[self.bottom + 1]:
            eligible_rules += 1

        if (
            config[self.top - 1] == config[self.bottom]
            and (config[self.top - 1] + 1) % 3 != config[self.top]
        ):
            eligible_rules += 1

        for i in range(self.bottom + 1, self.top):
            if eligible_rules > 1:
                return False

            if (config[i] + 1) % 3 == config[i - 1]:
                eligible_rules += 1

            if (config[i] + 1) % 3 == config[i + 1]:
                eligible_rules += 1

        return eligible_rules == 1
