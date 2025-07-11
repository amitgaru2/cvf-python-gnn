from dijkstra import DijkstraTokenRingCVFAnalysisV2


class DijkstraTokenRingCVFAnalysisV2EQ(DijkstraTokenRingCVFAnalysisV2):

    def rotate_bottom_top(self):
        """eq extension"""
        self.bottom = (self.bottom + 1) % len(self.nodes)
        self.top = (self.top + 1) % len(self.nodes)

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
