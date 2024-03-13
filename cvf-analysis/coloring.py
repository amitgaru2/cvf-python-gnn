from itertools import combinations

from analysis import Analysis, logger


class GraphColoringAnalysis(Analysis):
    results_dir = "results"

    def _find_invariants(self):
        for state in self.configurations:
            all_paths = combinations(range(len(state)), 2)
            for src, dest in all_paths:
                src_node, dest_node = self.nodes[src], self.nodes[dest]
                src_color, dest_color = state[src], state[dest]
                if dest_node in self.graph[src_node] and src_color == dest_color:
                    # found same color node between neighbors
                    break
            else:
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))
