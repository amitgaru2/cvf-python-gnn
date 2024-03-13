import copy
import logging

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

FullAnalysisType = "full"
PartialAnalysisType = "partial"


class Analysis:
    graphs_dir = "graphs"
    results_dir = "results"

    def __init__(self, graph) -> None:
        self.graph = graph
        self.nodes = list(graph.keys())
        self.node_positions = {v: i for i, v in enumerate(self.nodes)}
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}
        self.configurations = set()
        self.invariants = set()
        self.pts_rank = dict()

    def start(self):
        self._start()

    def _start(self):
        self._gen_configurations()
        self._find_invariants()

    def _gen_configurations(self):
        self.configurations = {tuple([0 for i in range(len(self.nodes))])}
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for n in self.nodes:
            node_pos = self.node_positions[n]
            config_copy = copy.deepcopy(self.configurations)
            for i in range(1, self.degree_of_nodes[n] + 1):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _find_invariants(self):
        raise NotImplemented

    def _init_pts_rank(self):
        for inv in self.invariants:
            self.pts_rank[inv] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}
