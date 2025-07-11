from simulation import SimulationMixin

from dijkstra_eq import DijkstraTokenRingCVFAnalysisV2EQ


class DijkstraEqSimulation(SimulationMixin, DijkstraTokenRingCVFAnalysisV2EQ):
    results_dir = "dijkstra_token_ring_eq"
