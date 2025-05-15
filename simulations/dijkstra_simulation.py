from simulation import SimulationMixin

from dijkstra import DijkstraTokenRingCVFAnalysisV2


class DijkstraSimulation(SimulationMixin, DijkstraTokenRingCVFAnalysisV2):
    results_dir = "dijkstra"
