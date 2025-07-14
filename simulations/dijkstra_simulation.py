from simulation import SimulationMixin
from simulation_v2 import SimulationMixinV2

from dijkstra import DijkstraTokenRingCVFAnalysisV2


class DijkstraSimulation(SimulationMixin, DijkstraTokenRingCVFAnalysisV2):
    results_dir = "dijkstra_token_ring"


class DijkstraSimulationV2(SimulationMixinV2, DijkstraTokenRingCVFAnalysisV2):
    results_dir = "dijkstra_token_ring_v2"
