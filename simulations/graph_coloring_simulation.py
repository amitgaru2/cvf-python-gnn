from simulation import SimulationMixin
from simulation_v2 import SimulationMixinV2

from graph_coloring import GraphColoringCVFAnalysisV2


class GraphColoringSimulation(SimulationMixin, GraphColoringCVFAnalysisV2):
    pass


class GraphColoringSimulationV2(SimulationMixinV2, GraphColoringCVFAnalysisV2):
    results_dir = "graph_coloring_v2"
