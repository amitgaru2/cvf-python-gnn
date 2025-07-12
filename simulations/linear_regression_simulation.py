from simulation import SimulationMixin

from linear_regression import LinearRegressionCVFAnalysisV2


class LinearRegressionSimulation(SimulationMixin, LinearRegressionCVFAnalysisV2):
    results_dir = "linear_regression"
