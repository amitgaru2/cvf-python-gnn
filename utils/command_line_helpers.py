import os
import logging


ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"

PROGRAM_CHOICES = [
    ColoringProgram,
    DijkstraProgram,
    MaxMatchingProgram,
    MaxIndependentSetProgram,
    LinearRegressionProgram,
]

GRAPHS_DIR = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs")


def get_graph(graph_names, logger=None):
    if logger is None:
        logger = logging.getLogger()

    for graph_name in graph_names:
        logger.debug('Locating Graph: "%s".', graph_name)
        full_path = os.path.join(GRAPHS_DIR, f"{graph_name}.txt")
        if not os.path.exists(full_path):
            logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
            continue

        graph = {}
        with open(full_path, "r") as f:
            line = f.readline()
            while line:
                node_edges = [int(i) for i in line.split()]
                node = node_edges[0]
                edges = node_edges[1:]
                graph[node] = set(edges)
                line = f.readline()

        yield graph_name, graph
